import torch
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Titan RTX (G102) 맞춤 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/app/deepseek-llm-7b-chat"
offload_path = "/app/offload"

torch.backends.cuda.matmul.allow_tf32 = True  # TF32 연산 활성화 (Volta 이상에서 최적화됨)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 8bit 대신 FP16 사용 (Titan RTX는 Ampere 이전 아키텍처이므로 8bit 효율이 낮음)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # FP16 사용
    device_map="auto",
    offload_folder=offload_path,
)

# CUDA 그래프 최적화 적용 (권장)
model = torch.compile(model, mode="reduce-overhead")

user_sessions = {}

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "messages": []})

@app.post("/generate/")
async def generate(request: Request, user_input: str = Form(...), session_id: str = Form("default")):
    if session_id not in user_sessions:
        user_sessions[session_id] = [{"role": "system", "content": "You are a helpful assistant."}]

    user_sessions[session_id].append({"role": "user", "content": user_input})

    chat_prompt = tokenizer.apply_chat_template(user_sessions[session_id], return_tensors="pt").to(device)
    attention_mask = torch.ones(chat_prompt.shape, dtype=torch.long).to(device)

    outputs = model.generate(
        input_ids=chat_prompt,
        attention_mask=attention_mask,
        num_beams=4,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    user_sessions[session_id].append({"role": "assistant", "content": response})

    return templates.TemplateResponse("index.html", {"request": request, "messages": user_sessions[session_id]})
