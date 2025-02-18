import torch
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/app/deepseek-llm-7b-chat"
offload_path = "/app/offload"

torch.backends.cuda.matmul.allow_tf32 = True  # TF32 연산 최적화

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=offload_path,
    # attn_implementation="flash_attention_2",
)
model.eval()  # 모델을 평가 모드로 설정하여 성능 최적화
model = torch.compile(model)
model.half()

user_sessions = {}
MAX_HISTORY = 5  # 최근 5개 대화만 유지하여 메모리 최적화

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "messages": []})

@app.post("/generate/")
async def generate(request: Request, user_input: str = Form(...), session_id: str = Form("default")):
    print("Generate response...")
    if session_id not in user_sessions:
        user_sessions[session_id] = [{"role": "system", "content": "You are a helpful assistant."}]

    user_sessions[session_id].append({"role": "user", "content": user_input})
    
    history = user_sessions[session_id][-MAX_HISTORY:]  # 최신 5개 메시지만 사용
    formatted_input = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    formatted_input += f"\nAssistant:"

    chat_prompt = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True).to(device)

    # 올바른 attention_mask 설정
    attention_mask = chat_prompt["attention_mask"].to(device)

    with torch.inference_mode():  # 그래디언트 계산 방지
        outputs = model.generate(
            input_ids=chat_prompt["input_ids"],
            attention_mask=attention_mask,  # 명시적으로 attention_mask 추가
            num_beams=1,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id  # pad_token_id 설정 유지
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    user_sessions[session_id].append({"role": "assistant", "content": response})

    return templates.TemplateResponse("index.html", {"request": request, "messages": user_sessions[session_id]})
