from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True

model_path = "/app/deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = model.to(device)

@app.post("/generate/")
async def generate(messages: list):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response}
