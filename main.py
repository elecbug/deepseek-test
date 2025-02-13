from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

device = "cpu"

model_path = "/app/deepseek-llm-7b-base"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

# device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.post("/generate/")
async def generate(messages: list):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=50, 
        num_beams=2,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response}
