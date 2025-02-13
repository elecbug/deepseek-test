from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
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

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/generate/")
async def generate(request: ChatRequest):
    messages = request.messages
    
    chat_prompt = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    attention_mask = torch.ones(chat_prompt.shape, dtype=torch.long).to(device)

    print(f"chat_prompt type: {type(chat_prompt)}")
    print(f"chat_prompt content: {chat_prompt}")

    outputs = model.generate(
        input_ids=chat_prompt,
        attention_mask=attention_mask,
        max_length=256,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}
