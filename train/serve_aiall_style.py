# train/serve_aiall_style.py
#!/usr/bin/env python3
"""
AIALL – Serve merged model with FULL AIALL STYLE ENGINE 8.0
FastAPI, Real-Time Context, CPU-Optimized, Streaming, STYLE 8.0
"""

import os
import platform
from datetime import datetime
from typing import AsyncGenerator, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.aiall_style import aiall_chat

# CPU OPTIMIZATION
torch.backends.mkldnn.enabled = True
torch.set_num_threads(max(1, os.cpu_count() // 2))

IS_LINUX = platform.system().lower().startswith("linux")
MERGED_OUTPUT_DIR = "aiall-merged"
MAX_LEN = 256


class HotSwapPayload(BaseModel):
    model_dir: str

    model_config = {
        "protected_namespaces": ()
    }


class ChatRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    prompt: str
    response: str


def build_realtime_context(prompt: str) -> str:
    parts = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"[TIME] Thời gian hiện tại: {current_time}")

    pl = prompt.lower()
    if any(w in pl for w in ["tin tức", "news", "báo"]):
        parts.append("[NEWS] Lưu ý: Thông tin thời sự có thể đã lỗi thời.")
    if any(w in pl for w in ["giá", "bitcoin", "btc", "chứng khoán", "stock"]):
        parts.append("[FINANCE] Lưu ý: Giá tài chính thay đổi liên tục.")
    if "thời tiết" in pl or "weather" in pl:
        parts.append("[WEATHER] Lưu ý: Dữ liệu thời tiết không phải thời gian thực.")
    if any(w in pl for w in ["lịch", "ngày", "tháng", "năm", "calendar"]):
        parts.append("[CALENDAR] Nhận biết thời gian hiện tại.")

    parts.append("[REAL-TIME NOTICE] Thông tin có thể đã lỗi thời.")

    return "\n".join(parts) + "\n\n"


def load_aiall_for_inference(model_dir: str):
    if not os.path.exists(model_dir):
        raise SystemExit(f"[ERROR] Merged model not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu")
    model = model.to(torch.float32)
    model.eval()
    return model, tokenizer


model, tokenizer = load_aiall_for_inference(MERGED_OUTPUT_DIR)


app = FastAPI(
    title="AIALL STYLE ENGINE 8.0 – Inference Service",
    version="1.0.0",
)


@app.get("/aiall/style-health")
def health():
    return {
        "status": "ok",
        "model_dir": MERGED_OUTPUT_DIR,
        "platform": platform.system(),
        "linux": IS_LINUX,
        "style_engine": "AIALL STYLE 8.0",
    }


@app.post("/aiall/style-chat", response_model=ChatResponse)
def aiall_style_chat(req: ChatRequest):
    realtime_context = build_realtime_context(req.prompt)
    full_prompt = realtime_context + req.prompt

    with torch.no_grad():
        response = aiall_chat(model, tokenizer, full_prompt, user_id=req.user_id)

    return ChatResponse(prompt=req.prompt, response=response)


async def stream_generate_style(prompt: str, user_id: Optional[str]) -> AsyncGenerator[str, None]:
    realtime_context = build_realtime_context(prompt)
    full_prompt = realtime_context + prompt

    with torch.no_grad():
        answer = aiall_chat(model, tokenizer, full_prompt, user_id=user_id)

    chunk_size = 64
    for i in range(0, len(answer), chunk_size):
        yield answer[i:i + chunk_size]


@app.post("/aiall/style-chat-stream")
async def aiall_style_chat_stream(req: ChatRequest):
    async def event_stream():
        async for chunk in stream_generate_style(req.prompt, req.user_id):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/aiall/style-reload")
def reload_model():
    global model, tokenizer
    model, tokenizer = load_aiall_for_inference(MERGED_OUTPUT_DIR)
    return {"status": "ok", "message": "Model reloaded successfully."}


@app.post("/aiall/style-hot-swap")
def hot_swap_model(payload: HotSwapPayload):
    global model, tokenizer, MERGED_OUTPUT_DIR
    new_dir = payload.model_dir

    if not os.path.exists(new_dir):
        return {"status": "error", "message": f"Model dir not found: {new_dir}"}

    MERGED_OUTPUT_DIR = new_dir
    model, tokenizer = load_aiall_for_inference(MERGED_OUTPUT_DIR)
    return {"status": "ok", "message": f"Model hot-swapped to {new_dir}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
