# train/serve_aiall.py
#!/usr/bin/env python3
"""
AIALL – Serve merged model (FastAPI, Real-Time Context, port 8001, CPU-Optimized, Streaming)
------------------------------------------------------------------
- Tối ưu:
  - mkldnn + thread limit
  - max_new_tokens thấp hơn (64)
  - no_grad + single-batch
  - thêm endpoint streaming /aiall/chat-stream
"""

import os
import platform
from datetime import datetime
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# CPU OPTIMIZATION
torch.backends.mkldnn.enabled = True
torch.set_num_threads(max(1, os.cpu_count() // 2))

IS_LINUX = platform.system().lower().startswith("linux")
MERGED_OUTPUT_DIR = "aiall-merged"


class HotSwapPayload(BaseModel):
    model_dir: str


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    prompt: str
    response: str


# ============================================================
#  REAL-TIME CONTEXT LAYER
# ============================================================

def build_realtime_context(prompt: str) -> str:
    parts = []

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"[TIME] Current datetime: {current_time}")

    pl = prompt.lower()

    if any(w in pl for w in ["tìm", "search", "google", "tra cứu"]):
        parts.append("[WEB] Web search: (stub) dữ liệu thời gian thực sẽ được tích hợp tại đây.")

    if "cơ sở dữ liệu" in pl or "database" in pl or "db" in pl:
        parts.append("[DB] Database: (stub) kết quả truy vấn DB sẽ được tích hợp tại đây.")

    if "sự kiện" in pl or "timeline" in pl:
        parts.append("[EVENTS] Timeline: (stub) danh sách sự kiện gần đây sẽ được tích hợp tại đây.")

    if any(w in pl for w in ["giá", "bitcoin", "btc", "chứng khoán", "stock"]):
        parts.append("[FINANCE] Financial: (stub) giá tài chính thời gian thực sẽ được tích hợp tại đây.")

    if "thời tiết" in pl or "weather" in pl:
        parts.append("[WEATHER] Weather: (stub) dữ liệu thời tiết thời gian thực sẽ được tích hợp tại đây.")

    if "tin tức" in pl or "news" in pl or "báo" in pl:
        parts.append("[NEWS] News: (stub) tin tức mới nhất sẽ được tích hợp tại đây.")

    if any(w in pl for w in ["lịch", "ngày", "tháng", "năm", "calendar"]):
        parts.append("[CALENDAR] Calendar: hệ thống nhận biết ngày/tháng/năm hiện tại và bối cảnh thời gian.")

    parts.append(
        "[REAL-TIME SYSTEM NOTICE] "
        "Nếu câu hỏi liên quan đến dữ liệu có thể thay đổi theo thời gian "
        "(sự kiện, giá, tin tức, thời tiết, lịch, v.v.), "
        "hãy luôn cân nhắc rằng thông tin trong mô hình có thể đã lỗi thời. "
        "Nếu không chắc chắn, hãy nói: "
        "'Thông tin tôi có thể đã lỗi thời. Bạn muốn tôi kiểm tra dữ liệu mới nhất không?'"
    )

    return "\n".join(parts) + "\n\n"


# ============================================================
#  LOAD MERGED MODEL (CPU-optimized)
# ============================================================

def load_aiall_for_inference(model_dir: str):
    if not os.path.exists(model_dir):
        raise SystemExit(f"[ERROR] Merged model not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cpu",
    )
    model = model.to(torch.float32)
    model.eval()
    return model, tokenizer


model, tokenizer = load_aiall_for_inference(MERGED_OUTPUT_DIR)

# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="AIALL Merged Model – Inference Service (CPU-Optimized, Streaming)",
    version="1.1.0",
)


@app.get("/aiall/health")
def health():
    return {
        "status": "ok",
        "model_dir": MERGED_OUTPUT_DIR,
        "platform": platform.system(),
        "linux": IS_LINUX,
    }


# ============================================================
#  NORMAL CHAT (non-streaming)
# ============================================================

@app.post("/aiall/chat", response_model=ChatResponse)
def aiall_chat(req: ChatRequest):
    prompt = req.prompt
    realtime_context = build_realtime_context(prompt)

    text = realtime_context + f"Instruction: {prompt}\nAnswer:"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,    # giảm thêm để phản hồi nhanh hơn
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ChatResponse(prompt=prompt, response=decoded)


# ============================================================
#  STREAMING CHAT (chunked text)
# ============================================================

async def stream_generate(prompt: str) -> AsyncGenerator[str, None]:
    realtime_context = build_realtime_context(prompt)
    text = realtime_context + f"Instruction: {prompt}\nAnswer:"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # generate full output once, then stream chunks
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # stream in small chunks
    chunk_size = 32
    for i in range(0, len(decoded), chunk_size):
        yield decoded[i:i + chunk_size]


@app.post("/aiall/chat-stream")
async def aiall_chat_stream(req: ChatRequest):
    async def event_stream():
        async for chunk in stream_generate(req.prompt):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/plain")


# ============================================================
#  RELOAD / HOT-SWAP
# ============================================================

@app.post("/aiall/reload")
def reload_model():
    global model, tokenizer
    model, tokenizer = load_aiall_for_inference(MERGED_OUTPUT_DIR)
    return {"status": "ok", "message": "Model reloaded successfully."}


@app.post("/aiall/hot-swap")
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



