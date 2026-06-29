# routes/aiall_inference.py
# ============================================================
#  AIALL CHAT / COMPLETION / EMBED / AUDIO / VISION / METRICS / STREAM
# ============================================================

import os
import psutil
import torch
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config_loader import load_runtime_config
import core.backends as be

from train.aiall_train import (
    load_aiall_for_inference,
    chat
)

router = APIRouter(prefix="/aiall", tags=["AIALL Inference"])


# ============================================================
#  CHAT
# ============================================================

@router.post("/chat")
def aiall_chat(prompt: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()
        response = chat(model, tokenizer, prompt)
        return {"model": "aiall", "prompt": prompt, "response": response}
    except Exception as e:
        return {"chat": "failed", "error": str(e)}


# ============================================================
#  COMPLETIONS
# ============================================================

@router.post("/completions")
def aiall_completions(prompt: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()
        response = chat(model, tokenizer, prompt)

        return {
            "id": "aiall-completion",
            "object": "text_completion",
            "model": "aiall",
            "choices": [{"index": 0, "text": response, "finish_reason": "stop"}]
        }
    except Exception as e:
        return {"completion": "failed", "error": str(e)}


# ============================================================
#  EMBEDDING
# ============================================================

@router.post("/embed")
def aiall_embed(text: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][0].mean(dim=0).tolist()

        return {
            "object": "embedding",
            "model": "aiall",
            "embedding": embedding,
            "dimension": len(embedding)
        }
    except Exception as e:
        return {"embed": "failed", "error": str(e)}


# ============================================================
#  AUDIO (Speech-to-Text)
# ============================================================

@router.post("/audio")
def aiall_audio(file: bytes, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        import speech_recognition as sr
        import io

        recognizer = sr.Recognizer()
        audio_data = sr.AudioFile(io.BytesIO(file))

        with audio_data as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language="vi-VN")

        return {"object": "audio.transcription", "model": "aiall", "text": text}
    except Exception as e:
        return {"audio": "failed", "error": str(e)}


# ============================================================
#  VISION (Image-to-Text)
# ============================================================

@router.post("/vision")
def aiall_vision(file: bytes, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(file)).convert("RGB")
        description = f"Hình ảnh có kích thước {img.width}x{img.height}."

        return {"object": "image.description", "model": "aiall", "description": description}
    except Exception as e:
        return {"vision": "failed", "error": str(e)}


# ============================================================
#  METRICS
# ============================================================

@router.get("/metrics")
def aiall_metrics():
    return {
        "model": "aiall",
        "adapter_exists": os.path.isdir("aiall-lora"),
        "merged_exists": os.path.isdir("aiall-merged"),
        "backends": be.load_backends(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        }
    }


# ============================================================
#  STREAM (SSE)
# ============================================================

@router.post("/stream")
async def aiall_stream(prompt: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()

        async def token_generator():
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            for tok in decoded.split():
                yield f"data: {tok}\n\n"
                await asyncio.sleep(0.01)

            yield "data: [DONE]\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    except Exception as e:
        return {"stream": "failed", "error": str(e)}
