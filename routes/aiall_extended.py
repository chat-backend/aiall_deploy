# routes/aiall_extended.py
# ============================================================
#  AIALL EXTENDED FEATURES (Moderation / Translate / TTS / Assistant / Secure Chat / Finetune / Files)
# ============================================================

import json
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from config_loader import load_runtime_config

from train.aiall_train import (
    load_aiall_for_inference,
    chat,
    train_aiall,
    merge_lora,
    register_aiall_backend
)

router = APIRouter(prefix="/aiall", tags=["AIALL Extended"])


# ============================================================
#  MODERATION
# ============================================================

@router.post("/moderation")
def aiall_moderation(text: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    flags = {
        "violence": any(w in text.lower() for w in ["kill", "fight", "weapon"]),
        "hate": any(w in text.lower() for w in ["hate", "racist", "discrimination"]),
        "sexual": any(w in text.lower() for w in ["sex", "nude", "porn"]),
        "self_harm": any(w in text.lower() for w in ["suicide", "self-harm"]),
    }

    return {
        "object": "moderation",
        "model": "aiall",
        "input": text,
        "flags": flags,
        "safe": not any(flags.values())
    }


# ============================================================
#  TRANSLATE
# ============================================================

@router.post("/translate")
def aiall_translate(text: str, target_lang: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()
        prompt = f"Dịch đoạn văn sau sang {target_lang}: {text}"
        translated = chat(model, tokenizer, prompt)

        return {
            "object": "translation",
            "model": "aiall",
            "source": text,
            "target_lang": target_lang,
            "translated": translated
        }
    except Exception as e:
        return {"translate": "failed", "error": str(e)}


# ============================================================
#  TTS (Text-to-Speech)
# ============================================================

@router.post("/tts")
def aiall_tts(text: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        from gtts import gTTS
        import io

        tts = gTTS(text=text, lang="vi")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        return StreamingResponse(buf, media_type="audio/mpeg")

    except Exception as e:
        return {"tts": "failed", "error": str(e)}


# ============================================================
#  ASSISTANT (Multi-message Chat)
# ============================================================

@router.post("/assistant")
def aiall_assistant(messages: list, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()

        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.upper()}: {content}\n"

        prompt += "ASSISTANT:"

        reply = chat(model, tokenizer, prompt)

        return {
            "object": "chat.completion",
            "model": "aiall",
            "messages": messages,
            "reply": reply
        }

    except Exception as e:
        return {"assistant": "failed", "error": str(e)}


# ============================================================
#  SECURE CHAT (AES-256)
# ============================================================

@router.post("/secure-chat")
def aiall_secure_chat(prompt: str, token: str = "", key: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        from Crypto.Cipher import AES
        import base64

        key_bytes = key.encode()
        if len(key_bytes) not in [16, 24, 32]:
            return {"secure_chat": "failed", "error": "AES key must be 16/24/32 bytes"}

        decoded = base64.b64decode(prompt)
        nonce = decoded[:16]
        ciphertext = decoded[16:]

        cipher = AES.new(key_bytes, AES.MODE_EAX, nonce=nonce)
        decrypted_prompt = cipher.decrypt(ciphertext).decode()

        model, tokenizer = load_aiall_for_inference()
        reply = chat(model, tokenizer, decrypted_prompt)

        cipher2 = AES.new(key_bytes, AES.MODE_EAX)
        encrypted = cipher2.nonce + cipher2.encrypt(reply.encode())
        encrypted_b64 = base64.b64encode(encrypted).decode()

        return {
            "object": "secure.chat",
            "model": "aiall",
            "encrypted_reply": encrypted_b64
        }

    except Exception as e:
        return {"secure_chat": "failed", "error": str(e)}


# ============================================================
#  FINETUNE (Train + Merge + Register)
# ============================================================

@router.post("/finetune")
def aiall_finetune(dataset_path: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        train_aiall()
        merge_lora()
        register_aiall_backend(host="127.0.0.1", port=8000)

        return {
            "object": "finetune",
            "model": "aiall",
            "dataset": dataset_path,
            "status": "completed"
        }

    except Exception as e:
        return {"finetune": "failed", "error": str(e)}


# ============================================================
#  FILE UPLOAD (JSONL Dataset)
# ============================================================

@router.post("/files")
async def aiall_files(file: UploadFile = File(...), token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        if not file.filename.endswith(".jsonl"):
            return {"upload": "failed", "error": "file must be .jsonl"}

        contents = await file.read()
        with open("aiall_data.jsonl", "wb") as f:
            f.write(contents)

        return {
            "object": "file.upload",
            "filename": file.filename,
            "size": len(contents),
            "status": "uploaded"
        }

    except Exception as e:
        return {"upload": "failed", "error": str(e)}
