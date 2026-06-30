# routes/aiall_model.py
# ============================================================
#  AIALL MODEL MANAGEMENT (TRAIN / MERGE / TEST / REGISTER / SERVE / RESET)
# ============================================================

import os
from fastapi import APIRouter
from config_loader import load_runtime_config
import core.backends as be
import core.nginx as ngx

from train.aiall_train import (
    train_aiall,
    merge_lora,
    load_aiall_for_inference,
    chat,
    register_aiall_backend
)

router = APIRouter(prefix="/aiall", tags=["AIALL Model Management"])


# ============================================================
#  TRAIN
# ============================================================

@router.post("/train")
def aiall_train_endpoint(token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        train_aiall()
        return {"aiall": "train_completed"}
    except Exception as e:
        return {"aiall": "train_failed", "error": str(e)}


# ============================================================
#  MERGE
# ============================================================

@router.post("/merge")
def aiall_merge_endpoint(token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        merge_lora()
        return {"aiall": "merge_completed"}
    except Exception as e:
        return {"aiall": "merge_failed", "error": str(e)}


# ============================================================
#  TEST
# ============================================================

@router.get("/test")
def aiall_test_endpoint(token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()
        prompt = "Giới thiệu bản thân là aiall bằng tiếng Việt, thân thiện, cảm xúc."
        response = chat(model, tokenizer, prompt)
        return {"prompt": prompt, "response": response}
    except Exception as e:
        return {"aiall": "test_failed", "error": str(e)}


# ============================================================
#  REGISTER BACKEND
# ============================================================

@router.post("/register")
def aiall_register_endpoint(host: str = "127.0.0.1", port: int = 8000, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        register_aiall_backend(host=host, port=port)
        return {
            "aiall": "register_completed",
            "backend": f"{host}:{port}",
            "url": f"{cfg.base_url}/aiall/"
        }
    except Exception as e:
        return {"aiall": "register_failed", "error": str(e)}


# ============================================================
#  INFO
# ============================================================

@router.get("/info")
def aiall_info():
    cfg = load_runtime_config()
    return {
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "model_token": cfg.model_token,
        "default_max_tokens": cfg.default_max_tokens,
        "default_min_tokens": cfg.default_min_tokens,
        "temperature": cfg.default_temperature,
        "top_p": cfg.default_top_p,
        "backends": be.load_backends(),
    }


# ============================================================
#  SERVE (AUTO START VLLM)
# ============================================================

@router.post("/serve")
def aiall_serve(host: str = "127.0.0.1", port: int = 8000, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    if not os.path.isdir("aiall-merged"):
        return {"serve": "failed", "error": "model not merged yet"}

    try:
        import subprocess

        cmd = [
            "vllm",
            "serve",
            "--model", "aiall-merged",
            "--host", host,
            "--port", str(port)
        ]

        subprocess.Popen(cmd)
        register_aiall_backend(host=host, port=port)

        return {
            "serve": "started",
            "backend": f"{host}:{port}",
            "url": f"{cfg.base_url}/aiall/"
        }
    except Exception as e:
        return {"serve": "failed", "error": str(e)}


# ============================================================
#  RESET
# ============================================================

@router.post("/reset")
def aiall_reset(token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        from config import MODEL_TOKEN_FILE, BACKENDS_CONFIG

        for path in ["aiall-lora", "aiall-merged"]:
            if os.path.isdir(path):
                os.system(f"rm -rf {path}")

        if MODEL_TOKEN_FILE.exists():
            MODEL_TOKEN_FILE.unlink()

        if BACKENDS_CONFIG.exists():
            BACKENDS_CONFIG.unlink()

        ngx.generate_upstream_block()
        ngx.reload_nginx()

        return {"reset": "completed"}
    except Exception as e:
        return {"reset": "failed", "error": str(e)}


# ============================================================
#  AUTH
# ============================================================

@router.post("/auth")
def aiall_auth(token: str):
    cfg = load_runtime_config()
    if token == cfg.model_token:
        return {"auth": "success", "model": "aiall"}
    return {"auth": "failed", "error": "invalid token"}


# ============================================================
#  MODELS
# ============================================================

@router.get("/models")
def aiall_models():
    return {
        "adapter_exists": os.path.isdir("aiall-lora"),
        "merged_exists": os.path.isdir("aiall-merged"),
        "backends": be.load_backends(),
        "model_token": load_runtime_config().model_token,
    }
