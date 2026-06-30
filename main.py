# main.py
#!/usr/bin/env python3
"""
AIALL vLLM Gateway – System Service (FastAPI, FULL INTEGRATION, CROSS-PLATFORM SAFE)
------------------------------------------------------------------------------------
- Không xử lý API AI chính (đã đi qua Nginx → vLLM backend)
- Hiển thị thông tin cấu hình runtime
- Quản lý backend (add/remove/drain/undrain)
- Health-check cluster
- Auto-update, auto-drain, rolling-restart (Linux-only)
- Tích hợp train/merge/load/test mô hình AIALL qua endpoint riêng
- Xem MODEL_TOKEN (ẩn giá trị, chỉ hash)
- Xem trạng thái Real-Time Context Layer
"""

import os
import platform
import hashlib
import subprocess
from datetime import datetime

from fastapi import FastAPI
import uvicorn

from config_loader import load_runtime_config
cfg = load_runtime_config()

from routes.aiall_model import router as aiall_model_router
from routes.aiall_inference import router as aiall_inference_router
from routes.aiall_extended import router as aiall_extended_router
from routes.aiall_tools import router as aiall_tools_router
from routes.aiall_advanced import router as aiall_advanced_router

IS_LINUX = platform.system().lower().startswith("linux")

# ===== Import deploy_main logic (Linux-only) =====
if IS_LINUX:
    from deploy_main import (
        full_deploy,
        auto_update_mode,
        health_check,
        auto_drain,
        rolling_restart,
    )
else:
    full_deploy = None
    auto_update_mode = None
    health_check = None
    auto_drain = None
    rolling_restart = None

# ===== Backend manager =====
if IS_LINUX:
    import core.backends as be
    import core.nginx as ngx
else:
    import core.backends as be
    ngx = None  # Nginx không dùng trên Windows


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


app = FastAPI(
    title="AIALL vLLM Gateway – System Service",
    version="3.0.0",
)


# ============================================================
#  ROOT ENDPOINT
# ============================================================

@app.get("/")
def index():
    return {
        "service": "AIALL vLLM Gateway – System Service",
        "status": "running",
        "base_url": cfg.base_url,
        "api": {
            "chat": cfg.url_chat,
            "completion": cfg.url_completion,
            "models": cfg.url_models,
        },
        "auth": {
            "api_key": "hidden",
            "model_token": "hidden",
        },
        "default_params": {
            "max_tokens": cfg.default_max_tokens,
            "min_tokens": cfg.default_min_tokens,
            "temperature": cfg.default_temperature,
            "top_p": cfg.default_top_p,
        },
        "backends": be.load_backends(),
        "environment": {
            "is_linux": IS_LINUX,
            "is_root": is_root() if IS_LINUX else None,
        },
        "note": "API chính chạy qua Nginx → vLLM backend (OpenAI-compatible)",
    }


# ============================================================
#  RUNTIME CONFIG VIEW
# ============================================================

@app.get("/system/config")
def system_config():
    return {
        "base_url": cfg.base_url,
        "api_chat": cfg.api_chat,
        "api_completion": cfg.api_completion,
        "api_models": cfg.api_models,
        "url_chat": cfg.url_chat,
        "url_completion": cfg.url_completion,
        "url_models": cfg.url_models,
        "default_params": {
            "max_tokens": cfg.default_max_tokens,
            "min_tokens": cfg.default_min_tokens,
            "temperature": cfg.default_temperature,
            "top_p": cfg.default_top_p,
        },
    }


# ============================================================
#  MODEL TOKEN HASH VIEW
# ============================================================

@app.get("/system/model-token")
def system_model_token():
    raw = cfg.model_token.strip()
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return {
        "model_token_hash": hashed,
        "note": "Giá trị thật của MODEL_TOKEN được ẩn để đảm bảo an toàn."
    }


# ============================================================
#  REAL-TIME CONTEXT STATUS
# ============================================================

@app.get("/system/realtime")
def system_realtime():
    return {
        "realtime_context": "enabled",
        "sources": {
            "time": True,
            "web_search": "stub",
            "database": "stub",
            "events": "stub",
            "finance": "stub",
            "weather": "stub",
            "news": "stub",
            "calendar": True,
        },
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Real-Time Context Layer đang hoạt động ở chế độ stub. Có thể nâng cấp để dùng API thật.",
    }


# ============================================================
#  SYSTEM HEALTH
# ============================================================

@app.get("/system/health")
def system_health():
    return {"status": "ok"}


# ============================================================
#  CLUSTER HEALTH
# ============================================================

@app.get("/cluster/health")
def cluster_health():
    if not IS_LINUX:
        return {"cluster": "unknown", "error": "cluster health is Linux-only"}
    try:
        health_check()
        return {"cluster": "healthy"}
    except Exception as e:
        return {"cluster": "unhealthy", "error": str(e)}


# ============================================================
#  TRIGGER TRAIN PIPELINE
# ============================================================

@app.post("/cluster/train")
def cluster_train():
    if not IS_LINUX:
        return {"train": "unsupported", "error": "train pipeline is Linux-only"}
    if not is_root():
        return {"train": "failed", "error": "train requires root (sudo)"}

    try:
        result = subprocess.run(
            ["python3", "train/train_pipeline.py"],
            capture_output=True,
            text=True,
        )
        return {
            "train": "completed",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"train": "failed", "error": str(e)}


# ============================================================
#  FULL DEPLOY
# ============================================================

@app.post("/cluster/deploy")
def cluster_deploy():
    if not IS_LINUX:
        return {"deploy": "unsupported", "error": "deploy is Linux-only"}
    if not is_root():
        return {"deploy": "failed", "error": "deploy requires root (sudo)"}
    try:
        full_deploy()
        return {"deploy": "completed"}
    except Exception as e:
        return {"deploy": "failed", "error": str(e)}


# ============================================================
#  AUTO UPDATE
# ============================================================

@app.post("/cluster/update")
def cluster_update():
    if not IS_LINUX:
        return {"update": "unsupported", "error": "update is Linux-only"}
    if not is_root():
        return {"update": "failed", "error": "update requires root (sudo)"}
    try:
        auto_update_mode()
        return {"update": "started"}
    except Exception as e:
        return {"update": "failed", "error": str(e)}


# ============================================================
#  AUTO DRAIN
# ============================================================

@app.post("/cluster/auto-drain")
def cluster_auto_drain():
    if not IS_LINUX:
        return {"auto_drain": "unsupported", "error": "auto-drain is Linux-only"}
    if not is_root():
        return {"auto_drain": "failed", "error": "auto-drain requires root (sudo)"}
    try:
        auto_drain()
        return {"auto_drain": "started"}
    except Exception as e:
        return {"auto_drain": "failed", "error": str(e)}


# ============================================================
#  ROLLING RESTART
# ============================================================

@app.post("/cluster/rolling-restart")
def cluster_rolling_restart():
    if not IS_LINUX:
        return {"rolling_restart": "unsupported", "error": "rolling-restart is Linux-only"}
    if not is_root():
        return {"rolling_restart": "failed", "error": "rolling-restart requires root (sudo)"}
    try:
        rolling_restart()
        return {"rolling_restart": "started"}
    except Exception as e:
        return {"rolling_restart": "failed", "error": str(e)}


# ============================================================
#  BACKEND MANAGEMENT
# ============================================================

@app.post("/backend/add")
def add_backend(backend: str):
    try:
        be.add_backend(backend)
        if IS_LINUX and ngx is not None:
            ngx.generate_upstream_block()
            ngx.reload_nginx()
        return {"backend": backend, "status": "added"}
    except Exception as e:
        return {"backend": backend, "status": "failed", "error": str(e)}


@app.post("/backend/remove")
def remove_backend(backend: str):
    try:
        be.remove_backend(backend)
        if IS_LINUX and ngx is not None:
            ngx.generate_upstream_block()
            ngx.reload_nginx()
        return {"backend": backend, "status": "removed"}
    except Exception as e:
        return {"backend": backend, "status": "failed", "error": str(e)}


@app.post("/backend/drain")
def drain_backend(backend: str):
    try:
        be.drain_backend(backend)
        if IS_LINUX and ngx is not None:
            ngx.generate_upstream_block()
            ngx.reload_nginx()
        return {"backend": backend, "status": "drained"}
    except Exception as e:
        return {"backend": backend, "status": "failed", "error": str(e)}


@app.post("/backend/undrain")
def undrain_backend(backend: str):
    try:
        be.undrain_backend(backend)
        if IS_LINUX and ngx is not None:
            ngx.generate_upstream_block()
            ngx.reload_nginx()
        return {"backend": backend, "status": "undrained"}
    except Exception as e:
        return {"backend": backend, "status": "failed", "error": str(e)}


# ============================================================
#  ROUTES REGISTRATION
# ============================================================

app.include_router(aiall_model_router)
app.include_router(aiall_inference_router)
app.include_router(aiall_extended_router)
app.include_router(aiall_tools_router)
app.include_router(aiall_advanced_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6001)


