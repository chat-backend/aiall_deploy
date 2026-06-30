# ============================================================
#  MODEL INFO API
# routes/aiall_model_info.py
# ============================================================

from fastapi import APIRouter
import os
from datetime import datetime
from config import MODEL_TOKEN_FILE, PROJECT_CONFIG_FILE
import core.backends as be
from core.health_cluster import health_check

router = APIRouter()

@router.get("/system/model-info")
def model_info():
    # Tên mô hình cố định của AIALL
    model_name = "aiall"
    model_base = "Qwen/Qwen2.5-1.5B"
    model_dir = "aiall-merged"

    # Đọc model token
    model_token = None
    if MODEL_TOKEN_FILE.exists():
        for line in MODEL_TOKEN_FILE.read_text().splitlines():
            if line.startswith("AIALL_MODEL_TOKEN="):
                model_token = line.split("=", 1)[1].strip()

    # Đọc version từ project config
    version = None
    if PROJECT_CONFIG_FILE.exists():
        for line in PROJECT_CONFIG_FILE.read_text().splitlines():
            if line.startswith("CONFIG_VERSION="):
                version = line.split("=", 1)[1].strip()

    # Backend đang active
    backends = be.load_backends()

    # Cluster health
    try:
        health = health_check(return_status=True)
    except Exception:
        health = "unknown"

    # Build time = thời điểm merge model
    merged_path = os.path.join(os.getcwd(), model_dir)
    if os.path.exists(merged_path):
        build_time = datetime.fromtimestamp(os.path.getmtime(merged_path)).isoformat()
    else:
        build_time = None

    return {
        "model_name": model_name,
        "model_base": model_base,
        "model_dir": model_dir,
        "model_token": model_token,
        "version": version,
        "primary_api_url": "https://api.aiallplatform.com/v1/",
        "backends": backends,
        "cluster_status": health,
        "build_time": build_time,
        "api_usage": {
            "chat": "/v1/chat/completions",
            "completion": "/v1/completions",
            "models": "/v1/models",
            "merged_chat": "/aiall/chat"
        }
    }

