# main.py
#!/usr/bin/env python3
"""
AIALL Gateway – Unified System Service (Ubuntu-Friendly, Auto-Reload, Auto-Check)
--------------------------------------------------------------------------------
- Không phụ thuộc root
- Tự động kiểm tra lỗi file quan trọng (train, serve, deploy)
- Tự động restart backend AIALL STYLE ENGINE 8.0 khi file thay đổi
- Tích hợp đầy đủ router của hệ thống AIALL
- Soft cluster mode: không crash nếu thiếu Nginx/vLLM
"""

import os
import platform
import hashlib
import subprocess
import time
from datetime import datetime
from threading import Thread

from fastapi import FastAPI
import uvicorn

from config import init_config_system, load_runtime_config

# ============================================================
#  INIT CONFIG
# ============================================================

init_config_system()
cfg = load_runtime_config()

# ============================================================
#  ROUTERS IMPORT
# ============================================================

from routes.aiall_model import router as aiall_model_router
from routes.aiall_inference import router as aiall_inference_router
from routes.aiall_extended import router as aiall_extended_router
from routes.aiall_tools import router as aiall_tools_router
from routes.aiall_advanced import router as aiall_advanced_router
from routes.aiall_model_info import router as model_info_router
from routes.aiall_model_stats import router as model_stats_router
from routes.aiall_model_reload import router as model_reload_router
from routes.aiall_cluster_info import router as cluster_info_router
from routes.aiall_backend_stats import router as backend_stats_router
from routes.aiall_model_checksum import router as model_checksum_router
from routes.aiall_model_history_extended import router as model_history_extended_router
from routes.aiall_cluster_repair_smart import router as cluster_repair_smart_router
from routes.aiall_cluster_rebalance_smart import router as cluster_rebalance_smart_router
from routes.aiall_model_hot_swap import router as model_hot_swap_router

# ============================================================
#  CLUSTER FUNCTIONS (SOFT MODE)
# ============================================================

IS_LINUX = platform.system().lower().startswith("linux")
if IS_LINUX:
    try:
        from core.deploy_aiall_url import (
            full_deploy,
            auto_update_mode,
            health_check,
            auto_drain,
            rolling_restart,
        )
    except Exception:
        full_deploy = None
        auto_update_mode = None
        health_check = None
        auto_drain = None
        rolling_restart = None
else:
    full_deploy = None
    auto_update_mode = None
    health_check = None
    auto_drain = None
    rolling_restart = None

# ============================================================
#  BACKEND MANAGER
# ============================================================

try:
    import core.backends as be
except Exception:
    be = None


def is_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


# ============================================================
#  AUTO FILE CHECKER & AUTO-RESTART SERVE ENGINE
# ============================================================

WATCH_FILES = [
    "train/aiall_style.py",
    "train/aiall_full_pipeline.py",
    "train/serve_aiall_style.py",
    "core/deploy_aiall_url.py",
]

FILE_HASH = {}
SERVE_PROCESS = None


def file_hash(path: str):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def check_python_syntax(path: str):
    try:
        subprocess.check_output(["python3", "-m", "py_compile", path])
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.output.decode() if e.output else "Syntax error"


def restart_serve_engine(reason: str = "file change"):
    global SERVE_PROCESS
    try:
        if SERVE_PROCESS:
            print(f"[AUTO-RESTART] Stopping old serve_aiall_style.py (reason: {reason})...")
            SERVE_PROCESS.terminate()
            time.sleep(1)

        print("[AUTO-RESTART] Starting new serve_aiall_style.py...")
        SERVE_PROCESS = subprocess.Popen(
            ["python3", "train/serve_aiall_style.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[AUTO-RESTART] serve_aiall_style.py restarted successfully (PID={SERVE_PROCESS.pid}).")
    except Exception as e:
        print(f"[AUTO-RESTART] Failed to restart serve engine: {e}")


def watch_files_loop():
    while True:
        for f in WATCH_FILES:
            h = file_hash(f)
            if f not in FILE_HASH:
                FILE_HASH[f] = h
                continue

            if h != FILE_HASH[f]:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                old_hash = FILE_HASH[f]

                print(f"[AUTO-RELOAD] {timestamp} – File changed: {f}")
                print(f"[AUTO-RELOAD] Old hash: {old_hash}")
                print(f"[AUTO-RELOAD] New hash: {h}")

                ok, err = check_python_syntax(f)
                if not ok:
                    print(f"[ERROR] Syntax error in {f}:\n{err}")
                else:
                    print(f"[OK] {f} syntax valid.")
                    if "serve_aiall_style.py" in f:
                        restart_serve_engine(reason=f)

                FILE_HASH[f] = h

        time.sleep(2)


Thread(target=watch_files_loop, daemon=True).start()

# Khởi động serve engine ngay khi gateway chạy
restart_serve_engine(reason="gateway startup")

# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="AIALL Gateway – Unified System Service",
    version="4.0.0",
)

# ============================================================
#  ROOT ENDPOINT
# ============================================================

@app.get("/")
def index():
    return {
        "service": "AIALL Gateway – Unified System Service",
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
        "backends": be.load_backends() if be else [],
        "environment": {
            "is_linux": IS_LINUX,
            "is_root": is_root() if IS_LINUX else None,
        },
        "note": "Gateway đang chạy chế độ Ubuntu-friendly (không yêu cầu root).",
    }

# ============================================================
#  SYSTEM HEALTH & AUTO-RELOAD STATUS
# ============================================================

@app.get("/system/health")
def system_health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/system/auto-reload-status")
def auto_reload_status():
    out = {}
    for f in WATCH_FILES:
        out[f] = {
            "last_hash": FILE_HASH.get(f),
            "exists": os.path.exists(f),
        }
    return {
        "status": "watching",
        "files": out,
        "serve_running": SERVE_PROCESS is not None,
        "serve_pid": SERVE_PROCESS.pid if SERVE_PROCESS else None,
    }

# ============================================================
#  FILE CHECK STATUS
# ============================================================

@app.get("/system/file-check")
def system_file_check():
    out = {}
    for f in WATCH_FILES:
        ok, err = check_python_syntax(f)
        out[f] = "OK" if ok else f"ERROR: {err}"
    return out

# ============================================================
#  CLUSTER HEALTH (soft mode)
# ============================================================

@app.get("/cluster/health")
def cluster_health():
    if not IS_LINUX or health_check is None:
        return {"cluster": "unknown", "note": "Linux-only"}
    try:
        health_check()
        return {"cluster": "healthy"}
    except Exception as e:
        return {"cluster": "unhealthy", "error": str(e)}

# ============================================================
#  ROUTES REGISTRATION
# ============================================================

app.include_router(aiall_model_router)
app.include_router(aiall_inference_router)
app.include_router(aiall_extended_router)
app.include_router(aiall_tools_router)
app.include_router(aiall_advanced_router)
app.include_router(model_info_router)
app.include_router(model_stats_router)
app.include_router(model_reload_router)
app.include_router(cluster_info_router)
app.include_router(backend_stats_router)
app.include_router(model_checksum_router)
app.include_router(model_history_extended_router)
app.include_router(cluster_repair_smart_router)
app.include_router(cluster_rebalance_smart_router)
app.include_router(model_hot_swap_router)

# ============================================================
#  RUN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6001)
