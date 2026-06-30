# ============================================================
#  MODEL RELOAD API (Improved)
# routes/aiall_model_reload.py
# ============================================================

from fastapi import APIRouter
import requests

router = APIRouter()

SERVE_AIALL_URL = "http://127.0.0.1:8001/aiall/reload"

@router.post("/system/model-reload")
def model_reload():
    try:
        r = requests.post(SERVE_AIALL_URL, timeout=10)

        if r.status_code != 200:
            return {
                "status": "error",
                "message": f"Backend returned HTTP {r.status_code}",
                "backend_response": r.text
            }

        return {
            "status": "ok",
            "message": "Model reload signal sent to serve_aiall backend.",
            "backend_response": r.text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

