# routes/aiall_model_hot_swap.py
# ============================================================

from fastapi import APIRouter
import requests
from pydantic import BaseModel

router = APIRouter()

SERVE_AIALL_URL = "http://127.0.0.1:8001/aiall/hot-swap"

class HotSwapRequest(BaseModel):
    model_dir: str

@router.post("/system/model-hot-swap")
def model_hot_swap(req: HotSwapRequest):
    try:
        r = requests.post(SERVE_AIALL_URL, json={"model_dir": req.model_dir}, timeout=10)
        return {
            "status": "ok",
            "message": "Hot-swap signal sent to serve_aiall backend.",
            "backend_response": r.json()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
