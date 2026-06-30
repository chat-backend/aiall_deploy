# ============================================================
#  BACKEND STATS API (Final Version)
# routes/aiall_backend_stats.py
# ============================================================

from fastapi import APIRouter
import time
import requests
import psutil
import core.backends as be

router = APIRouter()

def normalize_backend_url(url: str):
    # Loại bỏ prefix nếu backend trả về dạng "http://host:port"
    return url.replace("http://", "").replace("https://", "")

def ping_backend(url: str):
    url = normalize_backend_url(url)
    start = time.time()

    try:
        r = requests.get(f"http://{url}/aiall/health", timeout=3)
        latency = (time.time() - start) * 1000
        return {
            "reachable": True,
            "latency_ms": round(latency, 2),
            "status": r.json()
        }
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {
            "reachable": False,
            "latency_ms": round(latency, 2),
            "error": str(e)
        }

@router.get("/system/backend-stats")
def backend_stats():
    backends = be.load_backends()
    drain_map = be.load_drain_status()

    # Đo CPU nhanh, không delay API
    cpu_load = psutil.cpu_percent(interval=0)

    stats = {}
    for backend in backends:
        stats[backend] = {
            "drain": drain_map.get(backend, False),
            "cpu_load_percent": cpu_load,
            "ping": ping_backend(backend)
        }

    return {
        "backend_stats": stats,
        "updated_at": time.time()
    }

