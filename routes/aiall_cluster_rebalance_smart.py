# ============================================================
#  CLUSTER REBALANCE SMART API (Improved)
# routes/aiall_cluster_rebalance_smart.py
# ============================================================

from fastapi import APIRouter
import time
import requests
import core.backends as be

router = APIRouter()

def normalize_backend_url(url: str):
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
            "status": r.json(),
            "error": None
        }
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {
            "reachable": False,
            "latency_ms": round(latency, 2),
            "status": None,
            "error": str(e)
        }

@router.post("/system/cluster-rebalance-smart")
def cluster_rebalance_smart():
    backends = be.load_backends()
    drain_map = be.load_drain_status()

    actions = []
    evaluation = {}

    for backend in backends:
        info = ping_backend(backend)
        evaluation[backend] = info

        # RULE 1: unreachable → drain
        if not info["reachable"]:
            if not drain_map.get(backend, False):
                be.drain_backend(backend)
                actions.append({"backend": backend, "action": "drain", "reason": "unreachable"})
            continue

        # RULE 2: high latency → drain
        if info["latency_ms"] > 500:
            if not drain_map.get(backend, False):
                be.drain_backend(backend)
                actions.append({"backend": backend, "action": "drain", "reason": "high_latency"})
            continue

        # RULE 3: healthy → undrain
        if info["latency_ms"] < 200:
            if drain_map.get(backend, False):
                be.undrain_backend(backend)
                actions.append({"backend": backend, "action": "undrain", "reason": "healthy"})

    return {
        "evaluation": evaluation,
        "actions": actions,
        "updated_at": time.time(),
        "message": "Smart cluster rebalance executed."
    }

