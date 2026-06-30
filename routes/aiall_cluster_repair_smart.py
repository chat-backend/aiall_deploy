# ============================================================
#  CLUSTER REPAIR SMART API
# routes/aiall_cluster_repair_smart.py
# ============================================================

from fastapi import APIRouter
import subprocess
import requests
import core.backends as be
from core.health_cluster import health_check

router = APIRouter()

SERVICE_MAP = {
    "127.0.0.1:8000": "aiall-vllm-backend",
    "127.0.0.1:8001": "aiall-serve-backend",
}

def restart_service(service_name: str):
    try:
        subprocess.run(["systemctl", "restart", service_name], check=True)
        return {"service": service_name, "status": "restarted"}
    except Exception as e:
        return {"service": service_name, "status": "error", "error": str(e)}

def ping_backend(url: str):
    try:
        r = requests.get(f"http://{url}/aiall/health", timeout=3)
        return {"reachable": True, "status": r.json(), "error": None}
    except Exception as e:
        return {"reachable": False, "status": None, "error": str(e)}

@router.post("/system/cluster-repair-smart")
def cluster_repair_smart():
    backends = be.load_backends()
    drain_map = be.load_drain_status()

    try:
        cluster_status = health_check(return_status=True)
    except Exception:
        cluster_status = "unknown"

    actions = []
    evaluation = {}

    for backend in backends:
        info = ping_backend(backend)
        evaluation[backend] = {
            "reachable": info["reachable"],
            "drain": drain_map.get(backend, False),
            "error": info["error"],
        }

        # RULE 0: Backend đang drain → không đụng vào
        if drain_map.get(backend, False):
            actions.append({"backend": backend, "action": "skip", "reason": "drain"})
            continue

        # RULE 1: Backend unreachable → restart service
        if not info["reachable"]:
            svc = SERVICE_MAP.get(backend)
            if svc:
                res = restart_service(svc)
                actions.append({"backend": backend, "action": "restart", "reason": "unreachable", "service_result": res})
            else:
                actions.append({"backend": backend, "action": "skip", "reason": "no_service_map"})
            continue

        # RULE 2: Backend reachable → không restart
        actions.append({"backend": backend, "action": "skip", "reason": "healthy_or_minor_issue"})

    return {
        "cluster_status_before": cluster_status,
        "evaluation": evaluation,
        "repair_actions": actions,
        "message": "Smart cluster repair executed."
    }
