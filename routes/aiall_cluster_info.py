# ============================================================
#  CLUSTER INFO API (Improved)
# routes/aiall_cluster_info.py
# ============================================================

from fastapi import APIRouter
import psutil
import core.backends as be
from core.health_cluster import health_check
import time

router = APIRouter()

@router.get("/system/cluster-info")
def cluster_info():
    backends = be.load_backends()
    drain_map = be.load_drain_status()

    try:
        cluster_status = health_check(return_status=True)
    except Exception:
        cluster_status = "unknown"

    # Không delay API
    cpu_load = psutil.cpu_percent(interval=0)

    return {
        "cluster_status": cluster_status,
        "backends": backends,
        "drain_status": drain_map,
        "cpu_load_percent": cpu_load,
        "updated_at": time.time()
    }

