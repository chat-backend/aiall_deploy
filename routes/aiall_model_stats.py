# ============================================================
#  MODEL STATS API (Improved)
# routes/aiall_model_stats.py
# ============================================================

from fastapi import APIRouter
import os
import re
from datetime import datetime

router = APIRouter()

LOG_FILE = "/root/aiall_deploy/gateway_request.log"

@router.get("/system/model-stats")
def model_stats():
    if not os.path.exists(LOG_FILE):
        return {
            "request_count": 0,
            "avg_latency_ms": 0,
            "total_tokens": 0,
            "message": "No stats available yet."
        }

    request_count = 0
    total_latency = 0
    total_tokens = 0

    with open(LOG_FILE, "r") as f:
        for line in f:
            if "/v1/chat/completions" not in line:
                continue

            request_count += 1

            lat_match = re.search(r"latency=(\d+)ms", line)
            tok_match = re.search(r"tokens=(\d+)", line)

            if lat_match:
                total_latency += int(lat_match.group(1))

            if tok_match:
                total_tokens += int(tok_match.group(1))

    avg_latency = total_latency / request_count if request_count > 0 else 0

    return {
        "request_count": request_count,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "updated_at": datetime.now().isoformat()
    }

