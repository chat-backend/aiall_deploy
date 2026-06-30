# ============================================================
#  MODEL VERSION HISTORY EXTENDED API (Clean Version – No Regex Errors)
#  routes/aiall_model_history_extended.py
# ============================================================

from fastapi import APIRouter
import os
from datetime import datetime

router = APIRouter()

HISTORY_FILE = "/root/aiall_deploy/model_history.log"

@router.get("/system/model-version-history-extended")
def model_version_history_extended():
    if not os.path.exists(HISTORY_FILE):
        return {
            "history": [],
            "updated_at": datetime.now().isoformat(),
            "message": "No history available."
        }

    entries = []

    with open(HISTORY_FILE, "r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            # Mặc định
            action = None
            timestamp = None
            model_dir = None
            version = None
            checksum = None

            # Ví dụ raw:
            # [TRAIN] 2025-01-01T12:00:00 model_dir=aiall-lora version=unknown checksum=none

            parts = raw.split()

            # Bắt action
            if parts[0].startswith("[") and parts[0].endswith("]"):
                action = parts[0].replace("[", "").replace("]", "")

            # Bắt timestamp
            if len(parts) > 1:
                timestamp = parts[1]

            # Bắt các trường key=value
            for p in parts[2:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k == "model_dir":
                        model_dir = v
                    elif k == "version":
                        version = v
                    elif k == "checksum":
                        checksum = v

            entries.append({
                "action": action,
                "timestamp": timestamp,
                "model_dir": model_dir,
                "version": version,
                "checksum": checksum,
                "raw": raw
            })

    return {
        "history": entries,
        "updated_at": datetime.now().isoformat(),
        "message": "Extended model version history loaded."
    }

