# ============================================================
#  MODEL CHECKSUM API (Improved)
# routes/aiall_model_checksum.py
# ============================================================

from fastapi import APIRouter
import os
import hashlib

router = APIRouter()

MODEL_DIR = "/root/aiall_deploy/aiall-merged"

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

@router.get("/system/model-checksum")
def model_checksum():
    if not os.path.exists(MODEL_DIR):
        return {"error": "Model directory not found."}

    checksums = {}

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, MODEL_DIR)

            try:
                checksums[rel_path] = sha256_of_file(full_path)
            except Exception as e:
                checksums[rel_path] = f"error: {e}"

    return {
        "model_dir": MODEL_DIR,
        "checksums": checksums
    }

