# train/aiall_full_pipeline.py
#!/usr/bin/env python3
"""
AIALL – FULL TRAIN PIPELINE (CPU-Optimized, Smart Auto-Rollback)
---------------------------------------------------------------------
Lifecycle:
- STEP 0: Environment & sanity checks
- STEP 1: Validate & preview dataset
- STEP 2: Train LoRA adapter (via deploy_aiall_models_train)
- STEP 3: Validate LoRA + merge → full model
- STEP 4: Deep smoke test inference (SAFE CHAT)
- STEP 5: Register backend into gateway
"""

import sys
import os
import shutil
import traceback

from train.deploy_aiall_models_train import (
    load_base_model,
    load_dataset_tokenized,
    train_aiall,
    merge_lora,
    load_aiall_for_inference,
    register_aiall_backend,
)

LORA_DIR = "aiall-lora"
MERGED_DIR = "aiall-merged"

BACKUP_DIR = os.path.expanduser("~/aiall_merged_backup_pipeline_extreme")
LOG_FILE = os.path.expanduser("~/aiall_train_pipeline_extreme.log")


def log(msg: str):
    print(msg)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ============================================================
#  ENVIRONMENT & PATH CHECKS
# ============================================================

def ensure_paths():
    cwd = os.getcwd()
    log(f"[ENV] Current working directory: {cwd}")
    if not os.path.exists("train"):
        log("[ERROR] Không tìm thấy thư mục train/ trong cwd.")
        sys.exit(1)


def step_0_env_check():
    log("\n=== STEP 0: ENVIRONMENT & SANITY CHECKS ===")
    ensure_paths()
    log("[ENV] Paths OK, CPU-SAFE pipeline ready.")


# ============================================================
#  BACKUP & ROLLBACK MERGED MODEL
# ============================================================

def backup_merged_model():
    if os.path.exists(MERGED_DIR):
        log(f"[BACKUP] Backup merged model từ {MERGED_DIR} → {BACKUP_DIR}")
        try:
            if os.path.exists(BACKUP_DIR):
                shutil.rmtree(BACKUP_DIR)
            shutil.copytree(MERGED_DIR, BACKUP_DIR)
            log("[BACKUP] Backup merged model thành công.")
        except Exception as e:
            log(f"[WARN] Backup merged model thất bại: {e}")
            log(traceback.format_exc())
    else:
        log("[BACKUP] Không có merged model hiện tại, bỏ qua backup.")


def rollback_merged_model(reason: str):
    log(f"[ROLLBACK] Triggered rollback do lỗi: {reason}")
    if os.path.exists(BACKUP_DIR):
        log(f"[ROLLBACK] Khôi phục merged model từ {BACKUP_DIR} → {MERGED_DIR}")
        try:
            if os.path.exists(MERGED_DIR):
                shutil.rmtree(MERGED_DIR)
            shutil.copytree(BACKUP_DIR, MERGED_DIR)
            log("[ROLLBACK] Khôi phục merged model thành công.")
        except Exception as e:
            log(f"[ROLLBACK] Khôi phục merged model thất bại: {e}")
            log(traceback.format_exc())
    else:
        log("[ROLLBACK] Không có backup để khôi phục.")


# ============================================================
#  STEP 1: VALIDATE & PREVIEW DATASET
# ============================================================

def step_1_validate_and_preview():
    log("\n=== STEP 1: VALIDATE & PREVIEW DATASET ===")
    try:
        base_model, tokenizer = load_base_model()
        del base_model

        tokenized = load_dataset_tokenized(tokenizer)
        size = len(tokenized)
        log(f"[DATA] Dataset size sau tokenize: {size}")
        if size == 0:
            raise RuntimeError("Dataset rỗng sau khi tokenize.")

        preview_count = min(3, size)
        for i in range(preview_count):
            log(f"[DATA SAMPLE {i}] {tokenized[i]}")
    except Exception as e:
        log(f"[ERROR] STEP 1 failed: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 1 – dataset invalid")
        sys.exit(1)

    log("=== STEP 1 DONE ===")


# ============================================================
#  STEP 2: TRAIN LoRA ADAPTER
# ============================================================

def step_2_train_lora():
    log("\n=== STEP 2: TRAIN LoRA ADAPTER ===")
    try:
        train_aiall()
    except Exception as e:
        log(f"[ERROR] STEP 2 failed: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 2 – training failed")
        sys.exit(1)

    if not os.path.exists(LORA_DIR) or not os.listdir(LORA_DIR):
        log("[ERROR] LoRA adapter không tồn tại hoặc rỗng.")
        rollback_merged_model("STEP 2 – LoRA invalid")
        sys.exit(1)

    log("=== STEP 2 DONE ===")


# ============================================================
#  STEP 3: MERGE LoRA → FULL MODEL
# ============================================================

def step_3_validate_and_merge():
    log("\n=== STEP 3: MERGE LoRA → FULL MODEL ===")

    if not os.path.exists(LORA_DIR) or not os.listdir(LORA_DIR):
        log("[ERROR] LoRA adapter không tồn tại hoặc rỗng.")
        rollback_merged_model("STEP 3 – LoRA invalid")
        sys.exit(1)

    try:
        merge_lora()
    except Exception as e:
        log(f"[ERROR] STEP 3 failed: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 3 – merge failed")
        sys.exit(1)

    if not os.path.exists(MERGED_DIR) or not os.listdir(MERGED_DIR):
        log("[ERROR] Merged model không tồn tại hoặc rỗng.")
        rollback_merged_model("STEP 3 – merged invalid")
        sys.exit(1)

    log("=== STEP 3 DONE ===")


# ============================================================
#  SAFE CHAT (DIRECT GENERATE)
# ============================================================

def safe_chat(model, tokenizer, prompt: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    if inputs["input_ids"].shape[1] == 0:
        return "[SMOKE] Input rỗng."

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ============================================================
#  STEP 4: DEEP SMOKE TEST
# ============================================================

def step_4_deep_smoke_test():
    log("\n=== STEP 4: DEEP SMOKE TEST INFERENCE ===")

    if not os.path.exists(MERGED_DIR):
        log("[ERROR] Merged model không tồn tại.")
        rollback_merged_model("STEP 4 – merged missing")
        sys.exit(1)

    try:
        model, tokenizer = load_aiall_for_inference()
        prompts = [
            "Xin chào AIALL, hãy giới thiệu ngắn gọn về bản thân.",
            "Giải thích ngắn gọn về khái niệm LoRA.",
            "Tại sao cần kiểm tra mô hình sau khi train?",
        ]
        for i, p in enumerate(prompts):
            response = safe_chat(model, tokenizer, p)
            log(f"[SMOKE {i}] Prompt: {p}")
            log(f"[SMOKE {i}] Response: {response}")
    except Exception as e:
        log(f"[ERROR] STEP 4 failed: {e}")
        log(traceback.format_exc())
        log("[WARN] Không rollback vì lỗi inference không phá model.")

    log("=== STEP 4 DONE ===")


# ============================================================
#  STEP 5: REGISTER BACKEND
# ============================================================

def step_5_register_backend():
    log("\n=== STEP 5: REGISTER BACKEND ===")

    try:
        register_aiall_backend("127.0.0.1", 8001)
    except Exception as e:
        log(f"[ERROR] STEP 5 failed: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 5 – backend failed")
        sys.exit(1)

    log("=== STEP 5 DONE ===")
    log("=== URL: https://api.aiallplatform.com/aiall/ ===")


# ============================================================
#  MAIN ENTRYPOINT
# ============================================================

def main():
    log("=== AIALL FULL TRAIN PIPELINE START ===")
    step_0_env_check()
    backup_merged_model()
    step_1_validate_and_preview()
    step_2_train_lora()
    step_3_validate_and_merge()
    step_4_deep_smoke_test()
    step_5_register_backend()
    log("=== AIALL FULL TRAIN PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
