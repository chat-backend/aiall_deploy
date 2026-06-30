# train/train_pipeline.py
#!/usr/bin/env python3
"""
AIALL – FULL TRAIN PIPELINE (CPU-Optimized, Extreme + Smart Auto-Rollback)
---------------------------------------------------------------------
- STEP 0: Environment & sanity checks
- STEP 1: Validate & preview dataset
- STEP 2: Train LoRA adapter (fast)
- STEP 3: Validate LoRA + merge → full model
- STEP 4: Deep smoke test inference (multi-prompt)
- STEP 5: Register backend into gateway

- Smart Auto-rollback:
  - Backup aiall-merged trước khi train
  - Nếu bất kỳ step nào fail → restore backup
  - Log rõ nguyên nhân lỗi từng step
"""

import platform
import sys
import os
import shutil
import traceback

from train.aiall_train import (
    load_base_model,
    load_dataset_tokenized,
    train_aiall,
    merge_lora,
    load_aiall_for_inference,
    chat,
    register_aiall_backend,
)

IS_LINUX = platform.system().lower().startswith("linux")
LORA_DIR = "aiall-lora"
MERGED_DIR = "aiall-merged"
BACKUP_DIR = "/root/aiall_merged_backup_pipeline_extreme"
LOG_FILE = "/root/aiall_train_pipeline_extreme.log"


def log(msg: str):
    print(msg)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        # nếu log fail thì bỏ qua, không chặn pipeline
        pass


def ensure_linux():
    if not IS_LINUX:
        log("[ERROR] Full train pipeline chỉ hỗ trợ trên Linux.")
        sys.exit(1)


def ensure_paths():
    cwd = os.getcwd()
    log(f"[ENV] Current working directory: {cwd}")
    if not os.path.exists("train"):
        log("[ERROR] Không tìm thấy thư mục train/ trong cwd.")
        sys.exit(1)


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


def step_0_env_check():
    log("\n=== STEP 0: ENVIRONMENT & SANITY CHECKS ===")
    ensure_linux()
    ensure_paths()
    log("[ENV] Linux OK, paths OK.")


def step_1_validate_and_preview():
    log("\n=== STEP 1: VALIDATE & PREVIEW DATASET (CPU MODE, FAST) ===")
    try:
        _, tokenizer = load_base_model()
        tokenized = load_dataset_tokenized(tokenizer)
        size = len(tokenized)
        log(f"[DATA] Dataset size sau tokenize: {size}")
        if size == 0:
            raise RuntimeError("Dataset rỗng sau khi tokenize.")

        # preview vài mẫu
        preview_count = min(3, size)
        for i in range(preview_count):
            log(f"[DATA SAMPLE {i}] {tokenized[i]}")
    except Exception as e:
        log(f"[ERROR] STEP 1 failed: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 1 – dataset invalid")
        sys.exit(1)

    log("=== STEP 1 DONE: Dataset hợp lệ, có dữ liệu để train. ===")


def step_2_train_lora():
    log("\n=== STEP 2: TRAIN LoRA ADAPTER (CPU MODE, FAST) ===")
    log("⚠ CPU MODE: Training đã được tối ưu để ~30 phút (fast config).")
    try:
        train_aiall()
    except Exception as e:
        log(f"[ERROR] STEP 2 failed during training: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 2 – training failed")
        sys.exit(1)

    if not os.path.exists(LORA_DIR):
        log(f"[ERROR] Sau khi train không tìm thấy thư mục LoRA: {LORA_DIR}")
        rollback_merged_model("STEP 2 – LoRA dir missing")
        sys.exit(1)

    log("=== STEP 2 DONE: LoRA adapter saved to aiall-lora/ ===")


def step_3_validate_and_merge():
    log("\n=== STEP 3: VALIDATE LoRA & MERGE → FULL MODEL (CPU MODE, FAST) ===")

    if not os.path.exists(LORA_DIR):
        log(f"[ERROR] Không tìm thấy thư mục LoRA: {LORA_DIR}")
        log("Bạn cần chạy STEP 2 trước khi merge.")
        rollback_merged_model("STEP 3 – LoRA dir missing")
        sys.exit(1)

    # quick sanity check: LoRA dir không rỗng
    if not os.listdir(LORA_DIR):
        log(f"[ERROR] Thư mục LoRA rỗng: {LORA_DIR}")
        rollback_merged_model("STEP 3 – LoRA empty")
        sys.exit(1)

    try:
        merge_lora()
    except Exception as e:
        log(f"[ERROR] STEP 3 failed during merge: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 3 – merge failed")
        sys.exit(1)

    if not os.path.exists(MERGED_DIR):
        log(f"[ERROR] Merge xong nhưng không tìm thấy merged model: {MERGED_DIR}")
        rollback_merged_model("STEP 3 – merged dir missing")
        sys.exit(1)

    # quick sanity check: merged dir không rỗng
    if not os.listdir(MERGED_DIR):
        log(f"[ERROR] Thư mục merged rỗng: {MERGED_DIR}")
        rollback_merged_model("STEP 3 – merged empty")
        sys.exit(1)

    log("=== STEP 3 DONE: Merged model saved to aiall-merged/ ===")


def step_4_deep_smoke_test():
    log("\n=== STEP 4: DEEP SMOKE TEST INFERENCE (CPU MODE, FAST) ===")

    if not os.path.exists(MERGED_DIR):
        log(f"[ERROR] Không tìm thấy merged model: {MERGED_DIR}")
        log("Bạn cần chạy STEP 3 trước khi inference.")
        rollback_merged_model("STEP 4 – merged dir missing")
        sys.exit(1)

    try:
        model, tokenizer = load_aiall_for_inference()
        prompts = [
            "Xin chào AIALL, hãy giới thiệu ngắn gọn về bản thân.",
            "Giải thích ngắn gọn về khái niệm 'LoRA' trong huấn luyện mô hình ngôn ngữ.",
            "Hãy trả lời bằng tiếng Việt: Tại sao cần kiểm tra sức khỏe mô hình sau khi train?",
        ]
        for i, p in enumerate(prompts):
            response = chat(model, tokenizer, p)
            log(f"[SMOKE {i}] Prompt: {p}")
            log(f"[SMOKE {i}] Response: {response}")
            if not response or len(response.strip()) == 0:
                raise RuntimeError(f"Response rỗng cho prompt index {i}")
    except Exception as e:
        log(f"[ERROR] STEP 4 failed during inference: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 4 – inference failed")
        sys.exit(1)

    log("=== STEP 4 DONE: Deep smoke test inference OK (multi-prompt). ===")


def step_5_register_backend():
    log("\n=== STEP 5: REGISTER BACKEND INTO GATEWAY (CPU MODE) ===")

    if not os.path.exists(MERGED_DIR):
        log("[WARN] Chưa có merged model, nhưng vẫn cố đăng ký backend.")
        log("Khuyến nghị: chỉ register backend sau khi STEP 3 hoàn tất.")

    try:
        register_aiall_backend("127.0.0.1", 8001)
    except Exception as e:
        log(f"[ERROR] STEP 5 failed during backend registration: {e}")
        log(traceback.format_exc())
        rollback_merged_model("STEP 5 – backend registration failed")
        sys.exit(1)

    log("=== STEP 5 DONE: Backend registered into vLLM cluster ===")
    log("=== URL CHÍNH THỨC: https://api.aiallplatform.com/aiall/ ===")


def main():
    log("=== AIALL FULL TRAIN PIPELINE EXTREME START (CPU MODE, FAST + SMART AUTO-ROLLBACK) ===")
    step_0_env_check()
    backup_merged_model()
    step_1_validate_and_preview()
    step_2_train_lora()
    step_3_validate_and_merge()
    step_4_deep_smoke_test()
    step_5_register_backend()
    log("\n=== AIALL FULL TRAIN PIPELINE EXTREME COMPLETE (CPU MODE, FAST) ===")


if __name__ == "__main__":
    main()






