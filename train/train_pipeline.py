# train/train_pipeline.py
#!/usr/bin/env python3
"""
AIALL – FULL TRAIN PIPELINE (CPU-Optimized, Balanced Speed & Quality)
---------------------------------------------------------------------
- STEP 1: Preview dataset (CPU)
- STEP 2: Train LoRA adapter (CPU, optimized)
- STEP 3: Merge LoRA → full model (CPU)
- STEP 4: Smoke test inference (CPU)
- STEP 5: Register backend into vLLM gateway (Linux-only)

Đồng bộ hoàn toàn với:
- train/aiall_train.py
- train/serve_aiall.py
"""

import platform
import sys
import os

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


def ensure_linux():
    if not IS_LINUX:
        print("[ERROR] Full train pipeline chỉ hỗ trợ trên Linux.")
        sys.exit(1)


def step_1_preview():
    print("\n=== STEP 1: PREVIEW DATASET (CPU MODE, FAST) ===")
    try:
        # Chỉ load tokenizer, không cần model để giảm thời gian
        _, tokenizer = load_base_model()
        tokenized = load_dataset_tokenized(tokenizer)
        print("Dataset size:", len(tokenized))
        if len(tokenized) > 0:
            print("Sample:", tokenized[0])
        else:
            print("[WARN] Dataset rỗng sau khi tokenize.")
    except Exception as e:
        print(f"[ERROR] STEP 1 failed: {e}")
        sys.exit(1)


def step_2_train():
    print("\n=== STEP 2: TRAIN LoRA ADAPTER (CPU MODE, OPTIMIZED) ===")
    print("⚠ CPU MODE: Training sẽ chậm nhưng đã được tối ưu để nhẹ nhất có thể.")
    try:
        train_aiall()
    except Exception as e:
        print(f"[ERROR] STEP 2 failed during training: {e}")
        sys.exit(1)

    if not os.path.exists(LORA_DIR):
        print("[ERROR] Sau khi train không tìm thấy thư mục LoRA:", LORA_DIR)
        sys.exit(1)

    print("=== STEP 2 DONE: LoRA adapter saved to aiall-lora/ ===")


def step_3_merge():
    print("\n=== STEP 3: MERGE LoRA → FULL MODEL (CPU MODE, OPTIMIZED) ===")

    if not os.path.exists(LORA_DIR):
        print("[ERROR] Không tìm thấy thư mục LoRA:", LORA_DIR)
        print("Bạn cần chạy STEP 2 trước khi merge.")
        sys.exit(1)

    try:
        merge_lora()
    except Exception as e:
        print(f"[ERROR] STEP 3 failed during merge: {e}")
        sys.exit(1)

    if not os.path.exists(MERGED_DIR):
        print("[ERROR] Merge xong nhưng không tìm thấy merged model:", MERGED_DIR)
        sys.exit(1)

    print("=== STEP 3 DONE: Merged model saved to aiall-merged/ ===")


def step_4_inference_smoke_test():
    print("\n=== STEP 4: SMOKE TEST INFERENCE (CPU MODE, OPTIMIZED) ===")

    if not os.path.exists(MERGED_DIR):
        print("[ERROR] Không tìm thấy merged model:", MERGED_DIR)
        print("Bạn cần chạy STEP 3 trước khi inference.")
        sys.exit(1)

    try:
        model, tokenizer = load_aiall_for_inference()
        prompt = "Xin chào AIALL, hãy giới thiệu ngắn gọn về bản thân."
        response = chat(model, tokenizer, prompt)
        print("Prompt:", prompt)
        print("Response:", response)
    except Exception as e:
        print(f"[ERROR] STEP 4 failed during inference: {e}")
        sys.exit(1)

    print("=== STEP 4 DONE: Smoke test inference OK ===")


def step_5_register_backend():
    print("\n=== STEP 5: REGISTER BACKEND INTO GATEWAY (CPU MODE) ===")

    if not os.path.exists(MERGED_DIR):
        print("[WARN] Chưa có merged model, nhưng vẫn cố đăng ký backend.")
        print("Khuyến nghị: chỉ register backend sau khi STEP 3 hoàn tất.")

    try:
        register_aiall_backend("127.0.0.1", 8001)
    except Exception as e:
        print(f"[ERROR] STEP 5 failed during backend registration: {e}")
        sys.exit(1)

    print("=== STEP 5 DONE: Backend registered into vLLM cluster ===")
    print("=== URL CHÍNH THỨC: https://api.aiallplatform.com/aiall/ ===")


def main():
    ensure_linux()

    print("=== AIALL FULL TRAIN PIPELINE START (CPU MODE, OPTIMIZED) ===")
    step_1_preview()
    step_2_train()
    step_3_merge()
    step_4_inference_smoke_test()
    step_5_register_backend()
    print("\n=== AIALL FULL TRAIN PIPELINE COMPLETE (CPU MODE) ===")


if __name__ == "__main__":
    main()





