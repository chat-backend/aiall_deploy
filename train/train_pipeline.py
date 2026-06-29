# train/train_pipeline.py
#!/usr/bin/env python3
"""
AIALL – FULL TRAIN PIPELINE (Linux Production, GPU/CPU-aware)
-------------------------------------------------------------
- Nếu có GPU:
    - Train LoRA trên base model Qwen2.5-1.5B
    - Lưu adapter vào aiall-lora/
    - Merge LoRA vào full model aiall-merged/
    - Cập nhật MODEL_TOKEN
    - Đăng ký backend vào gateway (Nginx)
    - Smoke test inference với lớp Real-Time Context

- Nếu KHÔNG có GPU:
    - Chỉ preview dataset (nhẹ, chạy được trên CPU)
    - Đăng ký backend vào gateway (Nginx)
    - Bỏ qua train, merge, smoke test để tránh treo trên CPU
"""

import platform
import sys
import os

import torch  # thêm để kiểm tra GPU

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


def has_gpu():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def step_1_preview():
    print("\n=== STEP 1: PREVIEW DATASET ===")
    _, tokenizer = load_base_model()
    tokenized = load_dataset_tokenized(tokenizer)
    print("Dataset size:", len(tokenized))
    print("Sample:", tokenized[0])


def step_2_train():
    print("\n=== STEP 2: TRAIN LoRA ADAPTER ===")
    train_aiall()
    if not os.path.exists(LORA_DIR):
        print("[ERROR] Sau khi train không tìm thấy thư mục LoRA:", LORA_DIR)
        sys.exit(1)
    print("=== STEP 2 DONE: LoRA adapter saved to aiall-lora/ ===")


def step_3_merge():
    print("\n=== STEP 3: MERGE LoRA → FULL MODEL ===")

    if not os.path.exists(LORA_DIR):
        print("[ERROR] Không tìm thấy thư mục LoRA:", LORA_DIR)
        print("Bạn cần chạy STEP 2 trước khi merge.")
        sys.exit(1)

    merge_lora()

    if not os.path.exists(MERGED_DIR):
        print("[ERROR] Merge xong nhưng không tìm thấy merged model:", MERGED_DIR)
        sys.exit(1)

    print("=== STEP 3 DONE: Merged model saved to aiall-merged/ ===")


def step_4_inference_smoke_test():
    print("\n=== STEP 4: SMOKE TEST INFERENCE (WITH REAL-TIME CONTEXT) ===")

    if not os.path.exists(MERGED_DIR):
        print("[ERROR] Không tìm thấy merged model:", MERGED_DIR)
        print("Bạn cần chạy STEP 3 trước khi inference.")
        sys.exit(1)

    model, tokenizer = load_aiall_for_inference()
    prompt = "Xin chào AIALL, hãy giới thiệu ngắn gọn về bản thân."
    response = chat(model, tokenizer, prompt)
    print("Prompt:", prompt)
    print("Response:", response)


def step_5_register_backend():
    print("\n=== STEP 5: REGISTER BACKEND INTO GATEWAY ===")
    register_aiall_backend("127.0.0.1", 8000)
    print("=== STEP 5 DONE: Backend registered ===")


def main():
    ensure_linux()

    if has_gpu():
        # FULL PIPELINE – GPU MODE
        print("=== AIALL FULL TRAIN PIPELINE START (GPU MODE) ===")
        step_1_preview()
        step_2_train()
        step_3_merge()
        step_4_inference_smoke_test()
        step_5_register_backend()
        print("\n=== AIALL FULL TRAIN PIPELINE COMPLETE (GPU MODE) ===")
    else:
        # CPU MODE – NO TRAIN
        print("=== AIALL TRAIN PIPELINE START (CPU MODE, NO GPU) ===")
        print("⚠ WARNING: Không có GPU → bỏ qua train LoRA, merge, smoke test.")
        step_1_preview()
        step_5_register_backend()
        print("\n=== AIALL TRAIN PIPELINE COMPLETE (CPU MODE) ===")


if __name__ == "__main__":
    main()


