#!/usr/bin/env python3
"""
AIALL – TRAIN MODULE TEST SUITE
--------------------------------
Chạy được trên Windows Dev Mode và Linux Production.

Bao gồm:
- Test load base model
- Test load dataset
- Test LoRA build
- Test training (Linux-only)
- Test merge LoRA
- Test inference
- Test backend registration
"""

import platform
import torch

from train.aiall_train import (
    load_base_model,
    load_dataset_tokenized,
    build_lora_model,
    train_aiall,
    merge_lora,
    load_aiall_for_inference,
    chat,
    register_aiall_backend,
)

IS_LINUX = platform.system().lower().startswith("linux")


# ============================================================
# 1. TEST LOAD BASE MODEL
# ============================================================

def test_load_base_model():
    print("\n=== TEST: LOAD BASE MODEL ===")
    model, tokenizer = load_base_model()
    print("Model loaded:", type(model))
    print("Tokenizer loaded:", type(tokenizer))


# ============================================================
# 2. TEST LOAD DATASET
# ============================================================

def test_load_dataset():
    print("\n=== TEST: LOAD DATASET ===")
    _, tokenizer = load_base_model()
    tokenized = load_dataset_tokenized(tokenizer)
    print("Dataset size:", len(tokenized))
    print("Sample:", tokenized[0])


# ============================================================
# 3. TEST BUILD LORA MODEL
# ============================================================

def test_build_lora():
    print("\n=== TEST: BUILD LoRA MODEL ===")
    base_model, _ = load_base_model()
    lora_model = build_lora_model(base_model)
    print("LoRA model built:", type(lora_model))


# ============================================================
# 4. TEST TRAIN (LINUX ONLY)
# ============================================================

def test_train():
    print("\n=== TEST: TRAIN AIALL ===")
    if not IS_LINUX:
        print("Skipping training — Windows Dev Mode")
        return
    train_aiall()


# ============================================================
# 5. TEST MERGE LoRA → FULL MODEL
# ============================================================

def test_merge():
    print("\n=== TEST: MERGE LoRA → FULL MODEL ===")
    merge_lora()


# ============================================================
# 6. TEST LOAD MERGED MODEL FOR INFERENCE
# ============================================================

def test_inference():
    print("\n=== TEST: INFERENCE ===")
    model, tokenizer = load_aiall_for_inference()
    response = chat(model, tokenizer, "Xin chào AIALL, bạn khỏe không?")
    print("Model response:", response)


# ============================================================
# 7. TEST REGISTER BACKEND
# ============================================================

def test_register_backend():
    print("\n=== TEST: REGISTER BACKEND ===")
    if not IS_LINUX:
        print("Skipping backend registration — Windows Dev Mode")
        return
    register_aiall_backend("127.0.0.1", 8000)


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == "__main__":
    print("=== AIALL TRAIN MODULE TEST SUITE START ===")

    test_load_base_model()
    test_load_dataset()
    test_build_lora()

    test_train()          # Linux-only
    test_merge()
    test_inference()

    test_register_backend()  # Linux-only

    print("\n=== AIALL TRAIN MODULE TEST SUITE COMPLETE ===")
