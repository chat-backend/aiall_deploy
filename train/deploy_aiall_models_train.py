# train/deploy_aiall_models_train.py
#!/usr/bin/env python3
"""
AIALL – LoRA Training & Real-Time Inference Module
CPU-Optimized, QUALITY + VIETNAMESE + AIALL STYLE

Lifecycle:
- System & CPU setup
- Paths & global config
- Base model & tokenizer
- Dataset loading & preprocessing
- Training (via training_module)
- LoRA merge → full model
- Backend registration
- Inference loading
- Chat interface (AIALL style)
"""

import os
import platform
from datetime import datetime

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.aiall_style import aiall_chat
from train.training_module import train_lora_model
import core.backends as be

# ============================================================
#  SYSTEM & CPU SETUP
# ============================================================

def is_real_linux() -> bool:
    return platform.system().lower() == "linux"

IS_LINUX = is_real_linux()

if IS_LINUX:
    import core.nginx as ngx
else:
    ngx = None

# CPU optimization for 1.5B on CPU
torch.backends.mkldnn.enabled = True
torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.set_default_dtype(torch.float32)

# ============================================================
#  PATHS & GLOBAL CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DATA_FILE = os.path.join(os.path.dirname(__file__), "aiall_data.jsonl")

LORA_OUTPUT_DIR = "aiall-lora"
MERGED_OUTPUT_DIR = "aiall-merged"

TRAIN_BATCH = 4
GRAD_ACC = 2
LR = 1e-4
EPOCHS = 2
MAX_LEN = 256
MAX_SAMPLES = 300

LOG_FILE = os.path.expanduser("~/aiall_logs/model_history.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

# ============================================================
#  BASE MODEL & TOKENIZER
# ============================================================

def load_base_model():
    """
    Load base Qwen model + tokenizer trên CPU,
    cấu hình cho training LoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer

# ============================================================
#  DATASET LOADING & PREPROCESSING
# ============================================================

def load_dataset_tokenized(tokenizer):
    """
    Load dataset từ JSONL và tokenize theo format Instruction/Answer
    để mô hình quen phong cách AIALL.
    """
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"[ERROR] Dataset file not found: {DATA_FILE}")

    dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]

    if len(dataset) > MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    def preprocess(example):
        text = f"Instruction: {example['instruction']}\nAnswer: {example['output']}"
        tokens = tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(preprocess, remove_columns=dataset.column_names)

# ============================================================
#  TRAINING (LoRA VIA MODULE)
# ============================================================

def train_aiall():
    """
    Train LoRA adapter cho AIALL trên Linux thật.
    """
    if not IS_LINUX:
        raise SystemExit("[ERROR] Training is only supported on Linux.")

    base_model, tokenizer = load_base_model()
    tokenized_dataset = load_dataset_tokenized(tokenizer)

    train_lora_model(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        output_dir=LORA_OUTPUT_DIR,
        log_file=LOG_FILE,
        train_batch=TRAIN_BATCH,
        grad_acc=GRAD_ACC,
        lr=LR,
        epochs=EPOCHS,
        warmup_steps=40,
    )

# ============================================================
#  MERGE LoRA → FULL MODEL
# ============================================================

def merge_lora():
    """
    Merge LoRA adapter vào full Qwen model và lưu cùng tokenizer.
    """
    if not os.path.exists(LORA_OUTPUT_DIR):
        raise SystemExit("[ERROR] LoRA adapter not found.")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")

    merged = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    merged = merged.merge_and_unload()

    merged.save_pretrained(MERGED_OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    with open(LOG_FILE, "a") as h:
        h.write(
            f"[MERGE] {datetime.now().isoformat()} model_dir={MERGED_OUTPUT_DIR}\n"
        )

# ============================================================
#  BACKEND REGISTRATION
# ============================================================

def register_aiall_backend(host: str = "127.0.0.1", port: int = 8001):
    """
    Đăng ký backend AIALL vào gateway nội bộ và (nếu có) Nginx upstream.
    """
    backend = f"{host}:{port}"

    try:
        be.add_backend(backend)
    except Exception:
        # Không để backend lỗi làm crash hệ thống
        return

    if not IS_LINUX or ngx is None:
        return

    nginx_conf_dir = "/etc/nginx/conf.d"
    if not os.path.exists(nginx_conf_dir):
        return

    try:
        ngx.generate_upstream_block()
        ngx.reload_nginx()
    except Exception:
        # Nginx lỗi thì bỏ qua, backend vẫn hoạt động nội bộ
        return

# ============================================================
#  INFERENCE LOADING
# ============================================================

def load_aiall_for_inference():
    """
    Load merged model + tokenizer cho inference thời gian thực trên CPU.
    """
    if not os.path.exists(MERGED_OUTPUT_DIR):
        raise SystemExit("[ERROR] Merged model not found.")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForCausalLM.from_pretrained(MERGED_OUTPUT_DIR, device_map="cpu")
    model.config.use_cache = True
    model.eval()

    return model, tokenizer

# ============================================================
#  CHAT INTERFACE (AIALL STYLE)
# ============================================================

def chat(model, tokenizer, prompt: str):
    """
    Hàm chat chính, sử dụng aiall_chat để áp dụng phong cách AIALL:
    - Ưu tiên tiếng Việt
    - Giải thích rõ ràng, thân thiện, có cấu trúc
    """
    if not prompt.strip():
        return "Bạn chưa nhập nội dung."
    return aiall_chat(model, tokenizer, prompt)
