# train/aiall_train.py
#!/usr/bin/env python3
"""
AIALL – LoRA Training & Real-Time Inference Module
--------------------------------------------------
- Train LoRA trên base model Qwen2.5-1.5B
- Lưu adapter vào aiall-lora/
- Merge LoRA vào full model aiall-merged/
- Cập nhật MODEL_TOKEN
- Đăng ký backend vào gateway (Linux-only)
- Load merged model để inference
- Tích hợp lớp Real-Time Context (thời gian, web/db/events/finance/weather/news/calendar)
  để mô hình không bị "đóng băng" theo thời gian train.
"""

import os
import platform
import torch
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

from secrets import token_hex
from config import MODEL_TOKEN_FILE

IS_LINUX = platform.system().lower().startswith("linux")

# ===== Backend + Nginx integration =====
import core.backends as be
if IS_LINUX:
    import core.nginx as ngx
else:
    ngx = None  # Nginx không dùng trên Windows / non-Linux

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DATA_FILE = "aiall_data.jsonl"
LORA_OUTPUT_DIR = "aiall-lora"
MERGED_OUTPUT_DIR = "aiall-merged"


# ============================================================
#  REAL-TIME CONTEXT LAYER (STUBS)
# ============================================================

def build_realtime_context(prompt: str) -> str:
    """
    Xây dựng context thời gian thực để inject vào prompt.
    Đây là lớp "không đóng băng" – mô hình luôn biết thời gian hiện tại
    và có thể mở rộng ra web/db/finance/weather/news/calendar.

    Hiện tại: stub, bạn có thể nối với các service thực tế sau.
    """

    parts = []

    # Thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"[TIME] Current datetime: {current_time}")

    pl = prompt.lower()

    # Web search (stub)
    if any(w in pl for w in ["tìm", "search", "google", "tra cứu"]):
        parts.append("[WEB] Web search: (stub) dữ liệu thời gian thực sẽ được tích hợp tại đây.")

    # Database (stub)
    if "cơ sở dữ liệu" in pl or "database" in pl or "db" in pl:
        parts.append("[DB] Database: (stub) kết quả truy vấn DB sẽ được tích hợp tại đây.")

    # Event timeline (stub)
    if "sự kiện" in pl or "timeline" in pl:
        parts.append("[EVENTS] Timeline: (stub) danh sách sự kiện gần đây sẽ được tích hợp tại đây.")

    # Financial data (stub)
    if any(w in pl for w in ["giá", "bitcoin", "btc", "chứng khoán", "stock"]):
        parts.append("[FINANCE] Financial: (stub) giá tài chính thời gian thực sẽ được tích hợp tại đây.")

    # Weather data (stub)
    if "thời tiết" in pl or "weather" in pl:
        parts.append("[WEATHER] Weather: (stub) dữ liệu thời tiết thời gian thực sẽ được tích hợp tại đây.")

    # News data (stub)
    if "tin tức" in pl or "news" in pl or "báo" in pl:
        parts.append("[NEWS] News: (stub) tin tức mới nhất sẽ được tích hợp tại đây.")

    # Calendar awareness (stub)
    if any(w in pl for w in ["lịch", "ngày", "tháng", "năm", "calendar"]):
        parts.append("[CALENDAR] Calendar: hệ thống nhận biết ngày/tháng/năm hiện tại và bối cảnh thời gian.")

    # Hướng dẫn mô hình về tính thời gian thực
    parts.append(
        "[REAL-TIME SYSTEM NOTICE] "
        "Nếu câu hỏi liên quan đến dữ liệu có thể thay đổi theo thời gian "
        "(sự kiện, giá, tin tức, thời tiết, lịch, v.v.), "
        "hãy luôn cân nhắc rằng thông tin trong mô hình có thể đã lỗi thời. "
        "Nếu không chắc chắn, hãy nói: "
        "'Thông tin tôi có thể đã lỗi thời. Bạn muốn tôi kiểm tra dữ liệu mới nhất không?'"
    )

    return "\n".join(parts) + "\n\n"


# ============================================================
#  LOAD BASE MODEL
# ============================================================

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    return model, tokenizer


# ============================================================
#  LOAD + TOKENIZE DATASET
# ============================================================

def load_dataset_tokenized(tokenizer):
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"[ERROR] Dataset file not found: {DATA_FILE}")

    dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]

    def preprocess(example):
        instruction = example["instruction"]
        output = example["output"]
        text = f"Instruction: {instruction}\nAnswer: {output}"

        tokens = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(preprocess, remove_columns=dataset.column_names)


# ============================================================
#  BUILD LoRA MODEL
# ============================================================

def build_lora_model(base_model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_config)


# ============================================================
#  TRAIN AIALL (LoRA)
# ============================================================

def train_aiall():
    if not IS_LINUX:
        raise SystemExit("[ERROR] Training is only supported on Linux.")

    base_model, tokenizer = load_base_model()
    tokenized_dataset = load_dataset_tokenized(tokenizer)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir="aiall-model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        evaluation_strategy="no",
        fp16=True,
    )

    lora_model = build_lora_model(base_model)

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
    lora_model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    new_token = token_hex(64)
    MODEL_TOKEN_FILE.write_text(f"AIALL_MODEL_TOKEN={new_token}\n")
    print(f"=== NEW MODEL_TOKEN GENERATED === {new_token}")
    print("=== TRAINING DONE. ADAPTER SAVED TO aiall-lora ===")


# ============================================================
#  MERGE LoRA → FULL MODEL
# ============================================================

def merge_lora():
    print("=== MERGING LoRA INTO FULL MODEL ===")

    if not os.path.exists(LORA_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] LoRA adapter not found: {LORA_OUTPUT_DIR}")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    merged = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    merged = merged.merge_and_unload()

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_OUTPUT_DIR)
    print("=== MERGED MODEL SAVED TO aiall-merged ===")


# ============================================================
#  REGISTER AIALL BACKEND (URL + MODEL)
# ============================================================

def register_aiall_backend(host="127.0.0.1", port=8000):
    """
    Tự động đăng ký mô hình AIALL vào gateway (Linux-only):
    - add backend
    - generate upstream block
    - reload nginx
    - tạo URL chính thức: https://api.aiallplatform.com/aiall/
    """
    if not IS_LINUX:
        print("[WARN] Backend registration skipped — Linux-only feature.")
        return

    backend = f"{host}:{port}"
    print(f"=== REGISTER AIALL BACKEND: {backend} ===")

    be.add_backend(backend)
    ngx.generate_upstream_block()
    ngx.reload_nginx()

    print("=== AIALL BACKEND REGISTERED SUCCESSFULLY ===")
    print("=== URL CHÍNH THỨC: https://api.aiallplatform.com/aiall/ ===")


# ============================================================
#  LOAD AIALL FOR INFERENCE
# ============================================================

def load_aiall_for_inference():
    if not os.path.exists(MERGED_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] Merged model not found: {MERGED_OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MERGED_OUTPUT_DIR)
    model.eval()
    return model, tokenizer


# ============================================================
#  CHAT FUNCTION (WITH REAL-TIME CONTEXT)
# ============================================================

def chat(model, tokenizer, prompt: str):
    realtime_context = build_realtime_context(prompt)

    text = (
        realtime_context +
        f"Instruction: {prompt}\nAnswer:"
    )

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

