# train/aiall_train.py
#!/usr/bin/env python3
"""
AIALL – LoRA Training & Real-Time Inference Module (CPU-Optimized, Balanced Speed & Quality)
--------------------------------------------------
- Train LoRA trên base model Qwen2.5-1.5B (tối ưu cho CPU, ~40–60 phút)
- Lưu adapter vào aiall-lora/
- Merge LoRA vào full model aiall-merged/
- Cập nhật MODEL_TOKEN
- Đăng ký backend vào vLLM cluster (Linux-only)
- Load merged model để inference
- Tích hợp lớp Real-Time Context (thời gian, web/db/events/finance/weather/news/calendar)
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

# Bật MKL-DNN để tối ưu CPU
torch.backends.mkldnn.enabled = True

IS_LINUX = platform.system().lower().startswith("linux")

# ===== Backend + Nginx integration =====
import core.backends as be
if IS_LINUX:
    import core.nginx as ngx
else:
    ngx = None  # Nginx không dùng trên Windows / non-Linux

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DATA_FILE = os.path.join(os.path.dirname(__file__), "aiall_data.jsonl")
LORA_OUTPUT_DIR = "aiall-lora"
MERGED_OUTPUT_DIR = "aiall-merged"

# ===== Training config (CPU-optimized, balanced) =====
OUTPUT_DIR = LORA_OUTPUT_DIR
TRAIN_BATCH = 1
GRAD_ACC = 1
LR = 2e-4
EPOCHS = 1
MAX_LEN = 256
MAX_SAMPLES = 500          # giới hạn nhẹ để giữ chất lượng nhưng giảm thời gian

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Giới hạn số thread để tránh oversubscription
torch.set_num_threads(max(1, os.cpu_count() // 2))


# ============================================================
#  REAL-TIME CONTEXT LAYER (STUBS)
# ============================================================

def build_realtime_context(prompt: str) -> str:
    parts = []

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"[TIME] Current datetime: {current_time}")

    pl = prompt.lower()

    if any(w in pl for w in ["tìm", "search", "google", "tra cứu"]):
        parts.append("[WEB] Web search: (stub) dữ liệu thời gian thực sẽ được tích hợp tại đây.")

    if "cơ sở dữ liệu" in pl or "database" in pl or "db" in pl:
        parts.append("[DB] Database: (stub) kết quả truy vấn DB sẽ được tích hợp tại đây.")

    if "sự kiện" in pl or "timeline" in pl:
        parts.append("[EVENTS] Timeline: (stub) danh sách sự kiện gần đây sẽ được tích hợp tại đây.")

    if any(w in pl for w in ["giá", "bitcoin", "btc", "chứng khoán", "stock"]):
        parts.append("[FINANCE] Financial: (stub) giá tài chính thời gian thực sẽ được tích hợp tại đây.")

    if "thời tiết" in pl or "weather" in pl:
        parts.append("[WEATHER] Weather: (stub) dữ liệu thời tiết thời gian thực sẽ được tích hợp tại đây.")

    if "tin tức" in pl or "news" in pl or "báo" in pl:
        parts.append("[NEWS] News: (stub) tin tức mới nhất sẽ được tích hợp tại đây.")

    if any(w in pl for w in ["lịch", "ngày", "tháng", "năm", "calendar"]):
        parts.append("[CALENDAR] Calendar: hệ thống nhận biết ngày/tháng/năm hiện tại và bối cảnh thời gian.")

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
#  LOAD BASE MODEL (CPU-optimized)
# ============================================================

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
    )
    model = model.to(torch.float32)          # CPU: float32 ổn định và nhanh hơn bf16
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer


# ============================================================
#  LOAD + TOKENIZE DATASET (CPU-optimized, balanced)
# ============================================================

def load_dataset_tokenized(tokenizer):
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"[ERROR] Dataset file not found: {DATA_FILE}")

    dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]

    # Giới hạn số mẫu để thời gian train < 1 tiếng nhưng vẫn đủ đa dạng
    if len(dataset) > MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    def preprocess(example):
        instruction = example["instruction"]
        output = example["output"]
        text = f"Instruction: {instruction}\nAnswer: {output}"

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
#  BUILD LoRA MODEL (CPU-optimized, balanced)
# ============================================================

def build_lora_model(base_model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj",   # bỏ gate_proj để giảm tải CPU
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_config)


# ============================================================
#  TRAIN AIALL (LoRA, CPU-optimized, balanced)
# ============================================================

def train_aiall():
    if not IS_LINUX:
        raise SystemExit("[ERROR] Training is only supported on Linux."]

    base_model, tokenizer = load_base_model()
    tokenized_dataset = load_dataset_tokenized(tokenizer)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,

        fp16=False,
        bf16=False,
        use_cpu=True,

        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,

        dataloader_num_workers=2,   # tăng tốc dataloader
        report_to="none",
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

    with open("/root/aiall_deploy/model_history.log", "a") as h:
        h.write(
            f"[TRAIN_BALANCED] {datetime.now().isoformat()} "
            f"model_dir={LORA_OUTPUT_DIR} "
            f"version=balanced "
            f"checksum=none\n"
        )


# ============================================================
#  MERGE LoRA → FULL MODEL (CPU-optimized)
# ============================================================

def merge_lora():
    print("=== MERGING LoRA INTO FULL MODEL (CPU MODE) ===")

    if not os.path.exists(LORA_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] LoRA adapter not found: {LORA_OUTPUT_DIR}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
    )
    base_model = base_model.to(torch.float32)

    merged = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    merged = merged.merge_and_unload()

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_OUTPUT_DIR)
    print("=== MERGED MODEL SAVED TO aiall-merged ===")

    with open("/root/aiall_deploy/model_history.log", "a") as h:
        h.write(
            f"[MERGE_BALANCED] {datetime.now().isoformat()} "
            f"model_dir={MERGED_OUTPUT_DIR} "
            f"version=balanced "
            f"checksum=none\n"
        )


# ============================================================
#  REGISTER AIALL BACKEND (URL + MODEL)
# ============================================================

def register_aiall_backend(host="127.0.0.1", port=8001):
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
#  LOAD AIALL FOR INFERENCE (CPU-optimized)
# ============================================================

def load_aiall_for_inference():
    if not os.path.exists(MERGED_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] Merged model not found: {MERGED_OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_OUTPUT_DIR,
        device_map="cpu",
    )
    model = model.to(torch.float32)
    model.eval()
    return model, tokenizer


# ============================================================
#  CHAT FUNCTION (WITH REAL-TIME CONTEXT, CPU-optimized)
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
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

