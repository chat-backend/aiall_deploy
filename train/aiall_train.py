# train/aiall_train.py
#!/usr/bin/env python3
"""
AIALL – LoRA Training & Real-Time Inference Module (CPU-Optimized, Extreme 4.0 ~10 minutes)
------------------------------------------------------------------
- Cực hạn 4.0:
  - MAX_SAMPLES = 60
  - MAX_LEN = 96
  - batch size = 5
  - LoRA r = 2 (siêu nhẹ)
  - warmup_steps rất thấp
  - logging/save rất thưa
  - early-stop theo loss (đơn giản)
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

# CPU EXTREME OPTIMIZATION 4.0
torch.backends.mkldnn.enabled = True
torch.set_num_threads(max(1, os.cpu_count() // 2))

IS_LINUX = platform.system().lower().startswith("linux")

# Backend + Nginx integration
import core.backends as be
if IS_LINUX:
    import core.nginx as ngx
else:
    ngx = None

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DATA_FILE = os.path.join(os.path.dirname(__file__), "aiall_data.jsonl")
LORA_OUTPUT_DIR = "aiall-lora"
MERGED_OUTPUT_DIR = "aiall-merged"

# Training config – tuned for ~10 minutes on CPU
OUTPUT_DIR = LORA_OUTPUT_DIR
TRAIN_BATCH = 5
GRAD_ACC = 1
LR = 2e-4
EPOCHS = 1
MAX_LEN = 96
MAX_SAMPLES = 60

EARLY_STOP_MIN_LOSS_IMPROVEMENT = 0.001  # nếu loss không cải thiện hơn ngưỡng này → dừng sớm
EARLY_STOP_PATIENCE = 3                  # số lần liên tiếp không cải thiện

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOG FILE (HOME, không dùng /root) ---
LOG_FILE = os.path.expanduser("~/aiall_logs/model_history.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# ============================================================
#  REAL-TIME CONTEXT LAYER
# ============================================================

def build_realtime_context(prompt: str) -> str:
    from datetime import datetime

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
        "hãy luôn cân nhắc rằng thông tin trong mô hình có thể đã lỗi thời."
    )

    return "\n".join(parts) + "\n\n"


# ============================================================
#  LOAD BASE MODEL
# ============================================================

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
    model = model.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


# ============================================================
#  LOAD + TOKENIZE DATASET
# ============================================================

def load_dataset_tokenized(tokenizer):
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
#  BUILD LoRA MODEL (super light)
# ============================================================

def build_lora_model(base_model):
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_config)


# ============================================================
#  TRAIN AIALL (EXTREME 4.0, EARLY-STOP BY LOSS)
# ============================================================

class LossEarlyStopTrainer(Trainer):
    def __init__(self, min_improvement: float, patience: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_improvement = min_improvement
        self._patience = patience
        self._best_loss = None
        self._no_improve_count = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        loss_value = loss.detach().cpu().item() if hasattr(loss, "detach") else float(loss)

        if self._best_loss is None or loss_value < self._best_loss - self._min_improvement:
            self._best_loss = loss_value
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self._patience:
            self.control.should_training_stop = True

        return loss


def train_aiall():
    if not IS_LINUX:
        raise SystemExit("[ERROR] Training is only supported on Linux.")

    base_model, tokenizer = load_base_model()
    tokenized_dataset = load_dataset_tokenized(tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=60,      # rất thưa
        save_steps=1500,       # rất thưa
        save_total_limit=1,
        fp16=False,
        bf16=False,
        use_cpu=True,
        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_steps=10,       # rất thấp
        dataloader_num_workers=2,
        report_to="none",
    )

    lora_model = build_lora_model(base_model)

    trainer = LossEarlyStopTrainer(
        min_improvement=EARLY_STOP_MIN_LOSS_IMPROVEMENT,
        patience=EARLY_STOP_PATIENCE,
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
    print("=== TRAINING DONE. ADAPTER SAVED TO aiall-lora (EXTREME 4.0) ===")

    # Ghi log vào HOME
    with open(LOG_FILE, "a") as h:
        h.write(
            f"[TRAIN_EXTREME_4] {datetime.now().isoformat()} "
            f"model_dir={LORA_OUTPUT_DIR} version=extreme_4 checksum=none\n"
        )


# ============================================================
#  MERGE LoRA → FULL MODEL
# ============================================================

def merge_lora():
    print("=== MERGING LoRA INTO FULL MODEL (CPU MODE, EXTREME 4.0) ===")

    if not os.path.exists(LORA_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] LoRA adapter not found: {LORA_OUTPUT_DIR}")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
    base_model = base_model.to(torch.float32)

    merged = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    merged = merged.merge_and_unload()

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_OUTPUT_DIR)
    print("=== MERGED MODEL SAVED TO aiall-merged ===")

    # Ghi log vào HOME
    with open(LOG_FILE, "a") as h:
        h.write(
            f"[MERGE_EXTREME_4] {datetime.now().isoformat()} "
            f"model_dir={MERGED_OUTPUT_DIR} version=extreme_4 checksum=none\n"
        )


# ============================================================
#  REGISTER BACKEND
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
#  LOAD AIALL FOR INFERENCE
# ============================================================

def load_aiall_for_inference():
    if not os.path.exists(MERGED_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] Merged model not found: {MERGED_OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = MAX_LEN   # ⭐ FIX QUAN TRỌNG

    model = AutoModelForCausalLM.from_pretrained(MERGED_OUTPUT_DIR, device_map="cpu")
    model = model.to(torch.float32)
    model.eval()
    return model, tokenizer

# ============================================================
#  CHAT FUNCTION
# ============================================================

def chat(model, tokenizer, prompt: str):
    realtime_context = build_realtime_context(prompt)
    text = realtime_context + f"Instruction: {prompt}\nAnswer:"

    # Lần 1: tokenize với full context
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )

    # Nếu rỗng → thử lại chỉ với prompt
    if inputs["input_ids"].numel() == 0:
        fallback_text = f"Instruction: {prompt}\nAnswer:"
        inputs = tokenizer(
            fallback_text,
            return_tensors="pt",
            truncation=True,
            padding=False,
        )

    # Nếu vẫn rỗng nữa → ép một token tối thiểu
    if inputs["input_ids"].numel() == 0:
        inputs = {
            "input_ids": torch.tensor([[tokenizer.eos_token_id]]),
            "attention_mask": torch.tensor([[1]]),
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)










