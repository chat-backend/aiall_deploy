# train/aiall_train.py
#!/usr/bin/env python3
"""
AIALL – LoRA Training & Real-Time Inference Module
CPU-Optimized, QUALITY + VIETNAMESE + AIALL STYLE
------------------------------------------------------------------
Nâng cấp chính:
- Train nhanh hơn nhưng vẫn chất lượng:
  - MAX_SAMPLES = 300 (giảm nhẹ để tăng tốc, vẫn đủ học)
  - MAX_LEN = 256
  - batch size = 4, grad_acc = 2
  - LoRA r = 12 (vừa phải, an toàn cho 1.5B trên CPU)
  - warmup_steps = 40, epochs = 2, early-stop mềm

- Trả lời dài hơn, tự nhiên hơn:
  - max_new_tokens = 320
  - temperature = 0.8, top_p = 0.92

- Ưu tiên tiếng Việt:
  - context layer nhận diện tiếng Việt
  - prompt định hình phong cách AIALL tiếng Việt

- Phong cách AIALL:
  - luôn giới thiệu là AI trợ lý AIALL
  - ưu tiên giải thích rõ ràng, thân thiện, có cấu trúc
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

from train.aiall_style import aiall_chat

# ============================================================
#  CPU OPTIMIZATION – ưu tiên chất lượng nhưng vẫn nhanh
# ============================================================

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

# Training config – QUALITY + SPEED trên CPU
OUTPUT_DIR = LORA_OUTPUT_DIR
TRAIN_BATCH = 4
GRAD_ACC = 2
LR = 1e-4
EPOCHS = 2
MAX_LEN = 256
MAX_SAMPLES = 300  # giảm nhẹ để train nhanh hơn

EARLY_STOP_MIN_LOSS_IMPROVEMENT = 0.0007
EARLY_STOP_PATIENCE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOG FILE (HOME, không dùng /root) ---
LOG_FILE = os.path.expanduser("~/aiall_logs/model_history.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


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
        # Giữ cấu trúc Instruction/Answer để mô hình quen phong cách AIALL
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
#  BUILD LoRA MODEL (QUALITY + SPEED)
# ============================================================

def build_lora_model(base_model):
    lora_config = LoraConfig(
        r=12,  # giảm nhẹ từ 16 xuống 12 để train nhanh hơn nhưng vẫn đủ học
        lora_alpha=24,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_config)


# ============================================================
#  TRAIN AIALL (QUALITY + SPEED, EARLY-STOP BY LOSS)
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
        logging_steps=40,
        save_steps=400,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        use_cpu=True,
        optim="adamw_torch",
        max_grad_norm=0.6,
        warmup_steps=40,
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
    print("=== TRAINING DONE. ADAPTER SAVED TO aiall-lora (QUALITY+SPEED MODE) ===")

    with open(LOG_FILE, "a") as h:
        h.write(
            f"[TRAIN_QUALITY_SPEED] {datetime.now().isoformat()} "
            f"model_dir={LORA_OUTPUT_DIR} version=quality_speed checksum=none\n"
        )


# ============================================================
#  MERGE LoRA → FULL MODEL
# ============================================================

def merge_lora():
    print("=== MERGING LoRA INTO FULL MODEL (CPU MODE, QUALITY+SPEED) ===")

    if not os.path.exists(LORA_OUTPUT_DIR):
        raise SystemExit(f"[ERROR] LoRA adapter not found: {LORA_OUTPUT_DIR}")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
    base_model = base_model.to(torch.float32)

    merged = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    merged = merged.merge_and_unload()

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_OUTPUT_DIR)
    print("=== MERGED MODEL SAVED TO aiall-merged ===")

    with open(LOG_FILE, "a") as h:
        h.write(
            f"[MERGE_QUALITY_SPEED] {datetime.now().isoformat()} "
            f"model_dir={MERGED_OUTPUT_DIR} version=quality_speed checksum=none\n"
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
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForCausalLM.from_pretrained(MERGED_OUTPUT_DIR, device_map="cpu")
    model = model.to(torch.float32)
    model.eval()
    return model, tokenizer


# ============================================================
#  CHAT FUNCTION – DÀI HƠN, TỰ NHIÊN HƠN, ƯU TIÊN TIẾNG VIỆT
# ============================================================

def chat(model, tokenizer, prompt: str):
    return aiall_chat(model, tokenizer, prompt)













