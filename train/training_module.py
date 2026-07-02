# train/training_module.py
"""
AIALL – Training Module (LoRA, CPU-SAFE)
Tách riêng để file chính gọn, dễ quản lý và nâng cấp.
"""

import os
from datetime import datetime
from secrets import token_hex

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from config import MODEL_TOKEN_FILE


def build_lora_adapter(base_model):
    """
    Tạo LoRA adapter CPU-SAFE, giữ nguyên API cũ.
    """
    lora_config = LoraConfig(
        r=4,                     # CPU-friendly
        lora_alpha=16,
        target_modules=["c_attn"],   # Qwen attention modules tương tự GPT-2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_config)


def train_lora_model(
    base_model,
    tokenizer,
    tokenized_dataset,
    output_dir,
    log_file,
    train_batch=4,
    grad_acc=2,
    lr=1e-4,
    epochs=2,
    warmup_steps=40,
):
    """
    Hàm train LoRA – phiên bản CPU-SAFE.
    Giữ nguyên API cũ để file chính không cần sửa.
    """

    # CPU optimization
    torch.backends.mkldnn.enabled = True
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.cuda.is_available = lambda: False

    base_model.to("cpu")
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    # Collator cho causal LM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # TrainingArguments – CPU-SAFE
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_steps=999999,
        save_total_limit=1,
        fp16=False,              # CPU-safe
        bf16=False,              # CPU-safe
        optim="adamw_torch",
        warmup_steps=warmup_steps,
        dataloader_num_workers=0,   # CPU-safe
        report_to="none",
    )

    # Build LoRA adapter (CPU-safe)
    lora_model = build_lora_adapter(base_model)

    # Trainer CPU-SAFE
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    print("=== TRAINING START (CPU-SAFE) ===")
    trainer.train()
    print("=== TRAINING COMPLETE (CPU-SAFE) ===")

    # Save adapter + tokenizer
    lora_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Generate new model token
    new_token = token_hex(64)
    MODEL_TOKEN_FILE.write_text(f"AIALL_MODEL_TOKEN={new_token}\n")

    # Log
    with open(log_file, "a") as h:
        h.write(
            f"[TRAIN] {datetime.now().isoformat()} model_dir={output_dir}\n"
        )

    return True
