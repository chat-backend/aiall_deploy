# train/training_module.py
"""
AIALL – Training Module (LoRA)
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
    Tạo LoRA adapter đúng chuẩn như file chính yêu cầu.
    """
    lora_config = LoraConfig(
        r=12,
        lora_alpha=24,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
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
    Hàm train LoRA – tách riêng để file chính gọn hơn.
    Đồng bộ 100% với logic file chính.
    """

    # Đảm bảo cấu hình training giống file chính
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    # Collator cho causal LM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_steps=999999,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        warmup_steps=warmup_steps,
        dataloader_num_workers=2,
        report_to="none",
    )

    # Build LoRA adapter (đúng cấu hình)
    lora_model = build_lora_adapter(base_model)

    # Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    print("=== TRAINING START ===")
    trainer.train()
    print("=== TRAINING COMPLETE ===")

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
