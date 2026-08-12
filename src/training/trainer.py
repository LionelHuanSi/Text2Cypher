import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from configs.config import TRAINING_CONFIG, OUTPUTS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("StudentTrainer")


def _is_bf16_supported() -> bool:
    """Check hardware compute capability >= 8.0 (Ampere+) for native bf16."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def train_student_model(
    model_id: str,
    dataset_path: Path,
    output_model_name: str,
    epochs: int = TRAINING_CONFIG["num_epochs"],
):
    """
    Fine-tunes a student SLM (e.g. Qwen2.5-1.5B/3B) using 4-bit QLoRA via
    HuggingFace PEFT + trl SFTTrainer. Compatible across all trl versions.
    """
    output_dir = OUTPUTS_DIR / output_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Student SFT Training ===")
    logger.info(f" - Base Model ID: {model_id}")
    logger.info(f" - Dataset Path: {dataset_path}")
    logger.info(f" - Output Adapter Dir: {output_dir}")

    # ------------------------------------------------------------------ #
    # 1. Load Dataset
    # ------------------------------------------------------------------ #
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hf_dataset = Dataset.from_list(data)
    logger.info(f"Loaded {len(hf_dataset)} samples for fine-tuning.")

    # ------------------------------------------------------------------ #
    # 2. Precision: T4 uses fp16; Ampere+ (A100/L4) uses bf16
    # ------------------------------------------------------------------ #
    use_bf16 = _is_bf16_supported()
    use_fp16 = not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else "N/A"
    logger.info(f"Precision: {'bf16' if use_bf16 else 'fp16'} (GPU compute cap: {cap})")

    # ------------------------------------------------------------------ #
    # 3. 4-bit QLoRA Quantization Config (NF4 + double quant)
    # ------------------------------------------------------------------ #
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------------------------------------------ #
    # 4. Load Base Model + Tokenizer
    # ------------------------------------------------------------------ #
    logger.info("Loading base model with 4-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------ #
    # 5. LoRA Adapter Config
    # ------------------------------------------------------------------ #
    peft_config = LoraConfig(
        r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        target_modules=TRAINING_CONFIG["target_modules"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ------------------------------------------------------------------ #
    # 6. SFTTrainer via TrainingArguments (VRAM Optimized for T4 GPU)
    # ------------------------------------------------------------------ #
    # Batch size = 2 per GPU, accumulation = 16 => Total batch size = 32 (Same effective batch size, 75% less VRAM)
    per_device_bs = 2
    grad_accum_steps = 16

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_train_epochs=epochs,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=TRAINING_CONFIG["logging_steps"],
        optim="paged_adamw_8bit",
        weight_decay=TRAINING_CONFIG["weight_decay"],
        lr_scheduler_type="linear",
        seed=TRAINING_CONFIG["seed"],
        output_dir=str(output_dir),
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=min(TRAINING_CONFIG["max_seq_length"], 1280),
        peft_config=peft_config,
        args=training_args,
    )

    # ------------------------------------------------------------------ #
    # 7. Resume or Start Training
    # ------------------------------------------------------------------ #
    torch.cuda.empty_cache()
    last_checkpoint = None
    if output_dir.is_dir():
        last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        logger.info("Starting training from scratch...")
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    # ------------------------------------------------------------------ #
    # 8. Save LoRA adapter
    # ------------------------------------------------------------------ #
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Saved trained LoRA adapter to '{output_dir}'.")
    return trainer_stats

