import os
import sys
import torch
import logging
from pathlib import Path
from datasets import Dataset

from configs.config import TRAINING_CONFIG, OUTPUTS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("StudentTrainer")

def train_student_model(
    model_id: str,
    dataset_path: Path,
    output_model_name: str,
    epochs: int = TRAINING_CONFIG["num_epochs"]
):
    """
    Fine-tunes student SLM model (e.g. Qwen2.5-1.5B/3B) using Unsloth / QLoRA 4-bit SFT.
    """
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    output_dir = OUTPUTS_DIR / output_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Student SFT Training ===")
    logger.info(f" - Base Model ID: {model_id}")
    logger.info(f" - Dataset Path: {dataset_path}")
    logger.info(f" - Output Adapter Dir: {output_dir}")

    # Load dataset
    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hf_dataset = Dataset.from_list(data)
    logger.info(f"Loaded {len(hf_dataset)} samples for fine-tuning.")

    # Attempt Unsloth FastLanguageModel loading
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=TRAINING_CONFIG["max_seq_length"],
            load_in_4bit=TRAINING_CONFIG["load_in_4bit"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=TRAINING_CONFIG["lora_r"],
            target_modules=TRAINING_CONFIG["target_modules"],
            lora_alpha=TRAINING_CONFIG["lora_alpha"],
            lora_dropout=TRAINING_CONFIG["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=TRAINING_CONFIG["seed"],
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=TRAINING_CONFIG["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
                gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
                warmup_steps=TRAINING_CONFIG["warmup_steps"],
                num_train_epochs=epochs,
                learning_rate=TRAINING_CONFIG["learning_rate"],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=TRAINING_CONFIG["logging_steps"],
                optim="adamw_8bit",
                weight_decay=TRAINING_CONFIG["weight_decay"],
                lr_scheduler_type="linear",
                seed=TRAINING_CONFIG["seed"],
                output_dir=str(output_dir),
                save_strategy="epoch",
            ),
        )

        logger.info("Executing training loop...")
        trainer_stats = trainer.train()
        
        # Save LoRA adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Saved trained adapter successfully to '{output_dir}'.")
        return trainer_stats

    except Exception as e:
        logger.warning(f"Unsloth initialization failed ({e}). Falling back to standard HuggingFace PEFT Trainer...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=TRAINING_CONFIG["lora_r"],
            lora_alpha=TRAINING_CONFIG["lora_alpha"],
            target_modules=TRAINING_CONFIG["target_modules"],
            lora_dropout=TRAINING_CONFIG["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=TRAINING_CONFIG["max_seq_length"],
            args=TrainingArguments(
                per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
                gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
                warmup_steps=TRAINING_CONFIG["warmup_steps"],
                num_train_epochs=epochs,
                learning_rate=TRAINING_CONFIG["learning_rate"],
                fp16=True,
                logging_steps=TRAINING_CONFIG["logging_steps"],
                optim="paged_adamw_8bit",
                output_dir=str(output_dir),
                save_strategy="epoch",
            ),
        )

        logger.info("Executing standard HuggingFace PEFT training loop...")
        trainer_stats = trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Saved trained HF adapter successfully to '{output_dir}'.")
        return trainer_stats
