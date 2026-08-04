import os
import inspect
import logging
from typing import List, Dict
from configs.config import STUDENT_MODEL_ID, TRAINING_CONFIG, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_student_model(dataset_samples: List[Dict], output_model_dir: str):
    """
    Fine-tunes the Student model using Unsloth (or Hugging Face PEFT fallback for local execution).
    """
    logger.info(f"Starting Student Fine-Tuning for target model: '{STUDENT_MODEL_ID}'...")
    logger.info(f"Training dataset size: {len(dataset_samples)} samples.")
    
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        logger.info("Using Unsloth FastLanguageModel Engine...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=STUDENT_MODEL_ID,
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

        hf_dataset = Dataset.from_list(dataset_samples)

        sft_kwargs = {
            "dataset_text_field": "text",
            "per_device_train_batch_size": TRAINING_CONFIG["batch_size"],
            "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
            "warmup_steps": TRAINING_CONFIG["warmup_steps"],
            "num_train_epochs": TRAINING_CONFIG["num_epochs"],
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "fp16": False,
            "bf16": False,
            "logging_steps": TRAINING_CONFIG["logging_steps"],
            "optim": "adamw_8bit",
            "weight_decay": TRAINING_CONFIG["weight_decay"],
            "lr_scheduler_type": "linear",
            "seed": TRAINING_CONFIG["seed"],
            "output_dir": output_model_dir,
        }
        sig = inspect.signature(SFTConfig.__init__)
        if "max_seq_length" in sig.parameters:
            sft_kwargs["max_seq_length"] = TRAINING_CONFIG["max_seq_length"]
        elif "max_length" in sig.parameters:
            sft_kwargs["max_length"] = TRAINING_CONFIG["max_seq_length"]

        sft_config = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            train_dataset=hf_dataset,
            args=sft_config,
            peft_config=model.peft_config,
            processing_class=tokenizer,
        )

        logger.info("Executing SFT Training loop...")
        trainer.train()

        os.makedirs(output_model_dir, exist_ok=True)
        model.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        logger.info(f"Student model training complete! Artifacts saved to '{output_model_dir}'.")

    except Exception as e:
        logger.warning(f"Unsloth execution failed or not detected: {e}. Falling back to standard HuggingFace PEFT Engine...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID,
            low_cpu_mem_usage=True,
        )

        peft_config = LoraConfig(
            r=TRAINING_CONFIG["lora_r"],
            lora_alpha=TRAINING_CONFIG["lora_alpha"],
            target_modules=TRAINING_CONFIG["target_modules"],
            lora_dropout=TRAINING_CONFIG["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Không gọi get_peft_model() ở đây - để SFTTrainer tự xử lý via peft_config
        hf_dataset = Dataset.from_list(dataset_samples)

        import torch
        use_gpu = torch.cuda.is_available()
        sft_kwargs = {
            "dataset_text_field": "text",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": TRAINING_CONFIG["num_epochs"],
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "logging_steps": 5,
            "seed": TRAINING_CONFIG["seed"],
            "output_dir": output_model_dir,
            "fp16": False,   # Tắt FP16 - không dùng AMP để tránh lỗi CPU/GPU
            "bf16": False,   # Tắt BF16 - T4 không hỗ trợ, CPU cũng không
            "use_cpu": not use_gpu,  # Tự động phát hiện CPU/GPU (thay thế no_cuda deprecated)
        }
        sig = inspect.signature(SFTConfig.__init__)
        if "max_seq_length" in sig.parameters:
            sft_kwargs["max_seq_length"] = TRAINING_CONFIG["max_seq_length"]
        elif "max_length" in sig.parameters:
            sft_kwargs["max_length"] = TRAINING_CONFIG["max_seq_length"]

        sft_config = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            train_dataset=hf_dataset,
            args=sft_config,
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        logger.info("Executing Fallback SFT Training loop...")
        trainer.train()

        os.makedirs(output_model_dir, exist_ok=True)
        model.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        logger.info(f"Student model fallback training complete! Artifacts saved to '{output_model_dir}'.")
