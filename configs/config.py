import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure required directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
HF_DATASET_NAME = "neo4j/text2cypher-2024v1"

# Data File Paths
TRAIN_CLEANED_PATH = PROCESSED_DATA_DIR / "train_cleaned.json"
TEST_FULL_PATH = PROCESSED_DATA_DIR / "test_full.json"
TEST_EXECUTABLE_PATH = PROCESSED_DATA_DIR / "test_executable.json"

# Full 37k Distillation & Training File Paths
DISTILLATION_TRAIN_37K_PATH = PROCESSED_DATA_DIR / "clean_distillation_train_37k.json"
TRAIN_BASELINE_37K_PATH = PROCESSED_DATA_DIR / "train_baseline_37k.json"
TRAIN_KD_37K_PATH = PROCESSED_DATA_DIR / "train_kd_37k.json"

# Default Model Identifiers
TEACHER_MODEL_ID = os.getenv("TEACHER_MODEL_ID", "ag/gemini-3.6-flash-high") # Options: ag/gemini-3.6-flash-high, gemini-1.5-flash, gpt-4o-mini
STUDENT_MODEL_ID = os.getenv("STUDENT_MODEL_ID", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit") # Secondary: Qwen2.5-3B-Instruct

# API Keys & Local Router Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOCAL_ROUTER_URL = os.getenv("LOCAL_ROUTER_URL", "http://localhost:20128/v1")
LOCAL_ROUTER_KEY = os.getenv("LOCAL_ROUTER_KEY", "sk-ac17cce818d45be5-f2gyor-f42f21f8")

# Training Configuration (QLoRA / Unsloth)
TRAINING_CONFIG = {
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_steps": 20,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "seed": 42,
}

# Neo4j Database Credentials (For Execution Exact Match evaluation)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
