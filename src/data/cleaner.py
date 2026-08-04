import json
import logging
from pathlib import Path
from datasets import load_dataset
from configs.config import HF_DATASET_NAME, TRAIN_CLEANED_PATH, TEST_FULL_PATH, TEST_EXECUTABLE_PATH
from src.utils.logger import setup_logger

logger = setup_logger("DataCleaner")

def normalize_question(q: str) -> str:
    """Normalizes question string for exact de-leakage matching."""
    return " ".join(q.strip().lower().split())

def clean_and_prepare_dataset():
    """
    Downloads raw text2cypher dataset from HuggingFace, performs strict de-leakage
    by removing overlapping questions between train and test sets, and saves clean JSON files.
    """
    logger.info(f"Loading raw dataset '{HF_DATASET_NAME}' from HuggingFace...")
    ds = load_dataset(HF_DATASET_NAME)

    raw_train = ds["train"]
    raw_test = ds["test"]

    logger.info(f"Raw split sizes -> Train: {len(raw_train)}, Test: {len(raw_test)}")

    # Extract test question signatures
    test_questions = set(normalize_question(item["question"]) for item in raw_test if item.get("question"))

    clean_train_data = []
    leaked_count = 0

    for item in raw_train:
        q_norm = normalize_question(item["question"])
        if q_norm in test_questions:
            leaked_count += 1
        else:
            clean_train_data.append({
                "id": item.get("id"),
                "question": item["question"],
                "schema": item["schema"],
                "cypher": item["cypher"],
                "source": item.get("source", ""),
                "database_ref": item.get("database_ref", "")
            })

    logger.info(f"De-leakage Summary:")
    logger.info(f" - Raw Train Samples: {len(raw_train)}")
    logger.info(f" - Leaked Questions Removed: {leaked_count}")
    logger.info(f" - Cleaned Train Dataset: {len(clean_train_data)} samples")

    # Save cleaned train set
    with open(TRAIN_CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved clean train set to '{TRAIN_CLEANED_PATH}'")

    # Save full test set (4.8k)
    full_test_data = [
        {
            "id": item.get("id"),
            "question": item["question"],
            "schema": item["schema"],
            "cypher": item["cypher"],
            "source": item.get("source", ""),
            "database_ref": item.get("database_ref", "")
        }
        for item in raw_test
    ]
    with open(TEST_FULL_PATH, "w", encoding="utf-8") as f:
        json.dump(full_test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved full test set ({len(full_test_data)} samples) to '{TEST_FULL_PATH}'")

    # Save executable test set (samples with valid database_ref)
    exec_test_data = [item for item in full_test_data if item.get("database_ref")]
    with open(TEST_EXECUTABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(exec_test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved executable test set ({len(exec_test_data)} samples) to '{TEST_EXECUTABLE_PATH}'")

    return len(clean_train_data), len(full_test_data), len(exec_test_data)
