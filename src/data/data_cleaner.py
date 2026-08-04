import json
import logging
from typing import Dict, List, Tuple
from configs.config import TRAIN_CLEANED_PATH, TEST_FULL_PATH, TEST_EXECUTABLE_PATH, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_and_split_dataset(train_data: List[Dict], test_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    1. Removes data leakage by discarding any Train question that exact-matches a Test question.
    2. Filters Test split into test_full (unseen clean test set) and test_executable (only samples with database_reference).
    
    Returns: (train_cleaned, test_full_cleaned, test_executable_cleaned)
    """
    logger.info("Starting Data Leakage Audit and Executable Subset Filtering...")

    # Set of test questions for overlap checking
    test_questions = set(item["question"].strip().lower() for item in test_data)
    
    # 1. De-duplicate Train against Test questions
    train_cleaned = []
    leak_count = 0
    for item in train_data:
        q_norm = item["question"].strip().lower()
        if q_norm in test_questions:
            leak_count += 1
        else:
            train_cleaned.append(item)
            
    logger.info(f"Data Leakage Check Complete: Discarded {leak_count} training samples overlapping with Test questions.")
    logger.info(f"Clean Train Set Size: {len(train_cleaned)} samples (Original: {len(train_data)}).")

    # 2. Filter Executable Test Subset (Must have valid database_reference_alias / database_reference)
    test_executable = []
    for item in test_data:
        db_ref = item.get("database_reference_alias") or item.get("database_reference")
        if db_ref is not None:
            test_executable.append(item)
            
    logger.info(f"Test Split Statistics:")
    logger.info(f" - Full Test Set: {len(test_data)} samples.")
    logger.info(f" - Executable Subset: {len(test_executable)} samples ({len(test_executable)/len(test_data)*100:.2f}% of total test set).")

    # Save processed files to disk
    with open(TRAIN_CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(train_cleaned, f, ensure_ascii=False, indent=2)
        
    with open(TEST_FULL_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    with open(TEST_EXECUTABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_executable, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved processed dataset files to '{PROCESSED_DATA_DIR}'")
    return train_cleaned, test_data, test_executable
