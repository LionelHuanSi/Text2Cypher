import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.cleanup import purge_legacy_files
purge_legacy_files()

from src.data.cleaner import clean_and_prepare_dataset
from src.utils.logger import setup_logger

logger = setup_logger("Stage01_DataPrep")

def main():
    logger.info("=== STAGE 1: DATASET DOWNLOAD & DE-LEAKAGE CLEANING ===")
    train_count, test_count, exec_count = clean_and_prepare_dataset()
    logger.info(f"Stage 1 Complete: {train_count} clean train, {test_count} full test, {exec_count} executable test.")

if __name__ == "__main__":
    main()
