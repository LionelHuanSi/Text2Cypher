import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.cleanup import purge_legacy_files
purge_legacy_files()

from src.data.cleaner import clean_and_prepare_dataset
from src.utils.verifier import verify_stage01
from src.utils.logger import setup_logger

logger = setup_logger("Stage01_DataPrep")

def main():
    logger.info("=== STAGE 1: DATASET DOWNLOAD & DE-LEAKAGE CLEANING ===")
    train_count, test_count, exec_count = clean_and_prepare_dataset()
    logger.info(f"Stage 1 Execution Finished. Running Output Verification...")
    
    # Automated Output Verification Check
    is_ok = verify_stage01()
    if is_ok:
        logger.info("Stage 1 Output Verification PASSED 100%! Ready for Stage 2.")
    else:
        logger.error("Stage 1 Output Verification FAILED! Please inspect errors above.")

if __name__ == "__main__":
    main()
