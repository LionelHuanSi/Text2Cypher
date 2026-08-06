import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.formatter import export_sft_datasets
from src.utils.verifier import verify_stage03
from src.utils.logger import setup_logger

logger = setup_logger("Stage03_ExportSFT")

def main():
    logger.info("=== STAGE 3: EXPORT SFT DATASETS (BASELINE VS PROPOSED KD) ===")
    b_count, kd_count = export_sft_datasets()
    logger.info(f"Stage 3 Execution Finished. Running Output Verification...")
    verify_stage03()

if __name__ == "__main__":
    main()
