import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.config import STUDENT_MODEL_ID, TRAIN_BASELINE_37K_PATH, TRAIN_KD_37K_PATH
from src.training.trainer import train_student_model
from src.utils.logger import setup_logger

logger = setup_logger("Stage04_TrainStudent")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Student SLM Model for Text2Cypher")
    parser.add_argument("--mode", type=str, choices=["kd", "baseline"], default="kd", help="SFT mode: 'kd' (4-step JSON) or 'baseline' (Direct Cypher)")
    parser.add_argument("--model_id", type=str, default=STUDENT_MODEL_ID, help="HuggingFace / Unsloth Base Model ID")
    parser.add_argument("--output_name", type=str, default=None, help="Custom output adapter name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    logger.info("=== STAGE 4: STUDENT MODEL SFT TRAINING ===")

    if args.mode == "kd":
        dataset_path = TRAIN_KD_37K_PATH
        output_name = args.output_name or "qwen2.5_1.5b_student_kd"
    else:
        dataset_path = TRAIN_BASELINE_37K_PATH
        output_name = args.output_name or "qwen2.5_1.5b_student_baseline"

    if not dataset_path.exists():
        logger.error(f"Training dataset file not found: '{dataset_path}'. Please run Stage 3 script first!")
        return

    train_student_model(
        model_id=args.model_id,
        dataset_path=dataset_path,
        output_model_name=output_name,
        epochs=args.epochs
    )
    logger.info(f"Stage 4 Complete: Fine-tuning finished. Adapter saved under 'outputs/{output_name}'.")

if __name__ == "__main__":
    main()
