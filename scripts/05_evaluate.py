import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.config import OUTPUTS_DIR, STUDENT_MODEL_ID
from src.evaluation.evaluator import evaluate_model_on_testset
from src.utils.verifier import verify_stage05
from src.utils.logger import setup_logger

logger = setup_logger("Stage05_Evaluate")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Student Model Adapter on Test Dataset")
    parser.add_argument("--adapter", type=str, default="results/qwen2.5_1.5b_kd_adapter/final_adapter", help="Adapter folder path inside outputs/")
    parser.add_argument("--model_id", type=str, default=STUDENT_MODEL_ID, help="Base Model ID")
    parser.add_argument("--is_kd", action="store_true", default=True, help="Set True if adapter outputs 4-step JSON CoT")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    args = parser.parse_args()

    logger.info("=== STAGE 5: BENCHMARK EVALUATION ===")

    adapter_path = OUTPUTS_DIR / args.adapter
    if not adapter_path.exists():
        logger.error(f"Adapter directory not found: '{adapter_path}'")
        return

    results = evaluate_model_on_testset(
        adapter_path=adapter_path,
        base_model_id=args.model_id,
        is_kd=args.is_kd,
        sample_limit=args.limit
    )

    logger.info(f"Stage 5 Evaluation Finished. Running Output Verification...")
    verify_stage05(args.adapter)

if __name__ == "__main__":
    main()
