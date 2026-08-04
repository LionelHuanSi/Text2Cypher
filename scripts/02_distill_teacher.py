import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.config import TRAIN_CLEANED_PATH, DISTILLATION_TRAIN_37K_PATH, TEACHER_MODEL_ID
from src.prompts.teacher_prompts import create_teacher_prompt
from src.utils.json_parser import parse_cot_json
from src.extraction.validator import validate_teacher_ontology_and_cypher
from src.extraction.teacher import get_teacher_extractor
from src.utils.logger import setup_logger

logger = setup_logger("Stage02_DistillTeacher")

def save_atomic_checkpoint(filepath: Path, data: list):
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(filepath)

def main(sample_limit: int = None):
    logger.info("=== STAGE 2: FULL-SCALE TEACHER DISTILLATION & 3-TIER VALIDATION ===")

    if not TRAIN_CLEANED_PATH.exists():
        logger.error(f"File not found: '{TRAIN_CLEANED_PATH}'. Please run Stage 1 script first!")
        return

    with open(TRAIN_CLEANED_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    if sample_limit:
        logger.info(f"Limiting distillation scope to first {sample_limit} samples...")
        train_data = train_data[:sample_limit]

    # Resume from existing checkpoint
    existing_distillation_set = []
    processed_questions = set()

    if DISTILLATION_TRAIN_37K_PATH.exists():
        try:
            with open(DISTILLATION_TRAIN_37K_PATH, "r", encoding="utf-8") as f:
                existing_distillation_set = json.load(f)
            processed_questions = set(item["question"].strip().lower() for item in existing_distillation_set if isinstance(item, dict) and "question" in item)
            logger.info(f"Detected existing checkpoint: {len(existing_distillation_set)} valid samples saved. Resuming!")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    unprocessed_data = [item for item in train_data if item["question"].strip().lower() not in processed_questions]
    logger.info(f"Unprocessed samples remaining: {len(unprocessed_data)} / {len(train_data)} total.")

    if not unprocessed_data:
        logger.info("All requested samples have already been distilled! Nothing to do.")
        return

    extractor = get_teacher_extractor(TEACHER_MODEL_ID)
    clean_distillation_set = existing_distillation_set
    invalid_count = 0
    save_batch_size = 10

    for i, item in enumerate(tqdm(unprocessed_data, desc="Distilling 4-Step CoT Knowledge")):
        prompt = create_teacher_prompt(item["schema"], item["question"])
        raw_out = extractor.generate_single(prompt)

        if not raw_out:
            continue

        parsed = parse_cot_json(raw_out)
        is_valid, reason = validate_teacher_ontology_and_cypher(item["schema"], parsed)

        if is_valid:
            clean_distillation_set.append({
                "schema": item["schema"],
                "question": item["question"],
                "ground_truth_cypher": item.get("cypher", ""),
                "instance_extraction": parsed.get("instance_extraction", []),
                "relation_mapping": parsed.get("relation_mapping", []),
                "validation_check": parsed.get("validation_check", {"status": "PASS"}),
                "teacher_cypher": parsed.get("cypher", ""),
                "full_teacher_json": raw_out
            })
        else:
            invalid_count += 1

        if (i + 1) % save_batch_size == 0:
            save_atomic_checkpoint(DISTILLATION_TRAIN_37K_PATH, clean_distillation_set)
            logger.info(f"Checkpoint saved: Total {len(clean_distillation_set)} valid distillation samples.")

    save_atomic_checkpoint(DISTILLATION_TRAIN_37K_PATH, clean_distillation_set)

    logger.info(f"Stage 2 Execution Summary:")
    logger.info(f" - Total Valid Distillation Samples Saved: {len(clean_distillation_set)}")
    logger.info(f" - Total Invalid/Hallucinated Samples Rejected: {invalid_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()
    main(sample_limit=args.limit)
