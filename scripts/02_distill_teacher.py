import sys
import time
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
from src.utils.verifier import verify_stage02
from src.utils.logger import setup_logger

logger = setup_logger("Stage02_DistillTeacher")

def save_atomic_checkpoint(filepath: Path, data: list):
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Robust replace for Windows file locks
    for attempt in range(5):
        try:
            if filepath.exists():
                filepath.unlink()
            tmp_path.replace(filepath)
            break
        except Exception:
            time.sleep(0.2)
            if attempt == 4:
                # Direct write fallback
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

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

    # Resume by strict sample index count (no filtering/deduplication)
    existing_distillation_set = []
    if DISTILLATION_TRAIN_37K_PATH.exists():
        try:
            with open(DISTILLATION_TRAIN_37K_PATH, "r", encoding="utf-8") as f:
                existing_distillation_set = json.load(f)
            logger.info(f"Detected existing checkpoint: {len(existing_distillation_set)} samples loaded. Resuming!")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    start_index = len(existing_distillation_set)
    unprocessed_data = train_data[start_index:]
    logger.info(f"Resuming from index {start_index}. Remaining: {len(unprocessed_data)} / {len(train_data)} total.")

    if not unprocessed_data:
        logger.info("All requested samples have already been distilled! Running verification...")
        verify_stage02()
        return

    extractor = get_teacher_extractor(TEACHER_MODEL_ID)
    clean_distillation_set = existing_distillation_set
    invalid_count = 0
    save_batch_size = 10

    pbar = tqdm(unprocessed_data, desc="Distilling 4-Step CoT Knowledge", dynamic_ncols=True)
    for i, item in enumerate(pbar):
        prompt = create_teacher_prompt(item["schema"], item["question"])
        raw_out = extractor.generate_single(prompt)

        if not raw_out:
            continue

        parsed = parse_cot_json(raw_out)
        is_valid, reason = validate_teacher_ontology_and_cypher(item["schema"], parsed)

        if parsed:
            clean_distillation_set.append({
                "schema": item["schema"],
                "question": item["question"],
                "ground_truth_cypher": item.get("cypher", ""),
                "instance_extraction": parsed.get("instance_extraction", []),
                "relation_mapping": parsed.get("relation_mapping", []),
                "validation_check": parsed.get("validation_check", {"status": "PASS" if is_valid else "FAIL"}),
                "teacher_cypher": parsed.get("cypher", ""),
                "is_valid": is_valid,
                "invalid_reason": reason if not is_valid else "PASS",
                "full_teacher_json": raw_out
            })
            if not is_valid:
                invalid_count += 1
                logger.info(f"[Captured Raw Output] Sample saved (is_valid=False: {reason})")
        else:
            invalid_count += 1

        pbar.set_postfix({"saved": len(clean_distillation_set), "flagged_invalid": invalid_count})

        if (i + 1) % 10 == 0:
            save_atomic_checkpoint(DISTILLATION_TRAIN_37K_PATH, clean_distillation_set)

    save_atomic_checkpoint(DISTILLATION_TRAIN_37K_PATH, clean_distillation_set)

    logger.info(f"Stage 2 Execution Finished. Running Output Verification...")
    verify_stage02()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()
    main(sample_limit=args.limit)
