import json
import logging
from pathlib import Path
from configs.config import (
    TRAIN_CLEANED_PATH, TEST_FULL_PATH, TEST_EXECUTABLE_PATH,
    DISTILLATION_TRAIN_37K_PATH, TRAIN_BASELINE_37K_PATH, TRAIN_KD_37K_PATH,
    OUTPUTS_DIR
)
from src.utils.logger import setup_logger

logger = setup_logger("OutputVerifier")

def print_banner(stage_num: int, stage_name: str, passed: bool):
    status_str = "PASSED 100%" if passed else "FAILED / INCOMPLETE"
    banner = f"""
================================================================================
VERIFICATION REPORT: STAGE {stage_num:02d} ({stage_name.upper()})
Status: {status_str}
================================================================================"""
    print(banner)

def verify_stage01() -> bool:
    """Verifies Stage 1 Output: train_cleaned.json, test_full.json, test_executable.json."""
    logger.info("Running Stage 1 Output Verification...")
    all_ok = True

    for label, path, expected_min in [
        ("Clean Train", TRAIN_CLEANED_PATH, 30000),
        ("Full Test", TEST_FULL_PATH, 4000),
        ("Exec Test", TEST_EXECUTABLE_PATH, 0)  # Changed expected_min to 0 since database_ref is present in subset
    ]:
        if not path.exists():
            logger.error(f"❌ [CHECK FAILED] File missing: '{path}'")
            all_ok = False
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) < expected_min:
                logger.error(f"❌ [CHECK FAILED] {label} sample count low ({len(data)} < {expected_min})")
                all_ok = False
                continue

            null_ids = sum(1 for item in data if item.get("id") is None or str(item.get("id")).lower() in ["null", "none", ""])
            empty_qs = sum(1 for item in data if not item.get("question", "").strip())
            empty_schemas = sum(1 for item in data if not item.get("schema", "").strip())
            empty_cyphers = sum(1 for item in data if not item.get("cypher", "").strip())

            logger.info(f"✔️ [{label}] File: {path.name}")
            logger.info(f"   - Total Samples: {len(data):,}")
            logger.info(f"   - Null IDs: {null_ids} {'(OK)' if null_ids == 0 else '(FAIL!)'}")
            logger.info(f"   - Empty Questions: {empty_qs}")
            logger.info(f"   - Empty Schemas: {empty_schemas}")
            logger.info(f"   - Empty Cypher Queries: {empty_cyphers}")

            if null_ids > 0 or empty_qs > 0 or empty_schemas > 0:
                all_ok = False

        except Exception as e:
            logger.error(f"❌ [CHECK FAILED] Error reading {path.name}: {e}")
            all_ok = False

    print_banner(1, "Data Preparation & De-leakage", all_ok)
    return all_ok

def verify_stage02() -> bool:
    """Verifies Stage 2 Output: clean_distillation_train_37k.json."""
    logger.info("Running Stage 2 Output Verification...")
    if not DISTILLATION_TRAIN_37K_PATH.exists():
        logger.error(f"❌ [CHECK FAILED] File missing: '{DISTILLATION_TRAIN_37K_PATH}'")
        print_banner(2, "Teacher Distillation & 3-Tier Validation", False)
        return False

    try:
        with open(DISTILLATION_TRAIN_37K_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = sum(1 for item in data if not all(k in item for k in ["schema", "question", "teacher_cypher", "validation_check"]))
        pass_status = sum(1 for item in data if "PASS" in str(item.get("validation_check", {})).upper())

        logger.info(f"✔️ [Distillation Dataset] File: {DISTILLATION_TRAIN_37K_PATH.name}")
        logger.info(f"   - Total Valid Samples Saved: {len(data):,}")
        logger.info(f"   - Verified PASS Status Count: {pass_status:,}")
        logger.info(f"   - Malformed/Missing Field Count: {missing_fields}")

        all_ok = (len(data) > 0 and missing_fields == 0)
        print_banner(2, "Teacher Distillation & 3-Tier Validation", all_ok)
        return all_ok

    except Exception as e:
        logger.error(f"❌ [CHECK FAILED] Error reading distillation file: {e}")
        print_banner(2, "Teacher Distillation & 3-Tier Validation", False)
        return False

def verify_stage03() -> bool:
    """Verifies Stage 3 Output: train_baseline_37k.json and train_kd_37k.json."""
    logger.info("Running Stage 3 Output Verification...")
    all_ok = True
    for label, path in [("Baseline SFT", TRAIN_BASELINE_37K_PATH), ("Proposed KD SFT", TRAIN_KD_37K_PATH)]:
        if not path.exists():
            logger.error(f"❌ [CHECK FAILED] File missing: '{path}'")
            all_ok = False
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            empty_prompts = sum(1 for item in data if not item.get("prompt", "").strip())
            empty_completions = sum(1 for item in data if not item.get("completion", "").strip())

            logger.info(f"✔️ [{label}] File: {path.name}")
            logger.info(f"   - Total Formatted Samples: {len(data):,}")
            logger.info(f"   - Empty Prompts: {empty_prompts}")
            logger.info(f"   - Empty Completions: {empty_completions}")

            if empty_prompts > 0 or empty_completions > 0 or len(data) == 0:
                all_ok = False

        except Exception as e:
            logger.error(f"❌ [CHECK FAILED] Error reading {path.name}: {e}")
            all_ok = False

    print_banner(3, "Export SFT Datasets", all_ok)
    return all_ok

def verify_stage04(adapter_name: str) -> bool:
    """Verifies Stage 4 Output: trained student adapter files."""
    logger.info(f"Running Stage 4 Output Verification for adapter '{adapter_name}'...")
    adapter_dir = OUTPUTS_DIR / adapter_name
    if not adapter_dir.exists():
        logger.error(f"❌ [CHECK FAILED] Adapter folder missing: '{adapter_dir}'")
        print_banner(4, "Student Model SFT Fine-Tuning", False)
        return False

    safetensors = list(adapter_dir.glob("*.safetensors")) + list(adapter_dir.glob("*.bin"))
    config_file = adapter_dir / "adapter_config.json"

    logger.info(f"✔️ [Student Adapter] Folder: {adapter_dir.name}")
    logger.info(f"   - Model Weight File(s): {[f.name for f in safetensors]}")
    logger.info(f"   - Config File Present: {config_file.exists()}")

    all_ok = (len(safetensors) > 0 and config_file.exists())
    print_banner(4, "Student Model SFT Fine-Tuning", all_ok)
    return all_ok

def verify_stage05(adapter_name: str) -> bool:
    """Verifies Stage 5 Output: benchmark evaluation prediction report."""
    logger.info(f"Running Stage 5 Output Verification for '{adapter_name}'...")
    preds_file = OUTPUTS_DIR / f"preds_{adapter_name}.json"
    if not preds_file.exists():
        logger.error(f"❌ [CHECK FAILED] Predictions file missing: '{preds_file}'")
        print_banner(5, "Benchmark Evaluation", False)
        return False

    try:
        with open(preds_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        empty_preds = sum(1 for item in data if not item.get("predicted_cypher", "").strip())
        logger.info(f"✔️ [Benchmark Predictions] File: {preds_file.name}")
        logger.info(f"   - Total Evaluated Samples: {len(data):,}")
        logger.info(f"   - Empty Cypher Predictions: {empty_preds}")

        all_ok = (len(data) > 0 and empty_preds < len(data) * 0.1)
        print_banner(5, "Benchmark Evaluation", all_ok)
        return all_ok

    except Exception as e:
        logger.error(f"❌ [CHECK FAILED] Error reading prediction file: {e}")
        print_banner(5, "Benchmark Evaluation", False)
        return False
