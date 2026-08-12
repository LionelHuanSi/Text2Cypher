import json
import logging
from pathlib import Path
from configs.config import DISTILLATION_TRAIN_37K_PATH, TRAIN_BASELINE_37K_PATH, TRAIN_KD_37K_PATH
from src.prompts.teacher_prompts import create_student_prompt, create_baseline_prompt
from src.utils.logger import setup_logger

logger = setup_logger("DataFormatter")

def export_sft_datasets():
    """
    Reads the full-scale clean distillation dataset (37k) and exports two SFT training sets:
    1. Baseline SFT: Question + Schema -> Raw Cypher query
    2. Proposed KD: Question + Schema -> 4-Step Ontology CoT JSON
    """
    if not DISTILLATION_TRAIN_37K_PATH.exists():
        logger.error(f"File not found: '{DISTILLATION_TRAIN_37K_PATH}'. Please run Stage 2 distillation first!")
        return 0, 0

    with open(DISTILLATION_TRAIN_37K_PATH, "r", encoding="utf-8") as f:
        distillation_data = json.load(f)

    logger.info(f"Loaded {len(distillation_data)} valid distillation samples from '{DISTILLATION_TRAIN_37K_PATH}'.")

    baseline_samples = []
    kd_samples = []

    for item in distillation_data:
        schema = item["schema"]
        question = item["question"]
        # Filter out invalid / malformed samples
        if item.get("is_valid") is False:
            continue

        cypher = item.get("teacher_cypher", item.get("ground_truth_cypher", "")).strip()
        if not cypher:
            continue

        # 1. Direct SFT Baseline format
        baseline_prompt = create_baseline_prompt(schema, question)
        baseline_samples.append({
            "prompt": baseline_prompt,
            "completion": cypher,
            "text": f"{baseline_prompt}{cypher}"
        })

        # 2. Proposed KD (Strict 4-step JSON) format
        kd_prompt = create_student_prompt(schema, question)
        
        inst_ext = item.get("instance_extraction", [])
        if isinstance(inst_ext, str):
            inst_ext = [{"entity_or_property": "concept", "value": inst_ext, "entity_type": "Entity"}]
        elif not isinstance(inst_ext, list):
            inst_ext = [inst_ext] if inst_ext else []

        rel_map = item.get("relation_mapping", [])
        if isinstance(rel_map, str):
            rel_map = [{"source_node": "Node", "relation": rel_map, "target_node": "Node", "direction": "OUTGOING"}]
        elif not isinstance(rel_map, list):
            rel_map = [rel_map] if rel_map else []

        val_chk = item.get("validation_check", {"status": "PASS", "description": "Validation successful."})
        if isinstance(val_chk, str):
            val_chk = {"status": "PASS", "description": val_chk}
        elif not isinstance(val_chk, dict) or "status" not in val_chk:
            val_chk = {"status": "PASS", "description": "Validation successful."}

        cot_dict = {
            "instance_extraction": inst_ext,
            "relation_mapping": rel_map,
            "validation_check": val_chk,
            "cypher": cypher
        }
        json_output = json.dumps(cot_dict, ensure_ascii=False)

        kd_samples.append({
            "prompt": kd_prompt,
            "completion": json_output,
            "text": f"{kd_prompt}{json_output}"
        })

    # Save Baseline SFT dataset
    with open(TRAIN_BASELINE_37K_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved Baseline SFT dataset ({len(baseline_samples)} samples) to '{TRAIN_BASELINE_37K_PATH}'")

    # Save Proposed KD SFT dataset
    with open(TRAIN_KD_37K_PATH, "w", encoding="utf-8") as f:
        json.dump(kd_samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved Proposed KD dataset ({len(kd_samples)} samples) to '{TRAIN_KD_37K_PATH}'")

    return len(baseline_samples), len(kd_samples)
