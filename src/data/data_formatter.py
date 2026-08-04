import json
from typing import Dict
from src.prompts.teacher_prompts import create_student_prompt

def format_sample_for_student_baseline(sample: Dict) -> Dict:
    """
    Format for Baseline SFT Student (Question + Schema -> Direct Cypher)
    """
    prompt = create_student_prompt(sample["schema"], sample["question"])
    target = f"```cypher\n{sample.get('cypher', '')}\n```"
    return {
        "text": f"{prompt}{target}"
    }

def format_sample_for_student_kd(sample: Dict) -> Dict:
    """
    Format for Proposed Ontology-Aware KD Student (Question + Schema -> Dynamic 4-step Ontology Reasoning JSON)
    """
    prompt = create_student_prompt(sample["schema"], sample["question"])
    
    # Complete 4-step Ontology Reasoning JSON target for Student Knowledge Distillation
    target_dict = {
        "instance_extraction": sample.get("instance_extraction", sample.get("ontology_mapping", "")),
        "relation_mapping": sample.get("relation_mapping", ""),
        "validation_check": sample.get("validation_check", "PASS"),
        "cypher": sample.get("teacher_cypher", sample.get("cypher", ""))
    }
    target = json.dumps(target_dict, ensure_ascii=False, indent=2)
    return {
        "text": f"{prompt}{target}"
    }
