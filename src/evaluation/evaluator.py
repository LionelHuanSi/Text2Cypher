import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm

from configs.config import TEST_FULL_PATH, OUTPUTS_DIR
from src.evaluation.metrics import compute_google_bleu
from src.utils.json_parser import parse_cot_json
from src.utils.logger import setup_logger

logger = setup_logger("Evaluator")

def evaluate_model_on_testset(
    adapter_path: Path,
    base_model_id: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    is_kd: bool = True,
    test_path: Path = TEST_FULL_PATH,
    sample_limit: int = None
):
    """
    Evaluates a trained student model adapter on the test dataset.
    """
    if not test_path.exists():
        logger.error(f"Test dataset file not found: '{test_path}'")
        return {}

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if sample_limit:
        test_data = test_data[:sample_limit]

    logger.info(f"Evaluating model '{adapter_path.name}' on {len(test_data)} test samples...")

    # Load Model
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=2048,
        load_in_4bit=True
    )
    model.load_adapter(str(adapter_path))
    FastLanguageModel.for_inference(model)

    predictions = []
    references = []
    records = []

    for item in tqdm(test_data, desc="Evaluating Test Set"):
        schema = item["schema"]
        question = item["question"]
        gold_cypher = item.get("cypher", "")

        if is_kd:
            from src.prompts.teacher_prompts import create_student_prompt
            prompt = create_student_prompt(schema, question)
        else:
            from src.prompts.teacher_prompts import create_baseline_prompt
            prompt = create_baseline_prompt(schema, question)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        pred_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Extract Cypher if output is CoT JSON
        pred_cypher = pred_raw
        if is_kd:
            parsed = parse_cot_json(pred_raw)
            if parsed and "cypher" in parsed:
                pred_cypher = str(parsed["cypher"]).strip()

        predictions.append(pred_cypher)
        references.append(gold_cypher)

        records.append({
            "question": question,
            "gold_cypher": gold_cypher,
            "predicted_cypher": pred_cypher,
            "raw_output": pred_raw
        })

    bleu_score = compute_google_bleu(predictions, references)
    logger.info(f"=== EVALUATION RESULT: {adapter_path.name} ===")
    logger.info(f" - Google-BLEU Score: {bleu_score:.4f}")

    # Save predictions
    pred_save_path = OUTPUTS_DIR / f"preds_{adapter_path.name}.json"
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return {
        "model_name": adapter_path.name,
        "bleu_score": bleu_score,
        "total_samples": len(test_data),
        "predictions_path": str(pred_save_path)
    }
