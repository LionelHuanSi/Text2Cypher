import sacrebleu

def compute_google_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Computes Google-BLEU (SacreBLEU sentence_bleu avg) score between predicted Cypher queries and ground truth.
    """
    if not predictions or not references:
        return 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        p_clean = pred.strip()
        r_clean = ref.strip()
        if not r_clean:
            continue
        bleu = sacrebleu.sentence_bleu(p_clean, [r_clean]).score / 100.0
        scores.append(bleu)

    return sum(scores) / len(scores) if scores else 0.0

def compute_execution_exact_match(pred_results: list, gold_results: list) -> bool:
    """
    Checks if execution results of predicted Cypher match gold Cypher execution results.
    """
    if pred_results is None or gold_results is None:
        return False
    try:
        return sorted(str(x) for x in pred_results) == sorted(str(x) for x in gold_results)
    except Exception:
        return pred_results == gold_results
