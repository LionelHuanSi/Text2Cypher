import logging
from typing import List
import sacrebleu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_google_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Computes Google-BLEU (Sentence BLEU / SacreBLEU) score between predicted Cyphers and ground truth Cyphers.
    """
    if not predictions or not references:
        return 0.0

    # Ensure all inputs are strings
    preds = [p.strip() for p in predictions]
    refs = [[r.strip() for r in references]]

    try:
        bleu = sacrebleu.corpus_bleu(preds, refs)
        return round(bleu.score, 4)
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0
