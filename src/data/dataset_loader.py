import logging
from datasets import load_dataset, DatasetDict
from configs.config import HF_DATASET_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_text2cypher_dataset() -> DatasetDict:
    """
    Downloads and returns the official HuggingFace text2cypher-2024v1 dataset.
    Train split: 39,554 instances
    Test split: 4,833 instances
    """
    logger.info(f"Loading dataset '{HF_DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(HF_DATASET_NAME)
    logger.info(f"Dataset loaded successfully: Train samples = {len(dataset['train'])}, Test samples = {len(dataset['test'])}")
    return dataset
