import logging
from typing import List
from src.extraction.base_extractor import BaseTeacherExtractor
from configs.config import TEACHER_MODEL_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMTeacherExtractor(BaseTeacherExtractor):
    def __init__(self, model_id: str = TEACHER_MODEL_ID, tensor_parallel_size: int = 1):
        logger.info(f"Initializing Local vLLM Teacher Model: '{model_id}'...")
        try:
            # pyrefly: ignore [missing-import]
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024
            )
            logger.info("vLLM Teacher Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise e

    def generate_batch(self, prompts: List[str]) -> List[str]:
        logger.info(f"Running vLLM batch generation for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
