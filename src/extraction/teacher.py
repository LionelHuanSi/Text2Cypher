import os
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm

from configs.config import GEMINI_API_KEY, OPENAI_API_KEY, TEACHER_MODEL_ID
from src.prompts.teacher_prompts import create_teacher_prompt
from src.utils.json_parser import parse_cot_json
from src.extraction.validator import validate_teacher_ontology_and_cypher
from src.utils.logger import setup_logger

logger = setup_logger("TeacherDistiller")

class GeminiTeacherExtractor:
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini API extraction will fail if key is missing!")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini Client ({self.model_name}).")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API Client: {e}")

    def generate_single(self, prompt: str) -> str:
        if not self.client:
            self._init_client()
        if not self.client:
            return ""

        try:
            response = self.client.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024}
            )
            return response.text if response else ""
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("Gemini Rate Limit (429). Sleeping 10s...")
                time.sleep(10)
            else:
                logger.error(f"Gemini API Error: {e}")
            return ""

class OpenAITeacherExtractor:
    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = "gpt-4o-mini", base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-ac17cce818d45be5-f2gyor-f42f21f8")
        self.model_name = model_name
        self.base_url = base_url or os.getenv("LOCAL_ROUTER_URL", "http://localhost:20128/v1")
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            if self.base_url:
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                logger.info(f"Initialized OpenAI/LocalRouter Client ({self.model_name}) at '{self.base_url}'.")
            else:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI Client ({self.model_name}).")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI/LocalRouter API Client: {e}")

    def generate_single(self, prompt: str, max_retries: int = 5) -> str:
        if not self.client:
            self._init_client()
        if not self.client:
            return ""

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Bạn là mô hình chuyên gia trích xuất tri thức và sinh truy vấn Cypher chuẩn xác. Trả về kết quả dưới dạng JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1024
                )
                return response.choices[0].message.content if response.choices else ""
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str or "connection" in err_str:
                    sleep_time = min(60, 5 * (2 ** (attempt - 1)))
                    logger.warning(f"LocalRouter/API Warning (Attempt {attempt}/{max_retries}): {e}. Waiting {sleep_time}s before retrying...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"OpenAI/LocalRouter API Error: {e}")
                    break
        return ""

def get_teacher_extractor(provider: str = TEACHER_MODEL_ID, base_url: str = None, api_key: str = None):
    p_lower = provider.lower()
    if "ag/" in p_lower or "localhost" in str(base_url).lower() or "20128" in str(base_url):
        return OpenAITeacherExtractor(
            api_key=api_key or "sk-ac17cce818d45be5-f2gyor-f42f21f8",
            model_name=provider if "ag/" in provider else "ag/gemini-3.6-flash-high",
            base_url=base_url or "http://localhost:20128/v1"
        )
    elif "gemini" in p_lower and not ("ag/" in p_lower or "localhost" in p_lower):
        return GeminiTeacherExtractor(model_name=provider if "gemini" in provider else "gemini-1.5-flash")
    elif "gpt" in p_lower or "openai" in p_lower:
        return OpenAITeacherExtractor(model_name=provider if "gpt" in provider else "gpt-4o-mini", base_url=base_url)
    else:
        # Default fallback to Local Router if ag/ or custom model specified
        return OpenAITeacherExtractor(
            api_key=api_key or "sk-ac17cce818d45be5-f2gyor-f42f21f8",
            model_name=provider,
            base_url=base_url or "http://localhost:20128/v1"
        )
