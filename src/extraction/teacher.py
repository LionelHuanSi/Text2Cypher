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
                logger.warning(f"Gemini Rate Limit (429). Sleeping 10s...")
                time.sleep(10)
            else:
                logger.error(f"Gemini API Error: {e}")
            return ""

class OpenAITeacherExtractor:
    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set.")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI Client ({self.model_name}).")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API Client: {e}")

    def generate_single(self, prompt: str) -> str:
        if not self.client:
            self._init_client()
        if not self.client:
            return ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content if response.choices else ""
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            return ""

def get_teacher_extractor(provider: str = TEACHER_MODEL_ID):
    p_lower = provider.lower()
    if "gemini" in p_lower:
        return GeminiTeacherExtractor(model_name=provider if "gemini" in provider else "gemini-1.5-flash")
    elif "gpt" in p_lower or "openai" in p_lower:
        return OpenAITeacherExtractor(model_name=provider if "gpt" in provider else "gpt-4o-mini")
    else:
        logger.info(f"Falling back to Gemini Extractor for '{provider}'...")
        return GeminiTeacherExtractor()
