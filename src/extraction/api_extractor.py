import os
import time
import logging
import requests
from typing import List
from src.extraction.base_extractor import BaseTeacherExtractor
from configs.config import GEMINI_API_KEY, OPENAI_API_KEY
from src.prompts.teacher_prompts import TEACHER_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITeacherExtractor(BaseTeacherExtractor):
    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o-mini"):
        self.provider = provider.lower()
        self.model_name = model_name
        self.gemini_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.quota_exceeded = False
        logger.info(f"Initialized API Extractor with Provider='{self.provider}', Model='{self.model_name}'")

    def _call_gemini_single(self, prompt: str) -> str:
        if self.quota_exceeded:
            return ""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_key}"
        full_prompt = f"{TEACHER_SYSTEM_PROMPT}\n\n{prompt}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
        }
        try:
            res = requests.post(url, json=payload, timeout=60)
            res_json = res.json()
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                if "error" in res_json and ("quota" in str(res_json["error"]).lower() or "limit" in str(res_json["error"]).lower()):
                    logger.warning("Gemini API Quota Exceeded! Stopping extraction gracefully.")
                    self.quota_exceeded = True
                return ""
        except Exception as e:
            logger.error(f"Gemini API request error: {e}")
            return ""

    def _call_openai_single(self, prompt: str) -> str:
        if self.quota_exceeded:
            return ""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            res_json = res.json()
            if "choices" in res_json and len(res_json["choices"]) > 0:
                return res_json["choices"][0]["message"]["content"]
            else:
                err_msg = str(res_json.get("error", "")).lower()
                if "quota" in err_msg or "insufficient_quota" in err_msg or "billing" in err_msg:
                    logger.warning("OpenAI API Quota/Balance Exhausted! Gracefully saving progress and stopping extraction.")
                    self.quota_exceeded = True
                else:
                    logger.error(f"OpenAI API Error Response: {res_json}")
                return ""
        except Exception as e:
            logger.error(f"OpenAI API request error: {e}")
            return ""

    def generate_single(self, prompt: str) -> str:
        if self.provider == "gemini":
            return self._call_gemini_single(prompt)
        elif self.provider == "openai":
            return self._call_openai_single(prompt)
        return ""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        results = []
        for i, prompt in enumerate(prompts):
            if self.quota_exceeded:
                logger.warning(f"Quota exceeded signal detected. Stopping batch at prompt {i}/{len(prompts)}.")
                break
            if i > 0 and i % 5 == 0:
                logger.info(f"API Extractor Progress: {i}/{len(prompts)} completed...")
                time.sleep(0.2)
            out = self.generate_single(prompt)
            results.append(out)
        return results
