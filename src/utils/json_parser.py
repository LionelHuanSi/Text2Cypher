import json
import re

def parse_cot_json(text: str) -> dict:
    """
    Extracts and parses 4-step CoT JSON output from model response string.
    Returns a dict with keys: instance_extraction, relation_mapping, validation_check, cypher.
    """
    if not text:
        return {}
    
    # Try direct json parse first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Extract block inside ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Fallback to widest brace pair
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    return {}
