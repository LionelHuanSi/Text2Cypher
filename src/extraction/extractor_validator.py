import re
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CYPHER_KEYWORDS = {
    'match', 'return', 'where', 'and', 'or', 'as', 'limit', 'skip',
    'create', 'set', 'with', 'optional', 'call', 'unwind', 'count',
    'distinct', 'order', 'by', 'desc', 'asc', 'starts', 'ends', 'contains', 'in'
}

def validate_teacher_ontology_and_cypher(schema_str: str, parsed_output: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Advanced 3-tier Ontology & Schema Validation:
    Tier 1: JSON Structure & Validation Check Flag
    Tier 2: Term Extraction Regex across instances, relations, and Cypher
    Tier 3: Strict Schema T-Box cross-referencing to eliminate hallucinated classes/properties
    """
    if not isinstance(parsed_output, dict):
        return False, "Invalid JSON structure."

    cypher = parsed_output.get("cypher", "")
    instances = str(parsed_output.get("instance_extraction", ""))
    relations = str(parsed_output.get("relation_mapping", ""))
    val_check = str(parsed_output.get("validation_check", "")).upper()

    # Self-validation check by Teacher
    if "FAIL" in val_check:
        return False, f"Teacher self-validation check failed: {val_check}"

    if not cypher:
        return False, "Empty or missing Cypher query."

    schema_lower = schema_str.lower()

    # Regex search for terms inside backticks or relationship brackets: e.g. `Person`, [:ACTED_IN]
    raw_terms = re.findall(r'`([A-Za-z0-9_]+)`|\[[:]?([A-Za-z0-9_]+)\]|:\s*([A-Za-z0-9_]+)', f"{instances} {relations} {cypher}")
    
    terms_to_check = []
    for sub in raw_terms:
        for t in sub:
            if t:
                term_clean = t.lower()
                # Ignore pure numeric string literals and cypher syntax keywords
                if term_clean.isdigit():
                    continue
                if term_clean not in CYPHER_KEYWORDS and len(term_clean) > 2:
                    terms_to_check.append(term_clean)

    # Cross-reference terms against input schema
    hallucinated = []
    for term in set(terms_to_check):
        if term not in schema_lower:
            hallucinated.append(term)

    if hallucinated:
        return False, f"Teacher hallucinated ontology terms not in schema: {hallucinated}"

    return True, "PASS"
