def create_teacher_prompt(schema: str, question: str) -> str:
    """Creates system prompt for Teacher LLM (Gemini / GPT-4o) to generate 4-step CoT reasoning JSON."""
    return f"""### System:
You are an expert Knowledge Graph & Neo4j Cypher Database Engineer.
Translate the natural language Question into a precise Cypher query based on the given Graph Schema (Ontology).

You MUST return a single raw JSON object matching this EXACT schema structure:
{{
  "instance_extraction": [
    {{"entity_or_property": "string", "value": "string or number", "entity_type": "string"}}
  ],
  "relation_mapping": [
    {{"source_node": "string", "relation": "string", "target_node": "string", "direction": "OUTGOING|INCOMING"}}
  ],
  "validation_check": {{"status": "PASS", "description": "string"}},
  "cypher": "string"
}}

### Input:
Graph Schema (Ontology):
{schema}

Question:
{question}

### Response:
Return only the raw JSON object. DO NOT wrap in markdown blocks."""

def create_student_prompt(schema: str, question: str) -> str:
    """Creates prompt format for Student KD model to generate 4-step CoT JSON."""
    return f"""### System:
You are an expert Graph Database AI. Translate the natural language question into a Cypher query.
You MUST return a single raw JSON object matching this EXACT schema structure:
{{
  "instance_extraction": [
    {{"entity_or_property": "string", "value": "string or number", "entity_type": "string"}}
  ],
  "relation_mapping": [
    {{"source_node": "string", "relation": "string", "target_node": "string", "direction": "OUTGOING|INCOMING"}}
  ],
  "validation_check": {{"status": "PASS", "description": "string"}},
  "cypher": "string"
}}

### Input:
Ontology/Schema:
{schema}

Question:
{question}

### Response:
Return only the raw JSON object."""

def create_baseline_prompt(schema: str, question: str) -> str:
    """Creates prompt format for Direct SFT Student Baseline (Question + Schema -> Cypher)."""
    return f"""### System:
You are an expert Cypher Query Generator. Translate the question into a Cypher query using the provided schema.

### Input:
Schema:
{schema}

Question:
{question}

### Response:
"""
