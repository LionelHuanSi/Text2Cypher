def create_teacher_prompt(schema: str, question: str) -> str:
    """Creates system prompt for Teacher LLM (Gemini / GPT-4o) to generate 4-step CoT reasoning JSON."""
    return f"""### System:
You are an expert Knowledge Graph & Neo4j Cypher Database Engineer.
Your task is to translate a natural language Question into a precise Cypher query based on the given Graph Schema (Ontology).

You MUST follow the 4-step Ontology CoT reasoning process:
1. instance_extraction: Extract A-Box individuals, entity values, and datatype properties mentioned in the Question.
2. relation_mapping: Map entity relationships to T-Box Object Properties with exact Domain, Range, and Direction.
3. validation_check: Verify T-Box schema compliance, node labels, property names, and confirm with status 'PASS'.
4. cypher: Write the executable Cypher query without any markdown formatting wrappers.

### Input:
Graph Schema (Ontology):
{schema}

Question:
{question}

### Response:
Provide your reasoning and Cypher query in the exact raw JSON format matching the key schema: instance_extraction, relation_mapping, validation_check, cypher. DO NOT wrap in markdown code blocks.
"""

def create_student_prompt(schema: str, question: str) -> str:
    """Creates prompt format for Student KD model to generate 4-step CoT JSON."""
    return f"""### System:
You are an expert Graph Database AI. Translate the natural language question into a Cypher query.
You MUST follow the exact same Ontology Instance Extraction, Relation Mapping, and Validation Check process as an expert.

### Input:
Ontology/Schema:
{schema}

Question:
{question}

### Response:
Provide your reasoning and Cypher query in the exact same raw JSON format (instance_extraction, relation_mapping, validation_check, cypher). DO NOT wrap in markdown blocks."""

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
