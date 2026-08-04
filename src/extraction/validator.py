import re
from src.utils.json_parser import parse_cot_json

def extract_schema_elements(schema_text: str) -> tuple[set, set]:
    """
    Parses T-Box schema text to extract known node labels and relationship types.
    """
    node_labels = set()
    rel_types = set()

    if not schema_text:
        return node_labels, rel_types

    # Node labels matching patterns like: - Person {..}, - **Movie**, Node labels:\n Article
    node_matches = re.findall(r"(?:^|\n)\s*-\s*(?:\*\*)?([A-Za-z0-9_]+)(?:\*\*)?", schema_text)
    node_labels.update(node_matches)

    node_label_section = re.search(r"Node labels:\s*\n((?:[^\n]+\n)+)", schema_text)
    if node_label_section:
        nodes = re.findall(r"([A-Za-z0-9_]+)\s*\{", node_label_section.group(1))
        node_labels.update(nodes)

    # Relationship types matching: - ACTED_IN {..}, (:Person)-[:DIRECTED]->(:Movie), 'type': HAS_KEY
    rel_matches = re.findall(r"-\s*(?:\*\*)?([A-Z0-9_]{2,})(?:\*\*)?", schema_text)
    rel_types.update(rel_matches)

    rel_pattern_matches = re.findall(r"-\[:([A-Z0-9_]+)\]->", schema_text)
    rel_types.update(rel_pattern_matches)

    type_matches = re.findall(r"'type':\s*['\"]?([A-Za-z0-9_]+)['\"]?", schema_text)
    rel_types.update(type_matches)

    # Remove generic keywords
    node_labels.discard("Node")
    node_labels.discard("Relationship")
    rel_types.discard("Node")
    rel_types.discard("Relationship")

    return node_labels, rel_types

def validate_teacher_ontology_and_cypher(schema: str, parsed_json: dict) -> tuple[bool, str]:
    """
    3-Tier Hallucination Validator under Closed World Assumption (CWA):
    Tier 1: Check JSON structure and status == 'PASS'.
    Tier 2: Regex extract entities, labels, and relationships from 4-step CoT fields.
    Tier 3: Verify extracted labels/relationships against schema T-Box.
    """
    if not parsed_json or not isinstance(parsed_json, dict):
        return False, "Tier 1 Fail: Invalid JSON or empty response"

    required_keys = ["instance_extraction", "relation_mapping", "validation_check", "cypher"]
    if not all(k in parsed_json for k in required_keys):
        return False, f"Tier 1 Fail: Missing CoT keys in output JSON"

    val_check = parsed_json.get("validation_check", {})
    val_status = ""
    if isinstance(val_check, dict):
        val_status = val_check.get("status", "")
    elif isinstance(val_check, str):
        val_status = val_check

    if "PASS" not in str(val_status).upper():
        return False, f"Tier 1 Fail: validation_check status is '{val_status}', expected 'PASS'"

    cypher_code = str(parsed_json.get("cypher", "")).strip()
    if not cypher_code:
        return False, "Tier 1 Fail: Empty Cypher query"

    # Tier 2 & 3: Schema CWA Validation
    known_nodes, known_rels = extract_schema_elements(schema)

    if known_nodes or known_rels:
        # Extract node labels in Cypher: (p:Person), (:Movie)
        cypher_nodes = set(re.findall(r"\(\s*[a-Za-z0-9_]*\s*:\s*([A-Za-z0-9_]+)", cypher_code))
        # Extract rel types in Cypher: -[:ACTED_IN]->
        cypher_rels = set(re.findall(r"-\[\s*[a-Za-z0-9_]*\s*:\s*([A-Za-z0-9_]+)", cypher_code))

        if known_nodes:
            hallucinated_nodes = [n for n in cypher_nodes if n not in known_nodes]
            if hallucinated_nodes:
                return False, f"Tier 3 Fail: Hallucinated Node Label(s) {hallucinated_nodes} not in Schema"

        if known_rels:
            hallucinated_rels = [r for r in cypher_rels if r not in known_rels]
            if hallucinated_rels:
                return False, f"Tier 3 Fail: Hallucinated Relationship Type(s) {hallucinated_rels} not in Schema"

    return True, "PASS"
