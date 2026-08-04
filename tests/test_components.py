import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.json_parser import parse_teacher_or_student_output
from src.extraction.extractor_validator import validate_teacher_ontology_and_cypher

class TestKDComponents(unittest.TestCase):
    def test_json_parser_strict(self):
        valid_json = '{"ontology_mapping": "Person -> Node", "validation_trace": "Passed", "cypher": "MATCH (n:Person) RETURN n"}'
        parsed = parse_teacher_or_student_output(valid_json)
        self.assertEqual(parsed["cypher"], "MATCH (n:Person) RETURN n")

    def test_json_parser_markdown_fallback(self):
        markdown_json = '''Here is your output:
```json
{
  "ontology_mapping": "Movie -> Node",
  "validation_trace": "Valid",
  "cypher": "MATCH (m:Movie) RETURN m"
}
```
Thank you!'''
        parsed = parse_teacher_or_student_output(markdown_json)
        self.assertEqual(parsed["cypher"], "MATCH (m:Movie) RETURN m")

    def test_json_parser_cypher_regex_fallback(self):
        malformed_output = "I cannot generate JSON properly, but here is your query: MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a, m"
        parsed = parse_teacher_or_student_output(malformed_output)
        self.assertIn("MATCH (a:Actor)", parsed["cypher"])

    def test_hallucination_validator_valid(self):
        schema = "(:Person {name: STRING})-[:KNOWS]->(:Person)"
        output = {"ontology_mapping": "Person -> Node", "cypher": "MATCH (p:Person) RETURN p.name"}
        is_valid, reason = validate_teacher_ontology_and_cypher(schema, output)
        self.assertTrue(is_valid)

    def test_hallucination_validator_invalid_label(self):
        schema = "(:Person {name: STRING})-[:KNOWS]->(:Person)"
        output = {"ontology_mapping": "FakeNode -> Node", "cypher": "MATCH (f:FakeNode) RETURN f"}
        is_valid, reason = validate_teacher_ontology_and_cypher(schema, output)
        self.assertFalse(is_valid)
        self.assertIn("fakenode", reason.lower())

if __name__ == "__main__":
    unittest.main()
