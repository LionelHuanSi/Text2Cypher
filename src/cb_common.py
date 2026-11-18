"""Tiện ích chung cho CypherBench: nạp dữ liệu, schema, kết nối Neo4j và metric chính thức.

Metric/connector/schema được import nguyên văn từ repo tác giả (references/cypherbench @ 4cc727d),
không sửa, để số đo so sánh được với bài báo.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB_REPO = ROOT / "references/cypherbench"
RAW = ROOT / "data/cypherbench/raw"
sys.path.insert(0, str(CB_REPO))

from cypherbench.schema import Nl2CypherSample, PropertyGraphSchema, DataType  # noqa: E402
from cypherbench.neo4j_connector import Neo4jConnector  # noqa: E402
from cypherbench.metrics import execution_accuracy, executable, provenance_subgraph_jaccard_similarity  # noqa: E402

NEO4J_INFO = json.loads((CB_REPO / "neo4j_info.json").read_text())

# Prompt zero-shot nguyên văn của bài gốc (cypherbench/baseline/zero_shot_nl2cypher.py)
NL2CYPHER_PROMPT_DEFAULT = """Translate the question to Cypher query based on the schema of a Neo4j knowledge graph.
- Output the Cypher query in a single line, without any additional output or explanation. Do not wrap the query with any formatting like ```.
- Perform graph pattern matching in the `MATCH` clause if possible.
- Avoid listing the same entity multiple times in the results. However, if multiple distinct entities share the same name, their names should be repeated as separate entries.
- Do not return node objects. Instead, return entity names or properties.

Graph Schema:
{schema}

Question: {question}
Cypher: """


def load_split(split: str, graphs=None):
    items = [Nl2CypherSample(**x) for x in json.loads((RAW / f"{split}.json").read_text())]
    return [x for x in items if graphs is None or x.graph in graphs]


def schema_str(graph: str) -> str:
    """Giống load_schema_from_json của bài gốc."""
    path = RAW / f"graphs/schemas/{graph}_schema.json"
    schema = PropertyGraphSchema.from_json(json.loads(path.read_text()),
                                           add_meta_properties={"name": DataType.STR}).to_sorted()
    return schema.to_str(exclude_description=True)


def connector(graph: str) -> Neo4jConnector:
    return Neo4jConnector(name=graph, **NEO4J_INFO["full"][graph])


def clean_prediction(text: str) -> str:
    """Hậu xử lý giống bài gốc: bỏ fence ``` và khoảng trắng."""
    return text.replace("```cypher", "").replace("```", "").strip()
