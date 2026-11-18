# Ontology-Constrained Knowledge Distillation cho Text-to-Cypher

**Toàn bộ lựa chọn, kết quả và tiến độ nằm trong một file: [reports/BAO_CAO.md](reports/BAO_CAO.md).**

## Cấu trúc
| | |
|---|---|
| `reports/BAO_CAO.md` | báo cáo tiến độ duy nhất (cập nhật sau mỗi bước) |
| `src/cb_common.py` | prompt, schema, metric chính thức (import từ repo tác giả) |
| `src/cb_zero_shot.py` | sinh Cypher zero-shot bằng MLX |
| `src/cb_protocols.py`, `src/cb_run_protocol.py` | các giao thức suy luận (CypherBench, Cypher-RI) |
| `src/cb_evaluate.py` | chấm EX / PSJS / Exec. + CI 95%, theo từng đồ thị |
| `src/cb_error_analysis.py` | phân loại lỗi + kiểm vi phạm schema/ontology |
| `src/ontology/builder.py` | **Ontology Builder** tự động (Neo4j + Wikidata) |
| `src/ontology/card.py` | thẻ ontology cho prompt (auto / graphinfo / placebo) |
| `src/ontology/eval_graphinfo.py` | đối chiếu ontology tự xây với chú thích tay |
| `ontology/` | ontology đã dựng cho 10 đồ thị + cache Wikidata |
| `scripts/build_ontologies.sh` | dựng ontology lần lượt cho các đồ thị |
| `infra/cypherbench/compose.yaml` | Neo4j 7 đồ thị test + 3 đồ thị train (cổng như repo tác giả, chỉ chạy Neo4j) |
| `scripts/cb_load_graph.sh` | nạp đồ thị từ host bằng loader gốc của tác giả |
| `scripts/cb_eval_sequential.sh` | chấm lần lượt từng đồ thị (mỗi lúc một Neo4j) |
| `scripts/cb_run_zero_shot_all7.sh` | toàn bộ quy trình zero-shot 7 đồ thị |
| `data/cypherbench/manifest.json` | nguồn + checksum dữ liệu (`raw/` không vào git) |
| `references/cypherbench/` | repo tác giả @ `4cc727d` (không vào git) |

Phần MedT2C cũ: `~/Text2Cypher_medt2c_20261118.zip`.
