#!/bin/bash
# Chạy Ontology Builder lần lượt cho các đồ thị (mỗi lúc một Neo4j), rồi đối chiếu với graph_info.json.
# Dùng: scripts/build_ontologies.sh [graph ...]
set -e
cd "$(dirname "$0")/.."
GRAPHS=${@:-terrorist_attack soccer art company fictional_character flight_accident geography movie nba politics}
C="docker compose -f infra/cypherbench/compose.yaml -p cypherbench"
for g in $GRAPHS; do
  svc=${g//_/-}
  $C stop >/dev/null 2>&1 || true
  $C up -d "$svc" >/dev/null 2>&1
  until docker exec "cypherbench-$svc" cypher-shell -u neo4j -p cypherbench "RETURN 1" >/dev/null 2>&1; do sleep 5; done
  .venv-cb/bin/python src/ontology/builder.py --graph "$g"
done
$C stop >/dev/null 2>&1
.venv-cb/bin/python src/ontology/eval_graphinfo.py
