#!/bin/bash
# Nạp một đồ thị CypherBench vào Neo4j (container), chạy loader.py gốc của image trên host.
# Lý do: loader trong container bị OOM với movie/politics (VM Docker ~7,6 GB).
# loader.py/schema.py lấy nguyên văn từ megagonlabs/neo4j-with-loader:2.4 -> references/neo4j_loader/.
# Dùng: scripts/cb_load_graph.sh <graph>      (vd. movie)
set -e
cd "$(dirname "$0")/.."
g=$1; svc=${g//_/-}
port=$(python3 -c "import json,sys;print(json.load(open('references/cypherbench/neo4j_info.json'))['full'][sys.argv[1]]['port'])" "$g")
docker compose -f infra/cypherbench/compose.yaml -p cypherbench up -d "$svc" >/dev/null
until docker exec "cypherbench-$svc" cypher-shell -u neo4j -p cypherbench "RETURN 1" >/dev/null 2>&1; do sleep 5; done
cd references/neo4j_loader
NEO4J_AUTH=neo4j/cypherbench ../../.venv-cb/bin/python loader.py \
  --input_path "../../data/cypherbench/raw/graphs/simplekg/${g}_simplekg.json" \
  --uri "bolt://localhost:$port" --overwrite 2>&1 | grep -v -E "it/s\]|Index created|Dropped"
