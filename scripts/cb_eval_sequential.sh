#!/bin/bash
# Chấm lần lượt từng đồ thị test: bật Neo4j -> đợi sẵn sàng -> chấm -> tắt.
# VM Docker chỉ ~7,6 GB nên mỗi lúc chỉ bật một đồ thị. Đồ thị phải được nạp trước
# (scripts/cb_load_graph.sh); dữ liệu nằm trong volume của container nên stop/start không mất.
# Dùng: scripts/cb_eval_sequential.sh results/cypherbench/zero_shot/<model> [graph ...]
set -e
cd "$(dirname "$0")/.."
PRED_DIR=$1; shift
GRAPHS=${@:-company fictional_character flight_accident geography movie nba politics}
C="docker compose -f infra/cypherbench/compose.yaml -p cypherbench"
for g in $GRAPHS; do
  svc=${g//_/-}
  $C stop >/dev/null 2>&1 || true
  $C up -d "$svc" >/dev/null 2>&1
  until docker exec "cypherbench-$svc" cypher-shell -u neo4j -p cypherbench "RETURN 1" >/dev/null 2>&1; do sleep 5; done
  echo "== $g"
  .venv-cb/bin/python src/cb_evaluate.py --pred_dir "$PRED_DIR" --graphs "$g" > /dev/null
done
$C stop >/dev/null 2>&1
python3 -c "import json,sys; m=json.load(open(sys.argv[1]+'/metrics.json')); print(json.dumps({k:m[k] for k in ['n','overall','EX_ci95','by_graph']}, indent=1))" "$PRED_DIR"
