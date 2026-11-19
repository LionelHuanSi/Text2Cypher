#!/bin/bash
# Chấm NHIỀU thư mục dự đoán cùng lúc: mỗi đồ thị chỉ bật Neo4j một lần rồi chấm mọi thư mục.
# Dùng: scripts/cb_eval_multi.sh <pred_dir> [<pred_dir> ...]
# Tập dev: SPLIT=train GRAPHS="art soccer terrorist_attack" scripts/cb_eval_multi.sh <pred_dir> ...
set -e
cd "$(dirname "$0")/.."
C="docker compose -f infra/cypherbench/compose.yaml -p cypherbench"
SPLIT=${SPLIT:-test}
GRAPHS=${GRAPHS:-company fictional_character flight_accident geography movie nba politics}
for g in $GRAPHS; do
  svc=${g//_/-}
  $C stop >/dev/null 2>&1 || true
  $C up -d "$svc" >/dev/null 2>&1
  until docker exec "cypherbench-$svc" cypher-shell -u neo4j -p cypherbench "RETURN 1" >/dev/null 2>&1; do sleep 5; done
  for d in "$@"; do
    echo "$(date +%H:%M) $g $d"
    .venv-cb/bin/python src/cb_evaluate.py --pred_dir "$d" --graphs "$g" --split "$SPLIT" >/dev/null || echo "FAIL $g $d"
  done
done
$C stop >/dev/null 2>&1
for d in "$@"; do
  python3 -c "import json,sys;m=json.load(open(sys.argv[1]+'/metrics.json'));print(sys.argv[1],m['n'],m['overall'],m['EX_ci95'])" "$d"
done
