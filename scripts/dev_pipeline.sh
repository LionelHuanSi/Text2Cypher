#!/bin/bash
# Thí nghiệm pipeline Teacher + kiểm/sửa trên TẬP DEV (đồ thị train), 8 nhánh, rồi chấm.
# Dùng: scripts/dev_pipeline.sh  (sau khi có results/cypherbench/dev/qwen7b_instruct_A/predictions.jsonl)
set -e
cd "$(dirname "$0")/.."
D=results/cypherbench/dev
A=$D/qwen7b_instruct_A
GRAPHS="art soccer terrorist_attack"
P=".venv-cb/bin/python src/cb_pipeline.py --pred_dir $A --split train"
C="docker compose -f infra/cypherbench/compose.yaml -p cypherbench"
# 1) nhánh sửa tự động (không cần Neo4j, không gọi LLM)
$P --graphs $GRAPHS --level schema   --out $D/S
$P --graphs $GRAPHS --level ontology --out $D/O
$P --graphs $GRAPHS --level ontology --placebo --out $D/Op
# 2) nhánh có Teacher sửa lại (cần Neo4j cho EXPLAIN)
for g in $GRAPHS; do
  svc=${g//_/-}
  $C stop >/dev/null 2>&1 || true
  $C up -d "$svc" >/dev/null 2>&1
  until docker exec "cypherbench-$svc" cypher-shell -u neo4j -p cypherbench "RETURN 1" >/dev/null 2>&1; do sleep 5; done
  $P --graphs $g --level schema   --repair specific --out $D/S_R
  $P --graphs $g --level ontology --repair specific --out $D/O_R
  $P --graphs $g --level ontology --repair generic  --out $D/O_G
  $P --graphs $g --level ontology --placebo --repair specific --out $D/Op_R
done
# 3) chấm
SPLIT=train GRAPHS="$GRAPHS" scripts/cb_eval_multi.sh $A $D/S $D/O $D/Op $D/S_R $D/O_R $D/O_G $D/Op_R
