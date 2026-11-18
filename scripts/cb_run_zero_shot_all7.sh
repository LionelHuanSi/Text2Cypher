#!/bin/bash
# Toàn bộ quy trình zero-shot 7 đồ thị test: sinh (tự chạy tiếp nếu bị ngắt) -> nạp movie/politics -> chấm tuần tự.
# Dùng: nohup scripts/cb_run_zero_shot_all7.sh > results/cypherbench/zero_shot_all7_pipeline.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
P=${P:-results/cypherbench/zero_shot/qwen2.5-coder-7b-instruct}
.venv-cb/bin/python src/cb_zero_shot.py --model "$MODEL" \
  --graphs company fictional_character flight_accident geography movie nba politics --out "$P"
echo "gen done: $(wc -l < $P/predictions.jsonl)"
for g in movie politics; do
  echo "== load $g"; bash scripts/cb_load_graph.sh "$g" | tail -2
  docker compose -f infra/cypherbench/compose.yaml -p cypherbench stop >/dev/null 2>&1
done
bash scripts/cb_eval_sequential.sh "$P"
echo ALL_DONE
