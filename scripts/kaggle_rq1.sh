#!/bin/bash
# Sinh Cypher cho các nhánh RQ1 trên Kaggle (2x T4), tải kết quả về rồi chấm trên máy.
#   scripts/kaggle_rq1.sh push  "Qwen/Qwen2.5-Coder-7B-Instruct" "A B C D"   # tạo/cập nhật dataset + chạy kernel
#   scripts/kaggle_rq1.sh status
#   scripts/kaggle_rq1.sh pull                                               # tải về results/cypherbench/rq1_kaggle/
# Cần token Kaggle (~/.kaggle/kaggle.json, hoặc KAGGLE_API_TOKEN + KAGGLE_USERNAME). Prompt dựng trước bằng: cd src && ../.venv-cb/bin/python kaggle_build_prompts.py
set -e
cd "$(dirname "$0")/.."
K=.venv-cb/bin/kaggle
USER=${KAGGLE_USERNAME:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" 2>/dev/null)}
[ -n "$USER" ] || { echo "Thiếu tên Kaggle: đặt ~/.kaggle/kaggle.json hoặc KAGGLE_USERNAME"; exit 1; }
DS=text2cypher-rq1-prompts
KS=text2cypher-rq1-gen
case "$1" in
push)
  MODELS=${2:-Qwen/Qwen2.5-Coder-7B-Instruct}; ARMS=${3:-A B C D}
  cat > kaggle/dataset/dataset-metadata.json <<J
{"title": "$DS", "id": "$USER/$DS", "licenses": [{"name": "CC-BY-SA-4.0"}]}
J
  if $K datasets status "$USER/$DS" >/dev/null 2>&1; then
    $K datasets version -p kaggle/dataset -m "update $(date +%F_%H%M)" -q
  else
    $K datasets create -p kaggle/dataset -q
  fi
  mlist=$(python3 -c "import sys;print(repr(sys.argv[1].split()))" "$MODELS")
  alist=$(python3 -c "import sys;print(repr(sys.argv[1].split()))" "$ARMS")
  sed -e "s|__MODELS__|$mlist|" -e "s|__ARMS__|$alist|" kaggle/kernel/run_template.py > kaggle/kernel/run.py
  cat > kaggle/kernel/kernel-metadata.json <<J
{"id": "$USER/$KS", "title": "$KS", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$USER/$DS"], "competition_sources": [], "kernel_sources": []}
J
  # chờ dataset xử lý xong
  until $K datasets status "$USER/$DS" 2>/dev/null | grep -qi ready; do sleep 10; done
  $K kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4
  ;;
status) $K kernels status "$USER/$KS" ;;
pull)
  d=results/cypherbench/rq1_kaggle/_download; rm -rf "$d"; mkdir -p "$d"
  $K kernels output "$USER/$KS" -p "$d"
  tar xzf "$d/out.tgz" -C "$d"
  for m in "$d"/out/*/; do
    slug=$(basename "$m")
    for a in "$m"*/; do
      dst=results/cypherbench/rq1_kaggle/$slug/$(basename "$a"); mkdir -p "$dst"
      cp "$a"predictions.jsonl "$a"run_config.json "$dst"/; echo "$dst $(wc -l < "$dst/predictions.jsonl")"
    done
  done
  ;;
esac
