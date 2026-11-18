"""Chạy một giao thức suy luận (cb_chat | cb_raw | cypherri) trên test CypherBench bằng MLX, greedy.

--per_graph N: lấy mẫu phân tầng N câu mỗi đồ thị (seed cố định) để thử nhanh; 0 = toàn bộ.
Ghi predictions.jsonl (chạy tiếp được nếu bị ngắt) + run_config.json; chấm bằng cb_eval_sequential.sh.

Ví dụ:
  .venv-cb/bin/python src/cb_run_protocol.py --model Qwen/Qwen2.5-Coder-7B --protocol cypherri \
      --per_graph 30 --out results/cypherbench/protocol_probe/base_cypherri
  # nhánh ontology (RQ1): --protocol cb_chat_card --card auto|graphinfo|placebo
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

from cb_common import load_split
from cb_protocols import load_schema, run_cb_chat, run_cb_chat_card, run_cb_raw, run_cypherri
from ontology.card import render_card

TEST_GRAPHS = ["company", "fictional_character", "flight_accident", "geography", "movie", "nba", "politics"]


def subset(items, per_graph, seed=0):
    if not per_graph:
        return items
    out = []
    for g in TEST_GRAPHS:
        xs = sorted([x for x in items if x.graph == g], key=lambda x: x.qid)
        out += random.Random(f"{seed}-{g}").sample(xs, min(per_graph, len(xs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=["cb_chat", "cb_chat_card", "cb_raw", "cypherri"], required=True)
    ap.add_argument("--per_graph", type=int, default=0)
    ap.add_argument("--card", choices=["auto", "graphinfo", "placebo"], default=None,
                    help="nguồn thẻ ontology cho giao thức cb_chat_card")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(l)["qid"] for l in pred_path.read_text().splitlines() if l.strip()} if pred_path.exists() else set()
    items = subset(load_split("test", set(TEST_GRAPHS)), args.per_graph)
    schemas = {g: load_schema(g) for g in TEST_GRAPHS}
    cards = {g: render_card(g, args.card) for g in TEST_GRAPHS} if args.protocol == "cb_chat_card" else {}
    if cards:
        (out / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False))
    model, tok = load(args.model)
    sampler = make_sampler(temp=0.0)
    (out / "run_config.json").write_text(json.dumps({**vars(args), "n": len(items), "decoding": "greedy",
                                                      "backend": "mlx_lm bf16", "machine": platform.platform()}, indent=1))
    t0 = time.time()
    with pred_path.open("a") as f:
        for i, x in enumerate(items):
            if x.qid in done:
                continue
            s = schemas[x.graph]
            if args.protocol == "cypherri":
                r = run_cypherri(model, tok, sampler, s, x.nl_question)
            elif args.protocol == "cb_chat_card":
                r = run_cb_chat_card(model, tok, sampler, s.to_str(exclude_description=True), cards[x.graph], x.nl_question)
            elif args.protocol == "cb_raw":
                r = run_cb_raw(model, tok, sampler, s.to_str(exclude_description=True), x.nl_question)
            else:
                r = run_cb_chat(model, tok, sampler, s.to_str(exclude_description=True), x.nl_question)
            f.write(json.dumps({"qid": x.qid, "graph": x.graph, **r}, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0:
                print(f"{i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
