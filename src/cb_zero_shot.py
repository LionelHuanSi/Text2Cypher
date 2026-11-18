"""Zero-shot Text-to-Cypher trên CypherBench bằng MLX (Apple Silicon), tái lập giao thức bài gốc.

Giống bài gốc: prompt NL2CYPHER_PROMPT_DEFAULT làm một tin nhắn user, greedy (temperature 0),
dừng ở "\n```", bỏ fence ```. Ghi từng dòng JSONL để chạy tiếp được nếu bị ngắt.

Chạy:
  .venv-cb/bin/python src/cb_zero_shot.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --graphs flight_accident nba --out results/cypherbench/zero_shot/qwen2.5-coder-7b
"""
import argparse
import json
import platform
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from cb_common import NL2CYPHER_PROMPT_DEFAULT, clean_prediction, load_split, schema_str

STOP = "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["qid"] for l in pred_path.read_text().splitlines() if l.strip()}

    items = load_split(args.split, set(args.graphs))
    if args.limit:
        items = items[: args.limit]
    schemas = {g: schema_str(g) for g in args.graphs}

    model, tok = load(args.model)
    sampler = make_sampler(temp=0.0)
    (out / "run_config.json").write_text(json.dumps({
        "model": args.model, "backend": "mlx_lm", "dtype": "bf16 (weights as published)",
        "split": args.split, "graphs": args.graphs, "n": len(items), "decoding": "greedy",
        "max_tokens": args.max_tokens, "stop": STOP, "prompt": "NL2CYPHER_PROMPT_DEFAULT as single user message",
        "machine": platform.platform(),
    }, indent=1))

    t0 = time.time()
    with pred_path.open("a") as f:
        for i, x in enumerate(items):
            if x.qid in done:
                continue
            prompt = NL2CYPHER_PROMPT_DEFAULT.format(schema=schemas[x.graph], question=x.nl_question)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            raw = generate(model, tok, prompt=text, max_tokens=args.max_tokens, sampler=sampler)
            raw_cut = raw.split(STOP)[0]
            f.write(json.dumps({"qid": x.qid, "graph": x.graph, "raw": raw,
                                "pred_cypher": clean_prediction(raw_cut)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
