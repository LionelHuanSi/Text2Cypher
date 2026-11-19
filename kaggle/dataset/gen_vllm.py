"""Sinh Cypher cho các nhánh RQ1 trên Kaggle (2x T4), greedy, cùng giao thức với bản MLX trên máy.

- Đọc prompts.jsonl (dựng sẵn trên máy bằng src/kaggle_build_prompts.py).
- Áp chat template của model -> sinh greedy, max_tokens 512, dừng ở "\n```" -> bỏ fence (clean_prediction gốc).
- Ghi /kaggle/working/out/<model_slug>/<arm>/predictions.jsonl + run_config.json.
- Backend: vLLM (fp16, tensor_parallel=2). Nếu vLLM lỗi -> transformers batch greedy (chậm hơn, cùng kết quả về giao thức).

Dùng: python gen_vllm.py --prompts <dir>/prompts.jsonl --model Qwen/Qwen2.5-Coder-7B-Instruct [--arms A B C D]
"""
import argparse
import json
import os
import time
from pathlib import Path

STOP = "\n```"
MAX_TOKENS = 512


def clean_prediction(text):
    return text.replace("```cypher", "").replace("```", "").strip()


def gen_vllm(model_id, texts, tp):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, dtype="float16", tensor_parallel_size=tp, max_model_len=4096,
              gpu_memory_utilization=0.90, enforce_eager=False, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop=[STOP], include_stop_str_in_output=True)
    outs = llm.generate(texts, sp)
    return [o.outputs[0].text for o in outs], "vllm fp16 tp=%d" % tp


def gen_hf(model_id, texts, tok, bs=8):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res = [None] * len(texts)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            t = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            res[i] = t[: t.index(STOP) + len(STOP)] if STOP in t else t
        print(f"hf {b + len(idx)}/{len(texts)}", flush=True)
    return res, "transformers fp16 device_map=auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--out", default="/kaggle/working/out")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    rows = [r for r in rows if r["arm"] in args.arms]
    texts = [tok.apply_chat_template([{"role": "user", "content": r["user"]}], add_generation_prompt=True, tokenize=False)
             for r in rows]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    print(f"{len(rows)} prompts, prompt tokens max={max(lens)} mean={sum(lens) / len(lens):.0f}", flush=True)

    import torch
    tp = max(1, torch.cuda.device_count())
    t0 = time.time()
    if args.backend in ("auto", "vllm"):
        try:
            raws, backend = gen_vllm(args.model, texts, tp)
        except Exception as e:  # noqa: BLE001
            if args.backend == "vllm":
                raise
            print("vLLM lỗi, chuyển sang transformers:", repr(e)[:500], flush=True)
            raws, backend = gen_hf(args.model, texts, tok)
    else:
        raws, backend = gen_hf(args.model, texts, tok)
    dt = time.time() - t0
    print(f"generated {len(raws)} in {dt:.0f}s with {backend}", flush=True)

    slug = args.model.split("/")[-1].lower()
    for arm in args.arms:
        d = Path(args.out) / slug / arm
        d.mkdir(parents=True, exist_ok=True)
        with (d / "predictions.jsonl").open("w") as f:
            for r, raw in zip(rows, raws):
                if r["arm"] != arm:
                    continue
                f.write(json.dumps({"qid": r["qid"], "graph": r["graph"], "raw": raw,
                                    "pred_cypher": clean_prediction(raw.split(STOP)[0])}, ensure_ascii=False) + "\n")
        (d / "run_config.json").write_text(json.dumps({
            "model": args.model, "arm": arm, "backend": backend, "decoding": "greedy", "max_tokens": MAX_TOKENS,
            "stop": STOP, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "seconds_all_arms": round(dt), "n_all_arms": len(rows)}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
