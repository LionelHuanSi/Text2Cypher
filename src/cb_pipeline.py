"""Pipeline Teacher + kiểm/sửa (T4–T5) trên dự đoán có sẵn của Teacher.

Các bước cho mỗi câu:
  1. Sửa tự động theo mức tri thức --level (schema | ontology), --placebo để xáo ontology.
  2. (nếu --repair != none) Phát hiện vấn đề còn lại + lỗi thực thi (EXPLAIN trên Neo4j),
     cho Teacher sinh lại MỘT lần với thông báo:
        specific : thông báo cụ thể sinh ra từ schema/ontology (+ lỗi Neo4j nguyên văn)
        generic  : cùng tập câu bị đánh dấu, nhưng thông báo chung chung (đối chứng "chỉ hỏi lại")
  3. Sửa tự động lần nữa trên câu mới.
Ghi predictions.jsonl (chấm bằng cb_evaluate.py) + trace.jsonl (vấn đề, sửa, thông báo).

Ví dụ (tập dev = đồ thị train):
  .venv-cb/bin/python src/cb_pipeline.py --pred_dir results/cypherbench/dev/qwen7b_instruct_A --split train \
      --graphs soccer --level ontology --repair specific --out results/cypherbench/dev/pipe_onto_specific
"""
import argparse
import json
import time
from pathlib import Path

from cb_common import NL2CYPHER_PROMPT_DEFAULT, clean_prediction, connector, load_split, schema_str
from ontology.verifier import Ontology, verify

REPAIR = ("{base}\n\nYour previous query:\n{pred}\n\nProblems found by the checker:\n{msg}\n\n"
          "Rewrite the Cypher query to fix the problems. Output only the query in a single line.")
GENERIC = "The query may be incorrect. Please check it carefully against the schema and the question."


def exec_error(conn, q):
    if not q.strip():
        return "Empty query."
    try:
        conn.run_query("EXPLAIN " + q, timeout=20)
        return None
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0]
        return msg[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--level", choices=["none", "schema", "ontology"], required=True)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--repair", choices=["none", "specific", "generic"], default="none")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = {json.loads(l)["qid"]: json.loads(l) for l in open(Path(args.pred_dir) / "predictions.jsonl") if l.strip()}
    items = [x for x in load_split(args.split, set(args.graphs)) if x.qid in preds]
    pp, tp = out / "predictions.jsonl", out / "trace.jsonl"
    done = {json.loads(l)["qid"] for l in open(pp)} if pp.exists() else set()
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=1))

    model = tok = sampler = None
    if args.repair != "none":
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        from cb_protocols import _gen
        model, tok = load(args.model)
        sampler = make_sampler(temp=0.0)

    t0 = time.time()
    with pp.open("a") as fp, tp.open("a") as ft:
        for g in args.graphs:
            onto = Ontology(g, level=args.level if args.level != "none" else "schema", placebo=args.placebo)
            conn = connector(g) if args.repair != "none" else None
            sch = schema_str(g)
            gi = [x for x in items if x.graph == g and x.qid not in done]
            for i, x in enumerate(gi):
                q0 = preds[x.qid]["pred_cypher"]
                tr = {"qid": x.qid, "graph": g}
                if args.level == "none":
                    q1, applied, pending = q0, [], []
                else:
                    q1, applied, pending = verify(q0, x.nl_question, onto)
                tr["fix1"] = applied
                q = q1
                if args.repair != "none":
                    msgs = [p.message for p in pending]
                    err = exec_error(conn, q1)
                    if err:
                        msgs.append(f"Neo4j error: {err}")
                    tr["flags"] = [p.code for p in pending] + (["E_EXEC"] if err else [])
                    if msgs:
                        msg = "\n".join(f"- {m}" for m in msgs) if args.repair == "specific" else GENERIC
                        base = NL2CYPHER_PROMPT_DEFAULT.format(schema=sch, question=x.nl_question)
                        p = REPAIR.format(base=base, pred=q1, msg=msg)
                        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                                       add_generation_prompt=True, tokenize=False)
                        raw, _ = _gen(model, tok, text, 512, sampler, ["\n```"])
                        q2 = clean_prediction(raw.split("\n```")[0]) or q1
                        tr["message"] = msg
                        tr["repaired"] = q2
                        q, applied2, _ = verify(q2, x.nl_question, onto) if args.level != "none" else (q2, [], [])
                        tr["fix2"] = applied2
                fp.write(json.dumps({"qid": x.qid, "graph": g, "pred_cypher": q}, ensure_ascii=False) + "\n")
                ft.write(json.dumps(tr, ensure_ascii=False) + "\n")
                fp.flush()
                ft.flush()
                if i % 25 == 0:
                    print(f"{g} {i + 1}/{len(gi)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
