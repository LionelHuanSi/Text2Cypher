"""So sánh GHÉP CẶP hai lần chạy trên cùng tập câu hỏi (cùng qid).

Báo: chênh EX (B - A) tổng thể, theo đồ thị, theo loại match/return, kèm CI 95% bootstrap ghép cặp
(lấy mẫu lại câu hỏi, 10.000 lần), p-value McNemar (chính xác, hai phía), số câu A đúng/B sai và ngược lại.
Nếu có error_analysis (per-question) thì so thêm tỷ lệ vi phạm schema.

Dùng: .venv-cb/bin/python src/cb_compare.py --a <pred_dir A> --b <pred_dir B> [--out file.json]
"""
import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from cb_error_analysis import ORDER, check, classify
from cb_protocols import load_schema


def load_scored(d):
    rows = {}
    for f in sorted((Path(d) / "scored").glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def paired(xa, xb, rng, iters=10000):
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    d = xb - xa
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    a_only = int(((xa == 1) & (xb == 0)).sum())
    b_only = int(((xa == 0) & (xb == 1)).sum())
    return {"n": len(d), "A": round(100 * xa.mean(), 2), "B": round(100 * xb.mean(), 2),
            "delta": round(100 * d.mean(), 2), "ci95": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "A_only_correct": a_only, "B_only_correct": b_only, "mcnemar_p": float(f"{mcnemar(a_only, b_only):.2e}")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    A, B = load_scored(args.a), load_scored(args.b)
    qids = sorted(set(A) & set(B))
    rng = np.random.default_rng(args.seed)

    def ex(rows):
        return [1.0 if rows[q]["execution_accuracy"] == 1 else 0.0 for q in qids]

    out = {"a": args.a, "b": args.b, "n_common": len(qids), "overall": paired(ex(A), ex(B), rng)}
    for key in ("graph", "match", "return"):
        groups = defaultdict(list)
        for q in qids:
            groups[A[q][key]].append(q)
        out[f"by_{key}"] = {}
        for g, qs in sorted(groups.items()):
            out[f"by_{key}"][g] = paired([A[q]["execution_accuracy"] == 1 for q in qs],
                                         [B[q]["execution_accuracy"] == 1 for q in qs], rng, iters=2000)

    # vi phạm schema/ontology, cùng bộ kiểm như cb_error_analysis:
    #   any_violation : dự đoán có >= 1 vi phạm (bất kể đúng/sai)
    #   violation_class: câu sai được xếp vào unknown_element / reversed_direction / wrong_endpoint
    schemas = {g: load_schema(g) for g in {A[q]["graph"] for q in qids}}
    out["schema_violation"] = {}
    for metric in ("any_violation", "violation_class"):
        vals = {}
        for name, rows in (("A", A), ("B", B)):
            v = []
            for q in qids:
                r, sch = rows[q], schemas[rows[q]["graph"]]
                if metric == "any_violation":
                    v.append(1.0 if check(r["pred_cypher"] or "", sch) else 0.0)
                else:
                    v.append(1.0 if classify(r, sch)[0] in ORDER else 0.0)
            vals[name] = np.asarray(v)
        d = vals["B"] - vals["A"]
        boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
        out["schema_violation"][metric] = {
            "A": round(100 * vals["A"].mean(), 2), "B": round(100 * vals["B"].mean(), 2),
            "delta": round(100 * d.mean(), 2),
            "ci95": [round(100 * float(x), 2) for x in np.percentile(boots, [2.5, 97.5])]}
    cls = {}
    for name, rows in (("A", A), ("B", B)):
        c = defaultdict(int)
        for q in qids:
            c[classify(rows[q], schemas[rows[q]["graph"]])[0]] += 1
        cls[name] = dict(c)
    out["error_classes"] = cls

    o = out["overall"]
    print(f"EX  A={o['A']}  B={o['B']}  delta={o['delta']}  CI95={o['ci95']}  McNemar p={o['mcnemar_p']}"
          f"  (A-only {o['A_only_correct']}, B-only {o['B_only_correct']})")
    for k, s in out["schema_violation"].items():
        print(f"{k:16s} A={s['A']}  B={s['B']}  delta={s['delta']}  CI95={s['ci95']}")
    for k in sorted(set(out["error_classes"]["A"]) | set(out["error_classes"]["B"])):
        print(f"  class {k:20s} A={out['error_classes']['A'].get(k, 0):5d}  B={out['error_classes']['B'].get(k, 0):5d}")
    for g, v in out["by_graph"].items():
        print(f"  {g:22s} {v['A']:6.2f} -> {v['B']:6.2f}  delta {v['delta']:+6.2f}  CI {v['ci95']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
