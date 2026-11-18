"""Chấm dự đoán CypherBench bằng metric chính thức (EX, PSJS, Exec.) của bài gốc.

Logic giống references/cypherbench/cypherbench/evaluate.py; khác ở chỗ đọc predictions.jsonl
và thêm khoảng tin cậy bootstrap 95% cho EX.

Chạy: .venv-cb/bin/python src/cb_evaluate.py --pred_dir results/cypherbench/zero_shot/<model> [--graphs nba ...]
Với --graphs: chỉ chấm các đồ thị đó (Neo4j tương ứng phải đang chạy), lưu scored/<graph>.jsonl.
metrics.json luôn được tính lại từ mọi file scored/*.jsonl đã có, nên có thể chấm lần lượt từng đồ thị.
"""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cb_common import (connector, execution_accuracy, executable, load_split,
                       provenance_subgraph_jaccard_similarity)

RETURN_PATTERN_MAPPING = {  # nguyên văn evaluate.py
    "n_name": "n_name", "n_prop": "n_prop_combined", "n_name_prop": "n_prop_combined",
    "n_prop_distinct": "n_prop_combined", "n_prop_array_distinct": "n_prop_combined",
    "n_order_by": "n_order_by", "n_argmax": "n_argmax", "n_where": "n_where",
    "n_agg": "n_agg", "n_group_by": "n_group_by",
}
METRICS = {"execution_accuracy": execution_accuracy, "psjs": provenance_subgraph_jaccard_similarity,
           "executable": executable}


def avg(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--graphs", nargs="+", default=None)
    args = ap.parse_args()
    d = Path(args.pred_dir)
    preds = {json.loads(l)["qid"]: json.loads(l) for l in (d / "predictions.jsonl").read_text().splitlines() if l.strip()}
    graphs = args.graphs or sorted({p["graph"] for p in preds.values()})
    items = [x for x in load_split(args.split, set(graphs)) if x.qid in preds]
    conns = {g: connector(g) for g in graphs}
    sdir = d / "scored"
    sdir.mkdir(exist_ok=True)

    def score(x):
        pred = preds[x.qid]["pred_cypher"]
        if pred.endswith("<end_of_turn>"):
            pred = pred[: -len("<end_of_turn>")].strip()
        m = {k: f(pred_cypher=pred, target_cypher=x.gold_cypher, neo4j_connector=conns[x.graph])
             for k, f in METRICS.items()}
        return {"qid": x.qid, "graph": x.graph, "match": x.from_template.match_category,
                "return": x.from_template.return_pattern_id, "question": x.nl_question,
                "gold_cypher": x.gold_cypher, "pred_cypher": pred, **m}

    with ThreadPoolExecutor(args.threads) as ex:
        new = list(ex.map(score, items))
    for g in graphs:
        (sdir / f"{g}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new if r["graph"] == g))

    rows = [json.loads(l) for f in sorted(sdir.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    graphs = sorted({r["graph"] for r in rows})

    def group(key):
        g = defaultdict(list)
        for r in rows:
            k = key(r)
            if k is not None:
                g[k].append(r["execution_accuracy"])
        return {k: {"n": len(v), "EX": avg(v)} for k, v in sorted(g.items())}

    ex_all = [r["execution_accuracy"] for r in rows]
    agg = {
        "n": len(rows),
        "overall": {m: avg([r[m] for r in rows]) for m in METRICS},
        "EX_ci95": boot_ci(ex_all),
        "by_graph": {g: {"n": len(v), **{m: avg([r[m] for r in v]) for m in METRICS}}
                     for g in graphs for v in [[r for r in rows if r["graph"] == g]]},
        "by_match": group(lambda r: r["match"]),
        "by_return": group(lambda r: RETURN_PATTERN_MAPPING.get(r["return"])),
    }
    (d / "metrics.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
