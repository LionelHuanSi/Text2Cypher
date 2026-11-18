"""Đối chiếu ontology TỰ XÂY (ontology/<graph>.json) với chú thích TAY của tác giả (graph_info.json).

Với mỗi quan hệ (gộp các biến thể cặp kiểu theo nhãn quan hệ) so 6 trường:
  is_symmetric, is_time_sensitive, subj_cardinality, obj_cardinality, is_mandatory_subj, is_mandatory_obj.
Ngưỡng của Builder chỉ được chỉnh trên đồ thị TRAIN; đồ thị TEST báo cáo nguyên trạng.
Ngoài ra báo độ phủ Wikidata: tỷ lệ quan hệ suy ra được PID (support >= 0,5).

Chạy: .venv-cb/bin/python src/ontology/eval_graphinfo.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = {"art", "biology", "soccer", "terrorist_attack"}
FIELDS = [("is_symmetric", "symmetric"), ("is_time_sensitive", "time_sensitive"),
          ("subj_cardinality", "subj_cardinality"), ("obj_cardinality", "obj_cardinality"),
          ("is_mandatory_subj", "mandatory_subj"), ("is_mandatory_obj", "mandatory_obj")]


def aggregate(triples):
    """Gộp các biến thể của một nhãn quan hệ: 'one'/mandatory chỉ khi mọi biến thể đều vậy."""
    inf = [t["inferred"] for t in triples]
    return {
        "symmetric": any(i["symmetric"] for i in inf),
        "time_sensitive": any(i["time_sensitive"] for i in inf),
        "subj_cardinality": "one" if all(i["subj_cardinality"] == "one" for i in inf) else "many",
        "obj_cardinality": "one" if all(i["obj_cardinality"] == "one" for i in inf) else "many",
        "mandatory_subj": all(i["mandatory_subj"] for i in inf),
        "mandatory_obj": all(i["mandatory_obj"] for i in inf),
        "has_pid": any(i["wikidata_pid"] for i in inf),
    }


def main():
    gi = json.loads((ROOT / "data/cypherbench/raw/github/graph_info.json").read_text())
    res = {"train": defaultdict(lambda: [0, 0]), "test": defaultdict(lambda: [0, 0])}
    errors = defaultdict(list)
    per_graph = {}
    for g, v in sorted(gi.items()):
        f = ROOT / f"ontology/{g}.json"
        if not f.exists():
            continue
        onto = json.loads(f.read_text())
        by_label = defaultdict(list)
        for t in onto["relations"]:
            by_label[t["label"]].append(t)
        split = "train" if g in TRAIN else "test"
        ok_g, n_g = 0, 0
        for r in v["relations"]:
            if r["label"] not in by_label:
                errors["missing_relation"].append(f"{g}.{r['label']}")
                continue
            a = aggregate(by_label[r["label"]])
            for gold_k, our_k in FIELDS:
                hit = a[our_k] == r[gold_k]
                res[split][gold_k][0] += hit
                res[split][gold_k][1] += 1
                ok_g += hit
                n_g += 1
                if not hit:
                    errors[gold_k].append(f"{g}.{r['label']}: tay={r[gold_k]} tự xây={a[our_k]}")
            res[split]["wikidata_pid_coverage"][0] += a["has_pid"]
            res[split]["wikidata_pid_coverage"][1] += 1
        per_graph[g] = round(100 * ok_g / n_g, 1) if n_g else None

    out = {s: {k: {"agree_pct": round(100 * a / n, 1), "n": n} for k, (a, n) in d.items()} for s, d in res.items()}
    out["per_graph_agree_pct"] = per_graph
    out["disagreements"] = errors
    (ROOT / "ontology/eval_vs_graph_info.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for s in ("train", "test"):
        print(s, {k: v["agree_pct"] for k, v in out[s].items()}, "n_rel =", out[s].get("is_symmetric", {}).get("n"))
    print("per graph:", per_graph)


if __name__ == "__main__":
    main()
