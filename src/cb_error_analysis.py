"""Phân loại lỗi của dự đoán CypherBench bằng cách đối chiếu Cypher với schema (không cần Neo4j).

Mỗi câu sai (EX = 0) được gán vào loại ĐẦU TIÊN khớp:
  1. not_executable     : truy vấn không chạy được (cú pháp/runtime), theo metric Exec. chính thức
  2. unknown_element    : label / kiểu quan hệ / thuộc tính không có trong schema
  3. reversed_direction : (a:A)-[:R]->(b:B) sai hướng, nhưng (B)-[:R]->(A) có trong schema
  4. wrong_endpoint     : R có trong schema nhưng không nối cặp kiểu (A, B) theo hướng nào
  5. schema_valid_wrong : hợp lệ theo schema nhưng kết quả sai (ngữ nghĩa, lọc, gộp nhóm, liên kết thực thể…)
Loại 2–4 là "vi phạm schema/ontology", phát hiện tự động được trước khi thực thi.

Bộ phân tích dùng regex trên mẫu MATCH nên là xấp xỉ; độ tin cậy được kiểm bằng cách chạy trên gold
(gold phải ~0 vi phạm). Chạy: .venv-cb/bin/python src/cb_error_analysis.py --pred_dir <dir>
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from cb_protocols import load_schema

NODE = r"\(\s*([A-Za-z_]\w*)?\s*(?::\s*([A-Za-z_]\w*))?[^()]*\)"
REL = r"(<?-)\s*\[\s*([A-Za-z_]\w*)?\s*(?::\s*([A-Za-z_|:\w]+))?[^\]]*\]\s*(->?)"
CHAIN = re.compile(NODE + r"(?=\s*" + REL + r"\s*" + NODE + ")")
STR = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
LABELS = re.compile(r"\(\s*[A-Za-z_]?\w*\s*:\s*([A-Za-z_]\w*)")
RELTYPES = re.compile(r"\[\s*[A-Za-z_]?\w*\s*:\s*([A-Za-z_|:\w]+)")
PROP = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


def check(cypher: str, schema):
    """Trả về danh sách vi phạm [(loại, chi tiết)] của một truy vấn so với schema."""
    q = STR.sub("''", cypher)  # bỏ chuỗi để không bắt nhầm dấu chấm/ngoặc trong tên riêng
    ents = {e.label: e for e in schema.entities}
    rel_props = defaultdict(set)
    triples = set()
    for r in schema.relations:
        triples.add((r.subj_label, r.label, r.obj_label))
        rel_props[r.label] |= set(r.properties)
    rel_names = set(rel_props)
    out = []
    for lab in LABELS.findall(q):
        if lab not in ents:
            out.append(("unknown_element", f"label {lab}"))
    for grp in RELTYPES.findall(q):
        for t in re.split(r"[|:]", grp):
            if t and t not in rel_names:
                out.append(("unknown_element", f"rel {t}"))
    # biến -> label
    var_label = {}
    for m in re.finditer(NODE, q):
        if m.group(1) and m.group(2):
            var_label[m.group(1)] = m.group(2)
    var_rel = {}
    for m in CHAIN.finditer(q):
        a_var, a_lab, left, r_var, r_types, right, b_var, b_lab = m.groups()
        a_lab = a_lab or var_label.get(a_var)
        b_lab = b_lab or var_label.get(b_var)
        types = [t for t in re.split(r"[|:]", r_types or "") if t]
        if r_var and types:
            var_rel[r_var] = types
        if not (a_lab and b_lab and types) or a_lab not in ents or b_lab not in ents:
            continue
        if left == "<-" and right == "-":
            a_lab, b_lab = b_lab, a_lab
        elif not (left == "-" and right == "->"):
            continue  # vô hướng: chỉ kiểm cặp kiểu
        for t in types:
            if t not in rel_names:
                continue
            if (a_lab, t, b_lab) in triples:
                continue
            if (b_lab, t, a_lab) in triples:
                out.append(("reversed_direction", f"{a_lab}-[{t}]->{b_lab}"))
            else:
                out.append(("wrong_endpoint", f"{a_lab}-[{t}]->{b_lab}"))
    for v, p in PROP.findall(q):
        if v in var_label and var_label[v] in ents and p not in ents[var_label[v]].properties:
            out.append(("unknown_element", f"prop {var_label[v]}.{p}"))
        elif v in var_rel and all(p not in rel_props[t] for t in var_rel[v] if t in rel_props):
            out.append(("unknown_element", f"relprop {'/'.join(var_rel[v])}.{p}"))
    return out


ORDER = ["unknown_element", "reversed_direction", "wrong_endpoint"]


def classify(row, schema):
    if row["execution_accuracy"] == 1:
        return "correct", []
    v = check(row["pred_cypher"], schema)
    if row["executable"] == 0:
        return "not_executable", v
    for k in ORDER:
        if any(x[0] == k for x in v):
            return k, v
    return "schema_valid_wrong", v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    args = ap.parse_args()
    d = Path(args.pred_dir)
    rows = [json.loads(l) for f in sorted(glob.glob(str(d / "scored/*.jsonl"))) for l in open(f) if l.strip()]
    schemas = {g: load_schema(g) for g in {r["graph"] for r in rows}}

    gold_viol = [r["qid"] for r in rows if check(r["gold_cypher"], schemas[r["graph"]])]
    cats, by_graph, examples = Counter(), defaultdict(Counter), defaultdict(list)
    any_violation = 0
    for r in rows:
        c, v = classify(r, schemas[r["graph"]])
        cats[c] += 1
        by_graph[r["graph"]][c] += 1
        any_violation += bool(v) and r["execution_accuracy"] == 0
        if c not in ("correct",) and len(examples[c]) < 3:
            examples[c].append({"qid": r["qid"], "question": r["question"], "pred": r["pred_cypher"],
                                "gold": r["gold_cypher"], "violations": v[:3]})
    n = len(rows)
    res = {"n": n, "categories": {k: {"n": c, "pct_of_all": round(100 * c / n, 2)} for k, c in cats.most_common()},
           "wrong_with_any_schema_violation": any_violation,
           "gold_false_positive": {"n": len(gold_viol), "qids": gold_viol[:20]},
           "by_graph": {g: dict(c) for g, c in sorted(by_graph.items())}, "examples": examples}
    (d / "error_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ["n", "categories", "wrong_with_any_schema_violation", "gold_false_positive"]}, indent=1))


if __name__ == "__main__":
    main()
