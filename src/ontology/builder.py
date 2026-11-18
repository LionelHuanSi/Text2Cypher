"""Ontology Builder: xây ontology TỰ ĐỘNG cho một Neo4j bất kỳ.

Đầu vào: kết nối Neo4j (bắt buộc), Wikidata (tuỳ chọn, chỉ dùng khi thực thể mang QID).
Không dùng chú thích tay (graph_info.json) hay câu hỏi của benchmark.

Hai tầng:
  1. Introspection (KG bất kỳ): kiểu, quan hệ + hướng + cặp kiểu, kiểu thuộc tính, bản số và tính bắt buộc
     (thống kê bậc), tính tương hỗ (cạnh hai chiều), thuộc tính thời gian trên cạnh, từ vựng giá trị.
  2. Wikidata (khi có QID): suy ra PID của mỗi quan hệ từ các cạnh mẫu, rồi lấy nhãn, mô tả, bí danh,
     tính đối xứng, ràng buộc single-value, thuộc tính nghịch đảo; lớp Wikidata (P31) của mỗi label.

Chạy: .venv-cb/bin/python src/ontology/builder.py --graph nba        (Neo4j của đồ thị phải đang chạy)
Kết quả: ontology/<graph>.json
"""
import argparse
import collections
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ontology"
CACHE = ROOT / "ontology/cache"
UA = "ontology-builder-research/0.1 (academic research script)"
QID_RE = re.compile(r"\bQ[1-9]\d*\b")
TEMPORAL_NAME = re.compile(r"(^|_)(start|end|from|until|since|begin|year|date|time|period)(_|$)", re.I)
INTERVAL_START = re.compile(r"(^|_)(start|from|since|begin)(_|$)", re.I)
INTERVAL_END = re.compile(r"(^|_)(end|until|to)(_|$)", re.I)
META_KEYS = {"eid", "aliases", "description", "provenance"}  # thuộc tính meta phổ biến; vẫn được ghi nhận nhưng không vào thẻ
VOCAB_MAX = 30          # thuộc tính chuỗi có <= VOCAB_MAX giá trị khác nhau ...
VOCAB_RATIO = 0.5       # ... và mỗi giá trị lặp lại trung bình >= 2 lần thì là danh mục -> liệt kê từ vựng
EDGE_SAMPLE = 12        # số cạnh mẫu/quan hệ để suy PID


# ----------------------------------------------------------------------------- Neo4j
class Graph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def q(self, cypher, **params):
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]


def introspect(g: Graph) -> dict:
    labels = {r["l"]: r["c"] for r in g.q("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c")}
    node_props = collections.defaultdict(dict)
    for r in g.q("CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes "
                 "RETURN nodeLabels, propertyName, propertyTypes"):
        if r["propertyName"]:
            for l in r["nodeLabels"]:
                node_props[l][r["propertyName"]] = r["propertyTypes"]
    rel_props = collections.defaultdict(dict)
    for r in g.q("CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes "
                 "RETURN relType, propertyName, propertyTypes"):
        if r["propertyName"]:
            rel_props[r["relType"].strip(":`")][r["propertyName"]] = r["propertyTypes"]

    triples = g.q("MATCH (a)-[r]->(b) RETURN labels(a)[0] AS s, type(r) AS rel, labels(b)[0] AS o, count(*) AS n")
    rels = []
    for t in triples:
        s, rel, o = t["s"], t["rel"], t["o"]
        out = g.q(f"MATCH (a:`{s}`)-[:`{rel}`]->(b:`{o}`) WITH a, count(DISTINCT b) AS c "
                  "RETURN count(a) AS n, avg(c) AS avg, max(c) AS mx, "
                  "sum(CASE WHEN c = 1 THEN 1 ELSE 0 END) AS one")[0]
        inn = g.q(f"MATCH (a:`{s}`)-[:`{rel}`]->(b:`{o}`) WITH b, count(DISTINCT a) AS c "
                  "RETURN count(b) AS n, avg(c) AS avg, max(c) AS mx, "
                  "sum(CASE WHEN c = 1 THEN 1 ELSE 0 END) AS one")[0]
        recip = None
        if s == o:
            recip = g.q(f"MATCH (a:`{s}`)-[:`{rel}`]->(b:`{o}`) WHERE a <> b "
                        f"RETURN count(*) AS n, sum(CASE WHEN EXISTS {{ (b)-[:`{rel}`]->(a) }} THEN 1 ELSE 0 END) AS back")[0]
        props = rel_props.get(rel, {})
        temporal = {k: v for k, v in props.items()
                    if TEMPORAL_NAME.search(k) or any("Date" in x for x in v)}
        rels.append({
            "label": rel, "subj_label": s, "obj_label": o, "n_edges": t["n"],
            "properties": props, "temporal_properties": temporal,
            "stats": {
                "subj_participation": round(out["n"] / labels[s], 4) if labels.get(s) else None,
                "obj_participation": round(inn["n"] / labels[o], 4) if labels.get(o) else None,
                "objs_per_subj": {"avg": round(out["avg"] or 0, 3), "max": out["mx"],
                                  "frac_one": round(out["one"] / out["n"], 4) if out["n"] else None},
                "subjs_per_obj": {"avg": round(inn["avg"] or 0, 3), "max": inn["mx"],
                                  "frac_one": round(inn["one"] / inn["n"], 4) if inn["n"] else None},
                "reciprocity": round(recip["back"] / recip["n"], 4) if recip and recip["n"] else None,
            },
        })

    entities = []
    for l, cnt in sorted(labels.items()):
        props = {}
        for p, types in sorted(node_props.get(l, {}).items()):
            info = {"types": types}
            if p not in META_KEYS and p != "name" and any(t in ("String", "StringArray") for t in types):
                is_list = "StringArray" in types
                expr = f"UNWIND n.`{p}` AS v" if is_list else f"WITH n.`{p}` AS v"
                d = g.q(f"MATCH (n:`{l}`) WHERE n.`{p}` IS NOT NULL {expr} "
                        "RETURN count(DISTINCT v) AS k, count(*) AS n")[0]
                info["n_distinct"] = d["k"]
                if d["k"] <= VOCAB_MAX and d["k"] <= VOCAB_RATIO * d["n"]:
                    info["vocab"] = [r["v"] for r in g.q(
                        f"MATCH (n:`{l}`) WHERE n.`{p}` IS NOT NULL {expr} "
                        "RETURN v, count(*) AS c ORDER BY c DESC")]
            props[p] = info
        entities.append({"label": l, "count": cnt, "properties": props})
    return {"entities": entities, "relations": rels}


def find_qid_key(g: Graph, entities) -> str | None:
    """Tìm thuộc tính chứa QID Wikidata (vd. eid = 'Team#Q121783'), không giả định tên trường."""
    l = entities[0]["label"]
    row = g.q(f"MATCH (n:`{l}`) RETURN properties(n) AS p LIMIT 1")[0]["p"]
    for k, v in row.items():
        if isinstance(v, str) and QID_RE.search(v):
            return k
    return None


# ----------------------------------------------------------------------------- Wikidata
class Wikidata:
    """Truy cập Wikidata qua Action API (wbgetentities), cache gọn trên đĩa.

    Không dùng SPARQL (WDQS hay bị giới hạn tốc độ). Mỗi thực thể lưu: nhãn, mô tả, bí danh tiếng Anh
    và claims dạng {PID: [QID/PID đích]}.
    """

    def __init__(self, cache_file: Path):
        self.f = cache_file
        self.c = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        self.dirty = 0

    def _fetch(self, ids):
        """Lấy một lô thực thể; lỗi thì thử lại, rồi chia đôi lô (thực thể lớn như quốc gia dễ quá thời gian)."""
        url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(ids), "props": "labels|descriptions|aliases|claims",
            "languages": "en", "format": "json"})
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                data = json.loads(urllib.request.urlopen(req, timeout=180).read())
                if "entities" in data:
                    return data["entities"]
            except Exception:
                pass
            time.sleep(3 * (attempt + 1))
        if len(ids) > 1:
            h = len(ids) // 2
            return {**self._fetch(ids[:h]), **self._fetch(ids[h:])}
        print(f"  wikidata: bỏ qua {ids[0]} (không lấy được)", flush=True)
        return {ids[0]: {}}

    def entities(self, ids):
        ids = sorted(set(ids))
        todo = [i for i in ids if i not in self.c]
        for k in range(0, len(todo), 50):
            for i, ent in self._fetch(todo[k:k + 50]).items():
                claims = {}
                for pid, cs in ent.get("claims", {}).items():
                    vals = []
                    for c in cs:
                        dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                        if isinstance(dv, dict) and "id" in dv:
                            vals.append(dv["id"])
                    if vals:
                        claims[pid] = vals
                self.c[i] = {"label": ent.get("labels", {}).get("en", {}).get("value"),
                             "description": ent.get("descriptions", {}).get("en", {}).get("value"),
                             "aliases": [a["value"] for a in ent.get("aliases", {}).get("en", [])][:8],
                             "claims": claims}
            time.sleep(0.2)
        if todo:
            self.f.parent.mkdir(parents=True, exist_ok=True)
            self.f.write_text(json.dumps(self.c))
        return {i: self.c.get(i, {}) for i in ids}


def _claim_ids(ent, pid):
    return ent.get("claims", {}).get(pid, [])


SYMMETRIC_PROPERTY = "Q18647518"     # instance of: symmetric property
TRANSITIVE_PROPERTY = "Q18647515"    # instance of: transitive property
C_SINGLE_VALUE = "Q19474404"         # property constraint: single-value
C_SINGLE_BEST = "Q52060874"          # property constraint: single-best-value
C_SYMMETRIC = "Q21510862"            # property constraint: symmetric
C_INVERSE = "Q21510855"              # property constraint: inverse


def enrich_wikidata(g: Graph, onto: dict, qid_key: str, wd: Wikidata):
    def qid(v):
        m = QID_RE.search(v or "")
        return m.group(0) if m else None

    pids_needed = set()
    for r in onto["relations"]:
        rows = g.q(f"MATCH (a:`{r['subj_label']}`)-[:`{r['label']}`]->(b:`{r['obj_label']}`) "
                   f"RETURN a.`{qid_key}` AS s, b.`{qid_key}` AS o LIMIT {EDGE_SAMPLE}")
        pairs = [(qid(x["s"]), qid(x["o"])) for x in rows if qid(x["s"]) and qid(x["o"])]
        if not pairs:
            continue
        ents = wd.entities([q for pr in pairs for q in pr])
        fwd, bwd = collections.Counter(), collections.Counter()
        for sq, oq in pairs:
            for pid, vals in ents.get(sq, {}).get("claims", {}).items():
                if oq in vals:
                    fwd[pid] += 1
            for pid, vals in ents.get(oq, {}).get("claims", {}).items():
                if sq in vals:
                    bwd[pid] += 1
        cand = [(c / len(pairs), p, "forward") for p, c in fwd.items()] + \
               [(c / len(pairs), p, "inverse") for p, c in bwd.items()]
        cand.sort(reverse=True)
        r["wikidata"] = {"n_sample_edges": len(pairs),
                         "candidates": [{"pid": p, "direction": d, "support": round(s, 3)} for s, p, d in cand[:5]]}
        if cand:
            pids_needed.update(p for _, p, _ in cand[:3])

    # lớp Wikidata của mỗi label (P31 phổ biến nhất trên thực thể mẫu)
    for e in onto["entities"]:
        rows = g.q(f"MATCH (n:`{e['label']}`) RETURN n.`{qid_key}` AS q LIMIT 20")
        qids = [qid(x["q"]) for x in rows if qid(x["q"])]
        if not qids:
            continue
        ents = wd.entities(qids)
        cls = collections.Counter(c for q in qids for c in set(_claim_ids(ents.get(q, {}), "P31")))
        e["wikidata_classes"] = [{"qid": c, "support": round(n / len(qids), 3)} for c, n in cls.most_common(3)]
        pids_needed.update(c for c, _ in cls.most_common(3))

    info = wd.entities(sorted(pids_needed))
    def describe(i):
        ent = info.get(i, {})
        d = {"id": i, "label": ent.get("label"), "description": ent.get("description"),
             "aliases": ent.get("aliases", [])}
        if i.startswith("P"):
            constraints = _claim_ids(ent, "P2302")
            d["symmetric"] = SYMMETRIC_PROPERTY in _claim_ids(ent, "P31") or C_SYMMETRIC in constraints
            d["single_value"] = C_SINGLE_VALUE in constraints or C_SINGLE_BEST in constraints
            d["transitive"] = TRANSITIVE_PROPERTY in _claim_ids(ent, "P31")
            d["inverse_property"] = (_claim_ids(ent, "P1696") or [None])[0]
        return d
    for r in onto["relations"]:
        for c in r.get("wikidata", {}).get("candidates", []):
            c.update({k: v for k, v in describe(c["pid"]).items() if k != "id"})
    for e in onto["entities"]:
        for c in e.get("wikidata_classes", []):
            c.update({k: v for k, v in describe(c["qid"]).items() if k != "id"})


# ----------------------------------------------------------------------------- ontology suy luận
TAU_ONE = 0.95      # >= 95% đỉnh có đúng 1 láng giềng -> bản số "one"
TAU_MAND = 0.95     # >= 95% đỉnh của kiểu tham gia -> "mandatory"
TAU_RECIP = 0.5     # >= 50% cạnh có cạnh ngược -> đối xứng (theo dữ liệu)
TAU_PID = 0.5       # PID phải giải thích >= 50% cạnh mẫu


STOP = {"has", "is", "of", "by", "the", "in", "to", "with", "a", "an", "on", "for", "from", "at"}


def _tokens(text):
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "").replace("_", " ").lower()
    return {w[:5] for w in re.findall(r"[a-z]+", words) if w not in STOP}


def pick_pid(rel_label, candidates):
    """Chọn PID: phủ >= TAU_PID cạnh mẫu; xếp theo độ phủ + độ khớp tên (nhãn/bí danh PID với tên quan hệ),
    ưu tiên chiều thuận khi hoà. Tên quan hệ là một phần của KG nên vẫn hoàn toàn tự động."""
    rt = _tokens(rel_label)
    best, best_score = None, -1
    for c in candidates:
        if c["support"] < TAU_PID:
            continue
        names = [c.get("label") or ""] + (c.get("aliases") or [])
        lex = max((len(rt & _tokens(n)) / len(rt) for n in names if rt), default=0)
        score = c["support"] + 0.5 * lex + (0.01 if c["direction"] == "forward" else 0)
        if score > best_score:
            best, best_score = c, score
    return best


def infer(onto: dict) -> dict:
    """Từ thống kê + Wikidata suy ra các khẳng định ontology cho từng quan hệ."""
    for r in onto["relations"]:
        st = r["stats"]
        wd = (r.get("wikidata") or {}).get("candidates") or []
        top = pick_pid(r["label"], wd)
        sym_data = st["reciprocity"] is not None and st["reciprocity"] >= TAU_RECIP
        sym_wd = bool(top and top.get("symmetric")) and r["subj_label"] == r["obj_label"]
        if sym_wd and top["direction"] == "inverse":  # quan hệ đối xứng: chiều không quan trọng
            top = {**top, "direction": "forward"}
        obj_one = (st["objs_per_subj"]["frac_one"] or 0) >= TAU_ONE
        subj_one = (st["subjs_per_obj"]["frac_one"] or 0) >= TAU_ONE
        has_interval = any(INTERVAL_START.search(k) or INTERVAL_END.search(k) for k in r["temporal_properties"])
        # ràng buộc Wikidata "một giá trị" chỉ đúng tại một thời điểm -> không áp cho quan hệ có khoảng hiệu lực
        if top and top.get("single_value") and not has_interval:
            if top["direction"] == "forward":
                obj_one = True
            else:
                subj_one = True
        r["inferred"] = {
            "symmetric": sym_data or sym_wd,
            "symmetric_evidence": [x for x, ok in [("data_reciprocity", sym_data), ("wikidata", sym_wd)] if ok],
            # khoảng hiệu lực (start/end) -> quan hệ phụ thuộc thời gian; 'year' đơn lẻ -> thời điểm sự kiện
            "time_sensitive": any(INTERVAL_START.search(k) for k in r["temporal_properties"])
                              or any(INTERVAL_END.search(k) for k in r["temporal_properties"]),
            "interval_properties": sorted(k for k in r["temporal_properties"]
                                          if INTERVAL_START.search(k) or INTERVAL_END.search(k)),
            "event_time_properties": sorted(k for k in r["temporal_properties"]
                                            if not (INTERVAL_START.search(k) or INTERVAL_END.search(k))),
            "obj_cardinality": "one" if obj_one else "many",
            "subj_cardinality": "one" if subj_one else "many",
            "mandatory_subj": (st["subj_participation"] or 0) >= TAU_MAND,
            "mandatory_obj": (st["obj_participation"] or 0) >= TAU_MAND,
            "transitive": bool(top and top.get("transitive")) and r["subj_label"] == r["obj_label"],
            "wikidata_pid": top["pid"] if top else None,
            "wikidata_direction": top["direction"] if top else None,
            "meaning": (top or {}).get("description"),
            "reading": (top or {}).get("label"),
            "aliases": (top or {}).get("aliases", []),
        }
    return onto


def build(graph: str, uri: str, user: str, password: str, use_wikidata=True) -> dict:
    g = Graph(uri, user, password)
    t0 = time.time()
    onto = {"graph": graph, "source": uri, "builder_version": "0.1"}
    onto.update(introspect(g))
    qk = find_qid_key(g, onto["entities"])
    onto["qid_key"] = qk
    if use_wikidata and qk:
        enrich_wikidata(g, onto, qk, Wikidata(CACHE / "wikidata.json"))
    infer(onto)
    onto["build_seconds"] = round(time.time() - t0, 1)
    return onto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--no_wikidata", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    info = json.loads((ROOT / "references/cypherbench/neo4j_info.json").read_text())["full"][args.graph]
    onto = build(args.graph, f"bolt://{info['host']}:{info['port']}", info["username"], info["password"],
                 use_wikidata=not args.no_wikidata)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{args.graph}.json").write_text(json.dumps(onto, indent=1, ensure_ascii=False, default=str))
    print(f"{args.graph}: {len(onto['entities'])} labels, {len(onto['relations'])} triples, {onto['build_seconds']}s")


if __name__ == "__main__":
    main()
