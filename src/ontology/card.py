"""Chuyển ontology thành THẺ ONTOLOGY (văn bản ngắn) để ghép sau schema trong prompt.

Ba nguồn dùng CÙNG một khuôn thẻ, để so sánh công bằng (chỉ khác nội dung tri thức):
  auto      : ontology tự xây (ontology/<graph>.json): bản số, bắt buộc, thời gian, nghĩa Wikidata, từ vựng
  graphinfo : chú thích tay của tác giả (graph_info.json): chỉ có đối xứng, thời gian, bản số, bắt buộc
  placebo   : khuôn giống 'auto' nhưng nội dung bị xáo giữa các quan hệ (sai lệch có kiểm soát), cùng độ dài

Dùng: from ontology.card import render_card; render_card("nba", source="auto")
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _short(text, n=90):
    if not text:
        return text
    text = text.split(". ")[0].split("; ")[0]
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _facts_auto(graph):
    onto = json.loads((ROOT / f"ontology/{graph}.json").read_text())
    rels = []
    for t in sorted(onto["relations"], key=lambda t: (t["label"], t["subj_label"], t["obj_label"])):
        i = t["inferred"]
        st = t["stats"]
        meaning = None
        if i["wikidata_pid"]:
            meaning = _short(i["meaning"]) or i["reading"]
            if i["wikidata_direction"] == "inverse":
                meaning = f"inverse of '{i['reading']}' ({meaning})"
            elif i["reading"]:
                meaning = f"'{i['reading']}': {meaning}" if i["meaning"] else f"'{i['reading']}'"
        rels.append({
            "s": t["subj_label"], "r": t["label"], "o": t["obj_label"], "meaning": meaning,
            "obj_card": i["obj_cardinality"], "subj_card": i["subj_cardinality"],
            "mand_subj": i["mandatory_subj"], "subj_part": st["subj_participation"],
            "symmetric": i["symmetric"], "interval": i.get("interval_properties", []),
            "event_time": i.get("event_time_properties", []),
        })
    ents = []
    for e in onto["entities"]:
        cls = [c["label"] for c in e.get("wikidata_classes", []) if c.get("label") and c["support"] >= 0.5]
        vocab = {p: v["vocab"] for p, v in e["properties"].items() if v.get("vocab")}
        ents.append({"label": e["label"], "classes": cls[:2], "vocab": vocab})
    return rels, ents


def _facts_graphinfo(graph):
    gi = json.loads((ROOT / "data/cypherbench/raw/github/graph_info.json").read_text())[graph]
    sch = json.loads((ROOT / f"data/cypherbench/raw/graphs/schemas/{graph}_schema.json").read_text())
    rprops = {(r["label"], r["subj_label"], r["obj_label"]): r["properties"] for r in sch["relations"]}
    rels = []
    for r in sorted(gi["relations"], key=lambda r: r["label"]):
        for v in r["variants"]:
            lab, s, o = v.split("#")
            props = rprops.get((lab, s, o), {})
            interval = sorted(k for k in props if k.startswith(("start", "end"))) if r["is_time_sensitive"] else []
            rels.append({"s": s, "r": lab, "o": o, "meaning": None,
                         "obj_card": r["obj_cardinality"], "subj_card": r["subj_cardinality"],
                         "mand_subj": r["is_mandatory_subj"], "subj_part": None,
                         "symmetric": r["is_symmetric"], "interval": interval, "event_time": []})
    return rels, []


def _placebo(rels, ents, seed=0):
    """Giữ nguyên khuôn và độ dài, xáo nội dung giữa các quan hệ/thực thể."""
    rng = random.Random(seed)
    keys = ["meaning", "obj_card", "subj_card", "mand_subj", "subj_part", "symmetric", "interval", "event_time"]
    pool = [{k: r[k] for k in keys} for r in rels]
    rng.shuffle(pool)
    rels = [{**r, **p} for r, p in zip(rels, pool)]
    vpool = [e["vocab"] for e in ents]
    cpool = [e["classes"] for e in ents]
    rng.shuffle(vpool)
    rng.shuffle(cpool)
    ents = [{**e, "vocab": v, "classes": c} for e, v, c in zip(ents, vpool, cpool)]
    return rels, ents


def render_card(graph, source="auto", max_vocab=12):
    if source == "graphinfo":
        rels, ents = _facts_graphinfo(graph)
    else:
        rels, ents = _facts_auto(graph)
        if source == "placebo":
            rels, ents = _placebo(rels, ents)
    lines = ["Ontology notes:"]
    for r in rels:
        parts = [f"(:{r['s']})-[:{r['r']}]->(:{r['o']})"]
        if r["meaning"]:
            parts.append(f"meaning: {r['meaning']}")
        parts.append(f"each {r['s']} has {'one' if r['obj_card'] == 'one' else 'many'} {r['o']}; "
                     f"each {r['o']} has {'one' if r['subj_card'] == 'one' else 'many'} {r['s']}")
        if not r["mand_subj"]:
            pct = f" (only {round(100 * r['subj_part'])}%)" if r["subj_part"] is not None else ""
            parts.append(f"not every {r['s']} has this relation{pct}")
        if r["symmetric"]:
            parts.append("symmetric: match it without direction")
        if r["interval"]:
            parts.append(f"time-dependent, valid from/to: {', '.join(r['interval'])} (missing end = still valid)")
        if r["event_time"]:
            parts.append(f"event time: {', '.join(r['event_time'])}")
        lines.append("- " + "; ".join(parts))
    for e in ents:
        bits = []
        if e["classes"]:
            bits.append("type: " + " / ".join(e["classes"]))
        for p, vals in e["vocab"].items():
            vs = ", ".join(repr(x) for x in vals[:max_vocab]) + (", ..." if len(vals) > max_vocab else "")
            bits.append(f"{p} values: {vs}")
        if bits:
            lines.append(f"- {e['label']}: " + "; ".join(bits))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(render_card(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto"))
