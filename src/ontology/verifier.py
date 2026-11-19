"""Bộ KIỂM / SỬA Cypher theo schema và ontology (tầng T4–T5 của pipeline Teacher + ontology).

Hai tầng tri thức, tách riêng để so sánh công bằng:
  schema   : chỉ dùng nhãn, kiểu quan hệ, cặp kiểu hai đầu + hướng (thông tin có trong schema gốc).
  ontology : schema + tri thức ontology tự xây: khoảng thời gian, tính bắt buộc (tỷ lệ tham gia thật),
             nghĩa Wikidata của quan hệ.

Mỗi vấn đề (Issue) hoặc SỬA TỰ ĐỘNG được (fix), hoặc sinh THÔNG BÁO LỖI để Teacher sinh lại (message).

  Mã               Tầng      Phát hiện                                                      Xử lý
  S_REL_TYPE       schema    kiểu quan hệ không có trong schema, giữa hai nhãn đã biết      thay bằng kiểu hợp lệ gần nhất
  S_DIRECTION      schema    (A)-[R]->(B) sai hướng, schema chỉ có (B)-[R]->(A)             lật hướng
  S_UNKNOWN        schema    nhãn / kiểu quan hệ / thuộc tính không tồn tại                 thông báo
  S_ENDPOINT       schema    R có trong schema nhưng không nối (A, B)                        thông báo
  O_TIME_EQ        ontology  quan hệ có khoảng thời gian bị so BẰNG trên start/end           viết lại thành lọc khoảng
  O_TIME_OPEN      ontology  lọc end >= Y mà quên "end rỗng = còn hiệu lực"                  thêm OR end IS NULL
  O_TIME_MISSING   ontology  câu hỏi có năm, dùng quan hệ có khoảng thời gian, không lọc     thêm lọc khoảng (khi rõ ràng)
  O_OPTIONAL       ontology  đếm/gom qua quan hệ KHÔNG bắt buộc trong MATCH bắt buộc         thông báo
  O_SEMANTIC_DIR   ontology  quan hệ nối cùng kiểu (A)-[R]->(A): hướng theo nghĩa Wikidata   thông báo

Placebo (đối chứng): giữ nguyên schema, xáo các dữ kiện ontology giữa các quan hệ cùng đồ thị.

Dùng: from ontology.verifier import Ontology, verify
      onto = Ontology("company", level="ontology", placebo=False)
      fixed_query, issues = verify(query, question, onto)
"""
import difflib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- ontology view


@dataclass
class RelInfo:
    subj: str
    label: str
    obj: str
    props: set
    interval: list = field(default_factory=list)
    event_time: list = field(default_factory=list)
    mandatory_subj: bool = True
    mandatory_obj: bool = True
    subj_part: float = 1.0
    obj_part: float = 1.0
    reading: str = ""
    meaning: str = ""
    wd_direction: str = ""


ONTO_FIELDS = ["interval", "event_time", "mandatory_subj", "mandatory_obj", "subj_part", "obj_part",
               "reading", "meaning", "wd_direction"]


class Ontology:
    def __init__(self, graph, level="ontology", placebo=False, seed=0):
        assert level in ("schema", "ontology")
        self.graph, self.level, self.placebo = graph, level, placebo
        o = json.loads((ROOT / f"ontology/{graph}.json").read_text())
        self.ent_props = {e["label"]: set(e["properties"]) | {"name"} for e in o["entities"]}
        self.rels = {}
        for t in o["relations"]:
            i, st = t["inferred"], t["stats"]
            self.rels[(t["subj_label"], t["label"], t["obj_label"])] = RelInfo(
                t["subj_label"], t["label"], t["obj_label"], set(t["properties"]),
                interval=sorted(i.get("interval_properties", [])), event_time=i.get("event_time_properties", []),
                mandatory_subj=i["mandatory_subj"], mandatory_obj=i["mandatory_obj"],
                subj_part=st["subj_participation"], obj_part=st.get("obj_participation", 1.0),
                reading=i.get("reading") or "", meaning=i.get("meaning") or "", wd_direction=i.get("wikidata_direction") or "")
        if placebo:
            rng = random.Random(f"{seed}-{graph}")
            pool = [{k: getattr(r, k) for k in ONTO_FIELDS} for r in self.rels.values()]
            rng.shuffle(pool)
            for r, p in zip(self.rels.values(), pool):
                for k, v in p.items():
                    setattr(r, k, v)
        self.rel_names = {r.label for r in self.rels.values()}
        self.rel_props = {}
        for r in self.rels.values():
            self.rel_props.setdefault(r.label, set()).update(r.props)

    def get(self, s, r, o):
        return self.rels.get((s, r, o))

    def between(self, a, b):
        """Các kiểu quan hệ nối a và b (bất kể hướng)."""
        return sorted({r.label for r in self.rels.values() if {r.subj, r.obj} == {a, b} or (a == b == r.subj == r.obj)})


# ---------------------------------------------------------------- Cypher pattern parsing

STR = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
NODE_RE = r"\(\s*(?P<{p}v>[A-Za-z_]\w*)?\s*(?::\s*(?P<{p}l>[A-Za-z_]\w*))?(?:\s*:\s*[A-Za-z_]\w*)*\s*(?P<{p}m>\{{[^{{}}]*\}})?\s*\)"
REL_RE = (r"(?P<left><-|-)\s*\[\s*(?P<rv>[A-Za-z_]\w*)?\s*(?::\s*(?P<rt>[A-Za-z_]\w*(?:\s*\|\s*:?\s*[A-Za-z_]\w*)*))?"
          r"\s*(?P<star>\*[\d.]*)?\s*(?P<rm>\{[^{}]*\})?\s*\]\s*(?P<right>->|-)")
HOP = re.compile("(?=(?P<a>" + NODE_RE.format(p="a") + r")\s*(?P<rel>" + REL_RE + r")\s*(?P<b>" + NODE_RE.format(p="b") + "))")
NODE = re.compile(NODE_RE.format(p="n"))
CLAUSE = re.compile(r"\b(OPTIONAL\s+MATCH|MATCH|WHERE|WITH|RETURN|UNWIND|CALL|UNION|ORDER\s+BY|SET|CREATE|MERGE)\b", re.I)
AGG = re.compile(r"\b(count|collect|sum|avg|min|max)\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_]\w*)", re.I)
YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")


def mask_strings(q):
    """Thay nội dung chuỗi bằng 'x' (giữ nguyên độ dài) để regex không bắt nhầm trong tên riêng."""
    return STR.sub(lambda m: m.group(0)[0] + "x" * (len(m.group(0)) - 2) + m.group(0)[-1], q)


@dataclass
class Hop:
    a_var: str
    a_lab: str
    a_props: str
    b_var: str
    b_lab: str
    b_props: str
    rel_var: str
    rel_types: list
    left: str
    right: str
    rel_props: str
    clause: str          # MATCH | OPTIONAL MATCH | WHERE | ... (mệnh đề chứa hop)
    rel_span: tuple      # vị trí của cả quan hệ (gồm mũi tên) trong câu
    bracket_span: tuple  # vị trí của [ ... ]
    star: str

    @property
    def directed(self):
        return (self.left, self.right) in (("-", "->"), ("<-", "-"))

    def oriented(self):
        """(chủ thể, đối tượng) theo hướng đã viết; None nếu vô hướng."""
        if (self.left, self.right) == ("-", "->"):
            return self.a_lab, self.b_lab
        if (self.left, self.right) == ("<-", "-"):
            return self.b_lab, self.a_lab
        return None


def parse(q):
    m_q = mask_strings(q)
    var_label = {}
    for m in NODE.finditer(m_q):
        if m.group("nv") and m.group("nl"):
            var_label.setdefault(m.group("nv"), m.group("nl"))
    clauses = [(m.start(), re.sub(r"\s+", " ", m.group(1).upper())) for m in CLAUSE.finditer(m_q)]

    def clause_at(pos):
        c = "?"
        for p, name in clauses:
            if p <= pos:
                c = name
        return c

    hops = []
    for m in HOP.finditer(m_q):
        g = m.groupdict()
        types = [t.strip(" :") for t in re.split(r"\|", g["rt"])] if g["rt"] else []
        bs = m.start("rel") + m_q[m.start("rel"):m.end("rel")].index("[")
        be = m.start("rel") + m_q[m.start("rel"):m.end("rel")].rindex("]") + 1
        hops.append(Hop(
            a_var=g["av"], a_lab=g["al"] or var_label.get(g["av"]), a_props=g["am"] or "",
            b_var=g["bv"], b_lab=g["bl"] or var_label.get(g["bv"]), b_props=g["bm"] or "",
            rel_var=g["rv"], rel_types=types, left=g["left"], right=g["right"], rel_props=g["rm"] or "",
            clause=clause_at(m.start()), rel_span=(m.start("rel"), m.end("rel")), bracket_span=(bs, be),
            star=g["star"] or ""))
    return hops, var_label, m_q


# ---------------------------------------------------------------- issues


@dataclass
class Issue:
    code: str
    layer: str           # schema | ontology
    detail: str
    message: str = ""    # thông báo cho Teacher (nếu không sửa tự động được)
    fix: object = None   # hàm q -> q' (nếu sửa tự động được)


def _replace(q, span, new):
    return q[:span[0]] + new + q[span[1]:]


def _insert_where(q, after_pos, cond):
    """Thêm điều kiện vào mệnh đề WHERE ngay sau vị trí after_pos (tạo WHERE nếu chưa có)."""
    m_q = mask_strings(q)
    nxt = CLAUSE.search(m_q, after_pos)
    if nxt and nxt.group(1).upper() == "WHERE":
        return q[: nxt.end()] + f" {cond} AND" + q[nxt.end():]
    i = nxt.start() if nxt else len(q)
    return q[:i].rstrip() + f" WHERE {cond} " + q[i:]


def _ensure_rel_var(q, h, name):
    """Đặt tên biến cho quan hệ nếu chưa có; trả (q mới, tên biến, vị trí cuối [ ])."""
    if h.rel_var:
        return q, h.rel_var, h.bracket_span[1]
    s, e = h.bracket_span
    br = q[s:e]
    new = "[" + name + br[1:].lstrip()
    q = _replace(q, (s, e), new)
    return q, name, s + len(new)


def schema_issues(q, onto):
    hops, var_label, m_q = parse(q)
    out = []
    for lab in set(re.findall(r"\(\s*[A-Za-z_]?\w*\s*:\s*([A-Za-z_]\w*)", m_q)):
        if lab not in onto.ent_props:
            out.append(Issue("S_UNKNOWN", "schema", f"label {lab}",
                             f"Node label :{lab} does not exist in the schema."))
    for h in hops:
        for t in h.rel_types:
            if t in onto.rel_names:
                continue
            cands = onto.between(h.a_lab, h.b_lab) if h.a_lab in onto.ent_props and h.b_lab in onto.ent_props else []
            if cands:
                best = max(cands, key=lambda c: difflib.SequenceMatcher(None, c.lower(), t.lower()).ratio())
                s, e = h.bracket_span

                def fx(qq, s=s, e=e, t=t, best=best):
                    return qq[:s] + re.sub(r"(:\s*)" + re.escape(t) + r"\b", r"\g<1>" + best, qq[s:e], count=1) + qq[e:]
                out.append(Issue("S_REL_TYPE", "schema", f"{t} -> {best}", fix=fx))
            else:
                out.append(Issue("S_UNKNOWN", "schema", f"rel {t}",
                                 f"Relationship type :{t} does not exist in the schema."))
        o = h.oriented()
        if not o or not all(o) or not h.rel_types:
            continue
        s_lab, o_lab = o
        if s_lab not in onto.ent_props or o_lab not in onto.ent_props:
            continue
        for t in h.rel_types:
            if t not in onto.rel_names or onto.get(s_lab, t, o_lab):
                continue
            if onto.get(o_lab, t, s_lab):
                def fx(qq, h=h):
                    s, e = h.rel_span
                    seg = qq[s:e]
                    bs, be = h.bracket_span[0] - s, h.bracket_span[1] - s
                    nl, nr = ("<-", "-") if (h.left, h.right) == ("-", "->") else ("-", "->")
                    return qq[:s] + nl + seg[bs:be] + nr + qq[e:]
                out.append(Issue("S_DIRECTION", "schema", f"({s_lab})-[{t}]->({o_lab}) reversed", fix=fx))
                break
            out.append(Issue("S_ENDPOINT", "schema", f"({s_lab})-[{t}]->({o_lab})",
                             f"Relationship :{t} does not connect :{s_lab} to :{o_lab} in the schema."))
    # thuộc tính
    for v, p in re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", m_q):
        lab = var_label.get(v)
        if lab in onto.ent_props and p not in onto.ent_props[lab]:
            out.append(Issue("S_UNKNOWN", "schema", f"prop {lab}.{p}",
                             f"Property {p} does not exist on :{lab}."))
    return out


def ontology_issues(q, question, onto):
    hops, var_label, m_q = parse(q)
    out = []
    years = sorted(set(YEAR.findall(question)))
    agg_vars = {m.group(2) for m in AGG.finditer(m_q)}
    for k, h in enumerate(hops):
        o = h.oriented()
        labs = o if o else (h.a_lab, h.b_lab)
        if not all(labs):
            continue
        for t in h.rel_types:
            r = onto.get(labs[0], t, labs[1]) or (None if o else onto.get(labs[1], t, labs[0]))
            if not r:
                continue
            start = next((p for p in r.interval if p.startswith("start")), None)
            end = next((p for p in r.interval if p.startswith("end")), None)
            # --- O_TIME_EQ: {start_x: Y, end_x: Y2} trong map của quan hệ
            if start and end and h.rel_props:
                kv = {a: b.strip() for a, b in
                      re.findall(r"(\w+)\s*:\s*([^,}]+)", q[h.bracket_span[0]:h.bracket_span[1]].split("{", 1)[1])}
                lit = re.compile(r"^(-?\d+|\$\w+|null|'[^']*'|\"[^\"]*\")$", re.I)
                tv = {a: b for a, b in kv.items() if a in (start, end)}
                if tv and all(lit.match(b) for b in tv.values()):
                    def fx(qq, h=h, kv=kv, tv=tv, start=start, end=end, k=k):
                        s, e = h.bracket_span
                        br = qq[s:e]
                        rest = ", ".join(f"{a}: {b}" for a, b in kv.items() if a not in (start, end))
                        var = h.rel_var or f"r_t{k}"
                        types = br[br.index(":"):].split("{")[0].strip() if ":" in br else ""
                        new = f"[{var}{types}" + (f" {{{rest}}}" if rest else "") + "]"
                        qq = qq[:s] + new + qq[e:]
                        y_s, y_e = tv.get(start), tv.get(end)
                        if y_e and y_e.lower() == "null":
                            # "end rỗng" = còn hiệu lực ở hiện tại (ngữ nghĩa khoảng của ontology)
                            cond = f"{var}.{start} IS NOT NULL AND {var}.{end} IS NULL"
                        else:
                            y_s = y_s if y_s and y_s.lower() != "null" else y_e
                            y_e = y_e or y_s
                            cond = f"{var}.{start} <= {y_s} AND ({var}.{end} >= {y_e} OR {var}.{end} IS NULL)"
                        return _insert_where(qq, s + len(new), cond)
                    out.append(Issue("O_TIME_EQ", "ontology", f"{t}: equality on {start}/{end}", fix=fx))
                    continue
            # --- O_TIME_OPEN: var.end >= Y không kèm IS NULL
            if end and h.rel_var:
                pat = re.compile(rf"\b{h.rel_var}\.{end}\s*>=?\s*([\w.$]+)")
                if pat.search(m_q) and not re.search(rf"\b{h.rel_var}\.{end}\s+IS\s+NULL", m_q, re.I):
                    def fx(qq, var=h.rel_var, end=end):
                        mm = re.search(rf"\b{var}\.{end}\s*>=?\s*[\w.$]+", mask_strings(qq))
                        if not mm:
                            return qq
                        return qq[:mm.start()] + f"({qq[mm.start():mm.end()]} OR {var}.{end} IS NULL)" + qq[mm.end():]
                    out.append(Issue("O_TIME_OPEN", "ontology", f"{t}: open interval end", fix=fx))
            # --- O_TIME_MISSING: câu hỏi có đúng một năm, quan hệ có khoảng, không lọc start/end, năm chưa xuất hiện
            if start and end and len(years) == 1 and years[0] not in m_q:
                used = h.rel_var and re.search(rf"\b{h.rel_var}\.({start}|{end})\b", m_q)
                n_interval_hops = sum(1 for hh in hops for tt in hh.rel_types
                                      if any(rr.label == tt and rr.interval for rr in onto.rels.values()))
                if not used and n_interval_hops == 1:
                    y = years[0]

                    def fx(qq, h=h, start=start, end=end, y=y, k=k):
                        hh = parse(qq)[0]
                        cur = next((x for x in hh if x.rel_span[0] == h.rel_span[0]), h)
                        qq, var, pos = _ensure_rel_var(qq, cur, f"r_t{k}")
                        cond = f"{var}.{start} <= {y} AND ({var}.{end} >= {y} OR {var}.{end} IS NULL)"
                        return _insert_where(qq, pos, cond)
                    out.append(Issue("O_TIME_MISSING", "ontology", f"{t}: year {y} not applied", fix=fx))
            # --- O_OPTIONAL: đếm/gom qua quan hệ không bắt buộc trong MATCH bắt buộc
            if h.clause == "MATCH" and agg_vars:
                for g_var, g_lab, x_var, g_is_subj in ((h.a_var, labs[0] if o else h.a_lab, h.b_var, True),
                                                       (h.b_var, labs[1] if o else h.b_lab, h.a_var, False)):
                    # xác định vai trò chủ thể/đối tượng của g theo hướng thực của quan hệ
                    if o:
                        g_is_subj = (g_var == (h.a_var if (h.left, h.right) == ("-", "->") else h.b_var))
                    if not g_var or (x_var not in agg_vars and h.rel_var not in agg_vars) or g_var in agg_vars:
                        continue
                    mand = r.mandatory_subj if g_is_subj else r.mandatory_obj
                    part = r.subj_part if g_is_subj else r.obj_part
                    # g phải đã bị ràng buộc bởi điều kiện khác (quan hệ khác hoặc thuộc tính)
                    other = [hh for hh in hops if hh is not h and g_var in (hh.a_var, hh.b_var)]
                    if mand or not other:
                        continue
                    g_label = r.subj if g_is_subj else r.obj
                    out.append(Issue(
                        "O_OPTIONAL", "ontology", f"{t} optional for {g_label} ({100 * part:.1f}%)",
                        message=(f"Relationship (:{r.subj})-[:{t}]->(:{r.obj}) is optional for :{g_label}: only "
                                 f"{100 * part:.1f}% of {g_label} nodes have it. The query counts/aggregates over it "
                                 f"inside a required MATCH, so every {g_label} without it is dropped. If the question "
                                 f"asks for each {g_label} (including those with none) together with a count, match "
                                 f"this relationship with OPTIONAL MATCH and count DISTINCT target nodes.")))
            # --- O_SEMANTIC_DIR: quan hệ nối cùng kiểu, nhắc nghĩa theo hướng (Wikidata)
            if r.subj == r.obj and r.reading and o:
                if r.wd_direction == "inverse":
                    sense = f"x is the '{r.reading}' of y"
                else:
                    sense = f"y is the '{r.reading}' of x"
                out.append(Issue(
                    "O_SEMANTIC_DIR", "ontology", f"{t}: {sense}",
                    message=(f"In (x:{r.subj})-[:{t}]->(y:{r.obj}), {sense} (Wikidata: {r.meaning or r.reading}). "
                             f"Check that the direction of :{t} in the query matches the question; reverse it if not.")))
    # loại trùng
    seen, uniq = set(), []
    for i in out:
        if (i.code, i.detail) not in seen:
            seen.add((i.code, i.detail))
            uniq.append(i)
    return uniq


def detect(q, question, onto):
    iss = schema_issues(q, onto)
    if onto.level == "ontology":
        iss += ontology_issues(q, question, onto)
    return iss


def verify(q, question, onto, max_rounds=8):
    """Áp các sửa tự động cho tới khi hết; trả (câu đã sửa, các vấn đề đã sửa, các vấn đề cần thông báo)."""
    applied = []
    for _ in range(max_rounds):
        iss = detect(q, question, onto)
        fixable = [i for i in iss if i.fix]
        if not fixable:
            break
        i = fixable[0]
        try:
            new = i.fix(q)
        except Exception as e:  # noqa: BLE001 — sửa lỗi phải an toàn, không làm hỏng câu
            applied.append(f"{i.code}: FAILED {e!r}")
            break
        if new == q:
            break
        q = new
        applied.append(f"{i.code}: {i.detail}")
    pending = [i for i in detect(q, question, onto) if not i.fix and i.message]
    return q, applied, pending
