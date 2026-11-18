"""Các giao thức suy luận để đối chiếu mốc zero-shot 13,12 EX mà Cypher-RI báo cho Qwen2.5-Coder-7B.

Bài Cypher-RI không công bố code và không nói mốc 13,12 được đo thế nào. Ta thử các giả thuyết:

  cb_chat     : prompt CypherBench gốc, bọc chat template (giao thức đã chạy ở cb_zero_shot.py).
  cb_raw      : prompt CypherBench gốc dạng văn bản thô, không chat template (hợp với model base);
                cắt ở dòng đầu tiên khác rỗng (prompt yêu cầu "single line").
  cypherri    : template Table 1 của Cypher-RI (văn bản thô). Hai lượt:
                (1) model nhận schema rút gọn (chỉ tên entity và relation), suy nghĩ, chọn schema trong
                    <json>…</json>; (2) hệ thống chèn <schema> đầy đủ của phần đã chọn (chọn không hợp lệ
                    thì chèn toàn bộ schema), model suy nghĩ tiếp và trả Cypher trong <answer>…</answer>.

Giả định của ta (bài không nêu): schema rút gọn = JSON {"name", "entities": [label], "relations": [label]};
JSON chọn hợp lệ khi parse được và mọi tên đều có trong schema; giới hạn sinh 2.000 token (Table 6); greedy.
"""
import json
import re

from mlx_lm import stream_generate

from cb_common import NL2CYPHER_PROMPT_DEFAULT, RAW, DataType, PropertyGraphSchema, clean_prediction

CYPHERRI_TEMPLATE = (
    "A conversation between a User and an Assistant. The user asks the Assistant to translate the question "
    "to Cypher query based on a simplified version of the schema of a graph database. The Assistant first "
    "identifies the necessary schema components required to answer the User’s question within <think> "
    "</think> tags. Next, the Assistant outputs the needed schema in JSON format within <json> </json> tags. "
    "If the selected schema is valid, the Assistant is provided with the complete version of the selected "
    "schema enclosed within <schema> tags, or it is provided with the complete version of the full schema. "
    "After that, the Assistant explains the reasoning process for generating the Cypher query within <think> "
    "</think> tags. Finally, the Assistant provides the Cypher query enclosed within <answer> </answer> tags. "
    "For example, <think> This is the reasoning process. </think> <json> {\"name\": \"\", \"entities\": [], "
    "\"relations\": []} </json> <schema> This is the complete version of the schema. </schema> <think> This "
    "is the reasoning process. </think> <answer> This is the Cypher query </answer>. "
    "Graph schema: {schema} User: {question} Assistant:"
)


def load_schema(graph):
    data = json.loads((RAW / f"graphs/schemas/{graph}_schema.json").read_text())
    return PropertyGraphSchema.from_json(data, add_meta_properties={"name": DataType.STR}).to_sorted()


def simplified_schema(s: PropertyGraphSchema) -> str:
    return json.dumps({"name": s.name, "entities": [e.label for e in s.entities],
                       "relations": sorted({r.label for r in s.relations})})


def select_schema(s: PropertyGraphSchema, json_text: str):
    """Trả (schema_str, hợp_lệ). Chọn không hợp lệ -> toàn bộ schema (theo Table 1)."""
    full = s.to_str(exclude_description=True)
    try:
        sel = json.loads(json_text)
        ents, rels = set(sel.get("entities", [])), set(sel.get("relations", []))
    except Exception:
        return full, False
    known_e, known_r = {e.label for e in s.entities}, {r.label for r in s.relations}
    if not (ents or rels) or not ents <= known_e or not rels <= known_r:
        return full, False
    rel_objs = [r for r in s.relations if r.label in rels]
    ents |= {r.subj_label for r in rel_objs} | {r.obj_label for r in rel_objs}
    sub = PropertyGraphSchema(name=s.name, entities=[e for e in s.entities if e.label in ents], relations=rel_objs)
    return sub.to_str(exclude_description=True), True


def _gen(model, tok, prompt, max_tokens, sampler, stops):
    out, n = "", 0
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
        out += r.text
        n = r.generation_tokens
        hit = [s for s in stops if s in out]
        if hit:
            out = out[: out.index(hit[0]) + len(hit[0])]
            break
    return out, n


def run_cb_chat(model, tok, sampler, schema_s, question, max_tokens=512):
    prompt = NL2CYPHER_PROMPT_DEFAULT.format(schema=schema_s, question=question)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    raw, _ = _gen(model, tok, text, max_tokens, sampler, ["\n```"])
    return {"raw": raw, "pred_cypher": clean_prediction(raw.split("\n```")[0])}


def run_cb_chat_card(model, tok, sampler, schema_s, card, question, max_tokens=512):
    """Prompt CypherBench gốc + THẺ ONTOLOGY ghép ngay sau schema (nhánh B/C/D của thí nghiệm RQ1)."""
    prompt = NL2CYPHER_PROMPT_DEFAULT.format(schema=schema_s + "\n\n" + card, question=question)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    raw, _ = _gen(model, tok, text, max_tokens, sampler, ["\n```"])
    return {"raw": raw, "pred_cypher": clean_prediction(raw.split("\n```")[0])}


def run_cb_raw(model, tok, sampler, schema_s, question, max_tokens=512):
    prompt = NL2CYPHER_PROMPT_DEFAULT.format(schema=schema_s, question=question)
    raw, _ = _gen(model, tok, prompt, max_tokens, sampler, ["\nQuestion:", "\n\n"])
    lines = [l for l in clean_prediction(raw).splitlines() if l.strip()]
    return {"raw": raw, "pred_cypher": lines[0].strip() if lines else ""}


def run_cypherri(model, tok, sampler, schema: PropertyGraphSchema, question, max_tokens=2000):
    prompt = CYPHERRI_TEMPLATE.replace("{schema}", simplified_schema(schema)).replace("{question}", question)
    part1, n1 = _gen(model, tok, prompt, max_tokens, sampler, ["</json>"])
    m = re.search(r"<json>(.*?)</json>", part1, re.S)
    sel_str, valid = select_schema(schema, m.group(1).strip() if m else "")
    if not m:
        # Không có <json>: vẫn chèn toàn bộ schema để model trả lời (như nhánh "không hợp lệ").
        part1 = part1.rstrip() + " </json>"
    ctx = prompt + part1 + f" <schema> {sel_str} </schema>"
    part2, _ = _gen(model, tok, ctx, max(64, max_tokens - n1), sampler, ["</answer>"])
    a = re.search(r"<answer>(.*?)(</answer>|$)", part2, re.S)
    pred = clean_prediction(a.group(1)) if a else ""
    pred = re.sub(r"\s*\n\s*", " ", pred).strip()
    return {"raw": part1 + " <schema>…</schema>" + part2, "pred_cypher": pred,
            "json_found": bool(m), "selection_valid": valid, "answer_found": bool(a)}
