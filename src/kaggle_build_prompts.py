"""Dựng sẵn prompt cho 4 nhánh RQ1 (A/B/C/D) để sinh trên Kaggle (vLLM), đảm bảo prompt GIỐNG HỆT bản MLX.

  A  : prompt CypherBench gốc, schema rút gọn                (= cb_zero_shot / cb_chat)
  B  : A + thẻ graph_info (chú thích tay của tác giả)        (= cb_chat_card --card graphinfo)
  C  : A + thẻ ontology tự xây                               (= cb_chat_card --card auto)
  D  : A + thẻ placebo (cùng khuôn, nội dung bị xáo)         (= cb_chat_card --card placebo)

Ghi kaggle/dataset/prompts.jsonl: {"arm","qid","graph","user"} — "user" là nội dung tin nhắn user;
chat template được áp trên Kaggle bằng tokenizer của từng model.
Chạy: cd src && ../.venv-cb/bin/python kaggle_build_prompts.py
"""
import hashlib
import json
from pathlib import Path

from cb_common import NL2CYPHER_PROMPT_DEFAULT, load_split
from cb_protocols import load_schema
from ontology.card import render_card

ROOT = Path(__file__).resolve().parents[1]
TEST_GRAPHS = ["company", "fictional_character", "flight_accident", "geography", "movie", "nba", "politics"]
ARMS = {"A": None, "B": "graphinfo", "C": "auto", "D": "placebo"}


def main():
    out = ROOT / "kaggle/dataset"
    out.mkdir(parents=True, exist_ok=True)
    items = load_split("test", set(TEST_GRAPHS))
    schemas = {g: load_schema(g).to_str(exclude_description=True) for g in TEST_GRAPHS}
    n = 0
    with (out / "prompts.jsonl").open("w") as f:
        for arm, src in ARMS.items():
            cards = {g: render_card(g, src) for g in TEST_GRAPHS} if src else {}
            for x in items:
                s = schemas[x.graph] + ("\n\n" + cards[x.graph] if src else "")
                user = NL2CYPHER_PROMPT_DEFAULT.format(schema=s, question=x.nl_question)
                f.write(json.dumps({"arm": arm, "qid": x.qid, "graph": x.graph, "user": user}, ensure_ascii=False) + "\n")
                n += 1
    sha = hashlib.sha256((out / "prompts.jsonl").read_bytes()).hexdigest()[:16]
    (out / "prompts_meta.json").write_text(json.dumps({"n": n, "arms": list(ARMS), "sha256_16": sha}, indent=1))
    print(n, "prompts", sha)


if __name__ == "__main__":
    main()
