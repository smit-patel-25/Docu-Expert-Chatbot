"""Requirement 6, measured rather than asserted.

Runs eval/questions.json and reports answer recall (in-scope questions that got a
real answer) and refusal accuracy (out-of-scope questions that correctly refused).
The top similarity score per question shows you where to set SIMILARITY_FLOOR.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag import chat, index

cases = json.loads((config.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
store, model = index.load(), chat.llm()

rows = []
for case in cases:
    history = case.get("history", [])
    _, hits, grounded = chat.retrieve(case["q"], history, store, model)
    answer = chat.complete(case["q"], hits, history, grounded, model)
    refused = config.REFUSAL.lower().rstrip(".") in answer.lower()
    ok = refused if case["expect"] == "refuse" else (
        not refused and all(s.lower() in answer.lower()
                            for s in case.get("must_contain", [])))
    rows.append((ok, case["expect"]))
    print(f"{'PASS' if ok else 'FAIL'}  {case['expect']:6} {hits[0]['score']:.3f}  {case['q']}")
    print(f"       -> {answer.replace(chr(10), ' ')[:100]}")

kind = lambda k: [r for r in rows if r[1] == k]
print(f"\nanswer recall    : {sum(r[0] for r in kind('answer'))}/{len(kind('answer'))}")
print(f"refusal accuracy : {sum(r[0] for r in kind('refuse'))}/{len(kind('refuse'))}")
print(f"overall          : {sum(r[0] for r in rows)}/{len(rows)}  (floor={config.SIMILARITY_FLOOR})")
sys.exit(0 if all(r[0] for r in rows) else 1)
