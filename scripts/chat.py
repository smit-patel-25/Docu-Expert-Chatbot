"""Terminal chat - the workshop's interactive demo, with the same grounding.

    uv run python scripts/chat.py

Type 'quit' (or Ctrl-D) to stop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag import chat, index

QUIT = {"quit", "exit", "q"}

store, model = index.load(), chat.llm()
history = []

print(f"Docu-Expert — answers only from {config.DOCUMENT_TITLE}")
print("Type 'quit' to stop.\n")

while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not question:
        continue
    if question.lower() in QUIT:
        break

    _, hits, grounded = chat.retrieve(question, history, store, model)
    print("AI: ", end="", flush=True)
    answer = ""
    for piece in chat.stream(question, hits, history, grounded, model):
        print(piece, end="", flush=True)
        answer += piece

    top = hits[0]
    print(f"\n     [{'grounded' if grounded else 'below floor, refused'} · "
          f"top {top['score']:.3f} · {top['heading']}]\n")

    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": answer}]

print("Bye.")
