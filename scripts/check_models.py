"""Step 0: never trust a model name from a tutorial. Print what THIS key can use.

LangChain has no model-listing call, so this uses the Google SDK directly — the
same thing the workshop notebook does.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai

import config

# Hold the client in a variable: models.list() is a lazy pager, and a throwaway
# client gets garbage-collected (closing its connection) mid-iteration.
client = genai.Client(api_key=config.API_KEY)

chat, embed = [], []
for m in client.models.list():
    actions = set(m.supported_actions or [])
    if "generateContent" in actions:
        chat.append(m.name.replace("models/", ""))
    if "embedContent" in actions:
        embed.append(m.name)

for label, pinned, pool in (("CHAT_MODEL", config.CHAT_MODEL, chat),
                            ("EMBED_MODEL", config.EMBED_MODEL, embed)):
    live = pinned in pool or pinned.replace("models/", "") in pool
    print(f"{label:12} = {pinned:30} {'OK' if live else 'NOT AVAILABLE TO THIS KEY'}")
    if not live:
        print(f"  available: {', '.join(sorted(pool))}")
