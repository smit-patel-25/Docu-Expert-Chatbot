"""Settings. Verify the model names are live: uv run python scripts/check_models.py"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")

if not API_KEY:
    # Streamlit Community Cloud supplies the key through its Secrets panel rather
    # than a .env file. Wrapped, because the CLI scripts run without Streamlit.
    try:
        import streamlit as st

        API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

# Pinned to explicit versions, not the "-latest" aliases: an alias moves under you,
# which is what pinning is meant to prevent.
# flash-lite, not flash: the free tier allows only 5 requests/minute for the bigger
# flash models, which is unusable when each question costs 1-2 calls.
CHAT_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"

ROOT = Path(__file__).parent
DOCUMENT = ROOT / "data" / "raw" / "laws-of-the-game.md"
INDEX_PATH = ROOT / "data" / "index" / "vectorstore.json"
MANIFEST_PATH = ROOT / "data" / "index" / "document.json"

# Shown in the UI so a user knows what the bot is and isn't able to answer.
DOCUMENT_TITLE = "IFAB Laws of the Game 2024/25"
DOCUMENT_SUBTITLE = "The 17 Laws of association football"
DOCUMENT_SOURCE_URL = "https://www.theifab.com/laws-of-the-game-documents/"

CHUNK_SIZE, CHUNK_OVERLAP = 900, 150
TOP_K = 5

# If the best chunk scores below this, the question is out-of-scope and is refused
# without calling the LLM at all. Tune with scripts/evaluate.py.
SIMILARITY_FLOOR = 0.70
HISTORY_TURNS = 4

REFUSAL = "I don't have that information."
