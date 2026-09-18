"""Settings. Verify the model names are live: uv run python scripts/check_models.py"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")

# Pinned to explicit versions, not the "-latest" aliases: an alias moves under you,
# which is what pinning is meant to prevent.
# flash-lite, not flash: the free tier allows only 5 requests/minute for the bigger
# flash models, which is unusable when each question costs 1-2 calls.
CHAT_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"

ROOT = Path(__file__).parent
DOCUMENT = ROOT / "data" / "raw" / "laws-of-the-game.md"
INDEX_PATH = ROOT / "data" / "index" / "vectorstore.json"

CHUNK_SIZE, CHUNK_OVERLAP = 900, 150
TOP_K = 5

# If the best chunk scores below this, the question is out-of-scope and is refused
# without calling the LLM at all. Tune with scripts/evaluate.py.
SIMILARITY_FLOOR = 0.70
HISTORY_TURNS = 4

REFUSAL = "I don't have that information."
