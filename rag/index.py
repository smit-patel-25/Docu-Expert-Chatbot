"""Steps 3 and 4 - embed the chunks into a vector store, and search it.

LangChain's InMemoryVectorStore does the embedding, storage and similarity search.
It also dumps to and loads from a file, so the app starts from a built index
instead of re-embedding the document on every boot.
"""
import math
import re
import time

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

_DELAY = re.compile(r"retry in ([\d.]+)s")


def retry(call, attempts: int = 6):
    """Free-tier quotas are strict (chat can be 5 requests/minute) and Google
    also returns transient 503s, so every API call goes through this."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            transient = any(code in str(exc) for code in ("429", "503", "RESOURCE_EXHAUSTED"))
            if not transient or attempt == attempts - 1:
                raise
            found = _DELAY.search(str(exc))
            wait = math.ceil(float(found.group(1))) + 1 if found else 10 * (attempt + 1)
            print(f"   busy, waiting {wait}s")
            time.sleep(wait)


def embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.API_KEY)


def build(chunks: list[Document]) -> InMemoryVectorStore:
    """Embed every chunk and save the store to disk."""
    store = InMemoryVectorStore(embeddings())
    batch = 50  # the free tier counts each text, so send them in groups
    for i in range(0, len(chunks), batch):
        if i:
            time.sleep(35)
        retry(lambda: store.add_documents(chunks[i:i + batch]))
        print(f"   embedded {min(i + batch, len(chunks))}/{len(chunks)}")
    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(config.INDEX_PATH))
    return store


def load() -> InMemoryVectorStore:
    if not config.INDEX_PATH.exists():
        raise FileNotFoundError("No index yet. Run: uv run python scripts/ingest.py")
    return InMemoryVectorStore.load(str(config.INDEX_PATH), embeddings())


def search(store: InMemoryVectorStore, question: str,
           k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """Top k chunks with their similarity scores, best first."""
    return retry(lambda: store.similarity_search_with_score(question, k=k))
