"""Step 2 - split the document into chunks that carry their heading."""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _last_before(markers: list[tuple[int, str]], pos: int) -> str:
    """The most recent heading at or before `pos`."""
    return next((text for start, text in reversed(markers) if start <= pos), "")


def split(text: str) -> list[Document]:
    """Return LangChain Documents tagged with "Law 11 – Offside > 1. Offside position"."""
    laws = [(m.start(), m.group(1)) for m in re.finditer(r"^#{1,2} (.+)$", text, re.M)]
    parts = [(m.start(), m.group(1)) for m in re.finditer(r"^### (.+)$", text, re.M)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        add_start_index=True,
    )

    chunks = []
    for doc in splitter.create_documents([text]):
        body = doc.page_content.strip()
        if len(body) < 40:
            continue
        # Attribute by the chunk's midpoint: a chunk that starts on the last line
        # of Law 1 but mostly covers Law 2 belongs to Law 2.
        middle = doc.metadata["start_index"] + len(doc.page_content) // 2
        law, part = _last_before(laws, middle), _last_before(parts, middle)
        heading = f"{law} > {part}" if part else law
        # Prepending the heading lets "how big is the goal area?" match a chunk
        # whose body never says "Law 1".
        chunks.append(Document(page_content=f"{heading}\n\n{body}",
                               metadata={"heading": heading, "text": body}))
    return chunks
