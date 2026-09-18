"""Steps 1-3: load, chunk, embed, save. Run once; the app just loads the index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import config
from rag import chunker, index, loader

print(f"1. loading {config.DOCUMENT.name}")
text = loader.load(config.DOCUMENT)
print(f"   {len(text):,} characters")

print("2. chunking")
chunks = chunker.split(text)
print(f"   {len(chunks)} chunks, "
      f"{len({c.metadata['heading'] for c in chunks})} distinct headings")

print("3. embedding into the vector store")
index.build(chunks)

# A manifest so the app can tell users what the bot covers, without loading the
# source document at runtime.
sections = list(dict.fromkeys(c.metadata["heading"].split(" > ")[0] for c in chunks))
config.MANIFEST_PATH.write_text(json.dumps({
    "title": config.DOCUMENT_TITLE,
    "subtitle": config.DOCUMENT_SUBTITLE,
    "source_url": config.DOCUMENT_SOURCE_URL,
    "file": config.DOCUMENT.name,
    "chunks": len(chunks),
    "sections": sections,
}, indent=2), encoding="utf-8")
print(f"\nindex written to {config.INDEX_PATH.name}")
print(f"manifest written: {len(sections)} sections")
