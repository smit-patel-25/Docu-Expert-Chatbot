"""Steps 1-3: load, chunk, embed, save. Run once; the app just loads the index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
print(f"\nindex written to {config.INDEX_PATH.name}")
