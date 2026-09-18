"""Step 1 - load the document."""
from pathlib import Path


def load(path: Path) -> str:
    """Markdown and text load directly; PDFs go through pypdf."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")
