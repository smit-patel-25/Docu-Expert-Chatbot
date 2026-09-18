# Source document

`laws-of-the-game.md` is the IFAB **Laws of the Game 2024/25** — a public
document — converted from the official PDF to markdown, trimmed to Laws 1–17.

The conversion was a **one-off**. It isn't part of the app, because messy PDF
extraction belongs in a prep step, not in code you have to read every time. The
app just reads a clean markdown file, which is why `rag/loader.py` is six lines.

To redo it from source:

```bash
curl -L -o laws-of-the-game.pdf \
  "https://downloads.theifab.com/downloads/laws-of-the-game-2024-25?l=en"
pdftotext -layout laws-of-the-game.pdf - > raw.txt   # then tidy into markdown
```

`rag/loader.py` still reads `.pdf` directly if you point `config.DOCUMENT` at
one — you just get whatever text the PDF's layout yields.
