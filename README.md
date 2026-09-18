# Docu-Expert-Chatbot

A chatbot that answers questions from a single document and remembers the
conversation. It answers **only** from that document, and says
*"I don't have that information."* when the answer isn't there.

Python · LangChain · Google Gemini · Streamlit — the workshop stack, built out
into a deployable app.

The document is the IFAB **Laws of the Game 2024/25** (Laws 1–17, public). A
rulebook, because its answers are crisp and verifiable and because out-of-scope
questions are obvious: *"who won the 2022 World Cup?"* is simply not in a
rulebook, which makes the refusal behaviour easy to show.

## The six requirements, mapped to the code

| # | Requirement | Where | LangChain piece |
|---|---|---|---|
| 1 | Load a document | `rag/loader.py` | — |
| 2 | Split into chunks | `rag/chunker.py` | `RecursiveCharacterTextSplitter` |
| 3 | Embed and store | `rag/index.py` | `GoogleGenerativeAIEmbeddings`, `InMemoryVectorStore` |
| 4 | Retrieve relevant chunks | `rag/index.py`, `rag/chat.py` | `similarity_search_with_score` |
| 5 | Chunks + history + question → LLM | `rag/chat.py` | `ChatGoogleGenerativeAI`, `System/Human/AIMessage` |
| 6 | Answer only from the document | `rag/chat.py` + `SIMILARITY_FLOOR` | — |

## Results

`scripts/evaluate.py` runs 13 cases — 8 answerable from the document, 5
deliberately not, plus one that only works if memory works:

```
answer recall    : 8/8
refusal accuracy : 5/5
overall          : 13/13  (floor=0.7)
```

## Two things that make it more than the notebook

**Conversation memory is applied before retrieval.** A follow-up like *"and can
that be reduced?"* carries nothing worth embedding. `chat.rewrite()` turns it
into a standalone question using the last few turns, and *that* is what gets
searched. Without this, follow-ups retrieve garbage.

**Refusal is enforced twice.** The system message forbids outside knowledge, but
prompts leak. So retrieval also has a floor: if the best chunk scores below
`SIMILARITY_FLOOR`, the bot refuses *without calling the LLM at all*.

The floor is tuned, not guessed. On the eval set:

| | similarity range |
|---|---|
| in-scope questions | 0.739 – 0.806 |
| out-of-scope questions | 0.545 – 0.677 |

0.70 sits in that gap, so all five out-of-scope questions are refused
deterministically rather than relying on the prompt. Re-tune if you change the
document — `evaluate.py` prints the score for every question.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync

cp .env.example .env        # add your Gemini API key
# free key: https://aistudio.google.com/apikey
```

## Run

```bash
uv run python scripts/check_models.py   # are the pinned models live for your key?
uv run python scripts/ingest.py         # load -> chunk -> embed -> data/index/
uv run streamlit run app.py             # web UI
uv run python scripts/chat.py           # or the terminal version; type 'quit' to stop
```

The terminal version mirrors the workshop's final interactive cell and prints the
top similarity score and matched section after each answer, so you can watch
retrieval work. The web UI shows the same thing in its right-hand panel.

## Notes from building it

**Free-tier quotas are the main obstacle.** Embeddings are capped around 100
texts/minute and the bigger chat models at **5 requests/minute** — unusable when
each question costs one or two calls. Hence `gemini-3.1-flash-lite`, batching
with pauses in `index.build()`, and `index.retry()` wrapping every API call to
back off on 429s and Google's transient 503s.

**Pin explicit model versions.** `gemini-flash-latest` is an alias that moves
under you, which is what pinning is meant to prevent. `check_models.py` tells you
immediately if a pin has gone stale.

**Gemini 3 returns content as blocks.** `.content` can be a list of blocks (text
plus an internal thought signature) rather than a string, so `chat.text_of()`
pulls out just the visible text — the same helper the workshop notebook needs.

**The document is markdown, not the original PDF.** The source is a 230-page
designed PDF whose text layer has no clean structure; recovering it took ~150
lines of parsing heuristics, so that was done **once**, offline, and the clean
markdown is what's committed. Prep work belongs in a prep step, not in code you
re-read every time. See `data/raw/README.md`.

## Swapping the document

Point `config.DOCUMENT` at any `.md`, `.txt` or `.pdf`, re-run `scripts/ingest.py`,
and update `eval/questions.json` and `data/suggestions.json` to match.

## Deploying

[Streamlit Community Cloud](https://share.streamlit.io) — free, deploys from
GitHub, no card needed.

1. Push the repo to GitHub (public is simplest; the free tier allows one private app).
2. On share.streamlit.io: **New app** → pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Python 3.12** (it won't read `.python-version`).
4. **Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Deploy.

Notes:

- Dependencies come from `requirements.txt`, not `pyproject.toml`. Regenerate it
  after changing dependencies: `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`
- `data/index/vectorstore.json` (9.4 MB) is committed on purpose, so the deployed
  app never runs ingestion at boot — which would need embedding quota it doesn't have.
- `.env` is gitignored; `config.py` falls back to `st.secrets` so the same code
  works locally and deployed.
- Apps sleep after inactivity, so the first load after a pause takes ~30s.
