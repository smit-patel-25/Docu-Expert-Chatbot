"""Steps 4b, 5 and 6 - conversation memory, then the grounded answer.

Memory is applied *before* retrieval: a follow-up like "and can that be reduced?"
carries nothing worth embedding, so it's rewritten into a standalone question
using the history, and that is what gets searched.

Grounding is enforced twice, because a prompt alone leaks:
  1. the similarity floor, which refuses without calling the LLM at all
  2. the system message, for questions that clear the floor but still aren't
     answered by the retrieved text
"""
import logging
from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from rag import index

# LangChain calls generate_content directly, which makes the SDK log an
# automatic-function-calling advisory on every turn. It is noise here.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

SYSTEM = f"""You answer questions about one document, using the numbered SOURCES below.

- Use the SOURCES only. Never use outside or general knowledge.
- If the SOURCES don't answer the question, reply exactly: {config.REFUSAL}
- Cite what you used inline as [1], [2].
- If the SOURCES only partly cover it, answer that part and say what's missing.
- Use the conversation to understand the question, never as a source of facts.
- Be concise and use the document's own wording."""

REWRITE = """Rewrite the user's last message as a standalone question, resolving any
pronouns from the conversation. Output only the question.

Conversation:
{history}

Last message: {question}"""


def llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL, temperature=temperature,
        google_api_key=config.API_KEY)


def text_of(message) -> str:
    """Gemini 3 models return content as blocks (text plus a thought signature)
    rather than a plain string, so pull out just the visible text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def _history_messages(history: list[dict]) -> list:
    return [HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in history[-config.HISTORY_TURNS * 2:]]


def rewrite(question: str, history: list[dict], model: ChatGoogleGenerativeAI) -> str:
    if not history:
        return question
    recent = "\n".join(f"{m['role']}: {m['content']}"
                       for m in history[-config.HISTORY_TURNS * 2:])
    reply = index.retry(lambda: model.invoke(
        REWRITE.format(history=recent, question=question)))
    return text_of(reply).strip() or question


def retrieve(question: str, history: list[dict], store,
             model: ChatGoogleGenerativeAI) -> tuple[str, list[dict], bool]:
    """Return (standalone question, hits, grounded?)."""
    standalone = rewrite(question, history, model)
    scored = index.search(store, standalone)
    hits = [{"heading": doc.metadata["heading"], "text": doc.metadata["text"],
             "score": float(score)} for doc, score in scored]
    return standalone, hits, bool(hits) and hits[0]["score"] >= config.SIMILARITY_FLOOR


def format_sources(hits: list[dict]) -> str:
    return "\n\n".join(f"[{n}] ({h['heading']})\n{h['text']}"
                       for n, h in enumerate(hits, start=1))


def stream(question: str, hits: list[dict], history: list[dict], grounded: bool,
           model: ChatGoogleGenerativeAI) -> Iterator[str]:
    """Yield the answer in pieces. Refuses up front when retrieval found nothing."""
    if not grounded:
        yield config.REFUSAL
        return
    messages = [SystemMessage(content=SYSTEM), *_history_messages(history),
                HumanMessage(content=f"SOURCES:\n{format_sources(hits)}"
                                     f"\n\nQUESTION: {question}")]
    for chunk in index.retry(lambda: model.stream(messages)):
        if piece := text_of(chunk):
            yield piece


def complete(question: str, hits: list[dict], history: list[dict], grounded: bool,
             model: ChatGoogleGenerativeAI) -> str:
    """Non-streaming, for the eval harness."""
    return "".join(stream(question, hits, history, grounded, model))
