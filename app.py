"""Docu-Expert - answers only from one document, and remembers the conversation.

    uv run streamlit run app.py
"""
import json

import streamlit as st

import config
from rag import chat, index

st.set_page_config(page_title="Docu-Expert", page_icon="📄", layout="wide")


# Streamlit re-runs this file on every interaction, so the vector store and model
# must be cached or they'd be rebuilt on each keystroke.
@st.cache_resource(show_spinner="Loading index...")
def bootstrap():
    return index.load(), chat.llm()


@st.cache_data
def manifest() -> dict:
    """What the index actually contains — written by scripts/ingest.py."""
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"title": config.DOCUMENT.name, "sections": [], "chunks": 0}


if not config.API_KEY:
    st.error("GEMINI_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")
    st.stop()

try:
    store, model = bootstrap()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("hits", [])
st.session_state.setdefault("grounded", None)

chat_col, source_col = st.columns([2, 1], gap="large")

doc = manifest()

with chat_col:
    st.title("Docu-Expert")
    st.caption(
        f"Every answer comes only from **{doc['title']}**"
        + (f" — {doc['subtitle']}" if doc.get("subtitle") else "")
        + f". Anything outside it gets *\"{config.REFUSAL}\"*"
    )

    with st.expander(f"What's in this document? "
                     f"({len(doc['sections'])} sections, {doc['chunks']} chunks)"):
        st.markdown("\n".join(f"- {section}" for section in doc["sections"]))
        if doc.get("source_url"):
            st.markdown(f"[View the original document]({doc['source_url']})")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        st.write("Try one of these:")
        for i, q in enumerate(json.loads((config.ROOT / "data" / "suggestions.json")
                                         .read_text(encoding="utf-8"))):
            if st.button(q, key=f"s{i}", use_container_width=True):
                st.session_state.ask = q

question = st.chat_input("Ask about the document...") or st.session_state.pop("ask", None)

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                _, hits, grounded = chat.retrieve(question, history, store, model)
            reply = st.write_stream(chat.stream(question, hits, history, grounded, model))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.hits, st.session_state.grounded = hits, grounded

with source_col:
    st.subheader("Retrieved context")
    if st.session_state.grounded is None:
        st.info("Ask a question to see which parts of the document were used.")
    else:
        if st.session_state.grounded:
            st.success("Grounded — answered from the document")
        else:
            st.warning(f"Best match scored below the {config.SIMILARITY_FLOOR} floor "
                       "— refused without calling the model")
        for n, hit in enumerate(st.session_state.hits, start=1):
            with st.expander(f"[{n}] {hit['score']:.3f} — {hit['heading']}"):
                st.markdown(hit["text"])
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.clear()
            st.rerun()
