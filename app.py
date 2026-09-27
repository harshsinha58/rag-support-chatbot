"""
Main customer-facing chat UI.
Run with: streamlit run app.py
"""
import os
import uuid

import streamlit as st

from src import config
from src.ingest import ingest_files
from src.rag_chain import answer_question
from src.handoff import create_ticket

st.set_page_config(page_title="Support Chatbot", page_icon="💬", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": ..., "content": ..., "citations": [...]}]

st.title("💬 Customer Support Chatbot")
st.caption("Answers are grounded in the documents uploaded below — nothing else.")

with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index documents"):
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        saved_paths = []
        for uf in uploaded_files:
            path = os.path.join(config.DOCS_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(path)

        with st.spinner("Chunking and indexing documents..."):
            try:
                summary = ingest_files(saved_paths, append=True)
                st.success(
                    f"Indexed {summary['chunks_created']} chunks from "
                    f"{len(summary['files_ingested'])} file(s)."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(f"LLM provider: **{config.LLM_PROVIDER}**")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}`")

    if st.button("🔴 I need a human"):
        ticket = create_ticket(
            st.session_state.session_id,
            question="(manually requested from sidebar)",
            reason="user_requested",
        )
        st.info(f"Ticket #{ticket['id']} created. A team member will follow up.")

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['source']}** — {c['section']}")

# --- Chat input ---
if question := st.chat_input("Ask a question about our product..."):
    st.session_state.messages.append({"role": "user", "content": question, "citations": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(st.session_state.session_id, question)
            except Exception as e:
                result = {
                    "answer": f"Something went wrong while generating an answer: {e}",
                    "citations": [],
                    "is_unknown": True,
                    "needs_handoff": False,
                }

        st.markdown(result["answer"])

        if result["citations"]:
            with st.expander("Sources"):
                for c in result["citations"]:
                    st.markdown(f"- **{c['source']}** — {c['section']}")

        if result["needs_handoff"]:
            ticket = create_ticket(st.session_state.session_id, question, reason="user_requested")
            st.info(f"🎫 Ticket #{ticket['id']} created.")

        elif result["is_unknown"]:
            ticket = create_ticket(question=question, session_id=st.session_state.session_id,
                                    reason="bot_could_not_help")
            st.caption(f"Logged as unanswered (ticket #{ticket['id']}) for the support team to review.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
    })
