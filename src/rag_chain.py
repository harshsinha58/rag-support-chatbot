"""
Core question-answering logic.

Flow per question:
  1. Retrieve top-k chunks from FAISS with similarity distances.
  2. If the best match is too dissimilar (distance > MAX_RETRIEVAL_DISTANCE),
     skip the LLM entirely and return "I don't know" — this is the
     hallucination guard the project spec asks for.
  3. Otherwise, build a prompt with the retrieved chunks + recent chat
     history, ask the LLM, and return the answer plus the source
     citations that were actually used.
  4. Every unanswered question is logged for the admin page.
"""
import json
import os
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

from src import config
from src.ingest import load_vectorstore
from src.llm import get_llm

# In-memory per-session chat history: {session_id: [(role, text), ...]}
_SESSION_HISTORY: dict[str, list[tuple[str, str]]] = {}

SYSTEM_PROMPT = """You are a customer support assistant. Answer the user's question
using ONLY the context provided below, which comes from the company's own documents.

Rules:
- If the context does not contain enough information to answer confidently,
  say exactly: "I don't have enough information in the documents to answer that."
  Do not guess or use outside knowledge.
- Keep answers concise and directly useful to a customer.
- Do not mention "the context" or "the documents" explicitly in your answer;
  just answer naturally as support staff would.

Context:
{context}

Recent conversation:
{history}
"""

HANDOFF_TRIGGERS = ["talk to a human", "talk to human", "human agent", "speak to someone", "real person"]


def get_session_history(session_id: str) -> list[tuple[str, str]]:
    return _SESSION_HISTORY.get(session_id, [])


def _append_history(session_id: str, role: str, text: str):
    _SESSION_HISTORY.setdefault(session_id, []).append((role, text))
    # Keep only the last 10 turns to bound prompt size
    _SESSION_HISTORY[session_id] = _SESSION_HISTORY[session_id][-10:]


def _format_history(session_id: str) -> str:
    turns = get_session_history(session_id)
    if not turns:
        return "(no earlier messages)"
    return "\n".join(f"{role}: {text}" for role, text in turns)


def _log_unanswered(session_id: str, question: str):
    os.makedirs(os.path.dirname(config.UNANSWERED_LOG), exist_ok=True)
    entries = []
    if os.path.exists(config.UNANSWERED_LOG):
        with open(config.UNANSWERED_LOG, "r") as f:
            entries = json.load(f)
    entries.append({
        "session_id": session_id,
        "question": question,
        "timestamp": datetime.utcnow().isoformat(),
    })
    with open(config.UNANSWERED_LOG, "w") as f:
        json.dump(entries, f, indent=2)


def wants_human_handoff(question: str) -> bool:
    q = question.lower()
    return any(trigger in q for trigger in HANDOFF_TRIGGERS)


def answer_question(session_id: str, question: str) -> dict:
    """
    Returns:
      {
        "answer": str,
        "citations": [{"source": ..., "section": ...}, ...],
        "is_unknown": bool,
        "needs_handoff": bool,
      }
    """
    if wants_human_handoff(question):
        _append_history(session_id, "user", question)
        return {
            "answer": "Of course — I'll connect you with a human team member. "
                       "I've created a support ticket so someone can follow up with you.",
            "citations": [],
            "is_unknown": False,
            "needs_handoff": True,
        }

    store = load_vectorstore()
    if store is None:
        return {
            "answer": "No documents have been uploaded yet, so I can't answer questions "
                       "until some are indexed. Please upload documents first.",
            "citations": [],
            "is_unknown": True,
            "needs_handoff": False,
        }

    results = store.similarity_search_with_score(question, k=config.RETRIEVAL_K)

    if not results or results[0][1] > config.MAX_RETRIEVAL_DISTANCE:
        _append_history(session_id, "user", question)
        _log_unanswered(session_id, question)
        answer = "I don't have enough information in the documents to answer that."
        _append_history(session_id, "assistant", answer)
        return {
            "answer": answer,
            "citations": [],
            "is_unknown": True,
            "needs_handoff": False,
        }

    context_parts = []
    citations = []
    seen = set()
    for doc, score in results:
        context_parts.append(doc.page_content)
        key = (doc.metadata.get("source"), doc.metadata.get("section"))
        if key not in seen:
            seen.add(key)
            citations.append({"source": key[0], "section": key[1]})

    context = "\n\n---\n\n".join(context_parts)
    history_text = _format_history(session_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(context=context, history=history_text)),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"question": question})
    answer_text = response.content if hasattr(response, "content") else str(response)

    is_unknown = "don't have enough information" in answer_text.lower()
    if is_unknown:
        _log_unanswered(session_id, question)
        citations = []

    _append_history(session_id, "user", question)
    _append_history(session_id, "assistant", answer_text)

    return {
        "answer": answer_text,
        "citations": citations,
        "is_unknown": is_unknown,
        "needs_handoff": False,
    }
