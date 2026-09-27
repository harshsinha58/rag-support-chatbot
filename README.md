# RAG-Based Customer Support Chatbot

A chatbot that answers customer questions strictly from a company's own
documents (PDF, TXT, MD), cites the section it pulled the answer from,
admits when it doesn't know, and hands off to a human when needed.

## Features

- 📄 Upload PDFs / TXT / MD files and index them into a FAISS vector store
- 🔍 Answers include **citations** (source file + section/page)
- 🚫 Says **"I don't have enough information"** instead of guessing, using a
  similarity-distance threshold on retrieval, checked *before* the LLM is
  even called
- 🙋 **Human handoff**: triggered either when the user explicitly asks
  ("talk to a human") or automatically when the bot can't answer — both
  create a ticket (JSON log) and can optionally send an email alert
- 💬 **Session-based conversation history** (multi-turn context within one
  browser session)
- 🛠️ **Admin dashboard** (stretch goal): shows unanswered questions, open
  tickets, and indexed documents
- 🔌 Swappable LLM backend: Gemini / Claude / OpenAI via one env var

## Architecture

```
┌─────────────┐      upload       ┌──────────────┐
│  Streamlit  │ ───────────────▶  │  ingest.py   │
│   app.py    │                   │ (chunk+embed)│
└─────┬───────┘                   └──────┬───────┘
      │ question                         │ persist
      ▼                                  ▼
┌─────────────┐   similarity     ┌──────────────┐
│ rag_chain.py│ ───────────────▶ │ FAISS index  │
│  (retrieval │ ◀─────────────── │ (data/       │
│  + prompt)  │   top-k chunks   │  vectorstore)│
└─────┬───────┘                  └──────────────┘
      │ if confident: call LLM
      ▼
┌─────────────┐
│   llm.py    │  (Gemini / Claude / OpenAI)
└─────────────┘

      │ if not confident OR user asks for a human
      ▼
┌─────────────┐
│ handoff.py  │ → tickets.json (+ optional email)
└─────────────┘
```

**Why this design:**
- **FAISS over a hosted vector DB** — no external service/account needed to
  demo the project locally; the same LangChain interface would let it be
  swapped for Chroma/Pinecone with a one-line change in `ingest.py`.
- **Local embeddings (sentence-transformers)** — keeps ingestion free and
  fast, and separates the "understand text" cost from the "generate answer"
  cost, which only happens after we already know we have a confident match.
- **Distance-threshold gate before calling the LLM** — the "I don't know"
  requirement is enforced structurally (no chunk close enough → no LLM call
  → no chance of hallucination), not just via a prompt instruction, which
  is a much weaker guarantee.
- **JSON-file tickets instead of a real CRM** — keeps the handoff feature
  demonstrable without needing HubSpot/Zendesk credentials; `handoff.py`
  isolates this so swapping in a real API later only touches one file.

## Tech Used

- Python, Streamlit (chat UI + admin dashboard)
- LangChain (document loading, splitting, retrieval, prompting)
- FAISS (vector store)
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- Gemini / Claude / OpenAI (pluggable) for generation
- Plain JSON files for tickets and unanswered-question logs

## Setup

```bash
git clone <your-repo-url>
cd rag-support-bot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY,
# and set LLM_PROVIDER accordingly)

streamlit run app.py
```

A sample FAQ document is already included at `data/docs/sample_faq.md` —
upload it from the sidebar and click **Index documents** to try the bot
immediately without needing your own files.

The admin dashboard is auto-available in the Streamlit sidebar under **Admin**.

## Error Handling & Edge Cases Covered

- No documents indexed yet → bot tells the user instead of crashing
- Unsupported file type on upload → clear `ValueError` surfaced in the UI
- LLM call failure (bad API key, rate limit, network) → caught and shown
  as a chat message instead of crashing the app
- Empty/low-confidence retrieval → routed to "I don't know" + logged, never
  sent to the LLM
- Email handoff misconfigured or failing → fails silently (logged to
  console) so a broken SMTP setup never blocks ticket creation itself

## What I'd Improve With More Time

- Replace the L2-distance threshold with a re-ranking step (e.g. a cross-encoder)
  for a more reliable confidence signal than raw FAISS distance
- Move tickets/unanswered logs from JSON files to a real database (SQLite at
  minimum) to handle concurrent sessions safely
- Add streaming responses instead of waiting for the full LLM output
- Add authentication so the admin dashboard isn't publicly viewable
- Support website URLs as a document source (not just file upload), per the
  original spec's "website pages" mention
- Add automated tests around the retrieval-threshold logic, since it's the
  core correctness guarantee of the whole project
- Package as an embeddable widget (stretch goal) via a lightweight iframe +
  a small FastAPI backend instead of Streamlit, since Streamlit isn't
  naturally embeddable on a third-party website
