"""
Ingestion pipeline: load documents (PDF, TXT, MD), split into chunks with
section-aware metadata, embed them, and persist to a FAISS index on disk.

Each chunk stores metadata used later for citations:
  - source: original filename
  - section: a human-readable label (e.g. "Page 3" or "Section: Refunds")
"""
import os
import json
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src import config


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def _load_single_file(file_path: str):
    """Load one file into LangChain Documents, tagging section metadata."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for d in docs:
            page_num = d.metadata.get("page", 0) + 1
            d.metadata["source"] = filename
            d.metadata["section"] = f"Page {page_num}"
        return docs

    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
            d.metadata["section"] = "Full document"
        return docs

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_files(file_paths: list[str], append: bool = True) -> dict:
    """
    Ingest a list of file paths into the FAISS vector store.
    Returns a small summary dict for the UI to display.
    """
    all_docs = []
    for fp in file_paths:
        all_docs.extend(_load_single_file(fp))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # Give each chunk a stable, human-readable section label including its
    # position within the source doc, so citations are specific.
    counter = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counter[src] = counter.get(src, 0) + 1
        base_section = c.metadata.get("section", "Section")
        c.metadata["section"] = f"{base_section} (chunk {counter[src]})"

    embeddings = _get_embeddings()

    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
    index_file = os.path.join(config.VECTORSTORE_DIR, "index.faiss")

    if append and os.path.exists(index_file):
        store = FAISS.load_local(
            config.VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(config.VECTORSTORE_DIR)

    _update_manifest(file_paths)

    return {
        "files_ingested": [os.path.basename(f) for f in file_paths],
        "chunks_created": len(chunks),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _update_manifest(file_paths: list[str]):
    """Keep a simple record of which files have been ingested, for the admin page."""
    manifest_path = os.path.join(config.VECTORSTORE_DIR, "manifest.json")
    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
    for fp in file_paths:
        existing.append(
            {"filename": os.path.basename(fp), "ingested_at": datetime.utcnow().isoformat()}
        )
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2)


def load_vectorstore():
    """Load the persisted FAISS index, or None if it doesn't exist yet."""
    index_file = os.path.join(config.VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_file):
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(
        config.VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )


def get_manifest() -> list[dict]:
    manifest_path = os.path.join(config.VECTORSTORE_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as f:
        return json.load(f)
