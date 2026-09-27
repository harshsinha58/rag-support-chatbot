"""Central place for all environment-driven configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_RETRIEVAL_DISTANCE = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "1.3"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

ENABLE_EMAIL_HANDOFF = os.getenv("ENABLE_EMAIL_HANDOFF", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
HANDOFF_TO_EMAIL = os.getenv("HANDOFF_TO_EMAIL", "")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
TICKETS_LOG = os.path.join(BASE_DIR, "data", "logs", "tickets.json")
UNANSWERED_LOG = os.path.join(BASE_DIR, "data", "logs", "unanswered.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # local, free, same family you used before
