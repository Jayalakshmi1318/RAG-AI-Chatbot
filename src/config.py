import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# --- Google Gemini Settings ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Default to current Gemini models; configurable via .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()

# --- RAG & Chunking Parameters ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "6"))
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.65"))

# --- Storage Settings ---
CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))


def validate_config() -> dict[str, bool | str]:
    """
    Validates essential configuration settings.
    Returns status and descriptive error message if invalid.
    """
    if not GEMINI_API_KEY:
        return {
            "valid": False,
            "message": "GEMINI_API_KEY is not set. Please add it to your .env file or environment variables.",
        }
    return {"valid": True, "message": "Configuration valid."}
