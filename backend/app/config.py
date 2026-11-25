"""Central configuration. Values come from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(BASE_DIR.parent / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'spam_detector.db'}")
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = BASE_DIR / "data"
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
STORE_EMAIL_TEXT = _bool("STORE_EMAIL_TEXT", True)
MAX_SUBJECT_CHARS = int(os.getenv("MAX_SUBJECT_CHARS", "300"))
MAX_BODY_CHARS = int(os.getenv("MAX_BODY_CHARS", "20000"))

MODEL_FILENAME = "spam_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"

MAX_USERS = int(os.getenv("MAX_USERS", "50"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
FRONTEND_DIR = BASE_DIR.parent / "frontend"
