import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    def __init__(self) -> None:
        self.data_dir = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
        self.db_path = self.data_dir / "app.db"
        self.upload_dir = self.data_dir / "uploads"
        self.static_dir = BASE_DIR / "static"

        # LLM provider: auto | groq | gemini | ollama | anthropic | openai-compatible
        self.llm_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        self.llm_model = os.getenv("LLM_MODEL", "").strip() or None          # overrides the provider default
        self.llm_base_url = os.getenv("LLM_BASE_URL", "").strip() or None
        fallbacks = os.getenv("LLM_FALLBACK_MODELS")                           # comma-separated; empty = none
        self.llm_fallback_models = (None if fallbacks is None or fallbacks.strip() == "default"
                                    else [m.strip() for m in fallbacks.split(",") if m.strip()])
        self.llm_vision_model = os.getenv("LLM_VISION_MODEL", "").strip() or None  # OCR model if different
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        self.cookie_secure = _bool("COOKIE_SECURE")

        # Mandatory knowledge-base limits.
        self.max_documents = 10
        self.max_pages = 20
        self.max_file_bytes = 10 * 1024 * 1024
        self.ocr_concurrency = 3                # scanned pages OCR'd in parallel

        self.session_ttl_hours = 24 * 7
        self.max_question_chars = 1000
        self.ask_rate_limit_per_minute = 30


settings = Settings()
