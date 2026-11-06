"""
config/settings.py
==================
Centralized application settings loaded from environment variables and .env file.
Provides a strongly-typed, immutable Settings dataclass with sensible defaults.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    # Resolve project root (parent of config folder)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application settings schema."""
    # Paths
    project_root: Path = PROJECT_ROOT
    dataset_path: Path = PROJECT_ROOT / "cricket_log.txt"
    data_dir: Path = PROJECT_ROOT / "data"

    # Google Gemini Settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # MongoDB Settings
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "cricket_db")

    # Qdrant Settings
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "cricket_events")
    qdrant_local_path: str = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant_storage")

    # Embedding Settings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_dimension: int = 384

    # Django Backend Settings
    django_secret_key: str = os.getenv(
        "DJANGO_SECRET_KEY",
        "django-insecure-cricket-semantic-search-2026-key"
    )
    django_debug: bool = os.getenv("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")
    django_allowed_hosts: str = os.getenv("DJANGO_ALLOWED_HOSTS", "*")
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton instance of Settings."""
    return Settings()
