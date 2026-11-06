"""
database/mongo_client.py
========================
Robust MongoDB connection manager using PyMongo.
Includes automatic fallback to mongomock when a live MongoDB server is not running,
allowing full offline development, testing, and container portability.
"""

import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from config.settings import get_settings

logger = logging.getLogger(__name__)

_mongo_client: Optional[MongoClient] = None
_is_mock: bool = False
_is_seeding: bool = False


def get_mongo_client(uri: Optional[str] = None, timeout_ms: int = 1500) -> MongoClient:
    """
    Returns an active MongoClient.
    Tries live MongoDB first; falls back to mongomock if live server is unreachable.
    """
    global _mongo_client, _is_mock

    if _mongo_client is not None:
        return _mongo_client

    settings = get_settings()
    target_uri = uri or settings.mongodb_uri

    try:
        # Attempt connection to live MongoDB instance
        client = MongoClient(target_uri, serverSelectionTimeoutMS=timeout_ms)
        # Verify server is actually reachable
        client.admin.command("ping")
        logger.info("Connected successfully to live MongoDB at %s", target_uri)
        _mongo_client = client
        _is_mock = False
        return _mongo_client
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as exc:
        logger.warning(
            "Live MongoDB server unreachable at %s (%s). Falling back to in-memory mongomock.",
            target_uri,
            exc,
        )
        try:
            import mongomock
            _mongo_client = mongomock.MongoClient()
            _is_mock = True
            logger.info("Initialized mongomock client successfully.")
            return _mongo_client
        except ImportError:
            raise RuntimeError(
                f"Cannot connect to MongoDB at {target_uri} and mongomock is not installed."
            ) from exc


def get_mongo_database(db_name: Optional[str] = None) -> Database:
    """Returns the application MongoDB Database instance."""
    global _is_seeding
    settings = get_settings()
    target_db_name = db_name or settings.mongodb_db_name
    client = get_mongo_client()
    db = client[target_db_name]
    if _is_mock and not _is_seeding:
        _is_seeding = True
        try:
            from .schema import CRICKET_EVENTS_COLLECTION
            if db[CRICKET_EVENTS_COLLECTION].count_documents({}) == 0 and settings.dataset_path.exists():
                from .ingest_mongo import run_full_mongo_ingestion
                run_full_mongo_ingestion()
        except Exception as e:
            logger.warning("Auto-seed mongomock encountered: %s", e)
        finally:
            _is_seeding = False
    return db


def is_using_mock_mongo() -> bool:
    """Returns True if the current connection is using mongomock."""
    global _is_mock
    return _is_mock


def close_mongo_connection() -> None:
    """Closes the current MongoDB connection and resets singleton."""
    global _mongo_client, _is_mock
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
        _mongo_client = None
        _is_mock = False
