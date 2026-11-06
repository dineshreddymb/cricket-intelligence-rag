"""
vector_store/qdrant_manager.py
==============================
Connection manager and collection lifecycle controller for Qdrant vector database.
Supports dual-mode execution:
1. Live Qdrant server connection via HTTP (for Podman/Docker environment)
2. Embedded local disk/in-memory Qdrant instance (for zero-dependency offline dev & testing)
"""

import logging
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType

from config.settings import get_settings

logger = logging.getLogger(__name__)

_qdrant_client: Optional[QdrantClient] = None
_is_local_qdrant: bool = False


def get_qdrant_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    local_path: Optional[str] = None,
    timeout: float = 1.5,
) -> QdrantClient:
    """
    Returns an active QdrantClient.
    Attempts network connection to Qdrant server; falls back seamlessly to
    embedded local disk storage if server is unreachable.
    """
    global _qdrant_client, _is_local_qdrant

    if _qdrant_client is not None:
        return _qdrant_client

    settings = get_settings()
    target_host = host or settings.qdrant_host
    target_port = port or settings.qdrant_port

    # 1. Attempt connection to live Qdrant server (HTTP)
    try:
        client = QdrantClient(host=target_host, port=target_port, timeout=timeout)
        # Verify connection by listing collections
        client.get_collections()
        logger.info("Connected successfully to Qdrant server at %s:%s", target_host, target_port)
        _qdrant_client = client
        _is_local_qdrant = False
        return _qdrant_client
    except Exception as exc:
        logger.warning(
            "Live Qdrant server not accessible at %s:%s (%s). Falling back to embedded local storage.",
            target_host,
            target_port,
            exc,
        )

    # 2. Fallback to embedded local Qdrant engine
    storage_path = local_path or settings.qdrant_local_path
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    logger.info("Initializing embedded Qdrant at %s", storage_path)
    try:
        client = QdrantClient(path=storage_path)
        _qdrant_client = client
        _is_local_qdrant = True
        return _qdrant_client
    except Exception as exc:
        logger.warning(
            "Local storage folder %s locked by another process (%s). Falling back to temporary in-memory Qdrant.",
            storage_path,
            exc,
        )
        return QdrantClient(":memory:")


def is_using_local_qdrant() -> bool:
    """Returns True if the current client is using local embedded Qdrant."""
    global _is_local_qdrant
    return _is_local_qdrant


def reset_qdrant_client() -> None:
    """Resets the singleton client instance (useful for test isolation)."""
    global _qdrant_client, _is_local_qdrant
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None
        _is_local_qdrant = False


def ensure_cricket_collection(
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
    dimension: int = 384,
    recreate: bool = False,
) -> str:
    """
    Ensures that the target vector collection exists with the required schema and payload indexes.
    """
    target_client = client or get_qdrant_client()
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name

    existing_collections = [c.name for c in target_client.get_collections().collections]

    if recreate and name in existing_collections:
        logger.info("Recreating Qdrant collection: %s", name)
        target_client.delete_collection(name)
        existing_collections.remove(name)

    if name not in existing_collections:
        logger.info("Creating Qdrant collection '%s' (dim=%d, metric=Cosine)", name, dimension)
        target_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

        # Create payload indexes on remote server Qdrant (local Qdrant does not require explicit payload indexes)
        if not is_using_local_qdrant():
            for field_name in ["match_id", "team", "player", "event", "status", "event_id"]:
                try:
                    target_client.create_payload_index(
                        collection_name=name,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    logger.debug("Payload index creation notice for %s: %s", field_name, e)

    return name
