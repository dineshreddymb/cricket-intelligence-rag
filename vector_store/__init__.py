"""
vector_store/__init__.py
Vector embedding and Qdrant storage package.
"""
from .embedder import get_embedder, EmbeddingModel
from .text_repr import create_record_text_representation
from .qdrant_manager import get_qdrant_client, ensure_cricket_collection
from .indexer import index_cricket_records, get_qdrant_collection_info

__all__ = [
    "get_embedder",
    "EmbeddingModel",
    "create_record_text_representation",
    "get_qdrant_client",
    "ensure_cricket_collection",
    "index_cricket_records",
    "get_qdrant_collection_info",
]
