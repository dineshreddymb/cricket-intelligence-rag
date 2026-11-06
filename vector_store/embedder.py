"""
vector_store/embedder.py
========================
Dense vector embedding generation using SentenceTransformers.
Model: all-MiniLM-L6-v2 (384 dimensions, Cosine distance).
Provides cached model loading and efficient batch vectorization.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = logging.getLogger(__name__)

_embedder_instance: "EmbeddingModel | None" = None


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating dense text vectors."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        logger.info("Loading SentenceTransformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded successfully. Embedding dimension: %d", self.dimension)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Encodes a list of text strings into normalized float vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encodes a single user query string into a normalized float vector."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


def get_embedder() -> EmbeddingModel:
    """Singleton getter for the EmbeddingModel instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = EmbeddingModel()
    return _embedder_instance
