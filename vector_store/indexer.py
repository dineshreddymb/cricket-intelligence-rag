"""
vector_store/indexer.py
=======================
Batch vectorization and indexing of cricket log records into Qdrant.
Ensures 1:1 correspondence between Qdrant points and MongoDB authoritative documents
via unique event_id and line_number mapping.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from ingestion.cleaner import CleanedCricketRecord
from ingestion.pipeline import run_ingestion_pipeline
from .embedder import get_embedder, EmbeddingModel
from .text_repr import create_record_text_representation
from .qdrant_manager import get_qdrant_client, ensure_cricket_collection
from config.settings import get_settings

logger = logging.getLogger(__name__)


def index_cricket_records(
    records: List[CleanedCricketRecord],
    client: Optional[QdrantClient] = None,
    embedder: Optional[EmbeddingModel] = None,
    collection_name: Optional[str] = None,
    batch_size: int = 100,
    recreate: bool = False,
) -> Dict[str, Any]:
    """
    Transforms cleaned records into text representations, generates dense embeddings,
    and indexes them with metadata into Qdrant.
    
    Returns:
        dict: Indexing metrics including total indexed, collection name, and vector size.
    """
    target_client = client or get_qdrant_client()
    target_embedder = embedder or get_embedder()
    target_collection = collection_name or get_settings().qdrant_collection_name

    # Ensure collection exists
    ensure_cricket_collection(
        client=target_client,
        collection_name=target_collection,
        dimension=target_embedder.dimension,
        recreate=recreate,
    )

    # Optimization: If collection already has all points and recreate is False, skip embedding calculation
    if not recreate:
        try:
            col_info = target_client.get_collection(target_collection)
            if col_info.points_count >= len(records) and len(records) > 0:
                logger.info(
                    "Qdrant collection '%s' already contains %d points on SSD. Skipping embedding calculation.",
                    target_collection,
                    col_info.points_count,
                )
                return {
                    "total_records_processed": len(records),
                    "total_points_indexed": col_info.points_count,
                    "collection_name": target_collection,
                    "vector_dimension": target_embedder.dimension,
                    "skipped_reembedding": True,
                }
        except Exception as exc:
            logger.debug("Could not verify point count before embedding: %s", exc)

    if not records:
        return {
            "total_records_processed": 0,
            "total_points_indexed": 0,
            "collection_name": target_collection,
        }

    # Generate text representations
    texts = [create_record_text_representation(r) for r in records]

    # Batch embed texts
    logger.info("Generating embeddings for %d records...", len(texts))
    vectors = target_embedder.embed_texts(texts, batch_size=batch_size)

    # Prepare Qdrant PointStructs
    points: List[PointStruct] = []
    for idx, (record, vector, text_repr) in enumerate(zip(records, vectors, texts)):
        point = PointStruct(
            id=record.line_number,  # Deterministic point ID 1-500
            vector=vector,
            payload={
                "event_id": record.event_id,
                "line_number": record.line_number,
                "match_id": record.match_id,
                "team": record.team,
                "player": record.player,
                "event": record.event,
                "runs": record.runs,
                "over": record.over,
                "status": record.status,
                "commentary": record.commentary,
                "text_repr": text_repr,
                "player_role": record.player_role,
                "venue": record.venue,
                "weather_condition": record.weather_condition,
            },
        )
        points.append(point)

    # Batch upload points into Qdrant
    logger.info("Uploading %d points to Qdrant collection '%s'...", len(points), target_collection)
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        target_client.upsert(collection_name=target_collection, points=batch)

    # Verify collection info
    col_info = target_client.get_collection(target_collection)
    total_points = col_info.points_count

    summary = {
        "total_records_processed": len(records),
        "total_points_indexed": total_points,
        "collection_name": target_collection,
        "vector_dimension": target_embedder.dimension,
    }
    logger.info("Qdrant indexing complete: %s", summary)
    return summary


def get_qdrant_collection_info(
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves operational statistics for the cricket vector collection."""
    target_client = client or get_qdrant_client()
    name = collection_name or get_settings().qdrant_collection_name
    info = target_client.get_collection(name)
    return {
        "name": name,
        "points_count": info.points_count,
        "status": str(info.status),
        "vector_size": info.config.params.vectors.size,
        "distance": str(info.config.params.vectors.distance),
    }


def run_full_qdrant_indexing(
    recreate: bool = True,
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads raw dataset, parses, and indexes full 500 records into Qdrant."""
    valid_records, _, _ = run_ingestion_pipeline()
    return index_cricket_records(
        valid_records,
        client=client,
        collection_name=collection_name,
        recreate=recreate,
    )
