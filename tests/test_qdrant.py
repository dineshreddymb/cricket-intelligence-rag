"""
tests/test_qdrant.py
====================
Test suite for Module 4: Embeddings and Qdrant Vector Storage.
Verifies text representation, SentenceTransformer embedding dimensions,
collection lifecycle, batch indexing of 500 records, and MongoDB ID mapping.
"""

import pytest
from qdrant_client import QdrantClient

from vector_store.embedder import get_embedder
from vector_store.text_repr import create_record_text_representation
from vector_store.qdrant_manager import (
    get_qdrant_client,
    ensure_cricket_collection,
    reset_qdrant_client,
)
from vector_store.indexer import index_cricket_records, get_qdrant_collection_info
from ingestion.pipeline import run_ingestion_pipeline
from ingestion.cleaner import CleanedCricketRecord


@pytest.fixture(scope="module")
def qdrant_test_setup():
    """Initializes in-memory Qdrant client and indexes the full dataset."""
    # Use in-memory Qdrant instance for fast, isolated test execution
    client = QdrantClient(":memory:")
    embedder = get_embedder()
    test_collection = "test_cricket_events"

    valid_records, _, _ = run_ingestion_pipeline()
    index_result = index_cricket_records(
        records=valid_records,
        client=client,
        embedder=embedder,
        collection_name=test_collection,
        recreate=True,
    )

    yield {
        "client": client,
        "embedder": embedder,
        "collection": test_collection,
        "index_result": index_result,
        "records": valid_records,
    }


class TestTextRepresentationAndEmbedder:
    """Verifies dense representation construction and vector generation."""

    def test_text_representation_structure(self):
        sample = {
            "match_id": "MATCH001",
            "team": "Team B",
            "player": "Player 9",
            "event": "LBW given",
            "runs": 5,
            "over": 37,
            "status": "pending review",
            "commentary": "Player 9 () performed LBW given.",
            "player_role": "all-rounder",
            "venue": "Stadium 2",
            "weather_condition": "overcast",
        }
        text = create_record_text_representation(sample)
        assert "Match: MATCH001" in text
        assert "Team: Team B" in text
        assert "Player: Player 9" in text
        assert "Role: all-rounder" in text
        assert "Venue: Stadium 2" in text
        assert "Weather: overcast" in text

    def test_embedder_dimension_is_384(self):
        embedder = get_embedder()
        assert embedder.dimension == 384
        vec = embedder.embed_query("Who hit a boundary in over 10?")
        assert len(vec) == 384
        assert isinstance(vec[0], float)


class TestQdrantIndexingAndMapping:
    """Verifies vector storage, collection parameters, and MongoDB ID correlation."""

    def test_collection_contains_500_vectors(self, qdrant_test_setup):
        client = qdrant_test_setup["client"]
        col = qdrant_test_setup["collection"]
        info = get_qdrant_collection_info(client=client, collection_name=col)

        assert info["points_count"] == 500
        assert info["vector_size"] == 384
        assert "Cosine" in info["distance"]

    def test_qdrant_point_maps_to_mongodb_event_id(self, qdrant_test_setup):
        client = qdrant_test_setup["client"]
        col = qdrant_test_setup["collection"]

        # Retrieve first point (ID=1)
        points = client.retrieve(collection_name=col, ids=[1])
        assert len(points) == 1
        payload = points[0].payload

        assert payload["event_id"] == "EVT_0001"
        assert payload["line_number"] == 1
        assert payload["match_id"] == "MATCH001"
        assert payload["team"] == "Team B"
        assert payload["event"] == "LBW given"

    def test_qdrant_vector_search_round_trip(self, qdrant_test_setup):
        client = qdrant_test_setup["client"]
        embedder = qdrant_test_setup["embedder"]
        col = qdrant_test_setup["collection"]

        query = "LBW given pending review over 37"
        query_vec = embedder.embed_query(query)

        response = client.query_points(
            collection_name=col,
            query=query_vec,
            limit=3,
        )
        results = response.points

        assert len(results) > 0
        top_match = results[0]
        assert top_match.score > 0.0
        assert "event_id" in top_match.payload
        # First point EVT_0001 should be among top results for this specific text
        top_ids = [r.payload["event_id"] for r in results]
        assert "EVT_0001" in top_ids
