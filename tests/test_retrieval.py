"""
tests/test_retrieval.py
=======================
Test suite for Module 5: Retrieval Layer and Quality Evaluation.
Verifies dense vector search, MongoDB hydration, metadata filtering,
and quantitative benchmark evaluation (MRR and Precision@K).
"""

import pytest
from qdrant_client import QdrantClient

from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.embedder import get_embedder
from vector_store.indexer import index_cricket_records
from retrieval.semantic_search import search_cricket_events
from retrieval.evaluator import evaluate_retrieval_system


@pytest.fixture(scope="module")
def retrieval_environment():
    """Seeds both MongoDB (in-memory/mock) and Qdrant with all 500 records."""
    db = get_mongo_database()
    valid_records, _, _ = run_ingestion_pipeline()
    ingest_records_to_mongo(valid_records, db=db)

    client = QdrantClient(":memory:")
    embedder = get_embedder()
    collection_name = "eval_cricket_events"

    index_cricket_records(
        records=valid_records,
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        recreate=True,
    )

    yield {
        "client": client,
        "embedder": embedder,
        "collection": collection_name,
    }


class TestSemanticRetrieval:
    """Verifies vector search and MongoDB hydration."""

    def test_semantic_search_unfiltered(self, retrieval_environment):
        client = retrieval_environment["client"]
        embedder = retrieval_environment["embedder"]
        col = retrieval_environment["collection"]

        hits = search_cricket_events(
            query="LBW given decision pending review",
            limit=5,
            client=client,
            embedder=embedder,
            collection_name=col,
        )

        assert len(hits) == 5
        top_hit = hits[0]
        assert top_hit.event == "LBW given"
        assert top_hit.status == "pending review"
        assert top_hit.score > 0.5
        # Verify MongoDB hydration: fields like line_number, raw_source_line are populated
        assert top_hit.line_number > 0
        assert len(top_hit.commentary) > 0

    def test_semantic_search_with_match_filter(self, retrieval_environment):
        client = retrieval_environment["client"]
        embedder = retrieval_environment["embedder"]
        col = retrieval_environment["collection"]

        hits = search_cricket_events(
            query="boundary hit",
            match_id="MATCH002",
            limit=5,
            client=client,
            embedder=embedder,
            collection_name=col,
        )

        assert len(hits) > 0
        for hit in hits:
            assert hit.match_id == "MATCH002"

    def test_semantic_search_with_team_filter(self, retrieval_environment):
        client = retrieval_environment["client"]
        embedder = retrieval_environment["embedder"]
        col = retrieval_environment["collection"]

        hits = search_cricket_events(
            query="wicket taken",
            team="Team A",
            limit=5,
            client=client,
            embedder=embedder,
            collection_name=col,
        )

        assert len(hits) > 0
        for hit in hits:
            assert hit.team == "Team A"


class TestRetrievalEvaluationBenchmark:
    """Runs automated benchmark queries to evaluate retrieval performance."""

    def test_retrieval_benchmark_metrics(self, retrieval_environment):
        client = retrieval_environment["client"]
        embedder = retrieval_environment["embedder"]
        col = retrieval_environment["collection"]

        report = evaluate_retrieval_system(
            collection_name=col,
            client=client,
            embedder=embedder,
            top_k=5,
        )

        assert report.total_queries == 10
        assert report.mean_reciprocal_rank >= 0.70, (
            f"Expected MRR >= 0.70, observed {report.mean_reciprocal_rank}"
        )
        assert len(report.query_results) == 10
        assert len(report.strategy_recommendation) > 0
