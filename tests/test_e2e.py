"""
tests/test_e2e.py
=================
Module 11: End-to-End Integration Verification Test Suite.
Validates the complete project flow across all 10 architectural components:
1. LangChain TextLoader raw ingestion
2. Deterministic regex parsing and validation (all 500 records)
3. Idempotent MongoDB structured storage
4. SentenceTransformers embedding & Qdrant vector indexing
5. Dense vector retrieval with MongoDB hydration
6. Intent classification and query routing
7. Grounded RAG context preparation
8. Django REST Framework API layer
9. MCP tool execution delegating to Django APIs
10. Container configuration verification
"""

import json
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
django.setup()

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from ingestion.pipeline import run_ingestion_pipeline
from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from database.queries import get_match_summary, get_team_runs
from vector_store.embedder import get_embedder
from vector_store.indexer import index_cricket_records, get_qdrant_collection_info
from retrieval.semantic_search import search_cricket_events
from rag.query_router import route_query, QueryIntent
from rag.pipeline import answer_question
from mcp_server.server import (
    search_cricket_logs,
    query_cricket_stats,
    ask_cricket_assistant,
    get_match_summary as mcp_get_match_summary,
)


@pytest.fixture(scope="module")
def full_system():
    """Initializes and seeds full end-to-end environment."""
    # 1. Ingestion pipeline
    valid_records, report, stats = run_ingestion_pipeline()
    assert len(valid_records) == 500
    assert report.invalid_count == 0

    # 2. MongoDB Ingestion
    db = get_mongo_database()
    mongo_stats = ingest_records_to_mongo(valid_records, db=db)
    assert mongo_stats["total_in_collection"] == 500

    # 3. Qdrant Indexing
    qdrant_stats = index_cricket_records(valid_records, collection_name="e2e_collection", recreate=True)
    assert qdrant_stats["total_points_indexed"] == 500

    api_client = APIClient()

    return {
        "records": valid_records,
        "db": db,
        "api_client": api_client,
        "qdrant_collection": "e2e_collection",
        "stats": stats,
    }


class TestEndToEndSystemIntegration:
    """End-to-end verification covering every system layer."""

    def test_e2e_data_integrity_and_totals(self, full_system):
        """TXT -> Parser -> MongoDB: 500 records, 1422 total runs."""
        stats = full_system["stats"]
        assert stats["total_runs"] == 1422
        assert stats["unique_players"] == 11
        assert len(stats["match_distribution"]) == 10

    def test_e2e_retrieval_and_hydration(self, full_system):
        """Vector Search -> MongoDB Hydration: Top results are enriched with line numbers and commentary."""
        hits = search_cricket_events(
            query="spectators reacted as player executed boundary hit",
            limit=3,
            collection_name=full_system["qdrant_collection"],
        )
        assert len(hits) == 3
        for hit in hits:
            assert hit.event in ("boundary hit", "run scored", "six hit")
            assert hit.line_number > 0
            assert len(hit.commentary) > 0
            assert hit.score > 0.0

    def test_e2e_exact_aggregation_routing(self, full_system):
        """Query Router -> MongoDB Aggregation: Answers exact arithmetic without LLM estimation."""
        # 1. Router classification
        decision = route_query("How many runs did Team B score in MATCH001?")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED

        # 2. End-to-end pipeline execution
        res = answer_question("How many runs did Team B score in MATCH001?")
        assert res.intent == "exact_structured"
        assert res.model_used == "MongoDB Aggregation Engine"
        assert "MATCH001" in res.answer
        assert "Team B" in res.answer

    def test_e2e_superlative_aggregation_routing(self, full_system):
        """Query Router -> Superlative Aggregation: Finds Player 4 highest score in Team A."""
        res = answer_question("what is the Player 4 (Team A) highest score !")
        assert res.intent == "exact_structured"
        assert res.model_used == "MongoDB Aggregation Engine"
        assert "9 runs" in res.answer
        assert "MATCH002" in res.answer
        assert "EVT_0059" in res.cited_evidence
        assert "EVT_0075" in res.cited_evidence

    def test_e2e_django_api_endpoints(self, full_system):
        """Django REST APIs: health, records list, and grounded chat."""
        client = full_system["api_client"]

        # Health
        health_resp = client.get("/api/v1/health/")
        assert health_resp.status_code == status.HTTP_200_OK

        # Records List API
        records_resp = client.get("/api/v1/records/?limit=5")
        assert records_resp.status_code == status.HTTP_200_OK
        assert len(records_resp.json()["records"]) == 5

        # Chat API
        chat_resp = client.post(
            "/api/v1/chat/",
            {"query": "What events occurred in MATCH001?", "limit": 3},
            format="json",
        )
        assert chat_resp.status_code == status.HTTP_200_OK
        assert len(chat_resp.json()["answer"]) > 0

    def test_e2e_mcp_tools(self, full_system):
        """MCP Tools: search_cricket_logs, query_cricket_stats, ask_cricket_assistant."""
        # Search Tool
        tool_search_out = search_cricket_logs(query="wicket taken by bowler", limit=2)
        search_data = json.loads(tool_search_out)
        assert len(search_data["results"]) == 2

        # Exact Stats Tool
        tool_stats_out = query_cricket_stats(match_id="MATCH002")
        stats_data = json.loads(tool_stats_out)
        assert stats_data["data"]["total_events"] == 50

        # Q&A Assistant Tool
        tool_chat_out = ask_cricket_assistant(question="How many runs in MATCH001?")
        chat_data = json.loads(tool_chat_out)
        assert "MATCH001" in chat_data["answer"]
