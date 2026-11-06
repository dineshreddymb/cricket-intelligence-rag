"""
tests/test_mcp.py
=================
Test suite for Module 8: Model Context Protocol (MCP) server and tools.
Verifies tool definitions, argument serialization, and communication with Django REST APIs.
"""

import json
import pytest

from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.indexer import run_full_qdrant_indexing
from mcp_server.server import (
    search_cricket_logs,
    query_cricket_stats,
    ask_cricket_assistant,
    get_match_summary,
    api_client,
)


@pytest.fixture(scope="module", autouse=True)
def setup_mcp_environment():
    """Seeds database and vector store before running MCP tests."""
    db = get_mongo_database()
    valid_records, _, _ = run_ingestion_pipeline()
    ingest_records_to_mongo(valid_records, db=db)
    try:
        run_full_qdrant_indexing(recreate=False)
    except Exception:
        pass


class TestMCPTools:
    """Verifies that all MCP tools execute successfully and return valid JSON."""

    def test_mcp_api_client_health(self):
        health = api_client.health_check()
        assert "status" in health
        assert health["status"] == "healthy"

    def test_search_cricket_logs_tool(self):
        output = search_cricket_logs(query="LBW given pending review", limit=3)
        data = json.loads(output)
        assert "results" in data
        assert len(data["results"]) == 3
        assert "event_id" in data["results"][0]

    def test_query_cricket_stats_tool(self):
        output = query_cricket_stats(match_id="MATCH001", team="Team B")
        data = json.loads(output)
        assert data["type"] == "team_runs"
        assert data["data"]["team"] == "Team B"
        assert data["data"]["total_runs"] > 0

    def test_get_match_summary_tool(self):
        output = get_match_summary(match_id="MATCH001")
        data = json.loads(output)
        assert data["type"] == "match_summary"
        assert data["data"]["total_events"] == 50
        assert data["data"]["total_runs"] > 0

    def test_ask_cricket_assistant_tool(self):
        output = ask_cricket_assistant(question="How many runs did Team B score in MATCH001?")
        data = json.loads(output)
        assert "answer" in data
        assert "intent" in data
        assert "MATCH001" in data["answer"]

    def test_mcp_hybrid_search_extras(self):
        output = search_cricket_logs(query="deliveries that resulted in extra runs", event="extras", limit=3)
        data = json.loads(output)
        assert "results" in data
        assert len(data["results"]) > 0
        for item in data["results"]:
            assert item["event"] in ["wide", "no ball"]
