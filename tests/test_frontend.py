"""
tests/test_frontend.py
======================
Test suite for Module 9: Streamlit Frontend Communication Layer.
Verifies that the FrontendAPIService communicates with Django REST Framework APIs,
retrieves valid response schemas, and preserves separation of concerns.
"""

import os
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
import django
django.setup()
from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.indexer import run_full_qdrant_indexing
from frontend.api_service import FrontendAPIService


@pytest.fixture(scope="module", autouse=True)
def setup_frontend_environment():
    """Seeds database and vector store before running frontend service tests."""
    db = get_mongo_database()
    valid_records, _, _ = run_ingestion_pipeline()
    ingest_records_to_mongo(valid_records, db=db)
    try:
        run_full_qdrant_indexing(recreate=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def mock_httpx_if_offline(monkeypatch):
    """If live Django server is not actively bound on port 8000, routes in-process via DRF APIClient."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 8000))
    sock.close()
    if result != 0:
        from rest_framework.test import APIClient
        api_client = APIClient()

        class MockResponse:
            def __init__(self, data, status_code=200):
                self._data = data
                self.status_code = status_code

            def json(self):
                return self._data

        class MockHTTPXClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get(self, url):
                path = url.split("8000")[-1]
                resp = api_client.get(path)
                return MockResponse(resp.data, resp.status_code)

            def post(self, url, json=None):
                path = url.split("8000")[-1]
                resp = api_client.post(path, data=json, format="json")
                return MockResponse(resp.data, resp.status_code)

        monkeypatch.setattr("httpx.Client", MockHTTPXClient)


class TestFrontendAPIService:
    """Verifies that FrontendAPIService interacts with backend endpoints correctly."""

    def test_service_health_check(self):
        service = FrontendAPIService()
        health = service.check_health()
        assert "status" in health
        assert health["status"] == "healthy"

    def test_service_chat_delegation(self):
        service = FrontendAPIService()
        resp = service.ask_chat("How many runs did Team B score in MATCH001?")
        assert "answer" in resp
        assert resp["intent"] == "exact_structured"
        assert "MATCH001" in resp["answer"]

    def test_service_chat_semantic_delegation(self):
        service = FrontendAPIService()
        resp = service.ask_chat("LBW decisions pending review", limit=3)
        assert "answer" in resp
        assert "cited_evidence" in resp

    def test_service_chat_superlative_query(self):
        service = FrontendAPIService()
        resp = service.ask_chat("what is the Player 4 (Team A) highest score !")
        assert "answer" in resp
        assert resp["intent"] == "exact_structured"
        assert "9 runs" in resp["answer"]
        assert "MATCH002" in resp["answer"]
