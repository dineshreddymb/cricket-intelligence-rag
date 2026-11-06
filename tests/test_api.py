"""
tests/test_api.py
=================
Test suite for Module 7: Django REST Framework APIs for Grounded Q&A Assistant.
Uses DRF APIClient to verify HTTP status codes, request serialization,
response schemas, and orchestration across all active API endpoints.
"""

import os
import django

# Configure Django settings before importing rest_framework
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
django.setup()

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.indexer import run_full_qdrant_indexing


@pytest.fixture(scope="module")
def api_test_client():
    """Initializes and seeds database/vector store, then provides APIClient."""
    import django
    django.setup()

    db = get_mongo_database()
    valid_records, _, _ = run_ingestion_pipeline()
    ingest_records_to_mongo(valid_records, db=db)
    run_full_qdrant_indexing(recreate=False)

    client = APIClient()
    return client


class TestHealthEndpoint:
    """Verifies GET /api/v1/health/"""

    def test_health_check_returns_200(self, api_test_client):
        response = api_test_client.get("/api/v1/health/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "mongodb" in data
        assert "qdrant" in data
        assert "dataset" in data
        assert data["dataset"]["present"] is True


class TestChatRAGAPI:
    """Verifies POST /api/v1/chat/"""

    def test_chat_exact_structured_query(self, api_test_client):
        payload = {"query": "How many runs did Team B score in MATCH001?"}
        response = api_test_client.post("/api/v1/chat/", payload, format="json")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert data["intent"] == "exact_structured"
        assert "MATCH001" in data["answer"]
        assert "Team B" in data["answer"]

    def test_chat_superlative_player_score(self, api_test_client):
        payload = {"query": "what is the Player 4 (Team A) highest score !"}
        response = api_test_client.post("/api/v1/chat/", payload, format="json")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["intent"] == "exact_structured"
        assert data["model_used"] == "MongoDB Aggregation Engine"
        assert "9 runs" in data["answer"]
        assert "MATCH002" in data["answer"]
        assert "EVT_0059" in data["cited_evidence"]
        assert "EVT_0075" in data["cited_evidence"]

    def test_chat_century_inquiry(self, api_test_client):
        payload = {"query": "Which player scored the highest century in the tournament, and in which match did it happen?"}
        response = api_test_client.post("/api/v1/chat/", payload, format="json")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["intent"] == "exact_structured"
        assert data["model_used"] == "MongoDB Aggregation Engine"
        assert "No player scored a century" in data["answer"]

    def test_chat_semantic_rag_query(self, api_test_client):
        payload = {"query": "What events occurred during overcast conditions?", "limit": 3}
        response = api_test_client.post("/api/v1/chat/", payload, format="json")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["answer"]) > 0
        assert len(data["cited_evidence"]) > 0
        assert "retrieval_scope" in data
        assert "limitations" in data


class TestRecordsListAPI:
    """Verifies GET /api/v1/records/"""

    def test_get_paginated_records(self, api_test_client):
        response = api_test_client.get("/api/v1/records/?limit=5&skip=0")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_matching"] == 500
        assert data["returned_count"] == 5
        assert len(data["records"]) == 5
