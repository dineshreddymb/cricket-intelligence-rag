"""
frontend/api_service.py
=======================
Frontend communication layer that exclusively interfaces with the Django REST API over HTTP.
Ensures the UI does not bypass backend business logic or access databases directly.
"""

import logging
from typing import Dict, Any, Optional
import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


class FrontendAPIService:
    """Service client for Streamlit frontend calling Django REST APIs exclusively over HTTP."""

    def __init__(self, api_base_url: Optional[str] = None):
        settings = get_settings()
        self.base_url = (api_base_url or settings.api_base_url or "http://127.0.0.1:8000").rstrip("/")
        self.timeout = 45.0

    def check_health(self) -> Dict[str, Any]:
        """Checks API health and database connectivity."""
        url = f"{self.base_url}/api/v1/health/"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                return resp.json()
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {
                "status": "degraded",
                "error": f"Cannot reach Django backend at {url}: {e}",
                "mongodb": {"status": "unreachable", "records_stored": 0},
                "qdrant": {"status": "unreachable", "collections": []},
            }

    def ask_chat(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Submits user question to POST /api/v1/chat/"""
        url = f"{self.base_url}/api/v1/chat/"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json={"query": query, "limit": limit})
                return resp.json()
        except Exception as e:
            logger.error("Chat call failed: %s", e)
            return {
                "answer": f"Backend communication error: {e}. Is Django running on {self.base_url}?",
                "intent": "error",
                "cited_evidence": [],
                "model_used": "N/A",
                "raw_evidence": [],
            }
