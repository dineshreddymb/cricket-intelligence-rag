"""
api/views.py
============
Django REST Framework API views coordinating Grounded Q&A Assistant,
authoritative log browsing, ingestion, and system health monitoring.
Delegates business logic to rag, retrieval, and database modules.
"""

import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .serializers import (
    ChatRequestSerializer,
    IngestionTriggerSerializer,
)
from rag.pipeline import answer_question
from database.queries import filter_events
from database.mongo_client import get_mongo_database, is_using_mock_mongo
from database.ingest_mongo import run_full_mongo_ingestion
from vector_store.qdrant_manager import get_qdrant_client, is_using_local_qdrant
from vector_store.indexer import run_full_qdrant_indexing
from config.settings import get_settings

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """
    GET /api/v1/health/
    Verifies operational status of MongoDB, Qdrant, Embedding model, and dataset.
    """

    def get(self, request):
        settings = get_settings()

        # Check MongoDB
        try:
            db = get_mongo_database()
            mongo_count = db["cricket_events"].count_documents({})
            mongo_status = "connected (mongomock)" if is_using_mock_mongo() else "connected (live)"
        except Exception as e:
            mongo_status = f"unhealthy: {str(e)}"
            mongo_count = 0

        # Check Qdrant
        try:
            q_client = get_qdrant_client()
            cols = [c.name for c in q_client.get_collections().collections]
            if settings.qdrant_collection_name not in cols:
                from vector_store.qdrant_manager import reset_qdrant_client
                reset_qdrant_client()
                q_client = get_qdrant_client()
                cols = [c.name for c in q_client.get_collections().collections]
            qdrant_status = "connected (local/embedded)" if is_using_local_qdrant() else "connected (server)"
        except Exception as e:
            qdrant_status = f"unhealthy: {str(e)}"
            cols = []

        # Check dataset file
        dataset_exists = settings.dataset_path.exists()
        dataset_size = settings.dataset_path.stat().st_size if dataset_exists else 0

        is_healthy = dataset_exists and "unhealthy" not in mongo_status and "unhealthy" not in qdrant_status

        return Response(
            {
                "status": "healthy" if is_healthy else "degraded",
                "mongodb": {
                    "status": mongo_status,
                    "records_stored": mongo_count,
                },
                "qdrant": {
                    "status": qdrant_status,
                    "collections": cols,
                },
                "dataset": {
                    "present": dataset_exists,
                    "path": str(settings.dataset_path.resolve()),
                    "size_bytes": dataset_size,
                },
                "embedding_model": settings.embedding_model_name,
                "llm_model": settings.gemini_model,
            },
            status=status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ChatRAGView(APIView):
    """
    POST /api/v1/chat/
    Unified conversational entrypoint for Grounded Q&A Assistant.
    Intelligently routes between exact MongoDB calculations and Qdrant + Gemini grounded RAG.
    """

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": "Validation Error", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = serializer.validated_data
        response = answer_question(
            query=data["query"],
            limit=data["limit"],
            override_intent=data.get("override_intent"),
        )

        return Response(response.to_dict(), status=status.HTTP_200_OK)


class RecordsListView(APIView):
    """
    GET /api/v1/records/
    Paginated browser for authoritative log records with filtering.
    """

    def get(self, request):
        match_id = request.query_params.get("match_id")
        team = request.query_params.get("team")
        player = request.query_params.get("player")
        event = request.query_params.get("event")
        status_param = request.query_params.get("status")
        limit = int(request.query_params.get("limit", 50))
        skip = int(request.query_params.get("skip", 0))

        result = filter_events(
            match_id=match_id,
            team=team,
            player=player,
            event=event,
            status=status_param,
            limit=limit,
            skip=skip,
        )
        return Response(result, status=status.HTTP_200_OK)


class TriggerIngestionView(APIView):
    """
    POST /api/v1/ingest/
    Triggers ingestion from cricket_log.txt into MongoDB and Qdrant.
    """

    def post(self, request):
        serializer = IngestionTriggerSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        recreate_qdrant = serializer.validated_data.get("recreate_qdrant", False)

        try:
            mongo_res = run_full_mongo_ingestion()
            qdrant_res = run_full_qdrant_indexing(recreate=recreate_qdrant)

            return Response(
                {
                    "status": "success",
                    "mongo_ingestion": mongo_res,
                    "qdrant_indexing": qdrant_res,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            return Response(
                {"status": "error", "message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
