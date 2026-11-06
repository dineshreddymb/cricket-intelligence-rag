"""
api/urls.py
===========
URL routing for the Cricket Log Grounded Q&A Assistant API (v1).
"""

from django.urls import path
from .views import (
    HealthCheckView,
    ChatRAGView,
    RecordsListView,
    TriggerIngestionView,
)

app_name = "api"

urlpatterns = [
    path("health/", HealthCheckView.as_view(), name="health_check"),
    path("chat/", ChatRAGView.as_view(), name="chat_rag"),
    path("records/", RecordsListView.as_view(), name="records_list"),
    path("ingest/", TriggerIngestionView.as_view(), name="trigger_ingestion"),
]
