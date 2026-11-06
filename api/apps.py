"""
api/apps.py
Django application configuration.
"""
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    verbose_name = 'Cricket Semantic Search API'

    def ready(self):
        """Auto-populates in-memory MongoDB on startup if database has 0 records."""
        import os
        import sys

        # Execute only in worker process or when running without reloader
        if os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv or "runserver" in sys.argv:
            try:
                from database.mongo_client import get_mongo_database
                from database.ingest_mongo import run_full_mongo_ingestion
                db = get_mongo_database()
                if db["cricket_events"].count_documents({}) == 0:
                    run_full_mongo_ingestion()
            except Exception:
                pass
