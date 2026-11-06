"""
database/__init__.py
MongoDB connection and structured data management package.
"""
from .mongo_client import get_mongo_database, get_mongo_client, close_mongo_connection
from .schema import get_cricket_event_validator, CRICKET_EVENTS_COLLECTION
from .ingest_mongo import ingest_records_to_mongo
from .queries import (
    get_match_summary,
    get_team_runs,
    get_player_statistics,
    get_event_counts_by_status,
    filter_events,
)

__all__ = [
    "get_mongo_database",
    "get_mongo_client",
    "close_mongo_connection",
    "get_cricket_event_validator",
    "CRICKET_EVENTS_COLLECTION",
    "ingest_records_to_mongo",
    "get_match_summary",
    "get_team_runs",
    "get_player_statistics",
    "get_event_counts_by_status",
    "filter_events",
]
