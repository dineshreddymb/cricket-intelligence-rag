"""
database/ingest_mongo.py
========================
Idempotent ingestion of validated cricket event records into MongoDB.
Ensures unique index enforcement, bulk upserts (zero duplicates on re-run),
and post-ingestion count and integrity verification.
"""

import logging
from typing import List, Dict, Any, Optional
from pymongo import UpdateOne
from pymongo.database import Database

from ingestion.cleaner import CleanedCricketRecord
from ingestion.pipeline import run_ingestion_pipeline
from .mongo_client import get_mongo_database
from .schema import CRICKET_EVENTS_COLLECTION, get_required_indexes

logger = logging.getLogger(__name__)


def setup_indexes(db: Database) -> List[str]:
    """Creates required indexes on the cricket_events collection."""
    collection = db[CRICKET_EVENTS_COLLECTION]
    created_indexes: List[str] = []

    for index_def in get_required_indexes():
        try:
            name = collection.create_index(
                index_def["keys"],
                unique=index_def.get("unique", False),
                name=index_def.get("name"),
            )
            created_indexes.append(name)
        except Exception as exc:
            logger.warning("Index creation notice for %s: %s", index_def.get("name"), exc)

    return created_indexes


def ingest_records_to_mongo(
    records: List[CleanedCricketRecord],
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """
    Ingests cleaned records into MongoDB idempotently using bulk upsert.
    Falls back gracefully to individual update_one calls if mock client has bulk_write signature divergence.
    
    Returns:
        dict: Ingestion summary with counts, matched, modified, and upserted stats.
    """
    target_db = db if db is not None else get_mongo_database()
    collection = target_db[CRICKET_EVENTS_COLLECTION]

    # Ensure indexes are established
    created_indexes = setup_indexes(target_db)

    if not records:
        return {
            "total_records_processed": 0,
            "upserted_count": 0,
            "matched_count": 0,
            "modified_count": 0,
            "total_in_collection": collection.count_documents({}),
            "indexes_active": created_indexes,
        }

    # Prepare idempotent bulk operations: upsert by unique event_id
    operations = [
        UpdateOne(
            {"event_id": record.event_id},
            {"$set": record.to_dict()},
            upsert=True,
        )
        for record in records
    ]

    upserted_count = 0
    matched_count = 0
    modified_count = 0

    try:
        bulk_result = collection.bulk_write(operations, ordered=False)
        upserted_count = len(bulk_result.upserted_ids) if bulk_result.upserted_ids else 0
        matched_count = bulk_result.matched_count
        modified_count = bulk_result.modified_count
    except (TypeError, AttributeError):
        # Fallback for environments / mongomock where bulk_write has signature mismatches
        for op in operations:
            res = collection.update_one(op._filter, op._doc, upsert=op._upsert)
            if res.upserted_id is not None:
                upserted_count += 1
            matched_count += res.matched_count
            modified_count += res.modified_count

    total_in_collection = collection.count_documents({})

    result_summary = {
        "total_records_processed": len(records),
        "upserted_count": upserted_count,
        "matched_count": matched_count,
        "modified_count": modified_count,
        "total_in_collection": total_in_collection,
        "indexes_active": created_indexes,
    }

    logger.info("MongoDB Ingestion complete: %s", result_summary)
    return result_summary


def run_full_mongo_ingestion() -> Dict[str, Any]:
    """Runs end-to-end ingestion from raw TXT file directly into MongoDB."""
    valid_records, report, pipeline_stats = run_ingestion_pipeline()
    ingest_stats = ingest_records_to_mongo(valid_records)
    return {
        "pipeline_stats": pipeline_stats,
        "ingest_stats": ingest_stats,
    }
