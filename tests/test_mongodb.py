"""
tests/test_mongodb.py
=====================
Test suite for Module 3: MongoDB structured storage, idempotent ingestion,
indexing, and exact analytical queries.
"""

import pytest

from database.mongo_client import get_mongo_database, close_mongo_connection
from database.ingest_mongo import run_full_mongo_ingestion, ingest_records_to_mongo
from database.schema import CRICKET_EVENTS_COLLECTION
from database.queries import (
    get_match_summary,
    get_team_runs,
    get_player_statistics,
    get_event_counts_by_status,
    get_records_by_ids,
    filter_events,
    get_player_highest_score,
    get_tournament_highest_innings,
    get_tournament_top_run_scorers,
    get_top_performers,
    get_tournament_top_wicket_takers,
    get_tournament_top_stumpings,
    get_tournament_top_dismissals,
    get_tournament_top_catches,
)
from ingestion.pipeline import run_ingestion_pipeline


@pytest.fixture(scope="module")
def populated_db():
    """Initializes and ingests all 500 records into the test database."""
    db = get_mongo_database()
    # Clear collection for clean test slate
    db[CRICKET_EVENTS_COLLECTION].delete_many({})
    
    valid_records, report, stats = run_ingestion_pipeline()
    ingest_result = ingest_records_to_mongo(valid_records, db=db)
    yield db
    # Teardown
    close_mongo_connection()


class TestMongoStorageAndIngestion:
    """Verifies collection setup, schema validation, and idempotent bulk ingestion."""

    def test_database_connection_established(self, populated_db):
        assert populated_db is not None
        collection = populated_db[CRICKET_EVENTS_COLLECTION]
        assert collection is not None

    def test_exact_500_records_stored(self, populated_db):
        collection = populated_db[CRICKET_EVENTS_COLLECTION]
        count = collection.count_documents({})
        assert count == 500, f"Expected 500 documents, found {count}"

    def test_idempotent_ingestion_no_duplicates(self, populated_db):
        """Re-ingesting the dataset must not create any duplicate records."""
        valid_records, _, _ = run_ingestion_pipeline()
        second_ingest = ingest_records_to_mongo(valid_records, db=populated_db)
        
        # Total in collection must strictly remain 500
        assert second_ingest["total_in_collection"] == 500
        # No new documents should be upserted
        assert second_ingest["upserted_count"] == 0
        # All 500 should be matched
        assert second_ingest["matched_count"] == 500


class TestAuthoritativeQueries:
    """Verifies deterministic analytical queries over MongoDB records."""

    def test_total_dataset_runs_equals_1422(self, populated_db):
        collection = populated_db[CRICKET_EVENTS_COLLECTION]
        pipeline = [{"$group": {"_id": None, "total": {"$sum": "$runs"}}}]
        result = list(collection.aggregate(pipeline))
        assert len(result) == 1
        assert result[0]["total"] == 1422

    def test_match_summary_aggregation(self, populated_db):
        summary = get_match_summary("MATCH001", db=populated_db)
        assert summary["found"] is True
        assert summary["total_events"] == 50
        assert summary["total_runs"] > 0
        assert len(summary["team_runs"]) > 0

    def test_team_runs_calculation(self, populated_db):
        res = get_team_runs("MATCH001", "Team B", db=populated_db)
        assert res["team"] == "Team B"
        assert res["event_count"] > 0
        assert res["total_runs"] >= 0

    def test_player_statistics(self, populated_db):
        stats = get_player_statistics("Player 9", db=populated_db)
        assert stats["found"] is True
        assert stats["total_events"] > 0
        assert stats["total_runs"] >= 0
        assert "LBW given" in stats["event_breakdown"]

    def test_filter_events(self, populated_db):
        filtered = filter_events(match_id="MATCH001", status="pending review", db=populated_db)
        assert filtered["total_matching"] > 0
        for doc in filtered["records"]:
            assert doc["match_id"] == "MATCH001"
            assert doc["status"] == "pending review"

    def test_hydrate_records_by_ids(self, populated_db):
        hydrated = get_records_by_ids(["EVT_0001", "EVT_0002"], db=populated_db)
        assert len(hydrated) == 2
        assert "EVT_0001" in hydrated
        assert "EVT_0002" in hydrated
        assert hydrated["EVT_0001"]["line_number"] == 1
        assert hydrated["EVT_0002"]["line_number"] == 2

    def test_player_highest_score_with_team(self, populated_db):
        res = get_player_highest_score("Player 4", team="Team A", db=populated_db)
        assert res["found"] is True
        assert res["player"] == "Player 4"
        assert res["team"] == "Team A"
        assert res["highest_score"] == 9
        assert res["highest_match_id"] == "MATCH002"
        assert res["total_runs"] == 34
        assert res["event_count"] == 14
        assert "EVT_0059" in res["highest_event_ids"]
        assert "EVT_0075" in res["highest_event_ids"]

    def test_player_highest_score_all_teams(self, populated_db):
        res = get_player_highest_score("Player 4", db=populated_db)
        assert res["found"] is True
        assert res["highest_score"] == 25
        assert res["highest_match_id"] == "MATCH005"
        assert res["total_runs"] == 143

    def test_player_highest_score_nonexistent_player(self, populated_db):
        res = get_player_highest_score("Player 999", db=populated_db)
        assert res["found"] is False
        assert res["highest_score"] == 0
        assert res["highest_match_id"] is None

    def test_tournament_highest_innings(self, populated_db):
        innings = get_tournament_highest_innings(limit=5, db=populated_db)
        assert len(innings) == 5
        # Verify descending sort order
        scores = [item["total_runs"] for item in innings]
        assert scores == sorted(scores, reverse=True)
        assert innings[0]["total_runs"] > 0
        assert len(innings[0]["event_ids"]) > 0

    def test_tournament_top_run_scorers(self, populated_db):
        top_scorers = get_tournament_top_run_scorers(limit=5, db=populated_db)
        assert len(top_scorers) == 5
        scores = [item["total_runs"] for item in top_scorers]
        assert scores == sorted(scores, reverse=True)
        assert top_scorers[0]["player"] == "Player 1"
        assert top_scorers[0]["total_runs"] == 177

    def test_tournament_top_stumpings(self, populated_db):
        top_stumpings = get_tournament_top_stumpings(limit=5, db=populated_db)
        assert len(top_stumpings) == 5
        counts = [item["metric_value"] for item in top_stumpings]
        assert counts == sorted(counts, reverse=True)
        assert top_stumpings[0]["player"] == "Player 1"
        assert top_stumpings[0]["metric_value"] == 10
        assert len(top_stumpings[0]["event_ids"]) == 10

    def test_tournament_top_wicket_takers(self, populated_db):
        top_wickets = get_tournament_top_wicket_takers(limit=5, db=populated_db)
        assert len(top_wickets) == 5
        counts = [item["metric_value"] for item in top_wickets]
        assert counts == sorted(counts, reverse=True)
        assert top_wickets[0]["player"] == "Player 2"
        assert top_wickets[0]["metric_value"] == 8
        assert len(top_wickets[0]["event_ids"]) == 8

    def test_tournament_top_dismissals(self, populated_db):
        top_dismissals = get_tournament_top_dismissals(limit=5, db=populated_db)
        assert len(top_dismissals) == 5
        counts = [item["metric_value"] for item in top_dismissals]
        assert counts == sorted(counts, reverse=True)
        assert top_dismissals[0]["metric_value"] == 24
        assert len(top_dismissals[0]["event_ids"]) == 24
