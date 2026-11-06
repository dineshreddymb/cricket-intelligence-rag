"""
tests/test_ingestion.py
=======================
Comprehensive test suite for Module 2: Data Ingestion and Preparation.
Covers deterministic parsing, cleaning, subfield extraction, validation edge cases,
and full dataset-level verification (500 records, 1422 total runs, 0 dropped).
"""

import pytest
from pathlib import Path

from ingestion.parser import parse_log_line, RawCricketRecord
from ingestion.cleaner import clean_and_enrich_record, extract_commentary_subfields, CleanedCricketRecord
from ingestion.validator import validate_record, ValidationReport
from ingestion.pipeline import run_ingestion_pipeline
from config.settings import get_settings


class TestDeterministicParser:
    """Tests for regex-based line parsing."""

    def test_parse_standard_record(self):
        line = (
            "[2025-09-22 00:23:24] Match:MATCH001 Team:Team B Player:Player 9 "
            "Event:LBW given Runs:5 Over:37 Status:pending review "
            "Commentary:Player 9 () performed LBW given, adding 5 runs in over 37."
        )
        record = parse_log_line(line, line_number=1)
        assert record is not None
        assert record.line_number == 1
        assert record.raw_timestamp == "2025-09-22 00:23:24"
        assert record.match_id == "MATCH001"
        assert record.team == "Team B"
        assert record.player == "Player 9"
        assert record.event == "LBW given"
        assert record.raw_runs == "5"
        assert record.raw_over == "37"
        assert record.status == "pending review"
        assert "performed LBW given" in record.commentary

    def test_parse_multiword_event_and_status(self):
        line = (
            "[2025-09-27 07:37:24] Match:MATCH001 Team:Team C Player:Player 7 "
            "Event:wicket taken Runs:1 Over:33 Status:verified "
            "Commentary:Unexpected wicket taken by Player 7 (all-rounder) altered the flow of the game."
        )
        record = parse_log_line(line, line_number=2)
        assert record is not None
        assert record.event == "wicket taken"
        assert record.status == "verified"

    def test_parse_malformed_line_returns_none(self):
        bad_line = "This is a random corrupted text line that does not match cricket log pattern"
        assert parse_log_line(bad_line, line_number=999) is None


class TestCommentaryCleanerAndEnrichment:
    """Tests for commentary subfield extraction and record enrichment."""

    def test_extract_empty_parentheses_role(self):
        commentary = "Player 9 () performed LBW given, adding 5 runs in over 37."
        role, venue, weather = extract_commentary_subfields(commentary)
        assert role is None
        assert venue is None
        assert weather is None

    def test_extract_named_role(self):
        commentary = "Unexpected wicket taken by Player 7 (all-rounder) altered the flow of the game."
        role, venue, weather = extract_commentary_subfields(commentary)
        assert role == "all-rounder"
        assert venue is None
        assert weather is None

    def test_extract_venue_and_weather(self):
        commentary = "In over 15 at Stadium 2 (overcast), Player 2 was involved in over completed."
        role, venue, weather = extract_commentary_subfields(commentary)
        assert role is None
        assert venue == "Stadium 2"
        assert weather == "overcast"

    def test_clean_and_enrich_deterministic_id(self):
        raw = RawCricketRecord(
            line_number=42,
            raw_timestamp="2025-09-22 00:23:24",
            match_id="MATCH001",
            team="Team B",
            player="Player 9",
            event="LBW given",
            raw_runs="5",
            raw_over="37",
            status="pending review",
            commentary="Player 9 () performed LBW given, adding 5 runs in over 37.",
            raw_line="raw text line",
        )
        cleaned = clean_and_enrich_record(raw)
        assert cleaned.event_id == "EVT_0042"
        assert cleaned.runs == 5
        assert cleaned.over == 37
        assert cleaned.timestamp == "2025-09-22T00:23:24"
        assert cleaned.player_role is None


class TestValidationRules:
    """Tests for record domain validation."""

    def test_valid_record_produces_no_errors(self):
        raw = RawCricketRecord(
            line_number=1,
            raw_timestamp="2025-09-22 00:23:24",
            match_id="MATCH001",
            team="Team A",
            player="Player 1",
            event="boundary hit",
            raw_runs="4",
            raw_over="10",
            status="confirmed",
            commentary="Solid boundary hit.",
            raw_line="raw text",
        )
        cleaned = clean_and_enrich_record(raw)
        errors = validate_record(cleaned)
        assert errors == []

    def test_invalid_runs_flagged(self):
        # Construct record with invalid runs directly
        record = CleanedCricketRecord(
            event_id="EVT_0001",
            line_number=1,
            timestamp="2025-09-22T00:23:24",
            match_id="MATCH001",
            team="Team A",
            player="Player 1",
            event="boundary hit",
            runs=10,  # Invalid: > 6
            over=10,
            status="confirmed",
            commentary="test",
            player_role=None,
            venue=None,
            weather_condition=None,
            raw_source_line="test",
        )
        errors = validate_record(record)
        assert any("Runs out of valid range" in e for e in errors)

    def test_invalid_team_flagged(self):
        record = CleanedCricketRecord(
            event_id="EVT_0001",
            line_number=1,
            timestamp="2025-09-22T00:23:24",
            match_id="MATCH001",
            team="Team Z",  # Invalid team
            player="Player 1",
            event="boundary hit",
            runs=4,
            over=10,
            status="confirmed",
            commentary="test",
            player_role=None,
            venue=None,
            weather_condition=None,
            raw_source_line="test",
        )
        errors = validate_record(record)
        assert any("Unexpected team" in e for e in errors)


class TestFullDatasetIntegration:
    """Integration test executing the pipeline over the entire 500-record dataset."""

    def test_full_dataset_ingestion(self):
        settings = get_settings()
        assert settings.dataset_path.exists(), f"Dataset missing at {settings.dataset_path}"

        valid_records, report, stats = run_ingestion_pipeline(settings.dataset_path)

        # 1. Total records accounted for: exactly 500
        assert len(valid_records) == 500, f"Expected 500 valid records, got {len(valid_records)}"
        assert report.total_processed == 500
        assert report.valid_count == 500
        assert report.invalid_count == 0
        assert stats["unparsed_count"] == 0

        # 2. Total runs match ground truth
        assert stats["total_runs"] == 1422, f"Expected 1422 total runs, got {stats['total_runs']}"

        # 3. Exactly 10 unique matches with 50 records each
        assert len(stats["match_distribution"]) == 10
        for match_id, count in stats["match_distribution"].items():
            assert count == 50, f"Match {match_id} has {count} records, expected 50"

        # 4. Exactly 11 unique players
        assert stats["unique_players"] == 11

        # 5. All 10 event types are present
        assert len(stats["event_distribution"]) == 10

        # 6. Verify deterministic sequence of event_ids
        for idx, rec in enumerate(valid_records, start=1):
            assert rec.event_id == f"EVT_{idx:04d}"
            assert rec.line_number == idx
