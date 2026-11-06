"""
ingestion/cleaner.py
====================
Cleans, type-casts, enriches, and assigns deterministic identifiers to parsed records.
Extracts optional metadata (player_role, venue, weather_condition) from commentary text
while preserving original commentary and source-line traceability.
"""

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

from .parser import RawCricketRecord


# Regex patterns for commentary extraction
ROLE_REGEX = re.compile(r"Player \d+\s*\((all-rounder|bowler|batsman|fielder)\)", re.IGNORECASE)
VENUE_REGEX = re.compile(r"at (Stadium \d+)", re.IGNORECASE)
WEATHER_REGEX = re.compile(r"\((windy|overcast|humid|rainy|sunny)\)", re.IGNORECASE)


@dataclass(frozen=True)
class CleanedCricketRecord:
    """Validated, normalized, and enriched cricket event record."""
    event_id: str                   # Deterministic unique ID, e.g. "EVT_0001"
    line_number: int                # Source file 1-based line number
    timestamp: str                  # ISO-8601 formatted timestamp: YYYY-MM-DDTHH:MM:SS
    match_id: str                   # Standard match ID, e.g. "MATCH001"
    team: str                       # Team name: "Team A", "Team B", "Team C", "Team D"
    player: str                     # Player name: "Player 1" - "Player 11"
    event: str                      # Event type: "LBW given", "wicket taken", etc.
    runs: int                       # Runs scored on this event (0 - 6)
    over: int                       # Over in which event occurred (1 - 50)
    status: str                     # Verification status: "confirmed", "disputed", etc.
    commentary: str                 # Verbatim original commentary
    player_role: Optional[str]      # "batsman", "bowler", "all-rounder", "fielder", or None
    venue: Optional[str]            # "Stadium 1", "Stadium 2", etc., or None
    weather_condition: Optional[str]# "windy", "overcast", etc., or None
    raw_source_line: str            # Traceability back to exact raw source text

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to a dictionary suitable for MongoDB storage."""
        return asdict(self)


def extract_commentary_subfields(commentary: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts optional subfields from commentary:
    - player_role: extracted from 'Player X (role)'
    - venue: extracted from 'at Stadium X'
    - weather_condition: extracted from '(condition)'
    
    Returns (player_role, venue, weather_condition), with None for any missing/empty values.
    """
    # 1. Player Role
    role_match = ROLE_REGEX.search(commentary)
    player_role = role_match.group(1).lower() if role_match else None

    # 2. Venue
    venue_match = VENUE_REGEX.search(commentary)
    venue = venue_match.group(1) if venue_match else None

    # 3. Weather Condition
    weather_match = WEATHER_REGEX.search(commentary)
    weather_condition = weather_match.group(1).lower() if weather_match else None

    return player_role, venue, weather_condition


def clean_and_enrich_record(raw: RawCricketRecord) -> CleanedCricketRecord:
    """
    Converts a RawCricketRecord into a CleanedCricketRecord.
    Performs type conversion, ISO datetime normalization, commentary subfield
    extraction, and deterministic event_id generation.
    """
    # Deterministic event_id padded to 4 digits (EVT_0001, EVT_0002, ...)
    event_id = f"EVT_{raw.line_number:04d}"

    # Parse and normalize timestamp to standard ISO-8601
    dt = datetime.strptime(raw.raw_timestamp, "%Y-%m-%d %H:%M:%S")
    iso_timestamp = dt.isoformat()

    # Numeric conversions
    runs = int(raw.raw_runs)
    over = int(raw.raw_over)

    # Subfield extraction from commentary
    player_role, venue, weather_condition = extract_commentary_subfields(raw.commentary)

    return CleanedCricketRecord(
        event_id=event_id,
        line_number=raw.line_number,
        timestamp=iso_timestamp,
        match_id=raw.match_id,
        team=raw.team,
        player=raw.player,
        event=raw.event,
        runs=runs,
        over=over,
        status=raw.status,
        commentary=raw.commentary,
        player_role=player_role,
        venue=venue,
        weather_condition=weather_condition,
        raw_source_line=raw.raw_line,
    )
