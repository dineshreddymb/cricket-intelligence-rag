"""
ingestion/validator.py
======================
Comprehensive validation rules for cleaned cricket records.
Validates required fields, value ranges, allowed enumerations, and data types.
Records all validation errors without silently dropping data.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .cleaner import CleanedCricketRecord


ALLOWED_TEAMS = {"Team A", "Team B", "Team C", "Team D"}

ALLOWED_EVENTS = {
    "LBW given",
    "run scored",
    "wicket taken",
    "stumping",
    "over completed",
    "boundary hit",
    "no ball",
    "six hit",
    "catch taken",
    "wide",
}

ALLOWED_STATUSES = {
    "pending review",
    "disputed",
    "confirmed",
    "verified",
}

ALLOWED_ROLES = {"batsman", "bowler", "all-rounder", "fielder"}

ALLOWED_WEATHER = {"windy", "overcast", "humid", "rainy", "sunny"}

MATCH_ID_REGEX = re.compile(r"^MATCH\d{3}$")
PLAYER_ID_REGEX = re.compile(r"^Player \d+$")
EVENT_ID_REGEX = re.compile(r"^EVT_\d{4}$")


@dataclass
class ValidationReport:
    """Detailed summary of the validation outcome across the dataset."""
    total_processed: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    errors_by_record: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_all_valid(self) -> bool:
        return self.invalid_count == 0 and self.total_processed > 0


def validate_record(record: CleanedCricketRecord) -> List[str]:
    """
    Validates a single CleanedCricketRecord against all domain rules.
    Returns a list of error descriptions (empty list means record is valid).
    """
    errors: List[str] = []

    # 1. Event ID
    if not EVENT_ID_REGEX.match(record.event_id):
        errors.append(f"Invalid event_id format: '{record.event_id}'. Expected 'EVT_XXXX'.")

    # 2. Match ID
    if not MATCH_ID_REGEX.match(record.match_id):
        errors.append(f"Invalid match_id format: '{record.match_id}'. Expected 'MATCHXXX'.")

    # 3. Team
    if record.team not in ALLOWED_TEAMS:
        errors.append(f"Unexpected team: '{record.team}'. Allowed: {ALLOWED_TEAMS}.")

    # 4. Player
    if not PLAYER_ID_REGEX.match(record.player):
        errors.append(f"Invalid player format: '{record.player}'. Expected 'Player N'.")

    # 5. Event
    if record.event not in ALLOWED_EVENTS:
        errors.append(f"Unexpected event: '{record.event}'. Allowed: {ALLOWED_EVENTS}.")

    # 6. Runs (0 to 6)
    if not (0 <= record.runs <= 6):
        errors.append(f"Runs out of valid range [0, 6]: {record.runs}.")

    # 7. Over (1 to 50)
    if not (1 <= record.over <= 50):
        errors.append(f"Over out of valid range [1, 50]: {record.over}.")

    # 8. Status
    if record.status not in ALLOWED_STATUSES:
        errors.append(f"Unexpected status: '{record.status}'. Allowed: {ALLOWED_STATUSES}.")

    # 9. Commentary
    if not record.commentary or not record.commentary.strip():
        errors.append("Commentary is empty or missing.")

    # 10. Optional Player Role
    if record.player_role is not None and record.player_role not in ALLOWED_ROLES:
        errors.append(f"Unexpected player_role: '{record.player_role}'. Allowed: {ALLOWED_ROLES}.")

    # 11. Optional Weather Condition
    if record.weather_condition is not None and record.weather_condition not in ALLOWED_WEATHER:
        errors.append(f"Unexpected weather: '{record.weather_condition}'. Allowed: {ALLOWED_WEATHER}.")

    return errors
