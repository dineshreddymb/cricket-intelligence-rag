"""
ingestion/parser.py
===================
Deterministic regex-based parsing of raw cricket log lines.
Uses regular expressions validated against the full 500-record dataset.
Does not use LLMs for parsing to ensure 100% determinism and reproducibility.
"""

import re
from dataclasses import dataclass
from typing import Optional


# Pre-compiled regex with named groups for fast and deterministic parsing
LOG_LINE_REGEX = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"Match:(?P<match_id>\S+)\s+"
    r"Team:(?P<team>.+?)\s+"
    r"Player:(?P<player>.+?)\s+"
    r"Event:(?P<event>.+?)\s+"
    r"Runs:(?P<runs>\d+)\s+"
    r"Over:(?P<over>\d+)\s+"
    r"Status:(?P<status>.+?)\s+"
    r"Commentary:(?P<commentary>.+)$"
)


@dataclass(frozen=True)
class RawCricketRecord:
    """Represents a syntactically parsed raw record before cleaning/enrichment."""
    line_number: int
    raw_timestamp: str
    match_id: str
    team: str
    player: str
    event: str
    raw_runs: str
    raw_over: str
    status: str
    commentary: str
    raw_line: str


def parse_log_line(line: str, line_number: int = 0) -> Optional[RawCricketRecord]:
    """
    Parses a single log line into a RawCricketRecord using deterministic regex.
    Returns None if the line does not strictly match the expected pattern.
    """
    match = LOG_LINE_REGEX.match(line.strip())
    if not match:
        return None

    groups = match.groupdict()
    return RawCricketRecord(
        line_number=line_number,
        raw_timestamp=groups["timestamp"],
        match_id=groups["match_id"].strip(),
        team=groups["team"].strip(),
        player=groups["player"].strip(),
        event=groups["event"].strip(),
        raw_runs=groups["runs"].strip(),
        raw_over=groups["over"].strip(),
        status=groups["status"].strip(),
        commentary=groups["commentary"].strip(),
        raw_line=line.strip(),
    )
