"""
vector_store/text_repr.py
=========================
Formats validated structured cricket records into dense text representations
optimized for semantic similarity search in vector space.
"""

from typing import Union, Dict, Any
from ingestion.cleaner import CleanedCricketRecord


def create_record_text_representation(record: Union[CleanedCricketRecord, Dict[str, Any]]) -> str:
    """
    Constructs a structured natural-language representation of a cricket event.
    Combines structured dimensions and unstructured commentary to maximize retrieval accuracy.
    """
    if isinstance(record, CleanedCricketRecord):
        rec = record.to_dict()
    else:
        rec = record

    base_repr = (
        f"Match: {rec.get('match_id')} | "
        f"Team: {rec.get('team')} | "
        f"Player: {rec.get('player')} | "
        f"Event: {rec.get('event')} | "
        f"Runs: {rec.get('runs')} | "
        f"Over: {rec.get('over')} | "
        f"Status: {rec.get('status')} | "
        f"Commentary: {rec.get('commentary')}"
    )

    # Append optional subfield context when explicitly available
    extras = []
    if rec.get("player_role"):
        extras.append(f"Role: {rec['player_role']}")
    if rec.get("venue"):
        extras.append(f"Venue: {rec['venue']}")
    if rec.get("weather_condition"):
        extras.append(f"Weather: {rec['weather_condition']}")

    if extras:
        base_repr += " [" + ", ".join(extras) + "]"

    return base_repr
