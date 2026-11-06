"""
database/schema.py
==================
MongoDB collection specifications, validation schema, and index definitions
for authoritative structured cricket match records.
"""

from typing import Dict, Any, List, Tuple


CRICKET_EVENTS_COLLECTION = "cricket_events"


def get_cricket_event_validator() -> Dict[str, Any]:
    """Returns MongoDB JSON Schema validator for the cricket_events collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "event_id",
                "line_number",
                "timestamp",
                "match_id",
                "team",
                "player",
                "event",
                "runs",
                "over",
                "status",
                "commentary",
                "raw_source_line",
            ],
            "properties": {
                "event_id": {
                    "bsonType": "string",
                    "pattern": "^EVT_\\d{4}$",
                    "description": "Unique deterministic event ID (EVT_XXXX)",
                },
                "line_number": {
                    "bsonType": "int",
                    "minimum": 1,
                    "description": "1-based source line index",
                },
                "timestamp": {
                    "bsonType": "string",
                    "description": "ISO-8601 formatted timestamp",
                },
                "match_id": {
                    "bsonType": "string",
                    "pattern": "^MATCH\\d{3}$",
                    "description": "Match identifier",
                },
                "team": {
                    "enum": ["Team A", "Team B", "Team C", "Team D"],
                    "description": "Team identifier",
                },
                "player": {
                    "bsonType": "string",
                    "description": "Player identifier",
                },
                "event": {
                    "bsonType": "string",
                    "description": "Cricket event type",
                },
                "runs": {
                    "bsonType": "int",
                    "minimum": 0,
                    "maximum": 6,
                    "description": "Runs scored (0 to 6)",
                },
                "over": {
                    "bsonType": "int",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Over number (1 to 50)",
                },
                "status": {
                    "enum": ["pending review", "disputed", "confirmed", "verified"],
                    "description": "Event verification status",
                },
                "commentary": {
                    "bsonType": "string",
                    "description": "Verbatim commentary text",
                },
                "player_role": {
                    "bsonType": ["string", "null"],
                    "description": "Optional player role if explicitly stated",
                },
                "venue": {
                    "bsonType": ["string", "null"],
                    "description": "Optional stadium venue if explicitly stated",
                },
                "weather_condition": {
                    "bsonType": ["string", "null"],
                    "description": "Optional weather condition if explicitly stated",
                },
                "raw_source_line": {
                    "bsonType": "string",
                    "description": "Verbatim original unparsed line",
                },
            },
        }
    }


def get_required_indexes() -> List[Dict[str, Any]]:
    """Returns list of index definitions for optimal query performance and uniqueness."""
    return [
        # 1. Unique index on event_id for primary lookup and idempotency
        {"keys": [("event_id", 1)], "unique": True, "name": "idx_event_id_unique"},
        # 2. Filter indexes
        {"keys": [("match_id", 1)], "unique": False, "name": "idx_match_id"},
        {"keys": [("team", 1)], "unique": False, "name": "idx_team"},
        {"keys": [("player", 1)], "unique": False, "name": "idx_player"},
        {"keys": [("event", 1)], "unique": False, "name": "idx_event"},
        {"keys": [("status", 1)], "unique": False, "name": "idx_status"},
        # 3. Compound index for match over progression
        {"keys": [("match_id", 1), ("over", 1)], "unique": False, "name": "idx_match_over"},
    ]
