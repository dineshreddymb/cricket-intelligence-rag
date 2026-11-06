"""
rag/query_router.py
===================
Intelligent query intent classifier and router.
Differentiates between:
1. EXACT_STRUCTURED: Exact arithmetic, aggregations, counts, and sums -> routed directly to MongoDB
2. SEMANTIC_RAG: Natural language search, commentary summaries, and qualitative questions -> routed to Qdrant + Gemini
3. HYBRID: Filtered semantic search (e.g., "describe LBW events in MATCH001")
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from rag.entity_normalizer import extract_all_entities


class QueryIntent(str, Enum):
    EXACT_STRUCTURED = "exact_structured"
    SEMANTIC_RAG = "semantic_rag"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class QueryRoutingDecision:
    intent: QueryIntent
    extracted_match_id: Optional[str] = None
    extracted_team: Optional[str] = None
    extracted_teams: List[str] = field(default_factory=list)
    extracted_player: Optional[str] = None
    extracted_role: Optional[str] = None
    extracted_event: Optional[str] = None
    extracted_status: Optional[str] = None
    extracted_statuses: List[str] = field(default_factory=list)
    extracted_over: Optional[int] = None
    min_over: Optional[int] = None
    max_over: Optional[int] = None
    extracted_venue: Optional[str] = None
    extracted_weather: Optional[str] = None
    missing_venue: bool = False
    missing_role: bool = False
    reason: str = ""


# Keywords indicating exact structured aggregation intent
AGGREGATION_KEYWORDS = [
    # Natural question stems
    "how much did",
    "how much",
    "how many did",
    "what did",
    "what was the score",
    "what were the runs",
    "runs scored by",
    "score scored by",
    "total score scored by",
    "score of",
    "runs of",
    # Classic aggregation phrases
    "how many runs",
    "total runs",
    "score in",
    "total score",
    "highest score",
    "highest runs",
    "lowest score",
    "lowest runs",
    "maximum runs",
    "max runs",
    "minimum runs",
    "min runs",
    "most runs",
    "top score",
    "top scorer",
    "best score",
    "most wickets",
    "highest wickets",
    "how many wickets",
    "total wickets",
    "most stumpings",
    "highest stumpings",
    "total stumpings",
    "most dismissals",
    "highest dismissals",
    "total dismissals",
    "most catches",
    "highest catches",
    "total catches",
    "most extras",
    "highest extras",
    "total extras",
    "how many extras",
    "most extra runs",
    "total extra runs",
    "top bowler",
    "best bowler",
    "leading wicket taker",
    "top wicket taker",
    "leading run scorer",
    "top run scorer",
    "most lbws",
    "total lbws",
    "how many events",
    "count of",
    "number of events",
    "how many boundaries",
    "how many sixes",
    "total sixes",
    "total boundaries",
    "century",
    "centuries",
    "hundred",
    "highest century",
    "statistics for",
    "stats for",
    "highest individual",
    "scored the highest",
    "who scored the most",
    "who scored most",
    "who has the highest",
]

EVENT_KEYWORDS = {
    # Extras
    "extra runs": "extras",
    "extra run": "extras",
    "extras": "extras",
    "extra": "extras",
    "no ball": "no ball",
    "no balls": "no ball",
    "wide": "wide",
    "wides": "wide",
    "illegal delivery": "extras",
    "illegal deliveries": "extras",

    # Dismissals & Wickets
    "lbw": "LBW given",
    "leg before": "LBW given",
    "wicket": "wicket taken",
    "wickets": "wicket taken",
    "bowled": "wicket taken",
    "stumping": "stumping",
    "stumpings": "stumping",
    "stumped": "stumping",
    "catch": "catch taken",
    "catches": "catch taken",
    "caught": "catch taken",
    "dismissal": "dismissal",
    "dismissals": "dismissal",

    # Boundaries
    "boundary": "boundary hit",
    "boundaries": "boundary hit",
    "four": "boundary hit",
    "fours": "boundary hit",
    "six": "six hit",
    "sixes": "six hit",
    "maximum": "six hit",
    "maximums": "six hit",

    # Other
    "run scored": "run scored",
    "over completed": "over completed",
}

STATUS_KEYWORDS = {
    "pending": "pending review",
    "review": "pending review",
    "disputed": "disputed",
    "confirmed": "confirmed",
    "verified": "verified",
}


def route_query(query: str) -> QueryRoutingDecision:
    """
    Analyzes query text, extracts cricket entities, and determines the optimal execution path.
    Uses the generalized entity normalizer for real-world variations.
    """
    clean_query = query.strip()
    lower_query = clean_query.lower()

    # 1. Master Generalized Entity Extraction
    entities = extract_all_entities(clean_query)
    extracted_match = entities.get("match_id")
    extracted_team = entities.get("team")
    extracted_teams = entities.get("teams", [])
    extracted_player = entities.get("player")
    extracted_role = entities.get("role")
    extracted_over = entities.get("over")
    min_over = entities.get("min_over")
    max_over = entities.get("max_over")
    extracted_event = entities.get("event")
    extracted_status = entities.get("status")
    extracted_statuses = entities.get("statuses", [])
    extracted_venue = entities.get("venue")
    extracted_weather = entities.get("weather")
    missing_venue = entities.get("missing_venue", False)
    missing_role = entities.get("missing_role", False)

    # 2. Intent classification
    is_aggregation = any(kw in lower_query for kw in AGGREGATION_KEYWORDS)
    has_superlative = any(w in lower_query for w in ["highest", "lowest", "max", "maximum", "most", "top", "best", "leading"])
    has_metric = any(w in lower_query for w in [
        "score", "run", "runs", "wicket", "wickets", "stumping", "stumpings",
        "dismissal", "dismissals", "catch", "catches", "lbw", "six", "sixes",
        "boundary", "boundaries", "no ball", "wide", "extra", "extras", "century", "centuries", "total", "bowler"
    ])
    is_superlative_stat = has_superlative and has_metric

    if is_aggregation or is_superlative_stat:
        return QueryRoutingDecision(
            intent=QueryIntent.EXACT_STRUCTURED,
            extracted_match_id=extracted_match,
            extracted_team=extracted_team,
            extracted_teams=extracted_teams,
            extracted_player=extracted_player,
            extracted_role=extracted_role,
            extracted_event=extracted_event,
            extracted_status=extracted_status,
            extracted_statuses=extracted_statuses,
            extracted_over=extracted_over,
            min_over=min_over,
            max_over=max_over,
            extracted_venue=extracted_venue,
            extracted_weather=extracted_weather,
            missing_venue=missing_venue,
            missing_role=missing_role,
            reason="Query requests exact numerical statistics, superlatives, or aggregations; routed to MongoDB aggregation.",
        )

    if extracted_match or extracted_team or extracted_player or len(extracted_teams) > 1 or extracted_role:
        return QueryRoutingDecision(
            intent=QueryIntent.HYBRID,
            extracted_match_id=extracted_match,
            extracted_team=extracted_team,
            extracted_teams=extracted_teams,
            extracted_player=extracted_player,
            extracted_role=extracted_role,
            extracted_event=extracted_event,
            extracted_status=extracted_status,
            extracted_statuses=extracted_statuses,
            extracted_over=extracted_over,
            min_over=min_over,
            max_over=max_over,
            extracted_venue=extracted_venue,
            extracted_weather=extracted_weather,
            missing_venue=missing_venue,
            missing_role=missing_role,
            reason="Query contains specific entity filters; routed to filtered semantic retrieval.",
        )

    return QueryRoutingDecision(
        intent=QueryIntent.SEMANTIC_RAG,
        extracted_match_id=None,
        extracted_team=None,
        extracted_teams=extracted_teams,
        extracted_player=None,
        extracted_role=extracted_role,
        extracted_event=extracted_event,
        extracted_status=extracted_status,
        extracted_statuses=extracted_statuses,
        extracted_over=extracted_over,
        min_over=min_over,
        max_over=max_over,
        extracted_venue=extracted_venue,
        extracted_weather=extracted_weather,
        missing_venue=missing_venue,
        missing_role=missing_role,
        reason="Query is qualitative or explorative; routed to Qdrant vector retrieval + Gemini RAG.",
    )
