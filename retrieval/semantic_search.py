"""
retrieval/semantic_search.py
============================
Core retrieval engine combining dense vector search in Qdrant with
authoritative record hydration from MongoDB.
Supports structured metadata filtering (match, team, player, event, status).
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, Range

from config.settings import get_settings
from vector_store.embedder import get_embedder, EmbeddingModel
from vector_store.qdrant_manager import get_qdrant_client
from database.queries import get_records_by_ids

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResultItem:
    """Represents a single hydrated semantic search result."""
    event_id: str
    score: float
    rank: int
    match_id: str
    team: str
    player: str
    event: str
    runs: int
    over: int
    status: str
    commentary: str
    player_role: Optional[str]
    venue: Optional[str]
    weather_condition: Optional[str]
    line_number: int
    raw_source_line: str
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "match_id": self.match_id,
            "team": self.team,
            "player": self.player,
            "event": self.event,
            "runs": self.runs,
            "over": self.over,
            "status": self.status,
            "commentary": self.commentary,
            "player_role": self.player_role,
            "venue": self.venue,
            "weather_condition": self.weather_condition,
            "timestamp": self.timestamp,
            "line_number": self.line_number,
        }


def build_qdrant_filter(
    match_id: Optional[str] = None,
    team: Optional[str] = None,
    teams: Optional[List[str]] = None,
    player: Optional[str] = None,
    role: Optional[str] = None,
    event: Optional[str] = None,
    status: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    min_over: Optional[int] = None,
    max_over: Optional[int] = None,
    venue: Optional[str] = None,
    weather: Optional[str] = None,
) -> Optional[Filter]:
    """Constructs a Qdrant Filter object from keyword constraints."""
    conditions = []
    if match_id:
        conditions.append(FieldCondition(key="match_id", match=MatchValue(value=match_id)))
    if teams and len(teams) > 1:
        conditions.append(FieldCondition(key="team", match=MatchAny(any=teams)))
    elif team:
        conditions.append(FieldCondition(key="team", match=MatchValue(value=team)))
    elif teams and len(teams) == 1:
        conditions.append(FieldCondition(key="team", match=MatchValue(value=teams[0])))
    if player:
        conditions.append(FieldCondition(key="player", match=MatchValue(value=player)))
    if role:
        conditions.append(FieldCondition(key="player_role", match=MatchValue(value=role)))
    if event:
        if event in ("extras", "extra"):
            conditions.append(FieldCondition(key="event", match=MatchAny(any=["wide", "no ball"])))
        elif event in ("dismissal", "dismissals"):
            conditions.append(FieldCondition(key="event", match=MatchAny(any=["wicket taken", "stumping", "LBW given", "catch taken"])))
        elif event in ("fielding dismissals", "fielding dismissal"):
            conditions.append(FieldCondition(key="event", match=MatchAny(any=["stumping", "catch taken"])))
        elif event in ("boundary", "boundaries"):
            conditions.append(FieldCondition(key="event", match=MatchAny(any=["boundary hit", "six hit"])))
        else:
            conditions.append(FieldCondition(key="event", match=MatchValue(value=event)))
    if statuses and len(statuses) > 1:
        conditions.append(FieldCondition(key="status", match=MatchAny(any=statuses)))
    elif status:
        conditions.append(FieldCondition(key="status", match=MatchValue(value=status)))
    elif statuses and len(statuses) == 1:
        conditions.append(FieldCondition(key="status", match=MatchValue(value=statuses[0])))
    if min_over is not None or max_over is not None:
        range_kwargs = {}
        if min_over is not None:
            range_kwargs["gte"] = min_over
        if max_over is not None:
            range_kwargs["lte"] = max_over
        conditions.append(FieldCondition(key="over", range=Range(**range_kwargs)))
    if venue:
        conditions.append(FieldCondition(key="venue", match=MatchValue(value=venue)))
    if weather:
        conditions.append(FieldCondition(key="weather_condition", match=MatchValue(value=weather)))

    if not conditions:
        return None
    return Filter(must=conditions)


def search_cricket_events(
    query: str,
    limit: int = 5,
    match_id: Optional[str] = None,
    team: Optional[str] = None,
    teams: Optional[List[str]] = None,
    player: Optional[str] = None,
    role: Optional[str] = None,
    event: Optional[str] = None,
    status: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    min_over: Optional[int] = None,
    max_over: Optional[int] = None,
    venue: Optional[str] = None,
    weather: Optional[str] = None,
    score_threshold: float = 0.0,
    client: Optional[QdrantClient] = None,
    embedder: Optional[EmbeddingModel] = None,
    collection_name: Optional[str] = None,
) -> List[SearchResultItem]:
    """
    Executes semantic similarity search with optional metadata filters,
    hydrating full authoritative record payloads from MongoDB.
    """
    target_client = client or get_qdrant_client()
    target_embedder = embedder or get_embedder()
    target_collection = collection_name or get_settings().qdrant_collection_name

    # Ensure collection exists; re-check disk or auto-index if transiently in-memory
    try:
        existing = [c.name for c in target_client.get_collections().collections]
    except Exception:
        existing = []

    if target_collection not in existing:
        from vector_store.qdrant_manager import reset_qdrant_client
        reset_qdrant_client()
        target_client = client or get_qdrant_client()
        try:
            existing = [c.name for c in target_client.get_collections().collections]
        except Exception:
            existing = []

        if target_collection not in existing:
            logger.warning("Qdrant collection '%s' missing. Auto-indexing records.", target_collection)
            from vector_store.indexer import run_full_qdrant_indexing
            run_full_qdrant_indexing(recreate=False, client=target_client, collection_name=target_collection)

    # 1. Embed user query into vector space
    query_vector = target_embedder.embed_query(query)

    # 2. Build metadata filter
    qdrant_filter = build_qdrant_filter(
        match_id=match_id,
        team=team,
        teams=teams,
        player=player,
        role=role,
        event=event,
        status=status,
        statuses=statuses,
        min_over=min_over,
        max_over=max_over,
        venue=venue,
        weather=weather,
    )

    # 3. Search Qdrant vector index
    try:
        response = target_client.query_points(
            collection_name=target_collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0.0 else None,
        )
        scored_points = response.points
    except Exception as exc:
        logger.error("Error during Qdrant query_points: %s", exc)
        return []

    if not scored_points:
        return []

    # 4. Extract candidate event IDs in rank order
    candidate_event_ids = [pt.payload["event_id"] for pt in scored_points if "event_id" in pt.payload]

    # 5. Hydrate authoritative documents from MongoDB
    hydrated_docs = get_records_by_ids(candidate_event_ids)

    # 6. Assemble ranked search results
    results: List[SearchResultItem] = []
    for rank, point in enumerate(scored_points, start=1):
        event_id = point.payload.get("event_id")
        doc = hydrated_docs.get(event_id) or point.payload

        item = SearchResultItem(
            event_id=event_id,
            score=float(point.score),
            rank=rank,
            match_id=doc.get("match_id", ""),
            team=doc.get("team", ""),
            player=doc.get("player", ""),
            event=doc.get("event", ""),
            runs=int(doc.get("runs", 0)),
            over=int(doc.get("over", 0)),
            status=doc.get("status", ""),
            commentary=doc.get("commentary", ""),
            player_role=doc.get("player_role"),
            venue=doc.get("venue"),
            weather_condition=doc.get("weather_condition"),
            timestamp=doc.get("timestamp"),
            line_number=int(doc.get("line_number", 0)),
            raw_source_line=doc.get("raw_source_line", ""),
        )
        results.append(item)

    return results
