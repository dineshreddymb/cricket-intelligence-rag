"""
retrieval/hybrid_search.py
==========================
State-of-the-art Hybrid Information Retrieval combining:
1. Vector Search (Semantic) via Qdrant -> Top 50
2. Keyword Search (Lexical) via BM25 -> Top 50
3. Reciprocal Rank Fusion (RRF) -> Top 50 Merged Candidates
4. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) -> Top-K Context (K=5)
5. Hydration into authoritative SearchResultItem objects.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .semantic_search import (
    search_cricket_events as search_dense_events,
    SearchResultItem,
)
from .bm25_search import search_bm25
from database.queries import get_records_by_ids

logger = logging.getLogger(__name__)

_cross_encoder_instance = None
_cross_encoder_failed = False


def get_cross_encoder():
    """Lazy singleton getter for the Cross-Encoder model."""
    global _cross_encoder_instance, _cross_encoder_failed
    if _cross_encoder_failed:
        return None
    if _cross_encoder_instance is None:
        try:
            from sentence_transformers import CrossEncoder
            # ms-marco-MiniLM-L-6-v2: fast, lightweight, standard for passage reranking
            _cross_encoder_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Initialized Cross-Encoder reranker successfully.")
        except Exception as exc:
            logger.warning("Could not initialize Cross-Encoder (%s); falling back to RRF rank.", exc)
            _cross_encoder_failed = True
            return None
    return _cross_encoder_instance


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, int]],  # (event_id, rank 1-based)
    bm25_results: List[Tuple[str, int]],   # (event_id, rank 1-based)
    k: int = 60,
    top_n: int = 50,
) -> List[Tuple[str, float]]:
    """
    Combines ranked results from Dense Vector search and BM25 Lexical search
    using standard Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum(1 / (k + rank_m(d)))
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for event_id, rank in dense_results:
        rrf_scores[event_id] += 1.0 / (k + rank)

    for event_id, rank in bm25_results:
        rrf_scores[event_id] += 1.0 / (k + rank)

    sorted_pool = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pool[:top_n]


def _balance_multi_team_selection(
    scored_items: List[Tuple[Dict[str, Any], float]],
    teams: Optional[List[str]],
    limit: int,
) -> List[Tuple[Dict[str, Any], float]]:
    """Ensures balanced candidate representation across all requested teams."""
    if not teams or len(teams) <= 1:
        return scored_items[:limit]

    by_team: Dict[str, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
    for doc, score in scored_items:
        t = doc.get("team")
        if t in teams:
            by_team[t].append((doc, score))

    balanced: List[Tuple[Dict[str, Any], float]] = []
    per_team_quota = max(1, limit // len(teams))
    for t in teams:
        balanced.extend(by_team[t][:per_team_quota])

    # Fill remaining capacity from highest scoring unselected candidates
    chosen_ids = {d.get("event_id") for d, _ in balanced}
    for doc, score in scored_items:
        if len(balanced) >= limit:
            break
        if doc.get("event_id") not in chosen_ids:
            balanced.append((doc, score))
            chosen_ids.add(doc.get("event_id"))

    return balanced[:limit]


def hybrid_search_cricket_events(
    query: str,
    limit: int = 5,
    candidate_limit: int = 50,
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
    missing_venue: bool = False,
    missing_role: bool = False,
    client: Any = None,
    embedder: Any = None,
    collection_name: Optional[str] = None,
) -> List[SearchResultItem]:
    """
    End-to-end Hybrid Search pipeline:
    1. Vector Search (Qdrant) -> candidate_limit (50)
    2. BM25 Lexical Search -> candidate_limit (50)
    3. RRF Fusion -> Top 50 Merged Pool
    4. Cross-Encoder Reranker -> Top-K (limit=5)
    5. MongoDB Hydration & Return
    """
    # 1. Dense Vector Search (Top 50)
    try:
        dense_hits = search_dense_events(
            query=query,
            limit=candidate_limit,
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
            client=client,
            embedder=embedder,
            collection_name=collection_name,
        )
        dense_ranked = [(hit.event_id, idx + 1) for idx, hit in enumerate(dense_hits)]
    except Exception as exc:
        logger.warning("Dense vector search failed in hybrid pipeline: %s", exc)
        dense_hits = []
        dense_ranked = []

    # 2. BM25 Lexical Search (Top 50)
    try:
        bm25_raw = search_bm25(query=query, limit=candidate_limit)
        bm25_ranked = [(event_id, idx + 1) for idx, (event_id, _) in enumerate(bm25_raw)]
    except Exception as exc:
        logger.warning("BM25 lexical search failed in hybrid pipeline: %s", exc)
        bm25_ranked = []

    # If both returned nothing, return empty
    if not dense_ranked and not bm25_ranked:
        return []

    # 3. Reciprocal Rank Fusion (RRF) -> Top 50 Merged
    fused_pool = reciprocal_rank_fusion(
        dense_results=dense_ranked,
        bm25_results=bm25_ranked,
        k=60,
        top_n=candidate_limit,
    )

    fused_event_ids = [event_id for event_id, _ in fused_pool]
    mongo_docs = get_records_by_ids(fused_event_ids)

    # Filter documents according to optional metadata constraints if present
    candidates: List[Dict[str, Any]] = []
    for event_id in fused_event_ids:
        doc = mongo_docs.get(event_id)
        if not doc:
            continue
        if match_id and doc.get("match_id") != match_id:
            continue
        if teams and len(teams) > 0:
            if doc.get("team") not in teams:
                continue
        elif team and doc.get("team") != team:
            continue
        if player and doc.get("player") != player:
            continue
        if min_over is not None and int(doc.get("over", 0)) < min_over:
            continue
        if max_over is not None and int(doc.get("over", 0)) > max_over:
            continue
        if venue and doc.get("venue") != venue:
            continue
        if weather and doc.get("weather_condition") != weather:
            continue
        if event:
            if event in ("extras", "extra"):
                if doc.get("event") not in ("wide", "no ball"):
                    continue
            elif event in ("dismissal", "dismissals"):
                if doc.get("event") not in ("wicket taken", "stumping", "LBW given", "catch taken"):
                    continue
            elif event in ("fielding dismissals", "fielding dismissal"):
                if doc.get("event") not in ("stumping", "catch taken"):
                    continue
            elif event in ("boundary", "boundaries"):
                if doc.get("event") not in ("boundary hit", "six hit"):
                    continue
            elif doc.get("event") != event:
                continue
        if role and doc.get("player_role") != role:
            continue
        if missing_venue and (doc.get("venue") is not None or not doc.get("weather_condition")):
            continue
        if missing_role and doc.get("player_role") is not None:
            continue
        if statuses and len(statuses) > 0:
            if doc.get("status") not in statuses:
                continue
        elif status and doc.get("status") != status:
            continue
        candidates.append(doc)

    if not candidates:
        return []

    # 4. Cross-Encoder Reranker
    reranker = get_cross_encoder()
    if reranker is not None and len(candidates) > 1:
        try:
            def _format_reranker_text(d: Dict[str, Any]) -> str:
                ev = d.get("event", "")
                extra_note = ""
                if ev in ("wide", "no ball"):
                    extra_note = f" (extra delivery conceded to batting team: {d.get('runs')} extra runs)"
                v_str = f" | Venue: {d['venue']}" if d.get("venue") else ""
                w_str = f" | Weather: {d['weather_condition']}" if d.get("weather_condition") else ""
                if d.get("player_role"):
                    r_str = f" ({d['player_role']})"
                elif re.search(r"Player \d+\s*\(\)", d.get("commentary", "")):
                    r_str = " (player role is missing from commentary: empty role '()' placeholder)"
                else:
                    r_str = " (missing role / unstated role)"
                ts_str = f" | Date: {str(d['timestamp'])[:10]}" if d.get("timestamp") else ""
                return (
                    f"Match {d.get('match_id')}, Over {d.get('over')}: {d.get('team')} - {d.get('player')}{r_str} "
                    f"| Event: {ev}{extra_note} ({d.get('runs')} runs) | Status: {d.get('status')}{v_str}{w_str}{ts_str} | {d.get('commentary', '')}"
                )

            pairs = [
                (query, _format_reranker_text(d))
                for d in candidates
            ]
            scores = reranker.predict(pairs)
            scored_candidates = list(zip(candidates, [float(s) for s in scores]))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            final_selection = _balance_multi_team_selection(scored_candidates, teams, limit)
        except Exception as exc:
            logger.warning("Cross-Encoder scoring failed: %s; falling back to RRF ranking.", exc)
            fallback_items = [(d, 1.0 / (idx + 1)) for idx, d in enumerate(candidates)]
            final_selection = _balance_multi_team_selection(fallback_items, teams, limit)
    else:
        fallback_items = [(d, 1.0 / (idx + 1)) for idx, d in enumerate(candidates)]
        final_selection = _balance_multi_team_selection(fallback_items, teams, limit)

    # 5. Build final SearchResultItem objects
    results: List[SearchResultItem] = []
    for rank_idx, (doc, score) in enumerate(final_selection, start=1):
        item = SearchResultItem(
            event_id=doc.get("event_id", ""),
            score=round(score, 4),
            rank=rank_idx,
            match_id=doc.get("match_id", ""),
            team=doc.get("team", ""),
            player=doc.get("player", ""),
            event=doc.get("event", ""),
            runs=doc.get("runs", 0),
            over=doc.get("over", 0),
            status=doc.get("status", ""),
            commentary=doc.get("commentary", ""),
            player_role=doc.get("player_role"),
            venue=doc.get("venue"),
            weather_condition=doc.get("weather_condition"),
            timestamp=doc.get("timestamp"),
            line_number=doc.get("line_number", 0),
            raw_source_line=doc.get("raw_source_line", ""),
        )
        results.append(item)

    return results
