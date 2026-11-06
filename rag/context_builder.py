"""
rag/context_builder.py
======================
Formats hydrated candidate records into a strictly bounded, citeable evidence package
for LLM grounding. Ensures full source traceability and explicit limitation disclosures.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from retrieval.semantic_search import SearchResultItem


@dataclass(frozen=True)
class GroundedEvidencePackage:
    """Bounded evidence container sent to Gemini alongside the user query."""
    formatted_context: str
    cited_event_ids: List[str]
    evidence_count: int
    retrieval_scope: str
    limitations: str
    raw_evidence_items: List[Dict[str, Any]]


def build_evidence_context(
    search_hits: List[SearchResultItem],
    max_items: int = 5,
) -> GroundedEvidencePackage:
    """
    Constructs a structured evidence block from top search hits.
    Each item is prefixed with a unique [Evidence EVT_XXXX] tag for verbatim citation.
    """
    selected_hits = search_hits[:max_items]

    if not selected_hits:
        return GroundedEvidencePackage(
            formatted_context="NO MATCHING EVIDENCE FOUND IN DATASET.",
            cited_event_ids=[],
            evidence_count=0,
            retrieval_scope="0 candidate records retrieved from Qdrant.",
            limitations="No relevant evidence records matched the query criteria in the dataset.",
            raw_evidence_items=[],
        )

    lines: List[str] = []
    cited_ids: List[str] = []
    raw_items: List[Dict[str, Any]] = []

    for item in selected_hits:
        cited_ids.append(item.event_id)
        raw_items.append(item.to_dict())

        extras = []
        if item.player_role:
            extras.append(f"Role: {item.player_role}")
        else:
            extras.append("Role: Unspecified / Missing")
        if item.venue:
            extras.append(f"Venue: {item.venue}")
        else:
            extras.append("Venue: Unspecified / Missing")
        if item.weather_condition:
            extras.append(f"Weather: {item.weather_condition}")
        if item.timestamp:
            date_val = str(item.timestamp).split()[0] if ' ' in str(item.timestamp) else str(item.timestamp)[:10]
            extras.append(f"Date: {date_val}")
        extra_str = f" [{' | '.join(extras)}]" if extras else ""

        line_str = (
            f"[Evidence {item.event_id}] "
            f"Match: {item.match_id} | "
            f"Team: {item.team} | "
            f"Player: {item.player} | "
            f"Event: {item.event} | "
            f"Runs: {item.runs} | "
            f"Over: {item.over} | "
            f"Status: {item.status} | "
            f"Commentary: \"{item.commentary}\""
            f"{extra_str} (Similarity Score: {item.score:.3f})"
        )
        lines.append(line_str)

    formatted_context = "\n\n".join(lines)
    scope = f"Retrieved top {len(selected_hits)} semantically similar events from 500 total dataset records."
    limitations = (
        "Semantic similarity retrieves candidate events based on vector proximity. "
        "Top-K retrieved records do not represent complete dataset totals; "
        "exact numerical calculations must use structured database queries."
    )

    return GroundedEvidencePackage(
        formatted_context=formatted_context,
        cited_event_ids=cited_ids,
        evidence_count=len(selected_hits),
        retrieval_scope=scope,
        limitations=limitations,
        raw_evidence_items=raw_items,
    )
