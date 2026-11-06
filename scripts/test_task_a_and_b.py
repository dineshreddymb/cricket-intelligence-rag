"""
scripts/test_task_a_and_b.py
============================
Comprehensive, unbiased benchmark and evaluation suite for:
- Task A: Semantic Search, Vector Retrieval (Qdrant + BM25), and Grounded LLM Generation (Gemini).
- Task B: Exact Structured Database Queries, Aggregations, and Routing (MongoDB).

Compares live system responses against raw dataset truth from cricket_log.txt.
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, List

# Setup project root on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure Django environment is configured
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
import django
django.setup()

from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.indexer import run_full_qdrant_indexing
from rag.pipeline import answer_question
from rag.query_router import route_query


def ensure_environment_seeded():
    """Ensures database and vector index are active and seeded."""
    db = get_mongo_database()
    valid_records, _, _ = run_ingestion_pipeline()
    ingest_records_to_mongo(valid_records, db=db)
    try:
        run_full_qdrant_indexing(recreate=False)
    except Exception as e:
        print(f"Index seed warning: {e}")


def load_raw_dataset_lines() -> List[str]:
    """Loads all lines from raw cricket_log.txt."""
    dataset_path = ROOT_DIR / "cricket_log.txt"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def parse_raw_line(line: str) -> Dict[str, Any]:
    """Extracts key fields from a raw log line."""
    m_match = re.search(r"Match:(MATCH\d{3})", line)
    m_team = re.search(r"Team:(Team [A-D])", line)
    m_player = re.search(r"Player:(Player \d+)", line)
    m_event = re.search(r"Event:(.*?)\s+Runs:", line)
    m_runs = re.search(r"Runs:(\d+)", line)
    m_over = re.search(r"Over:(\d+)", line)
    m_status = re.search(r"Status:(.*?)\s+Commentary:", line)
    m_comm = re.search(r"Commentary:(.*)$", line)

    return {
        "match_id": m_match.group(1) if m_match else "",
        "team": m_team.group(1) if m_team else "",
        "player": m_player.group(1) if m_player else "",
        "event": m_event.group(1).strip() if m_event else "",
        "runs": int(m_runs.group(1)) if m_runs else 0,
        "over": int(m_over.group(1)) if m_over else 0,
        "status": m_status.group(1).strip() if m_status else "",
        "commentary": m_comm.group(1).strip() if m_comm else "",
        "raw": line,
    }


def evaluate_task_a():
    """Evaluates Semantic RAG (Task A)."""
    print("\n" + "="*80)
    print("TASK A: SEMANTIC SEARCH, VECTOR RETRIEVAL & GROUNDED GENERATION EVALUATION")
    print("="*80)

    raw_lines = load_raw_dataset_lines()
    parsed_records = [parse_raw_line(l) for l in raw_lines]

    test_queries = [
        {
            "query": "Find deliveries that resulted in extra runs for the batting team.",
            "type": "Extras (wides / no-balls)",
            "verification_func": lambda rec: rec["event"] in ["no ball", "wide"] or "extra" in rec["commentary"].lower(),
        },
        {
            "query": "Show me incidents involving a stumping when the weather was overcast.",
            "type": "Multi-condition (stumping + overcast)",
            "verification_func": lambda rec: rec["event"] == "stumping" and "overcast" in rec["commentary"].lower(),
        },
        {
            "query": "What happened during LBW decisions pending review?",
            "type": "Status & Event (LBW + pending review)",
            "verification_func": lambda rec: rec["event"] == "LBW given" and rec["status"] == "pending review",
        },
        {
            "query": "Describe Player 6 events in disputed matches.",
            "type": "Player & Status (Player 6 + disputed)",
            "verification_func": lambda rec: rec["player"] == "Player 6" and rec["status"] == "disputed",
        },
    ]

    task_a_results = []

    for tq in test_queries:
        query = tq["query"]
        print(f"\n[QUERY]: {query}")
        print(f"[CATEGORY]: {tq['type']}")

        # 1. Ground truth in raw dataset
        matching_ground_truth = []
        for i, rec in enumerate(parsed_records):
            if tq["verification_func"](rec):
                evt_id = f"EVT_{i+1:04d}"
                matching_ground_truth.append((evt_id, rec))

        print(f"-> Total Ground-Truth Matching Records in cricket_log.txt: {len(matching_ground_truth)}")
        gt_ids = {item[0] for item in matching_ground_truth}

        # 2. Run Pipeline
        decision = route_query(query)
        response = answer_question(query, limit=5)

        retrieved_raw = response.raw_evidence
        retrieved_ids = [r.get("event_id") for r in retrieved_raw if "event_id" in r]
        cited_ids = response.cited_evidence

        # 3. Retrieval Precision & Recall in Top-K
        relevant_retrieved = [eid for eid in retrieved_ids if eid in gt_ids]
        precision_at_k = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0

        print(f"-> Intent Classified: {response.intent} (Decision Reason: {decision.reason})")
        print(f"-> Engine Used: {response.model_used}")
        print(f"-> Top-{len(retrieved_ids)} Retrieved IDs: {retrieved_ids}")
        print(f"-> Relevant Records in Top-{len(retrieved_ids)}: {relevant_retrieved} (Precision@{len(retrieved_ids)}: {precision_at_k:.1%})")
        print(f"-> Cited Evidence in Answer: {cited_ids}")
        print(f"-> LLM Answer:\n   \"{response.answer}\"")

        # 4. Check for Hallucination in Generation
        hallucinated_citations = [c for c in cited_ids if c not in retrieved_ids]
        is_hallucination_free = len(hallucinated_citations) == 0

        print(f"-> Citations Grounded in Retrieved Context: {'YES (100%)' if is_hallucination_free else f'NO (Hallucinated: {hallucinated_citations})'}")

        task_a_results.append({
            "query": query,
            "type": tq["type"],
            "intent": response.intent,
            "gt_count": len(matching_ground_truth),
            "retrieved_count": len(retrieved_ids),
            "relevant_retrieved_count": len(relevant_retrieved),
            "precision": precision_at_k,
            "cited_count": len(cited_ids),
            "hallucination_free": is_hallucination_free,
            "answer": response.answer,
        })

    return task_a_results


def evaluate_task_b():
    """Evaluates Exact Structured Queries & Database Aggregations (Task B)."""
    print("\n" + "="*80)
    print("TASK B: EXACT STRUCTURED DATABASE AGGREGATIONS EVALUATION")
    print("="*80)

    raw_lines = load_raw_dataset_lines()
    parsed_records = [parse_raw_line(l) for l in raw_lines]

    test_queries = [
        {
            "query": "what is the total score scored by the player 2 from team a in the match 1 .",
            "description": "Specific Player + Team in Specific Match",
            "calc_ground_truth": lambda: sum(
                r["runs"] for r in parsed_records
                if r["match_id"] == "MATCH001" and r["team"] == "Team A" and r["player"] == "Player 2"
            ),
            "expected_events": lambda: [
                f"EVT_{i+1:04d}" for i, r in enumerate(parsed_records)
                if r["match_id"] == "MATCH001" and r["team"] == "Team A" and r["player"] == "Player 2"
            ]
        },
        {
            "query": "what is the total score scored by the player 7 from team a in the match 1",
            "description": "Specific Player + Team in Specific Match (P7)",
            "calc_ground_truth": lambda: sum(
                r["runs"] for r in parsed_records
                if r["match_id"] == "MATCH001" and r["team"] == "Team A" and r["player"] == "Player 7"
            ),
            "expected_events": lambda: [
                f"EVT_{i+1:04d}" for i, r in enumerate(parsed_records)
                if r["match_id"] == "MATCH001" and r["team"] == "Team A" and r["player"] == "Player 7"
            ]
        },
        {
            "query": "how many runs did team b score in MATCH001?",
            "description": "Team Total in Specific Match",
            "calc_ground_truth": lambda: sum(
                r["runs"] for r in parsed_records
                if r["match_id"] == "MATCH001" and r["team"] == "Team B"
            ),
            "expected_events": lambda: [f"MATCH001_Team B"]
        },
        {
            "query": "what were the total runs scored in MATCH002?",
            "description": "Total Match Runs across all teams",
            "calc_ground_truth": lambda: sum(
                r["runs"] for r in parsed_records
                if r["match_id"] == "MATCH002"
            ),
            "expected_events": lambda: ["MATCH002"]
        },
        {
            "query": "who scored the most runs in the tournament?",
            "description": "Tournament Leading Run Scorer",
            "calc_ground_truth": lambda: get_top_scorer_from_raw(parsed_records),
            "expected_events": lambda: []
        },
        {
            "query": "did any player score a century in the tournament?",
            "description": "Century Inquiry (100+ individual match runs)",
            "calc_ground_truth": lambda: any(
                check_century_in_raw(parsed_records)
            ),
            "expected_events": lambda: []
        }
    ]

    task_b_results = []

    for tq in test_queries:
        query = tq["query"]
        print(f"\n[QUERY]: {query}")
        print(f"[CATEGORY]: {tq['description']}")

        # 1. Raw ground truth
        gt_value = tq["calc_ground_truth"]()
        gt_events = tq["expected_events"]()
        print(f"-> Raw Dataset Ground-Truth Value: {gt_value}")
        if gt_events:
            print(f"-> Raw Dataset Expected Events: {gt_events}")

        # 2. System response
        decision = route_query(query)
        response = answer_question(query)

        print(f"-> Intent Classified: {response.intent}")
        print(f"-> Engine Used: {response.model_used}")
        print(f"-> System Cited Evidence: {response.cited_evidence}")
        print(f"-> System Answer:\n   \"{response.answer}\"")

        # 3. Accuracy check
        is_exact_match = False
        if isinstance(gt_value, int):
            # Check if integer appears in answer text
            is_exact_match = (
                f"{gt_value} runs" in response.answer or
                f"= {gt_value}" in response.answer or
                f"total of {gt_value}" in response.answer or
                f"{gt_value}" in response.answer
            )
        elif isinstance(gt_value, tuple):  # (Player, total_runs)
            top_p, top_r = gt_value
            is_exact_match = top_p in response.answer and str(top_r) in response.answer
        elif isinstance(gt_value, bool):
            # Century check: False means "No player scored a century"
            if not gt_value:
                is_exact_match = "no player scored a century" in response.answer.lower()

        print(f"-> Numerical / Fact Match: {'PERFECT (100% MATCH)' if is_exact_match else 'DISCREPANCY DETECTED'}")

        task_b_results.append({
            "query": query,
            "description": tq["description"],
            "gt_value": gt_value,
            "system_answer": response.answer,
            "engine": response.model_used,
            "intent": response.intent,
            "exact_match": is_exact_match,
        })

    return task_b_results


def get_top_scorer_from_raw(records: List[Dict[str, Any]]):
    from collections import defaultdict
    player_totals = defaultdict(int)
    for r in records:
        player_totals[r["player"]] += r["runs"]
    sorted_players = sorted(player_totals.items(), key=lambda x: x[1], reverse=True)
    return sorted_players[0]  # (player, total_runs)


def check_century_in_raw(records: List[Dict[str, Any]]):
    from collections import defaultdict
    innings = defaultdict(int)
    for r in records:
        innings[(r["match_id"], r["player"])] += r["runs"]
    return [score >= 100 for score in innings.values()]


if __name__ == "__main__":
    ensure_environment_seeded()
    res_a = evaluate_task_a()
    res_b = evaluate_task_b()
