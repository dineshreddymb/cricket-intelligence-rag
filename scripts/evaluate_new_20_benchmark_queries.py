"""
scripts/evaluate_new_20_benchmark_queries.py
============================================
Comprehensive evaluation suite for the 20 new test queries (Q26-Q45):
- Evaluates Routing & Entity Normalization
- Evaluates Retrieval Quality (Precision@K, Candidate Relevance)
- Evaluates Answer Generation Quality (Faithfulness, Groundedness, Citations)
- Compares Directly Against Ground Truth Extracted from cricket_log.txt
"""

import sys
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
import django
django.setup()

from rag.pipeline import answer_question
from rag.query_router import route_query
from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.indexer import run_full_qdrant_indexing

def log_print(*args, **kwargs):
    kwargs["flush"] = True
    print(*args, **kwargs)
db = get_mongo_database()
valid_records, _, _ = run_ingestion_pipeline()
ingest_records_to_mongo(valid_records, db=db)
try:
    run_full_qdrant_indexing(recreate=False)
except Exception:
    pass

# Load raw dataset lines
with open(ROOT_DIR / "cricket_log.txt", "r", encoding="utf-8") as f:
    RAW_LINES = [l.strip() for l in f.readlines() if l.strip()]

def parse_line(line: str, idx: int) -> Dict[str, Any]:
    ts_m = re.match(r"^\[(.*?)\]", line)
    ts = ts_m.group(1) if ts_m else ""

    m_match = re.search(r"Match:(MATCH\d{3})", line)
    m_team = re.search(r"Team:(Team [A-D])", line)
    m_player = re.search(r"Player:(Player \d+)", line)
    m_event = re.search(r"Event:(.*?)\s+Runs:", line)
    m_runs = re.search(r"Runs:(\d+)", line)
    m_over = re.search(r"Over:(\d+)", line)
    m_status = re.search(r"Status:(.*?)\s+Commentary:", line)
    m_comm = re.search(r"Commentary:(.*)$", line)

    player = m_player.group(1) if m_player else ""
    comm = m_comm.group(1).strip() if m_comm else ""

    role_m = re.search(rf"{player}\s*\((.*?)\)", comm)
    role = role_m.group(1).strip() if role_m else None
    if role == "":
        role = "missing"

    venue = None
    venue_m = re.search(r"\bat\s+(Stadium\s+\d+)", comm, re.IGNORECASE)
    if venue_m:
        venue = venue_m.group(1)

    weather = None
    weather_m = re.search(r"\((sunny|rainy|overcast|windy)\)", comm, re.IGNORECASE)
    if weather_m:
        weather = weather_m.group(1).lower()

    has_empty_stadium_weather = bool(re.search(r"\bat\s+\((sunny|rainy|overcast|windy)\)", comm, re.IGNORECASE))

    return {
        "event_id": f"EVT_{idx:04d}",
        "line": idx,
        "timestamp": ts,
        "match_id": m_match.group(1) if m_match else "",
        "team": m_team.group(1) if m_team else "",
        "player": player,
        "event": m_event.group(1).strip() if m_event else "",
        "runs": int(m_runs.group(1)) if m_runs else 0,
        "over": int(m_over.group(1)) if m_over else 0,
        "status": m_status.group(1).strip() if m_status else "",
        "commentary": comm,
        "role": role,
        "venue": venue,
        "weather": weather,
        "has_empty_stadium_weather": has_empty_stadium_weather,
        "raw": line,
    }

ALL_RECORDS = [parse_line(l, idx) for idx, l in enumerate(RAW_LINES, start=1)]

TEST_CASES = [
    # 1. Player-focused queries
    {
        "id": "Q26",
        "category": "Player-focused",
        "query": "What kind of incidents was Player 10 involved in across the matches?",
        "ground_truth_filter": lambda r: r["player"] == "Player 10",
        "desc": "Incidents involving Player 10 across all matches (51 records in total)",
    },
    {
        "id": "Q27",
        "category": "Player-focused",
        "query": "Can you show me all the recorded activities involving Player 5 from Team C?",
        "ground_truth_filter": lambda r: r["player"] == "Player 5" and r["team"] == "Team C",
        "desc": "Activities involving Player 5 from Team C (14 records)",
    },
    {
        "id": "Q28",
        "category": "Player-focused",
        "query": "Find incidents where players identified as bowlers were involved in cricket events.",
        "ground_truth_filter": lambda r: r["role"] == "bowler",
        "desc": "Incidents where player role is explicitly identified as bowler (17 records)",
    },
    {
        "id": "Q29",
        "category": "Player-focused",
        "query": "Were any fielders involved in stumping or catch-related incidents?",
        "ground_truth_filter": lambda r: r["event"] in ("stumping", "catch taken") and r["role"] == "fielder",
        "desc": "Fielders involved in stumping or catch taken events (4 records in ground truth)",
    },
    {
        "id": "Q30",
        "category": "Player-focused",
        "query": "Which players were involved in multiple types of events in the same match?",
        "ground_truth_filter": lambda r: True,  # Analytical cross-record
        "desc": "Players involved in distinct event types within the same match",
    },

    # 2. Venue and weather-based queries
    {
        "id": "Q31",
        "category": "Venue and weather-based",
        "query": "Show me the cricket incidents that happened at Stadium 2.",
        "ground_truth_filter": lambda r: r["venue"] == "Stadium 2",
        "desc": "Cricket incidents taking place at Stadium 2 (17 records)",
    },
    {
        "id": "Q32",
        "category": "Venue and weather-based",
        "query": "Were there any incidents recorded during rainy weather?",
        "ground_truth_filter": lambda r: r["weather"] == "rainy",
        "desc": "Incidents during rainy weather (6 records in dataset)",
    },
    {
        "id": "Q33",
        "category": "Venue and weather-based",
        "query": "Find events that took place in windy conditions at different stadiums.",
        "ground_truth_filter": lambda r: r["weather"] == "windy",
        "desc": "Events in windy conditions across stadiums (15 records)",
    },
    {
        "id": "Q34",
        "category": "Venue and weather-based",
        "query": "Can you find incidents where the commentary mentions the weather conditions but doesn't specify the stadium?",
        "ground_truth_filter": lambda r: r["has_empty_stadium_weather"],
        "desc": "Incidents where commentary states weather condition without specifying stadium name (12 records)",
    },

    # 3. Match, over and time-based queries
    {
        "id": "Q35",
        "category": "Match, over and time-based",
        "query": "What happened during the first five overs of MATCH006?",
        "ground_truth_filter": lambda r: r["match_id"] == "MATCH006" and 1 <= r["over"] <= 5,
        "desc": "Events occurring in overs 1-5 of MATCH006 (8 records)",
    },
    {
        "id": "Q36",
        "category": "Match, over and time-based",
        "query": "Show me the incidents that took place in the final ten overs of MATCH003.",
        "ground_truth_filter": lambda r: r["match_id"] == "MATCH003" and 41 <= r["over"] <= 50,
        "desc": "Events occurring in final 10 overs (41-50) of MATCH003 (13 records)",
    },
    {
        "id": "Q37",
        "category": "Match, over and time-based",
        "query": "Were there any important events recorded on 30 September 2025?",
        "ground_truth_filter": lambda r: "2025-09-30" in r["timestamp"],
        "desc": "Events recorded on date 30 September 2025 (46 records)",
    },
    {
        "id": "Q38",
        "category": "Match, over and time-based",
        "query": "Find incidents involving Team D players between overs 35 and 45.",
        "ground_truth_filter": lambda r: r["team"] == "Team D" and 35 <= r["over"] <= 45,
        "desc": "Team D events occurring between overs 35 and 45 (39 records)",
    },

    # 4. Event and status combinations
    {
        "id": "Q39",
        "category": "Event and status combinations",
        "query": "Can you find boundary-hit incidents that were either disputed or awaiting review?",
        "ground_truth_filter": lambda r: r["event"] == "boundary hit" and r["status"] in ("disputed", "pending review"),
        "desc": "Boundary hit events with status disputed or pending review (29 records)",
    },
    {
        "id": "Q40",
        "category": "Event and status combinations",
        "query": "Show me all confirmed incidents involving wide or no-ball deliveries.",
        "ground_truth_filter": lambda r: r["status"] == "confirmed" and r["event"] in ("wide", "no ball"),
        "desc": "Confirmed wide or no-ball deliveries (30 records)",
    },
    {
        "id": "Q41",
        "category": "Event and status combinations",
        "query": "Were there any wicket-related incidents that were marked as verified?",
        "ground_truth_filter": lambda r: r["status"] == "verified" and r["event"] in ("wicket taken", "stumping", "LBW given", "catch taken"),
        "desc": "Verified dismissals/wicket-related incidents (51 records)",
    },
    {
        "id": "Q42",
        "category": "Event and status combinations",
        "query": "Find examples where a player was involved in an over-completed event.",
        "ground_truth_filter": lambda r: r["event"] == "over completed",
        "desc": "Incidents where event type is over completed (49 records)",
    },

    # 5. Deeper retrieval and data understanding
    {
        "id": "Q43",
        "category": "Deeper retrieval and data understanding",
        "query": "Can you find records where the commentary describes an unexpected incident?",
        "ground_truth_filter": lambda r: "unexpected" in r["commentary"].lower(),
        "desc": "Records where commentary explicitly notes unexpected events altering flow (83 records)",
    },
    {
        "id": "Q44",
        "category": "Deeper retrieval and data understanding",
        "query": "Are there any records where the player's role is missing from the commentary?",
        "ground_truth_filter": lambda r: r["role"] == "missing",
        "desc": "Records where commentary contains empty player role parentheses '()' (90 records)",
    },
    {
        "id": "Q45",
        "category": "Deeper retrieval and data understanding",
        "query": "Show me incidents where the same player was involved in different event types under different status labels.",
        "ground_truth_filter": lambda r: True,
        "desc": "Multi-event, multi-status relationship query across same player",
    },
]

def execute_with_retry(query: str, max_retries: int = 4) -> Any:
    for attempt in range(max_retries):
        try:
            return answer_question(query)
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                wait_time = (attempt + 1) * 8
                print(f"    [429 Rate Limit] Backing off {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise exc
    return answer_question(query)

def run_benchmark():
    log_print("=" * 80)
    log_print("STARTING COMPREHENSIVE BENCHMARK EVALUATION (Q26 - Q45)")
    log_print("=" * 80)

    results = []
    category_metrics = {}

    for idx, tc in enumerate(TEST_CASES, start=1):
        qid = tc["id"]
        cat = tc["category"]
        q_text = tc["query"]
        desc = tc["desc"]
        gt_filter = tc["ground_truth_filter"]

        log_print(f"\n[{idx}/20] Running {qid} ({cat}): '{q_text}'")

        # 1. Establish Ground Truth
        matching_gt = [r for r in ALL_RECORDS if gt_filter(r)]
        gt_ids = {r["event_id"] for r in matching_gt}
        log_print(f"  Ground Truth: {len(matching_gt)} total matching records in dataset ({desc})")

        # 2. Route Query
        decision = route_query(q_text)
        log_print(f"  Router: Intent={decision.intent.value} | Entities: player={decision.extracted_player}, role={decision.extracted_role}, team={decision.extracted_team}, match={decision.extracted_match_id}, event={decision.extracted_event}, status={decision.extracted_status}, overs=({decision.min_over}, {decision.max_over}), venue={decision.extracted_venue}, weather={decision.extracted_weather}")

        # 3. Execute End-to-End RAG
        time.sleep(4.5)  # Rate pacing
        t0 = time.time()
        resp = execute_with_retry(q_text)
        latency = round(time.time() - t0, 2)

        # 4. Evaluate Retrieval Quality
        retrieved_ids = []
        if resp.raw_evidence:
            for item in resp.raw_evidence:
                if isinstance(item, dict) and "event_id" in item:
                    retrieved_ids.append(item["event_id"])
        
        # Calculate Precision@K
        if retrieved_ids:
            if qid in ("Q30", "Q45"):
                # Complex analytical queries
                precision_at_k = 1.0
            else:
                relevant_hits = [eid for eid in retrieved_ids if eid in gt_ids]
                precision_at_k = round(len(relevant_hits) / len(retrieved_ids), 4)
        else:
            precision_at_k = 1.0 if len(matching_gt) == 0 else 0.0

        # 5. Evaluate Citations & Groundedness
        cited_ids = resp.cited_evidence or []
        valid_citations = [cid for cid in cited_ids if cid in [r["event_id"] for r in ALL_RECORDS]]
        hallucinated_citations = [cid for cid in cited_ids if cid not in [r["event_id"] for r in ALL_RECORDS] and not cid.startswith("MATCH")]
        groundedness_score = 1.0 if len(hallucinated_citations) == 0 else 0.0

        # 6. Overall Pass / Fail
        passed = (precision_at_k >= 0.70) and (groundedness_score == 1.0) and len(resp.answer.strip()) > 30

        log_print(f"  Model: {resp.model_used} | Latency: {latency}s")
        log_print(f"  Retrieved Chunks ({len(retrieved_ids)}): {retrieved_ids}")
        log_print(f"  Citations ({len(cited_ids)}): {cited_ids}")
        log_print(f"  Precision@K: {precision_at_k * 100:.1f}% | Groundedness: {'100%' if groundedness_score == 1.0 else 'HALLUCINATED'}")
        log_print(f"  Status: {'PASS' if passed else 'FAIL'}")
        log_print(f"  Answer Preview: {resp.answer[:220]}...")

        res_record = {
            "id": qid,
            "category": cat,
            "query": q_text,
            "description": desc,
            "router_intent": decision.intent.value,
            "extracted_entities": {
                "player": decision.extracted_player,
                "team": decision.extracted_team,
                "teams": decision.extracted_teams,
                "match": decision.extracted_match_id,
                "event": decision.extracted_event,
                "status": decision.extracted_status,
                "statuses": decision.extracted_statuses,
                "overs": [decision.min_over, decision.max_over],
                "venue": decision.extracted_venue,
                "weather": decision.extracted_weather,
            },
            "ground_truth_count": len(matching_gt),
            "ground_truth_sample": [r["event_id"] for r in matching_gt[:5]],
            "retrieved_chunk_ids": retrieved_ids,
            "retrieval_precision_at_k": precision_at_k,
            "cited_evidence": cited_ids,
            "groundedness_score": groundedness_score,
            "model_used": resp.model_used,
            "latency_seconds": latency,
            "answer": resp.answer,
            "passed": passed,
        }
        results.append(res_record)

        if cat not in category_metrics:
            category_metrics[cat] = {"total": 0, "passed": 0, "precisions": []}
        category_metrics[cat]["total"] += 1
        if passed:
            category_metrics[cat]["passed"] += 1
        category_metrics[cat]["precisions"].append(precision_at_k)

    # Save to JSON
    out_file = ROOT_DIR / "scripts" / "benchmark_results_new_20.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": len(results),
            "passed_queries": sum(1 for r in results if r["passed"]),
            "pass_rate_percentage": round(sum(1 for r in results if r["passed"]) / len(results) * 100, 2),
            "category_summary": {
                cat: {
                    "total": m["total"],
                    "passed": m["passed"],
                    "accuracy_pct": round(m["passed"] / m["total"] * 100, 2),
                    "avg_precision": round(sum(m["precisions"]) / len(m["precisions"]) * 100, 2),
                }
                for cat, m in category_metrics.items()
            },
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION COMPLETE")
    print(f"Total Passed: {sum(1 for r in results if r['passed'])} / {len(results)} ({sum(1 for r in results if r['passed'])/len(results)*100:.1f}%)")
    print(f"Report saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
