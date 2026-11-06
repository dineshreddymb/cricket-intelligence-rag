"""
scripts/evaluate_20_benchmark_queries.py
========================================
Comprehensive automated benchmark running all 20 user evaluation queries.
Validates each response against raw ground truth in cricket_log.txt:
- Evaluates Query Intent & Routing
- Evaluates Retrieval Quality (Precision@K, Top-K Relevance)
- Evaluates Generation Quality (Faithfulness, Grounded Citations, Hallucination Check)
- Evaluates Real-world Data Alignment
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
import django
django.setup()

from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.indexer import run_full_qdrant_indexing
from rag.pipeline import answer_question
from rag.query_router import route_query

# Seed environment
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
    m_match = re.search(r"Match:(MATCH\d{3})", line)
    m_team = re.search(r"Team:(Team [A-D])", line)
    m_player = re.search(r"Player:(Player \d+)", line)
    m_event = re.search(r"Event:(.*?)\s+Runs:", line)
    m_runs = re.search(r"Runs:(\d+)", line)
    m_over = re.search(r"Over:(\d+)", line)
    m_status = re.search(r"Status:(.*?)\s+Commentary:", line)
    m_comm = re.search(r"Commentary:(.*)$", line)

    comm = m_comm.group(1).strip() if m_comm else ""
    # Extract role from commentary if present
    role_m = re.search(r"\((all-rounder|batsman|bowler|fielder)\)", comm, re.IGNORECASE)
    role = role_m.group(1).lower() if role_m else None

    # Extract venue & weather from commentary
    venue_m = re.search(r"at (Stadium \d+)", comm)
    venue = venue_m.group(1) if venue_m else None

    weather_m = re.search(r"\((overcast|humid|sunny|rainy)\)", comm)
    weather = weather_m.group(1).lower() if weather_m else None

    return {
        "event_id": f"EVT_{idx+1:04d}",
        "line_number": idx + 1,
        "match_id": m_match.group(1) if m_match else "",
        "team": m_team.group(1) if m_team else "",
        "player": m_player.group(1) if m_player else "",
        "event": m_event.group(1).strip() if m_event else "",
        "runs": int(m_runs.group(1)) if m_runs else 0,
        "over": int(m_over.group(1)) if m_over else 0,
        "status": m_status.group(1).strip() if m_status else "",
        "commentary": comm,
        "player_role": role,
        "venue": venue,
        "weather_condition": weather,
    }

PARSED_RECORDS = [parse_line(l, i) for i, l in enumerate(RAW_LINES)]

BENCHMARK_QUERIES = [
    # 1. Semantic search – 5 queries
    {
        "id": "Q01",
        "category": "1. Semantic Search",
        "query": "Can you find moments where a player was given out because of an LBW decision?",
        "ground_truth_filter": lambda r: r["event"] == "LBW given",
        "description": "LBW dismissals"
    },
    {
        "id": "Q02",
        "category": "1. Semantic Search",
        "query": "Show me instances where the fielding team managed to take a catch.",
        "ground_truth_filter": lambda r: r["event"] == "catch taken",
        "description": "Catches taken by fielding team"
    },
    {
        "id": "Q03",
        "category": "1. Semantic Search",
        "query": "Were there any moments when a player was dismissed through stumping?",
        "ground_truth_filter": lambda r: r["event"] == "stumping",
        "description": "Stumping dismissals"
    },
    {
        "id": "Q04",
        "category": "1. Semantic Search",
        "query": "Find incidents where players hit boundaries or sixes.",
        "ground_truth_filter": lambda r: r["event"] in ["boundary hit", "six hit"],
        "description": "Boundaries (4s) or Sixes (6s)"
    },
    {
        "id": "Q05",
        "category": "1. Semantic Search",
        "query": "Can you find situations where the batting side received extra runs through wide or no-ball deliveries?",
        "ground_truth_filter": lambda r: r["event"] in ["wide", "no ball"] or "extra" in r["commentary"].lower(),
        "description": "Extras (wide or no-ball)"
    },

    # 2. Contextual understanding – 5 queries
    {
        "id": "Q06",
        "category": "2. Contextual Understanding",
        "query": "Were there any incidents that created excitement among the spectators?",
        "ground_truth_filter": lambda r: "spectators" in r["commentary"].lower() or "spectators reacted" in r["commentary"].lower(),
        "description": "Spectator excitement commentary"
    },
    {
        "id": "Q07",
        "category": "2. Contextual Understanding",
        "query": "Find moments where more than one cricket event was mentioned in the same commentary.",
        "ground_truth_filter": lambda r: "multiple events observed" in r["commentary"].lower() or "while another" in r["commentary"].lower(),
        "description": "Multiple events in single commentary"
    },
    {
        "id": "Q08",
        "category": "2. Contextual Understanding",
        "query": "Show me incidents where an unexpected event changed the flow of the game.",
        "ground_truth_filter": lambda r: "altered the flow of the game" in r["commentary"].lower() or "unexpected" in r["commentary"].lower(),
        "description": "Unexpected event altering flow of game"
    },
    {
        "id": "Q09",
        "category": "2. Contextual Understanding",
        "query": "Find incidents where a player's event was still awaiting review or confirmation.",
        "ground_truth_filter": lambda r: r["status"] in ["pending review", "confirmed"],
        "description": "Status: pending review or awaiting confirmation"
    },
    {
        "id": "Q10",
        "category": "2. Contextual Understanding",
        "query": "Were there any controversial or disputed incidents during the matches?",
        "ground_truth_filter": lambda r: r["status"] == "disputed" or "disputed" in r["commentary"].lower(),
        "description": "Disputed / controversial incidents"
    },

    # 3. Combined filtering – 5 queries
    {
        "id": "Q11",
        "category": "3. Combined Filtering",
        "query": "In MATCH004, can you find any LBW decisions that were disputed?",
        "ground_truth_filter": lambda r: r["match_id"] == "MATCH004" and r["event"] == "LBW given" and r["status"] == "disputed",
        "description": "MATCH004 + LBW given + disputed"
    },
    {
        "id": "Q12",
        "category": "3. Combined Filtering",
        "query": "Show me incidents involving all-rounders where a wicket was taken.",
        "ground_truth_filter": lambda r: r["event"] == "wicket taken" and r["player_role"] == "all-rounder",
        "description": "All-rounders + wicket taken"
    },
    {
        "id": "Q13",
        "category": "3. Combined Filtering",
        "query": "Were there any six-hit incidents in Stadium 3 during sunny weather?",
        "ground_truth_filter": lambda r: r["event"] == "six hit" and r["venue"] == "Stadium 3" and r["weather_condition"] == "sunny",
        "description": "six hit + Stadium 3 + sunny"
    },
    {
        "id": "Q14",
        "category": "3. Combined Filtering",
        "query": "Find wide-ball incidents that happened in the later overs.",
        "ground_truth_filter": lambda r: r["event"] == "wide" and r["over"] >= 30,
        "description": "wide + over >= 30"
    },
    {
        "id": "Q15",
        "category": "3. Combined Filtering",
        "query": "Can you show me boundary-hit incidents involving Team A players in MATCH007?",
        "ground_truth_filter": lambda r: r["match_id"] == "MATCH007" and r["team"] == "Team A" and r["event"] == "boundary hit",
        "description": "MATCH007 + Team A + boundary hit"
    },

    # 4. Answer generation and reasoning – 5 queries
    {
        "id": "Q16",
        "category": "4. Answer Generation & Reasoning",
        "query": "Give me a summary of the different dismissal-related incidents recorded in MATCH008.",
        "ground_truth_filter": lambda r: r["match_id"] == "MATCH008" and r["event"] in ["wicket taken", "LBW given", "stumping", "catch taken"],
        "description": "MATCH008 dismissals summary"
    },
    {
        "id": "Q17",
        "category": "4. Answer Generation & Reasoning",
        "query": "Which incidents are still awaiting verification, and what happened in those incidents?",
        "ground_truth_filter": lambda r: r["status"] == "pending review",
        "description": "Pending review incidents & actions"
    },
    {
        "id": "Q18",
        "category": "4. Answer Generation & Reasoning",
        "query": "Compare the types of incidents recorded for Team A and Team B.",
        "ground_truth_filter": lambda r: r["team"] in ["Team A", "Team B"],
        "description": "Comparative breakdown for Team A vs Team B"
    },
    {
        "id": "Q19",
        "category": "4. Answer Generation & Reasoning",
        "query": "Find examples where the commentary mentions one event but also refers to another event in the same over.",
        "ground_truth_filter": lambda r: "multiple events observed" in r["commentary"].lower() or "while another" in r["commentary"].lower(),
        "description": "Co-occurring events in same over commentary"
    },
    {
        "id": "Q20",
        "category": "4. Answer Generation & Reasoning",
        "query": "Show me a few examples where the recorded runs were zero and explain what happened in those incidents.",
        "ground_truth_filter": lambda r: r["runs"] == 0,
        "description": "Zero runs incidents explanation"
    },
]

import time

results = []

print("="*90)
print("EXECUTING 20-QUERY BENCHMARK SUITE (with 4.5s pacing for Free Tier RPM)")
print("="*90)

for bq in BENCHMARK_QUERIES:
    qid = bq["id"]
    cat = bq["category"]
    query = bq["query"]
    desc = bq["description"]

    # 1. Calculate ground-truth matching records
    matching_gt = [r for r in PARSED_RECORDS if bq["ground_truth_filter"](r)]
    gt_ids = set(r["event_id"] for r in matching_gt)

    # 2. Execute Pipeline with retry on rate limit
    decision = route_query(query)
    
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = answer_question(query, limit=5)
            # If answer contains 429 quota error, retry after pause
            if "quota" in response.answer.lower() or "429" in response.answer.lower() or "resource_exhausted" in response.answer.lower():
                print(f"     [Quota backoff on {qid}, pausing 12s...]")
                time.sleep(12.0)
                continue
            break
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower():
                print(f"     [429 rate limit on {qid}, pausing 12s...]")
                time.sleep(12.0)
            else:
                raise exc

    if response is None:
        response = answer_question(query, limit=5)

    retrieved_raw = response.raw_evidence
    retrieved_ids = [r.get("event_id") for r in retrieved_raw if "event_id" in r]
    cited_ids = response.cited_evidence

    # 3. Compute retrieval precision against ground truth
    relevant_retrieved = [eid for eid in retrieved_ids if eid in gt_ids]
    precision_at_k = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0

    # 4. Check for hallucination
    hallucinated = [c for c in cited_ids if c.startswith("EVT_") and c not in retrieved_ids]
    is_grounded = len(hallucinated) == 0

    record_summary = {
        "id": qid,
        "category": cat,
        "query": query,
        "description": desc,
        "gt_count": len(matching_gt),
        "intent": response.intent,
        "engine": response.model_used,
        "retrieved_ids": retrieved_ids,
        "relevant_retrieved": relevant_retrieved,
        "precision_at_k": precision_at_k,
        "cited_ids": cited_ids,
        "is_grounded": is_grounded,
        "answer": response.answer,
    }
    results.append(record_summary)

    print(f"\n[{qid}] ({cat}) \"{query}\"")
    print(f"     Ground Truth Count in cricket_log.txt: {len(matching_gt)}")
    print(f"     Intent: {response.intent} | Engine: {response.model_used}")
    print(f"     Retrieved ({len(retrieved_ids)}): {retrieved_ids}")
    print(f"     Relevant in Top-{len(retrieved_ids)}: {relevant_retrieved} (Precision: {precision_at_k:.1%})")
    print(f"     Cited: {cited_ids} | Grounded: {is_grounded}")
    print(f"     Answer:\n     \"{response.answer.strip()[:180]}...\"")

    # Free tier RPM pacing delay
    time.sleep(4.5)

# Save structured benchmark output
output_path = ROOT_DIR / "scripts" / "benchmark_results_20.json"
with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, indent=2)

print("\n" + "="*90)
print(f"BENCHMARK COMPLETE. Results saved to {output_path}")
print("="*90)
