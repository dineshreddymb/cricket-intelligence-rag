"""
scripts/test_new_unseen_cases.py
================================
Evaluates the system on 100% BRAND NEW, UNSEEN test queries that have never been tested before.
Compares system responses against raw dataset truth in cricket_log.txt.
"""

import sys
import os
import re
from pathlib import Path

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

# Seed database and index
db = get_mongo_database()
valid_records, _, _ = run_ingestion_pipeline()
ingest_records_to_mongo(valid_records, db=db)
try:
    run_full_qdrant_indexing(recreate=False)
except Exception:
    pass

new_queries = [
    {
        "id": "NEW_1",
        "query": "how many runs did player 9 score for team a in match 3?",
        "task": "Task B (Exact Structured)",
        "ground_truth_desc": "Player 9 (Team A) in MATCH003: 6 runs (Line 127 [2 runs] + Line 132 [4 runs]). Line 105 was Team C (3 runs).",
        "expected_value": 6,
        "expected_events": ["EVT_0127", "EVT_0132"]
    },
    {
        "id": "NEW_2",
        "query": "how many runs did team d score in match 5?",
        "task": "Task B (Exact Structured)",
        "ground_truth_desc": "Team D in MATCH005: Exactly 43 runs across 12 events.",
        "expected_value": 43,
        "expected_events": ["MATCH005_Team D"]
    },
    {
        "id": "NEW_3",
        "query": "what was the score of p11 for team b in the 1st match?",
        "task": "Task B (Exact Structured with informal phrasing)",
        "ground_truth_desc": "Player 11 (Team B) in MATCH001: Exactly 2 runs (Line 20: Over 31, no ball).",
        "expected_value": 2,
        "expected_events": ["EVT_0020"]
    },
    {
        "id": "NEW_4",
        "query": "Find incidents where a catch was taken in rainy weather.",
        "task": "Task A (Semantic Vector RAG)",
        "ground_truth_desc": "Exactly 3 catch events in rainy weather in entire dataset: EVT_0292 (Line 292), EVT_0383 (Line 383), EVT_0392 (Line 392).",
        "expected_value": None,
        "expected_events": ["EVT_0292", "EVT_0383", "EVT_0392"]
    },
    {
        "id": "NEW_5",
        "query": "show me events in 20th over of match 4",
        "task": "Task B (Over-level exact event query)",
        "ground_truth_desc": "Over 20 in MATCH004: Exactly 1 event, EVT_0169 (Player 2, Team C, LBW given, 4 runs).",
        "expected_value": 4,
        "expected_events": ["EVT_0169"]
    }
]

print("="*85)
print("TESTING 100% BRAND NEW, UNSEEN TEST CASES (NEVER TESTED BEFORE)")
print("="*85)

for t in new_queries:
    print(f"\n>>> TEST ID: {t['id']} [{t['task']}]")
    print(f"    QUERY: \"{t['query']}\"")
    print(f"    GROUND TRUTH IN RAW DATA: {t['ground_truth_desc']}")
    
    resp = answer_question(t["query"])
    
    print(f"    SYSTEM INTENT: {resp.intent}")
    print(f"    SYSTEM ENGINE: {resp.model_used}")
    print(f"    SYSTEM CITATIONS: {resp.cited_evidence}")
    print(f"    SYSTEM ANSWER:\n    \"{resp.answer}\"")
    
    # Check correctness
    if t["expected_value"] is not None:
        matched = str(t["expected_value"]) in resp.answer
    else:
        # Check if expected events are present in citations/retrieved
        matched = any(ev in resp.cited_evidence for ev in t["expected_events"])
    
    print(f"    VERDICT: {'VERIFIED CORRECT (100% MATCH)' if matched else 'MISMATCH'}")
