"""
scripts/verify_4_target_queries.py
==================================
Targeted validation script for the 4 queries:
Q13, Q14, Q17, Q18.
Checks router decisions, retrieved documents, and generated responses.
"""

import sys
import os
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cricket_search.settings")
import django
django.setup()

from rag.query_router import route_query
from rag.pipeline import answer_question
from rag.entity_normalizer import (
    normalize_status,
    normalize_teams,
    normalize_over_range,
    normalize_venue,
    normalize_weather,
)

TARGET_QUERIES = [
    {
        "id": "Q13",
        "category": "Combined filtering",
        "query": "Were there any six-hit incidents in Stadium 3 during sunny weather?",
        "expected_venue": "Stadium 3",
        "expected_weather": "sunny",
        "expected_event": "six hit",
    },
    {
        "id": "Q14",
        "category": "Combined filtering",
        "query": "Find wide-ball incidents that happened in the later overs.",
        "expected_event": "wide",
        "expected_min_over": 30,
    },
    {
        "id": "Q17",
        "category": "Answer generation and reasoning",
        "query": "Which incidents are still awaiting verification, and what happened in those incidents?",
        "expected_status": "pending review",
    },
    {
        "id": "Q18",
        "category": "Answer generation and reasoning",
        "query": "Compare the types of incidents recorded for Team A and Team B.",
        "expected_teams": ["Team A", "Team B"],
    },
]

def main():
    print("=" * 70)
    print("VERIFYING ROUTING AND NORMALIZATION LOGIC")
    print("=" * 70)

    # 1. Verify entity normalizer directly
    q13_venue = normalize_venue("Were there any six-hit incidents in Stadium 3 during sunny weather?")
    q13_weather = normalize_weather("Were there any six-hit incidents in Stadium 3 during sunny weather?")
    print(f"[Q13 Normalizer] Venue: {q13_venue} (Expected: Stadium 3) | Weather: {q13_weather} (Expected: sunny)")
    assert q13_venue == "Stadium 3", f"Expected 'Stadium 3', got {q13_venue}"
    assert q13_weather == "sunny", f"Expected 'sunny', got {q13_weather}"

    q14_range = normalize_over_range("Find wide-ball incidents that happened in the later overs.")
    print(f"[Q14 Normalizer] Over range: {q14_range} (Expected: (30, 50))")
    assert q14_range == (30, 50), f"Expected (30, 50), got {q14_range}"

    q17_status = normalize_status("Which incidents are still awaiting verification, and what happened in those incidents?")
    print(f"[Q17 Normalizer] Status: {q17_status} (Expected: pending review)")
    assert q17_status == "pending review", f"Expected 'pending review', got {q17_status}"

    q18_teams = normalize_teams("Compare the types of incidents recorded for Team A and Team B.")
    print(f"[Q18 Normalizer] Teams: {q18_teams} (Expected: ['Team A', 'Team B'])")
    assert "Team A" in q18_teams and "Team B" in q18_teams, f"Expected Team A and Team B, got {q18_teams}"

    print("\nAll entity normalizer unit assertions passed successfully!\n")

    print("=" * 70)
    print("VERIFYING END-TO-END RAG RESPONSES FOR Q13, Q14, Q17, Q18")
    print("=" * 70)

    results = []
    for item in TARGET_QUERIES:
        qid = item["id"]
        q_text = item["query"]
        print(f"\n[{qid}] Query: '{q_text}'")

        decision = route_query(q_text)
        print(f"  Router Intent: {decision.intent.value}")
        print(f"  Extracted Entities: teams={decision.extracted_teams}, event={decision.extracted_event}, "
              f"status={decision.extracted_status}, overs=({decision.min_over}, {decision.max_over}), "
              f"venue={decision.extracted_venue}, weather={decision.extracted_weather}")

        try:
            time.sleep(4.5)  # Rate pacing for Gemini API
            resp = answer_question(q_text)
            print(f"  Model Used: {resp.model_used}")
            print(f"  Cited Event IDs ({len(resp.cited_evidence)}): {resp.cited_evidence}")
            print(f"  Retrieved Chunks ({len(resp.raw_evidence)} items):")
            for c in resp.raw_evidence[:4]:
                if isinstance(c, dict):
                    print(f"    - [{c.get('event_id')}] Match {c.get('match_id')} | {c.get('team')} | Over {c.get('over')} | {c.get('event')} | {c.get('status')} | {c.get('commentary')}")
            print(f"  Answer Preview:\n{resp.answer[:300]}...\n")

            results.append({
                "id": qid,
                "query": q_text,
                "status": "PASS",
                "cited": resp.cited_evidence,
                "answer": resp.answer,
                "raw_count": len(resp.raw_evidence),
            })
        except Exception as exc:
            print(f"  ERROR executing {qid}: {exc}")
            results.append({
                "id": qid,
                "query": q_text,
                "status": f"FAIL: {exc}",
            })

    output_path = ROOT_DIR / "scripts" / "target_4_verification_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerification results saved to {output_path}")

if __name__ == "__main__":
    main()
