"""
scripts/analyze_new_20_queries_ground_truth.py
=============================================
Deep ground-truth inspection of cricket_log.txt for all 20 new test queries (Q26-Q45).
Parses every line and finds the true matching records.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT_DIR / "cricket_log.txt"

def parse_all_records() -> List[Dict[str, Any]]:
    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            # Timestamp
            ts_m = re.match(r"^\[(.*?)\]", line)
            ts = ts_m.group(1) if ts_m else ""

            # Match
            m_match = re.search(r"Match:(MATCH\d{3})", line)
            match_id = m_match.group(1) if m_match else ""

            # Team
            m_team = re.search(r"Team:(Team [A-D])", line)
            team = m_team.group(1) if m_team else ""

            # Player
            m_player = re.search(r"Player:(Player \d+)", line)
            player = m_player.group(1) if m_player else ""

            # Event
            m_event = re.search(r"Event:(.*?)\s+Runs:", line)
            event = m_event.group(1).strip() if m_event else ""

            # Runs
            m_runs = re.search(r"Runs:(\d+)", line)
            runs = int(m_runs.group(1)) if m_runs else 0

            # Over
            m_over = re.search(r"Over:(\d+)", line)
            over = int(m_over.group(1)) if m_over else 0

            # Status
            m_status = re.search(r"Status:(.*?)\s+Commentary:", line)
            status = m_status.group(1).strip() if m_status else ""

            # Commentary
            m_comm = re.search(r"Commentary:(.*)$", line)
            commentary = m_comm.group(1).strip() if m_comm else ""

            # Role from commentary: e.g. "Player 7 (all-rounder)" or "Player 5 ()"
            role_m = re.search(rf"{player}\s*\((.*?)\)", commentary)
            role = role_m.group(1).strip() if role_m else None
            if role == "":
                role = "missing"

            # Venue and weather from commentary: e.g. "In over 15 at Stadium 2 (overcast)" or "at (sunny)"
            venue = None
            venue_m = re.search(r"\bat\s+(Stadium\s+\d+)", commentary, re.IGNORECASE)
            if venue_m:
                venue = venue_m.group(1)

            weather = None
            weather_m = re.search(r"\((sunny|rainy|overcast|windy)\)", commentary, re.IGNORECASE)
            if weather_m:
                weather = weather_m.group(1).lower()

            # Empty weather or venue
            has_empty_stadium_weather = bool(re.search(r"\bat\s+\((sunny|rainy|overcast|windy)\)", commentary, re.IGNORECASE))

            event_id = f"EVT_{idx:04d}"

            records.append({
                "line": idx,
                "event_id": event_id,
                "timestamp": ts,
                "match_id": match_id,
                "team": team,
                "player": player,
                "event": event,
                "runs": runs,
                "over": over,
                "status": status,
                "commentary": commentary,
                "role": role,
                "venue": venue,
                "weather": weather,
                "has_empty_stadium_weather": has_empty_stadium_weather,
            })
    return records

def analyze():
    records = parse_all_records()
    print(f"Total parsed records: {len(records)}")

    # Q26: What kind of incidents was Player 10 involved in across the matches?
    p10_recs = [r for r in records if r["player"] == "Player 10"]
    p10_events = defaultdict(int)
    for r in p10_recs:
        p10_events[r["event"]] += 1
    print(f"\n[Q26] Player 10 Total Records: {len(p10_recs)}, Event Breakdown: {dict(p10_events)}")

    # Q27: Can you show me all the recorded activities involving Player 5 from Team C?
    p5_c_recs = [r for r in records if r["player"] == "Player 5" and r["team"] == "Team C"]
    print(f"\n[Q27] Player 5 from Team C: {len(p5_c_recs)} records:")
    for r in p5_c_recs:
        print(f"  - {r['event_id']}: Match {r['match_id']} | Over {r['over']} | {r['event']} ({r['runs']} runs) | Status: {r['status']}")

    # Q28: Find incidents where players identified as bowlers were involved in cricket events.
    bowler_recs = [r for r in records if r["role"] == "bowler"]
    print(f"\n[Q28] Bowler records: {len(bowler_recs)} records across dataset.")
    for r in bowler_recs[:5]:
        print(f"  - {r['event_id']}: {r['player']} (bowler) | {r['event']} | {r['status']}")

    # Q29: Were any fielders involved in stumping or catch-related incidents?
    stump_catch = [r for r in records if r["event"] in ("stumping", "catch taken")]
    print(f"\n[Q29] Stumping or catch taken records: {len(stump_catch)} records. Roles present:")
    roles_in_sc = defaultdict(int)
    for r in stump_catch:
        roles_in_sc[r["role"]] += 1
    print(f"  Roles: {dict(roles_in_sc)}")

    # Q30: Which players were involved in multiple types of events in the same match?
    player_match_events = defaultdict(lambda: defaultdict(set))
    for r in records:
        player_match_events[r["match_id"]][r["player"]].add(r["event"])
    multi_event_players = []
    for m, pdict in player_match_events.items():
        for p, evs in pdict.items():
            if len(evs) > 1:
                multi_event_players.append((m, p, list(evs)))
    print(f"\n[Q30] Players involved in multiple event types in same match: {len(multi_event_players)} instances.")
    for m, p, evs in multi_event_players[:5]:
        print(f"  - {m}: {p} involved in {evs}")

    # Q31: Show me the cricket incidents that happened at Stadium 2.
    s2_recs = [r for r in records if r["venue"] == "Stadium 2"]
    print(f"\n[Q31] Stadium 2 Total Records: {len(s2_recs)}")
    for r in s2_recs[:5]:
        print(f"  - {r['event_id']}: {r['match_id']} | Over {r['over']} | {r['player']} | {r['event']}")

    # Q32: Were there any incidents recorded during rainy weather?
    rainy_recs = [r for r in records if r["weather"] == "rainy"]
    print(f"\n[Q32] Rainy weather Total Records: {len(rainy_recs)}")
    for r in rainy_recs[:5]:
        print(f"  - {r['event_id']}: {r['match_id']} | Over {r['over']} | {r['player']} | {r['event']} | venue: {r['venue']}")

    # Q33: Find events that took place in windy conditions at different stadiums.
    windy_recs = [r for r in records if r["weather"] == "windy"]
    windy_venues = defaultdict(list)
    for r in windy_recs:
        windy_venues[r["venue"] or "unspecified"].append(r["event_id"])
    print(f"\n[Q33] Windy weather records: {len(windy_recs)}, Venues: {dict((k, len(v)) for k, v in windy_venues.items())}")

    # Q34: Can you find incidents where the commentary mentions the weather conditions but doesn't specify the stadium?
    weather_no_venue = [r for r in records if r["has_empty_stadium_weather"]]
    print(f"\n[Q34] Weather mentioned without stadium: {len(weather_no_venue)} records.")
    for r in weather_no_venue[:5]:
        print(f"  - {r['event_id']}: {r['match_id']} | Over {r['over']} | {r['weather']} | Commentary: {r['commentary']}")

    # Q35: What happened during the first five overs of MATCH006?
    m6_early = [r for r in records if r["match_id"] == "MATCH006" and 1 <= r["over"] <= 5]
    print(f"\n[Q35] MATCH006 Overs 1-5: {len(m6_early)} records.")
    for r in m6_early:
        print(f"  - {r['event_id']}: Over {r['over']} | {r['team']} | {r['player']} | {r['event']} ({r['runs']} runs) | Status: {r['status']}")

    # Q36: Show me the incidents that took place in the final ten overs of MATCH003.
    m3_late = [r for r in records if r["match_id"] == "MATCH003" and 41 <= r["over"] <= 50]
    print(f"\n[Q36] MATCH003 Final 10 overs (41-50): {len(m3_late)} records.")
    for r in m3_late:
        print(f"  - {r['event_id']}: Over {r['over']} | {r['team']} | {r['player']} | {r['event']} ({r['runs']} runs)")

    # Q37: Were there any important events recorded on 30 September 2025?
    sep30_recs = [r for r in records if "2025-09-30" in r["timestamp"]]
    print(f"\n[Q37] 30 September 2025: {len(sep30_recs)} records.")
    for r in sep30_recs[:5]:
        print(f"  - {r['event_id']}: {r['match_id']} | Over {r['over']} | {r['player']} | {r['event']} | {r['status']}")

    # Q38: Find incidents involving Team D players between overs 35 and 45.
    d_35_45 = [r for r in records if r["team"] == "Team D" and 35 <= r["over"] <= 45]
    print(f"\n[Q38] Team D between overs 35 and 45: {len(d_35_45)} records.")
    for r in d_35_45[:5]:
        print(f"  - {r['event_id']}: Match {r['match_id']} | Over {r['over']} | {r['player']} | {r['event']} ({r['runs']} runs)")

    # Q39: Can you find boundary-hit incidents that were either disputed or awaiting review?
    b_disp_rev = [r for r in records if r["event"] == "boundary hit" and r["status"] in ("disputed", "pending review")]
    print(f"\n[Q39] Boundary hit disputed or pending review: {len(b_disp_rev)} records.")
    for r in b_disp_rev[:5]:
        print(f"  - {r['event_id']}: Match {r['match_id']} | Over {r['over']} | {r['player']} | {r['status']}")

    # Q40: Show me all confirmed incidents involving wide or no-ball deliveries.
    conf_extras = [r for r in records if r["status"] == "confirmed" and r["event"] in ("wide", "no ball")]
    print(f"\n[Q40] Confirmed wide or no-ball: {len(conf_extras)} records.")
    for r in conf_extras[:5]:
        print(f"  - {r['event_id']}: Match {r['match_id']} | Over {r['over']} | {r['event']} | {r['status']}")

    # Q41: Were there any wicket-related incidents that were marked as verified?
    ver_wickets = [r for r in records if r["status"] == "verified" and r["event"] in ("wicket taken", "stumping", "LBW given", "catch taken")]
    print(f"\n[Q41] Verified wicket-related incidents: {len(ver_wickets)} records.")
    for r in ver_wickets[:5]:
        print(f"  - {r['event_id']}: Match {r['match_id']} | Over {r['over']} | {r['event']} | {r['status']}")

    # Q42: Find examples where a player was involved in an over-completed event.
    over_comp = [r for r in records if r["event"] == "over completed"]
    print(f"\n[Q42] Over completed records: {len(over_comp)} records.")
    for r in over_comp[:5]:
        print(f"  - {r['event_id']}: Match {r['match_id']} | Over {r['over']} | {r['player']} | {r['status']}")

    # Q43: Can you find records where the commentary describes an unexpected incident?
    unexp_recs = [r for r in records if "unexpected" in r["commentary"].lower()]
    print(f"\n[Q43] Unexpected incident commentary: {len(unexp_recs)} records.")
    for r in unexp_recs[:5]:
        print(f"  - {r['event_id']}: {r['match_id']} | Over {r['over']} | {r['commentary']}")

    # Q44: Are there any records where the player's role is missing from the commentary?
    missing_role = [r for r in records if r["role"] == "missing"]
    print(f"\n[Q44] Records with missing player role '()': {len(missing_role)} records.")
    for r in missing_role[:5]:
        print(f"  - {r['event_id']}: {r['player']} | Commentary: {r['commentary']}")

    # Q45: Show me incidents where the same player was involved in different event types under different status labels.
    p_event_status = defaultdict(lambda: defaultdict(set))
    for r in records:
        p_event_status[r["player"]][r["event"]].add(r["status"])
    multi_status_players = []
    for p, edict in p_event_status.items():
        if len(edict) > 1:
            all_statuses = set()
            for s_set in edict.values():
                all_statuses.update(s_set)
            if len(all_statuses) > 1:
                multi_status_players.append((p, dict((k, list(v)) for k, v in edict.items())))
    print(f"\n[Q45] Players with different events under different statuses: {len(multi_status_players)} players.")
    for p, edict in multi_status_players[:3]:
        print(f"  - {p}: {edict}")

if __name__ == "__main__":
    analyze()
