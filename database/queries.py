"""
database/queries.py
===================
Authoritative structured database query functions.
These deterministic functions are used to answer exact numerical, statistical,
and filtered queries without depending on vector Top-K or LLM estimation.
"""

from typing import List, Dict, Any, Optional
from pymongo.database import Database
from .mongo_client import get_mongo_database
from .schema import CRICKET_EVENTS_COLLECTION


def get_collection(db: Optional[Database] = None):
    target_db = db if db is not None else get_mongo_database()
    return target_db[CRICKET_EVENTS_COLLECTION]


def get_records_by_ids(event_ids: List[str], db: Optional[Database] = None) -> Dict[str, Dict[str, Any]]:
    """
    Hydrates full authoritative records from MongoDB given a list of event_ids.
    Used by the RAG retrieval layer to resolve vector candidate IDs to full documents.
    """
    if not event_ids:
        return {}

    col = get_collection(db)
    cursor = col.find({"event_id": {"$in": event_ids}}, {"_id": 0})
    return {doc["event_id"]: doc for doc in cursor}


def get_match_summary(match_id: str, db: Optional[Database] = None) -> Dict[str, Any]:
    """
    Calculates exact totals for a match:
    total runs, runs by team, events breakdown, and event count.
    """
    col = get_collection(db)
    match_docs = list(col.find({"match_id": match_id}, {"_id": 0}))

    if not match_docs:
        return {"match_id": match_id, "found": False, "total_events": 0, "total_runs": 0}

    total_runs = sum(d["runs"] for d in match_docs)
    team_runs: Dict[str, int] = {}
    event_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}

    for d in match_docs:
        team_runs[d["team"]] = team_runs.get(d["team"], 0) + d["runs"]
        event_counts[d["event"]] = event_counts.get(d["event"], 0) + 1
        status_counts[d["status"]] = status_counts.get(d["status"], 0) + 1

    return {
        "match_id": match_id,
        "found": True,
        "total_events": len(match_docs),
        "total_runs": total_runs,
        "team_runs": team_runs,
        "event_counts": event_counts,
        "status_counts": status_counts,
    }


def get_team_runs(match_id: str, team: str, db: Optional[Database] = None) -> Dict[str, Any]:
    """Returns exact total runs scored by a specific team in a match."""
    col = get_collection(db)
    docs = list(col.find({"match_id": match_id, "team": team}, {"runs": 1, "_id": 0}))
    total_runs = sum(d.get("runs", 0) for d in docs)
    return {
        "match_id": match_id,
        "team": team,
        "event_count": len(docs),
        "total_runs": total_runs,
    }


def get_player_statistics(player: str, db: Optional[Database] = None) -> Dict[str, Any]:
    """Aggregates all events, runs, and teams for a given player."""
    col = get_collection(db)
    docs = list(col.find({"player": player}, {"_id": 0}))

    if not docs:
        return {"player": player, "found": False, "total_events": 0, "total_runs": 0}

    total_runs = sum(d["runs"] for d in docs)
    event_counts: Dict[str, int] = {}
    matches_involved = set()
    teams_involved = set()

    for d in docs:
        event_counts[d["event"]] = event_counts.get(d["event"], 0) + 1
        matches_involved.add(d["match_id"])
        teams_involved.add(d["team"])

    return {
        "player": player,
        "found": True,
        "total_events": len(docs),
        "total_runs": total_runs,
        "matches_count": len(matches_involved),
        "matches": sorted(list(matches_involved)),
        "teams": sorted(list(teams_involved)),
        "event_breakdown": event_counts,
    }


def get_event_counts_by_status(
    status: Optional[str] = None,
    event: Optional[str] = None,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """Exact count of events filtered by status and/or event type."""
    col = get_collection(db)
    query: Dict[str, Any] = {}
    if status:
        query["status"] = status
    if event:
        query["event"] = event

    count = col.count_documents(query)
    sample_records = list(col.find(query, {"_id": 0}).limit(10))

    return {
        "filter": {"status": status, "event": event},
        "matching_count": count,
        "sample_records": sample_records,
    }


def get_player_highest_score(
    player: str,
    team: Optional[str] = None,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """
    Calculates the exact highest score in a single match for a player,
    optionally filtered by team. Also returns the events in that match and match-by-match totals.
    """
    col = get_collection(db)
    query: Dict[str, Any] = {"player": player}
    if team:
        query["team"] = team

    docs = list(col.find(query, {"_id": 0}))
    if not docs:
        return {
            "player": player,
            "team": team,
            "found": False,
            "highest_score": 0,
            "highest_match_id": None,
            "total_runs": 0,
            "event_count": 0,
            "highest_match_events": [],
            "highest_event_ids": [],
            "match_scores": {},
        }

    from collections import defaultdict
    match_runs: Dict[str, int] = defaultdict(int)
    match_events: Dict[str, list] = defaultdict(list)

    for d in docs:
        m = d["match_id"]
        match_runs[m] += d.get("runs", 0)
        match_events[m].append(d)

    sorted_matches = sorted(match_runs.items(), key=lambda x: x[1], reverse=True)
    best_match_id, best_score = sorted_matches[0]
    best_events = match_events[best_match_id]
    best_event_ids = [e["event_id"] for e in best_events]

    return {
        "player": player,
        "team": team,
        "found": True,
        "highest_score": best_score,
        "highest_match_id": best_match_id,
        "total_runs": sum(d.get("runs", 0) for d in docs),
        "event_count": len(docs),
        "highest_match_events": best_events,
        "highest_event_ids": best_event_ids,
        "match_scores": dict(sorted_matches),
    }


def get_tournament_highest_innings(
    limit: int = 5,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Calculates the highest individual scores by a player in any single match across the tournament."""
    col = get_collection(db)
    docs = list(col.find({}, {"_id": 0}))
    from collections import defaultdict
    innings_runs: Dict[tuple, int] = defaultdict(int)
    innings_events: Dict[tuple, list] = defaultdict(list)

    for d in docs:
        key = (d["player"], d["match_id"], d["team"])
        innings_runs[key] += d.get("runs", 0)
        innings_events[key].append(d["event_id"])

    sorted_innings = sorted(innings_runs.items(), key=lambda x: x[1], reverse=True)
    results = []
    for (p, m, t), runs in sorted_innings[:limit]:
        results.append({
            "player": p,
            "match_id": m,
            "team": t,
            "total_runs": runs,
            "event_ids": innings_events[(p, m, t)],
        })
    return results


def get_tournament_top_run_scorers(
    limit: int = 5,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """Calculates cumulative run scorers across the entire tournament."""
    col = get_collection(db)
    docs = list(col.find({}, {"_id": 0}))
    from collections import defaultdict
    player_runs: Dict[str, int] = defaultdict(int)
    player_events: Dict[str, int] = defaultdict(int)
    player_matches: Dict[str, set] = defaultdict(set)

    for d in docs:
        p = d["player"]
        player_runs[p] += d.get("runs", 0)
        player_events[p] += 1
        player_matches[p].add(d["match_id"])

    sorted_players = sorted(player_runs.items(), key=lambda x: x[1], reverse=True)
    results = []
    for p, runs in sorted_players[:limit]:
        results.append({
            "player": p,
            "total_runs": runs,
            "event_count": player_events[p],
            "matches_count": len(player_matches[p]),
        })
    return results


DISMISSAL_EVENTS = ["wicket taken", "stumping", "LBW given", "catch taken"]


def get_top_performers(
    event_types: Optional[List[str]] = None,
    metric: str = "count",   # 'count' (for event frequency) or 'sum' (for runs scored)
    group_by: str = "player", # 'player' or 'team'
    limit: int = 5,
    team: Optional[str] = None,
    match_id: Optional[str] = None,
    status: Optional[str] = None,
    db: Optional[Database] = None,
) -> List[Dict[str, Any]]:
    """
    Generalized, dynamic aggregation across all tournament records.
    Supports counting any event types (stumpings, wickets, sixes, boundaries, all dismissals)
    or summing metrics (runs) without hardcoding individual statistics.
    """
    col = get_collection(db)
    query: Dict[str, Any] = {}
    if event_types:
        query["event"] = {"$in": event_types} if len(event_types) > 1 else event_types[0]
    if team:
        query["team"] = team
    if match_id:
        query["match_id"] = match_id
    if status:
        query["status"] = status

    docs = list(col.find(query, {"_id": 0}))
    from collections import defaultdict

    entity_metrics: Dict[str, int] = defaultdict(int)
    entity_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    entity_event_ids: Dict[str, List[str]] = defaultdict(list)
    entity_matches: Dict[str, set] = defaultdict(set)
    entity_teams: Dict[str, set] = defaultdict(set)

    for d in docs:
        key = d.get(group_by, "Unknown")
        if metric == "sum":
            entity_metrics[key] += d.get("runs", 0)
        else:
            entity_metrics[key] += 1
        entity_events[key].append(d)
        entity_event_ids[key].append(d["event_id"])
        entity_matches[key].add(d.get("match_id", ""))
        entity_teams[key].add(d.get("team", ""))

    sorted_entities = sorted(entity_metrics.items(), key=lambda x: x[1], reverse=True)
    results = []
    for entity, val in sorted_entities[:limit]:
        results.append({
            group_by: entity,
            "metric_value": val,
            "metric_type": metric,
            "event_count": len(entity_events[entity]),
            "event_ids": entity_event_ids[entity],
            "matches_count": len(entity_matches[entity]),
            "matches": sorted(list(entity_matches[entity])),
            "teams": sorted(list(entity_teams[entity])),
            "events_sample": entity_events[entity][:5],
        })
    return results


def get_tournament_top_wicket_takers(limit: int = 5, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Calculates top wicket takers across the entire tournament."""
    return get_top_performers(event_types=["wicket taken"], metric="count", group_by="player", limit=limit, db=db)


def get_tournament_top_stumpings(limit: int = 5, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Calculates top stumping executors across the entire tournament."""
    return get_top_performers(event_types=["stumping"], metric="count", group_by="player", limit=limit, db=db)


def get_tournament_top_dismissals(limit: int = 5, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Calculates leading players involved in all dismissals (wickets, stumpings, LBW, catches)."""
    return get_top_performers(event_types=DISMISSAL_EVENTS, metric="count", group_by="player", limit=limit, db=db)


def get_tournament_top_catches(limit: int = 5, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Calculates top catch takers across the entire tournament."""
    return get_top_performers(event_types=["catch taken"], metric="count", group_by="player", limit=limit, db=db)


def get_tournament_top_extras(limit: int = 5, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Returns top teams associated with the most extras (wides and no balls)."""
    return get_top_performers(
        event_types=["wide", "no ball"],
        metric="sum",
        group_by="team",
        limit=limit,
        db=db,
    )


def filter_events(
    match_id: Optional[str] = None,
    team: Optional[str] = None,
    player: Optional[str] = None,
    event: Optional[str] = None,
    status: Optional[str] = None,
    over_min: Optional[int] = None,
    over_max: Optional[int] = None,
    limit: int = 50,
    skip: int = 0,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """Flexible structured filtering across authoritative records."""
    col = get_collection(db)
    query: Dict[str, Any] = {}

    if match_id:
        query["match_id"] = match_id
    if team:
        query["team"] = team
    if player:
        query["player"] = player
    if event:
        if event in ("extras", "extra"):
            query["event"] = {"$in": ["wide", "no ball"]}
        elif event in ("dismissal", "dismissals"):
            query["event"] = {"$in": ["wicket taken", "stumping", "LBW given", "catch taken"]}
        elif event in ("boundary", "boundaries"):
            query["event"] = {"$in": ["boundary hit", "six hit"]}
        else:
            query["event"] = event
    if status:
        query["status"] = status

    if over_min is not None or over_max is not None:
        query["over"] = {}
        if over_min is not None:
            query["over"]["$gte"] = over_min
        if over_max is not None:
            query["over"]["$lte"] = over_max

    total_matching = col.count_documents(query)
    records = list(col.find(query, {"_id": 0}).skip(skip).limit(limit))

    return {
        "query": query,
        "total_matching": total_matching,
        "returned_count": len(records),
        "skip": skip,
        "limit": limit,
        "records": records,
    }
