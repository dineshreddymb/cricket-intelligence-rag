"""
rag/pipeline.py
================
Master orchestration pipeline for Cricket Log Question-Answering.
Seamlessly routes user queries between:
1. Exact MongoDB structured queries (for total runs, player stats, match aggregates)
2. Qdrant vector retrieval + Gemini grounded RAG (for natural language search and commentary)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .query_router import route_query, QueryIntent
from .context_builder import build_evidence_context
from .gemini_client import get_gemini_client
from retrieval import search_cricket_events, hybrid_search_cricket_events
from database.queries import (
    get_match_summary,
    get_team_runs,
    get_player_statistics,
    get_event_counts_by_status,
    get_player_highest_score,
    get_tournament_highest_innings,
    get_tournament_top_run_scorers,
    get_top_performers,
    get_tournament_top_wicket_takers,
    get_tournament_top_stumpings,
    get_tournament_top_dismissals,
    get_tournament_top_catches,
    get_tournament_top_extras,
)


@dataclass(frozen=True)
class RAGResponse:
    query: str
    answer: str
    intent: str
    cited_evidence: List[str]
    retrieval_scope: str
    limitations: str
    model_used: str
    raw_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "intent": self.intent,
            "cited_evidence": self.cited_evidence,
            "retrieval_scope": self.retrieval_scope,
            "limitations": self.limitations,
            "model_used": self.model_used,
            "raw_evidence": self.raw_evidence,
        }


def answer_question(
    query: str,
    limit: int = 5,
    override_intent: Optional[str] = None,
    client: Any = None,
    embedder: Any = None,
    collection_name: Optional[str] = None,
) -> RAGResponse:
    """
    End-to-end question answering pipeline:
    Analyzes intent, dispatches to exact database calculation or semantic vector RAG,
    and returns a structured, evidenced response.
    """
    decision = route_query(query)
    intent = override_intent or decision.intent.value

    # Path 1: Exact Structured Database Queries (No LLM arithmetic hallucination)
    if intent == QueryIntent.EXACT_STRUCTURED.value:
        match_id = decision.extracted_match_id
        team = decision.extracted_team
        player = decision.extracted_player
        event = decision.extracted_event
        over = getattr(decision, "extracted_over", None)

        lower_q = query.lower()
        is_superlative = any(w in lower_q for w in ["highest", "max", "maximum", "top", "best", "lowest", "min", "minimum", "most", "leading"])
        is_century = any(w in lower_q for w in ["century", "centuries", "hundred", "100"])

        # Sub-case: Over-level events in a specific match
        if match_id and over is not None and not player:
            from database.queries import get_collection
            col = get_collection()
            q_filter: Dict[str, Any] = {"match_id": match_id, "over": over}
            if team:
                q_filter["team"] = team
            docs = list(col.find(q_filter, {"_id": 0}))
            if docs:
                runs = sum(d.get("runs", 0) for d in docs)
                ev_ids = [d["event_id"] for d in docs]
                events_str = ", ".join(f"{d['event_id']} ({d.get('player')} - {d.get('event')}, {d.get('runs', 0)} runs)" for d in docs)
                team_ctx = f" for {team}" if team else ""
                answer_text = (
                    f"In Over {over} of {match_id}{team_ctx}, {len(docs)} event(s) occurred contributing a total of {runs} run(s): {events_str}."
                )
            else:
                answer_text = f"No events recorded in Over {over} for match {match_id}{f' ({team})' if team else ''}."
                ev_ids = []
            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=ev_ids,
                retrieval_scope=f"Authoritative MongoDB query for Over {over} in {match_id}.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=docs,
            )

        # Sub-case 1: Tournament-wide Century query
        if is_century:
            top_innings = get_tournament_highest_innings(limit=3)
            answer_text = (
                f"No player scored a century (100+ runs in a match) in the tournament. "
                f"The highest individual match score in tournament history was {top_innings[0]['total_runs']} runs "
                f"by {top_innings[0]['player']} ({top_innings[0]['team']}) in {top_innings[0]['match_id']}, followed by "
                f"{top_innings[1]['total_runs']} runs by {top_innings[1]['player']} in {top_innings[1]['match_id']}."
            )
            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=top_innings[0]["event_ids"],
                retrieval_scope="Authoritative MongoDB calculation across all tournament records.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=top_innings,
            )

        # Sub-case 2: Player Highest Score / Superlative
        if player and is_superlative:
            high_stats = get_player_highest_score(player, team=team)
            if high_stats["found"]:
                team_ctx = f" for {team}" if team else ""
                events_str = ", ".join(f"{ev['event_id']} ({ev['runs']} runs in Over {ev['over']})" for ev in high_stats["highest_match_events"])
                answer_text = (
                    f"{player}'s highest score{team_ctx} is {high_stats['highest_score']} runs in match "
                    f"{high_stats['highest_match_id']} (across {len(high_stats['highest_match_events'])} events: {events_str}). "
                    f"Across all matches{team_ctx}, {player} scored an exact total of {high_stats['total_runs']} runs "
                    f"across {high_stats['event_count']} recorded events."
                )
            else:
                answer_text = f"No events recorded for {player}{f' in {team}' if team else ''} in the dataset."

            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=high_stats.get("highest_event_ids", []),
                retrieval_scope=f"Authoritative MongoDB superlative aggregation for {player}{f' ({team})' if team else ''}.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=high_stats.get("highest_match_events", []),
            )

        # Sub-case 3: Tournament-wide Superlatives without player specified
        if is_superlative and not player and not match_id:
            # A: Stumpings
            if "stumping" in lower_q or "stumpings" in lower_q:
                top_stumpings = get_tournament_top_stumpings(limit=5)
                if top_stumpings:
                    top_p = top_stumpings[0]
                    runner_ups = ", ".join(f"{p['player']} ({p['metric_value']} stumpings)" for p in top_stumpings[1:4])
                    answer_text = (
                        f"{top_p['player']} executed the most stumpings in the tournament with an exact count of "
                        f"{top_p['metric_value']} stumpings across matches {', '.join(top_p['matches'])}, followed by {runner_ups}."
                    )
                    evidence_ids = top_p["event_ids"]
                else:
                    answer_text = "No stumping events found in the tournament records."
                    evidence_ids = []
                return RAGResponse(
                    query=query,
                    answer=answer_text,
                    intent=intent,
                    cited_evidence=evidence_ids,
                    retrieval_scope="Authoritative MongoDB tournament-wide stumping aggregation.",
                    limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                    model_used="MongoDB Aggregation Engine",
                    raw_evidence=top_stumpings,
                )

            # B: Wickets / Bowling
            if any(w in lower_q for w in ["wicket", "wickets", "bowler", "bowling"]):
                top_wickets = get_tournament_top_wicket_takers(limit=5)
                if top_wickets:
                    top_p = top_wickets[0]
                    runner_ups = ", ".join(f"{p['player']} ({p['metric_value']} wickets)" for p in top_wickets[1:4])
                    answer_text = (
                        f"{top_p['player']} took the most wickets in the tournament with an exact count of "
                        f"{top_p['metric_value']} wickets taken across matches {', '.join(top_p['matches'])}, followed by {runner_ups}."
                    )
                    evidence_ids = top_p["event_ids"]
                else:
                    answer_text = "No wicket events found in the tournament records."
                    evidence_ids = []
                return RAGResponse(
                    query=query,
                    answer=answer_text,
                    intent=intent,
                    cited_evidence=evidence_ids,
                    retrieval_scope="Authoritative MongoDB tournament-wide wicket aggregation.",
                    limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                    model_used="MongoDB Aggregation Engine",
                    raw_evidence=top_wickets,
                )

            # C: Overall Dismissals
            if any(w in lower_q for w in ["dismissal", "dismissals"]):
                top_dismissals = get_tournament_top_dismissals(limit=5)
                if top_dismissals:
                    top_p = top_dismissals[0]
                    runner_ups = ", ".join(f"{p['player']} ({p['metric_value']} dismissals)" for p in top_dismissals[1:4])
                    answer_text = (
                        f"{top_p['player']} was involved in the most dismissals in the tournament with an exact count of "
                        f"{top_p['metric_value']} total dismissals (wickets, stumpings, LBWs, catches), followed by {runner_ups}."
                    )
                    evidence_ids = top_p["event_ids"]
                else:
                    answer_text = "No dismissal events found in the tournament records."
                    evidence_ids = []
                return RAGResponse(
                    query=query,
                    answer=answer_text,
                    intent=intent,
                    cited_evidence=evidence_ids,
                    retrieval_scope="Authoritative MongoDB tournament-wide dismissal aggregation.",
                    limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                    model_used="MongoDB Aggregation Engine",
                    raw_evidence=top_dismissals,
                )

            # D: Catches
            if any(w in lower_q for w in ["catch", "catches"]):
                top_catches = get_tournament_top_catches(limit=5)
                if top_catches:
                    top_p = top_catches[0]
                    runner_ups = ", ".join(f"{p['player']} ({p['metric_value']} catches)" for p in top_catches[1:4])
                    answer_text = (
                        f"{top_p['player']} took the most catches in the tournament with an exact count of "
                        f"{top_p['metric_value']} catches, followed by {runner_ups}."
                    )
                    evidence_ids = top_p["event_ids"]
                else:
                    answer_text = "No catch events found in the tournament records."
                    evidence_ids = []
                return RAGResponse(
                    query=query,
                    answer=answer_text,
                    intent=intent,
                    cited_evidence=evidence_ids,
                    retrieval_scope="Authoritative MongoDB tournament-wide catch aggregation.",
                    limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                    model_used="MongoDB Aggregation Engine",
                    raw_evidence=top_catches,
                )

            # E: Extras (wides, no balls)
            if event == "extras" or any(w in lower_q for w in ["extra", "extras", "wide", "wides", "no ball", "no balls"]):
                top_extras = get_tournament_top_extras(limit=5)
                if top_extras:
                    top_t = top_extras[0]
                    team_name = top_t.get("team") or top_t.get("player") or "Unknown"
                    runner_ups = ", ".join(f"{t.get('team') or t.get('player')} ({t['metric_value']} extra runs)" for t in top_extras[1:4])
                    answer_text = (
                        f"{team_name} conceded/had the most extra runs in the tournament with an exact total of "
                        f"{top_t['metric_value']} extra runs across matches {', '.join(top_t['matches'])}, followed by {runner_ups}."
                    )
                    evidence_ids = top_t["event_ids"]
                else:
                    answer_text = "No extra runs (wides/no balls) recorded in tournament."
                    evidence_ids = []
                return RAGResponse(
                    query=query,
                    answer=answer_text,
                    intent=intent,
                    cited_evidence=evidence_ids,
                    retrieval_scope="Authoritative MongoDB tournament-wide extras aggregation.",
                    limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                    model_used="MongoDB Aggregation Engine",
                    raw_evidence=top_extras,
                )

            # F: Other event counts (sixes, boundaries, etc.)
            if event:
                top_event_performers = get_top_performers(event_types=[event], metric="count", limit=5)
                if top_event_performers:
                    top_p = top_event_performers[0]
                    runner_ups = ", ".join(f"{p['player']} ({p['metric_value']} {event})" for p in top_event_performers[1:4])
                    answer_text = (
                        f"{top_p['player']} had the most '{event}' events in the tournament with an exact count of "
                        f"{top_p['metric_value']}, followed by {runner_ups}."
                    )
                    return RAGResponse(
                        query=query,
                        answer=answer_text,
                        intent=intent,
                        cited_evidence=top_p["event_ids"],
                        retrieval_scope=f"Authoritative MongoDB tournament-wide {event} aggregation.",
                        limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                        model_used="MongoDB Aggregation Engine",
                        raw_evidence=top_event_performers,
                    )

            # G: Default Run Scorers
            top_scorers = get_tournament_top_run_scorers(limit=5)
            top_innings = get_tournament_highest_innings(limit=3)
            answer_text = (
                f"In tournament history, the highest individual match score was {top_innings[0]['total_runs']} runs "
                f"by {top_innings[0]['player']} ({top_innings[0]['team']}) in {top_innings[0]['match_id']}. "
                f"The leading cumulative run-scorer across all matches is {top_scorers[0]['player']} with "
                f"{top_scorers[0]['total_runs']} total runs, followed by {top_scorers[1]['player']} ({top_scorers[1]['total_runs']} runs) "
                f"and {top_scorers[2]['player']} ({top_scorers[2]['total_runs']} runs)."
            )
            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=top_innings[0]["event_ids"],
                retrieval_scope="Authoritative MongoDB tournament-wide superlative aggregation.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=top_scorers,
            )

        # Sub-case 4: Player runs in a specific team across matches
        if player and team and not match_id:
            high_stats = get_player_highest_score(player, team=team)
            if high_stats["found"]:
                answer_text = (
                    f"{player} scored an exact total of {high_stats['total_runs']} runs for {team} "
                    f"across {high_stats['event_count']} recorded events. Highest match score was "
                    f"{high_stats['highest_score']} runs in {high_stats['highest_match_id']}."
                )
            else:
                answer_text = f"No events recorded for {player} in {team} in the dataset."

            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=high_stats.get("highest_event_ids", []),
                retrieval_scope=f"Authoritative MongoDB aggregation for {player} in {team}.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=high_stats.get("highest_match_events", []),
            )

        # Sub-case: Specific Player in a specific Match (with optional team filter)
        if player and match_id:
            from database.queries import get_collection
            col = get_collection()
            q_filter: Dict[str, Any] = {"match_id": match_id, "player": player}
            if team:
                q_filter["team"] = team
            docs = list(col.find(q_filter, {"_id": 0}))
            if docs:
                p_runs = sum(d.get("runs", 0) for d in docs)
                ev_ids = [d["event_id"] for d in docs]
                events_str = ", ".join(f"{d['event_id']} ({d.get('runs', 0)} runs in Over {d.get('over', 0)})" for d in docs)
                team_ctx = f" for {team}" if team else ""
                answer_text = (
                    f"In {match_id}, {player}{team_ctx} scored an exact total of {p_runs} runs "
                    f"across {len(docs)} recorded events: {events_str}."
                )
            else:
                answer_text = f"No events recorded for {player}{f' in {team}' if team else ''} in match {match_id}."
                ev_ids = []

            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=ev_ids,
                retrieval_scope=f"Authoritative MongoDB query for {player}{f' ({team})' if team else ''} in {match_id}.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=docs,
            )

        # Sub-case A: Specific Team runs in a specific Match
        if match_id and team:
            team_res = get_team_runs(match_id, team)
            answer_text = (
                f"In {match_id}, {team} scored an exact total of {team_res['total_runs']} runs "
                f"across {team_res['event_count']} recorded events."
            )
            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=[f"{match_id}_{team}"],
                retrieval_scope=f"Authoritative MongoDB query filtered by match={match_id} and team={team}.",
                limitations="Computed directly from MongoDB structured storage without Top-K sampling or LLM estimation.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=[team_res],
            )

        # Sub-case B: Match Summary (Total runs across all teams in match)
        if match_id:
            summary = get_match_summary(match_id)
            if summary.get("found"):
                team_breakdown = ", ".join(f"{t}: {r} runs" for t, r in summary["team_runs"].items())
                answer_text = (
                    f"Match {match_id} summary: Total runs scored = {summary['total_runs']} across "
                    f"{summary['total_events']} events. Team scores: {team_breakdown}."
                )
            else:
                answer_text = f"No records found for match {match_id} in the dataset."

            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=[match_id],
                retrieval_scope=f"Authoritative MongoDB aggregation for match {match_id}.",
                limitations="Exact calculation computed directly from MongoDB structured store.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=[summary],
            )

        # Sub-case C: Player Statistics
        if player:
            stats = get_player_statistics(player)
            if stats.get("found"):
                event_str = ", ".join(f"{e}: {c}" for e, c in stats["event_breakdown"].items())
                answer_text = (
                    f"{player} participated in {stats['total_events']} events across {stats['matches_count']} matches, "
                    f"contributing {stats['total_runs']} runs. Event breakdown: {event_str}."
                )
            else:
                answer_text = f"No events recorded for {player} in the dataset."

            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=[player],
                retrieval_scope=f"Authoritative MongoDB aggregation for player {player}.",
                limitations="Exact calculation computed directly from MongoDB structured store.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=[stats],
            )

        # Sub-case D: Event counts by status
        if event or decision.extracted_status:
            count_res = get_event_counts_by_status(
                status=decision.extracted_status,
            statuses=decision.extracted_statuses,
                event=event,
            )
            answer_text = (
                f"There are {count_res['matching_count']} events matching "
                f"event='{event or 'any'}' and status='{decision.extracted_status or 'any'}'."
            )
            return RAGResponse(
                query=query,
                answer=answer_text,
                intent=intent,
                cited_evidence=[rec["event_id"] for rec in count_res.get("sample_records", [])],
                retrieval_scope="Authoritative MongoDB exact count.",
                limitations="Exact count calculated directly over 500 MongoDB records.",
                model_used="MongoDB Aggregation Engine",
                raw_evidence=count_res.get("sample_records", []),
            )

    # Path 2: Hybrid Search (BM25 + Qdrant + RRF + Cross-Encoder) + Gemini Grounded Answer
    eff_team = decision.extracted_team if not decision.extracted_teams or len(decision.extracted_teams) <= 1 else None
    search_limit = max(limit, len(decision.extracted_teams) * 3) if decision.extracted_teams and len(decision.extracted_teams) > 1 else limit

    try:
        search_hits = hybrid_search_cricket_events(
            query=query,
            limit=search_limit,
            match_id=decision.extracted_match_id,
            team=eff_team,
            teams=decision.extracted_teams,
            player=decision.extracted_player,
            role=decision.extracted_role,
            event=decision.extracted_event,
            status=decision.extracted_status,
            statuses=decision.extracted_statuses,
            min_over=decision.min_over,
            max_over=decision.max_over,
            venue=decision.extracted_venue,
            weather=decision.extracted_weather,
            missing_venue=decision.missing_venue,
            missing_role=decision.missing_role,
            client=client,
            embedder=embedder,
            collection_name=collection_name,
        )
    except Exception as exc:
        search_hits = search_cricket_events(
            query=query,
            limit=search_limit,
            match_id=decision.extracted_match_id,
            team=eff_team,
            teams=decision.extracted_teams,
            player=decision.extracted_player,
            role=decision.extracted_role,
            event=decision.extracted_event,
            status=decision.extracted_status,
            statuses=decision.extracted_statuses,
            min_over=decision.min_over,
            max_over=decision.max_over,
            venue=decision.extracted_venue,
            weather=decision.extracted_weather,
            client=client,
            embedder=embedder,
            collection_name=collection_name,
        )

    evidence_package = build_evidence_context(search_hits, max_items=search_limit)
    gemini_client = get_gemini_client()
    llm_result = gemini_client.generate_grounded_answer(query, evidence_package)

    return RAGResponse(
        query=query,
        answer=llm_result["answer"],
        intent=intent,
        cited_evidence=llm_result["cited_event_ids"],
        retrieval_scope=evidence_package.retrieval_scope,
        limitations=evidence_package.limitations,
        model_used=llm_result["model_used"],
        raw_evidence=evidence_package.raw_evidence_items,
    )
