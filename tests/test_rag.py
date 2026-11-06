"""
tests/test_rag.py
=================
Test suite for Module 6: RAG Pipeline, Query Routing, and Gemini Grounding.
Verifies intent classification, exact structured routing to MongoDB,
and evidence-grounded semantic response formatting.
"""

import pytest
from qdrant_client import QdrantClient

from database.mongo_client import get_mongo_database
from database.ingest_mongo import ingest_records_to_mongo
from ingestion.pipeline import run_ingestion_pipeline
from vector_store.embedder import get_embedder
from vector_store.indexer import index_cricket_records
from rag.query_router import route_query, QueryIntent
from rag.context_builder import build_evidence_context
from rag.pipeline import answer_question
from retrieval.semantic_search import search_cricket_events


@pytest.fixture(scope="module")
def rag_test_environment():
    """Seeds both MongoDB and Qdrant in-memory with full dataset."""
    db = get_mongo_database()
    valid_records, _, _ = run_ingestion_pipeline()
    ingest_records_to_mongo(valid_records, db=db)

    client = QdrantClient(":memory:")
    embedder = get_embedder()
    col = "rag_test_cricket_events"

    index_cricket_records(
        records=valid_records,
        client=client,
        embedder=embedder,
        collection_name=col,
        recreate=True,
    )

    yield {
        "client": client,
        "embedder": embedder,
        "collection": col,
    }


class TestQueryRouter:
    """Verifies intent classification and entity parsing."""

    def test_exact_aggregation_intent(self):
        decision = route_query("How many runs did Team B score in MATCH001?")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED
        assert decision.extracted_match_id == "MATCH001"
        assert decision.extracted_team == "Team B"

    def test_player_stats_exact_intent(self):
        decision = route_query("Total runs and statistics for Player 9")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED
        assert decision.extracted_player == "Player 9"

    def test_semantic_exploration_intent(self):
        decision = route_query("Tell me about controversial stumping events in overcast conditions")
        assert decision.intent == QueryIntent.SEMANTIC_RAG
        assert decision.extracted_match_id is None

    def test_hybrid_filtered_intent(self):
        decision = route_query("Show me boundary hit events in MATCH002")
        assert decision.intent == QueryIntent.HYBRID
        assert decision.extracted_match_id == "MATCH002"
        assert decision.extracted_event == "boundary hit"

    def test_superlative_routing_player_and_team(self):
        decision = route_query("what is the Player 4 (Team A) highest score !")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED
        assert decision.extracted_player == "Player 4"
        assert decision.extracted_team == "Team A"

    def test_superlative_routing_overall_player(self):
        decision = route_query("What is the highest score by Player 4?")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED
        assert decision.extracted_player == "Player 4"

    def test_superlative_routing_tournament_century(self):
        decision = route_query("Which player scored the highest century in the tournament, and in which match did it happen?")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED

    def test_superlative_routing_tournament_top_scorer(self):
        decision = route_query("Who scored the most runs in the tournament?")
        assert decision.intent == QueryIntent.EXACT_STRUCTURED


class TestEvidenceContextBuilder:
    """Verifies bounded context formatting and limitation disclosures."""

    def test_context_formatting_with_citations(self, rag_test_environment):
        client = rag_test_environment["client"]
        embedder = rag_test_environment["embedder"]
        col = rag_test_environment["collection"]

        hits = search_cricket_events(
            query="LBW given pending review",
            limit=3,
            client=client,
            embedder=embedder,
            collection_name=col,
        )

        pkg = build_evidence_context(hits, max_items=3)
        assert pkg.evidence_count == 3
        assert len(pkg.cited_event_ids) == 3
        assert "[Evidence EVT_" in pkg.formatted_context
        assert "Similarity Score:" in pkg.formatted_context
        assert len(pkg.limitations) > 0


class TestRAGPipelineExecution:
    """Verifies end-to-end question answering paths."""

    def test_exact_pipeline_team_runs(self, rag_test_environment):
        # Exact structured path
        response = answer_question("How many runs did Team B score in MATCH001?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert "MATCH001" in response.answer
        assert "Team B" in response.answer
        assert "runs" in response.answer
        assert response.model_used == "MongoDB Aggregation Engine"

    def test_exact_pipeline_match_summary(self, rag_test_environment):
        response = answer_question("What is the total score in MATCH001?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert "MATCH001" in response.answer
        assert "Total runs" in response.answer
        assert response.model_used == "MongoDB Aggregation Engine"

    def test_exact_pipeline_player_highest_score_team_a(self, rag_test_environment):
        response = answer_question("what is the Player 4 (Team A) highest score !")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "Player 4" in response.answer
        assert "9 runs" in response.answer
        assert "MATCH002" in response.answer
        assert "34 runs" in response.answer
        assert "14 recorded events" in response.answer
        assert "EVT_0059" in response.cited_evidence
        assert "EVT_0075" in response.cited_evidence

    def test_exact_pipeline_player_highest_score_overall(self, rag_test_environment):
        response = answer_question("What is the highest score by Player 4?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "Player 4" in response.answer
        assert "25 runs" in response.answer
        assert "MATCH005" in response.answer
        assert len(response.cited_evidence) > 0

    def test_exact_pipeline_century_inquiry(self, rag_test_environment):
        response = answer_question("Which player scored the highest century in the tournament, and in which match did it happen?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "No player scored a century" in response.answer
        assert len(response.cited_evidence) > 0

    def test_exact_pipeline_tournament_top_scorer(self, rag_test_environment):
        response = answer_question("Who scored the most runs in the tournament?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "Player 1" in response.answer
        assert "177 total runs" in response.answer

    def test_exact_pipeline_most_stumpings(self, rag_test_environment):
        response = answer_question("Who took the most stumpings in the tournament?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "Player 1" in response.answer
        assert "10 stumpings" in response.answer
        assert len(response.cited_evidence) > 0

    def test_exact_pipeline_most_wickets(self, rag_test_environment):
        response = answer_question("Which bowler took the most wickets in the tournament?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "Player 2" in response.answer
        assert "8 wickets" in response.answer
        assert len(response.cited_evidence) > 0

    def test_exact_pipeline_most_dismissals(self, rag_test_environment):
        response = answer_question("Which player took the most dismissals in the tournament?")
        assert response.intent == QueryIntent.EXACT_STRUCTURED.value
        assert response.model_used == "MongoDB Aggregation Engine"
        assert "Player 9" in response.answer
        assert "24 total dismissals" in response.answer
        assert len(response.cited_evidence) > 0

    def test_semantic_pipeline_execution(self, rag_test_environment):
        client = rag_test_environment["client"]
        embedder = rag_test_environment["embedder"]
        col = rag_test_environment["collection"]

        response = answer_question(
            query="What happened during LBW decisions that are pending review?",
            limit=3,
            client=client,
            embedder=embedder,
            collection_name=col,
        )

        assert len(response.answer) > 0
        assert len(response.cited_evidence) > 0
        assert "EVT_" in response.cited_evidence[0]
        assert "500 total dataset records" in response.retrieval_scope
        assert len(response.limitations) > 0
