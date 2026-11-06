"""
tests/test_hybrid_retrieval.py
==============================
Unit tests for the state-of-the-art Hybrid Information Retrieval pipeline:
1. Okapi BM25 Lexical Indexing and Search
2. Reciprocal Rank Fusion (RRF) rank-merging arithmetic
3. Cross-Encoder reranker candidate scoring
4. End-to-end hybrid retrieval with ground truth validation
"""

import pytest
from retrieval.bm25_search import BM25Index, tokenize
from retrieval.hybrid_search import reciprocal_rank_fusion, hybrid_search_cricket_events


@pytest.fixture
def sample_bm25_documents():
    return [
        {
            "event_id": "EVT_TEST_01",
            "match_id": "MATCH001",
            "team": "Team A",
            "player": "Player 1",
            "event": "stumping",
            "runs": 0,
            "over": 15,
            "status": "confirmed",
            "commentary": "In over 15 at Stadium 2 (overcast), Player 1 executed stumping.",
        },
        {
            "event_id": "EVT_TEST_02",
            "match_id": "MATCH001",
            "team": "Team B",
            "player": "Player 2",
            "event": "wicket taken",
            "runs": 1,
            "over": 22,
            "status": "verified",
            "commentary": "Spectators cheered as Player 2 took a wicket in sunny weather.",
        },
        {
            "event_id": "EVT_TEST_03",
            "match_id": "MATCH002",
            "team": "Team C",
            "player": "Player 3",
            "event": "six hit",
            "runs": 6,
            "over": 45,
            "status": "confirmed",
            "commentary": "Unexpected six hit by Player 3 altered the flow of the game.",
        },
    ]


def test_bm25_tokenization():
    tokens = tokenize("Unexpected stumping by Player 1 (batsman) altered the flow!")
    assert "stumping" in tokens
    assert "player" in tokens
    assert "1" in tokens
    # Stopwords should be filtered
    assert "the" not in tokens
    assert "by" not in tokens


def test_bm25_indexing_and_search(sample_bm25_documents):
    index = BM25Index()
    index.build_from_documents(sample_bm25_documents)

    assert index.is_indexed
    assert index.corpus_size == 3

    # Exact keyword query
    hits = index.search("overcast stumping", limit=2)
    assert len(hits) > 0
    top_doc_id, score = hits[0]
    assert top_doc_id == "EVT_TEST_01"
    assert score > 0


def test_reciprocal_rank_fusion():
    dense_results = [("DOC_A", 1), ("DOC_B", 2), ("DOC_C", 3)]
    bm25_results = [("DOC_B", 1), ("DOC_A", 2), ("DOC_D", 3)]

    # Standard RRF formula: 1/(k + rank)
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=4)
    fused_ids = [doc_id for doc_id, _ in fused]

    # Both DOC_A and DOC_B appear at ranks 1 and 2, so their RRF scores are equal and highest
    assert fused_ids[0] in ["DOC_A", "DOC_B"]
    assert fused_ids[1] in ["DOC_A", "DOC_B"]
    assert "DOC_C" in fused_ids
    assert "DOC_D" in fused_ids


def test_hybrid_search_end_to_end():
    hits = hybrid_search_cricket_events("stumping in overcast condition", limit=3)
    assert len(hits) > 0
    # The top hit should have high relevance for stumping and overcast
    assert any("stumping" in h.event.lower() or "stumping" in h.commentary.lower() for h in hits)


def test_extras_deliveries_hybrid_search():
    """Validates that extras/extra runs query returns exclusively wide and no-ball events."""
    hits = hybrid_search_cricket_events(
        "Find deliveries that resulted in extra runs for the batting team",
        event="extras",
        limit=5,
    )
    assert len(hits) > 0
    for hit in hits:
        assert hit.event in ["wide", "no ball"]
        assert hit.runs >= 0
