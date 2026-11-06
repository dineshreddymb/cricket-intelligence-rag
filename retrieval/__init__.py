"""
retrieval/__init__.py
Semantic search, filtering, hydration, and retrieval evaluation package.
"""
from .semantic_search import search_cricket_events, SearchResultItem
from .bm25_search import search_bm25, BM25Index, get_bm25_index
from .hybrid_search import hybrid_search_cricket_events
from .evaluator import evaluate_retrieval_system, EvaluationReport

__all__ = [
    "search_cricket_events",
    "hybrid_search_cricket_events",
    "search_bm25",
    "BM25Index",
    "get_bm25_index",
    "SearchResultItem",
    "evaluate_retrieval_system",
    "EvaluationReport",
]
