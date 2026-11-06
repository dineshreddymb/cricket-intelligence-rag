"""
rag/__init__.py
Retrieval-Augmented Generation (RAG) and Gemini LLM integration package.
"""
from .context_builder import build_evidence_context, GroundedEvidencePackage
from .query_router import route_query, QueryIntent, QueryRoutingDecision
from .gemini_client import get_gemini_client, GeminiClient
from .pipeline import answer_question, RAGResponse

__all__ = [
    "build_evidence_context",
    "GroundedEvidencePackage",
    "route_query",
    "QueryIntent",
    "QueryRoutingDecision",
    "get_gemini_client",
    "GeminiClient",
    "answer_question",
    "RAGResponse",
]
