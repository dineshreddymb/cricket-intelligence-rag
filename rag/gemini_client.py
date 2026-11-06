"""
rag/gemini_client.py
====================
Integration with Google Gemini using the official google-genai SDK.
Implements strict evidence grounding, anti-hallucination guardrails,
citation enforcement, and graceful error/quota fallback.
"""

import logging
from typing import Dict, Any, Optional
from config.settings import get_settings
from .context_builder import GroundedEvidencePackage

logger = logging.getLogger(__name__)

_gemini_client_instance: "GeminiClient | None" = None

SYSTEM_INSTRUCTION = (
    "You are an expert AI cricket assistant answering user questions based exclusively on the "
    "provided match event logs. Follow these strict rules:\n"
    "1. Answer ONLY using the facts stated in the provided [Evidence EVT_XXXX] blocks.\n"
    "2. Always cite the specific [Evidence EVT_XXXX] IDs supporting every factual statement.\n"
    "3. NEVER fabricate player roles, venues, weather, winners, or match totals not present in the evidence.\n"
    "4. If the retrieved evidence does not contain sufficient details to answer, state clearly: "
    "\"The provided records do not contain sufficient evidence to answer this question.\"\n"
    "5. Be concise, precise, and professional."
)


class GeminiClient:
    """Wrapper around official google-genai SDK with grounding enforcement."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self._sdk_client = None

        if self.api_key:
            try:
                from google import genai
                self._sdk_client = genai.Client(api_key=self.api_key)
                logger.info("Initialized official google-genai Client (model: %s)", self.model_name)
            except Exception as e:
                logger.warning("Failed to initialize google-genai SDK: %s", e)

    def generate_grounded_answer(
        self,
        query: str,
        evidence: GroundedEvidencePackage,
    ) -> Dict[str, Any]:
        """
        Submits query and evidence package to Gemini for grounded answer synthesis.
        """
        # If no evidence was retrieved
        if evidence.evidence_count == 0:
            return {
                "answer": "No relevant cricket events found in the dataset matching your question.",
                "cited_event_ids": [],
                "model_used": self.model_name,
                "grounded": True,
                "error": None,
            }

        # Format user prompt with explicit evidence injection
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"=== RETRIEVED EVIDENCE FROM DATASET ===\n"
            f"{evidence.formatted_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{query}\n\n"
            f"=== GROUNDED ANSWER (Cite supporting [Evidence EVT_XXXX] tags) ==="
        )

        if not self._sdk_client:
            # Deterministic fallback when API key is missing or SDK client not configured
            fallback_answer = (
                f"Retrieved {evidence.evidence_count} relevant events from the dataset: "
                f"{', '.join(evidence.cited_event_ids)}. "
                f"Top event: {evidence.raw_evidence_items[0].get('commentary', '') if evidence.raw_evidence_items else 'N/A'}"
            )
            return {
                "answer": fallback_answer,
                "cited_event_ids": evidence.cited_event_ids,
                "model_used": "local-fallback (Gemini API key not configured)",
                "grounded": True,
                "error": "Gemini API key missing or SDK uninitialized",
            }

        candidate_models = [self.model_name]
        for fallback in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-pro-preview"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model in candidate_models:
            try:
                response = self._sdk_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return {
                    "answer": response.text.strip(),
                    "cited_event_ids": evidence.cited_event_ids,
                    "model_used": model,
                    "grounded": True,
                    "error": None,
                }
            except Exception as exc:
                last_error = exc
                logger.warning("Gemini model %s call failed (%s). Retrying next candidate...", model, exc)
                continue

        logger.error("All Gemini candidate models failed: %s", last_error)
        fallback_answer = (
            f"Based on retrieved evidence ({', '.join(evidence.cited_event_ids)}): "
            f"{evidence.raw_evidence_items[0].get('commentary', '') if evidence.raw_evidence_items else ''} "
            f"[Note: LLM synthesis encountered a network/quota limit: {str(last_error)}]"
        )
        return {
            "answer": fallback_answer,
            "cited_event_ids": evidence.cited_event_ids,
            "model_used": self.model_name,
            "grounded": True,
            "error": str(last_error),
        }


def get_gemini_client() -> GeminiClient:
    """Singleton getter for GeminiClient."""
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = GeminiClient()
    return _gemini_client_instance
