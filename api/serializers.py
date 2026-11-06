"""
api/serializers.py
==================
Django REST Framework request and response serializers with strict input validation.
"""

from rest_framework import serializers


class SemanticSearchRequestSerializer(serializers.Serializer):
    """Input validation for semantic search requests."""
    query = serializers.CharField(
        required=True,
        min_length=2,
        max_length=500,
        help_text="Natural language query string",
    )
    limit = serializers.IntegerField(
        required=False,
        default=5,
        min_value=1,
        max_value=50,
        help_text="Maximum number of candidate records to retrieve",
    )
    match_id = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    team = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    player = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    event = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    status = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    score_threshold = serializers.FloatField(
        required=False,
        default=0.0,
        min_value=0.0,
        max_value=1.0,
        help_text="Minimum cosine similarity cutoff",
    )


class ChatRequestSerializer(serializers.Serializer):
    """Input validation for end-to-end RAG question answering."""
    query = serializers.CharField(
        required=True,
        min_length=2,
        max_length=1000,
        help_text="User question about cricket events or match statistics",
    )
    limit = serializers.IntegerField(
        required=False,
        default=5,
        min_value=1,
        max_value=20,
        help_text="Number of evidence records to supply to Gemini",
    )


class ExactQueryRequestSerializer(serializers.Serializer):
    """Input validation for structured database aggregation/filter requests."""
    match_id = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    team = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    player = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    event = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    status = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    over_min = serializers.IntegerField(required=False, min_value=1, max_value=50, allow_null=True, default=None)
    over_max = serializers.IntegerField(required=False, min_value=1, max_value=50, allow_null=True, default=None)
    limit = serializers.IntegerField(required=False, default=50, min_value=1, max_value=500)
    skip = serializers.IntegerField(required=False, default=0, min_value=0)


class IngestionTriggerSerializer(serializers.Serializer):
    """Input validation for offline data ingestion re-run."""
    recreate_qdrant = serializers.BooleanField(required=False, default=False)
