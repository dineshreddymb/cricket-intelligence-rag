"""
cricket_search/urls.py
======================
Top-level URL routing configuration.
Exposes /api/v1/ endpoints routed to the api application.
"""

from django.urls import path, include

urlpatterns = [
    path("api/v1/", include("api.urls")),
]
