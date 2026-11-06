"""
cricket_search/settings.py
==========================
Django configuration settings for Cricket Log Semantic Search API backend.
Integrates Django REST Framework and application configuration from config.settings.
"""

from pathlib import Path
from config.settings import get_settings

app_config = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = app_config.django_secret_key
DEBUG = app_config.django_debug
ALLOWED_HOSTS = [h.strip() for h in app_config.django_allowed_hosts.split(",") if h.strip()] or ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",
    "api.apps.ApiConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "cricket_search.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [],
        },
    },
]

WSGI_APPLICATION = "cricket_search.wsgi.application"
ASGI_APPLICATION = "cricket_search.asgi.application"

# Internal SQLite in-memory database for Django internals (MongoDB is our authoritative domain database)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
    "UNAUTHENTICATED_USER": None,
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
