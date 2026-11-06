"""
cricket_search/asgi.py
ASGI config for cricket_search project.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cricket_search.settings')
application = get_asgi_application()
