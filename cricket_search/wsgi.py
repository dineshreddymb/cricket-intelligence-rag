"""
cricket_search/wsgi.py
WSGI config for cricket_search project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cricket_search.settings')
application = get_wsgi_application()
