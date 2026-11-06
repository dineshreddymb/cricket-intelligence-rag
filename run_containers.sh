#!/usr/bin/env bash
# =============================================================================
# run_containers.sh — Unix/Linux/macOS script to launch containerized services
# Detects podman compose or docker compose automatically.
# =============================================================================

set -e

echo "Checking container runtime..."

if command -v podman &> /dev/null; then
    echo "Using Podman Compose..."
    podman compose -f compose.yaml up -d --build
elif command -v docker &> /dev/null; then
    echo "Using Docker Compose..."
    docker compose -f compose.yaml up -d --build
else
    echo "Error: Neither Podman nor Docker was found on PATH."
    exit 1
fi

echo ""
echo "Services launching:"
echo "  - Django REST API:    http://localhost:8000/api/v1/health/"
echo "  - Streamlit Frontend: http://localhost:8501"
echo "  - Qdrant Vector DB:   http://localhost:6333/dashboard"
echo "  - MongoDB:            mongodb://localhost:27017"
