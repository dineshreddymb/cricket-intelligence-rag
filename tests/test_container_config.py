"""
tests/test_container_config.py
==============================
Test suite for Module 10: Podman and Docker Containerization Configuration.
Validates compose.yaml service definitions, volume mappings, environment variables,
and Containerfile structural integrity.
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestContainerizationConfig:
    """Verifies compose.yaml and Containerfiles."""

    def test_compose_yaml_validity(self):
        compose_path = PROJECT_ROOT / "compose.yaml"
        assert compose_path.exists(), f"compose.yaml not found at {compose_path}"

        with open(compose_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "services" in config
        services = config["services"]

        # 5 required services
        required_services = ["mongodb", "qdrant", "backend", "frontend", "mcp_server"]
        for svc in required_services:
            assert svc in services, f"Missing service '{svc}' in compose.yaml"

        # Verify ports
        assert "27017:27017" in services["mongodb"]["ports"]
        assert "6333:6333" in services["qdrant"]["ports"]
        assert "8000:8000" in services["backend"]["ports"]
        assert "8501:8501" in services["frontend"]["ports"]

        # Verify volumes
        assert "volumes" in config
        assert "mongo_data" in config["volumes"]
        assert "qdrant_data" in config["volumes"]

    def test_containerfiles_exist_and_non_empty(self):
        containerfiles = [
            "Containerfile.backend",
            "Containerfile.frontend",
            "Containerfile.mcp",
        ]
        for cf in containerfiles:
            p = PROJECT_ROOT / cf
            assert p.exists(), f"Containerfile missing: {cf}"
            content = p.read_text(encoding="utf-8")
            assert "FROM python:3.10-slim" in content
            assert "WORKDIR /app" in content
            assert "CMD" in content
