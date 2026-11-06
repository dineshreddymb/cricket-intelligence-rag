"""
tests/test_environment.py
=========================
Module 1 environment verification tests.

These tests confirm that the development environment is correctly set up
before we write any application code. They are intentionally narrow — each
one checks a specific fact that would cause silent, confusing failures in
later modules if wrong.

We do not test:
  - Whether pytest itself is importable (it is running this file; that is proof)
  - Whether the venv is active (verify manually with `where python`)
  - Whether .gitignore exists (verify by file inspection)

We do test:
  - Python version meets the architecture requirement (3.10+)
  - The dataset and documentation files that every future module depends on
"""

import sys
import pathlib

# Resolve the project root relative to this test file.
# This file is at: <project_root>/tests/test_environment.py
# So the project root is one level up: tests/../ = project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent


class TestPythonVersion:
    """Verify that the Python interpreter meets the project requirement."""

    def test_python_version_is_310_or_higher(self):
        """
        The project architecture requires Python 3.10+.

        Reasons:
          - Structural pattern matching (match/case) is available from 3.10.
          - Type hint improvements (X | Y union syntax) require 3.10.
          - LangChain, Qdrant client, and other dependencies target 3.10+.

        If this test fails, the developer is running the wrong Python version.
        The fix is to activate the correct interpreter, not to change this test.
        """
        current = sys.version_info
        assert current >= (3, 10), (
            f"Python 3.10+ required. "
            f"Current version: {current.major}.{current.minor}.{current.micro}. "
            f"Please use the correct Python interpreter."
        )


class TestRequiredProjectFiles:
    """
    Verify that the essential project input files are present and accessible.

    These files are the inputs to every future module:
      - cricket_log.txt   → loaded in Module 2 (ingestion)
      - .env.example      → copied to .env in Module 2 (configuration)
      - docs/             → architecture reference throughout all modules

    If any of these are missing, the problem should be caught here with a
    clear message rather than discovered as a confusing error in Module 2+.
    """

    def test_dataset_file_exists(self):
        """
        cricket_log.txt must exist and be a non-empty file.

        This is the primary dataset for the entire application. Without it,
        no ingestion, no embeddings, no search, and no RAG pipeline can run.
        """
        dataset = PROJECT_ROOT / "cricket_log.txt"
        assert dataset.exists(), (
            f"Dataset not found at: {dataset}\n"
            f"cricket_log.txt is required for Module 2 onwards. "
            f"Please ensure it is present in the project root."
        )
        assert dataset.is_file(), f"Expected a file at {dataset}, but found a directory."
        assert dataset.stat().st_size > 0, (
            f"cricket_log.txt exists but is empty. "
            f"The original dataset should be approximately 100KB."
        )

    def test_env_example_exists(self):
        """
        .env.example must exist.

        This template documents which environment variables the application
        requires (GEMINI_API_KEY, MONGODB_URI, etc.) without containing
        real credentials. It is committed to Git and used as the reference
        when creating the real .env file in Module 2.
        """
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), (
            f".env.example not found at: {env_example}\n"
            f"This file documents required environment variables. "
            f"It should be present and committed to version control."
        )

    def test_docs_directory_exists(self):
        """
        docs/ directory must exist and contain the architecture documents.

        The three architecture documents define the system design, technology
        stack, and pipeline flow. They are referenced throughout implementation
        to ensure we build what was agreed.
        """
        docs_dir = PROJECT_ROOT / "docs"
        assert docs_dir.exists(), (
            f"docs/ directory not found at: {docs_dir}"
        )
        assert docs_dir.is_dir(), (
            f"Expected a directory at {docs_dir}, but found a file."
        )

        # Check that the three approved architecture documents are present
        expected_docs = [
            "01_Cricket_Log_System_Architecture.docx",
            "02_Cricket_Log_Technology_Stack.docx",
            "03_Cricket_Log_Pipeline_and_Inference_Flow.docx",
        ]
        for doc_name in expected_docs:
            doc_path = docs_dir / doc_name
            assert doc_path.exists(), (
                f"Architecture document missing: {doc_name}\n"
                f"Expected at: {doc_path}"
            )
