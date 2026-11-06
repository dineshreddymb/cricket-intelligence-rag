"""
ingestion/loader.py
===================
Loads the raw cricket_log.txt dataset using LangChain's TextLoader.
Preserves metadata (source path, line numbers) for complete traceability.
"""

import hashlib
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from config.settings import get_settings


def calculate_file_hash(file_path: Path) -> str:
    """Computes the SHA-256 checksum of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest().upper()


def load_cricket_log(file_path: Path | None = None) -> Tuple[List[Document], dict]:
    """
    Loads the cricket log file using LangChain's TextLoader.
    
    Returns:
        tuple: (list of line-level Document objects, file metadata dictionary)
    """
    settings = get_settings()
    path = Path(file_path) if file_path else settings.dataset_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    # Compute checksum and file stats
    file_size = path.stat().st_size
    file_checksum = calculate_file_hash(path)

    # Use LangChain TextLoader to load the raw file
    loader = TextLoader(str(path.resolve()), encoding="utf-8")
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(f"TextLoader loaded 0 documents from {path}")

    raw_text = raw_docs[0].page_content
    lines = raw_text.splitlines()

    # Package each non-empty line as a LangChain Document with rich traceability metadata
    line_documents: List[Document] = []
    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        doc = Document(
            page_content=stripped,
            metadata={
                "source": str(path.resolve()),
                "line_number": line_idx,
                "file_size_bytes": file_size,
                "file_sha256": file_checksum,
            }
        )
        line_documents.append(doc)

    metadata_summary = {
        "file_path": str(path.resolve()),
        "file_size_bytes": file_size,
        "file_sha256": file_checksum,
        "total_lines_read": len(lines),
        "non_empty_documents": len(line_documents),
    }

    return line_documents, metadata_summary
