"""Document loading and chunking utilities."""

import os
from typing import List, Dict


def load_document(file_path: str) -> str:
    """Load the text content of a single file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        The file content as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents_from_directory(directory: str) -> List[Dict[str, str]]:
    """Load all ``.txt`` and ``.md`` files from *directory*.

    Args:
        directory: Path to the folder containing source documents.

    Returns:
        A list of dicts with ``content`` and ``source`` keys.
    """
    documents: List[Dict[str, str]] = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith((".txt", ".md")):
            content = load_document(filepath)
            documents.append({"content": content, "source": filename})
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split *text* into overlapping chunks.

    Args:
        text:       The input text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        A list of non-empty text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def prepare_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """Chunk each document and attach metadata.

    Args:
        documents:  Output of :func:`load_documents_from_directory`.
        chunk_size: Passed to :func:`chunk_text`.
        overlap:    Passed to :func:`chunk_text`.

    Returns:
        A list of dicts with ``content``, ``source``, and ``chunk_id`` keys.
    """
    prepared: List[Dict] = []
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            prepared.append(
                {
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_id": i,
                }
            )
    return prepared
