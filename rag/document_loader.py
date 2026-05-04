"""
Document Loader — reads .txt and .pdf files from a folder,
cleans them, and splits them into overlapping word-based chunks.

Each chunk is returned as (chunk_text, metadata_dict) so the indexer
can build the vector store and still know which source file each chunk
came from (useful for citing sources in answers).

Supported formats
-----------------
.txt  — read directly (UTF-8, errors ignored)
.pdf  — extracted via pypdf if installed, skipped otherwise

Chunking strategy
-----------------
Word-based sliding window with configurable chunk_size and overlap.
Using words (not tokens) keeps the implementation library-free and
produces predictable chunk sizes across languages.

    chunk_size = 200 words, overlap = 50 words
    → each chunk is ~1-2 short paragraphs
    → adjacent chunks share 50 words to avoid splitting mid-sentence
"""

import re
from pathlib import Path
from typing import List, Tuple


# -------------------------------------------------------------------------
# Type alias for clarity throughout the codebase
# -------------------------------------------------------------------------
Chunk = Tuple[str, dict]   # (text, {"source": filename, "chunk_id": int})


def load_documents(
    docs_dir: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Load all .txt (and .pdf if pypdf is installed) files from docs_dir.

    Args:
        docs_dir:   path to the folder containing document files
        chunk_size: target words per chunk
        overlap:    words shared between consecutive chunks

    Returns:
        list of (chunk_text, metadata) tuples, sorted by source filename
        then chunk index within that file.

    Raises:
        FileNotFoundError: if docs_dir does not exist
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents directory '{docs_dir}' not found. "
            "Create it and add .txt files before running ingest.py."
        )

    results: List[Chunk] = []

    # Sorted for deterministic ordering (important for reproducibility)
    files = sorted(docs_path.iterdir())
    if not files:
        raise ValueError(f"No files found in '{docs_dir}'")

    loaded_count = 0
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            raw_text = _read_pdf(file_path)
            if raw_text is None:
                continue  # pypdf not installed, skip silently
        else:
            continue  # skip .pyc, .DS_Store, etc.

        # Normalise whitespace: collapse runs of spaces/newlines to single space
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            print(f"[Loader] {file_path.name}: empty after cleaning, skipped")
            continue

        chunks = _split_into_chunks(text, chunk_size, overlap)
        for chunk_id, chunk_text in enumerate(chunks):
            results.append((
                chunk_text,
                {"source": file_path.name, "chunk_id": chunk_id},
            ))

        print(f"[Loader] {file_path.name}: {len(chunks)} chunks")
        loaded_count += 1

    if loaded_count == 0:
        raise ValueError(
            f"No readable .txt or .pdf files found in '{docs_dir}'. "
            "Add at least one .txt file."
        )

    print(f"[Loader] Loaded {loaded_count} file(s) → {len(results)} total chunks")
    return results


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Sliding-window word chunker.

    Returns a list of non-empty text chunks.  The last chunk may be shorter
    than chunk_size if the document does not divide evenly.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)   # step > 0 guaranteed
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step

    return chunks


def _read_pdf(file_path: Path) -> str | None:
    """
    Extract plain text from a PDF using pypdf.

    Returns None (and prints a warning) if pypdf is not installed.
    """
    try:
        from pypdf import PdfReader          # type: ignore
    except ImportError:
        print(
            f"[Loader] Warning: pypdf not installed — skipping PDF '{file_path.name}'. "
            "Install with: pip install pypdf"
        )
        return None

    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)
    return " ".join(pages_text)
