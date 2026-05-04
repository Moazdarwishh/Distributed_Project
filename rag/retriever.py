"""
RAG Retriever — query the persisted vector index.

On the first call to retrieve_context(), this module:
  1. Loads the sentence-transformer embedding model (once per process).
  2. Loads the NearestNeighbors index and chunk list from disk
     (built by running `python ingest.py`).

Every subsequent call is a fast in-memory query:
  • Embed the query (~10ms on CPU for all-MiniLM-L6-v2)
  • kNN search over the indexed embeddings (~1ms for a few hundred chunks)
  • Return the top-k chunks formatted as context for the LLM

Thread safety
-------------
The embedding model and index are loaded behind a double-checked lock so
they are initialised exactly once even when many worker threads call
retrieve_context() simultaneously on startup.
"""

import threading
import time
from typing import List

import numpy as np

from rag.indexer import load_index, index_exists

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons — set once, never replaced
_model  = None
_nn     = None     # sklearn NearestNeighbors fitted on indexed embeddings
_chunks = None     # list of (chunk_text, metadata)

_load_lock = threading.Lock()


# =========================================================================
# Public API
# =========================================================================

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Return the top-k most relevant document chunks as a context string.

    The context is formatted as a bulleted list with source attribution:
        - [filename.txt] (relevance: 0.92) chunk text …

    This format is fed directly into the LLM prompt.

    Args:
        query:  the user's natural-language question
        top_k:  number of chunks to retrieve (default: 3)

    Returns:
        Formatted context string, or empty string on error.
    """
    try:
        _ensure_loaded()
    except Exception as exc:
        print(f"[RAG] Failed to load index: {exc}")
        return ""

    # --- Embed the query ---
    query_vec = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)    # shape: (1, dim)

    # --- kNN search ---
    # distances are cosine *distances* (0 = identical, 2 = opposite)
    actual_k = min(top_k, len(_chunks))
    distances, indices = _nn.kneighbors(query_vec, n_neighbors=actual_k)

    # --- Build context string ---
    lines = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue
        chunk_text, meta = _chunks[idx]
        source = meta.get("source", "unknown")
        similarity = round(1.0 - float(dist), 3)
        lines.append(f"- [{source}] (relevance: {similarity}) {chunk_text}")

    return "\n".join(lines)


def retrieve_with_scores(query: str, top_k: int = 3) -> List[dict]:
    """
    Return top-k chunks as a list of dicts — useful for debugging/evaluation.

    Each dict: {"text": str, "source": str, "chunk_id": int, "score": float}
    """
    _ensure_loaded()

    query_vec = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    actual_k = min(top_k, len(_chunks))
    distances, indices = _nn.kneighbors(query_vec, n_neighbors=actual_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk_text, meta = _chunks[idx]
        results.append({
            "text": chunk_text,
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
            "score": round(1.0 - float(dist), 4),
        })
    return results


# =========================================================================
# Lazy loader — thread-safe double-checked locking
# =========================================================================

def _ensure_loaded():
    """Load the embedding model and index exactly once per process."""
    global _model, _nn, _chunks
    if _model is not None:
        return   # fast path

    with _load_lock:
        if _model is not None:
            return   # another thread raced us and won

        # Verify index exists before loading the heavy model
        if not index_exists():
            raise RuntimeError(
                "Vector index not found.\n"
                "Run `python ingest.py` first to build the index from docs/."
            )

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            ) from exc

        print(f"[RAG] Loading embedding model '{_MODEL_NAME}' …")
        t0 = time.time()
        _model = SentenceTransformer(_MODEL_NAME)

        print("[RAG] Loading vector index from disk …")
        _nn, _chunks = load_index()

        print(
            f"[RAG] Ready — {len(_chunks)} chunks indexed, "
            f"loaded in {time.time() - t0:.1f}s"
        )
