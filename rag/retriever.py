"""
Real RAG retriever using sentence-transformers + cosine similarity.

How it works:
  1. On first use, load `all-MiniLM-L6-v2` (~80MB, 384-dim embeddings)
     and embed every document in `knowledge_base.DOCUMENTS` into a matrix.
  2. For each query, embed the query into the same 384-dim space.
  3. Compute cosine similarity between the query vector and every
     document vector. Pick the top-k highest-scoring documents.
  4. Concatenate them into one context string for the LLM.

Why this design:
  - The embedding model is small and CPU-fast (~10ms per query).
  - We embed the KB exactly once at load time, then every retrieval is
    a single matrix-vector multiplication: O(num_docs * dim). For ~20
    docs that's effectively instant. For 100k+ docs you'd swap this for
    FAISS, but the *interface* would stay identical.
  - Lazy loading + double-checked locking, same pattern as llm/inference.py,
    so import is cheap and the model loads exactly once across all threads.
"""

import threading
import time
import numpy as np

from rag.knowledge_base import DOCUMENTS

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_doc_embeddings = None        # shape: (num_docs, 384)
_load_lock = threading.Lock()


def _ensure_loaded():
    """Load the embedding model and embed the KB once."""
    global _model, _doc_embeddings
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Run:\n"
                "  pip install sentence-transformers"
            ) from e

        print(f"[RAG] Loading {_MODEL_NAME} (first call only)...")
        t0 = time.time()
        _model = SentenceTransformer(_MODEL_NAME)
        # Embed all KB documents up front. normalize_embeddings=True means
        # each vector has unit length, which makes cosine similarity equal
        # to a plain dot product (faster + simpler).
        _doc_embeddings = _model.encode(
            DOCUMENTS,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        print(
            f"[RAG] Indexed {len(DOCUMENTS)} docs "
            f"into {_doc_embeddings.shape} in {time.time() - t0:.1f}s"
        )


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Return the top-k most relevant KB documents as a single context string.

    Args:
        query: the user's question.
        top_k: how many documents to include in the context.

    Returns:
        A newline-separated string of the top-k documents, prefixed for
        clarity. If something goes wrong, returns an empty string so the
        LLM can still attempt a (less grounded) answer.
    """
    _ensure_loaded()

    # Embed the query into the same vector space as the KB.
    query_vec = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]   # shape: (384,)

    # Cosine similarity = dot product, because both sides are unit-normalized.
    # Result shape: (num_docs,)
    scores = _doc_embeddings @ query_vec

    # argsort returns indices low->high; we want top-k highest, so reverse.
    top_indices = np.argsort(scores)[::-1][:top_k]

    chosen = [DOCUMENTS[i] for i in top_indices]
    return "\n".join(f"- {doc}" for doc in chosen)
