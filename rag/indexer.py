"""
Vector Index — build, save, and load the RAG search index.

Uses sentence-transformers for embedding and sklearn NearestNeighbors
(brute-force cosine similarity) as the vector search backend.  The index
and chunk list are persisted to disk so workers load them once at startup
rather than rebuilding on every run.

Why sklearn instead of FAISS?
------------------------------
FAISS is not always pre-installed and requires compilation.  For corpora
under ~100 k chunks, sklearn's brute-force kNN is fast enough (a query
over 1000 384-dim vectors takes ~0.5 ms on CPU) and requires no extra
installation.  The retriever interface is identical to what a FAISS
backend would expose, so swapping in FAISS later only changes this file.

Persisted files (inside the rag/ directory)
-------------------------------------------
  rag/embeddings.npy   — float32 array of shape (n_chunks, embed_dim)
  rag/chunks.pkl       — list of (chunk_text, metadata) tuples
"""

import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore

from rag.document_loader import Chunk

# Paths are relative to this file's directory (rag/)
_RAG_DIR = Path(__file__).parent
_EMBEDDINGS_PATH = _RAG_DIR / "embeddings.npy"
_CHUNKS_PATH     = _RAG_DIR / "chunks.pkl"

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================================
# Build & save
# =========================================================================

def build_and_save(
    chunks: List[Chunk],
    model_name: str = _DEFAULT_MODEL,
    embeddings_path: str | None = None,
    chunks_path: str | None = None,
    batch_size: int = 64,
) -> Tuple[NearestNeighbors, List[Chunk]]:
    """
    Embed all chunks and persist the index and chunk list to disk.

    Args:
        chunks:          list of (text, metadata) produced by document_loader
        model_name:      HuggingFace model ID for the sentence-transformer
        embeddings_path: override for the .npy file path
        chunks_path:     override for the .pkl file path
        batch_size:      embedding batch size (larger = faster but more RAM)

    Returns:
        (fitted NearestNeighbors index, chunks list)
    """
    emb_path = Path(embeddings_path or _EMBEDDINGS_PATH)
    chk_path = Path(chunks_path    or _CHUNKS_PATH)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed.\n"
            "Run: pip install sentence-transformers"
        ) from exc

    texts = [c[0] for c in chunks]

    print(f"[Indexer] Loading embedding model '{model_name}' …")
    t0 = time.time()
    model = SentenceTransformer(model_name)

    print(f"[Indexer] Embedding {len(texts)} chunks (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # unit vectors → cosine = dot product
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)

    print(f"[Indexer] Building NearestNeighbors index …")
    nn = _build_nn(embeddings)

    # --- Persist ---
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), embeddings)
    with open(chk_path, "wb") as f:
        pickle.dump(chunks, f)

    elapsed = time.time() - t0
    print(
        f"[Indexer] Done in {elapsed:.1f}s — "
        f"{len(chunks)} chunks, embeddings shape {embeddings.shape}\n"
        f"          Saved to: {emb_path}\n"
        f"                    {chk_path}"
    )
    return nn, chunks


# =========================================================================
# Load
# =========================================================================

def load_index(
    embeddings_path: str | None = None,
    chunks_path: str | None = None,
) -> Tuple[NearestNeighbors, List[Chunk]]:
    """
    Load a previously built index from disk.

    Raises:
        FileNotFoundError: if the index files don't exist (run ingest.py first)
    """
    emb_path = Path(embeddings_path or _EMBEDDINGS_PATH)
    chk_path = Path(chunks_path    or _CHUNKS_PATH)

    if not emb_path.exists() or not chk_path.exists():
        missing = [p for p in (emb_path, chk_path) if not p.exists()]
        raise FileNotFoundError(
            f"Index file(s) missing: {missing}\n"
            "Run `python ingest.py` first to build the index."
        )

    embeddings: np.ndarray = np.load(str(emb_path))
    with open(chk_path, "rb") as f:
        chunks: List[Chunk] = pickle.load(f)

    nn = _build_nn(embeddings)

    print(
        f"[Indexer] Loaded index: {embeddings.shape[0]} vectors, "
        f"dim={embeddings.shape[1]}, {len(chunks)} chunks"
    )
    return nn, chunks


def index_exists(
    embeddings_path: str | None = None,
    chunks_path: str | None = None,
) -> bool:
    """Return True if both index files exist on disk."""
    emb_path = Path(embeddings_path or _EMBEDDINGS_PATH)
    chk_path = Path(chunks_path    or _CHUNKS_PATH)
    return emb_path.exists() and chk_path.exists()


# =========================================================================
# Internal helpers
# =========================================================================

def _build_nn(embeddings: np.ndarray) -> NearestNeighbors:
    """
    Fit a NearestNeighbors model on the embedding matrix.

    algorithm='brute' + metric='cosine' performs exact cosine-similarity
    search — equivalent to FAISS IndexFlatIP on unit-normalised vectors,
    but with zero additional dependencies.
    """
    nn = NearestNeighbors(
        n_neighbors=min(10, len(embeddings)),  # can't ask for more than n_samples
        algorithm="brute",
        metric="cosine",
        n_jobs=-1,  # use all CPU cores
    )
    nn.fit(embeddings)
    return nn
