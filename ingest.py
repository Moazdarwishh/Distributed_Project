"""
Document Ingestion Script — run once before starting the system.

    python ingest.py [--docs-dir docs] [--chunk-size 200] [--overlap 50]

What it does
------------
1. Reads all .txt (and .pdf if pypdf is installed) files from docs/
2. Splits each file into overlapping word chunks
3. Embeds every chunk using sentence-transformers/all-MiniLM-L6-v2
4. Saves the embeddings to rag/embeddings.npy
5. Saves the chunk list to rag/chunks.pkl

After this script completes, the worker servers can load the index at
startup without re-embedding.  Re-run ingest.py whenever you add or
change documents in the docs/ folder.

Typical runtime: ~30 s on CPU for 5 documents (~1000 chunks).
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents and build the RAG vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Folder containing .txt / .pdf files to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Target words per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Words shared between consecutive chunks",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"ERROR: docs directory '{docs_dir}' not found.")
        print(f"Create it and add .txt files, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print("  RAG Document Ingestion")
    print(f"  docs_dir   = {docs_dir.resolve()}")
    print(f"  chunk_size = {args.chunk_size} words")
    print(f"  overlap    = {args.overlap} words")
    print("=" * 60)

    t0 = time.time()

    # --- Step 1: Load and chunk documents ---
    from rag.document_loader import load_documents
    chunks = load_documents(
        str(docs_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"\nStep 1 complete: {len(chunks)} chunks loaded.\n")

    # --- Step 2: Embed and save index ---
    from rag.indexer import build_and_save
    build_and_save(chunks)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Ingestion complete in {elapsed:.1f}s")
    print(f"  {len(chunks)} chunks indexed and saved.")
    print(f"  Run `python main.py` to start the distributed system.")
    print("=" * 60)


if __name__ == "__main__":
    main()
