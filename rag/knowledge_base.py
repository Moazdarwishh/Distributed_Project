"""
A small but real knowledge base for the RAG retriever to search over.

In a production system this would be thousands of documents loaded from
disk / a database / scraped pages. For the project demo we keep ~20
self-contained sentences about distributed computing, LLMs, and RAG so
that:
  - The retrieval step has something meaningful to find.
  - You can verify the system end-to-end by asking related questions.
  - Nothing has to be downloaded beyond the embedding model.

Each entry is one "document" — keep them short and focused. The retriever
treats each entry as an atomic unit: it returns the whole entry, not a
snippet from inside it.
"""

DOCUMENTS = [
    # --- Distributed systems basics ---
    "A distributed system is a collection of independent computers that "
    "appears to its users as a single coherent system.",
    "In distributed computing, fault tolerance is the property that enables "
    "a system to continue operating properly in the event of the failure of "
    "some of its components.",
    "Horizontal scaling means adding more machines to a system, while "
    "vertical scaling means adding more power (CPU, RAM) to an existing machine.",

    # --- Load balancing ---
    "A load balancer is a component that distributes incoming network "
    "requests across multiple backend servers to optimize resource use, "
    "maximize throughput, and minimize response time.",
    "Round Robin load balancing forwards each new request to the next "
    "server in a fixed cyclic order, regardless of current server load.",
    "Least Connections routing sends each new request to the backend "
    "server that currently has the fewest active connections.",
    "Load-aware routing uses live metrics such as CPU utilization or "
    "average response latency to pick the least busy backend for each request.",

    # --- LLM inference ---
    "Large Language Model (LLM) inference is the process of generating "
    "output tokens from a trained language model given an input prompt.",
    "GPU acceleration speeds up LLM inference by performing the matrix "
    "multiplications required for each layer in parallel across thousands of cores.",
    "Batching multiple requests together on a GPU improves throughput "
    "because the cost of loading model weights is amortized across many queries.",

    # --- RAG ---
    "Retrieval-Augmented Generation (RAG) is a technique that combines a "
    "retriever, which fetches relevant documents from a knowledge base, "
    "with a generative language model that produces the final answer.",
    "In a RAG pipeline, the retriever typically uses vector embeddings "
    "and similarity search to find the documents most relevant to the user query.",
    "Sentence embeddings map a piece of text to a fixed-length vector such "
    "that semantically similar texts end up close together in vector space.",
    "Cosine similarity measures the cosine of the angle between two "
    "vectors and is the standard similarity metric for text embeddings.",
    "FAISS (Facebook AI Similarity Search) is a library for efficient "
    "similarity search over large collections of dense vectors.",

    # --- Fault tolerance ---
    "A heartbeat mechanism is a periodic signal sent between distributed "
    "components to detect whether peers are still alive and reachable.",
    "Task reassignment ensures that if a worker fails mid-computation, the "
    "scheduler reroutes its pending tasks to a healthy worker so no work is lost.",
    "Replication is the practice of keeping multiple copies of data or "
    "computation on different nodes so the system survives individual failures.",

    # --- Concurrency ---
    "Concurrency is the ability of a system to handle multiple tasks at "
    "overlapping time periods, while parallelism executes them at the same instant.",
    "A thread pool is a fixed group of worker threads kept alive to "
    "service many short-lived tasks without paying thread-creation overhead.",
]
