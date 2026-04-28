# CSE354 Distributed Computing Project

**Title:** Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests.

A distributed system that takes concurrent user requests, runs them through a real RAG + LLM pipeline, and dynamically distributes the work across multiple simulated GPU worker nodes. Includes three load-balancing strategies, fault tolerance with automatic failover, and full performance metrics.

## Folder structure

```
Distributed_Project/
├── main.py                  - entry point; edit knobs at the top to run experiments
├── requirements.txt         - Python dependencies
├── README.md
│
├── client/
│   └── load_generator.py    - simulates N concurrent users with ThreadPoolExecutor
│
├── lb/
│   └── load_balancer.py     - Round Robin / Least Connections / Load-aware
│
├── master/
│   └── scheduler.py         - controller, aggregate metrics, heartbeat
│
├── workers/
│   └── gpu_worker.py        - simulated GPU node (real RAG + real LLM)
│
├── llm/
│   └── inference.py         - real flan-t5-small inference (lazy + thread-safe)
│
├── rag/
│   ├── retriever.py         - sentence-transformers + cosine similarity
│   └── knowledge_base.py    - in-memory KB documents
│
└── common/
    └── models.py            - Request / Response dataclasses
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads two models (one-time):
- `google/flan-t5-small` (~308MB) for LLM inference
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) for RAG embeddings

## Run

```bash
python main.py
```

Edit the CONFIG block at the top of `main.py` to change strategy, worker count, user count, or toggle the failure simulation.

## Experiments to run for the report

| Experiment | What to change |
|---|---|
| Strategy comparison | Set `STRATEGY` to each of `round_robin`, `least_connections`, `load_aware`. Compare throughput + per-worker breakdown. |
| Scaling test | Run with `NUM_USERS = 100, 250, 500, 1000` and plot throughput vs. users. |
| Fault tolerance | Run with `SIMULATE_FAILURE = True` and verify success rate is still 100% and the killed worker handled fewer requests. |
| Worker count sweep | Run with `NUM_WORKERS = 1, 2, 4, 8`. Shows scalability. |

## What's implemented

- Round Robin, Least Connections, and Load-aware load balancing
- Real LLM inference (flan-t5-small) — not a sleep stub
- Real RAG retrieval with sentence embeddings + cosine similarity
- Scheduler with aggregate metrics and background heartbeat
- ThreadPoolExecutor-based concurrent load generator
- Fault tolerance: per-worker health flags, dispatch-level retry, dead-node simulation
- p95 latency + per-worker request distribution in the summary

## Limitations

- Workers run as threads in one Python process. The interfaces (`process()`, `is_healthy()`, `heartbeat()`) are designed so each worker can be moved across the network later without changing the rest of the system.
- LLM inference is serialized by a lock (modeling a single GPU). For multi-GPU you would load one model per worker.
- The knowledge base is small (20 documents). Swap `rag/retriever.py` for a FAISS index when scaling up.
