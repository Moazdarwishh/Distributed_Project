# CSE354 Distributed Computing Project

**Title:** Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests.

A **real** distributed system: multiple worker processes communicate over genuine TCP sockets, the RAG pipeline ingests real documents from disk and searches a persisted vector index, and the LLM generates actual text. Nothing is simulated.

---

## Architecture

```
         ┌─────────────────────────────────────────┐
         │            Client (load test)            │
         │  ThreadPoolExecutor → Scheduler          │
         └───────────────────┬─────────────────────┘
                             │ handle_request()
                    ┌────────▼────────┐
                    │   Scheduler     │  ← heartbeat: GET /health every 5s
                    └────────┬────────┘
                             │ dispatch()
                    ┌────────▼────────┐
                    │  Load Balancer  │  round_robin | least_connections
                    │  (httpx client) │  | load_aware
                    └──┬───┬───┬──┬───┘
                       │   │   │  │  real HTTP POST /process
              ┌────────┘ ┌─┘ ┌─┘ └────────┐
          :8001       :8002 :8003        :8004
         Worker 0   Worker 1  …         Worker N
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────────┐
              │  RAG Retriever                    │
              │  sentence-transformers embed      │
              │  sklearn NearestNeighbors search  │
              │  rag/embeddings.npy  (disk)       │
              └──────────────┬───────────────────┘
                             │ retrieved context
              ┌──────────────▼───────────────────┐
              │  LLM: google/flan-t5-small (CPU)  │
              └──────────────────────────────────┘
```

## Folder Structure

```
Distributed_Project/
├── main.py                      entry point: spawn workers → run load test
├── ingest.py                    one-time: chunk docs, embed, save index
├── requirements.txt
├── README.md
│
├── docs/                        real knowledge documents (5 × .txt)
│   ├── distributed_systems.txt
│   ├── rag_and_llms.txt
│   ├── load_balancing.txt
│   ├── fault_tolerance.txt
│   └── llm_inference.txt
│
├── rag/
│   ├── document_loader.py       load + chunk .txt/.pdf files
│   ├── indexer.py               embed chunks → sklearn NearestNeighbors
│   │                            persist to rag/embeddings.npy + rag/chunks.pkl
│   ├── retriever.py             load index from disk, serve kNN queries
│   └── knowledge_base.py        (kept for reference)
│
├── workers/
│   ├── worker_server.py         real HTTP server (ThreadingHTTPServer)
│   │                            GET /health  GET /metrics  POST /process
│   └── gpu_worker.py            (kept for reference — original in-process version)
│
├── lb/
│   └── load_balancer.py         WorkerProxy (httpx) + LoadBalancer (3 strategies)
│
├── master/
│   └── scheduler.py             metrics + heartbeat (pings /health HTTP endpoint)
│
├── client/
│   └── load_generator.py        ThreadPoolExecutor load test + summary report
│
├── llm/
│   └── inference.py             flan-t5-small inference (lazy, thread-safe)
│
└── common/
    └── models.py                Request / Response dataclasses
```

---

## Setup

```bash
# Activate the existing venv (Python 3.14)
source .venv/bin/activate

# All dependencies are already installed in the venv.
# If you start fresh: pip install -r requirements.txt
```

First run will download two models (one-time, cached to ~/.cache/huggingface/):
- `sentence-transformers/all-MiniLM-L6-v2` (~80 MB) — for embeddings
- `google/flan-t5-small` (~308 MB) — for LLM inference

---

## Run

### Step 1 — Build the vector index (one-time)

```bash
python ingest.py
```

This reads the 5 documents in `docs/`, splits them into ~200-word chunks,
embeds each chunk with sentence-transformers, and saves the index to
`rag/embeddings.npy` and `rag/chunks.pkl`.

To add your own documents: drop `.txt` (or `.pdf` if pypdf is installed)
files into `docs/` and re-run `ingest.py`.

### Step 2 — Start the distributed system and run the load test

```bash
python main.py
```

This:
1. Spawns `NUM_WORKERS` independent worker server processes (ports 8001–8004)
2. Waits for all workers to respond to `GET /health`
3. Runs a load test with `NUM_USERS` concurrent requests
4. Prints latency, throughput, per-worker breakdown
5. Shuts down all worker processes cleanly

### Step 3 — Try a single query manually (optional)

```bash
# In one terminal: start a single worker
python -m workers.worker_server --port 8001 --worker-id 0

# In another terminal: send a request
curl -s -X POST http://localhost:8001/process \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "query": "How does least connections routing work?"}' | python -m json.tool
```

---

## Configuration (edit main.py)

| Variable | Default | Effect |
|---|---|---|
| `NUM_WORKERS` | `4` | Number of real worker processes |
| `NUM_USERS` | `100` | Total requests in the load test |
| `MAX_CONCURRENCY` | `20` | Max simultaneous in-flight HTTP requests |
| `STRATEGY` | `"load_aware"` | Routing algorithm |
| `SIMULATE_FAILURE` | `True` | Kill one worker mid-test |
| `FAILURE_VICTIM` | `1` | Which worker index (0-based) to kill |

---

## Experiments for the Report

| Experiment | What to change | What to measure |
|---|---|---|
| Strategy comparison | `STRATEGY = "round_robin"` / `"least_connections"` / `"load_aware"` | Throughput, avg latency, per-worker distribution |
| Scaling test | `NUM_USERS = 10, 50, 100, 200` | Wall time, throughput (req/s) |
| Fault tolerance | `SIMULATE_FAILURE = True` | Success rate stays 100%; killed worker shows fewer requests |
| Worker count | `NUM_WORKERS = 1, 2, 4` | Throughput scales with workers |

---

## What Changed (Simulation → Real System)

| Component | Simulation (before) | Real System (now) |
|---|---|---|
| Workers | Python objects in same process | Independent OS processes |
| Communication | Method calls | Real HTTP over TCP (httpx) |
| Health checking | Boolean flag | HTTP `GET /health` over network |
| Knowledge base | 20 hardcoded strings | 5 real documents in `docs/` |
| Document ingestion | None | `ingest.py` chunks & embeds files |
| Vector index | Numpy matrix rebuilt every run | sklearn NearestNeighbors persisted to disk |
| Retrieval | Brute-force numpy matmul | kNN search with source attribution |
