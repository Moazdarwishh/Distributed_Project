"""
Main entry point — Real Distributed RAG System.

Architecture
------------
         ┌─────────────────────────────────────────┐
         │            Client (load test)            │
         │  ThreadPoolExecutor → Scheduler          │
         └───────────────────┬─────────────────────┘
                             │ handle_request()
                    ┌────────▼────────┐
                    │   Scheduler     │  ← background heartbeat
                    │  (metrics +     │    pings /health on each worker
                    │   heartbeat)    │
                    └────────┬────────┘
                             │ dispatch()
                    ┌────────▼────────┐
                    │  Load Balancer  │  round_robin | least_connections
                    │  (HTTP client)  │  | load_aware
                    └──┬──────┬──┬───┘
                       │      │  │  real HTTP POST /process
              ┌────────▼─┐ ┌──▼──┴──┐ ┌──▼────────┐
              │ Worker 0 │ │Worker 1 │ │ Worker N  │
              │:8001     │ │:8002    │ │:800(N+1)  │
              └────┬─────┘ └──┬──────┘ └──┬────────┘
                   │          │            │
              ┌────▼──────────▼────────────▼────┐
              │  RAG Retriever (FAISS/sklearn)   │
              │  → sentence-transformers         │
              │  → rag/embeddings.npy (disk)     │
              └──────────────┬───────────────────┘
                             │ context
              ┌──────────────▼───────────────────┐
              │  LLM: flan-t5-small (CPU)         │
              │  → transformers + torch           │
              └──────────────────────────────────┘

Each worker is a real OS process running a ThreadingHTTPServer.
The load balancer talks to them over real TCP sockets on localhost.

=============================================================================
CONFIG — edit these to run different experiments
=============================================================================
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from lb.load_balancer import LoadBalancer, WorkerProxy
from master.scheduler import Scheduler
from client.load_generator import run_load_test

# ─── How many independent worker server processes to spawn ───────────────────
NUM_WORKERS      = 4

# ─── Base port: workers listen on BASE_PORT, BASE_PORT+1, … ─────────────────
BASE_PORT        = 8001

# ─── Load test parameters ────────────────────────────────────────────────────
NUM_USERS        = 100        # total requests to fire
MAX_CONCURRENCY  = 20         # max simultaneous in-flight HTTP requests

# ─── Which routing strategy to use ───────────────────────────────────────────
#     Options: "round_robin" | "least_connections" | "load_aware"
STRATEGY         = "load_aware"

# ─── Fault-tolerance demo: kill one worker mid-test ──────────────────────────
SIMULATE_FAILURE = True
FAILURE_DELAY_S  = 8.0        # seconds after load test starts
FAILURE_VICTIM   = 1          # which worker index (0-based) to kill

# ─── Scheduler heartbeat interval ────────────────────────────────────────────
HEARTBEAT_S      = 5.0

# =============================================================================


def wait_for_workers(urls: list, timeout: float = 120.0):
    """
    Block until all workers respond to GET /health, or raise RuntimeError.

    Uses a short poll loop with exponential backoff so we don't spam
    the workers during their startup phase (model loading can take 30s+).
    """
    deadline = time.time() + timeout
    pending  = set(urls)
    delay    = 0.5

    print(f"[Main] Waiting for {len(urls)} worker(s) to become healthy …")
    while pending and time.time() < deadline:
        for url in list(pending):
            try:
                r = httpx.get(f"{url}/health", timeout=2.0)
                if r.status_code == 200:
                    print(f"  [✓] {url}")
                    pending.discard(url)
            except Exception:
                pass
        if pending:
            time.sleep(min(delay, 3.0))
            delay *= 1.2

    if pending:
        raise RuntimeError(
            f"Workers did not become healthy in {timeout}s: {pending}"
        )

    print("[Main] All workers are healthy.\n")


def _schedule_failure(procs: list, delay: float, victim_idx: int):
    """Spawn a daemon thread that terminates one worker process after `delay` s."""
    def _kill():
        time.sleep(delay)
        proc = procs[victim_idx]
        if proc.poll() is None:   # still running
            print(f"\n!!! [FailureSim] Terminating worker {victim_idx} "
                  f"(pid={proc.pid}) !!!\n")
            proc.terminate()

    threading.Thread(target=_kill, daemon=True, name="failure-sim").start()


def main():
    # ------------------------------------------------------------------
    # 0. Verify the vector index exists
    # ------------------------------------------------------------------
    from rag.indexer import index_exists
    if not index_exists():
        print(
            "ERROR: RAG vector index not found.\n"
            "Run `python ingest.py` first to build the index from docs/.\n"
        )
        sys.exit(1)

    print("=" * 62)
    print(f"  Real Distributed RAG System")
    print(f"  strategy={STRATEGY}, workers={NUM_WORKERS}, "
          f"users={NUM_USERS}, concurrency≤{MAX_CONCURRENCY}")
    print("=" * 62 + "\n")

    # ------------------------------------------------------------------
    # 1. Spawn worker server processes
    # ------------------------------------------------------------------
    python_bin  = sys.executable
    worker_procs = []
    worker_urls  = []

    for i in range(NUM_WORKERS):
        port = BASE_PORT + i
        url  = f"http://localhost:{port}"
        proc = subprocess.Popen(
            [
                python_bin, "-m", "workers.worker_server",
                "--port",      str(port),
                "--worker-id", str(i),
            ],
            # Route worker stdout/stderr to the parent terminal so you can
            # see model loading logs.  Comment these out for silent runs.
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        worker_procs.append(proc)
        worker_urls.append(url)
        print(f"[Main] Spawned worker {i} on port {port} (pid={proc.pid})")

    # ------------------------------------------------------------------
    # 2. Wait for all workers to be ready
    # ------------------------------------------------------------------
    try:
        wait_for_workers(worker_urls, timeout=180.0)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        for p in worker_procs:
            p.terminate()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build the distributed system components
    # ------------------------------------------------------------------
    proxies   = [WorkerProxy(worker_id=i, url=u) for i, u in enumerate(worker_urls)]
    lb        = LoadBalancer(proxies, strategy=STRATEGY)
    scheduler = Scheduler(lb, heartbeat_interval=HEARTBEAT_S)

    # ------------------------------------------------------------------
    # 4. (Optional) schedule a worker kill to test fault tolerance
    # ------------------------------------------------------------------
    if SIMULATE_FAILURE:
        print(
            f"[Main] Fault-tolerance demo: worker {FAILURE_VICTIM} will be "
            f"killed {FAILURE_DELAY_S}s into the load test.\n"
        )
        _schedule_failure(worker_procs, FAILURE_DELAY_S, FAILURE_VICTIM)

    # ------------------------------------------------------------------
    # 5. Run the load test
    # ------------------------------------------------------------------
    try:
        run_load_test(
            scheduler,
            num_users=NUM_USERS,
            max_concurrency=MAX_CONCURRENCY,
            warmup=True,
        )
    finally:
        # Always print final reports even if the test raised
        print("\n[Scheduler report]")
        for k, v in scheduler.report().items():
            print(f"  {k:20s}: {v}")

        print("\n[Worker health]")
        for wid, info in scheduler.worker_summary().items():
            status = "UP  " if info["healthy"] else "DOWN"
            print(
                f"  worker {wid} [{status}] {info['url']}  "
                f"active={info['active_requests']}  "
                f"avg_lat={info['avg_latency_s']:.3f}s"
            )

        scheduler.stop()

        # ------------------------------------------------------------------
        # 6. Shut down worker processes
        # ------------------------------------------------------------------
        print("\n[Main] Shutting down worker processes …")
        for i, proc in enumerate(worker_procs):
            if proc.poll() is None:
                proc.terminate()
        for proc in worker_procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[Main] All workers stopped. Done.")


if __name__ == "__main__":
    main()
