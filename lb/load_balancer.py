"""
Load Balancer — HTTP-based, real network requests.

Architecture
------------
Instead of calling Python methods on in-process objects, this load
balancer sends real HTTP POST requests to worker server processes via
httpx.  Each remote worker is represented locally by a WorkerProxy that:

    • tracks health (via GET /health heartbeat)
    • tracks in-flight request count (incremented/decremented locally)
    • maintains a rolling average latency window (for load-aware routing)
    • retries on failure and marks misbehaving workers unhealthy

The three routing strategies are unchanged from the simulation:

    round_robin        — cycle through workers in order
    least_connections  — pick the worker with fewest in-flight requests
    load_aware         — pick the worker with the lowest expected wait:
                         (active_requests + 1) * (avg_latency + ε)

Fault tolerance
---------------
If a worker raises an exception or returns a non-200 status, the proxy
marks it unhealthy and the dispatcher retries on the next healthy worker.
The scheduler's heartbeat loop separately pings /health to detect silent
failures (workers that died without the LB noticing mid-flight).
"""

import itertools
import threading
import time
from collections import deque
from typing import List, Optional

import httpx

from common.models import Request, Response


# ===========================================================================
# WorkerProxy — one per remote HTTP worker
# ===========================================================================

class WorkerProxy:
    """
    Client-side proxy for one remote HTTP worker process.

    Tracks health and load metrics locally so routing decisions are
    O(1) without making extra HTTP calls.
    """

    def __init__(self, worker_id: int, url: str, timeout: float = 60.0):
        self.id  = worker_id
        self.url = url.rstrip("/")

        # --- Health ---
        self._healthy = True
        self._health_lock = threading.Lock()

        # --- Load metrics ---
        self._active: int = 0
        self._recent_latencies: deque = deque(maxlen=20)
        self._metrics_lock = threading.Lock()

        # Persistent HTTP client with connection pooling.
        # A single Client is shared across all threads; httpx.Client is
        # thread-safe as long as we don't mutate it after construction.
        self._client = httpx.Client(timeout=timeout)

    # -----------------------------------------------------------------------
    # Health API
    # -----------------------------------------------------------------------

    def is_healthy(self) -> bool:
        with self._health_lock:
            return self._healthy

    def mark_unhealthy(self):
        with self._health_lock:
            self._healthy = False
        print(f"[Proxy {self.id}] marked UNHEALTHY ({self.url})")

    def mark_healthy(self):
        with self._health_lock:
            self._healthy = True

    def heartbeat(self) -> bool:
        """
        Ping GET /health.  Updates health state and returns True if alive.
        Called periodically by the scheduler's background heartbeat thread.
        """
        try:
            resp = self._client.get(f"{self.url}/health", timeout=3.0)
            if resp.status_code == 200:
                self.mark_healthy()
                return True
        except Exception as exc:
            print(f"[Proxy {self.id}] heartbeat failed: {exc}")
        self.mark_unhealthy()
        return False

    # -----------------------------------------------------------------------
    # Load metrics API (read by routing strategies)
    # -----------------------------------------------------------------------

    def active_requests(self) -> int:
        with self._metrics_lock:
            return self._active

    def avg_latency(self) -> float:
        """Rolling average of the last 20 processed request latencies."""
        with self._metrics_lock:
            if not self._recent_latencies:
                return 0.0
            return sum(self._recent_latencies) / len(self._recent_latencies)

    # -----------------------------------------------------------------------
    # Main dispatch — sends a real HTTP request
    # -----------------------------------------------------------------------

    def process(self, request: Request) -> Response:
        """
        POST /process to the remote worker and return a Response.

        Raises RuntimeError (caught by LoadBalancer.dispatch) on:
            • worker is marked unhealthy before we even try
            • HTTP error status (4xx / 5xx)
            • network timeout or connection refused
        """
        if not self.is_healthy():
            raise RuntimeError(f"Worker {self.id} at {self.url} is unhealthy")

        with self._metrics_lock:
            self._active += 1

        wall_start = time.time()
        try:
            http_resp = self._client.post(
                f"{self.url}/process",
                json={"id": request.id, "query": request.query},
            )
            http_resp.raise_for_status()   # raises on 4xx / 5xx
            data = http_resp.json()

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Worker {self.id} returned HTTP {exc.response.status_code}"
            ) from exc

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise RuntimeError(
                f"Worker {self.id} at {self.url} unreachable: {exc}"
            ) from exc

        finally:
            with self._metrics_lock:
                self._active -= 1

        # Record wall-clock latency (includes network round-trip overhead)
        wall_latency = time.time() - wall_start
        with self._metrics_lock:
            self._recent_latencies.append(wall_latency)

        # Prefer the worker's own reported latency (pure compute time)
        # for the Response object; fall back to wall-clock if missing.
        reported_latency = data.get("latency", wall_latency)

        return Response(
            id=request.id,
            result=data.get("result", ""),
            latency=reported_latency,
            worker_id=self.id,
            success=data.get("success", True),
            error=data.get("error"),
        )

    def close(self):
        """Clean up the underlying httpx connection pool."""
        self._client.close()

    def __repr__(self):
        status = "UP" if self.is_healthy() else "DOWN"
        return (
            f"WorkerProxy(id={self.id}, url={self.url}, "
            f"status={status}, active={self.active_requests()}, "
            f"avg_lat={self.avg_latency():.3f}s)"
        )


# ===========================================================================
# LoadBalancer
# ===========================================================================

class LoadBalancer:
    """
    Routes requests to worker proxies using one of three strategies.

    Args:
        workers:   list of WorkerProxy instances (one per remote worker process)
        strategy:  "round_robin" | "least_connections" | "load_aware"
    """

    def __init__(self, workers: List[WorkerProxy], strategy: str = "round_robin"):
        if not workers:
            raise ValueError("LoadBalancer needs at least one worker")
        if strategy not in ("round_robin", "least_connections", "load_aware"):
            raise ValueError(f"Unknown strategy: '{strategy}'")

        self.workers  = workers
        self.strategy = strategy

        self._rr_iter = itertools.cycle(range(len(workers)))
        self._rr_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Strategy implementations
    # -----------------------------------------------------------------------

    def _pick_round_robin(self) -> Optional[WorkerProxy]:
        with self._rr_lock:
            for _ in range(len(self.workers)):
                idx = next(self._rr_iter)
                w   = self.workers[idx]
                if w.is_healthy():
                    return w
        return None

    def _pick_least_connections(self) -> Optional[WorkerProxy]:
        healthy = [w for w in self.workers if w.is_healthy()]
        return min(healthy, key=lambda w: w.active_requests()) if healthy else None

    def _pick_load_aware(self) -> Optional[WorkerProxy]:
        """
        Pick the worker with the lowest estimated wait time:
            (active_requests + 1) * (avg_latency + ε)

        +1 prevents a fresh worker (active=0) from scoring 0 regardless of
        latency.  ε prevents zero scoring when no latency history exists.
        """
        healthy = [w for w in self.workers if w.is_healthy()]
        return (
            min(healthy, key=lambda w: (w.active_requests() + 1) * (w.avg_latency() + 1e-3))
            if healthy else None
        )

    def _pick(self) -> Optional[WorkerProxy]:
        dispatch = {
            "round_robin":       self._pick_round_robin,
            "least_connections": self._pick_least_connections,
            "load_aware":        self._pick_load_aware,
        }
        return dispatch[self.strategy]()

    # -----------------------------------------------------------------------
    # Dispatch with retry-on-failure
    # -----------------------------------------------------------------------

    def dispatch(self, request: Request, max_retries: int = 2) -> Response:
        """
        Send request to the best available worker, retrying on failures.

        On each failure:
          - The offending worker is marked unhealthy.
          - The strategy picks the next best healthy worker.
          - Up to max_retries+1 total attempts are made.

        If all attempts fail, returns a failed Response (never raises).
        """
        last_err  = None
        tried_ids = set()

        for attempt in range(max_retries + 1):
            worker = self._pick()
            if worker is None:
                last_err = "No healthy workers available"
                break

            if worker.id in tried_ids:
                # This worker already failed — don't retry it
                # (may happen with round-robin when pool shrinks)
                continue
            tried_ids.add(worker.id)

            try:
                return worker.process(request)
            except Exception as exc:
                last_err = str(exc)
                worker.mark_unhealthy()
                print(
                    f"[LB] attempt {attempt+1}/{max_retries+1} "
                    f"failed on worker {worker.id}: {exc}"
                )

        return Response(
            id=request.id,
            result="",
            latency=0.0,
            worker_id=None,
            success=False,
            error=last_err or "dispatch failed",
        )

    def close(self):
        """Close all worker HTTP clients."""
        for w in self.workers:
            w.close()
