"""
Simulated GPU worker node.

In a real distributed system each GPUWorker would run on its own machine
behind a network endpoint. Here we model one as a Python object whose
`process(request)` method runs the RAG + LLM pipeline. The load balancer
talks to workers through the same interface either way, so when you later
move workers across the network (sockets / gRPC / HTTP), nothing else
in the codebase has to change.

Each worker tracks:
  - in-flight request count           -> used by Least-Connections balancing
  - rolling average of recent latency -> used by Load-aware balancing
  - a health flag                     -> used by every strategy + fault tolerance

The whole class is thread-safe because the load balancer can call
`process()` from many client threads simultaneously.
"""

import threading
import time
from collections import deque

from common.models import Request, Response
from rag.retriever import retrieve_context
from llm.inference import run_llm


# How many recent samples to average when computing avg_latency().
# Small enough that a slow node "feels slow" quickly, big enough that
# one outlier doesn't flip routing decisions.
_LATENCY_WINDOW = 20


class GPUWorker:
    def __init__(self, worker_id: int):
        self.id = worker_id

        # --- liveness ---
        self._healthy = True
        self._health_lock = threading.Lock()

        # --- load metrics ---
        self._active = 0                          # requests currently being processed
        self._recent_latencies = deque(maxlen=_LATENCY_WINDOW)
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Health / fault-tolerance API (called by load balancer + scheduler)
    # ------------------------------------------------------------------
    def is_healthy(self) -> bool:
        with self._health_lock:
            return self._healthy

    def mark_unhealthy(self):
        """Called by the load balancer when a process() call raises."""
        with self._health_lock:
            self._healthy = False
        print(f"[Worker {self.id}] marked UNHEALTHY")

    def simulate_failure(self):
        """Manually trip the worker — used by main.py to test recovery."""
        self.mark_unhealthy()

    def heartbeat(self):
        """
        Periodically called by the scheduler. In a real system this would
        ping the remote node. For the simulation we just keep the flag
        as-is; you could later add auto-recovery logic here.
        """
        # Hook for later: try to revive a dead worker after a cool-off period.
        return self.is_healthy()

    # ------------------------------------------------------------------
    # Load metrics API (read by the load balancer's strategies)
    # ------------------------------------------------------------------
    def active_requests(self) -> int:
        with self._metrics_lock:
            return self._active

    def avg_latency(self) -> float:
        """Rolling average of the last N processed requests (seconds)."""
        with self._metrics_lock:
            if not self._recent_latencies:
                return 0.0
            return sum(self._recent_latencies) / len(self._recent_latencies)

    # ------------------------------------------------------------------
    # The actual work
    # ------------------------------------------------------------------
    def process(self, request: Request) -> Response:
        """Run RAG + LLM for one request and return a Response."""
        # Refuse work if we've been marked dead. The load balancer should
        # never route here in that case, but defend in depth.
        if not self.is_healthy():
            raise RuntimeError(f"Worker {self.id} is down")

        # bump in-flight counter (visible to the load balancer immediately)
        with self._metrics_lock:
            self._active += 1

        start = time.time()
        try:
            print(f"[Worker {self.id}] processing request {request.id}")

            # --- RAG step: pull relevant context from the KB ---
            context = retrieve_context(request.query)

            # --- LLM step: generate the answer grounded in that context ---
            result = run_llm(request.query, context)

            latency = time.time() - start

            # remember this latency for load-aware routing
            with self._metrics_lock:
                self._recent_latencies.append(latency)

            return Response(
                id=request.id,
                result=result,
                latency=latency,
                worker_id=self.id,
                success=True,
            )

        finally:
            # whether we succeeded or raised, the request is no longer in flight
            with self._metrics_lock:
                self._active -= 1
