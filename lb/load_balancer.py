"""
Load Balancer.

Holds a list of GPUWorkers and routes incoming requests to one of them
according to a configurable strategy. Implements the three strategies
required by the project spec:

    round_robin       - cycle through workers in order
    least_connections - pick the worker with fewest in-flight requests
    load_aware        - pick the worker with the lowest combined
                        (in-flight count) * (recent average latency) score

The load balancer also handles the *fault-tolerance* contract:
  - Skip workers whose `is_healthy()` returns False.
  - If a worker raises during process(), mark it unhealthy and retry on
    a different worker so the request is not lost.
"""

import threading
import itertools

from common.models import Response


class LoadBalancer:
    def __init__(self, workers, strategy: str = "round_robin"):
        if not workers:
            raise ValueError("LoadBalancer needs at least one worker")
        if strategy not in ("round_robin", "least_connections", "load_aware"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self.workers = workers
        self.strategy = strategy

        # Round-robin needs a shared counter that's safe across threads.
        # itertools.cycle gives us "next index forever"; the lock makes
        # incrementing it atomic so two threads can't grab the same worker.
        self._rr_iter = itertools.cycle(range(len(workers)))
        self._rr_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    def _pick_round_robin(self):
        """Cycle through workers; skip unhealthy ones."""
        with self._rr_lock:
            for _ in range(len(self.workers)):
                idx = next(self._rr_iter)
                w = self.workers[idx]
                if w.is_healthy():
                    return w
        return None  # all workers down

    def _pick_least_connections(self):
        """Pick the healthy worker with the fewest in-flight requests."""
        healthy = [w for w in self.workers if w.is_healthy()]
        if not healthy:
            return None
        return min(healthy, key=lambda w: w.active_requests())

    def _pick_load_aware(self):
        """
        Pick the healthy worker with the lowest "expected wait" score:
            (active_requests + 1) * (avg_latency + small_epsilon)

        The +1 stops a fresh worker (active=0) from being scored 0
        regardless of its latency. The epsilon stops a worker with no
        latency history from also scoring 0. The product approximates
        the time a *new* request would have to wait.
        """
        healthy = [w for w in self.workers if w.is_healthy()]
        if not healthy:
            return None
        return min(
            healthy,
            key=lambda w: (w.active_requests() + 1) * (w.avg_latency() + 1e-3),
        )

    def _pick(self):
        """Strategy dispatch. Called once per dispatch attempt."""
        if self.strategy == "round_robin":
            return self._pick_round_robin()
        if self.strategy == "least_connections":
            return self._pick_least_connections()
        if self.strategy == "load_aware":
            return self._pick_load_aware()
        # unreachable - validated in __init__
        return None

    # ------------------------------------------------------------------
    # Dispatch with retry-on-failure (the fault-tolerance core)
    # ------------------------------------------------------------------
    def dispatch(self, request, max_retries: int = 2):
        """
        Send a request to a worker, with up to `max_retries` failovers.

        On exception:
          - mark the offending worker unhealthy
          - re-pick using the same strategy (now from the smaller pool)
          - retry the request

        If we run out of healthy workers, return a failed Response so
        the caller knows what happened (instead of raising up the stack
        and crashing the load test).
        """
        last_err = None
        tried_ids = set()

        for attempt in range(max_retries + 1):
            worker = self._pick()
            if worker is None:
                last_err = "No healthy workers available"
                break

            # Avoid hitting the same worker twice in one dispatch (would
            # be a guaranteed second failure if it's still broken).
            if worker.id in tried_ids:
                # Round Robin will eventually cycle back here; for the
                # other strategies the same worker may keep being "best"
                # if everyone else just got marked dead.
                continue
            tried_ids.add(worker.id)

            try:
                return worker.process(request)
            except Exception as e:
                last_err = f"worker {worker.id} failed: {e}"
                worker.mark_unhealthy()
                # loop and try the next one
                continue

        # Exhausted retries -> return a failure Response, don't raise.
        return Response(
            id=request.id,
            result="",
            latency=0.0,
            worker_id=None,
            success=False,
            error=last_err or "dispatch failed",
        )
