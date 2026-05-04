"""
Master Scheduler / Controller — HTTP-aware.

The Scheduler is the single front door of the distributed system.
Clients call handle_request(); the Scheduler routes through the
LoadBalancer (which in turn fires real HTTP requests to worker processes).

Responsibilities
----------------
1. Aggregate metrics — totals, success/failure counts, latency stats.
2. Background heartbeat thread — periodically calls worker.heartbeat()
   which pings the real GET /health HTTP endpoint on each worker.
   Silent failures (crashed processes) are caught here even if no
   request happens to be in-flight at that moment.
3. Clean lifecycle — stop() shuts down the heartbeat so the process
   can exit cleanly.

Changes from the simulation version
-------------------------------------
- The heartbeat now triggers real HTTP health checks (via WorkerProxy).
- Failed workers that recover can be re-detected as healthy by the
  heartbeat (mark_healthy is called on a successful ping).
- report() adds p50 / p95 latency percentiles.
"""

import threading
import time
from statistics import median
from typing import List

from common.models import Request, Response


def _percentile(values: list, p: float) -> float:
    """Stdlib-only percentile (no numpy dependency here)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class Scheduler:
    """
    Routes requests through the LoadBalancer and owns the heartbeat loop.

    Args:
        load_balancer:       a LoadBalancer instance wrapping WorkerProxy objects
        heartbeat_interval:  seconds between health check rounds (default: 5s)
    """

    def __init__(self, load_balancer, heartbeat_interval: float = 5.0):
        self.lb                  = load_balancer
        self.heartbeat_interval  = heartbeat_interval

        # --- Aggregate metrics ---
        self._metrics_lock   = threading.Lock()
        self.total_requests  = 0
        self.successful      = 0
        self.failed          = 0
        self._latencies: List[float] = []   # all successful latencies (for percentiles)

        # --- Heartbeat ---
        self._stop      = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="scheduler-heartbeat",
        )
        self._hb_thread.start()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def handle_request(self, request: Request) -> Response:
        """Dispatch one request and record aggregate metrics."""
        response = self.lb.dispatch(request)

        with self._metrics_lock:
            self.total_requests += 1
            if response.success:
                self.successful += 1
                self._latencies.append(response.latency)
            else:
                self.failed += 1

        return response

    def report(self) -> dict:
        """
        Snapshot of aggregate metrics.

        Returns a dict with:
            total, success, failed, success_rate,
            avg_latency_s, p50_latency_s, p95_latency_s
        """
        with self._metrics_lock:
            n   = self.successful
            lats = list(self._latencies)

        avg = sum(lats) / n if n else 0.0
        p50 = _percentile(lats, 0.50)
        p95 = _percentile(lats, 0.95)

        total = self.total_requests
        return {
            "total":          total,
            "success":        self.successful,
            "failed":         self.failed,
            "success_rate":   round(self.successful / total, 4) if total else 0.0,
            "avg_latency_s":  round(avg, 4),
            "p50_latency_s":  round(p50, 4),
            "p95_latency_s":  round(p95, 4),
        }

    def worker_summary(self) -> dict:
        """Return health and load snapshot for each worker proxy."""
        summary = {}
        for w in self.lb.workers:
            summary[w.id] = {
                "url":             w.url,
                "healthy":         w.is_healthy(),
                "active_requests": w.active_requests(),
                "avg_latency_s":   round(w.avg_latency(), 4),
            }
        return summary

    def stop(self):
        """Signal the heartbeat thread to exit and close HTTP clients."""
        self._stop.set()
        self._hb_thread.join(timeout=self.heartbeat_interval + 1)
        self.lb.close()

    # -----------------------------------------------------------------------
    # Background heartbeat loop
    # -----------------------------------------------------------------------

    def _heartbeat_loop(self):
        """
        Periodically ping every worker's /health endpoint.

        - Workers that fail to respond are marked unhealthy, taking them
          out of the routing pool until they recover.
        - Workers that recover (return 200 again) are automatically
          re-admitted into the pool (mark_healthy is called inside
          WorkerProxy.heartbeat()).
        - The loop uses Event.wait() rather than time.sleep() so stop()
          wakes it up immediately.
        """
        while not self._stop.is_set():
            for worker in self.lb.workers:
                alive = worker.heartbeat()
                status = "UP" if alive else "DOWN"
                print(
                    f"[Heartbeat] worker {worker.id} @ {worker.url} → {status}"
                )
            self._stop.wait(self.heartbeat_interval)
