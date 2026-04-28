"""
Master Scheduler / Controller.

This is the single front door of the system. The client doesn't talk to
the load balancer directly — it talks to a Scheduler instance. The
Scheduler owns three responsibilities the load balancer alone shouldn't:

  1. Aggregate metrics across all requests (totals, success rate, latency).
     The load balancer routes one request at a time and shouldn't have
     to know about "the whole load test."

  2. A background heartbeat thread that periodically checks each worker.
     This is what catches silent failures - workers that died without
     raising an exception during a process() call.

  3. A clean lifecycle: stop() shuts the heartbeat down so the program
     can exit cleanly after a test run.
"""

import threading
import time

from common.models import Request, Response


class Scheduler:
    def __init__(self, load_balancer, heartbeat_interval: float = 2.0):
        self.lb = load_balancer
        self.heartbeat_interval = heartbeat_interval

        # --- aggregate metrics (read by client / report()) ---
        self._metrics_lock = threading.Lock()
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.total_latency = 0.0     # sum of latencies of *successful* requests

        # --- heartbeat thread ---
        # Event = a thread-safe boolean we can flip from another thread.
        # The heartbeat loop checks it every iteration so it can exit promptly.
        self._stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,           # dies with the program if not stopped explicitly
            name="scheduler-heartbeat",
        )
        self._hb_thread.start()

    # ------------------------------------------------------------------
    # Public API used by the client / load generator
    # ------------------------------------------------------------------
    def handle_request(self, request: Request) -> Response:
        """Process one request end-to-end and update aggregate metrics."""
        response = self.lb.dispatch(request)

        with self._metrics_lock:
            self.total_requests += 1
            if response.success:
                self.successful += 1
                self.total_latency += response.latency
            else:
                self.failed += 1

        return response

    def report(self) -> dict:
        """Snapshot of the metrics. Safe to call any time."""
        with self._metrics_lock:
            avg = (
                self.total_latency / self.successful
                if self.successful else 0.0
            )
            return {
                "total": self.total_requests,
                "success": self.successful,
                "failed": self.failed,
                "avg_latency_s": round(avg, 4),
                "success_rate": (
                    round(self.successful / self.total_requests, 4)
                    if self.total_requests else 0.0
                ),
            }

    def stop(self):
        """Signal the heartbeat thread to exit. Call at program shutdown."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Background heartbeat — fault detection
    # ------------------------------------------------------------------
    def _heartbeat_loop(self):
        """
        Periodically poke every worker so silent failures get noticed.

        Without this, a worker that dies in a way that *doesn't* raise
        (e.g. it stops responding but its process is still alive in some
        future networked version) would only be discovered the next time
        a request is dispatched to it. The heartbeat catches it sooner
        and gives the load balancer a chance to skip it proactively.
        """
        while not self._stop.is_set():
            for w in self.lb.workers:
                # In the current single-process simulation, heartbeat()
                # just returns the worker's current health flag. When you
                # later move workers across the network, this becomes a
                # real ping / RPC call with a timeout.
                w.heartbeat()
            # Event.wait returns early if stop() is called - cleaner than
            # time.sleep() because we don't have to wait the full interval
            # before the program can exit.
            self._stop.wait(self.heartbeat_interval)
