"""
Real HTTP Worker Node.

Each worker is a standalone process running a ThreadingHTTPServer that
exposes three JSON endpoints:

    GET  /health    — liveness probe polled by the scheduler heartbeat
    GET  /metrics   — live load metrics (active requests, avg latency)
    POST /process   — main endpoint: runs the RAG + LLM pipeline

Endpoints return JSON on every call.  Errors are also returned as JSON
(never raw exceptions) so the load balancer can always parse the response.

Usage (launch one worker):
    python -m workers.worker_server --port 8001 --worker-id 0

The load balancer and main.py launch multiple workers on consecutive ports.

Design notes
------------
- ThreadingHTTPServer creates a new OS thread per connection.
  For our load test with ~100-1000 requests this is fine; for thousands
  of concurrent connections you would switch to asyncio + aiohttp.
- The RAG model and LLM are loaded lazily on the first /process call
  (identical to the original design) — startup is fast.
- We share a single threading.Lock for LLM inference (same as original)
  because PyTorch is not re-entrant across threads.
- Active request count and latency history are tracked in this process
  so /metrics always reflects the real current load.
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Per-worker metrics (module-level, shared across request threads)
# ---------------------------------------------------------------------------
_active_requests  = 0
_recent_latencies: deque = deque(maxlen=20)
_metrics_lock     = threading.Lock()

# ---------------------------------------------------------------------------
# Worker identity (set by --worker-id CLI flag)
# ---------------------------------------------------------------------------
WORKER_ID = 0


# ===========================================================================
# HTTP request handler
# ===========================================================================

class WorkerHandler(BaseHTTPRequestHandler):
    """Handle incoming HTTP requests for this worker."""

    # Silence the per-request access log — too noisy for load tests
    def log_message(self, fmt, *args):  # type: ignore[override]
        pass

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._handle_health()
        elif path == "/metrics":
            self._handle_metrics()
        else:
            self._send_json(404, {"error": f"Unknown path: {path}"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/process":
            self._handle_process()
        else:
            self._send_json(404, {"error": f"Unknown path: {path}"})

    # -----------------------------------------------------------------------
    # /health
    # -----------------------------------------------------------------------
    def _handle_health(self):
        self._send_json(200, {
            "status": "healthy",
            "worker_id": WORKER_ID,
            "timestamp": time.time(),
        })

    # -----------------------------------------------------------------------
    # /metrics
    # -----------------------------------------------------------------------
    def _handle_metrics(self):
        with _metrics_lock:
            active = _active_requests
            lats   = list(_recent_latencies)

        avg_lat = sum(lats) / len(lats) if lats else 0.0
        self._send_json(200, {
            "worker_id": WORKER_ID,
            "active_requests": active,
            "avg_latency": round(avg_lat, 4),
            "samples": len(lats),
        })

    # -----------------------------------------------------------------------
    # /process  — the main work endpoint
    # -----------------------------------------------------------------------
    def _handle_process(self):
        global _active_requests

        # --- Parse request body ---
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        try:
            body = self.rfile.read(length)
            payload = json.loads(body)
            request_id = int(payload["id"])
            query      = str(payload["query"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Bad request: {exc}"})
            return

        # --- Track in-flight count ---
        with _metrics_lock:
            _active_requests += 1

        start = time.time()
        try:
            print(
                f"[Worker {WORKER_ID}] req={request_id} "
                f"query='{query[:60]}{'…' if len(query) > 60 else ''}'"
            )

            # Import here so the server starts instantly;
            # models are loaded lazily on first call.
            from rag.retriever import retrieve_context
            from llm.inference import run_llm

            context = retrieve_context(query)
            result  = run_llm(query, context)
            latency = time.time() - start

            with _metrics_lock:
                _recent_latencies.append(latency)

            self._send_json(200, {
                "id":        request_id,
                "result":    result,
                "latency":   round(latency, 4),
                "worker_id": WORKER_ID,
                "success":   True,
            })

        except Exception as exc:
            latency = time.time() - start
            print(f"[Worker {WORKER_ID}] ERROR req={request_id}: {exc}")
            self._send_json(500, {
                "id":        request_id,
                "result":    "",
                "latency":   round(latency, 4),
                "worker_id": WORKER_ID,
                "success":   False,
                "error":     str(exc),
            })

        finally:
            with _metrics_lock:
                _active_requests -= 1

    # -----------------------------------------------------------------------
    # Helper: send a JSON response
    # -----------------------------------------------------------------------
    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ===========================================================================
# Entry point
# ===========================================================================

def run_server(port: int, worker_id: int):
    global WORKER_ID
    WORKER_ID = worker_id

    server = ThreadingHTTPServer(("localhost", port), WorkerHandler)
    print(
        f"[Worker {worker_id}] Listening on http://localhost:{port}  "
        f"(pid={__import__('os').getpid()})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[Worker {worker_id}] Shutting down.")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Start one RAG+LLM worker HTTP server")
    parser.add_argument("--port",      type=int, default=8001, help="Port to listen on")
    parser.add_argument("--worker-id", type=int, default=0,    help="Worker identifier")
    args = parser.parse_args()
    run_server(args.port, args.worker_id)


if __name__ == "__main__":
    main()
