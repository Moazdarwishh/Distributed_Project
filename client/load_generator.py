"""
Client-side load generator.

Simulates many concurrent users hitting the scheduler. Each "user" is
one task that:
  1. builds a Request
  2. calls scheduler.handle_request(req)
  3. records the Response

After all tasks finish we print an aggregate report:
    - total requests / successes / failures
    - wall-clock time
    - throughput (requests / second)
    - mean & p95 latency over successful requests
    - per-worker request count (proves the load was actually distributed)

We use ThreadPoolExecutor instead of spawning raw Threads because
materialising 1000 OS threads at once is wasteful. The pool keeps a
bounded set of workers and queues the rest, but from the system's
point of view there are still 1000 in-flight requests being handled.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from collections import Counter

from common.models import Request


# A small bag of varied queries. Each simulated user picks one.
# Keeping queries semantically diverse stress-tests the RAG retriever
# (different docs come back) instead of trivially caching one answer.
_QUERIES = [
    "How does round robin load balancing work?",
    "What is least connections routing?",
    "How does load-aware routing pick a backend?",
    "What is retrieval augmented generation?",
    "How does fault tolerance work in distributed systems?",
    "What does a heartbeat mechanism do?",
    "What is GPU acceleration for LLM inference?",
    "Why batch requests on a GPU?",
    "What is cosine similarity?",
    "What is horizontal scaling?",
]


def _percentile(values, p: float) -> float:
    """Tiny stdlib-only percentile (no numpy needed)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _user_task(scheduler, user_id: int):
    """One simulated user. Picks a query and calls the scheduler."""
    query = _QUERIES[user_id % len(_QUERIES)]
    req = Request(id=user_id, query=query)
    return scheduler.handle_request(req)


def run_load_test(
    scheduler,
    num_users: int = 1000,
    max_concurrency: int = 64,
    warmup: bool = True,
):
    """Fire `num_users` concurrent requests at the scheduler.

    Args:
        scheduler:        a master.scheduler.Scheduler instance
        num_users:        how many requests to send in total
        max_concurrency:  cap on simultaneous in-flight requests.
                          Real systems limit this; setting it equal to
                          num_users is fine for small tests but wastes
                          memory at 1000.
        warmup:           if True, send 1 request first to pay the
                          model-load cost before starting the timer.
                          Makes the reported throughput honest.

    Returns:
        list[Response] — every response collected, in completion order.
    """
    if warmup:
        print("[Client] Warm-up request (pays model-load cost)...")
        _user_task(scheduler, user_id=-1)

    print(f"[Client] Starting load test: {num_users} users, "
          f"concurrency<={max_concurrency}\n")

    results = []
    start = time.time()

    # ThreadPoolExecutor manages the worker threads for us. as_completed
    # yields each future the moment it finishes, so we collect responses
    # without having to wait for the slowest one.
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [
            pool.submit(_user_task, scheduler, i)
            for i in range(num_users)
        ]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                # Should never happen - LoadBalancer.dispatch returns a
                # failed Response instead of raising. But defend in depth.
                print(f"[Client] Task raised: {e}")

    elapsed = time.time() - start

    # ---- summary ----
    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success]
    latencies = [r.latency for r in successes]

    avg_lat = mean(latencies) if latencies else 0.0
    p95_lat = _percentile(latencies, 0.95)
    throughput = len(results) / elapsed if elapsed > 0 else 0.0

    # how many requests each worker actually handled
    worker_counts = Counter(r.worker_id for r in successes)

    print("\n========== Load Test Summary ==========")
    print(f"  Users          : {num_users}")
    print(f"  Concurrency    : {max_concurrency}")
    print(f"  Wall time      : {elapsed:.2f} s")
    print(f"  Throughput     : {throughput:.2f} req/s")
    print(f"  Successful     : {len(successes)}")
    print(f"  Failed         : {len(failures)}")
    print(f"  Avg latency    : {avg_lat:.3f} s")
    print(f"  p95 latency    : {p95_lat:.3f} s")
    print(f"  Per-worker     : {dict(sorted(worker_counts.items()))}")
    print("=======================================\n")

    return results
