"""
Main entry point for the distributed LLM serving system.

Wires together:
    GPUWorker(s)  ->  LoadBalancer  ->  Scheduler  ->  client load test

Edit the CONFIG block below to run different experiments without
changing any other file. Each experiment in your project report can be
reproduced by setting these knobs and re-running this script.
"""

import threading
import time

from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


# =============================================================================
# CONFIG — change these to run different experiments.
# =============================================================================

# How many simulated GPU nodes. More workers => more parallelism, but
# each worker still serializes its own LLM calls behind a lock, so the
# real ceiling is set by your CPU.
NUM_WORKERS = 4

# How many concurrent users to simulate.
# Project spec asks for 100 -> 1000 progression. Start small while
# debugging, then bump up for the headline number in the report.
NUM_USERS = 1000

# Cap on simultaneous in-flight requests. Real production load balancers
# also queue beyond this. Keep it well below NUM_USERS to actually test
# queueing behavior.
MAX_CONCURRENCY = 135

# Which load-balancing strategy to use this run.
# Options: "round_robin" | "least_connections" | "load_aware"
STRATEGY = "load_aware"

# If True, kill one worker partway through the test to demonstrate
# fault tolerance: dispatch should retry on a healthy worker, the load
# test should still report 0 failures, and the per-worker breakdown
# should show the killed worker handled fewer requests than the others.
SIMULATE_FAILURE = True
FAILURE_DELAY_S = 2.0      # how long after the test starts before we kill
FAILURE_VICTIM_ID = 1      # which worker id to kill

# =============================================================================


def schedule_failure(workers, delay: float, victim_id: int):
    """Spawn a daemon thread that kills one worker after `delay` seconds."""
    def _kill():
        time.sleep(delay)
        print(f"\n!!! [FailureSim] Killing worker {victim_id} !!!\n")
        workers[victim_id].simulate_failure()

    threading.Thread(target=_kill, daemon=True).start()


def main():
    print("=" * 60)
    print(f"  Distributed LLM Serving — strategy={STRATEGY}")
    print(f"  workers={NUM_WORKERS}, users={NUM_USERS}, "
          f"concurrency<={MAX_CONCURRENCY}")
    print("=" * 60)

    # 1. Build the simulated GPU cluster.
    workers = [GPUWorker(worker_id=i) for i in range(NUM_WORKERS)]

    # 2. Put a load balancer in front of them.
    lb = LoadBalancer(workers, strategy=STRATEGY)

    # 3. Put the master scheduler in front of the load balancer.
    scheduler = Scheduler(lb, heartbeat_interval=2.0)

    # 4. (Optional) schedule a worker-kill to test fault tolerance.
    if SIMULATE_FAILURE:
        schedule_failure(workers, FAILURE_DELAY_S, FAILURE_VICTIM_ID)

    # 5. Run the load test. This is the part the rubric cares about.
    try:
        run_load_test(
            scheduler,
            num_users=NUM_USERS,
            max_concurrency=MAX_CONCURRENCY,
            warmup=True,
        )
    finally:
        # Always print the scheduler-side report and stop the heartbeat,
        # even if the test crashed mid-way.
        print("[Scheduler report]", scheduler.report())
        # Show the final health of each worker (proves whether the
        # fault-tolerance simulation actually killed someone).
        print("[Worker health]", {
            w.id: ("UP" if w.is_healthy() else "DOWN") for w in workers
        })
        scheduler.stop()


if __name__ == "__main__":
    main()
