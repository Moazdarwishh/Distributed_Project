"""
Shared data models.

These two dataclasses are the "common language" of the whole system.
Every component (client, load balancer, worker, scheduler) creates,
reads, or returns one of these. Defining them in one place means every
component agrees on the exact shape of the data being passed around.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Request:
    """A single user query entering the system."""
    id: int                                       # unique id per request
    query: str                                    # the user's question
    created_at: float = field(default_factory=time.time)
    # `created_at` is auto-set to "now" when the Request is built.
    # We'll use it later to measure end-to-end latency (queue time + processing).


@dataclass
class Response:
    """The result returned to the user after processing."""
    id: int                                       # matches the Request.id
    result: str                                   # the LLM-generated answer
    latency: float                                # how long processing took (seconds)
    worker_id: Optional[int] = None               # which GPU worker handled it
    success: bool = True                          # False if the request failed
    error: Optional[str] = None                   # error message if success=False
