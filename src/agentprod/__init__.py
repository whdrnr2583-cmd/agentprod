"""agentprod — production patterns for indie AI agents.

Public API:
    Router      — complexity classification + cost-aware model selection
    Throttle    — async token bucket with jitter + timeout
    is_retryable, retry_call — retryable error detection + retry wrappers
    CostTracker — per-request token / dollar cost tracking
"""
from __future__ import annotations

from agentprod.router import (
    Complexity,
    Router,
    classify_complexity,
)
from agentprod.throttle import (
    Throttle,
    ThrottleTimeout,
)
from agentprod.retry import (
    DEFAULT_RETRYABLE_PATTERNS,
    is_retryable,
    retry_call,
    retry_async,
)
from agentprod.cost import (
    CostEntry,
    CostTracker,
    ModelPricing,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "Complexity",
    "Router",
    "classify_complexity",
    "Throttle",
    "ThrottleTimeout",
    "DEFAULT_RETRYABLE_PATTERNS",
    "is_retryable",
    "retry_call",
    "retry_async",
    "CostEntry",
    "CostTracker",
    "ModelPricing",
]
