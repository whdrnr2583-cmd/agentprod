"""Async token bucket throttle.

Pattern extracted from production stock-advisor (v17): an asyncio-friendly
token bucket that survives rate-limit bursts without cascading failures.

Why this exists when libraries like aiolimiter already do throttling:
  - Built-in jitter to avoid thundering herd when many coroutines wake at once.
  - Hard timeout that raises ThrottleTimeout — callers decide whether to retry
    or fail forward, instead of being silently held.
  - Optional structured metrics callback so you can ship throttle wait-times
    to your observability stack without wrapping the bucket.

Single-event-loop assumption: like aiolimiter, this is asyncio-only. Calling
from a threadpool or multiple loops will misbehave (threading.Lock is held
briefly inside, then released before await).

Usage:

    bucket = Throttle(capacity=12, refill_per_sec=12)
    await bucket.acquire(timeout=1.0)  # blocks up to 1s, raises ThrottleTimeout
    # ... make your API call here ...

For module-singleton pattern (one bucket per app key, shared across clients):

    _bucket: Throttle | None = None
    def get_bucket() -> Throttle:
        global _bucket
        if _bucket is None:
            _bucket = Throttle(capacity=int(os.environ.get("MY_RPS", "10")))
        return _bucket
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

MetricsCallback = Callable[[dict], Optional[Awaitable[None]]]


class ThrottleTimeout(Exception):
    """Raised when acquire() exceeds its timeout budget.

    Surface this to your caller — burst longer than `timeout` is usually a
    signal to drop the request and let the next cycle recover, not to push
    harder.
    """


class Throttle:
    """Asyncio-friendly token bucket with jitter, timeout, and optional metrics.

    Args:
        capacity: max tokens in bucket. Caps the burst size.
        refill_per_sec: tokens added per second. Equals sustained rps.
        jitter_ms: (min, max) jitter added to each sleep, in milliseconds.
            Prevents thundering herd. Set to (0, 0) to disable.
        on_acquire: optional callback fired after each acquire (success or
            timeout). Receives a dict with keys: wait_ms, timeout, label.
            Sync or async callable; exceptions inside are swallowed and logged
            to avoid blocking the bucket.

    Raises:
        ValueError: capacity or refill_per_sec <= 0.
    """

    # Floating-point precision guard — elapsed * refill can drift below 1.0
    # by sub-nanosecond amounts after many acquires.
    _TOKEN_EPS = 1e-9

    # Safety net to avoid infinite retry on pathological clock skew.
    _MAX_LOOP_ITERATIONS = 100

    def __init__(
        self,
        capacity: int = 10,
        refill_per_sec: int = 10,
        *,
        jitter_ms: tuple[int, int] = (5, 30),
        on_acquire: MetricsCallback | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if refill_per_sec <= 0:
            raise ValueError(f"refill_per_sec must be > 0, got {refill_per_sec}")
        if jitter_ms[0] < 0 or jitter_ms[1] < jitter_ms[0]:
            raise ValueError(f"invalid jitter_ms range: {jitter_ms}")

        self._capacity = float(capacity)
        self._refill_per_sec = float(refill_per_sec)
        self._jitter_min_ms, self._jitter_max_ms = jitter_ms
        self._on_acquire = on_acquire

        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return int(self._capacity)

    @property
    def refill_per_sec(self) -> int:
        return int(self._refill_per_sec)

    def _try_consume_locked(self, now: float) -> tuple[bool, float]:
        """Inside lock: try to take one token. Returns (success, wait_seconds)."""
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(
            self._capacity, self._tokens + elapsed * self._refill_per_sec
        )
        self._last_refill = now
        if self._tokens >= 1.0 - self._TOKEN_EPS:
            self._tokens -= 1.0
            return True, 0.0
        deficit = 1.0 - self._tokens
        return False, deficit / self._refill_per_sec

    async def acquire(
        self,
        timeout: float = 1.0,
        *,
        label: Optional[str] = None,
    ) -> float:
        """Acquire one token. Block up to `timeout` seconds for natural refill.

        Args:
            timeout: max cumulative wait in seconds. Exceeding it raises
                ThrottleTimeout. Recommended 1.0s for write-path calls,
                higher for read-path.
            label: optional identifier passed to the metrics callback (e.g.
                "GET /api/v1/quote"). Doesn't affect throttling.

        Returns:
            Actual seconds waited (0.0 if no burst).

        Raises:
            ThrottleTimeout: cumulative wait exceeded `timeout`.
            ValueError: timeout < 0.
        """
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout}")

        deadline = time.monotonic() + timeout
        total_waited = 0.0

        for _ in range(self._MAX_LOOP_ITERATIONS):
            now = time.monotonic()
            with self._lock:
                ok, wait = self._try_consume_locked(now)
            if ok:
                await self._fire_metric(total_waited, timeout=False, label=label)
                return total_waited

            jitter_ms = (
                random.randint(self._jitter_min_ms, self._jitter_max_ms)
                if self._jitter_max_ms > 0
                else 0
            )
            sleep_for = wait + (jitter_ms / 1000.0)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or sleep_for > remaining:
                await self._fire_metric(total_waited, timeout=True, label=label)
                raise ThrottleTimeout(
                    f"acquire timed out after {total_waited * 1000:.1f}ms "
                    f"(needed {sleep_for * 1000:.1f}ms more)"
                )
            await asyncio.sleep(sleep_for)
            total_waited += sleep_for

        await self._fire_metric(total_waited, timeout=True, label=label)
        raise ThrottleTimeout(
            f"acquire exceeded {self._MAX_LOOP_ITERATIONS} iterations "
            f"(waited {total_waited * 1000:.1f}ms) — likely clock skew or misconfig"
        )

    async def _fire_metric(
        self, wait_seconds: float, *, timeout: bool, label: Optional[str]
    ) -> None:
        if self._on_acquire is None:
            return
        record = {
            "wait_ms": round(wait_seconds * 1000.0, 3),
            "timeout": timeout,
            "label": label,
        }
        try:
            result = self._on_acquire(record)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001 — metrics must never break callers
            logger.debug("on_acquire callback raised (swallowed): %s", exc)

    # ── Test helpers ────────────────────────────────────────────────────────

    def _state_snapshot(self) -> dict:
        """Internal use only — testing / debugging."""
        with self._lock:
            return {
                "tokens": self._tokens,
                "last_refill": self._last_refill,
                "capacity": self._capacity,
                "refill_per_sec": self._refill_per_sec,
            }

    def _reset_for_test(self) -> None:
        """Internal use only — pytest fixture support."""
        with self._lock:
            self._tokens = self._capacity
            self._last_refill = time.monotonic()
