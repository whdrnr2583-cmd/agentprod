"""Retry helpers for AI-agent and API-call paths.

Pattern extracted from production stock-advisor (v17): cheap message-pattern
detection of transient failures (rate limits, 5xx, timeouts) plus thin
wrappers that delegate to tenacity if installed.

Why string matching of error messages instead of typed exceptions:
LLM provider SDKs (OpenAI, Anthropic, Google) wrap their HTTP errors in a
mess of provider-specific exception classes whose names and hierarchies keep
changing. The string of the underlying error message is the most stable
contract across versions.

Two entry points:

  is_retryable(exc) -> bool
      Pure decision function. Use this in your own retry loop or with any
      retry library.

  retry_call(fn, ...) / retry_async(fn, ...)
      Thin wrappers around tenacity. Optional dependency:
          pip install agentprod[retry]
      They fall back to a simple manual loop if tenacity is missing — useful
      so the same code path works in barebones environments.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Iterable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Default patterns matched (case-insensitive) against str(exception). These
# cover the union of error strings seen across OpenAI, Anthropic, Google
# Gemini, and bare httpx responses in v17 production.
DEFAULT_RETRYABLE_PATTERNS: tuple[str, ...] = (
    "rate limit",
    "ratelimit",
    "429",
    "503",
    "502",
    "500",
    "overloaded",
    "timeout",
    "timed out",
    "server error",
    "too many requests",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
)


def is_retryable(
    exc: BaseException,
    patterns: Iterable[str] = DEFAULT_RETRYABLE_PATTERNS,
) -> bool:
    """Decide whether `exc` looks like a transient failure worth retrying.

    Matches against str(exc) case-insensitively. Patterns are matched
    case-insensitively too, so caller-supplied codes like "EGW00201" or
    "MyError" work without forcing a particular case.

    Examples:
        >>> is_retryable(RuntimeError("rate limit exceeded"))
        True
        >>> is_retryable(ValueError("invalid input"))
        False
    """
    msg = str(exc).lower()
    return any(p.lower() in msg for p in patterns)


def _fallback_retry_sync(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    base_seconds: float,
    max_seconds: float,
    is_retryable_fn: Callable[[BaseException], bool],
) -> T:
    """Manual exponential backoff with jitter — used when tenacity is absent."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:
            if attempt >= max_attempts or not is_retryable_fn(exc):
                raise
            wait = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
            wait = wait * (0.5 + random.random())  # full jitter
            logger.warning(
                "retry_call: attempt %d failed (%s) — sleeping %.2fs",
                attempt, type(exc).__name__, wait,
            )
            import time
            time.sleep(wait)


async def _fallback_retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_seconds: float,
    max_seconds: float,
    is_retryable_fn: Callable[[BaseException], bool],
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except BaseException as exc:
            if attempt >= max_attempts or not is_retryable_fn(exc):
                raise
            wait = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
            wait = wait * (0.5 + random.random())
            logger.warning(
                "retry_async: attempt %d failed (%s) — sleeping %.2fs",
                attempt, type(exc).__name__, wait,
            )
            await asyncio.sleep(wait)


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_seconds: float = 1.0,
    max_seconds: float = 30.0,
    is_retryable_fn: Callable[[BaseException], bool] = is_retryable,
) -> T:
    """Sync retry wrapper. Uses tenacity if available, else manual backoff.

    Args:
        fn: zero-argument callable to retry.
        max_attempts: total attempts (including the first try).
        base_seconds: initial backoff. Doubles each retry, capped by max_seconds.
        max_seconds: hard cap on a single sleep.
        is_retryable_fn: predicate to decide if exception is retryable.

    Returns:
        Whatever `fn()` returns.

    Raises:
        Whatever `fn()` raises if non-retryable, or after exhausting attempts.
    """
    try:
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
        )
    except ImportError:
        return _fallback_retry_sync(
            fn,
            max_attempts=max_attempts,
            base_seconds=base_seconds,
            max_seconds=max_seconds,
            is_retryable_fn=is_retryable_fn,
        )

    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=base_seconds, max=max_seconds),
        retry=retry_if_exception(is_retryable_fn),
        reraise=True,
    )
    def _call() -> T:
        return fn()

    return _call()


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_seconds: float = 1.0,
    max_seconds: float = 30.0,
    is_retryable_fn: Callable[[BaseException], bool] = is_retryable,
) -> T:
    """Async retry wrapper. Same semantics as retry_call but for coroutines."""
    try:
        from tenacity import (
            AsyncRetrying,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
        )
    except ImportError:
        return await _fallback_retry_async(
            fn,
            max_attempts=max_attempts,
            base_seconds=base_seconds,
            max_seconds=max_seconds,
            is_retryable_fn=is_retryable_fn,
        )

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=base_seconds, max=max_seconds),
        retry=retry_if_exception(is_retryable_fn),
        reraise=True,
    ):
        with attempt:
            return await fn()
    # Unreachable — tenacity always either returns or raises.
    raise RuntimeError("retry_async: unreachable")
