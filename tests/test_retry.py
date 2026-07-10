from __future__ import annotations

import sys

import pytest

from agentprod import is_retryable, retry_async, retry_call


class TestIsRetryable:
    def test_rate_limit_message(self):
        assert is_retryable(RuntimeError("Rate limit exceeded"))

    def test_429_in_message(self):
        assert is_retryable(Exception("HTTP 429: Too Many Requests"))

    def test_500_in_message(self):
        assert is_retryable(Exception("server error 500"))

    def test_timeout(self):
        assert is_retryable(TimeoutError("connection timed out"))

    def test_non_retryable(self):
        assert not is_retryable(ValueError("invalid input"))

    def test_custom_pattern(self):
        assert is_retryable(
            Exception("EGW00201"),
            patterns=("EGW00201",),
        )


class TestRetryCall:
    def test_success_on_first_attempt(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        assert retry_call(fn) == "ok"
        assert calls["n"] == 1

    def test_retry_then_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("rate limit")
            return "ok"

        assert retry_call(fn, max_attempts=3, base_seconds=0.01) == "ok"
        assert calls["n"] == 2

    def test_non_retryable_raises_immediately(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("invalid")

        with pytest.raises(ValueError):
            retry_call(fn, max_attempts=5, base_seconds=0.01)
        assert calls["n"] == 1

    def test_exhaust_attempts(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise RuntimeError("rate limit")

        with pytest.raises(RuntimeError):
            retry_call(fn, max_attempts=2, base_seconds=0.01)
        assert calls["n"] == 2


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_success(self):
        async def fn():
            return "ok"

        assert await retry_async(fn) == "ok"

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("503 server error")
            return "ok"

        assert await retry_async(fn, max_attempts=3, base_seconds=0.01) == "ok"
        assert calls["n"] == 2


class TestMixedRetryableSequence:
    """Boundary case: a retryable error followed by a non-retryable one must
    stop immediately at the non-retryable failure, not burn through
    max_attempts. Confirms retry_if_exception(is_retryable_fn) semantics
    documented in the module docstring.
    """

    def test_stops_at_first_non_retryable_after_a_retryable_one(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("rate limit")
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            retry_call(fn, max_attempts=5, base_seconds=0.01)
        assert calls["n"] == 2  # not 5 — stopped at the non-retryable exception


class TestFallbackRetryPath:
    """Exercises _fallback_retry_sync / _fallback_retry_async directly by
    forcing `import tenacity` to fail, so the manual-backoff code path (used
    when tenacity isn't installed) gets real coverage instead of always
    routing through tenacity. Verifies it has the same raw-exception-reraise
    and is_retryable-string-matching semantics as the tenacity path.
    """

    @pytest.fixture(autouse=True)
    def _force_no_tenacity(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tenacity", None)

    def test_retry_then_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("rate limit")
            return "ok"

        assert retry_call(fn, max_attempts=3, base_seconds=0.01) == "ok"
        assert calls["n"] == 2

    def test_exhausts_and_reraises_raw_exception(self):
        def fn():
            raise RuntimeError("rate limit exceeded")

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            retry_call(fn, max_attempts=2, base_seconds=0.01)

    def test_non_retryable_raises_immediately(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("invalid")

        with pytest.raises(ValueError):
            retry_call(fn, max_attempts=5, base_seconds=0.01)
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_async_retry_then_success(self):
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("timeout")
            return "ok"

        assert await retry_async(fn, max_attempts=3, base_seconds=0.01) == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_async_exhausts_and_reraises_raw_exception(self):
        async def fn():
            raise RuntimeError("rate limit exceeded")

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            await retry_async(fn, max_attempts=2, base_seconds=0.01)
