from __future__ import annotations

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
