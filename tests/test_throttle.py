from __future__ import annotations

import asyncio
import time

import pytest

from agentprod import Throttle, ThrottleTimeout


class TestThrottle:
    @pytest.mark.asyncio
    async def test_immediate_acquire_within_capacity(self):
        bucket = Throttle(capacity=5, refill_per_sec=5, jitter_ms=(0, 0))
        # First 5 acquires should be instant
        for _ in range(5):
            waited = await bucket.acquire(timeout=1.0)
            assert waited == 0.0

    @pytest.mark.asyncio
    async def test_burst_then_wait(self):
        bucket = Throttle(capacity=2, refill_per_sec=10, jitter_ms=(0, 0))
        # Drain the bucket
        await bucket.acquire(timeout=0.5)
        await bucket.acquire(timeout=0.5)
        # Third should wait (bucket empty, refill at 10/s = 100ms per token)
        start = time.monotonic()
        waited = await bucket.acquire(timeout=1.0)
        elapsed = time.monotonic() - start
        assert waited > 0
        assert elapsed >= 0.05  # at least some wait happened

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        # Capacity 1, refill 1/s — second acquire with timeout=0.1 should fail
        bucket = Throttle(capacity=1, refill_per_sec=1, jitter_ms=(0, 0))
        await bucket.acquire(timeout=0.5)
        with pytest.raises(ThrottleTimeout):
            await bucket.acquire(timeout=0.1)

    @pytest.mark.asyncio
    async def test_metrics_callback_fires(self):
        records = []

        def on_acquire(record):
            records.append(record)

        bucket = Throttle(
            capacity=2,
            refill_per_sec=10,
            jitter_ms=(0, 0),
            on_acquire=on_acquire,
        )
        await bucket.acquire(timeout=1.0, label="GET /test")
        assert len(records) == 1
        assert records[0]["label"] == "GET /test"
        assert records[0]["timeout"] is False

    @pytest.mark.asyncio
    async def test_metrics_callback_async(self):
        records = []

        async def on_acquire(record):
            await asyncio.sleep(0)  # cooperative yield
            records.append(record)

        bucket = Throttle(capacity=1, refill_per_sec=10, on_acquire=on_acquire)
        await bucket.acquire(timeout=1.0)
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_metrics_callback_exception_swallowed(self):
        def on_acquire(record):
            raise RuntimeError("boom")

        bucket = Throttle(capacity=2, refill_per_sec=10, on_acquire=on_acquire)
        # Should not raise
        await bucket.acquire(timeout=1.0)

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            Throttle(capacity=0)

    def test_invalid_refill(self):
        with pytest.raises(ValueError, match="refill"):
            Throttle(capacity=5, refill_per_sec=-1)

    @pytest.mark.asyncio
    async def test_negative_timeout_raises(self):
        bucket = Throttle()
        with pytest.raises(ValueError, match="timeout"):
            await bucket.acquire(timeout=-1)
