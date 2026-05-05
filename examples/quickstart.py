"""agentprod quickstart — copy this into your agent code.

This example doesn't call a real LLM (no API key needed). It shows how the
four pieces fit together for one request:

    1. Router      — pick model based on query complexity
    2. Throttle    — respect provider rate limits with timeout
    3. retry_async — auto-retry on transient failures
    4. CostTracker — record what each call cost and which agent ran it

Run:
    python examples/quickstart.py
"""
from __future__ import annotations

import asyncio
import logging
import random

from agentprod import (
    Complexity,
    CostTracker,
    ModelPricing,
    Router,
    Throttle,
    retry_async,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── 1. Router: cost-aware model selection ───────────────────────────────────
router = Router(
    model_for={
        Complexity.SIMPLE: "gpt-4o-mini",
        Complexity.MODERATE: "gpt-4o",
        Complexity.COMPLEX: "claude-sonnet-4-6",
    },
    # Domain-specific keywords — bump finance queries to higher tier
    complex_keywords=("DCF", "valuation", "포트폴리오", "종합"),
)

# ── 2. Throttle: shared bucket for one provider/app key ─────────────────────
throttle = Throttle(
    capacity=10,
    refill_per_sec=10,
    on_acquire=lambda r: print(f"[throttle] {r}"),
)

# ── 3. Pricing & cost tracker ───────────────────────────────────────────────
PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini":     ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4o":          ModelPricing(input_per_1k=0.0025,  output_per_1k=0.01),
    "claude-sonnet-4-6": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
}
cost = CostTracker(jsonl_path=".data/cost.jsonl")


# ── 4. Fake LLM call (replace with your real client) ────────────────────────
async def fake_llm_call(model: str, prompt: str) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens). Sometimes fails."""
    if random.random() < 0.3:
        raise RuntimeError("503 server error: simulated transient failure")
    await asyncio.sleep(0.05)
    return f"[{model}] response to: {prompt[:30]}", len(prompt), 50


# ── End-to-end: one request through the stack ───────────────────────────────
async def handle_query(query: str, *, agent: str, user: str) -> str:
    # Pick the right tier for this query
    model, tier = router.select_with_reason(query)
    print(f"[router] '{query[:40]}' → {tier.value} → {model}")

    # Respect rate limit (raises ThrottleTimeout if bucket exhausted)
    await throttle.acquire(timeout=1.0, label=f"llm:{model}")

    # Auto-retry on transient errors
    async def call() -> tuple[str, int, int]:
        return await fake_llm_call(model, query)

    text, in_tok, out_tok = await retry_async(
        call, max_attempts=3, base_seconds=0.1
    )

    # Record cost with labels for slicing later
    entry = cost.record(
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        pricing=PRICING[model],
        labels={"agent": agent, "user": user},
    )
    print(f"[cost] +${entry.cost_usd:.6f} ({model}, agent={agent})")
    return text


async def main() -> None:
    queries = [
        ("what is the price of AAPL?", "fundamental_analyst", "u_001"),
        ("compare AAPL and MSFT cash flow over 5 years", "fundamental_analyst", "u_002"),
        ("DCF 밸류에이션 종합 분석 부탁해", "fundamental_analyst", "u_001"),
        ("hello", "router", "u_003"),
    ]
    for q, agent, user in queries:
        try:
            text = await handle_query(q, agent=agent, user=user)
            print(f"  → {text}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {type(exc).__name__}: {exc}\n")

    print("=" * 60)
    print(f"Total cost so far: ${cost.total_usd():.6f}")
    print(f"By agent: {cost.by_label('agent')}")
    print(f"By user:  {cost.by_label('user')}")
    print(f"By model: {cost.by_model()}")


if __name__ == "__main__":
    asyncio.run(main())
