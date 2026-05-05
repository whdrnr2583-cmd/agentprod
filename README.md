# agentprod

> Production patterns for indie AI agents — extracted from running a multi-LLM trading agent in production.

`agentprod` is a small Python library of the four things you reach for once your AI agent leaves your laptop and starts charging your credit card at 3 AM:

| Module | What it gives you | Why it exists |
|---|---|---|
| `Router` | Cost-aware model selection (cheapest model that meets the quality bar) | Burning Sonnet on "what is the price?" is how you go bankrupt |
| `Throttle` | Async token bucket with jitter + hard timeout | Provider rate limits don't just slow you down, they cascade |
| `retry_call` / `retry_async` | Pattern-based detection of transient failures | LLM SDKs change their exception classes every release; the error _string_ is stable |
| `CostTracker` | Per-call USD ledger with arbitrary labels (agent, user, route) | "Which agent burned $40 last night" is a question the provider dashboard can't answer |

No hard dependency on LangChain / LangGraph / OpenAI SDK. Bring your own LLM client. agentprod just gives you the production scaffolding around it.

---

## Status

**Alpha (v0.0.1).** APIs may change before 1.0. Battle-tested in one production system; tests cover the core paths but the public surface is intentionally small until usage shapes it.

---

## Install

```bash
# Pure stdlib — no required deps
pip install agentprod

# With tenacity for richer retry semantics
pip install "agentprod[retry]"
```

Python 3.10+.

---

## Quickstart

The full example is in [`examples/quickstart.py`](examples/quickstart.py). Skeleton:

```python
import asyncio
from agentprod import (
    Complexity, Router, Throttle, retry_async,
    CostTracker, ModelPricing,
)

router = Router(model_for={
    Complexity.SIMPLE:   "gpt-4o-mini",
    Complexity.MODERATE: "gpt-4o",
    Complexity.COMPLEX:  "claude-sonnet-4-6",
})
throttle = Throttle(capacity=10, refill_per_sec=10)
PRICING = {
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4o":      ModelPricing(input_per_1k=0.0025,  output_per_1k=0.01),
}
cost = CostTracker(jsonl_path=".data/cost.jsonl")

async def handle(query: str, *, agent: str) -> str:
    model = router.select(query)
    await throttle.acquire(timeout=1.0, label=f"llm:{model}")
    text, in_tok, out_tok = await retry_async(
        lambda: your_llm_call(model, query),
        max_attempts=3,
    )
    cost.record(
        model=model,
        input_tokens=in_tok, output_tokens=out_tok,
        pricing=PRICING[model],
        labels={"agent": agent},
    )
    return text
```

---

## Each piece in 30 seconds

### Router — cost-aware model selection

Pick the cheapest model that can handle the query:

```python
from agentprod import Complexity, Router

router = Router(
    model_for={
        Complexity.SIMPLE:   "gpt-4o-mini",
        Complexity.MODERATE: "gpt-4o",
        Complexity.COMPLEX:  "claude-sonnet-4-6",
    },
    # Optional: bump domain terms to a higher tier
    complex_keywords=("DCF", "valuation", "portfolio"),
    simple_keywords=("price of", "ticker"),
)

router.select("what is the price of AAPL?")
# → "gpt-4o-mini"

router.select("compare AAPL and MSFT cash flow over 5 years")
# → "claude-sonnet-4-6"
```

Three-tier classifier (`simple` / `moderate` / `complex`) using:
1. Simple-keyword regex (wins over everything — short queries shouldn't hit the expensive model just because they happen to contain a long word)
2. Complex-keyword count
3. Word-count thresholds (CJK width-aware — works on Korean / Japanese / Chinese mixed input)

### Throttle — asyncio token bucket

```python
from agentprod import Throttle, ThrottleTimeout

bucket = Throttle(
    capacity=12,             # max burst size
    refill_per_sec=12,       # sustained rps
    jitter_ms=(5, 30),       # avoid thundering herd
    on_acquire=lambda r: log.info("throttle wait: %s", r),
)

try:
    await bucket.acquire(timeout=1.0, label="GET /quote")
    # ... make your call ...
except ThrottleTimeout:
    # bucket couldn't free a slot in time — drop and try next cycle
    return None
```

Why not aiolimiter / asyncio-throttle? Two things:
- **Hard timeout** with explicit exception. Burst > timeout is usually a signal to drop the request, not to keep waiting.
- **Metrics callback**. Sync or async, exceptions swallowed. You ship throttle waits to your observability stack without wrapping the bucket.

### retry — pattern-based transient detection

```python
from agentprod import is_retryable, retry_call, retry_async

# Decision function — drop into any retry library
if is_retryable(exc):
    ...

# Or use the wrapper (uses tenacity if installed, manual backoff otherwise)
result = retry_call(
    lambda: openai_client.chat.completions.create(...),
    max_attempts=3,
)

# Async version
result = await retry_async(
    lambda: anthropic_client.messages.create(...),
    max_attempts=3,
)
```

Default patterns cover OpenAI / Anthropic / Google / bare httpx error strings: rate limit, 429, 500, 502, 503, overloaded, timeout, server error, too many requests, connection reset.

Why string matching: provider SDKs reshuffle their exception classes every release. The _message_ is the most stable contract.

### CostTracker — per-call ledger with labels

```python
from agentprod import CostTracker, ModelPricing

pricing = ModelPricing(
    input_per_1k=0.0025,
    output_per_1k=0.01,
    cached_input_per_1k=0.00125,  # optional, for providers with prompt caching
)

tracker = CostTracker(jsonl_path=".data/cost.jsonl")

tracker.record(
    model="gpt-4o",
    input_tokens=1234, output_tokens=567, cached_input_tokens=800,
    pricing=pricing,
    labels={"agent": "fundamental_analyst", "user": "u_123", "route": "/analyze"},
)

tracker.total_usd()                       # 12.4583
tracker.total_usd(where={"user": "u_123"})  # 0.42
tracker.by_label("agent")                  # {"fundamental_analyst": 0.42, ...}
tracker.by_model()                         # {"gpt-4o": 12.4583}
```

Why bring your own pricing: model prices change weekly. A library that ships its own catalog goes stale fast.

---

## Why these four

These are the four pieces I rebuilt in three different agent codebases before deciding to extract them once. Every production AI agent eventually needs:

1. **Cost discipline at the routing layer.** Per-call cost discipline alone isn't enough — by the time you see a $300 bill, the spend is sunk. Routing is where the economics start.
2. **Rate-limit resilience that doesn't cascade.** A single 429 turns into 50 once your retries pile up. Token bucket + hard timeout breaks the cascade.
3. **Retry that survives SDK upgrades.** I've had three OpenAI SDK upgrades break my retry code because the exception classes moved. String matching the message has outlived all of them.
4. **Cost attribution by label, not just total.** "We spent $40 last night" is useless. "The fundamental_analyst agent spent $38 on retries against gpt-4o" is fixable.

Everything else in your agent is your business logic and shouldn't live in a library.

---

## Non-goals

- **No LLM client wrapping.** Use OpenAI / Anthropic / LangChain / your own. agentprod gives you the scaffolding around the call, not the call itself.
- **No model catalog.** Prices change too fast.
- **No vector DB / RAG / evaluation.** Different problem domain.
- **No multiprocessing.** The Throttle is asyncio-only by design. If you need cross-process throttling, you want Redis-backed leaky bucket.

---

## Development

```bash
git clone https://github.com/whdrnr2583-cmd/agentprod
cd agentprod
pip install -e ".[dev]"
pytest
```

---

## License

MIT.
