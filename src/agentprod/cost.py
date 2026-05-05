"""Per-request token / dollar cost tracking.

Pattern extracted from production stock-advisor (v17): track cost at the
unit-of-work level (per request, per agent, per user), then aggregate.
Lets you answer "which agent burned my budget last week" without piping
through provider dashboards.

Pricing is supplied by the user — model prices change, so we don't ship a
catalog. Bring your own ModelPricing instances; the tracker just does the
math.

Usage:

    pricing = ModelPricing(input_per_1k=0.15, output_per_1k=0.60)
    tracker = CostTracker()
    tracker.record(
        model="gpt-4o",
        input_tokens=1234,
        output_tokens=567,
        pricing=pricing,
        labels={"agent": "fundamental_analyst", "user": "u_123"},
    )
    print(tracker.total_usd())
    print(tracker.by_label("agent"))
    # {'fundamental_analyst': 0.0019...}
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class ModelPricing:
    """Per-1K-token pricing in USD.

    Anthropic, OpenAI, and Google all quote prices per 1K (or 1M) tokens.
    Use whichever scale you like — just be consistent. Cached input usually
    has a separate (lower) rate; supply it if your provider distinguishes.
    """

    input_per_1k: float
    output_per_1k: float
    cached_input_per_1k: Optional[float] = None

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """USD cost for a single call."""
        non_cached_in = max(0, input_tokens - cached_input_tokens)
        cost = (non_cached_in / 1000.0) * self.input_per_1k
        cost += (output_tokens / 1000.0) * self.output_per_1k
        if cached_input_tokens and self.cached_input_per_1k is not None:
            cost += (cached_input_tokens / 1000.0) * self.cached_input_per_1k
        elif cached_input_tokens:
            # No cached price set — bill cached at full input rate (conservative).
            cost += (cached_input_tokens / 1000.0) * self.input_per_1k
        return cost


@dataclass
class CostEntry:
    """One LLM call's cost record."""

    ts: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    labels: dict[str, str] = field(default_factory=dict)


class CostTracker:
    """Thread-safe cost ledger with optional jsonl persistence.

    Args:
        jsonl_path: if given, every record() also appends one JSON line to
            this file. Append failures are logged at debug level and never
            raise — observability must not break the caller.
    """

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self._entries: list[CostEntry] = []
        self._lock = threading.Lock()
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self._jsonl_path is not None:
            try:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Surfaced via _append_jsonl on first write attempt.
                pass

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        pricing: ModelPricing,
        cached_input_tokens: int = 0,
        labels: Optional[dict[str, str]] = None,
        ts: Optional[str] = None,
    ) -> CostEntry:
        """Record one LLM call. Returns the entry written."""
        cost = pricing.cost(input_tokens, output_tokens, cached_input_tokens)
        entry = CostEntry(
            ts=ts or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=round(cost, 6),
            labels=dict(labels or {}),
        )
        with self._lock:
            self._entries.append(entry)
        self._append_jsonl(entry)
        return entry

    def total_usd(self, where: Optional[dict[str, str]] = None) -> float:
        """Sum USD across all entries (optionally filtered by labels)."""
        return sum(e.cost_usd for e in self._filter(where))

    def total_tokens(self, where: Optional[dict[str, str]] = None) -> dict[str, int]:
        """Token totals broken into input / output / cached / total."""
        entries = list(self._filter(where))
        in_t = sum(e.input_tokens for e in entries)
        out_t = sum(e.output_tokens for e in entries)
        cache_t = sum(e.cached_input_tokens for e in entries)
        return {
            "input": in_t,
            "output": out_t,
            "cached_input": cache_t,
            "total": in_t + out_t,
        }

    def by_label(self, key: str) -> dict[str, float]:
        """Sum USD grouped by the value of one label key.

        Example: by_label("agent") → {"fundamental_analyst": 0.42, "router": 0.01}
        Entries missing that label are bucketed under "(unset)".
        """
        out: dict[str, float] = {}
        with self._lock:
            for e in self._entries:
                k = e.labels.get(key, "(unset)")
                out[k] = out.get(k, 0.0) + e.cost_usd
        return {k: round(v, 6) for k, v in out.items()}

    def by_model(self) -> dict[str, float]:
        """Sum USD grouped by model name."""
        out: dict[str, float] = {}
        with self._lock:
            for e in self._entries:
                out[e.model] = out.get(e.model, 0.0) + e.cost_usd
        return {k: round(v, 6) for k, v in out.items()}

    def entries(self) -> list[CostEntry]:
        """Snapshot copy of all recorded entries."""
        with self._lock:
            return list(self._entries)

    def reset(self) -> None:
        """Drop in-memory entries. Does NOT delete the jsonl file."""
        with self._lock:
            self._entries.clear()

    # ── internals ───────────────────────────────────────────────────────────

    def _filter(
        self, where: Optional[dict[str, str]]
    ) -> Iterable[CostEntry]:
        if not where:
            with self._lock:
                yield from list(self._entries)
            return
        with self._lock:
            for e in self._entries:
                if all(e.labels.get(k) == v for k, v in where.items()):
                    yield e

    def _append_jsonl(self, entry: CostEntry) -> None:
        if self._jsonl_path is None:
            return
        try:
            line = json.dumps(asdict(entry), ensure_ascii=False)
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            import logging
            logging.getLogger(__name__).debug(
                "cost jsonl append failed (suppressed)"
            )
