"""Complexity classifier + cost-aware model router.

Pattern extracted from production stock-advisor (v17): classify each query into
simple / moderate / complex, then map to the cheapest model that meets the
quality bar for that tier.

The default heuristics are language-agnostic enough for English and CJK
(Korean / Japanese / Chinese): width-aware word counting + extensible keyword
sets you supply.

Typical wiring with three-tier model setup:

    >>> router = Router({
    ...     Complexity.SIMPLE:   "gpt-4o-mini",
    ...     Complexity.MODERATE: "gpt-4o",
    ...     Complexity.COMPLEX:  "claude-sonnet-4-6",
    ... })
    >>> router.select("what is the price of AAPL")
    'gpt-4o-mini'
    >>> router.select("compare AAPL and MSFT, then forecast which one outperforms")
    'claude-sonnet-4-6'

You can also extend complexity / simple keyword sets per domain:

    >>> router = Router(
    ...     model_for={...},
    ...     complex_keywords=["DCF", "valuation", "포트폴리오"],
    ...     simple_keywords=["price of", "현재 가격"],
    ... )
"""
from __future__ import annotations

import enum
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping


class Complexity(str, enum.Enum):
    """Three-tier complexity. String enum so it serializes cleanly into logs."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# Defaults tuned on real production traffic (v17). Keep deliberately small —
# users override via Router(complex_keywords=..., simple_keywords=...).
_DEFAULT_COMPLEX = (
    r"compare .+ (and|vs|versus) .+",
    r"comprehensive",
    r"forecast",
    r"predict",
    r"\bwhy\b",
    r"explain how",
    r"analy[sz]e",
)
_DEFAULT_SIMPLE = (
    r"what is the (price|value|name)",
    r"how much is",
    r"^\s*(get|fetch|show)\b",
    r"\?$",  # bare single-clause question often simple
)

# Word-count thresholds (after CJK width normalization).
_MODERATE_WORD_THRESHOLD = 8
_COMPLEX_WORD_THRESHOLD = 25


def _word_count(text: str) -> int:
    """CJK-aware word count.

    East-Asian wide characters count as 0.5 words each (matches v17 heuristic
    that worked across Korean / Japanese / Chinese mixed input).
    """
    if not text:
        return 0
    cjk = sum(
        1 for c in text if unicodedata.east_asian_width(c) in ("W", "F")
    )
    return len(text.split()) + cjk // 2


def classify_complexity(
    query: str,
    *,
    complex_keywords: Iterable[str] = _DEFAULT_COMPLEX,
    simple_keywords: Iterable[str] = _DEFAULT_SIMPLE,
    moderate_word_threshold: int = _MODERATE_WORD_THRESHOLD,
    complex_word_threshold: int = _COMPLEX_WORD_THRESHOLD,
) -> Complexity:
    """Classify query into Complexity tier.

    Decision order:
      1. Any simple-keyword match     → SIMPLE
      2. ≥2 complex-keyword matches   → COMPLEX
      3. 1 complex-keyword match      → MODERATE
      4. Word count > complex_word    → COMPLEX
      5. Word count > moderate_word   → MODERATE
      6. Otherwise                    → SIMPLE

    Order matters: simple-keyword wins over word count, because short queries
    like "what is the price of AAPL?" are SIMPLE even if the system was
    reading them as MODERATE on raw word count.
    """
    if not query:
        return Complexity.SIMPLE

    # Case-insensitive matching — caller patterns can be any case
    # ("DeFi", "DCF", "포트폴리오" all work without forcing lowercase).
    flags = re.IGNORECASE

    for pat in simple_keywords:
        if re.search(pat, query, flags):
            return Complexity.SIMPLE

    complex_hits = sum(1 for pat in complex_keywords if re.search(pat, query, flags))
    if complex_hits >= 2:
        return Complexity.COMPLEX
    if complex_hits == 1:
        return Complexity.MODERATE

    wc = _word_count(query)
    if wc > complex_word_threshold:
        return Complexity.COMPLEX
    if wc > moderate_word_threshold:
        return Complexity.MODERATE
    return Complexity.SIMPLE


@dataclass
class Router:
    """Cost-aware model router.

    Maps each Complexity tier to a model identifier. The router does not call
    LLMs itself — it returns the model name so you can plug it into your
    LangChain / OpenAI / Anthropic / LangGraph code unchanged.

    Args:
        model_for: dict mapping Complexity → model identifier. Must contain
            all three tiers.
        complex_keywords: extra regex patterns that bump a query toward COMPLEX.
        simple_keywords: extra regex patterns that pin a query to SIMPLE.
        moderate_word_threshold: word count above which → MODERATE.
        complex_word_threshold: word count above which → COMPLEX.

    Raises:
        ValueError: if model_for is missing any Complexity tier.
    """

    model_for: Mapping[Complexity, str]
    complex_keywords: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_COMPLEX)
    simple_keywords: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_SIMPLE)
    moderate_word_threshold: int = _MODERATE_WORD_THRESHOLD
    complex_word_threshold: int = _COMPLEX_WORD_THRESHOLD

    def __post_init__(self) -> None:
        missing = [c for c in Complexity if c not in self.model_for]
        if missing:
            raise ValueError(
                f"Router.model_for missing tiers: {[m.value for m in missing]}"
            )

    def classify(self, query: str) -> Complexity:
        """Public re-export of classify_complexity using this router's keywords."""
        return classify_complexity(
            query,
            complex_keywords=self.complex_keywords,
            simple_keywords=self.simple_keywords,
            moderate_word_threshold=self.moderate_word_threshold,
            complex_word_threshold=self.complex_word_threshold,
        )

    def select(self, query: str) -> str:
        """Classify query and return the model name for that tier."""
        return self.model_for[self.classify(query)]

    def select_with_reason(self, query: str) -> tuple[str, Complexity]:
        """Same as select(), but also returns the inferred Complexity for logging."""
        tier = self.classify(query)
        return self.model_for[tier], tier
