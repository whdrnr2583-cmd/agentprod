from __future__ import annotations

import pytest

from agentprod import Complexity, Router, classify_complexity


class TestClassifyComplexity:
    def test_empty_is_simple(self):
        assert classify_complexity("") == Complexity.SIMPLE

    def test_simple_keyword_wins_over_word_count(self):
        # Long-ish query but contains simple keyword
        q = "what is the price of Apple Inc that everyone keeps talking about today"
        assert classify_complexity(q) == Complexity.SIMPLE

    def test_two_complex_keywords(self):
        q = "predict and forecast the next move"
        assert classify_complexity(q) == Complexity.COMPLEX

    def test_one_complex_keyword(self):
        q = "explain how this happened"
        assert classify_complexity(q) == Complexity.MODERATE

    def test_long_query_is_complex(self):
        q = " ".join(["word"] * 30)
        assert classify_complexity(q) == Complexity.COMPLEX

    def test_medium_query_is_moderate(self):
        q = " ".join(["word"] * 12)
        assert classify_complexity(q) == Complexity.MODERATE

    def test_short_query_is_simple(self):
        assert classify_complexity("hello world") == Complexity.SIMPLE

    def test_cjk_word_count(self):
        # 30 CJK chars → counts as ~15 words → MODERATE
        q = "안녕하세요반갑습니다오늘은좋은날씨입니다어떻게지내시나요궁금합"
        assert classify_complexity(q) == Complexity.MODERATE

    def test_custom_complex_keywords(self):
        result = classify_complexity(
            "valuation please",
            complex_keywords=("valuation",),
        )
        assert result == Complexity.MODERATE


class TestRouter:
    def _three_tier(self) -> Router:
        return Router(
            model_for={
                Complexity.SIMPLE: "cheap-model",
                Complexity.MODERATE: "mid-model",
                Complexity.COMPLEX: "expensive-model",
            }
        )

    def test_select_simple(self):
        r = self._three_tier()
        assert r.select("what is the price") == "cheap-model"

    def test_select_complex(self):
        r = self._three_tier()
        assert r.select("predict and forecast") == "expensive-model"

    def test_select_with_reason(self):
        r = self._three_tier()
        model, tier = r.select_with_reason("predict and forecast")
        assert model == "expensive-model"
        assert tier == Complexity.COMPLEX

    def test_missing_tier_raises(self):
        with pytest.raises(ValueError, match="missing tiers"):
            Router(model_for={Complexity.SIMPLE: "x"})

    def test_custom_keywords_via_router(self):
        r = Router(
            model_for={
                Complexity.SIMPLE: "s",
                Complexity.MODERATE: "m",
                Complexity.COMPLEX: "c",
            },
            complex_keywords=("blockchain", "DeFi"),
        )
        # Two custom keyword hits → COMPLEX
        assert r.select("explain blockchain and DeFi") == "c"
