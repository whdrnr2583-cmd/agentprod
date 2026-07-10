from __future__ import annotations

import json

import pytest

from agentprod import CostTracker, ModelPricing


class TestModelPricing:
    def test_basic_cost(self):
        p = ModelPricing(input_per_1k=0.15, output_per_1k=0.60)
        # 1000 input + 1000 output = $0.15 + $0.60 = $0.75
        assert p.cost(1000, 1000) == pytest.approx(0.75)

    def test_partial_tokens(self):
        p = ModelPricing(input_per_1k=1.0, output_per_1k=2.0)
        # 500 input + 250 output = $0.50 + $0.50 = $1.00
        assert p.cost(500, 250) == pytest.approx(1.00)

    def test_cached_input_separate_rate(self):
        p = ModelPricing(
            input_per_1k=1.0,
            output_per_1k=1.0,
            cached_input_per_1k=0.1,
        )
        # 1000 input total, of which 800 cached
        # non-cached: 200 @ 1.0 = 0.2
        # cached: 800 @ 0.1 = 0.08
        # output: 0
        assert p.cost(1000, 0, cached_input_tokens=800) == pytest.approx(0.28)

    def test_cached_input_no_separate_rate_falls_back_to_full(self):
        p = ModelPricing(input_per_1k=1.0, output_per_1k=1.0)
        # No cached price → cached billed at full rate
        # 1000 input, 800 cached → 200@1.0 + 800@1.0 = 1.0
        assert p.cost(1000, 0, cached_input_tokens=800) == pytest.approx(1.0)


class TestCostTracker:
    def _pricing(self) -> ModelPricing:
        return ModelPricing(input_per_1k=0.10, output_per_1k=0.30)

    def test_record_and_total(self):
        t = CostTracker()
        t.record(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            pricing=self._pricing(),
        )
        # 0.10 + 0.15 = 0.25
        assert t.total_usd() == pytest.approx(0.25)

    def test_total_tokens(self):
        t = CostTracker()
        p = self._pricing()
        t.record(model="gpt-4o", input_tokens=100, output_tokens=50, pricing=p)
        t.record(model="gpt-4o", input_tokens=200, output_tokens=80, pricing=p)
        totals = t.total_tokens()
        assert totals["input"] == 300
        assert totals["output"] == 130
        assert totals["total"] == 430

    def test_by_label(self):
        t = CostTracker()
        p = self._pricing()
        t.record(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=0,
            pricing=p,
            labels={"agent": "router"},
        )
        t.record(
            model="gpt-4o",
            input_tokens=2000,
            output_tokens=0,
            pricing=p,
            labels={"agent": "analyst"},
        )
        result = t.by_label("agent")
        assert result["router"] == pytest.approx(0.10)
        assert result["analyst"] == pytest.approx(0.20)

    def test_by_label_missing_label_buckets_to_unset(self):
        t = CostTracker()
        p = self._pricing()
        t.record(model="m", input_tokens=1000, output_tokens=0, pricing=p)
        result = t.by_label("agent")
        assert "(unset)" in result

    def test_by_model(self):
        t = CostTracker()
        p = self._pricing()
        t.record(model="gpt-4o", input_tokens=1000, output_tokens=0, pricing=p)
        t.record(model="claude-sonnet", input_tokens=1000, output_tokens=0, pricing=p)
        result = t.by_model()
        assert result["gpt-4o"] == pytest.approx(0.10)
        assert result["claude-sonnet"] == pytest.approx(0.10)

    def test_jsonl_persistence(self, tmp_path):
        path = tmp_path / "cost.jsonl"
        t = CostTracker(jsonl_path=path)
        t.record(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            pricing=self._pricing(),
            labels={"agent": "test"},
        )
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model"] == "gpt-4o"
        assert record["input_tokens"] == 1000
        assert record["labels"] == {"agent": "test"}

    def test_filter_by_labels(self):
        t = CostTracker()
        p = self._pricing()
        t.record(
            model="m", input_tokens=1000, output_tokens=0,
            pricing=p, labels={"user": "a"},
        )
        t.record(
            model="m", input_tokens=1000, output_tokens=0,
            pricing=p, labels={"user": "b"},
        )
        assert t.total_usd(where={"user": "a"}) == pytest.approx(0.10)
        assert t.total_usd(where={"user": "b"}) == pytest.approx(0.10)
        assert t.total_usd() == pytest.approx(0.20)

    def test_reset(self):
        t = CostTracker()
        t.record(
            model="m", input_tokens=1000, output_tokens=0,
            pricing=self._pricing(),
        )
        assert len(t.entries()) == 1
        t.reset()
        assert len(t.entries()) == 0

    def test_unwritable_jsonl_directory_degrades_gracefully(self, tmp_path):
        """Synthetic fixture: point jsonl_path's parent at a location that is
        actually a file, not a directory. Both the constructor's mkdir()
        and _append_jsonl()'s open() then hit OSError. Per the class
        docstring ("Append failures are logged at debug level and never
        raise"), record() must still succeed and in-memory tracking must
        stay intact — this was previously untested (both except-OSError
        branches were uncovered).
        """
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("occupies the path a directory would need")
        jsonl_path = blocker / "cost.jsonl"

        t = CostTracker(jsonl_path=jsonl_path)  # must not raise
        entry = t.record(
            model="m", input_tokens=100, output_tokens=50,
            pricing=self._pricing(),
        )  # must not raise
        assert entry.cost_usd > 0
        assert len(t.entries()) == 1
        assert t.total_usd() == pytest.approx(entry.cost_usd)
        assert not jsonl_path.exists()
