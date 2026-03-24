"""Unit tests for Feature 40 — common/exception_engine.py.

Tests all pure detection functions and scoring in isolation (no DB required).
"""
from __future__ import annotations

import datetime
import pytest

from common.exception_engine import (
    detect_forecast_bias,
    detect_stockout_risk,
    detect_accuracy_drop,
    detect_excess_risk,
    score_exception,
    generate_headline,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config():
    return {
        "thresholds": {
            "forecast_bias": {
                "bias_pct_threshold": 20.0,
                "critical_pct_threshold": 40.0,
                "min_months": 3,
                "min_actual_units": 100,
            },
            "stockout_risk": {
                "dos_threshold": 14,
                "critical_dos_threshold": 7,
                "min_daily_sales": 1.0,
            },
            "accuracy_drop": {
                "accuracy_drop_pct": 15.0,
                "critical_drop_pct": 25.0,
                "min_recent_wape": 40.0,
                "baseline_months": 3,
            },
            "excess_risk": {
                "excess_dos_threshold": 90,
                "critical_dos_threshold": 180,
                "excess_months": 3,
                "carrying_cost_rate": 0.25,
            },
        },
        "severity_weights": {
            "financial_impact": 0.4,
            "rule_score": 0.4,
            "urgency": 0.2,
        },
    }


# ---------------------------------------------------------------------------
# detect_forecast_bias
# ---------------------------------------------------------------------------

class TestDetectForecastBias:
    def _bias_history(self, forecast_sum, actual_sum, months=3):
        return [
            {"month": datetime.date(2026, i + 1, 1), "forecast_sum": forecast_sum, "actual_sum": actual_sum}
            for i in range(months)
        ]

    def test_returns_none_when_below_threshold(self, base_config):
        history = self._bias_history(forecast_sum=105, actual_sum=100, months=3)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is None  # 5% bias < 20% threshold

    def test_detects_over_forecast_bias(self, base_config):
        # 50% over-forecast
        history = self._bias_history(forecast_sum=150, actual_sum=100, months=3)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is not None
        assert result["exception_type"] == "forecast_bias"
        assert result["supporting_data"]["direction"] == "over-forecast"
        assert result["supporting_data"]["bias_pct"] > 0

    def test_detects_under_forecast_bias(self, base_config):
        # 30% under-forecast
        history = self._bias_history(forecast_sum=70, actual_sum=100, months=3)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is not None
        assert result["exception_type"] == "forecast_bias"
        assert result["supporting_data"]["direction"] == "under-forecast"
        assert result["supporting_data"]["bias_pct"] < 0

    def test_returns_none_when_insufficient_months(self, base_config):
        history = self._bias_history(forecast_sum=200, actual_sum=100, months=2)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is None

    def test_returns_none_when_actual_below_min(self, base_config):
        # Very low actual volume
        history = self._bias_history(forecast_sum=60, actual_sum=30, months=3)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is None  # total_actual = 90 < 100 min

    def test_severity_in_range(self, base_config):
        history = self._bias_history(forecast_sum=160, actual_sum=100, months=3)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is not None
        assert 0.0 <= result["severity"] <= 1.0

    def test_critical_bias_higher_severity_than_high(self, base_config):
        # Critical: 50% bias
        critical_history = self._bias_history(forecast_sum=150, actual_sum=100, months=3)
        # High: 25% bias
        high_history = self._bias_history(forecast_sum=125, actual_sum=100, months=3)

        critical_result = detect_forecast_bias("ITEM1", "LOC1", critical_history, base_config)
        high_result = detect_forecast_bias("ITEM1", "LOC1", high_history, base_config)

        assert critical_result is not None
        assert high_result is not None
        assert critical_result["severity"] > high_result["severity"]

    def test_returns_none_for_oscillating_bias(self, base_config):
        # Alternating over/under — not consistent
        history = [
            {"month": datetime.date(2026, 1, 1), "forecast_sum": 130, "actual_sum": 100},
            {"month": datetime.date(2026, 2, 1), "forecast_sum": 70, "actual_sum": 100},
            {"month": datetime.date(2026, 3, 1), "forecast_sum": 130, "actual_sum": 100},
        ]
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is None  # inconsistent direction

    def test_output_has_required_keys(self, base_config):
        history = self._bias_history(forecast_sum=145, actual_sum=100, months=3)
        result = detect_forecast_bias("ITEM1", "LOC1", history, base_config)
        assert result is not None
        for key in ("exception_type", "item_id", "loc", "severity", "financial_impact",
                    "headline", "supporting_data", "month_start"):
            assert key in result


# ---------------------------------------------------------------------------
# detect_stockout_risk
# ---------------------------------------------------------------------------

class TestDetectStockoutRisk:
    def test_returns_none_when_dos_above_threshold(self, base_config):
        result = detect_stockout_risk("ITEM1", "LOC1", dos=20.0, is_below_ss=False, config=base_config)
        assert result is None

    def test_detects_low_dos(self, base_config):
        result = detect_stockout_risk("ITEM1", "LOC1", dos=8.0, is_below_ss=False, config=base_config)
        assert result is not None
        assert result["exception_type"] == "stockout_risk"

    def test_detects_when_below_ss_even_above_threshold(self, base_config):
        result = detect_stockout_risk("ITEM1", "LOC1", dos=20.0, is_below_ss=True, config=base_config)
        assert result is not None

    def test_severity_higher_for_critical_dos(self, base_config):
        critical = detect_stockout_risk("ITEM1", "LOC1", dos=3.0, is_below_ss=True, config=base_config)
        high = detect_stockout_risk("ITEM1", "LOC1", dos=10.0, is_below_ss=False, config=base_config)
        assert critical is not None
        assert high is not None
        assert critical["severity"] > high["severity"]

    def test_severity_in_range(self, base_config):
        result = detect_stockout_risk("ITEM1", "LOC1", dos=5.0, is_below_ss=True, config=base_config)
        assert result is not None
        assert 0.0 <= result["severity"] <= 1.0

    def test_zero_dos_produces_high_severity(self, base_config):
        result = detect_stockout_risk("ITEM1", "LOC1", dos=0.0, is_below_ss=True, config=base_config)
        assert result is not None
        assert result["severity"] >= 0.5  # should be high severity

    def test_output_has_required_keys(self, base_config):
        result = detect_stockout_risk("ITEM1", "LOC1", dos=5.0, is_below_ss=True, config=base_config)
        assert result is not None
        for key in ("exception_type", "item_id", "loc", "severity", "headline", "supporting_data"):
            assert key in result


# ---------------------------------------------------------------------------
# detect_accuracy_drop
# ---------------------------------------------------------------------------

class TestDetectAccuracyDrop:
    def test_returns_none_when_no_significant_drop(self, base_config):
        result = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=25.0, baseline_wape=20.0, config=base_config)
        assert result is None  # 5pp drop < 15pp threshold; wape < 40

    def test_detects_significant_wape_increase(self, base_config):
        result = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=40.0, baseline_wape=20.0, config=base_config)
        assert result is not None
        assert result["exception_type"] == "accuracy_drop"

    def test_detects_high_wape_even_without_big_delta(self, base_config):
        # WAPE 45% regardless of delta
        result = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=45.0, baseline_wape=40.0, config=base_config)
        assert result is not None

    def test_severity_higher_for_critical_drop(self, base_config):
        critical = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=60.0, baseline_wape=30.0, config=base_config)
        high = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=38.0, baseline_wape=20.0, config=base_config)
        assert critical is not None
        assert high is not None
        assert critical["severity"] >= high["severity"]

    def test_severity_in_range(self, base_config):
        result = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=50.0, baseline_wape=25.0, config=base_config)
        assert result is not None
        assert 0.0 <= result["severity"] <= 1.0

    def test_output_has_required_keys(self, base_config):
        result = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=42.0, baseline_wape=24.0, config=base_config)
        assert result is not None
        for key in ("exception_type", "item_id", "loc", "severity", "headline", "supporting_data"):
            assert key in result

    def test_supporting_data_has_wape_fields(self, base_config):
        result = detect_accuracy_drop("ITEM1", "LOC1", recent_wape=42.0, baseline_wape=24.0, config=base_config)
        assert result is not None
        sd = result["supporting_data"]
        assert "recent_wape" in sd
        assert "baseline_wape" in sd
        assert "wape_delta_pp" in sd
        assert abs(sd["wape_delta_pp"] - 18.0) < 0.01


# ---------------------------------------------------------------------------
# detect_excess_risk
# ---------------------------------------------------------------------------

class TestDetectExcessRisk:
    def test_returns_none_when_dos_below_threshold(self, base_config):
        result = detect_excess_risk("ITEM1", "LOC1", dos=50.0, config=base_config)
        assert result is None

    def test_detects_excess_dos(self, base_config):
        result = detect_excess_risk("ITEM1", "LOC1", dos=120.0, config=base_config)
        assert result is not None
        assert result["exception_type"] == "excess_risk"

    def test_detects_critical_excess_dos(self, base_config):
        result = detect_excess_risk("ITEM1", "LOC1", dos=200.0, config=base_config)
        assert result is not None
        assert result["severity"] >= 0.5

    def test_severity_higher_for_critical_dos(self, base_config):
        critical = detect_excess_risk("ITEM1", "LOC1", dos=200.0, config=base_config)
        high = detect_excess_risk("ITEM1", "LOC1", dos=100.0, config=base_config)
        assert critical is not None
        assert high is not None
        assert critical["severity"] > high["severity"]

    def test_severity_in_range(self, base_config):
        result = detect_excess_risk("ITEM1", "LOC1", dos=150.0, config=base_config)
        assert result is not None
        assert 0.0 <= result["severity"] <= 1.0

    def test_exactly_at_threshold_flags(self, base_config):
        result = detect_excess_risk("ITEM1", "LOC1", dos=90.0, config=base_config)
        assert result is not None  # >= threshold triggers flag

    def test_output_has_required_keys(self, base_config):
        result = detect_excess_risk("ITEM1", "LOC1", dos=120.0, config=base_config)
        assert result is not None
        for key in ("exception_type", "item_id", "loc", "severity", "headline", "supporting_data"):
            assert key in result


# ---------------------------------------------------------------------------
# score_exception
# ---------------------------------------------------------------------------

class TestScoreException:
    def test_score_in_range(self, base_config):
        score = score_exception({"rule_score": 0.8, "urgency": 0.9}, financial_impact=None, config=base_config)
        assert 0.0 <= score <= 1.0

    def test_higher_rule_score_raises_severity(self, base_config):
        low = score_exception({"rule_score": 0.1, "urgency": 0.1}, financial_impact=None, config=base_config)
        high = score_exception({"rule_score": 0.9, "urgency": 0.9}, financial_impact=None, config=base_config)
        assert high > low

    def test_financial_impact_increases_severity(self, base_config):
        no_impact = score_exception({"rule_score": 0.5, "urgency": 0.5}, financial_impact=None, config=base_config)
        with_impact = score_exception({"rule_score": 0.5, "urgency": 0.5}, financial_impact=100000.0, config=base_config)
        assert with_impact >= no_impact

    def test_zero_financial_impact_treated_as_none(self, base_config):
        no_impact = score_exception({"rule_score": 0.5, "urgency": 0.5}, financial_impact=None, config=base_config)
        zero_impact = score_exception({"rule_score": 0.5, "urgency": 0.5}, financial_impact=0.0, config=base_config)
        # Both paths should give similar scores (zero not treated as high impact)
        assert abs(no_impact - zero_impact) < 0.5  # should not differ drastically

    def test_score_capped_at_1(self, base_config):
        score = score_exception(
            {"rule_score": 1.0, "urgency": 1.0},
            financial_impact=1_000_000_000,
            config=base_config,
        )
        assert score <= 1.0

    def test_missing_weights_uses_defaults(self):
        # Config without explicit weights
        cfg = {}
        score = score_exception({"rule_score": 0.7, "urgency": 0.5}, financial_impact=None, config=cfg)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# generate_headline
# ---------------------------------------------------------------------------

class TestGenerateHeadline:
    def test_forecast_bias_headline(self):
        data = {
            "item_id": "100320",
            "loc": "LOC1",
            "supporting_data": {
                "bias_pct": 38.5,
                "months_evaluated": 3,
            },
        }
        headline = generate_headline("forecast_bias", data)
        assert "100320" in headline
        assert "LOC1" in headline
        assert "38" in headline  # bias percentage

    def test_stockout_risk_headline(self):
        data = {
            "item_id": "ITEM2",
            "loc": "LOC2",
            "supporting_data": {"dos": 6.5},
        }
        headline = generate_headline("stockout_risk", data)
        assert "ITEM2" in headline
        assert "LOC2" in headline
        assert "6.5" in headline

    def test_accuracy_drop_headline(self):
        data = {
            "item_id": "ITEM3",
            "loc": "LOC3",
            "supporting_data": {"wape_delta_pp": 18.3, "recent_wape": 42.0},
        }
        headline = generate_headline("accuracy_drop", data)
        assert "ITEM3" in headline
        assert "18.3" in headline

    def test_excess_risk_headline(self):
        data = {
            "item_id": "ITEM4",
            "loc": "LOC4",
            "supporting_data": {"dos": 130.0},
        }
        headline = generate_headline("excess_risk", data)
        assert "ITEM4" in headline
        assert "130" in headline

    def test_model_drift_headline(self):
        data = {"item_id": "ITEM5", "loc": "LOC5", "supporting_data": {}}
        headline = generate_headline("model_drift", data)
        assert "ITEM5" in headline
        assert "Model Drift" in headline or "model" in headline.lower()

    def test_new_item_headline(self):
        data = {
            "item_id": "ITEM6",
            "loc": "LOC6",
            "supporting_data": {"history_months": 1},
        }
        headline = generate_headline("new_item", data)
        assert "ITEM6" in headline

    def test_unknown_type_returns_fallback(self):
        data = {"item_id": "ITEM7", "loc": "LOC7", "supporting_data": {}}
        headline = generate_headline("unknown_type", data)
        assert "ITEM7" in headline
        assert "LOC7" in headline

    def test_headline_is_string(self, base_config):
        data = {
            "item_id": "X",
            "loc": "Y",
            "supporting_data": {"bias_pct": 25.0, "months_evaluated": 3},
        }
        headline = generate_headline("forecast_bias", data)
        assert isinstance(headline, str)
        assert len(headline) > 0


# ---------------------------------------------------------------------------
# Integration: detection → score → headline pipeline
# ---------------------------------------------------------------------------

class TestDetectionPipeline:
    def test_full_bias_pipeline(self, base_config):
        history = [
            {"month": datetime.date(2026, i + 1, 1), "forecast_sum": 140, "actual_sum": 100}
            for i in range(3)
        ]
        result = detect_forecast_bias("ITEMX", "LOCX", history, base_config)
        assert result is not None

        # Re-score with financial impact
        new_score = score_exception(
            {"rule_score": result["severity"], "urgency": 0.7},
            financial_impact=50000,
            config=base_config,
        )
        assert 0.0 <= new_score <= 1.0

        # Regenerate headline
        headline = generate_headline("forecast_bias", result)
        assert "ITEMX" in headline

    def test_full_stockout_pipeline(self, base_config):
        result = detect_stockout_risk("ITEMY", "LOCY", dos=5.0, is_below_ss=True, config=base_config)
        assert result is not None
        assert result["exception_type"] == "stockout_risk"

        new_score = score_exception(
            {"rule_score": result["severity"], "urgency": 1.0},
            financial_impact=200000,
            config=base_config,
        )
        assert 0.0 <= new_score <= 1.0

    def test_no_exception_returns_none_not_exception(self, base_config):
        # Normal inventory — no exception
        result = detect_stockout_risk("ITEMZ", "LOCZ", dos=30.0, is_below_ss=False, config=base_config)
        assert result is None

        # Normal WAPE — no accuracy drop
        result2 = detect_accuracy_drop("ITEMZ", "LOCZ", recent_wape=22.0, baseline_wape=20.0, config=base_config)
        assert result2 is None
