"""Tests for financial-impact estimators wired into every exception detector.

Gen-4 Roadmap 1.8 — detectors used to pass financial_impact=None, making the
0.4 severity weight dead. These tests lock in that every detector now emits
a positive dollar impact when unit economics are provided, and still returns
None when they aren't.
"""
from __future__ import annotations

import datetime

import pytest

from common.exception_engine import (
    detect_accuracy_drop,
    detect_excess_risk,
    detect_forecast_bias,
    detect_stockout_risk,
    estimate_financial_impact_accuracy,
    estimate_financial_impact_bias,
    estimate_financial_impact_excess,
    estimate_financial_impact_stockout,
)


@pytest.fixture
def cfg():
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
                "financial_horizon_days": 7,
            },
            "accuracy_drop": {
                "accuracy_drop_pct": 15.0,
                "critical_drop_pct": 25.0,
                "min_recent_wape": 40.0,
            },
            "excess_risk": {
                "excess_dos_threshold": 90,
                "critical_dos_threshold": 180,
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
# Estimator primitives
# ---------------------------------------------------------------------------

class TestEstimators:
    def test_bias_impact_uses_margin_over_cost(self):
        # margin is preferred over cost for bias estimate
        v_margin = estimate_financial_impact_bias(50.0, 1000.0, unit_cost=5.0, unit_margin=2.0)
        v_cost_only = estimate_financial_impact_bias(50.0, 1000.0, unit_cost=5.0, unit_margin=None)
        assert v_margin == pytest.approx(0.5 * 1000 * 2.0)
        assert v_cost_only == pytest.approx(0.5 * 1000 * 5.0)

    def test_bias_impact_none_when_economics_missing(self):
        assert estimate_financial_impact_bias(50.0, 1000.0, unit_cost=None, unit_margin=None) is None
        # Zero actual volume → None
        assert estimate_financial_impact_bias(50.0, 0.0, unit_cost=5.0, unit_margin=2.0) is None

    def test_stockout_impact_scales_with_exposure(self):
        # 7-day horizon, DOS=2 → 5 exposure days @ 10 units/day @ $3 margin = $150
        v = estimate_financial_impact_stockout(
            dos=2.0, daily_demand_rate=10.0, unit_margin=3.0, unit_cost=None
        )
        assert v == pytest.approx(5 * 10 * 3)
        # DOS already beyond horizon → 0
        v2 = estimate_financial_impact_stockout(
            dos=20.0, daily_demand_rate=10.0, unit_margin=3.0, unit_cost=None
        )
        assert v2 == 0.0

    def test_excess_impact_monthly_carrying_cost(self):
        # excess_days = 30, daily_demand = 5 → excess_units = 150
        # annual carrying = 150 * $4 * 0.25 = $150/yr → monthly = $12.5
        v = estimate_financial_impact_excess(
            dos=120.0, excess_dos_threshold=90.0, daily_demand_rate=5.0,
            unit_cost=4.0, carrying_cost_rate=0.25,
        )
        assert v == pytest.approx(150.0 * 4.0 * 0.25 / 12.0)

    def test_accuracy_impact_linear_in_delta(self):
        v = estimate_financial_impact_accuracy(
            wape_delta_pp=10.0, recent_actual_units=1000.0,
            unit_margin=2.0, unit_cost=None,
        )
        assert v == pytest.approx(0.10 * 1000 * 2.0)


# ---------------------------------------------------------------------------
# Every detector now emits a populated financial_impact when economics present
# ---------------------------------------------------------------------------

class TestDetectorsEmitFinancialImpact:
    def _bias_history(self, *, forecast_sum, actual_sum, months=3):
        return [
            {"month": datetime.date(2026, i + 1, 1),
             "forecast_sum": forecast_sum, "actual_sum": actual_sum}
            for i in range(months)
        ]

    def test_forecast_bias_detector_populates_fi(self, cfg):
        hist = self._bias_history(forecast_sum=160, actual_sum=100)  # 60% over
        out = detect_forecast_bias(
            "I1", "L1", hist, cfg, unit_cost=10.0, unit_margin=3.0
        )
        assert out is not None
        assert out["financial_impact"] is not None
        assert out["financial_impact"] > 0

    def test_forecast_bias_none_without_economics(self, cfg):
        hist = self._bias_history(forecast_sum=160, actual_sum=100)
        out = detect_forecast_bias("I1", "L1", hist, cfg)
        assert out is not None
        assert out["financial_impact"] is None

    def test_stockout_detector_populates_fi(self, cfg):
        out = detect_stockout_risk(
            "I1", "L1", dos=3.0, is_below_ss=True, config=cfg,
            daily_demand_rate=12.0, unit_cost=6.0, unit_margin=2.0,
        )
        assert out is not None
        # 4 exposure days * 12 * 2 = $96
        assert out["financial_impact"] == pytest.approx(4 * 12 * 2)

    def test_accuracy_drop_detector_populates_fi(self, cfg):
        out = detect_accuracy_drop(
            "I1", "L1", recent_wape=50.0, baseline_wape=30.0, config=cfg,
            recent_actual_units=500.0, unit_cost=8.0, unit_margin=3.0,
        )
        assert out is not None
        # delta = 20pp => 0.2 * 500 * 3 = 300
        assert out["financial_impact"] == pytest.approx(0.2 * 500 * 3)

    def test_excess_detector_populates_fi(self, cfg):
        out = detect_excess_risk(
            "I1", "L1", dos=150.0, config=cfg,
            daily_demand_rate=4.0, unit_cost=10.0,
        )
        assert out is not None
        # excess_days=60, units=240, annual=$600, monthly=$50
        assert out["financial_impact"] == pytest.approx(60 * 4 * 10 * 0.25 / 12.0)

    def test_severity_increases_with_financial_impact(self, cfg):
        # Same detector, same signal — bigger margin => bigger severity
        out_small = detect_stockout_risk(
            "I1", "L1", 3.0, True, cfg,
            daily_demand_rate=1.0, unit_margin=1.0, unit_cost=1.0,
        )
        out_big = detect_stockout_risk(
            "I1", "L1", 3.0, True, cfg,
            daily_demand_rate=500.0, unit_margin=500.0, unit_cost=500.0,
        )
        assert out_small is not None and out_big is not None
        assert out_big["severity"] > out_small["severity"]
