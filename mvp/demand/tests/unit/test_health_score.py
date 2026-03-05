"""Unit tests for IPfeature6 — Inventory Health Score scoring logic.

Tests the scoring CASE expressions and tier classification logic in isolation
(no DB required).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Pure-Python replicas of the SQL scoring logic
# ---------------------------------------------------------------------------

def score_ss_coverage(ss_combined, ss_coverage):
    """Component 1: SS Coverage (0–25 pts, 12 = neutral)."""
    if ss_combined is None:
        return 12
    cov = ss_coverage or 0
    if cov >= 1.5:
        return 25
    if cov >= 1.0:
        return 18
    if cov >= 0.5:
        return 10
    return 0


def score_dos_target(target_dos_min, target_dos_max, avg_daily_sls, eom_qty_on_hand):
    """Component 2: DOS Target Adherence (0–25 pts, 15 = neutral)."""
    if target_dos_min is None:
        return 15
    if avg_daily_sls == 0 and eom_qty_on_hand == 0:
        return 0
    if avg_daily_sls == 0:
        return 5
    dos = eom_qty_on_hand / avg_daily_sls
    if target_dos_min <= dos <= target_dos_max:
        return 25
    if dos > target_dos_max:
        return 10
    return 5  # below minimum


def score_stockout_risk(stockout_count_3m):
    """Component 3: Stockout Risk History (0–25 pts, 20 = neutral)."""
    if stockout_count_3m is None:
        return 20
    if stockout_count_3m == 0:
        return 25
    if stockout_count_3m == 1:
        return 15
    if stockout_count_3m == 2:
        return 8
    return 0  # 3+ = chronic


def score_forecast_accuracy(recent_wape):
    """Component 4: Forecast Accuracy (0–25 pts, 15 = neutral)."""
    if recent_wape is None:
        return 15
    if recent_wape < 0.15:
        return 25
    if recent_wape < 0.25:
        return 20
    if recent_wape < 0.40:
        return 15
    if recent_wape < 0.60:
        return 8
    return 0


def health_tier(health_score: int) -> str:
    """Map composite score to tier."""
    if health_score >= 80:
        return "healthy"
    if health_score >= 60:
        return "monitor"
    if health_score >= 40:
        return "at_risk"
    return "critical"


def composite(ss_combined, ss_coverage, target_dos_min, target_dos_max,
              avg_daily_sls, eom_qty_on_hand, stockout_count_3m, recent_wape) -> int:
    return (
        score_ss_coverage(ss_combined, ss_coverage)
        + score_dos_target(target_dos_min, target_dos_max, avg_daily_sls, eom_qty_on_hand)
        + score_stockout_risk(stockout_count_3m)
        + score_forecast_accuracy(recent_wape)
    )


# ---------------------------------------------------------------------------
# Tests: score_ss_coverage
# ---------------------------------------------------------------------------

class TestScoreSsCoverage:
    def test_null_ss_combined_returns_neutral(self):
        assert score_ss_coverage(None, None) == 12

    def test_high_coverage_returns_25(self):
        assert score_ss_coverage(100, 1.5) == 25

    def test_above_15_returns_25(self):
        assert score_ss_coverage(100, 2.0) == 25

    def test_exactly_1_returns_18(self):
        assert score_ss_coverage(100, 1.0) == 18

    def test_medium_coverage_returns_18(self):
        assert score_ss_coverage(100, 1.2) == 18

    def test_low_coverage_returns_10(self):
        assert score_ss_coverage(100, 0.5) == 10

    def test_very_low_coverage_returns_0(self):
        assert score_ss_coverage(100, 0.3) == 0

    def test_zero_coverage_returns_0(self):
        assert score_ss_coverage(100, 0.0) == 0

    def test_null_ss_coverage_treated_as_zero(self):
        # COALESCE(ss_coverage, 0) → 0 → returns 0
        assert score_ss_coverage(100, None) == 0


# ---------------------------------------------------------------------------
# Tests: score_dos_target
# ---------------------------------------------------------------------------

class TestScoresDosTarget:
    def test_null_target_returns_neutral(self):
        assert score_dos_target(None, None, 10, 100) == 15

    def test_stockout_zero_sales_zero_stock_returns_0(self):
        assert score_dos_target(10, 30, 0, 0) == 0

    def test_no_movement_returns_5(self):
        # avg_daily_sls=0 but stock > 0
        assert score_dos_target(10, 30, 0, 50) == 5

    def test_within_target_returns_25(self):
        # dos = 200/10 = 20, target 15–30
        assert score_dos_target(15, 30, 10, 200) == 25

    def test_excess_above_max_returns_10(self):
        # dos = 500/10 = 50, max=30
        assert score_dos_target(15, 30, 10, 500) == 10

    def test_below_min_returns_5(self):
        # dos = 50/10 = 5, min=15
        assert score_dos_target(15, 30, 10, 50) == 5


# ---------------------------------------------------------------------------
# Tests: score_stockout_risk
# ---------------------------------------------------------------------------

class TestScoreStockoutRisk:
    def test_null_returns_neutral(self):
        assert score_stockout_risk(None) == 20

    def test_no_stockouts_returns_25(self):
        assert score_stockout_risk(0) == 25

    def test_one_stockout_returns_15(self):
        assert score_stockout_risk(1) == 15

    def test_two_stockouts_returns_8(self):
        assert score_stockout_risk(2) == 8

    def test_three_stockouts_returns_0(self):
        assert score_stockout_risk(3) == 0

    def test_many_stockouts_returns_0(self):
        assert score_stockout_risk(10) == 0


# ---------------------------------------------------------------------------
# Tests: score_forecast_accuracy
# ---------------------------------------------------------------------------

class TestScoreForecastAccuracy:
    def test_null_wape_returns_neutral(self):
        assert score_forecast_accuracy(None) == 15

    def test_excellent_under_15pct_returns_25(self):
        assert score_forecast_accuracy(0.10) == 25

    def test_exactly_15pct_is_good(self):
        assert score_forecast_accuracy(0.15) == 20

    def test_good_under_25pct_returns_20(self):
        assert score_forecast_accuracy(0.20) == 20

    def test_fair_under_40pct_returns_15(self):
        assert score_forecast_accuracy(0.35) == 15

    def test_poor_under_60pct_returns_8(self):
        assert score_forecast_accuracy(0.50) == 8

    def test_very_poor_over_60pct_returns_0(self):
        assert score_forecast_accuracy(0.70) == 0

    def test_exactly_60pct_returns_0(self):
        assert score_forecast_accuracy(0.60) == 0


# ---------------------------------------------------------------------------
# Tests: health_tier
# ---------------------------------------------------------------------------

class TestHealthTier:
    def test_80_is_healthy(self):
        assert health_tier(80) == "healthy"

    def test_100_is_healthy(self):
        assert health_tier(100) == "healthy"

    def test_79_is_monitor(self):
        assert health_tier(79) == "monitor"

    def test_60_is_monitor(self):
        assert health_tier(60) == "monitor"

    def test_59_is_at_risk(self):
        assert health_tier(59) == "at_risk"

    def test_40_is_at_risk(self):
        assert health_tier(40) == "at_risk"

    def test_39_is_critical(self):
        assert health_tier(39) == "critical"

    def test_0_is_critical(self):
        assert health_tier(0) == "critical"


# ---------------------------------------------------------------------------
# Tests: composite score ranges
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_all_neutral_no_data_equals_62(self):
        # ss_neutral=12, dos_neutral=15, stockout_neutral=20, forecast_neutral=15 = 62
        score = composite(None, None, None, None, 10, 100, None, None)
        assert score == 62

    def test_perfect_score_equals_100(self):
        # ss=25, dos=25, stockout=25, forecast=25 = 100
        score = composite(100, 2.0, 15, 30, 10, 200, 0, 0.10)
        assert score == 100

    def test_worst_case_equals_0(self):
        # ss=0, dos=0, stockout=0, forecast=0 = 0
        score = composite(100, 0.0, 10, 30, 0, 0, 5, 0.90)
        assert score == 0

    def test_score_within_0_100(self):
        for ss_cov in [None, 0.0, 0.5, 1.0, 2.0]:
            for wape in [None, 0.10, 0.30, 0.70]:
                s = composite(100 if ss_cov else None, ss_cov, 10, 30, 5, 100, 1, wape)
                assert 0 <= s <= 100

    def test_neutral_tier_is_monitor(self):
        score = composite(None, None, None, None, 10, 100, None, None)
        assert health_tier(score) == "monitor"
