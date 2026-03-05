"""Unit tests for IPfeature7 — Exception Queue generation logic.

Tests the pure-Python detection and recommendation functions in isolation
(no DB required).
"""
from __future__ import annotations

import datetime
import pytest

from scripts.generate_replenishment_exceptions import (
    detect_exception_type,
    compute_recommendation,
)

TODAY = datetime.date(2026, 3, 4)


# ---------------------------------------------------------------------------
# detect_exception_type
# ---------------------------------------------------------------------------

class TestDetectExceptionType:
    def test_zero_qty_is_stockout_critical(self):
        exc, sev = detect_exception_type(0, ss_combined=100, reorder_point=200,
                                          current_dos=0, target_dos_max=30, avg_daily_sls=5)
        assert exc == "stockout"
        assert sev == "critical"

    def test_negative_qty_is_stockout(self):
        exc, sev = detect_exception_type(-5, ss_combined=100, reorder_point=200,
                                          current_dos=None, target_dos_max=30, avg_daily_sls=5)
        assert exc == "stockout"
        assert sev == "critical"

    def test_below_ss_critical_when_coverage_under_50pct(self):
        # current_qty=40, ss=100, ratio=0.4 < 0.5 → critical
        exc, sev = detect_exception_type(40, ss_combined=100, reorder_point=200,
                                          current_dos=8, target_dos_max=30, avg_daily_sls=5)
        assert exc == "below_ss"
        assert sev == "critical"

    def test_below_ss_high_when_coverage_above_50pct(self):
        # current_qty=60, ss=100, ratio=0.6 > 0.5 → high
        exc, sev = detect_exception_type(60, ss_combined=100, reorder_point=200,
                                          current_dos=12, target_dos_max=30, avg_daily_sls=5)
        assert exc == "below_ss"
        assert sev == "high"

    def test_below_rop_high(self):
        # qty=150 > ss=100, but qty <= rop=200
        exc, sev = detect_exception_type(150, ss_combined=100, reorder_point=200,
                                          current_dos=30, target_dos_max=60, avg_daily_sls=5)
        assert exc == "below_rop"
        assert sev == "high"

    def test_excess_medium_when_dos_under_180(self):
        # dos=100, target_dos_max=60, dos > 60 * 1.5=90 → excess; dos<180 → medium
        exc, sev = detect_exception_type(500, ss_combined=None, reorder_point=None,
                                          current_dos=100, target_dos_max=60, avg_daily_sls=5)
        assert exc == "excess"
        assert sev == "medium"

    def test_excess_low_when_dos_at_least_180(self):
        # dos=200, target_dos_max=60 → excess; dos>=180 → low
        exc, sev = detect_exception_type(1000, ss_combined=None, reorder_point=None,
                                          current_dos=200, target_dos_max=60, avg_daily_sls=5)
        assert exc == "excess"
        assert sev == "low"

    def test_zero_velocity_low(self):
        exc, sev = detect_exception_type(50, ss_combined=None, reorder_point=None,
                                          current_dos=None, target_dos_max=None, avg_daily_sls=0)
        assert exc == "zero_velocity"
        assert sev == "low"

    def test_no_exception_when_within_range(self):
        # qty=250, ss=100, rop=200, dos=50, target_max=60 → within target
        exc, sev = detect_exception_type(250, ss_combined=100, reorder_point=200,
                                          current_dos=50, target_dos_max=60, avg_daily_sls=5)
        assert exc is None
        assert sev is None

    def test_no_exception_when_ss_is_none(self):
        # No ss data — excess threshold not reached, not zero-velocity
        exc, sev = detect_exception_type(500, ss_combined=None, reorder_point=None,
                                          current_dos=20, target_dos_max=30, avg_daily_sls=5)
        assert exc is None

    def test_ss_zero_does_not_trigger_below_ss(self):
        # ss_combined=0 → no below_ss trigger
        exc, sev = detect_exception_type(50, ss_combined=0, reorder_point=None,
                                          current_dos=10, target_dos_max=30, avg_daily_sls=5)
        assert exc is None

    def test_rop_zero_does_not_trigger_below_rop(self):
        exc, sev = detect_exception_type(50, ss_combined=0, reorder_point=0,
                                          current_dos=10, target_dos_max=30, avg_daily_sls=5)
        assert exc is None


# ---------------------------------------------------------------------------
# compute_recommendation
# ---------------------------------------------------------------------------

class TestComputeRecommendation:
    def test_stockout_critical_orders_eoq(self):
        qty, order_by, receipt = compute_recommendation(
            "stockout", "critical",
            current_qty=0,
            ss_combined=200,
            effective_eoq=100,
            demand_mean_monthly=500,
            review_cycle_days=7,
            lead_time_mean_days=5,
            max_eoq_months_supply=6,
            today=TODAY,
        )
        # gap=200, rec=max(100, 200+50)=250
        assert qty > 0
        assert order_by == TODAY  # critical → today

    def test_high_severity_order_by_uses_review_cycle(self):
        qty, order_by, receipt = compute_recommendation(
            "below_rop", "high",
            current_qty=150,
            ss_combined=200,
            effective_eoq=100,
            demand_mean_monthly=500,
            review_cycle_days=7,
            lead_time_mean_days=10,
            max_eoq_months_supply=6,
            today=TODAY,
        )
        assert order_by == TODAY + datetime.timedelta(days=7)

    def test_expected_receipt_adds_lead_time(self):
        _, order_by, receipt = compute_recommendation(
            "below_ss", "high",
            current_qty=50,
            ss_combined=200,
            effective_eoq=100,
            demand_mean_monthly=300,
            review_cycle_days=7,
            lead_time_mean_days=14,
            max_eoq_months_supply=6,
            today=TODAY,
        )
        assert receipt == order_by + datetime.timedelta(days=14)

    def test_order_qty_at_least_effective_eoq(self):
        # gap=0, eoq=100 → max(100, 0+50)=100
        qty, _, _ = compute_recommendation(
            "below_rop", "high",
            current_qty=200,
            ss_combined=200,
            effective_eoq=100,
            demand_mean_monthly=300,
            review_cycle_days=7,
            lead_time_mean_days=5,
            max_eoq_months_supply=6,
            today=TODAY,
        )
        assert qty >= 100

    def test_order_qty_capped_at_max_months(self):
        # demand=1000/month, max=3 months → cap=3000
        qty, _, _ = compute_recommendation(
            "stockout", "critical",
            current_qty=0,
            ss_combined=5000,  # very large ss → gap=5000
            effective_eoq=100,
            demand_mean_monthly=1000,
            review_cycle_days=7,
            lead_time_mean_days=0,
            max_eoq_months_supply=3,
            today=TODAY,
        )
        assert qty <= 3000

    def test_excess_returns_zero_order(self):
        qty, order_by, receipt = compute_recommendation(
            "excess", "medium",
            current_qty=1000,
            ss_combined=None,
            effective_eoq=100,
            demand_mean_monthly=200,
            review_cycle_days=7,
            lead_time_mean_days=5,
            max_eoq_months_supply=6,
            today=TODAY,
        )
        assert qty == 0.0
        assert order_by is None
        assert receipt is None

    def test_zero_velocity_returns_zero_order(self):
        qty, order_by, receipt = compute_recommendation(
            "zero_velocity", "low",
            current_qty=500,
            ss_combined=None,
            effective_eoq=0,
            demand_mean_monthly=0,
            review_cycle_days=7,
            lead_time_mean_days=5,
            max_eoq_months_supply=6,
            today=TODAY,
        )
        assert qty == 0.0
        assert order_by is None
