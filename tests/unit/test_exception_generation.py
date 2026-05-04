"""Unit tests for IPfeature7 — Exception Queue generation logic.

Tests the pure-Python detection and recommendation functions in isolation
(no DB required).
"""
from __future__ import annotations

import datetime

import pytest

from scripts.inventory.generate_replenishment_exceptions import (
    compute_financial_impact,
    compute_recommendation,
    detect_exception_type,
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


# ---------------------------------------------------------------------------
# compute_financial_impact
# ---------------------------------------------------------------------------

class TestComputeFinancialImpact:
    def test_stockout_loss_of_sales(self):
        """Stockout exception computes lost sales based on demand and margin."""
        fin = compute_financial_impact(
            exception_type="stockout",
            current_qty=0,
            ss_combined=200,
            unit_cost=20.0,
            demand_mean_monthly=304.4,   # daily_demand = 10
            current_dos=0.0,
            lead_time_mean_days=14.0,
        )
        assert fin["unit_cost"] == 20.0
        assert fin["unit_margin"] == 6.0   # 20 * 0.30
        assert fin["daily_demand_rate"] == pytest.approx(10.0, abs=0.01)
        # days_at_risk = 14 - 0 = 14; loss_7d = 10 * 6 * 7 = 420
        assert fin["loss_of_sales_7d"] == pytest.approx(420.0, abs=1.0)
        # loss_30d = 10 * 6 * 14 = 840 (capped at days_at_risk=14 < 30)
        assert fin["loss_of_sales_30d"] == pytest.approx(840.0, abs=1.0)
        assert fin["monthly_holding_cost"] == 0.0
        assert fin["financial_impact_total"] == fin["loss_of_sales_7d"]

    def test_below_ss_uses_days_at_risk(self):
        """Below-SS computes days_at_risk = lead_time - current_dos."""
        fin = compute_financial_impact(
            exception_type="below_ss",
            current_qty=50,
            ss_combined=200,
            unit_cost=10.0,
            demand_mean_monthly=304.4,   # daily = 10
            current_dos=5.0,
            lead_time_mean_days=10.0,
        )
        # days_at_risk = 10 - 5 = 5; margin = 3.0
        # loss_7d = 10 * 3 * min(7, 5) = 150
        assert fin["loss_of_sales_7d"] == pytest.approx(150.0, abs=1.0)

    def test_excess_holding_cost(self):
        """Excess exception computes monthly holding cost."""
        fin = compute_financial_impact(
            exception_type="excess",
            current_qty=1000,
            ss_combined=200,
            unit_cost=10.0,
            demand_mean_monthly=100.0,
            current_dos=300.0,
            lead_time_mean_days=7.0,
        )
        # excess_qty = 1000 - 2*200 = 600
        # monthly_holding = 600 * 10 * 0.25 / 12 = 125.0
        assert fin["monthly_holding_cost"] == pytest.approx(125.0, abs=0.01)
        assert fin["loss_of_sales_7d"] == 0.0
        assert fin["financial_impact_total"] == fin["monthly_holding_cost"]

    def test_zero_velocity_no_impact(self):
        """Zero velocity exception has zero financial impact."""
        fin = compute_financial_impact(
            exception_type="zero_velocity",
            current_qty=500,
            ss_combined=None,
            unit_cost=10.0,
            demand_mean_monthly=0.0,
            current_dos=None,
            lead_time_mean_days=7.0,
        )
        assert fin["financial_impact_total"] == 0.0
        assert fin["loss_of_sales_7d"] == 0.0
        assert fin["monthly_holding_cost"] == 0.0

    def test_default_unit_cost_when_none(self):
        """Uses $10 default when unit_cost is None."""
        fin = compute_financial_impact(
            exception_type="stockout",
            current_qty=0,
            ss_combined=100,
            unit_cost=None,
            demand_mean_monthly=304.4,
            current_dos=0.0,
            lead_time_mean_days=7.0,
        )
        assert fin["unit_cost"] == 10.0
        assert fin["unit_margin"] == 3.0

    def test_default_unit_cost_when_zero(self):
        """Uses $10 default when unit_cost is 0."""
        fin = compute_financial_impact(
            exception_type="below_rop",
            current_qty=150,
            ss_combined=200,
            unit_cost=0.0,
            demand_mean_monthly=304.4,
            current_dos=30.0,
            lead_time_mean_days=14.0,
        )
        assert fin["unit_cost"] == 10.0

    def test_excess_no_ss_yields_full_qty_excess(self):
        """When ss_combined is None, excess_qty = current_qty - 0 = current_qty."""
        fin = compute_financial_impact(
            exception_type="excess",
            current_qty=500,
            ss_combined=None,
            unit_cost=20.0,
            demand_mean_monthly=100.0,
            current_dos=150.0,
            lead_time_mean_days=7.0,
        )
        # excess_qty = 500 - 0 = 500; holding = 500 * 20 * 0.25 / 12 = 208.33
        assert fin["monthly_holding_cost"] == pytest.approx(208.33, abs=0.01)

    def test_no_days_at_risk_when_dos_exceeds_lead_time(self):
        """When current_dos > lead_time, days_at_risk = 0 so no lost sales."""
        fin = compute_financial_impact(
            exception_type="below_rop",
            current_qty=150,
            ss_combined=200,
            unit_cost=10.0,
            demand_mean_monthly=304.4,
            current_dos=20.0,
            lead_time_mean_days=10.0,
        )
        # days_at_risk = max(0, 10 - 20) = 0
        assert fin["loss_of_sales_7d"] == 0.0
        assert fin["loss_of_sales_30d"] == 0.0

    def test_stockout_no_lead_time_assumes_7_days(self):
        """When lead_time is None and DOS=0 (stocked out), assume 7-day risk window."""
        fin = compute_financial_impact(
            exception_type="stockout",
            current_qty=0,
            ss_combined=100,
            unit_cost=10.0,
            demand_mean_monthly=304.4,   # daily = 10
            current_dos=0.0,
            lead_time_mean_days=None,    # no lead time data
        )
        # days_at_risk = 7 (fallback); margin = 3.0; loss_7d = 10 * 3 * 7 = 210
        assert fin["loss_of_sales_7d"] == pytest.approx(210.0, abs=1.0)
        assert fin["financial_impact_total"] == fin["loss_of_sales_7d"]

    def test_below_rop_no_lead_time_with_dos_is_zero(self):
        """When lead_time is None but DOS > 0, days_at_risk = 0."""
        fin = compute_financial_impact(
            exception_type="below_rop",
            current_qty=150,
            ss_combined=200,
            unit_cost=10.0,
            demand_mean_monthly=304.4,
            current_dos=10.0,
            lead_time_mean_days=None,
        )
        # DOS > 0 and no lead time: days_at_risk = 0
        assert fin["loss_of_sales_7d"] == 0.0
