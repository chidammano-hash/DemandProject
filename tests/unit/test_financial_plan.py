"""Unit tests for scripts/compute_financial_plan.py — F4.1."""

import pytest
from scripts.compute_financial_plan import (
    compute_inventory_value,
    compute_carrying_cost,
    compute_excess_value,
    compute_budget_utilization,
)


class TestComputeInventoryValue:
    def test_basic(self):
        assert compute_inventory_value(100.0, 24.00) == pytest.approx(2400.0, abs=1e-2)

    def test_zero_qty(self):
        assert compute_inventory_value(0.0, 24.00) == pytest.approx(0.0, abs=1e-4)

    def test_floor_at_zero(self):
        assert compute_inventory_value(-5.0, 24.00) == pytest.approx(0.0, abs=1e-4)


class TestComputeCarryingCost:
    def test_monthly_cost(self):
        # $10,000 × 25% / 12 = $208.33
        result = compute_carrying_cost(10000.0, 0.25, 1.0)
        assert result == pytest.approx(208.33, abs=0.01)

    def test_two_months(self):
        result = compute_carrying_cost(10000.0, 0.25, 2.0)
        assert result == pytest.approx(416.67, abs=0.01)

    def test_zero_value(self):
        assert compute_carrying_cost(0.0, 0.25) == pytest.approx(0.0, abs=1e-4)


class TestComputeExcessValue:
    def test_no_excess_within_threshold(self):
        # 500 units on hand, daily demand=10, threshold=180 days → max_normal=1800 → no excess
        result = compute_excess_value(500.0, 10.0, 24.0, 180)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_excess_beyond_threshold(self):
        # 5000 units on hand, daily demand=10, threshold=180 → max_normal=1800 → excess=3200
        result = compute_excess_value(5000.0, 10.0, 24.0, 180)
        assert result == pytest.approx(3200 * 24.0, abs=1e-2)

    def test_zero_demand_returns_zero(self):
        result = compute_excess_value(5000.0, 0.0, 24.0, 180)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_zero_cost_returns_zero(self):
        result = compute_excess_value(5000.0, 10.0, 0.0, 180)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestComputeBudgetUtilization:
    def test_under_budget(self):
        util, breached = compute_budget_utilization(800.0, 1000.0)
        assert util == pytest.approx(80.0, abs=1e-2)
        assert not breached

    def test_exactly_at_budget(self):
        util, breached = compute_budget_utilization(1000.0, 1000.0)
        assert util == pytest.approx(100.0, abs=1e-2)
        assert breached

    def test_over_budget(self):
        util, breached = compute_budget_utilization(1200.0, 1000.0)
        assert util == pytest.approx(120.0, abs=1e-2)
        assert breached

    def test_zero_budget_no_breach(self):
        util, breached = compute_budget_utilization(500.0, 0.0)
        assert util == pytest.approx(0.0, abs=1e-4)
        assert not breached
