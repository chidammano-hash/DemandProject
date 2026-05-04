"""Unit tests for scripts/run_supply_chain_scenario.py — F4.4."""

import pytest
from scripts.inventory.run_supply_chain_scenario import (
    compute_adjusted_lead_time,
    compute_available_supply,
    compute_stockout_days,
    compute_scenario_financial_impact,
)


class TestComputeAdjustedLeadTime:
    def test_supplier_delay_50pct_4weeks(self):
        # base=10, impact=50%, duration=4 weeks → increase = 4×7×0.50 = 14
        adj, increase = compute_adjusted_lead_time(10.0, "supplier_delay", 50.0, 4)
        assert adj == pytest.approx(24.0, abs=1e-1)
        assert increase == pytest.approx(14.0, abs=1e-1)

    def test_transport_disruption_100pct(self):
        # Full transport disruption for 2 weeks → +14 days
        adj, increase = compute_adjusted_lead_time(10.0, "transport_disruption", 100.0, 2)
        assert adj == pytest.approx(24.0, abs=1e-1)
        assert increase == pytest.approx(14.0, abs=1e-1)

    def test_capacity_constraint_no_lt_change(self):
        # Capacity constraint doesn't change lead time
        adj, increase = compute_adjusted_lead_time(10.0, "capacity_constraint", 30.0, 4)
        assert adj == pytest.approx(10.0, abs=1e-4)
        assert increase == pytest.approx(0.0, abs=1e-4)

    def test_zero_impact(self):
        adj, increase = compute_adjusted_lead_time(10.0, "supplier_delay", 0.0, 4)
        assert adj == pytest.approx(10.0, abs=1e-4)
        assert increase == pytest.approx(0.0, abs=1e-4)


class TestComputeAvailableSupply:
    def test_capacity_constraint_30pct(self):
        # 30% capacity constraint → 70% available
        avail, shortfall = compute_available_supply(100.0, "capacity_constraint", 30.0)
        assert avail == pytest.approx(70.0, abs=1e-2)
        assert shortfall == pytest.approx(30.0, abs=1e-2)

    def test_quality_hold_25pct(self):
        avail, shortfall = compute_available_supply(100.0, "quality_hold", 25.0)
        assert avail == pytest.approx(75.0, abs=1e-2)
        assert shortfall == pytest.approx(25.0, abs=1e-2)

    def test_supplier_delay_full_supply(self):
        # Supplier delay doesn't reduce supply quantity
        avail, shortfall = compute_available_supply(100.0, "supplier_delay", 50.0)
        assert avail == pytest.approx(100.0, abs=1e-4)
        assert shortfall == pytest.approx(0.0, abs=1e-4)

    def test_100pct_constraint(self):
        avail, shortfall = compute_available_supply(100.0, "capacity_constraint", 100.0)
        assert avail == pytest.approx(0.0, abs=1e-4)
        assert shortfall == pytest.approx(100.0, abs=1e-2)


class TestComputeStockoutDays:
    def test_no_stockout_sufficient_supply(self):
        # on_hand=500 + available=300 = 800 → at 45/day, depletion=17.8 days > LT=10
        result = compute_stockout_days(500.0, 45.0, 10.0, 300.0)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_stockout_when_insufficient(self):
        # on_hand=100 + available=0 → depletion=100/45=2.2 days; LT=24 → stockout=21.8 days
        result = compute_stockout_days(100.0, 45.0, 24.0, 0.0)
        assert result > 0

    def test_zero_demand_no_stockout(self):
        result = compute_stockout_days(100.0, 0.0, 24.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestComputeScenarioFinancialImpact:
    def test_stockout_cost(self):
        result = compute_scenario_financial_impact(
            stockout_days=5.0, daily_demand=10.0, unit_cost=24.0,
            stockout_cost_per_unit=10.0, excess_qty=0.0
        )
        assert result["stockout_units"] == pytest.approx(50.0, abs=1e-1)
        assert result["stockout_cost"] == pytest.approx(500.0, abs=1e-2)
        assert result["total_impact"] == pytest.approx(500.0, abs=1e-2)

    def test_no_disruption_zero_impact(self):
        result = compute_scenario_financial_impact(0.0, 10.0, 24.0, 10.0)
        assert result["total_impact"] == pytest.approx(0.0, abs=1e-4)

    def test_holding_cost_for_excess(self):
        result = compute_scenario_financial_impact(
            stockout_days=0.0, daily_demand=0.0, unit_cost=24.0,
            stockout_cost_per_unit=10.0, excess_qty=100.0, holding_cost_pct=0.52
        )
        # holding_cost = 100 × 24 × (0.52/52) = 24.0 per week
        assert result["holding_cost"] == pytest.approx(24.0, abs=1e-2)
