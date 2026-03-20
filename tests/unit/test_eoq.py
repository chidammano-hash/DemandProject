"""Unit tests for scripts/compute_eoq.py — IPfeature4."""

import math
import pytest

from scripts.compute_eoq import (
    compute_eoq,
    compute_effective_eoq,
    compute_eoq_metrics,
    sensitivity_curve,
)


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg():
    return {
        "costs": {
            "default_ordering_cost": 50.0,
            "default_holding_cost_pct": 0.25,
            "default_unit_cost": 10.0,
            "default_moq": 1.0,
        },
        "constraints": {
            "max_eoq_months_supply": 6.0,
            "min_annual_demand": 0.001,
        },
        "sensitivity": {
            "ordering_cost_min": 5.0,
            "ordering_cost_max": 200.0,
            "ordering_cost_steps": 10,
        },
    }


# ---------------------------------------------------------------------------
# compute_eoq
# ---------------------------------------------------------------------------

class TestComputeEoq:
    def test_known_result(self):
        """D=1200, S=50, H=0.25, C=10 → EOQ = sqrt(2*1200*50/(0.25*10)) ≈ 219.09."""
        result = compute_eoq(1200, 50, 0.25, 10)
        assert result is not None
        assert abs(result - 219.0890) < 0.01

    def test_zero_annual_demand_returns_none(self):
        assert compute_eoq(0, 50, 0.25, 10) is None

    def test_negative_annual_demand_returns_none(self):
        assert compute_eoq(-100, 50, 0.25, 10) is None

    def test_zero_holding_cost_returns_none(self):
        assert compute_eoq(1200, 50, 0.0, 10) is None

    def test_zero_unit_cost_returns_none(self):
        assert compute_eoq(1200, 50, 0.25, 0.0) is None

    def test_formula_doubles_with_double_demand(self):
        """EOQ ∝ sqrt(D), so doubling D → EOQ multiplied by sqrt(2)."""
        eoq1 = compute_eoq(1000, 50, 0.25, 10)
        eoq2 = compute_eoq(2000, 50, 0.25, 10)
        assert eoq1 is not None and eoq2 is not None
        assert abs(eoq2 / eoq1 - math.sqrt(2)) < 1e-6

    def test_formula_with_large_demand(self):
        result = compute_eoq(1_000_000, 50, 0.25, 10)
        assert result is not None
        assert result > 0

    def test_returns_float(self):
        result = compute_eoq(1200, 50, 0.25, 10)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_effective_eoq
# ---------------------------------------------------------------------------

class TestComputeEffectiveEoq:
    def test_eoq_above_moq_no_cap(self):
        """EOQ > MOQ, no cap → returns EOQ."""
        result = compute_effective_eoq(100.0, 50.0, 6.0, 100.0)
        assert result == 100.0

    def test_moq_floor_applied(self):
        """EOQ < MOQ → returns MOQ."""
        result = compute_effective_eoq(10.0, 50.0, 6.0, 100.0)
        assert result == 50.0

    def test_cap_applied_when_eoq_exceeds_months_supply(self):
        """EOQ = 700, max 6 months, demand = 100/month → cap = 600."""
        result = compute_effective_eoq(700.0, 1.0, 6.0, 100.0)
        assert result == 600.0

    def test_zero_demand_no_cap(self):
        """Zero demand → cap is infinity, so no cap applied."""
        result = compute_effective_eoq(200.0, 50.0, 6.0, 0.0)
        assert result == 200.0

    def test_moq_also_capped(self):
        """MOQ exceeds months-supply cap → result capped."""
        result = compute_effective_eoq(10.0, 800.0, 6.0, 100.0)
        assert result == 600.0

    def test_effective_eoq_never_below_moq(self):
        """effective_eoq >= moq always when cap > moq."""
        result = compute_effective_eoq(5.0, 20.0, 6.0, 100.0)
        assert result >= 20.0


# ---------------------------------------------------------------------------
# compute_eoq_metrics
# ---------------------------------------------------------------------------

class TestComputeEoqMetrics:
    def test_returns_none_for_zero_demand(self, cfg):
        result = compute_eoq_metrics(0.0, cfg)
        assert result is None

    def test_returns_none_below_min_annual_demand(self, cfg):
        """monthly demand so small that annual < min_annual_demand."""
        result = compute_eoq_metrics(0.00001, cfg)
        assert result is None

    def test_returns_dict_for_valid_demand(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        assert isinstance(result, dict)

    def test_all_expected_keys(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        expected_keys = {
            "annual_demand", "ordering_cost", "holding_cost_pct", "unit_cost", "moq",
            "eoq", "effective_eoq", "eoq_cycle_stock", "order_frequency",
            "annual_holding_cost", "annual_order_cost", "total_annual_cost",
        }
        assert expected_keys.issubset(result.keys())

    def test_annual_demand_is_12x_monthly(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        assert abs(result["annual_demand"] - 1200.0) < 1e-9

    def test_cycle_stock_is_half_effective_eoq(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        assert abs(result["eoq_cycle_stock"] - result["effective_eoq"] / 2.0) < 1e-9

    def test_total_cost_is_holding_plus_ordering(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        assert abs(result["total_annual_cost"] - (result["annual_holding_cost"] + result["annual_order_cost"])) < 1e-6

    def test_effective_eoq_ge_moq(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        assert result["effective_eoq"] >= result["moq"]

    def test_override_ordering_cost(self, cfg):
        result = compute_eoq_metrics(100.0, cfg, ordering_cost=100.0)
        assert result is not None
        assert result["ordering_cost"] == 100.0

    def test_override_unit_cost(self, cfg):
        result = compute_eoq_metrics(100.0, cfg, unit_cost=5.0)
        assert result is not None
        assert result["unit_cost"] == 5.0

    def test_override_moq(self, cfg):
        result = compute_eoq_metrics(100.0, cfg, moq=200.0)
        assert result is not None
        assert result["moq"] == 200.0
        assert result["effective_eoq"] >= 200.0

    def test_order_frequency_positive(self, cfg):
        result = compute_eoq_metrics(100.0, cfg)
        assert result is not None
        assert result["order_frequency"] > 0


# ---------------------------------------------------------------------------
# sensitivity_curve
# ---------------------------------------------------------------------------

class TestSensitivityCurve:
    def test_returns_list(self, cfg):
        result = sensitivity_curve(100.0, cfg)
        assert isinstance(result, list)

    def test_length_matches_steps(self, cfg):
        result = sensitivity_curve(100.0, cfg)
        assert len(result) == cfg["sensitivity"]["ordering_cost_steps"]

    def test_each_entry_has_required_keys(self, cfg):
        result = sensitivity_curve(100.0, cfg)
        for entry in result:
            assert "ordering_cost" in entry
            assert "eoq" in entry
            assert "effective_eoq" in entry
            assert "total_annual_cost" in entry

    def test_eoq_increases_with_ordering_cost(self, cfg):
        """Higher ordering cost → bigger EOQ (monotone relationship)."""
        result = sensitivity_curve(100.0, cfg)
        eoqs = [r["eoq"] for r in result]
        assert all(eoqs[i] <= eoqs[i + 1] for i in range(len(eoqs) - 1))

    def test_ordering_cost_range(self, cfg):
        result = sensitivity_curve(100.0, cfg)
        first = result[0]["ordering_cost"]
        last = result[-1]["ordering_cost"]
        assert abs(first - cfg["sensitivity"]["ordering_cost_min"]) < 0.01
        assert abs(last - cfg["sensitivity"]["ordering_cost_max"]) < 0.01

    def test_zero_demand_returns_empty_or_no_crash(self, cfg):
        """Zero demand should not crash, might return empty list."""
        result = sensitivity_curve(0.0, cfg)
        assert isinstance(result, list)
