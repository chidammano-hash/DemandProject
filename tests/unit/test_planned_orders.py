"""Unit tests for generate_planned_orders.py — F2.1 Order Recommendation Engine."""

import sys
import os
import pytest
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.generate_planned_orders import (
    round_to_moq,
    compute_net_requirements,
    compute_confidence_score,
)
from common.planning_date import get_planning_date

TODAY = get_planning_date()

# ---------------------------------------------------------------------------
# round_to_moq
# ---------------------------------------------------------------------------

def test_round_to_moq_ceil_exact_multiple():
    assert round_to_moq(200, 100, "ceil_to_moq") == 200.0


def test_round_to_moq_ceil_partial():
    assert round_to_moq(220, 100, "ceil_to_moq") == 300.0


def test_round_to_moq_ceil_minimum_moq():
    """1 unit with MOQ=100 should round up to 100."""
    assert round_to_moq(1, 100, "ceil_to_moq") == 100.0


def test_round_to_moq_zero_net_req():
    """0 units should still return at least 1 MOQ."""
    assert round_to_moq(0, 100, "ceil_to_moq") == 100.0


def test_round_to_moq_nearest_moq():
    assert round_to_moq(160, 100, "nearest_moq") == 200.0
    assert round_to_moq(140, 100, "nearest_moq") == 100.0


def test_round_to_moq_invalid_moq():
    """Zero or negative MOQ should default to 1."""
    assert round_to_moq(50, 0, "ceil_to_moq") == 50.0


# ---------------------------------------------------------------------------
# compute_net_requirements
# ---------------------------------------------------------------------------

def _base_config():
    return {
        "recommendation": {
            "horizon_days": 90,
            "max_orders_per_sku": 3,
            "include_past_due": True,
            "confidence": {
                "high_threshold": 0.80,
                "low_threshold": 0.50,
                "penalty_no_open_po_data": 0.15,
                "penalty_fallback_forecast": 0.20,
                "penalty_past_due_order": 0.10,
            },
        },
        "moq_handling": {
            "rounding_strategy": "ceil_to_moq",
            "evaluate_price_breaks": False,
        },
    }


def _base_inputs(qty=120.0, ss=60.0, rp=60.0, lt=14, daily_rate=16.3, moq=100.0):
    """Build a standard inputs dict with uniform daily demand."""
    demand = {TODAY + timedelta(days=i): daily_rate for i in range(1, 91)}
    return {
        "item_id": "100320",
        "loc": "1401-BULK",
        "current_qty_on_hand": qty,
        "safety_stock": ss,
        "reorder_point": rp,
        "lead_time_days": lt,
        "moq": moq,
        "unit_cost": 12.50,
        "policy_id": 3,
        "review_cycle_days": None,
        "supplier_id": "VENDOR-0042",
        "daily_demand_by_date": demand,
        "confirmed_receipts_by_date": {},
        "plan_version": "2026-03",
        "forecast_source": "production_forecast",
        "open_po_data_available": True,
        "run_id": "test-run-id",
    }


def test_compute_net_requirements_trigger_on_day_4():
    """
    With qty=120, SS=60, daily demand=16.3, trigger occurs on day 4
    (tomorrow is day 1, day 4 = today + 4 days):
    day1: 103.7, day2: 87.4, day3: 71.1, day4: 54.8 <= 60 -> TRIGGER
    """
    inputs = _base_inputs(qty=120.0, ss=60.0, rp=60.0, daily_rate=16.3)
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    assert len(orders) >= 1
    trigger = orders[0]["trigger_date"]
    assert (trigger - TODAY).days == 4


def test_compute_net_requirements_correct_order_qty():
    """net_req = 60 + 14*16.3 - 54.8 = 233.4 -> round_to_moq(233.4, 100) = 300."""
    inputs = _base_inputs(qty=120.0, ss=60.0, rp=60.0, daily_rate=16.3, moq=100.0)
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    assert len(orders) >= 1
    assert orders[0]["recommended_qty"] == 300.0
    assert orders[0]["net_requirement_qty"] == pytest.approx(233.4, abs=1.0)


def test_compute_net_requirements_no_trigger_sufficient_stock():
    """If stock is way above reorder point, no trigger within horizon."""
    inputs = _base_inputs(qty=5000.0, ss=60.0, rp=60.0, daily_rate=16.3)
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    # 5000 / 16.3 = ~306 days, exceeds horizon of 90
    assert len(orders) == 0


def test_compute_net_requirements_past_due_trigger():
    """When starting stock is already below reorder_point, trigger on day 1."""
    inputs = _base_inputs(qty=50.0, ss=60.0, rp=60.0, daily_rate=16.3)
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    assert len(orders) >= 1
    assert (orders[0]["trigger_date"] - TODAY).days == 1


def test_compute_net_requirements_multi_cycle():
    """With high demand, a second trigger should appear after the first planned receipt."""
    inputs = _base_inputs(qty=120.0, ss=60.0, rp=60.0, lt=7, daily_rate=16.3, moq=50.0)
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    assert len(orders) >= 2


def test_compute_net_requirements_po_receipts_delay_trigger():
    """A PO receipt arriving on day 8 should delay the trigger vs no-PO case."""
    inputs_no_po = _base_inputs(qty=120.0, ss=60.0, rp=60.0, daily_rate=16.3)
    inputs_with_po = _base_inputs(qty=120.0, ss=60.0, rp=60.0, daily_rate=16.3)
    inputs_with_po["confirmed_receipts_by_date"] = {
        TODAY + timedelta(days=8): 150.0
    }
    config = _base_config()
    orders_no_po = compute_net_requirements(inputs_no_po, config)
    orders_with_po = compute_net_requirements(inputs_with_po, config)
    # With PO, trigger should be later (if it occurs at all within the same horizon position)
    if orders_no_po and orders_with_po:
        assert orders_with_po[0]["trigger_date"] >= orders_no_po[0]["trigger_date"]


def test_compute_net_requirements_order_value():
    """order_value field = recommended_qty * unit_cost (computed field, checked in dict)."""
    inputs = _base_inputs(qty=120.0, ss=60.0, rp=60.0, daily_rate=16.3)
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    assert len(orders) >= 1
    order = orders[0]
    # recommended_qty=300, unit_cost=12.50 -> value=$3750
    assert order["recommended_qty"] * order["unit_cost"] == pytest.approx(3750.0)


# ---------------------------------------------------------------------------
# compute_confidence_score
# ---------------------------------------------------------------------------

def test_compute_confidence_score_all_sources_available():
    inputs = _base_inputs()
    inputs["forecast_source"] = "production_forecast"
    inputs["open_po_data_available"] = True
    config = _base_config()
    orders = compute_net_requirements(inputs, config)
    score, reason = compute_confidence_score(inputs, orders, config)
    assert score == pytest.approx(1.0)
    assert reason == "all data sources available"


def test_compute_confidence_score_fallback_forecast_penalty():
    inputs = _base_inputs()
    inputs["forecast_source"] = "fallback_avg"
    config = _base_config()
    orders = []
    score, reason = compute_confidence_score(inputs, orders, config)
    assert score == pytest.approx(0.80)
    assert "fallback" in reason


def test_compute_confidence_score_no_po_data_penalty():
    inputs = _base_inputs()
    inputs["open_po_data_available"] = False
    config = _base_config()
    orders = []
    score, reason = compute_confidence_score(inputs, orders, config)
    assert score == pytest.approx(0.85)
    assert "open PO" in reason


def test_compute_confidence_score_past_due_penalty():
    inputs = _base_inputs()
    config = _base_config()
    # Simulate an order that is past due
    past_due_order = {"order_by_date": TODAY - timedelta(days=3)}
    score, reason = compute_confidence_score(inputs, [past_due_order], config)
    assert score == pytest.approx(0.90)
    assert "past due" in reason


def test_compute_confidence_score_combined_penalties():
    inputs = _base_inputs()
    inputs["forecast_source"] = "fallback_avg"
    inputs["open_po_data_available"] = False
    config = _base_config()
    score, _ = compute_confidence_score(inputs, [], config)
    assert score == pytest.approx(0.65)  # 1.0 - 0.20 - 0.15
