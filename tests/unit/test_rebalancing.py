"""Unit tests for scripts/compute_rebalancing.py — pure logic, no DB."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inventory.compute_rebalancing import (
    assign_urgency,
    build_transfer_candidates,
    compute_financials,
    compute_network_balance,
    detect_imbalances,
    greedy_solver,
)


# ---------------------------------------------------------------------------
# detect_imbalances
# ---------------------------------------------------------------------------

def test_detect_imbalances_basic():
    """1 item, 2 locs: one excess (ratio 2.0 > 1.5), one shortage (ratio 0.5 < 0.8)."""
    state = {
        ("ITEM1", "LOC_A"): {
            "item_id": "ITEM1", "loc": "LOC_A",
            "on_hand": 200, "daily_sales": 10, "ss_target": 100,
            "dos": 20, "abc_vol": "A", "reorder_point": 80,
        },
        ("ITEM1", "LOC_B"): {
            "item_id": "ITEM1", "loc": "LOC_B",
            "on_hand": 50, "daily_sales": 10, "ss_target": 100,
            "dos": 5, "abc_vol": "A", "reorder_point": 80,
        },
    }
    result = detect_imbalances(state, excess_pct=1.50, shortage_pct=0.80)
    assert "ITEM1" in result
    assert len(result["ITEM1"]["excess"]) == 1
    assert len(result["ITEM1"]["shortage"]) == 1
    assert result["ITEM1"]["excess"][0]["loc"] == "LOC_A"
    assert result["ITEM1"]["shortage"][0]["loc"] == "LOC_B"


def test_detect_imbalances_no_shortage():
    """Both locs are excess — no item qualifies (need BOTH excess + shortage)."""
    state = {
        ("ITEM1", "LOC_A"): {
            "item_id": "ITEM1", "loc": "LOC_A",
            "on_hand": 200, "daily_sales": 10, "ss_target": 100,
            "dos": 20, "abc_vol": "A", "reorder_point": 80,
        },
        ("ITEM1", "LOC_B"): {
            "item_id": "ITEM1", "loc": "LOC_B",
            "on_hand": 200, "daily_sales": 10, "ss_target": 100,
            "dos": 20, "abc_vol": "A", "reorder_point": 80,
        },
    }
    result = detect_imbalances(state, excess_pct=1.50, shortage_pct=0.80)
    assert result == {}


def test_detect_imbalances_single_location():
    """Item at only 1 location — cannot have both excess and shortage."""
    state = {
        ("ITEM1", "LOC_A"): {
            "item_id": "ITEM1", "loc": "LOC_A",
            "on_hand": 200, "daily_sales": 10, "ss_target": 100,
            "dos": 20, "abc_vol": "A", "reorder_point": 80,
        },
    }
    result = detect_imbalances(state, excess_pct=1.50, shortage_pct=0.80)
    assert result == {}


def test_detect_imbalances_no_ss_target():
    """ss_target is None — location should be skipped entirely."""
    state = {
        ("ITEM1", "LOC_A"): {
            "item_id": "ITEM1", "loc": "LOC_A",
            "on_hand": 200, "daily_sales": 10, "ss_target": None,
            "dos": 20, "abc_vol": "A", "reorder_point": 80,
        },
        ("ITEM1", "LOC_B"): {
            "item_id": "ITEM1", "loc": "LOC_B",
            "on_hand": 50, "daily_sales": 10, "ss_target": 100,
            "dos": 5, "abc_vol": "A", "reorder_point": 80,
        },
    }
    result = detect_imbalances(state, excess_pct=1.50, shortage_pct=0.80)
    # LOC_A skipped (None ss), LOC_B is shortage but no excess peer → empty
    assert result == {}


# ---------------------------------------------------------------------------
# build_transfer_candidates
# ---------------------------------------------------------------------------

def _make_imbalances():
    """Helper: 1 item with 1 excess loc and 1 shortage loc."""
    return {
        "ITEM1": {
            "excess": [{
                "item_id": "ITEM1", "loc": "LOC_A",
                "on_hand": 200, "ss_target": 100, "dos": 20,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
            "shortage": [{
                "item_id": "ITEM1", "loc": "LOC_B",
                "on_hand": 50, "ss_target": 100, "dos": 5,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
        }
    }


def _base_config(**overrides):
    cfg = {
        "network": {
            "default_cost_per_unit": 0.50,
            "default_transfer_lt_days": 3,
            "default_min_transfer_qty": 10,
            "default_batch_size": 1,
        },
        "costs": {},
        "constraints": {"max_source_drawdown_pct": 0.30},
    }
    cfg.update(overrides)
    return cfg


def test_build_transfer_candidates_with_lane():
    """Matching lane exists — candidate should use lane's cost and LT."""
    lanes = [{
        "lane_id": "LANE1", "source_loc": "LOC_A", "dest_loc": "LOC_B",
        "transfer_mode": "truck", "cost_per_unit": 1.25,
        "handling_cost": 0, "freight_cost": 0, "receiving_cost": 0,
        "fixed_cost_per_shipment": 10.0, "transfer_lt_days": 5,
        "min_transfer_qty": 5, "max_transfer_qty": 500,
        "batch_size": 1, "max_shipments_per_week": 5,
        "max_receiving_units_per_period": None,
    }]
    result = build_transfer_candidates(_make_imbalances(), lanes, _base_config())
    assert len(result) == 1
    c = result[0]
    assert c["lane_id"] == "LANE1"
    assert c["cost_per_unit"] == 1.25
    assert c["transfer_lt_days"] == 5
    assert c["fixed_cost"] == 10.0


def test_build_transfer_candidates_no_lane():
    """No matching lane — should use default config values."""
    result = build_transfer_candidates(_make_imbalances(), [], _base_config())
    assert len(result) == 1
    c = result[0]
    assert c["lane_id"] is None
    assert c["cost_per_unit"] == 0.50
    assert c["transfer_lt_days"] == 3
    assert c["fixed_cost"] == 0


def test_build_transfer_candidates_batch_rounding():
    """batch_size=10, raw_qty=50 (min of excess=100, shortage=50, drawdown=60).
    floor(50/10)*10 = 50 → no rounding needed. Use raw_qty that forces rounding."""
    # Excess = 200-100=100, shortage = 100-65=35, drawdown = 200*0.30=60
    # raw_qty = min(100, 35, 60) = 35, floor(35/10)*10 = 30
    imbalances = {
        "ITEM1": {
            "excess": [{
                "item_id": "ITEM1", "loc": "LOC_A",
                "on_hand": 200, "ss_target": 100, "dos": 20,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
            "shortage": [{
                "item_id": "ITEM1", "loc": "LOC_B",
                "on_hand": 65, "ss_target": 100, "dos": 6.5,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
        }
    }
    lanes = [{
        "lane_id": "L1", "source_loc": "LOC_A", "dest_loc": "LOC_B",
        "transfer_mode": "truck", "cost_per_unit": 1.0,
        "handling_cost": 0, "freight_cost": 0, "receiving_cost": 0,
        "fixed_cost_per_shipment": 0, "transfer_lt_days": 3,
        "min_transfer_qty": 5, "max_transfer_qty": 500,
        "batch_size": 10, "max_shipments_per_week": 5,
        "max_receiving_units_per_period": None,
    }]
    result = build_transfer_candidates(imbalances, lanes, _base_config())
    assert len(result) == 1
    assert result[0]["recommended_qty"] == 30  # floor(35/10)*10


def test_build_transfer_candidates_below_min():
    """raw_qty after rounding < min_transfer_qty → candidate dropped."""
    # Excess = 200-100=100, shortage = 100-97=3, drawdown = 200*0.30=60
    # raw_qty = min(100, 3, 60) = 3, batch=1, 3 < min_qty=10 → skip
    imbalances = {
        "ITEM1": {
            "excess": [{
                "item_id": "ITEM1", "loc": "LOC_A",
                "on_hand": 200, "ss_target": 100, "dos": 20,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
            "shortage": [{
                "item_id": "ITEM1", "loc": "LOC_B",
                "on_hand": 97, "ss_target": 100, "dos": 9.7,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
        }
    }
    result = build_transfer_candidates(imbalances, [], _base_config())
    assert result == []


def test_build_transfer_candidates_drawdown_limit():
    """max_drawdown=0.30 limits qty to 30% of source on_hand."""
    # Excess = 200-100=100, shortage = 100-0=100, drawdown = 200*0.30=60
    # raw_qty = min(100, 100, 60) = 60
    imbalances = {
        "ITEM1": {
            "excess": [{
                "item_id": "ITEM1", "loc": "LOC_A",
                "on_hand": 200, "ss_target": 100, "dos": 20,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
            "shortage": [{
                "item_id": "ITEM1", "loc": "LOC_B",
                "on_hand": 0, "ss_target": 100, "dos": 0,
                "daily_sales": 10, "abc_vol": "A", "reorder_point": 80,
            }],
        }
    }
    result = build_transfer_candidates(imbalances, [], _base_config())
    assert len(result) == 1
    assert result[0]["recommended_qty"] == 60  # drawdown = 200 * 0.30


# ---------------------------------------------------------------------------
# compute_financials
# ---------------------------------------------------------------------------

def test_compute_financials():
    """Verify transfer_cost, stockout_cost_avoided, net_benefit, roi formulas."""
    config = {
        "costs": {
            "stockout_cost_multiplier": 5.0,
            "carrying_cost_annual_pct": 0.25,
        },
        "optimization": {"horizon_weeks": 4},
    }
    candidates = [{
        "recommended_qty": 50,
        "cost_per_unit": 1.0,
        "fixed_cost": 10.0,
        "source_excess_qty": 100,
        "dest_shortage_qty": 50,
        "dest_ss_target": 100,
    }]

    result = compute_financials(candidates, config)
    c = result[0]

    # transfer_cost = 50 * 1.0 + 10.0 = 60.0
    assert c["transfer_cost"] == 60.0

    # carrying_cost_saved = 100 * 1.0 * 0.25 * (28/365)
    horizon_days = 4 * 7  # 28
    expected_carrying = 100 * 1.0 * 0.25 * (horizon_days / 365)
    assert abs(c["carrying_cost_saved"] - expected_carrying) < 0.001

    # shortage_severity = min(1.0, 50/100) = 0.5
    # stockout_cost_avoided = min(50, 50) * 1.0 * 5.0 * 0.5 = 125.0
    assert c["stockout_cost_avoided"] == 125.0

    # net_benefit = 125.0 + carrying - 60.0
    expected_net = 125.0 + expected_carrying - 60.0
    assert abs(c["net_benefit"] - expected_net) < 0.001

    # roi = net_benefit / transfer_cost
    assert abs(c["roi"] - expected_net / 60.0) < 0.001


# ---------------------------------------------------------------------------
# assign_urgency
# ---------------------------------------------------------------------------

def test_assign_urgency_critical():
    """dest_dos=2, abc='A' → critical."""
    candidates = [{"dest_dos": 2, "abc_class": "A"}]
    result = assign_urgency(candidates)
    assert result[0]["urgency"] == "critical"


def test_assign_urgency_high():
    """dest_dos=5 (< 7) → high."""
    candidates = [{"dest_dos": 5, "abc_class": "B"}]
    result = assign_urgency(candidates)
    assert result[0]["urgency"] == "high"


def test_assign_urgency_medium():
    """dest_dos=10 (< 14) → medium."""
    candidates = [{"dest_dos": 10, "abc_class": "C"}]
    result = assign_urgency(candidates)
    assert result[0]["urgency"] == "medium"


def test_assign_urgency_low():
    """dest_dos=30 (>= 14) → low."""
    candidates = [{"dest_dos": 30, "abc_class": "A"}]
    result = assign_urgency(candidates)
    assert result[0]["urgency"] == "low"


# ---------------------------------------------------------------------------
# greedy_solver
# ---------------------------------------------------------------------------

def _make_candidate(
    item="ITEM1", src="LOC_A", dst="LOC_B",
    qty=50, cost_per_unit=1.0, fixed_cost=0,
    net_benefit=100, roi=2.0, urgency="high",
    source_excess=100, dest_shortage=50, dest_ss=100,
    dest_dos=5, abc="A",
):
    return {
        "item_id": item, "source_loc": src, "dest_loc": dst,
        "recommended_qty": qty, "cost_per_unit": cost_per_unit,
        "fixed_cost": fixed_cost, "transfer_cost": qty * cost_per_unit + fixed_cost,
        "net_benefit": net_benefit, "roi": roi, "urgency": urgency,
        "source_excess_qty": source_excess, "dest_shortage_qty": dest_shortage,
        "dest_ss_target": dest_ss, "dest_dos": dest_dos, "abc_class": abc,
        "source_on_hand": 200, "source_dos": 20, "source_ss_target": 100,
        "dest_on_hand": 50, "dest_reorder_point": 80,
        "stockout_cost_avoided": 125, "carrying_cost_saved": 2,
    }


def test_greedy_solver_basic():
    """2 candidates with positive benefit — both should be selected."""
    config = {"triggers": {"min_benefit_per_transfer": 5.0}}
    c1 = _make_candidate(dst="LOC_B", net_benefit=100, roi=2.0)
    c2 = _make_candidate(dst="LOC_C", net_benefit=50, roi=1.0,
                         dest_shortage=60, dest_ss=120)
    result = greedy_solver([c1, c2], config)
    assert len(result) == 2


def test_greedy_solver_negative_roi_skipped():
    """Candidate with net_benefit < min_benefit_per_transfer is filtered."""
    config = {"triggers": {"min_benefit_per_transfer": 5.0}}
    c1 = _make_candidate(net_benefit=100, roi=2.0)
    c2 = _make_candidate(dst="LOC_C", net_benefit=2.0, roi=0.01,
                         dest_shortage=60, dest_ss=120)
    result = greedy_solver([c1, c2], config)
    # c2 net_benefit (2.0) < min_benefit (5.0) → filtered
    assert len(result) == 1
    assert result[0]["dest_loc"] == "LOC_B"


# ---------------------------------------------------------------------------
# compute_network_balance
# ---------------------------------------------------------------------------

def test_compute_network_balance():
    """2 items, each with 2 locs with different DOS → avg CV > 0."""
    state = {
        ("ITEM1", "LOC_A"): {"dos": 30, "item_id": "ITEM1", "loc": "LOC_A",
                              "on_hand": 300, "daily_sales": 10, "ss_target": 100,
                              "abc_vol": "A", "reorder_point": 80},
        ("ITEM1", "LOC_B"): {"dos": 10, "item_id": "ITEM1", "loc": "LOC_B",
                              "on_hand": 100, "daily_sales": 10, "ss_target": 100,
                              "abc_vol": "A", "reorder_point": 80},
        ("ITEM2", "LOC_A"): {"dos": 20, "item_id": "ITEM2", "loc": "LOC_A",
                              "on_hand": 200, "daily_sales": 10, "ss_target": 100,
                              "abc_vol": "B", "reorder_point": 80},
        ("ITEM2", "LOC_B"): {"dos": 5, "item_id": "ITEM2", "loc": "LOC_B",
                              "on_hand": 50, "daily_sales": 10, "ss_target": 100,
                              "abc_vol": "B", "reorder_point": 80},
    }
    cv = compute_network_balance(state)
    assert cv > 0

    # Verify manually:
    # ITEM1: mean=20, var=((30-20)^2+(10-20)^2)/2=100, std=10, cv=10/20=0.5
    # ITEM2: mean=12.5, var=((20-12.5)^2+(5-12.5)^2)/2=56.25, std=7.5, cv=7.5/12.5=0.6
    # avg CV = (0.5 + 0.6) / 2 = 0.55
    assert abs(cv - 0.55) < 0.001
