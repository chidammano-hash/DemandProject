"""Unit tests for common/inventory/safety_stock.py.

These tests confirm the pure safety-stock formulas are importable and testable
without pulling in the CLI script (`scripts/inventory/compute_safety_stock.py`).
They intentionally cover a small, representative slice of behaviour — the
exhaustive formula matrix lives in `tests/unit/test_safety_stock.py`.
"""
from __future__ import annotations

import math

from common.inventory.safety_stock import (
    classify_xyz,
    compute_position_metrics,
    compute_reorder_point_periodic,
    compute_ss_components,
    compute_ss_demand,
    get_z_score,
)

Z_TABLE = {"0.90": 1.282, "0.95": 1.645, "0.98": 2.054}


def test_get_z_score_exact_and_fallback():
    """Exact key returns its Z; unknown level snaps to the closest key."""
    assert abs(get_z_score(0.95, Z_TABLE) - 1.645) < 1e-9
    # 0.96 is closest to 0.95 → 1.645
    assert abs(get_z_score(0.96, Z_TABLE) - 1.645) < 1e-9
    # Empty table → hard 95% fallback
    assert abs(get_z_score(0.95, {}) - 1.645) < 1e-9


def test_compute_ss_demand_known_value():
    """SS_demand = Z * sqrt(LT * sigma^2): 1.645 * sqrt(14 * 4) ≈ 12.31."""
    result = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=14.0)
    assert abs(result - 1.645 * math.sqrt(14.0 * 4.0)) < 1e-9


def test_compute_ss_demand_zero_variance_is_zero():
    """Zero sigma → zero demand component."""
    assert compute_ss_demand(z_score=2.054, sigma_demand=0.0, lt_mean_days=14.0) == 0.0


def test_classify_xyz_boundaries():
    """Default thresholds: <0.3 → X, <0.8 → Y, else Z; None → None."""
    assert classify_xyz(0.1) == "X"
    assert classify_xyz(0.5) == "Y"
    assert classify_xyz(1.2) == "Z"
    assert classify_xyz(None) is None


def test_compute_ss_components_combined_method():
    """Components dict exposes daily stats and labels lead-time-variable items 'combined'."""
    comps = compute_ss_components(
        z=1.645,
        demand_mean_monthly=304.4,  # → 10 units/day
        demand_std_monthly=10.0,
        lt_mean_days=14.0,
        lt_std_days=3.0,
    )
    assert abs(comps["avg_daily_demand"] - 10.0) < 1e-9
    assert comps["ss_method"] == "combined"
    # Combined SS is at least as large as the demand-only component.
    assert comps["ss_combined"] >= comps["ss_demand_only"] - 1e-9


def test_compute_position_metrics_below_ss():
    """On-hand under SS yields a shortfall gap, coverage < 1, is_below_ss True."""
    pos = compute_position_metrics(
        ss_combined=80.0,
        avg_daily_demand=10.0,
        lt_mean_days=14.0,
        current_qty_on_hand=50.0,
    )
    assert abs(pos["reorder_point"] - (10.0 * 14.0 + 80.0)) < 1e-9
    assert abs(pos["ss_coverage"] - 0.625) < 1e-9
    assert abs(pos["ss_gap"] - (-30.0)) < 1e-9
    assert pos["is_below_ss"] is True


def test_compute_reorder_point_periodic_adds_half_review_cycle():
    """Periodic ROP protects LT + R/2 of demand plus safety stock."""
    rop = compute_reorder_point_periodic(
        ss_combined=50.0,
        avg_daily_demand=10.0,
        lt_mean_days=14.0,
        review_cycle_days=7.0,
    )
    # 10 * (14 + 3.5) + 50 = 225
    assert abs(rop - 225.0) < 1e-9
