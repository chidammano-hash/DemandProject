"""Unit tests for scripts/compute_safety_stock.py — IPfeature3.

Tests the pure-Python safety stock computation functions in isolation
(no DB required). All formulas follow the spec in docs/design-specs/IPfeature3.md.

Key formula reference:
  SS_demand   = Z * sqrt(LT_mean_days * sigma_D_daily^2)
  SS_lt       = Z * avg_daily_demand * lt_std_days
  SS_combined = Z * sqrt(LT_mean_days * sigma_D_daily^2 + avg_daily_demand^2 * lt_std_days^2)
  ROP         = avg_daily_demand * LT_mean_days + SS_combined
  ss_coverage = current_qty / SS_combined
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scripts.compute_safety_stock import (
        compute_ss_demand,
        compute_ss_lt,
        compute_ss_combined,
        compute_reorder_point,
        compute_ss_coverage,
        get_z_score,
        apply_guard_rails,
    )
except ImportError:
    pytest.skip("compute_safety_stock not yet implemented", allow_module_level=True)


# ---------------------------------------------------------------------------
# Z-score lookup table (used across tests)
# ---------------------------------------------------------------------------

Z_TABLE = {
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.97: 1.881,
    0.98: 2.054,
    0.99: 2.326,
}


# ---------------------------------------------------------------------------
# get_z_score
# ---------------------------------------------------------------------------

def test_z_score_90():
    """90% service level → Z = 1.282."""
    z = get_z_score(0.90, Z_TABLE)
    assert abs(z - 1.282) < 1e-6


def test_z_score_95():
    """95% service level → Z = 1.645."""
    z = get_z_score(0.95, Z_TABLE)
    assert abs(z - 1.645) < 1e-6


def test_z_score_98():
    """98% service level → Z = 2.054."""
    z = get_z_score(0.98, Z_TABLE)
    assert abs(z - 2.054) < 1e-6


def test_z_score_99():
    """99% service level → Z = 2.326."""
    z = get_z_score(0.99, Z_TABLE)
    assert abs(z - 2.326) < 1e-6


def test_z_score_85():
    """85% service level → Z = 1.036."""
    z = get_z_score(0.85, Z_TABLE)
    assert abs(z - 1.036) < 1e-6


def test_z_score_97():
    """97% service level → Z = 1.881."""
    z = get_z_score(0.97, Z_TABLE)
    assert abs(z - 1.881) < 1e-6


def test_z_score_returns_float():
    z = get_z_score(0.95, Z_TABLE)
    assert isinstance(z, float)


def test_z_score_exact_key_lookup():
    """All keys in the table can be looked up exactly."""
    for sl, expected_z in Z_TABLE.items():
        z = get_z_score(sl, Z_TABLE)
        assert abs(z - expected_z) < 1e-6, f"Failed for SL={sl}"


# ---------------------------------------------------------------------------
# compute_ss_demand
# ---------------------------------------------------------------------------

def test_ss_demand_known_result():
    """Verified: Z=1.645, sigma_D_daily=2.0, LT_mean=14.
    SS_demand = 1.645 * sqrt(14 * 4) = 1.645 * 7.483... ≈ 12.31
    """
    result = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=14.0)
    assert result is not None
    assert abs(result - 12.31) < 0.05


def test_ss_demand_formula_uses_sqrt():
    """SS_demand = Z * sqrt(LT_mean * sigma^2). Increasing LT by 4x → doubles SS."""
    ss1 = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=10.0)
    ss4 = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=40.0)
    assert abs(ss4 / ss1 - 2.0) < 1e-6


def test_ss_demand_scales_with_z():
    """Doubling Z doubles SS_demand."""
    ss1 = compute_ss_demand(z_score=1.0, sigma_demand=2.0, lt_mean_days=14.0)
    ss2 = compute_ss_demand(z_score=2.0, sigma_demand=2.0, lt_mean_days=14.0)
    assert abs(ss2 / ss1 - 2.0) < 1e-6


def test_ss_demand_zero_sigma_returns_zero():
    """Zero demand variability → zero demand SS component."""
    result = compute_ss_demand(z_score=1.645, sigma_demand=0.0, lt_mean_days=14.0)
    assert result == 0.0 or abs(result) < 1e-9


def test_ss_demand_positive():
    """SS_demand is always non-negative for valid inputs."""
    result = compute_ss_demand(z_score=1.645, sigma_demand=3.0, lt_mean_days=7.0)
    assert result >= 0.0


def test_ss_demand_zero_lt_returns_zero():
    """Zero lead time mean → zero SS_demand."""
    result = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=0.0)
    assert result == 0.0 or abs(result) < 1e-9


# ---------------------------------------------------------------------------
# compute_ss_lt
# ---------------------------------------------------------------------------

def test_ss_lt_known_result():
    """Verified: Z=1.645, avg_daily=10, lt_std=3.0.
    SS_lt = 1.645 * 10 * 3.0 = 49.35
    """
    result = compute_ss_lt(z_score=1.645, avg_daily_demand=10.0, lt_std_days=3.0)
    assert result is not None
    assert abs(result - 49.35) < 0.01


def test_ss_lt_zero_lt_std_returns_zero():
    """Zero LT variability → zero LT SS component."""
    result = compute_ss_lt(z_score=1.645, avg_daily_demand=10.0, lt_std_days=0.0)
    assert result == 0.0 or abs(result) < 1e-9


def test_ss_lt_zero_demand_returns_zero():
    """Zero average demand → zero LT SS component (no demand, no risk)."""
    result = compute_ss_lt(z_score=1.645, avg_daily_demand=0.0, lt_std_days=3.0)
    assert result == 0.0 or abs(result) < 1e-9


def test_ss_lt_scales_linearly_with_demand():
    """SS_lt ∝ avg_daily_demand — doubling demand doubles SS_lt."""
    ss1 = compute_ss_lt(z_score=1.645, avg_daily_demand=5.0, lt_std_days=3.0)
    ss2 = compute_ss_lt(z_score=1.645, avg_daily_demand=10.0, lt_std_days=3.0)
    assert abs(ss2 / ss1 - 2.0) < 1e-6


def test_ss_lt_positive_for_valid_inputs():
    """SS_lt is always non-negative for valid inputs."""
    result = compute_ss_lt(z_score=2.054, avg_daily_demand=8.0, lt_std_days=2.5)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# compute_ss_combined
# ---------------------------------------------------------------------------

def test_ss_combined_known_result():
    """Verified: Z=1.645, LT_mean=14, sigma_D_daily=2.0, avg_daily=10, lt_std=3.0.
    SS_combined = 1.645 * sqrt(14*4 + 100*9) = 1.645 * sqrt(956) ≈ 50.87
    """
    # sqrt(14*4 + 100*9) = sqrt(56 + 900) = sqrt(956) ≈ 30.919
    # 1.645 * 30.919 ≈ 50.86
    result = compute_ss_combined(
        z_score=1.645,
        sigma_demand=2.0,
        lt_mean_days=14.0,
        avg_daily_demand=10.0,
        lt_std_days=3.0,
    )
    assert result is not None
    assert abs(result - 50.87) < 0.10


def test_ss_combined_geq_both_components():
    """Combined SS via uncorrelated formula: SS_combined >= max(SS_demand, SS_lt)."""
    ss_d = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=14.0)
    ss_lt = compute_ss_lt(z_score=1.645, avg_daily_demand=10.0, lt_std_days=3.0)
    ss_c = compute_ss_combined(
        z_score=1.645,
        sigma_demand=2.0,
        lt_mean_days=14.0,
        avg_daily_demand=10.0,
        lt_std_days=3.0,
    )
    assert ss_c >= max(ss_d, ss_lt) - 1e-9


def test_ss_combined_zero_lt_std_equals_demand_component():
    """When lt_std=0: SS_combined = SS_demand (only demand uncertainty)."""
    ss_c = compute_ss_combined(
        z_score=1.645,
        sigma_demand=2.0,
        lt_mean_days=14.0,
        avg_daily_demand=10.0,
        lt_std_days=0.0,
    )
    ss_d = compute_ss_demand(z_score=1.645, sigma_demand=2.0, lt_mean_days=14.0)
    assert abs(ss_c - ss_d) < 1e-6


def test_ss_combined_zero_demand_std_equals_lt_component():
    """When sigma_demand=0: SS_combined should equal SS_lt component."""
    ss_c = compute_ss_combined(
        z_score=1.645,
        sigma_demand=0.0,
        lt_mean_days=14.0,
        avg_daily_demand=10.0,
        lt_std_days=3.0,
    )
    ss_lt = compute_ss_lt(z_score=1.645, avg_daily_demand=10.0, lt_std_days=3.0)
    assert abs(ss_c - ss_lt) < 1e-6


def test_ss_combined_zero_demand_and_zero_std_returns_zero():
    """Zero mean demand + zero demand std → SS_combined = 0."""
    result = compute_ss_combined(
        z_score=1.645,
        sigma_demand=0.0,
        lt_mean_days=14.0,
        avg_daily_demand=0.0,
        lt_std_days=0.0,
    )
    assert result == 0.0 or abs(result) < 1e-9


def test_ss_combined_non_negative():
    """SS_combined is always non-negative."""
    result = compute_ss_combined(
        z_score=2.054,
        sigma_demand=5.0,
        lt_mean_days=21.0,
        avg_daily_demand=15.0,
        lt_std_days=4.0,
    )
    assert result >= 0.0


# ---------------------------------------------------------------------------
# compute_reorder_point
# ---------------------------------------------------------------------------

def test_rop_known_result():
    """ROP = avg_daily_demand * lt_mean_days + ss_combined.
    Example: avg_daily=10, LT_mean=14, ss_combined=50.87 → ROP ≈ 190.87
    """
    result = compute_reorder_point(
        ss_combined=50.87, avg_daily_demand=10.0, lt_mean_days=14.0
    )
    assert abs(result - 190.87) < 0.01


def test_rop_equals_lt_demand_plus_ss():
    """ROP must always equal lt_demand_component + ss_combined."""
    avg_daily = 8.0
    lt_mean = 20.0
    ss_combined = 75.0
    result = compute_reorder_point(ss_combined, avg_daily, lt_mean)
    expected = avg_daily * lt_mean + ss_combined
    assert abs(result - expected) < 1e-6


def test_rop_with_zero_ss():
    """When SS=0, ROP = avg_daily * lt_mean."""
    result = compute_reorder_point(
        ss_combined=0.0, avg_daily_demand=10.0, lt_mean_days=14.0
    )
    assert abs(result - 140.0) < 1e-6


def test_rop_with_zero_demand():
    """Zero demand → ROP = 0 + ss_combined = ss_combined."""
    result = compute_reorder_point(
        ss_combined=50.0, avg_daily_demand=0.0, lt_mean_days=14.0
    )
    assert abs(result - 50.0) < 1e-6


def test_rop_positive_for_valid_inputs():
    """ROP is always non-negative for valid inputs."""
    result = compute_reorder_point(
        ss_combined=30.0, avg_daily_demand=5.0, lt_mean_days=10.0
    )
    assert result >= 0.0


# ---------------------------------------------------------------------------
# compute_ss_coverage
# ---------------------------------------------------------------------------

def test_ss_coverage_below_target():
    """current_qty=50, ss=80 → ss_gap=-30, is_below_ss=True, coverage=0.625."""
    result = compute_ss_coverage(current_on_hand=50.0, ss_combined=80.0)
    assert result is not None
    assert abs(result - 0.625) < 1e-6


def test_ss_coverage_above_target():
    """current_qty=100, ss=80 → coverage=1.25."""
    result = compute_ss_coverage(current_on_hand=100.0, ss_combined=80.0)
    assert abs(result - 1.25) < 1e-6


def test_ss_coverage_exact_match():
    """On-hand exactly equals SS → coverage=1.0."""
    result = compute_ss_coverage(current_on_hand=80.0, ss_combined=80.0)
    assert abs(result - 1.0) < 1e-6


def test_ss_coverage_zero_ss_no_crash():
    """When SS=0, coverage should not raise ZeroDivisionError.
    Returns None or 0 or very large number — just must not crash.
    """
    try:
        result = compute_ss_coverage(current_on_hand=50.0, ss_combined=0.0)
        # If it returns a value, it should be None or a non-negative number
        if result is not None:
            assert result >= 0.0 or math.isinf(result)
    except ZeroDivisionError:
        pytest.fail("compute_ss_coverage raised ZeroDivisionError for ss_combined=0")


def test_ss_coverage_zero_on_hand():
    """Zero on-hand → coverage=0.0."""
    result = compute_ss_coverage(current_on_hand=0.0, ss_combined=80.0)
    assert result == 0.0 or abs(result) < 1e-9


def test_ss_coverage_returns_float():
    result = compute_ss_coverage(current_on_hand=100.0, ss_combined=80.0)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# apply_guard_rails
# ---------------------------------------------------------------------------

def test_guard_rails_below_min_is_clamped():
    """If SS < min_ss_days * avg_daily → SS clamped to min_ss_days * avg_daily.
    min_days=3, avg_daily=10 → min_ss = 30. If ss_days=1 → result should be 30.
    """
    avg_daily = 10.0
    result = apply_guard_rails(
        ss_days=1.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(result - 30.0) < 1e-6


def test_guard_rails_above_max_is_capped():
    """If SS > max_ss_days * avg_daily → SS capped at max_ss_days * avg_daily.
    max_days=120, avg_daily=10 → max_ss=1200. If ss_days=200 → result capped at 1200.
    """
    avg_daily = 10.0
    result = apply_guard_rails(
        ss_days=200.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(result - 1200.0) < 1e-6


def test_guard_rails_within_range_unchanged():
    """SS within [min, max] range passes through unchanged."""
    avg_daily = 10.0
    # ss_days=30 → SS=300 → within [30, 1200]
    result = apply_guard_rails(
        ss_days=30.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(result - 300.0) < 1e-6


def test_guard_rails_zero_demand_no_crash():
    """Zero demand should not raise ZeroDivisionError (min_ss = 0)."""
    try:
        result = apply_guard_rails(
            ss_days=0.0, avg_daily_demand=0.0, min_days=3, max_days=120
        )
        # If zero demand, guard rail floor produces 0 — acceptable
        assert result >= 0.0
    except ZeroDivisionError:
        pytest.fail("apply_guard_rails raised ZeroDivisionError for zero demand")


def test_guard_rails_result_is_non_negative():
    """Guard-railed SS is always non-negative."""
    avg_daily = 5.0
    result = apply_guard_rails(
        ss_days=0.5, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert result >= 0.0


def test_guard_rails_exact_min_boundary():
    """SS at exactly min_ss_days → no clamping."""
    avg_daily = 10.0
    # min_days=3, ss = 3*10=30 → exactly at boundary
    result = apply_guard_rails(
        ss_days=3.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(result - 30.0) < 1e-6


def test_guard_rails_exact_max_boundary():
    """SS at exactly max_ss_days → no capping."""
    avg_daily = 10.0
    # max_days=120, ss = 120*10=1200 → exactly at boundary
    result = apply_guard_rails(
        ss_days=120.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(result - 1200.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration / cross-formula checks
# ---------------------------------------------------------------------------

def test_full_formula_pipeline_verified_case():
    """End-to-end: sigma_D_daily=2.0, LT_mean=14, lt_std=3.0, Z=1.645, avg_daily=10.
    Expected:
      SS_demand = 1.645 * sqrt(14 * 4) = 12.31
      SS_lt     = 1.645 * 10 * 3 = 49.35
      SS_comb   = 1.645 * sqrt(56 + 900) ≈ 50.87
      ROP       = 10 * 14 + 50.87 = 190.87
    """
    z = 1.645
    sigma_d = 2.0
    lt_mean = 14.0
    lt_std = 3.0
    avg_daily = 10.0

    ss_d = compute_ss_demand(z, sigma_d, lt_mean)
    ss_lt = compute_ss_lt(z, avg_daily, lt_std)
    ss_c = compute_ss_combined(z, sigma_d, lt_mean, avg_daily, lt_std)
    rop = compute_reorder_point(ss_c, avg_daily, lt_mean)

    assert abs(ss_d - 12.31) < 0.05
    assert abs(ss_lt - 49.35) < 0.01
    assert abs(ss_c - 50.87) < 0.10
    assert abs(rop - 190.87) < 0.10


def test_ss_coverage_gap_relationship():
    """Verify: if on_hand < ss_combined → coverage < 1.0 (below SS target)."""
    on_hand = 50.0
    ss = 80.0
    coverage = compute_ss_coverage(on_hand, ss)
    assert coverage < 1.0
    # gap = on_hand - ss = -30 (negative = shortfall)
    gap = on_hand - ss
    assert gap < 0.0


def test_ss_coverage_above_target_relationship():
    """If on_hand > ss_combined → coverage > 1.0."""
    on_hand = 100.0
    ss = 80.0
    coverage = compute_ss_coverage(on_hand, ss)
    assert coverage > 1.0


def test_ss_combined_always_geq_components():
    """Combined SS (via Pythagorean formula) always >= both individual components."""
    test_cases = [
        (1.645, 2.0, 14.0, 10.0, 3.0),
        (2.054, 5.0, 21.0, 20.0, 4.0),
        (1.282, 1.0, 7.0, 3.0, 2.0),
        (2.326, 10.0, 30.0, 50.0, 5.0),
    ]
    for z, sigma_d, lt_mean, avg_daily, lt_std in test_cases:
        ss_d = compute_ss_demand(z, sigma_d, lt_mean)
        ss_lt = compute_ss_lt(z, avg_daily, lt_std)
        ss_c = compute_ss_combined(z, sigma_d, lt_mean, avg_daily, lt_std)
        assert ss_c >= ss_d - 1e-9, f"ss_combined < ss_demand for case {(z, sigma_d, lt_mean, avg_daily, lt_std)}"
        assert ss_c >= ss_lt - 1e-9, f"ss_combined < ss_lt for case {(z, sigma_d, lt_mean, avg_daily, lt_std)}"


def test_zero_demand_item_has_zero_ss():
    """Zero mean AND zero std demand → SS_combined = 0 (no safety stock needed)."""
    ss_c = compute_ss_combined(
        z_score=1.645,
        sigma_demand=0.0,
        lt_mean_days=14.0,
        avg_daily_demand=0.0,
        lt_std_days=3.0,
    )
    assert ss_c == 0.0 or abs(ss_c) < 1e-9
