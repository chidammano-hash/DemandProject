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

from datetime import date

try:
    from scripts.compute_safety_stock import (
        apply_guard_rails,
        apply_seasonal_adjustment,
        classify_xyz,
        compute_reorder_point,
        compute_seasonal_factors,
        compute_ss_combined,
        compute_ss_coverage,
        compute_ss_demand,
        compute_ss_lt,
        detect_outliers,
        get_service_level,
        get_z_score,
    )
except ImportError:
    pytest.skip("compute_safety_stock not yet implemented", allow_module_level=True)


# ---------------------------------------------------------------------------
# Z-score lookup table (used across tests)
# ---------------------------------------------------------------------------

Z_TABLE = {
    "0.85": 1.036,
    "0.90": 1.282,
    "0.95": 1.645,
    "0.97": 1.881,
    "0.98": 2.054,
    "0.99": 2.326,
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
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_days=1.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(clamped - 30.0) < 1e-6
    assert was_clamped is True


def test_guard_rails_above_max_is_capped():
    """If SS > max_ss_days * avg_daily → SS capped at max_ss_days * avg_daily.
    max_days=120, avg_daily=10 → max_ss=1200. If ss_days=200 → result capped at 1200.
    """
    avg_daily = 10.0
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_days=200.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(clamped - 1200.0) < 1e-6
    assert was_clamped is True


def test_guard_rails_within_range_unchanged():
    """SS within [min, max] range passes through unchanged."""
    avg_daily = 10.0
    # ss_days=30 → SS=300 → within [30, 1200]
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_days=30.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(clamped - 300.0) < 1e-6
    assert was_clamped is False


def test_guard_rails_zero_demand_no_crash():
    """Zero demand should not raise ZeroDivisionError (min_ss = 0)."""
    try:
        clamped, _, _, _ = apply_guard_rails(
            ss_days=0.0, avg_daily_demand=0.0, min_days=3, max_days=120
        )
        # If zero demand, guard rail floor produces 0 — acceptable
        assert clamped >= 0.0
    except ZeroDivisionError:
        pytest.fail("apply_guard_rails raised ZeroDivisionError for zero demand")


def test_guard_rails_result_is_non_negative():
    """Guard-railed SS is always non-negative."""
    avg_daily = 5.0
    clamped, _, _, _ = apply_guard_rails(
        ss_days=0.5, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert clamped >= 0.0


def test_guard_rails_exact_min_boundary():
    """SS at exactly min_ss_days → no clamping."""
    avg_daily = 10.0
    # min_days=3, ss = 3*10=30 → exactly at boundary
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_days=3.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(clamped - 30.0) < 1e-6
    assert was_clamped is False


def test_guard_rails_exact_max_boundary():
    """SS at exactly max_ss_days → no capping."""
    avg_daily = 10.0
    # max_days=120, ss = 120*10=1200 → exactly at boundary
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_days=120.0, avg_daily_demand=avg_daily, min_days=3, max_days=120
    )
    assert abs(clamped - 1200.0) < 1e-6
    assert was_clamped is False


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


# ---------------------------------------------------------------------------
# classify_xyz
# ---------------------------------------------------------------------------

def test_classify_xyz_stable():
    """CV < 0.3 → X (stable demand)."""
    assert classify_xyz(0.1) == "X"
    assert classify_xyz(0.0) == "X"
    assert classify_xyz(0.29) == "X"


def test_classify_xyz_moderate():
    """0.3 <= CV < 0.8 → Y (moderate variability)."""
    assert classify_xyz(0.3) == "Y"
    assert classify_xyz(0.5) == "Y"
    assert classify_xyz(0.79) == "Y"


def test_classify_xyz_volatile():
    """CV >= 0.8 → Z (volatile demand)."""
    assert classify_xyz(0.8) == "Z"
    assert classify_xyz(1.5) == "Z"
    assert classify_xyz(10.0) == "Z"


def test_classify_xyz_none_cv():
    """None demand_cv → None (unknown XYZ class)."""
    assert classify_xyz(None) is None


def test_classify_xyz_custom_thresholds():
    """Custom thresholds should override defaults."""
    thresholds = {"x_max": 0.2, "y_max": 0.6}
    assert classify_xyz(0.15, thresholds) == "X"
    assert classify_xyz(0.25, thresholds) == "Y"
    assert classify_xyz(0.65, thresholds) == "Z"


# ---------------------------------------------------------------------------
# get_service_level — ABC x XYZ matrix
# ---------------------------------------------------------------------------

# ABC-only service levels used as fallback
SERVICE_LEVELS = {"A": 0.98, "B": 0.95, "C": 0.90, "default": 0.95}

# Full 9-cell ABC x XYZ matrix
SERVICE_LEVEL_MATRIX = {
    "AX": 0.99, "AY": 0.98, "AZ": 0.97,
    "BX": 0.97, "BY": 0.95, "BZ": 0.93,
    "CX": 0.93, "CY": 0.90, "CZ": 0.85,
}


def test_service_level_matrix_ax():
    """AX segment (high value, stable) → 0.99."""
    sl, xyz, segment, reason = get_service_level(
        "A", SERVICE_LEVELS, demand_cv=0.1,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.99) < 1e-6
    assert xyz == "X"
    assert segment == "AX"
    assert reason is None


def test_service_level_matrix_bz():
    """BZ segment (medium value, volatile) → 0.93."""
    sl, xyz, segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=1.2,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.93) < 1e-6
    assert xyz == "Z"
    assert segment == "BZ"
    assert reason is None


def test_service_level_matrix_cy():
    """CY segment (low value, moderate) → 0.90."""
    sl, xyz, segment, reason = get_service_level(
        "C", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.90) < 1e-6
    assert xyz == "Y"
    assert segment == "CY"
    assert reason is None


def test_service_level_matrix_cz():
    """CZ segment (low value, volatile) → 0.85."""
    sl, xyz, segment, reason = get_service_level(
        "C", SERVICE_LEVELS, demand_cv=0.9,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.85) < 1e-6
    assert xyz == "Z"
    assert segment == "CZ"
    assert reason is None


def test_service_level_fallback_no_cv():
    """When demand_cv is None, falls back to ABC-only lookup."""
    sl, xyz, segment, reason = get_service_level(
        "A", SERVICE_LEVELS, demand_cv=None,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.98) < 1e-6  # ABC fallback for A
    assert xyz is None
    assert segment is None
    assert reason is None


def test_service_level_fallback_no_matrix():
    """When service_level_matrix is None, uses ABC-only lookup."""
    sl, xyz, segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=None,
    )
    assert abs(sl - 0.95) < 1e-6  # ABC fallback for B
    assert xyz == "Y"
    assert segment == "BY"  # segment is still computed for informational purposes
    assert reason is None


def test_service_level_fallback_no_abc():
    """When abc_vol is None, falls back to default service level."""
    sl, xyz, segment, reason = get_service_level(
        None, SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.95) < 1e-6  # default fallback
    assert xyz == "Y"
    assert segment is None  # can't form segment without abc
    assert reason is None


def test_service_level_backward_compat_abc_only():
    """Original 2-arg call still works for backward compatibility."""
    sl, xyz, segment, reason = get_service_level("A", SERVICE_LEVELS)
    assert abs(sl - 0.98) < 1e-6
    assert xyz is None
    assert segment is None
    assert reason is None


def test_service_level_all_nine_cells():
    """All 9 ABC x XYZ cells return correct service levels."""
    expected = {
        ("A", 0.1): (0.99, "X", "AX"),
        ("A", 0.5): (0.98, "Y", "AY"),
        ("A", 0.9): (0.97, "Z", "AZ"),
        ("B", 0.1): (0.97, "X", "BX"),
        ("B", 0.5): (0.95, "Y", "BY"),
        ("B", 0.9): (0.93, "Z", "BZ"),
        ("C", 0.1): (0.93, "X", "CX"),
        ("C", 0.5): (0.90, "Y", "CY"),
        ("C", 0.9): (0.85, "Z", "CZ"),
    }
    for (abc, cv), (exp_sl, exp_xyz, exp_seg) in expected.items():
        sl, xyz, segment, reason = get_service_level(
            abc, SERVICE_LEVELS, demand_cv=cv,
            service_level_matrix=SERVICE_LEVEL_MATRIX,
        )
        assert abs(sl - exp_sl) < 1e-6, f"Failed for {abc}{exp_xyz}: got {sl}, expected {exp_sl}"
        assert xyz == exp_xyz, f"Failed xyz for {abc}: got {xyz}"
        assert segment == exp_seg, f"Failed segment for {abc}{exp_xyz}: got {segment}"
        assert reason is None


# ---------------------------------------------------------------------------
# detect_outliers (MAD-based outlier detection)
# ---------------------------------------------------------------------------

def test_detect_outliers_no_outliers():
    """Uniform demand → zero outlier pct, not volatile."""
    history = [100.0] * 12
    pct, is_volatile = detect_outliers(history)
    assert pct == 0.0
    assert is_volatile is False


def test_detect_outliers_with_spike():
    """One extreme spike in 12 months of otherwise variable demand."""
    # Natural variability so MAD is non-zero, plus one huge outlier
    history = [90.0, 95.0, 105.0, 110.0, 100.0, 88.0,
               92.0, 108.0, 103.0, 97.0, 102.0, 10000.0]
    pct, is_volatile = detect_outliers(history)
    assert pct > 0.0  # at least 1 outlier detected
    # 1/12 = 8.3% < 20% → not volatile
    assert is_volatile is False


def test_detect_outliers_many_spikes_volatile():
    """Multiple extreme values in a naturally variable history → volatile."""
    # 8 normal months with some variance, 4 extreme months
    history = [90.0, 95.0, 105.0, 110.0, 100.0, 88.0,
               92.0, 108.0, 5000.0, 6000.0, 7000.0, 8000.0]
    pct, is_volatile = detect_outliers(history)
    assert pct > 0.20
    assert is_volatile is True


def test_detect_outliers_empty_history():
    """Empty demand history → no outliers, not volatile."""
    pct, is_volatile = detect_outliers([])
    assert pct == 0.0
    assert is_volatile is False


def test_detect_outliers_short_history():
    """Fewer than 3 months → no outliers, not volatile."""
    pct, is_volatile = detect_outliers([100.0, 200.0])
    assert pct == 0.0
    assert is_volatile is False


def test_detect_outliers_zero_mad():
    """All identical values → MAD=0 → no outliers."""
    history = [50.0] * 10
    pct, is_volatile = detect_outliers(history)
    assert pct == 0.0
    assert is_volatile is False


def test_detect_outliers_custom_threshold():
    """Lower threshold should flag more outliers."""
    history = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
               150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
    pct_strict, _ = detect_outliers(history, threshold=1.0)
    pct_loose, _ = detect_outliers(history, threshold=5.0)
    assert pct_strict >= pct_loose


def test_detect_outliers_returns_tuple():
    """Return type is (float, bool)."""
    pct, is_volatile = detect_outliers([100.0, 200.0, 300.0, 400.0])
    assert isinstance(pct, float)
    assert isinstance(is_volatile, bool)


# ---------------------------------------------------------------------------
# apply_guard_rails — ABC-specific bounds
# ---------------------------------------------------------------------------

GUARD_RAILS_CONFIG = {
    "A": {"min_ss_days": 5, "max_ss_days": 60},
    "B": {"min_ss_days": 3, "max_ss_days": 90},
    "C": {"min_ss_days": 1, "max_ss_days": 120},
    "default": {"min_ss_days": 3, "max_ss_days": 120},
    "zero_demand_min_units": 3,
}


def test_guard_rails_abc_a_tighter_max():
    """ABC class A should have tighter max (60 days).
    avg_daily=10, ss=700 (70 days) → capped at 60*10=600.
    """
    clamped, was_clamped, _, max_qty = apply_guard_rails(
        ss_combined=700.0, avg_daily_demand=10.0,
        abc_vol="A", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    assert abs(clamped - 600.0) < 1e-6
    assert was_clamped is True
    assert abs(max_qty - 600.0) < 1e-6


def test_guard_rails_abc_c_wider_bounds():
    """ABC class C allows up to 120 days.
    avg_daily=10, ss=700 (70 days) → within [1*10, 120*10] → no clamping.
    """
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_combined=700.0, avg_daily_demand=10.0,
        abc_vol="C", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    assert abs(clamped - 700.0) < 1e-6
    assert was_clamped is False


def test_guard_rails_abc_a_higher_min():
    """ABC class A has min 5 days.
    avg_daily=10, ss=20 (2 days) → clamped to 5*10=50.
    """
    clamped, was_clamped, min_qty, _ = apply_guard_rails(
        ss_combined=20.0, avg_daily_demand=10.0,
        abc_vol="A", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    assert abs(clamped - 50.0) < 1e-6
    assert was_clamped is True
    assert abs(min_qty - 50.0) < 1e-6


def test_guard_rails_abc_c_lower_min():
    """ABC class C has min 1 day.
    avg_daily=10, ss=5 (0.5 days) → clamped to 1*10=10.
    """
    clamped, was_clamped, min_qty, _ = apply_guard_rails(
        ss_combined=5.0, avg_daily_demand=10.0,
        abc_vol="C", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    assert abs(clamped - 10.0) < 1e-6
    assert was_clamped is True
    assert abs(min_qty - 10.0) < 1e-6


def test_guard_rails_unknown_abc_uses_default():
    """Unknown ABC class falls back to default bounds."""
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_combined=700.0, avg_daily_demand=10.0,
        abc_vol="D", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    # Default max = 120 days → 1200 → 700 is within range
    assert abs(clamped - 700.0) < 1e-6
    assert was_clamped is False


def test_guard_rails_zero_demand_with_config():
    """Zero demand + config → zero_demand_min_units floor applied."""
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_combined=0.0, avg_daily_demand=0.0,
        abc_vol="A", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    assert abs(clamped - 3.0) < 1e-6
    assert was_clamped is True


def test_guard_rails_backward_compat_no_config():
    """Without guard_rails_config, uses global min/max (backward compatible)."""
    clamped, was_clamped, _, _ = apply_guard_rails(
        ss_combined=20.0, avg_daily_demand=10.0,
        min_ss_days=3, max_ss_days=120,
    )
    # 20 < 3*10=30 → clamped to 30
    assert abs(clamped - 30.0) < 1e-6
    assert was_clamped is True


def test_guard_rails_returns_min_max_quantities():
    """Guard rails return the computed min/max quantities."""
    _, _, min_qty, max_qty = apply_guard_rails(
        ss_combined=500.0, avg_daily_demand=10.0,
        abc_vol="B", guard_rails_config=GUARD_RAILS_CONFIG,
    )
    # B: min=3 days, max=90 days → min_qty=30, max_qty=900
    assert abs(min_qty - 30.0) < 1e-6
    assert abs(max_qty - 900.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_seasonal_factors
# ---------------------------------------------------------------------------

def test_seasonal_factors_insufficient_data():
    """Fewer than min_history_months → all factors = 1.0 (no seasonality)."""
    history = [(date(2025, m, 1), 100.0) for m in range(1, 10)]  # 9 months
    factors = compute_seasonal_factors(history, min_history_months=12)
    assert all(abs(v - 1.0) < 1e-9 for v in factors.values())
    assert len(factors) == 12


def test_seasonal_factors_uniform_demand():
    """Uniform demand across all months → all factors = 1.0."""
    history = []
    for yr in (2024, 2025):
        for m in range(1, 13):
            history.append((date(yr, m, 1), 100.0))
    factors = compute_seasonal_factors(history, min_history_months=24)
    for m in range(1, 13):
        assert abs(factors[m] - 1.0) < 1e-9, f"Month {m}: expected 1.0, got {factors[m]}"


def test_seasonal_factors_peak_and_trough():
    """December demand is 2x average, June is 0.5x average.
    With uniform other months, December factor should be >1.0 and June <1.0.
    """
    history = []
    for yr in (2024, 2025):
        for m in range(1, 13):
            if m == 12:
                qty = 200.0  # peak
            elif m == 6:
                qty = 50.0   # trough
            else:
                qty = 100.0
            history.append((date(yr, m, 1), qty))
    factors = compute_seasonal_factors(history, min_history_months=24)
    assert factors[12] > 1.0, "December should be a peak month"
    assert factors[6] < 1.0, "June should be a trough month"
    # overall avg = (10*100 + 200 + 50) / 12 = 1250/12 ≈ 104.17
    # Dec factor = 200 / 104.17 ≈ 1.92
    # Jun factor = 50 / 104.17 ≈ 0.48
    assert abs(factors[12] - 200.0 / (1250.0 / 12)) < 0.01
    assert abs(factors[6] - 50.0 / (1250.0 / 12)) < 0.01


def test_seasonal_factors_zero_overall_demand():
    """All zero demand → all factors = 1.0 (avoid division by zero)."""
    history = [(date(2024, m, 1), 0.0) for m in range(1, 13)] * 2
    factors = compute_seasonal_factors(history, min_history_months=24)
    assert all(abs(v - 1.0) < 1e-9 for v in factors.values())


def test_seasonal_factors_returns_twelve_months():
    """Always returns exactly 12 month keys (1-12)."""
    history = [(date(2024, m, 1), float(m * 10)) for m in range(1, 13)] * 2
    factors = compute_seasonal_factors(history, min_history_months=24)
    assert set(factors.keys()) == set(range(1, 13))


def test_seasonal_factors_factors_sum_to_twelve():
    """Factors should approximately sum to 12 (since average factor = 1.0)."""
    history = []
    for yr in (2024, 2025):
        for m in range(1, 13):
            history.append((date(yr, m, 1), float(m * 10 + 50)))
    factors = compute_seasonal_factors(history, min_history_months=24)
    total = sum(factors.values())
    assert abs(total - 12.0) < 1e-6


def test_seasonal_factors_default_min_history():
    """Default min_history_months=24 requires 2 years of data."""
    # 23 months → insufficient
    history = [(date(2024, 1, 1), 100.0)] * 23
    factors = compute_seasonal_factors(history)
    assert all(abs(v - 1.0) < 1e-9 for v in factors.values())

    # 24 months → sufficient (but uniform, so still 1.0)
    history = [(date(2024, 1, 1), 100.0)] * 24
    factors = compute_seasonal_factors(history)
    # All mapped to month 1, so month 1 has avg=100, others have avg=0
    # overall_avg = 100/12 ≈ 8.33, month 1 factor = 100/8.33 = 12.0
    assert factors[1] > 1.0  # month 1 heavily overweighted


# ---------------------------------------------------------------------------
# apply_seasonal_adjustment
# ---------------------------------------------------------------------------

def test_seasonal_adjustment_peak_increases_ss():
    """Peak month (factor > 1.0) should increase safety stock."""
    ss_base = 100.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=1.5, dampening=1.0)
    # Full dampening: ss_adj = ss_base * sqrt(1.5) ≈ 122.47
    expected = 100.0 * math.sqrt(1.5)
    assert abs(ss_adj - expected) < 0.01


def test_seasonal_adjustment_trough_decreases_ss():
    """Trough month (factor < 1.0) should decrease safety stock."""
    ss_base = 100.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=0.5, dampening=1.0)
    # Full dampening: ss_adj = ss_base * sqrt(0.5) ≈ 70.71
    expected = 100.0 * math.sqrt(0.5)
    assert abs(ss_adj - expected) < 0.01
    assert ss_adj < ss_base


def test_seasonal_adjustment_factor_one_no_change():
    """Factor = 1.0 → no change regardless of dampening."""
    ss_base = 100.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=1.0, dampening=0.5)
    assert abs(ss_adj - ss_base) < 1e-9


def test_seasonal_adjustment_dampening_blends():
    """Dampening = 0.5 blends 50% seasonal + 50% base."""
    ss_base = 100.0
    factor = 2.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=factor, dampening=0.5)
    ss_scaled = ss_base * math.sqrt(factor)
    expected = 0.5 * ss_scaled + 0.5 * ss_base
    assert abs(ss_adj - expected) < 1e-9


def test_seasonal_adjustment_zero_dampening_no_seasonal():
    """Dampening = 0.0 means no seasonal adjustment (100% base)."""
    ss_base = 100.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=2.0, dampening=0.0)
    assert abs(ss_adj - ss_base) < 1e-9


def test_seasonal_adjustment_full_dampening():
    """Dampening = 1.0 means full seasonal adjustment (0% base)."""
    ss_base = 100.0
    factor = 1.5
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=factor, dampening=1.0)
    expected = ss_base * math.sqrt(factor)
    assert abs(ss_adj - expected) < 1e-9


def test_seasonal_adjustment_zero_factor_returns_base():
    """Factor <= 0 should return base SS unchanged (safety guard)."""
    ss_base = 100.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=0.0, dampening=0.5)
    assert abs(ss_adj - ss_base) < 1e-9

    ss_adj_neg = apply_seasonal_adjustment(ss_base, seasonal_factor=-1.0, dampening=0.5)
    assert abs(ss_adj_neg - ss_base) < 1e-9


def test_seasonal_adjustment_zero_ss_stays_zero():
    """Zero base SS stays zero regardless of seasonal factor."""
    ss_adj = apply_seasonal_adjustment(0.0, seasonal_factor=2.0, dampening=1.0)
    assert abs(ss_adj) < 1e-9


def test_seasonal_adjustment_sqrt_scaling():
    """SS scales with sqrt of demand because variance scales linearly.
    Factor 4.0 → sqrt(4) = 2.0 → SS doubles at full dampening.
    """
    ss_base = 50.0
    ss_adj = apply_seasonal_adjustment(ss_base, seasonal_factor=4.0, dampening=1.0)
    assert abs(ss_adj - 100.0) < 1e-9  # 50 * sqrt(4) = 100


# ---------------------------------------------------------------------------
# Dynamic service level adjustments (IPfeature11)
# ---------------------------------------------------------------------------

# Standard adjustments config used across dynamic SL tests
SL_ADJUSTMENTS = {
    "seasonal_peak_boost": 0.02,
    "seasonal_trough_relax": -0.01,
    "intermittent_relax": -0.02,
    "sl_floor": 0.80,
    "sl_ceiling": 0.995,
}


def test_sl_adjustment_no_adjustments_kwarg():
    """Without adjustments kwarg, no adjustments applied, reason is None."""
    sl, _xyz, _segment, reason = get_service_level(
        "A", SERVICE_LEVELS, demand_cv=0.1,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
    )
    assert abs(sl - 0.99) < 1e-6
    assert reason is None


def test_sl_adjustment_peak_season_boost():
    """Peak season adds seasonal_peak_boost to base SL."""
    sl, _xyz, _segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_peak_season=True,
        adjustments=SL_ADJUSTMENTS,
    )
    # Base BY = 0.95, + 0.02 peak boost = 0.97
    assert abs(sl - 0.97) < 1e-6
    assert reason is not None
    assert "seasonal_peak_boost(+0.02)" in reason


def test_sl_adjustment_trough_season_relax():
    """Trough season applies seasonal_trough_relax to base SL."""
    sl, _xyz, _segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_trough_season=True,
        adjustments=SL_ADJUSTMENTS,
    )
    # Base BY = 0.95, - 0.01 trough relax = 0.94
    assert abs(sl - 0.94) < 1e-6
    assert reason is not None
    assert "seasonal_trough_relax(-0.01)" in reason


def test_sl_adjustment_intermittent_relax():
    """High intermittency ratio applies intermittent_relax."""
    sl, _xyz, _segment, reason = get_service_level(
        "A", SERVICE_LEVELS, demand_cv=0.1,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        intermittency_ratio=0.6,
        adjustments=SL_ADJUSTMENTS,
    )
    # Base AX = 0.99, - 0.02 intermittent relax = 0.97
    assert abs(sl - 0.97) < 1e-6
    assert reason is not None
    assert "intermittent_relax(-0.02)" in reason


def test_sl_adjustment_intermittent_below_threshold():
    """Intermittency ratio <= 0.5 does not trigger adjustment."""
    sl, _xyz, _segment, reason = get_service_level(
        "A", SERVICE_LEVELS, demand_cv=0.1,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        intermittency_ratio=0.5,
        adjustments=SL_ADJUSTMENTS,
    )
    # Base AX = 0.99, no intermittent adjustment (ratio exactly 0.5, not > 0.5)
    assert abs(sl - 0.99) < 1e-6
    assert reason is None


def test_sl_adjustment_combined_peak_and_intermittent():
    """Multiple adjustments are applied additively and reasons concatenated."""
    sl, _xyz, _segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_peak_season=True,
        intermittency_ratio=0.7,
        adjustments=SL_ADJUSTMENTS,
    )
    # Base BY = 0.95, + 0.02 peak - 0.02 intermittent = 0.95
    assert abs(sl - 0.95) < 1e-6
    assert reason is not None
    assert "seasonal_peak_boost" in reason
    assert "intermittent_relax" in reason


def test_sl_adjustment_floor_clamp():
    """SL never goes below sl_floor."""
    sl, _xyz, _segment, _reason = get_service_level(
        "C", SERVICE_LEVELS, demand_cv=0.9,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_trough_season=True,
        intermittency_ratio=0.8,
        adjustments={
            "seasonal_trough_relax": -0.05,
            "intermittent_relax": -0.05,
            "sl_floor": 0.80,
            "sl_ceiling": 0.995,
        },
    )
    # Base CZ = 0.85, - 0.05 trough - 0.05 intermittent = 0.75 → clamped to 0.80
    assert abs(sl - 0.80) < 1e-6


def test_sl_adjustment_ceiling_clamp():
    """SL never goes above sl_ceiling."""
    sl, _xyz, _segment, _reason = get_service_level(
        "A", SERVICE_LEVELS, demand_cv=0.1,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_peak_season=True,
        adjustments={
            "seasonal_peak_boost": 0.05,
            "sl_floor": 0.80,
            "sl_ceiling": 0.995,
        },
    )
    # Base AX = 0.99, + 0.05 = 1.04 → clamped to 0.995
    assert abs(sl - 0.995) < 1e-6


def test_sl_adjustment_empty_adjustments_dict():
    """Empty adjustments dict applies no changes (no reasons)."""
    sl, _xyz, _segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_peak_season=True,
        adjustments={},
    )
    # Empty dict → adjustments is truthy but all .get() default to 0
    # Still clamped to [0.80, 0.995] but base BY=0.95 is within range
    assert abs(sl - 0.95) < 1e-6
    assert reason is None


def test_sl_adjustment_no_reason_when_no_flags():
    """When adjustments dict is provided but no flags are set, reason is None."""
    sl, _xyz, _segment, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        intermittency_ratio=0.3,
        adjustments=SL_ADJUSTMENTS,
    )
    # No peak, no trough, intermittency < 0.5 → no adjustments
    assert abs(sl - 0.95) < 1e-6
    assert reason is None


def test_sl_adjustment_peak_and_trough_mutually_exclusive():
    """Peak and trough are mutually exclusive — peak takes priority."""
    sl, _, _, reason = get_service_level(
        "B", SERVICE_LEVELS, demand_cv=0.5,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        is_peak_season=True,
        is_trough_season=True,
        adjustments=SL_ADJUSTMENTS,
    )
    # Peak takes priority (elif branch for trough)
    assert abs(sl - 0.97) < 1e-6
    assert "seasonal_peak_boost" in reason
    assert "seasonal_trough_relax" not in reason
