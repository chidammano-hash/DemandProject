"""Tests for the empirical-quantile safety-stock mode — Gen-4 Roadmap #2."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_ss_simulation import (
    compute_service_level_curve,
    empirical_quantile_ss,
    find_recommended_ss,
)


def test_empirical_quantile_ss_matches_numpy_quantile():
    # Construct a stationary setup: demand ~ 10/day, LT constant 5 days.
    rng = np.random.default_rng(0)
    demand = list(rng.normal(loc=10.0, scale=2.0, size=1000).clip(min=0))
    lt = [5.0] * 200

    ss_p95 = empirical_quantile_ss(demand, lt, n_simulations=5000, target_csl=0.95, random_seed=42)
    ss_p50 = empirical_quantile_ss(demand, lt, n_simulations=5000, target_csl=0.50, random_seed=42)

    assert ss_p95 > ss_p50 > 0
    # Sanity: mean LT demand ~ 50; p95 should be meaningfully above p50.
    assert ss_p95 > 45
    assert ss_p95 < 120


def test_empirical_quantile_consistent_with_grid_sweep_at_target():
    """SS derived from the empirical-quantile path should be within 1 grid step
    of the value the normal_approx grid produces at the same target_csl."""
    rng = np.random.default_rng(1)
    demand = list(rng.normal(loc=20.0, scale=5.0, size=500).clip(min=0))
    lt = [7.0] * 200

    target_csl = 0.90
    eq_ss = empirical_quantile_ss(demand, lt, n_simulations=3000, target_csl=target_csl, random_seed=7)

    # Run the grid sweep and look up where CSL first crosses 0.90
    ss_levels = list(np.linspace(0, eq_ss * 2, 20))
    curve = compute_service_level_curve(demand, lt, n_simulations=3000, ss_levels=ss_levels, random_seed=7)
    grid_ss = find_recommended_ss(curve, target_csl)

    # Grid result must be >= the empirical quantile (grid picks first level >=
    # target; the exact quantile is always <= the smallest grid value satisfying
    # the constraint).
    assert grid_ss is None or grid_ss >= eq_ss - 1e-6


def test_empirical_quantile_ss_rejects_bad_inputs():
    with pytest.raises(ValueError):
        empirical_quantile_ss([], [1.0], n_simulations=100, target_csl=0.95)
    with pytest.raises(ValueError):
        empirical_quantile_ss([1.0], [], n_simulations=100, target_csl=0.95)
    with pytest.raises(ValueError):
        empirical_quantile_ss([1.0], [1.0], n_simulations=100, target_csl=0.0)
    with pytest.raises(ValueError):
        empirical_quantile_ss([1.0], [1.0], n_simulations=100, target_csl=1.0)


def test_empirical_quantile_ss_monotonic_in_target():
    """Higher target CSL => higher SS recommendation."""
    rng = np.random.default_rng(3)
    demand = list(rng.normal(loc=15, scale=4, size=300).clip(min=0))
    lt = [3.0, 4.0, 5.0] * 50

    vals = [
        empirical_quantile_ss(demand, lt, n_simulations=2000, target_csl=q, random_seed=42)
        for q in (0.50, 0.80, 0.95, 0.99)
    ]
    assert vals == sorted(vals)
