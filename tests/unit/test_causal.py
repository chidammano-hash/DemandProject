"""Tests for common.ai.causal (OLS elasticity fallback)."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from common.ai.causal import (
    ElasticityResult,
    _two_sided_p_from_t,
    fit_linear_elasticities,
)


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(seed=42)
    n = 200
    X = rng.normal(size=(n, 2))
    true_coef = np.array([2.0, -1.5])
    noise = rng.normal(scale=0.1, size=n)
    y = X @ true_coef + noise

    results = fit_linear_elasticities(X, y, ["price_log", "promo_flag"])
    assert len(results) == 2
    by_name = {r.feature: r for r in results}
    assert pytest.approx(by_name["price_log"].coef, abs=0.1) == 2.0
    assert pytest.approx(by_name["promo_flag"].coef, abs=0.1) == -1.5
    assert all(r.method == "linear_regression" for r in results)
    assert all(r.n_obs == n for r in results)


def test_fit_p_value_small_for_strong_signal():
    rng = np.random.default_rng(seed=7)
    n = 300
    X = rng.normal(size=(n, 1))
    y = 3.0 * X.ravel() + rng.normal(scale=0.5, size=n)
    results = fit_linear_elasticities(X, y, ["strong"])
    assert results[0].p_value is not None
    assert results[0].p_value < 0.01


def test_fit_shape_validation():
    with pytest.raises(ValueError):
        fit_linear_elasticities(np.zeros((5, 2)), np.zeros(4), ["a", "b"])
    with pytest.raises(ValueError):
        fit_linear_elasticities(np.zeros((5, 2)), np.zeros(5), ["a"])


def test_fit_small_n_does_not_crash():
    # n = k + 1 → df = 0, p-values should be NaN/None but no crash.
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    results = fit_linear_elasticities(X, y, ["feat"])
    assert len(results) == 1
    assert isinstance(results[0], ElasticityResult)


def test_two_sided_p_monotonic():
    # Larger |t| -> smaller p-value.
    p_small_t = _two_sided_p_from_t(0.5, df=100)
    p_large_t = _two_sided_p_from_t(5.0, df=100)
    assert p_large_t < p_small_t
    assert 0 <= p_large_t <= 1
    assert 0 <= p_small_t <= 1


def test_two_sided_p_handles_zero_df():
    assert math.isnan(_two_sided_p_from_t(1.0, df=0))
