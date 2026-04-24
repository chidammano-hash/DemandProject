"""Unit tests for common.ml.crps — CRPS and pinball loss."""
from __future__ import annotations

import numpy as np
import pytest

from common.ml.crps import compute_crps, compute_pinball_loss


def test_pinball_loss_perfect_median():
    # When the median forecast equals the actual, pinball at q=0.5 is 0.
    actuals = np.array([10.0, 20.0, 30.0])
    forecasts = np.array([[10.0], [20.0], [30.0]])  # single-quantile forecasts
    loss = compute_pinball_loss(actuals, forecasts, [0.5])
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_pinball_loss_overforecast_q10():
    # When forecast > actual at q=0.1, the loss weights overshoot by 0.9.
    actuals = np.array([5.0])
    forecasts = np.array([[10.0]])
    # diff = a - q = -5; max(0.1 * -5, -0.9 * -5) = max(-0.5, 4.5) = 4.5
    loss = compute_pinball_loss(actuals, forecasts, [0.1])
    assert loss == pytest.approx(4.5, rel=1e-6)


def test_pinball_loss_none_reduce_shape():
    actuals = np.array([1.0, 2.0])
    forecasts = np.array([[1.0, 1.5], [2.0, 2.5]])
    loss = compute_pinball_loss(actuals, forecasts, [0.2, 0.8], reduce="none")
    assert loss.shape == (2, 2)
    # Row 0: q0=1.0 matches a=1.0 so first col is 0.
    assert loss[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_crps_non_negative():
    rng = np.random.default_rng(0)
    actuals = rng.normal(size=50)
    levels = np.array([0.1, 0.5, 0.9])
    # Use Gaussian quantiles so forecasts look plausible.
    from scipy.stats import norm  # optional
    try:
        qvals = norm.ppf(levels)
        forecasts = np.tile(qvals, (50, 1))
    except ImportError:  # pragma: no cover
        forecasts = np.tile([-1.28, 0.0, 1.28], (50, 1))
    crps = compute_crps(actuals, forecasts, levels)
    assert crps >= 0.0


def test_crps_equals_zero_when_forecasts_equal_actuals_on_single_quantile():
    actuals = np.array([1.0, 2.0, 3.0])
    forecasts = np.array([[1.0], [2.0], [3.0]])
    crps = compute_crps(actuals, forecasts, [0.5])
    assert crps == pytest.approx(0.0, abs=1e-9)


def test_compute_crps_input_validation():
    actuals = np.array([1.0, 2.0])
    # Wrong K for quantile levels
    with pytest.raises(ValueError):
        compute_crps(actuals, np.array([[1.0], [2.0]]), [0.1, 0.9])


def test_compute_crps_rejects_bad_levels():
    actuals = np.array([1.0])
    forecasts = np.array([[1.0]])
    with pytest.raises(ValueError):
        compute_pinball_loss(actuals, forecasts, [1.5])
