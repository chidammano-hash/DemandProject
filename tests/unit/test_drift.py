"""Unit tests for common.ai.drift — PSI & rolling WAPE."""
from __future__ import annotations

import numpy as np
import pytest

from common.ai.drift import (
    compute_psi,
    psi_signal,
    rolling_wape,
    wape_signal,
)


def test_psi_zero_when_distributions_match():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5000)
    # Same distribution sampled twice -> near-zero PSI.
    other = rng.normal(size=5000)
    psi = compute_psi(sample, other, bins=10)
    assert psi >= 0.0
    assert psi < 0.05


def test_psi_large_when_distributions_shift():
    rng = np.random.default_rng(1)
    baseline = rng.normal(loc=0.0, size=5000)
    shifted = rng.normal(loc=3.0, size=5000)  # mean shift by 3 sd
    psi = compute_psi(baseline, shifted, bins=10)
    assert psi > 0.2  # material drift threshold


def test_psi_rejects_empty_inputs():
    with pytest.raises(ValueError):
        compute_psi([], [1.0, 2.0])


def test_psi_rejects_bad_bins():
    with pytest.raises(ValueError):
        compute_psi([1.0, 2.0], [1.0, 2.0], bins=1)


def test_psi_signal_threshold_breached():
    rng = np.random.default_rng(2)
    base = rng.normal(loc=0.0, size=2000)
    curr = rng.normal(loc=5.0, size=2000)
    sig = psi_signal("lgbm_cluster", "demand_qty", base, curr, threshold=0.2)
    assert sig.model_id == "lgbm_cluster"
    assert sig.metric == "psi_demand_qty"
    assert sig.threshold_breached is True
    assert sig.value > 0.2


def test_rolling_wape_shapes_and_values():
    actuals = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    forecasts = np.array([12.0, 10.0, 10.0, 10.0, 10.0])
    out = rolling_wape(actuals, forecasts, window=3)
    assert out.shape == (3,)
    # Window 0 contains one error of 2 / 30 total demand.
    assert out[0] == pytest.approx(2.0 / 30.0)
    # Later windows perfect
    assert out[-1] == pytest.approx(0.0, abs=1e-9)


def test_rolling_wape_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rolling_wape([1.0, 2.0], [1.0], window=1)


def test_rolling_wape_rejects_bad_window():
    with pytest.raises(ValueError):
        rolling_wape([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], window=0)
    with pytest.raises(ValueError):
        rolling_wape([1.0, 2.0], [1.0, 2.0], window=5)


def test_wape_signal_breach_flag():
    actuals = np.array([10.0, 10.0, 10.0])
    forecasts = np.array([20.0, 20.0, 20.0])  # 100% over
    sig = wape_signal("lgbm_cluster", actuals, forecasts, window=3, threshold=0.5)
    assert sig.metric == "rolling_wape"
    assert sig.threshold_breached is True


def test_wape_signal_no_breach():
    actuals = np.array([10.0, 10.0, 10.0])
    forecasts = np.array([10.0, 10.0, 10.0])
    sig = wape_signal("lgbm_cluster", actuals, forecasts, window=3, threshold=0.1)
    assert sig.threshold_breached is False
    assert sig.value == pytest.approx(0.0)
