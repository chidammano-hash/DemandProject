"""Tests for common.ml.sensing — horizon-weighted blend math."""

from __future__ import annotations

import pytest

from common.ml.sensing import blend_forecasts


_CFG = {
    "short_weight_horizon_0_7": 0.8,
    "long_weight_horizon_30_plus": 0.95,
    "short_horizon_days": 7,
    "long_horizon_days": 30,
    "min_near_term_weight": 0.0,
}


def test_short_horizon_uses_near_term_weight():
    # Horizon 3 days -> w_near = 0.8 -> 0.8*100 + 0.2*200 = 120.
    blended = blend_forecasts(long_range=200.0, near_term=100.0, horizon_days_out=3,
                              weights_config=_CFG)
    assert blended == pytest.approx(120.0)


def test_at_short_boundary_exact_short_weight():
    blended = blend_forecasts(200.0, 100.0, 7, _CFG)
    assert blended == pytest.approx(0.8 * 100.0 + 0.2 * 200.0)


def test_long_horizon_uses_residual_weight():
    # long_weight = 0.95 -> w_near = 0.05; 0.05*100 + 0.95*200 = 195.
    blended = blend_forecasts(200.0, 100.0, 60, _CFG)
    assert blended == pytest.approx(0.05 * 100.0 + 0.95 * 200.0)


def test_at_long_boundary_exact_residual_weight():
    blended = blend_forecasts(200.0, 100.0, 30, _CFG)
    assert blended == pytest.approx(0.05 * 100.0 + 0.95 * 200.0)


def test_midpoint_linearly_interpolates():
    # At horizon 18.5 (midpoint of [7, 30]) the near-term weight is
    # midway between 0.8 and 0.05 -> 0.425.
    blended_18 = blend_forecasts(200.0, 100.0, 18, _CFG)
    blended_19 = blend_forecasts(200.0, 100.0, 19, _CFG)
    # Mono-decreasing near-term contribution -> blend mono-increasing toward long_range.
    assert blended_18 < blended_19


def test_zero_horizon_uses_short_weight():
    blended = blend_forecasts(200.0, 100.0, 0, _CFG)
    assert blended == pytest.approx(0.8 * 100.0 + 0.2 * 200.0)


def test_negative_horizon_raises():
    with pytest.raises(ValueError):
        blend_forecasts(100.0, 100.0, -1, _CFG)


def test_short_horizon_greater_than_long_raises():
    bad_cfg = dict(_CFG)
    bad_cfg["short_horizon_days"] = 40
    with pytest.raises(ValueError):
        blend_forecasts(100.0, 100.0, 10, bad_cfg)


def test_min_near_term_weight_floor_applied():
    # min_near_term_weight = 0.1 -> long-horizon weight never drops below 0.1.
    cfg = dict(_CFG, min_near_term_weight=0.1)
    blended = blend_forecasts(200.0, 100.0, 120, cfg)
    # Near-term weight clamped to 0.1 -> 0.1*100 + 0.9*200 = 190.
    assert blended == pytest.approx(190.0)


def test_defaults_used_when_config_keys_missing():
    # An empty config should fall back to hard-coded defaults matching sensing_config.yaml.
    blended = blend_forecasts(200.0, 100.0, 3, {})
    assert blended == pytest.approx(0.8 * 100.0 + 0.2 * 200.0)
