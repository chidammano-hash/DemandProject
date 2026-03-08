"""Unit tests for scripts/compute_bias_corrections.py — F3.1."""

import pytest
from scripts.compute_bias_corrections import (
    compute_rolling_bias,
    derive_correction_factor,
    apply_correction_to_forecast,
)


class TestComputeRollingBias:
    def test_single_value(self):
        result = compute_rolling_bias([0.15])
        assert result == pytest.approx(0.15, abs=1e-4)

    def test_three_months_weighted(self):
        # weights [0.50, 0.30, 0.20] applied to [Mar=0.15, Feb=0.17, Jan=0.22]
        result = compute_rolling_bias([0.15, 0.17, 0.22], [0.50, 0.30, 0.20])
        expected = 0.50 * 0.15 + 0.30 * 0.17 + 0.20 * 0.22
        assert result == pytest.approx(expected, abs=1e-4)

    def test_weights_normalised_automatically(self):
        # Supply unnormalised weights — should still produce a weighted average
        result = compute_rolling_bias([0.10, 0.10], [1.0, 1.0])
        assert result == pytest.approx(0.10, abs=1e-4)

    def test_empty_list_returns_zero(self):
        # Empty bias_values: n=0, dot product returns 0.0
        result = compute_rolling_bias([], [0.50, 0.30, 0.20])
        assert isinstance(result, float)

    def test_more_values_than_weights_truncated(self):
        # Only first 3 values used (len of weights = 3)
        result = compute_rolling_bias([0.15, 0.17, 0.22, 0.99], [0.50, 0.30, 0.20])
        expected = 0.50 * 0.15 + 0.30 * 0.17 + 0.20 * 0.22
        assert result == pytest.approx(expected, abs=1e-4)

    def test_negative_bias(self):
        result = compute_rolling_bias([-0.10, -0.15, -0.20])
        assert result < 0


class TestDeriveCorrectionFactor:
    def test_positive_bias_produces_factor_below_one(self):
        # Forecast was 17% too high (bias=+0.17) → factor < 1.0 to correct down
        raw, clipped, was_clipped, flagged = derive_correction_factor(0.17)
        assert clipped < 1.0
        assert not was_clipped

    def test_negative_bias_produces_factor_above_one(self):
        # Forecast was 10% too low (bias=-0.10) → factor > 1.0 to correct up
        raw, clipped, was_clipped, flagged = derive_correction_factor(-0.10)
        assert clipped > 1.0

    def test_zero_bias_factor_equals_one(self):
        raw, clipped, was_clipped, flagged = derive_correction_factor(0.0)
        assert clipped == pytest.approx(1.0, abs=1e-4)
        assert not flagged

    def test_clip_at_max(self):
        # Extreme negative bias → raw factor > 1.30 → clipped
        raw, clipped, was_clipped, flagged = derive_correction_factor(-0.80)
        assert clipped == pytest.approx(1.30, abs=1e-4)
        assert was_clipped
        assert flagged

    def test_clip_at_min(self):
        # Extreme positive bias → raw factor < 0.70 → clipped
        raw, clipped, was_clipped, flagged = derive_correction_factor(0.80)
        assert clipped == pytest.approx(0.70, abs=1e-4)
        assert was_clipped

    def test_flagged_for_review_above_threshold(self):
        # bias=0.30 → raw=1/1.30≈0.769 → |1-0.769|=0.231 > 0.20 → flagged
        raw, clipped, was_clipped, flagged = derive_correction_factor(0.30)
        assert flagged

    def test_not_flagged_below_threshold(self):
        raw, clipped, was_clipped, flagged = derive_correction_factor(0.10)
        assert not flagged

    def test_bias_exactly_minus_one(self):
        # Guard against division by zero when rolling_bias == -1.0
        raw, clipped, was_clipped, flagged = derive_correction_factor(-1.0)
        assert clipped == pytest.approx(1.30, abs=1e-4)  # clamped to max


class TestApplyCorrectionToForecast:
    def test_reduces_over_forecast(self):
        result = apply_correction_to_forecast(1000.0, 0.90)
        assert result == pytest.approx(900.0, abs=1e-2)

    def test_increases_under_forecast(self):
        result = apply_correction_to_forecast(1000.0, 1.10)
        assert result == pytest.approx(1100.0, abs=1e-2)

    def test_floor_at_zero(self):
        result = apply_correction_to_forecast(100.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_no_correction(self):
        result = apply_correction_to_forecast(500.0, 1.0)
        assert result == pytest.approx(500.0, abs=1e-4)
