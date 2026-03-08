"""Unit tests for scripts/compute_blended_forecast.py — F3.4."""

import pytest
from scripts.compute_blended_forecast import (
    compute_alpha,
    compute_velocity_signal,
    monthly_to_weekly,
    apply_dow_factor,
)


class TestComputeAlpha:
    def test_week_zero_is_pure_sensing(self):
        assert compute_alpha(0, 4) == pytest.approx(1.0, abs=1e-4)

    def test_at_horizon_is_zero(self):
        assert compute_alpha(4, 4) == pytest.approx(0.0, abs=1e-4)

    def test_linear_decay_midpoint(self):
        # week_offset=2, horizon=4 → 1 - 2/4 = 0.5
        assert compute_alpha(2, 4) == pytest.approx(0.5, abs=1e-4)

    def test_beyond_horizon_clipped_to_zero(self):
        assert compute_alpha(10, 4) == pytest.approx(0.0, abs=1e-4)

    def test_week_three_of_four(self):
        # 1 - 3/4 = 0.25
        assert compute_alpha(3, 4) == pytest.approx(0.25, abs=1e-4)

    def test_zero_horizon_returns_zero(self):
        assert compute_alpha(0, 0) == pytest.approx(0.0, abs=1e-4)

    def test_alpha_in_range(self):
        for offset in range(10):
            a = compute_alpha(offset, 4)
            assert 0.0 <= a <= 1.0


class TestComputeVelocitySignal:
    def test_normal_pace(self):
        # MTD=40, elapsed=10, days=30, hist_avg=4 → daily_run_rate=4, not capped
        proj, rate, spike, capped = compute_velocity_signal(40, 10, 30, 4.0)
        assert rate == pytest.approx(4.0, abs=1e-2)
        assert spike == pytest.approx(1.0, abs=1e-2)
        assert not capped
        assert proj == pytest.approx(120.0, abs=1e-2)

    def test_capped_spike(self):
        # MTD=160, elapsed=10, days=30, hist_avg=4 → spike=4.0 > 3.0 → capped at threshold
        proj, rate, spike, capped = compute_velocity_signal(160, 10, 30, 4.0)
        assert capped
        assert spike == pytest.approx(4.0, abs=1e-2)
        # Rate capped at 4.0 × 3.0 = 12.0 (outlier_threshold=3.0)
        assert rate == pytest.approx(12.0, abs=1e-2)

    def test_zero_elapsed_returns_historical(self):
        proj, rate, spike, capped = compute_velocity_signal(0, 0, 30, 5.0)
        assert proj == pytest.approx(150.0, abs=1e-2)  # 5.0 × 30
        assert not capped

    def test_zero_hist_avg_no_divide_by_zero(self):
        proj, rate, spike, capped = compute_velocity_signal(30, 10, 30, 0.0)
        assert spike == pytest.approx(1.0, abs=1e-2)  # default when hist_avg=0


class TestMonthlyToWeekly:
    def test_typical_conversion(self):
        result = monthly_to_weekly(100.0)
        assert result == pytest.approx(100 / 4.33, abs=1e-2)

    def test_zero_months_returns_zero(self):
        result = monthly_to_weekly(100.0, 0)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestApplyDowFactor:
    def test_default_factor_unchanged(self):
        assert apply_dow_factor(50.0) == pytest.approx(50.0, abs=1e-4)

    def test_custom_factor(self):
        assert apply_dow_factor(50.0, 1.2) == pytest.approx(60.0, abs=1e-4)
