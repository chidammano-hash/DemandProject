"""Unit tests for scripts/apply_event_adjustments.py — F4.3."""

import pytest
from scripts.forecasting.apply_event_adjustments import (
    apply_event_uplift,
    compute_event_impact_value,
    compute_post_event_accuracy,
)


class TestApplyEventUplift:
    def test_promo_multiplier(self):
        adjusted, delta = apply_event_uplift(450.0, 1.40)
        assert adjusted == pytest.approx(630.0, abs=1e-2)
        assert delta == pytest.approx(180.0, abs=1e-2)

    def test_phase_out_multiplier(self):
        adjusted, delta = apply_event_uplift(450.0, 0.30)
        assert adjusted == pytest.approx(135.0, abs=1e-2)
        assert delta == pytest.approx(-315.0, abs=1e-2)

    def test_additive_uplift(self):
        adjusted, delta = apply_event_uplift(450.0, 1.0, additive_qty=50.0)
        assert adjusted == pytest.approx(500.0, abs=1e-2)
        assert delta == pytest.approx(50.0, abs=1e-2)

    def test_hard_override(self):
        adjusted, delta = apply_event_uplift(450.0, 1.0, is_hard_override=True, override_qty=300.0)
        assert adjusted == pytest.approx(300.0, abs=1e-2)
        assert delta == pytest.approx(-150.0, abs=1e-2)

    def test_floor_at_zero(self):
        adjusted, delta = apply_event_uplift(450.0, 0.0)
        assert adjusted == pytest.approx(0.0, abs=1e-4)

    def test_max_multiplier_guard(self):
        # Multiplier 6.0 exceeds max=5.0 → clamped to 5.0
        adjusted, delta = apply_event_uplift(100.0, 6.0, max_multiplier=5.0)
        assert adjusted == pytest.approx(500.0, abs=1e-2)

    def test_no_uplift(self):
        adjusted, delta = apply_event_uplift(450.0, 1.0)
        assert adjusted == pytest.approx(450.0, abs=1e-2)
        assert delta == pytest.approx(0.0, abs=1e-4)

    def test_combined_multiplier_and_additive(self):
        # base=400, multiplier=1.25, additive=20 → 400×1.25 + 20 = 520
        adjusted, delta = apply_event_uplift(400.0, 1.25, additive_qty=20.0)
        assert adjusted == pytest.approx(520.0, abs=1e-2)


class TestComputeEventImpactValue:
    def test_positive_uplift(self):
        result = compute_event_impact_value(180.0, 24.0)
        assert result == pytest.approx(4320.0, abs=1e-2)

    def test_negative_uplift_magnitude(self):
        # Phase-out with -315 delta → impact = abs(-315) × cost
        result = compute_event_impact_value(-315.0, 24.0)
        assert result == pytest.approx(7560.0, abs=1e-2)

    def test_zero_delta(self):
        result = compute_event_impact_value(0.0, 24.0)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestComputePostEventAccuracy:
    def test_perfect_forecast(self):
        bias, abs_err = compute_post_event_accuracy(630.0, 630.0)
        assert bias == pytest.approx(0.0, abs=1e-4)
        assert abs_err == pytest.approx(0.0, abs=1e-4)

    def test_over_forecast(self):
        # forecast=700, actual=630 → bias = (700-630)/630 × 100 > 0
        bias, abs_err = compute_post_event_accuracy(700.0, 630.0)
        assert bias > 0
        assert abs_err > 0

    def test_under_forecast(self):
        bias, abs_err = compute_post_event_accuracy(500.0, 630.0)
        assert bias < 0

    def test_zero_actual_returns_zero(self):
        bias, abs_err = compute_post_event_accuracy(500.0, 0.0)
        assert bias == 0.0
        assert abs_err == 0.0
