"""Unit tests for F2.2 — generate_quantile_forecasts.py."""

import math
import sys
import os
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.generate_quantile_forecasts import (
    compute_sigma_forecast,
    compute_sigma_combined,
    get_weekly_weights,
    disaggregate_to_weekly,
)


# ---------------------------------------------------------------------------
# compute_sigma_forecast
# ---------------------------------------------------------------------------

def test_sigma_forecast_basic():
    """σ_f = (P90 - P10) / 2.5632."""
    result = compute_sigma_forecast(320.0, 580.0)
    assert result == pytest.approx((580 - 320) / 2.5632, rel=1e-4)


def test_sigma_forecast_zero_when_equal():
    """P10 == P90 → σ_f = 0."""
    assert compute_sigma_forecast(450.0, 450.0) == 0.0


def test_sigma_forecast_zero_when_p90_less():
    """P90 < P10 (degenerate) → σ_f = 0."""
    assert compute_sigma_forecast(500.0, 400.0) == 0.0


def test_sigma_forecast_example_from_spec():
    """Spec example: P10=320, P90=580 → σ_f ≈ 101.6."""
    result = compute_sigma_forecast(320.0, 580.0)
    assert result == pytest.approx(101.6, abs=0.5)


# ---------------------------------------------------------------------------
# compute_sigma_combined
# ---------------------------------------------------------------------------

def test_sigma_combined_basic():
    """σ_combined = sqrt(σ_f² + σ_d²)."""
    sf, sd = 101.6, 80.0
    expected = math.sqrt(sf**2 + sd**2)
    assert compute_sigma_combined(sf, sd) == pytest.approx(expected, rel=1e-6)


def test_sigma_combined_zero_sigma_d():
    """If σ_d = 0, σ_combined = σ_f."""
    assert compute_sigma_combined(100.0, 0.0) == pytest.approx(100.0)


def test_sigma_combined_zero_sigma_f():
    """If σ_f = 0, σ_combined = σ_d."""
    assert compute_sigma_combined(0.0, 80.0) == pytest.approx(80.0)


def test_sigma_combined_example_from_spec():
    """Spec: σ_f=101.6, σ_d=80 → σ_combined ≈ 129.3."""
    result = compute_sigma_combined(101.6, 80.0)
    assert result == pytest.approx(129.3, abs=0.5)


# ---------------------------------------------------------------------------
# get_weekly_weights
# ---------------------------------------------------------------------------

def test_weekly_weights_sum_to_approximately_one():
    """Weights for a full 30-day month must sum to 1.0."""
    plan_month = date(2026, 4, 1)  # April has 30 days
    weights = get_weekly_weights(plan_month)
    total = sum(w for _, w in weights)
    assert total == pytest.approx(1.0, abs=0.001)


def test_weekly_weights_returns_list_of_tuples():
    plan_month = date(2026, 4, 1)
    weights = get_weekly_weights(plan_month)
    assert isinstance(weights, list)
    assert all(isinstance(wk, tuple) and len(wk) == 2 for wk in weights)


def test_weekly_weights_week_starts_are_mondays():
    """Each week_start must be a Monday (weekday() == 0)."""
    plan_month = date(2026, 4, 1)
    weights = get_weekly_weights(plan_month)
    for week_start, _ in weights:
        assert week_start.weekday() == 0, f"{week_start} is not a Monday"


def test_weekly_weights_all_positive():
    plan_month = date(2026, 4, 1)
    weights = get_weekly_weights(plan_month)
    assert all(w > 0 for _, w in weights)


def test_weekly_weights_31_day_month():
    """March has 31 days — weights should still sum to ≈ 1."""
    plan_month = date(2026, 3, 1)
    weights = get_weekly_weights(plan_month)
    total = sum(w for _, w in weights)
    assert total == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# disaggregate_to_weekly
# ---------------------------------------------------------------------------

def _make_monthly_row(plan_month: date, quantile: float, qty: float) -> dict:
    return {
        "item_no": "100320",
        "loc": "1401-BULK",
        "plan_month": plan_month,
        "quantile": quantile,
        "forecast_qty": qty,
    }


def test_disaggregate_weekly_sum_approx_monthly():
    """Sum of weekly forecast_qty should ≈ monthly forecast_qty."""
    plan_month = date(2026, 4, 1)
    monthly_rows = [_make_monthly_row(plan_month, 0.50, 450.0)]
    weekly = disaggregate_to_weekly(monthly_rows, "test-version")
    total = sum(r["forecast_qty"] for r in weekly)
    assert total == pytest.approx(450.0, abs=1.0)


def test_disaggregate_weekly_has_iso_week_and_year():
    plan_month = date(2026, 4, 1)
    monthly_rows = [_make_monthly_row(plan_month, 0.50, 100.0)]
    weekly = disaggregate_to_weekly(monthly_rows, "v1")
    for r in weekly:
        assert "iso_week" in r
        assert "iso_year" in r
        assert 1 <= r["iso_week"] <= 53


def test_disaggregate_weekly_preserves_quantile():
    plan_month = date(2026, 4, 1)
    for q in [0.10, 0.50, 0.90]:
        monthly_rows = [_make_monthly_row(plan_month, q, 300.0)]
        weekly = disaggregate_to_weekly(monthly_rows, "v1")
        assert all(r["quantile"] == q for r in weekly)


def test_disaggregate_weekly_preserves_version():
    plan_month = date(2026, 4, 1)
    monthly_rows = [_make_monthly_row(plan_month, 0.50, 200.0)]
    weekly = disaggregate_to_weekly(monthly_rows, "my-version")
    assert all(r["plan_version"] == "my-version" for r in weekly)
