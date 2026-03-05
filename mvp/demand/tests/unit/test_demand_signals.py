"""Unit tests for IPfeature9 demand sensing pure functions."""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.compute_demand_signals import (
    compute_projected_monthly,
    compute_demand_vs_forecast_pct,
    classify_signal_type,
    compute_signal_strength,
    classify_alert_priority,
    compute_projected_stockout,
    MIN_DAY_OF_MONTH,
)


# ---------------------------------------------------------------------------
# compute_projected_monthly
# ---------------------------------------------------------------------------

def test_projection_basic():
    result = compute_projected_monthly(50.0, 15, 31)
    assert result == pytest.approx(50.0 * (31 / 15), rel=1e-6)


def test_projection_returns_none_below_min_day():
    assert compute_projected_monthly(100.0, MIN_DAY_OF_MONTH - 1, 30) is None


def test_projection_zero_day_of_month():
    assert compute_projected_monthly(100.0, 0, 30) is None


def test_projection_day_equals_days_in_month():
    result = compute_projected_monthly(300.0, 30, 30)
    assert result == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# compute_demand_vs_forecast_pct
# ---------------------------------------------------------------------------

def test_demand_vs_forecast_above():
    result = compute_demand_vs_forecast_pct(115.0, 100.0)
    assert result == pytest.approx(15.0)


def test_demand_vs_forecast_below():
    result = compute_demand_vs_forecast_pct(75.0, 100.0)
    assert result == pytest.approx(-25.0)


def test_demand_vs_forecast_zero_forecast():
    assert compute_demand_vs_forecast_pct(100.0, 0.0) is None


def test_demand_vs_forecast_none_forecast():
    assert compute_demand_vs_forecast_pct(100.0, None) is None


def test_demand_vs_forecast_none_projected():
    assert compute_demand_vs_forecast_pct(None, 100.0) is None


# ---------------------------------------------------------------------------
# classify_signal_type
# ---------------------------------------------------------------------------

def test_signal_above_plan():
    assert classify_signal_type(15.0) == "above_plan"


def test_signal_below_plan():
    assert classify_signal_type(-25.0) == "below_plan"


def test_signal_on_plan_positive():
    assert classify_signal_type(5.0) == "on_plan"


def test_signal_on_plan_negative():
    assert classify_signal_type(-5.0) == "on_plan"


def test_signal_on_plan_none():
    assert classify_signal_type(None) == "on_plan"


# ---------------------------------------------------------------------------
# compute_signal_strength
# ---------------------------------------------------------------------------

def test_signal_strength():
    assert compute_signal_strength(30.0) == pytest.approx(0.30)


def test_signal_strength_negative():
    assert compute_signal_strength(-40.0) == pytest.approx(0.40)


def test_signal_strength_none():
    assert compute_signal_strength(None) == 0.0


# ---------------------------------------------------------------------------
# classify_alert_priority
# ---------------------------------------------------------------------------

def test_alert_urgent_stockout_below_ss():
    assert classify_alert_priority(True, True, 15.0) == "urgent"


def test_alert_watch_high_deviation():
    assert classify_alert_priority(False, False, 25.0) == "watch"


def test_alert_watch_below_plan_high():
    assert classify_alert_priority(False, False, -25.0) == "watch"


def test_alert_none_low_deviation():
    assert classify_alert_priority(False, False, 5.0) == "none"


def test_alert_not_urgent_if_not_below_ss():
    # projected_stockout=True but is_below_ss=False → watch (if deviation > threshold)
    assert classify_alert_priority(True, False, 25.0) == "watch"


# ---------------------------------------------------------------------------
# compute_projected_stockout
# ---------------------------------------------------------------------------

def test_projected_stockout_true():
    # daily demand = 10, 15 days remaining, on hand = 100 → need 150, have 100 → stockout
    assert compute_projected_stockout(10.0, 15, 100.0) is True


def test_projected_stockout_false():
    # daily demand = 5, 10 days remaining, on hand = 100 → need 50, have 100 → ok
    assert compute_projected_stockout(5.0, 10, 100.0) is False


def test_projected_stockout_zero_days_remaining():
    assert compute_projected_stockout(100.0, 0, 50.0) is False
