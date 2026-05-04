"""Unit tests for IPfeature9 demand sensing pure functions."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inventory.compute_demand_signals import (
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


# ---------------------------------------------------------------------------
# Batched upsert in run()
# ---------------------------------------------------------------------------

class TestDemandSignalsBatching:
    """Verify the run() function batches upserts in 10K chunks."""

    @patch("scripts.inventory.compute_demand_signals.get_planning_date", return_value=date(2026, 3, 15))
    @patch("scripts.inventory.compute_demand_signals.get_db_params", return_value={})
    @patch("scripts.inventory.compute_demand_signals.psycopg")
    def test_upsert_batches_at_10k(self, mock_psycopg, mock_db, mock_pd):
        """Signals exceeding 10K should be split into multiple executemany calls."""
        from scripts.inventory.compute_demand_signals import run

        # Build mock connection for the read phase
        mock_read_conn = MagicMock()
        mock_read_cur = MagicMock()
        mock_read_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_read_cur)
        mock_read_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Build mock connection for the write phase
        mock_write_conn = MagicMock()
        mock_write_cur = MagicMock()
        mock_write_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_write_cur)
        mock_write_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # psycopg.connect is called twice: once for reads, once for writes
        mock_psycopg.connect.return_value.__enter__ = MagicMock(
            side_effect=[mock_read_conn, mock_write_conn]
        )
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate 25K snapshot rows that produce 25K signals
        n_signals = 25_000
        snapshot_rows = [
            (f"ITEM-{i}", "LOC-A", 100.0, 500.0)
            for i in range(n_signals)
        ]
        forecast_rows = [
            (f"ITEM-{i}", "LOC-A", 200.0)
            for i in range(n_signals)
        ]
        ss_rows = [
            (f"ITEM-{i}", "LOC-A", 50.0, False)
            for i in range(n_signals)
        ]

        # Mock the read cursor fetchall calls in sequence:
        # 1. fetchone for max snapshot_date
        # 2. fetchall for snapshot rows
        # 3. fetchall for forecast rows
        # 4. fetchall for safety stock rows
        mock_read_cur.fetchone.return_value = (date(2026, 3, 15),)
        mock_read_cur.fetchall.side_effect = [snapshot_rows, forecast_rows, ss_rows]

        result = run(signal_date=date(2026, 3, 15), dry_run=False)

        assert result["inserted"] == n_signals
        # Write cursor should have 3 batches: 10K + 10K + 5K
        assert mock_write_cur.executemany.call_count == 3
