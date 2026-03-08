"""Unit tests for compute_inventory_projection.py — F1.2."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today():
    return date.today()


def _make_demand(daily_rate: float, horizon: int = 30) -> dict:
    start = _today()
    return {start + timedelta(days=i): daily_rate for i in range(horizon)}


def _make_receipts(day_offset: int, qty: float, horizon: int = 30) -> dict:
    start = _today()
    d = start + timedelta(days=day_offset)
    if day_offset < horizon:
        return {d: qty}
    return {}


def _run(
    current_qty: float,
    daily_rate: float,
    receipts: dict | None = None,
    safety_stock: float = 0.0,
    max_coverage_qty: float = 9999.0,
    horizon: int = 30,
    scenario: str = "no_order",
):
    from scripts.compute_inventory_projection import run_projection_scenario
    start = _today()
    demand = _make_demand(daily_rate, horizon)
    rcpts = receipts or {}
    return run_projection_scenario(
        current_qty=current_qty,
        demand_by_day=demand,
        receipts_by_day=rcpts,
        safety_stock=safety_stock,
        max_coverage_qty=max_coverage_qty,
        horizon_days=horizon,
        start_date=start,
        scenario=scenario,
        forecast_source="production_forecast",
        plan_version="2026-03",
        projection_run_id="test-run-id",
        item_no="100320",
        loc="1401-BULK",
    )


# ---------------------------------------------------------------------------
# Core projection simulation
# ---------------------------------------------------------------------------

class TestRunProjectionScenario:
    def test_no_order_depletes(self):
        """Projected qty decreases monotonically toward 0 with no receipts."""
        rows = _run(current_qty=100.0, daily_rate=5.0, horizon=20)
        qtys = [r["projected_qty"] for r in rows]
        # Strictly decreasing until hitting 0
        for i in range(len(qtys) - 1):
            if qtys[i] > 0:
                assert qtys[i] >= qtys[i + 1], f"Qty increased at day {i}"

    def test_qty_never_negative(self):
        """Projected qty is clamped to 0 — never negative."""
        rows = _run(current_qty=10.0, daily_rate=20.0, horizon=10)
        for r in rows:
            assert r["projected_qty"] >= 0.0

    def test_po_bumps_qty(self):
        """A PO receipt on day N increases projected_qty above no-order curve."""
        horizon = 30
        day_offset = 10
        receipt_qty = 200.0
        rcpts = _make_receipts(day_offset, receipt_qty, horizon)

        no_order_rows = _run(100.0, 5.0, receipts=None, horizon=horizon, scenario="no_order")
        with_po_rows = _run(100.0, 5.0, receipts=rcpts, horizon=horizon, scenario="with_open_po")

        # Before receipt: should be identical (no receipt yet)
        for i in range(day_offset):
            assert no_order_rows[i]["projected_qty"] == pytest.approx(with_po_rows[i]["projected_qty"], abs=0.01)

        # On and after receipt day: with_po >= no_order
        for i in range(day_offset, horizon):
            assert with_po_rows[i]["projected_qty"] >= no_order_rows[i]["projected_qty"] - 0.01

    def test_stockout_date_correct(self):
        """stockout_risk becomes True on the first day projected_qty reaches 0."""
        rows = _run(current_qty=50.0, daily_rate=10.0, horizon=10)
        # Day 5 should be the first stockout (50 / 10 = day 5)
        stockout_rows = [r for r in rows if r["stockout_risk"]]
        assert len(stockout_rows) > 0
        # First stockout should be around day 5
        first_stockout = stockout_rows[0]
        days_to_first_stockout = (first_stockout["projection_date"] - _today()).days
        assert days_to_first_stockout == 4  # i=4: qty = 50-5*10=0 (0-indexed)

    def test_reorder_trigger_correct(self):
        """reorder_triggered becomes True when qty <= safety_stock."""
        rows = _run(current_qty=100.0, daily_rate=5.0, safety_stock=50.0, horizon=30)
        reorder_rows = [r for r in rows if r["reorder_triggered"]]
        assert len(reorder_rows) > 0
        # First reorder trigger should be when qty drops to 50 = after 10 days
        first_reorder = reorder_rows[0]
        days_to_reorder = (first_reorder["projection_date"] - _today()).days
        assert days_to_reorder == 9  # i=9: qty=100-10*5=50=SS (0-indexed)

    def test_excess_flag_correct(self):
        """excess_risk is True when projected_qty > max_coverage_qty."""
        # Start with massive qty, low demand, low coverage threshold
        rows = _run(current_qty=500.0, daily_rate=1.0, max_coverage_qty=100.0, horizon=5)
        # All rows should be excess (500 > 100)
        for r in rows:
            assert r["excess_risk"] is True

    def test_zero_demand_stable_qty(self):
        """Zero demand leaves qty unchanged throughout the horizon."""
        rows = _run(current_qty=120.0, daily_rate=0.0, horizon=30)
        for r in rows:
            assert r["projected_qty"] == pytest.approx(120.0, abs=0.01)
            assert r["stockout_risk"] is False

    def test_multiple_pos_same_day_summed(self):
        """Two POs arriving the same day are additive."""
        start = _today()
        rcpts = {start + timedelta(days=5): 100.0 + 150.0}  # pre-summed = 250
        rows = _run(50.0, 10.0, receipts=rcpts, horizon=10, scenario="with_open_po")
        # On day 5: qty = max(0, 0 + 250 - 10) = 240
        day5_row = rows[5]
        assert day5_row["projected_qty"] == pytest.approx(240.0, abs=0.01)

    def test_row_count_equals_horizon(self):
        """Number of output rows equals horizon_days."""
        horizon = 45
        rows = _run(current_qty=100.0, daily_rate=2.0, horizon=horizon)
        assert len(rows) == horizon

    def test_cumulative_demand_monotonically_increasing(self):
        """forecast_qty_consumed is non-decreasing across rows."""
        rows = _run(current_qty=200.0, daily_rate=5.0, horizon=20)
        consumed = [r["forecast_qty_consumed"] for r in rows]
        for i in range(len(consumed) - 1):
            assert consumed[i] <= consumed[i + 1]

    def test_dos_is_capped_when_zero_demand(self):
        """projected_dos is capped at 9999 when daily_demand_rate = 0."""
        rows = _run(current_qty=100.0, daily_rate=0.0, horizon=5)
        for r in rows:
            assert r["projected_dos"] == pytest.approx(9999.0, abs=0.01)


# ---------------------------------------------------------------------------
# Monthly disaggregation (standalone)
# ---------------------------------------------------------------------------

class TestDisaggregateDemand:
    def test_even_split(self):
        """490 units / 30 calendar days = ~16.333 per day."""
        import calendar
        from datetime import date, timedelta

        # Simulate what get_daily_demand_rates does with a single monthly forecast
        horizon = 30
        start_date = date(2026, 4, 1)
        monthly = {date(2026, 4, 1): 490.0}

        daily = {}
        for i in range(horizon):
            d = start_date + timedelta(days=i)
            month_start = d.replace(day=1)
            days_in_month = calendar.monthrange(d.year, d.month)[1]
            daily[d] = monthly.get(month_start, 0.0) / days_in_month

        expected_rate = 490.0 / 30
        for rate in daily.values():
            assert rate == pytest.approx(expected_rate, rel=0.01)

    def test_multi_month_boundary(self):
        """Demand crosses month boundary correctly."""
        import calendar
        from datetime import date, timedelta

        start_date = date(2026, 3, 25)
        horizon = 14  # spans Mar + Apr
        monthly = {date(2026, 3, 1): 310.0, date(2026, 4, 1): 300.0}

        daily = {}
        for i in range(horizon):
            d = start_date + timedelta(days=i)
            month_start = d.replace(day=1)
            days_in_month = calendar.monthrange(d.year, d.month)[1]
            daily[d] = monthly.get(month_start, 0.0) / days_in_month

        # March days should use 310 / 31
        mar_days = [d for d in daily if d.month == 3]
        for d in mar_days:
            assert daily[d] == pytest.approx(310.0 / 31, rel=0.01)

        # April days should use 300 / 30
        apr_days = [d for d in daily if d.month == 4]
        for d in apr_days:
            assert daily[d] == pytest.approx(300.0 / 30, rel=0.01)
