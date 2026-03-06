"""Unit tests for intra-month stockout detection logic.

IPfeature14 — Intra-Month Stockout Detection.
Tests mirror the SQL formulas defined in sql/034_create_intramonth_stockout.sql
and the refresh logic in scripts/refresh_intramonth_stockout.py.

All tests are pure Python (no DB/infrastructure required).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# daily_sls derivation: GREATEST(mtd_sales - LAG(mtd_sales), 0)
# ---------------------------------------------------------------------------

def test_daily_sls_normal_increment():
    """Normal day: today's MTD minus yesterday's MTD gives positive daily sales."""
    prev_mtd = 50.0
    curr_mtd = 65.0
    daily_sls = max(0.0, curr_mtd - prev_mtd)
    assert daily_sls == 15.0


def test_daily_sls_clamp_negative():
    """Negative delta (LAG artifact or data correction) must be clamped to 0."""
    prev_mtd = 70.0
    curr_mtd = 65.0
    daily_sls = max(0.0, curr_mtd - prev_mtd)
    assert daily_sls == 0.0


def test_daily_sls_zero_delta():
    """Zero delta → zero daily sales (stockout day candidate)."""
    prev_mtd = 40.0
    curr_mtd = 40.0
    daily_sls = max(0.0, curr_mtd - prev_mtd)
    assert daily_sls == 0.0


def test_daily_sls_first_day_of_month():
    """First day of month: LAG defaults to 0, so daily_sls = mtd_sales."""
    prev_mtd = 0.0  # LAG default for first row in partition
    curr_mtd = 30.0
    daily_sls = max(0.0, curr_mtd - prev_mtd)
    assert daily_sls == 30.0


# ---------------------------------------------------------------------------
# Stockout day detection: qty_on_hand <= 0
# ---------------------------------------------------------------------------

def test_stockout_day_zero_on_hand():
    """qty_on_hand == 0 → stockout day."""
    qty_on_hand = 0.0
    is_stockout = qty_on_hand <= 0
    assert is_stockout is True


def test_stockout_day_negative_on_hand():
    """Negative on-hand (data artifact) is also treated as a stockout day."""
    qty_on_hand = -1.0
    is_stockout = qty_on_hand <= 0
    assert is_stockout is True


def test_non_stockout_day():
    """Positive on-hand → not a stockout day."""
    qty_on_hand = 10.0
    is_stockout = qty_on_hand <= 0
    assert is_stockout is False


# ---------------------------------------------------------------------------
# Stockout day rate: stockout_days / snapshot_days
# ---------------------------------------------------------------------------

def test_stockout_rate_partial():
    """3 stockout days out of 30 → 10% rate."""
    stockout_days, total_days = 3, 30
    rate = stockout_days / max(total_days, 1)
    assert abs(rate - 0.10) < 0.001


def test_stockout_rate_zero():
    """No stockout days → rate is 0."""
    rate = 0 / max(30, 1)
    assert rate == 0.0


def test_stockout_rate_full_month():
    """All 30 days are stockout → rate is 1.0."""
    rate = 30 / max(30, 1)
    assert rate == 1.0


def test_stockout_rate_zero_snapshot_days_guard():
    """Guard against zero snapshot_days to avoid division by zero."""
    stockout_days, total_days = 0, 0
    rate = stockout_days / max(total_days, 1)
    assert rate == 0.0


# ---------------------------------------------------------------------------
# had_full_stockout: at least 1 stockout day in month
# ---------------------------------------------------------------------------

def test_had_full_stockout_true():
    """One stockout day → had_full_stockout is True."""
    stockout_days = 1
    had_full_stockout = stockout_days >= 1
    assert had_full_stockout is True


def test_had_full_stockout_false():
    """Zero stockout days → had_full_stockout is False."""
    stockout_days = 0
    had_full_stockout = stockout_days >= 1
    assert had_full_stockout is False


# ---------------------------------------------------------------------------
# had_extended_stockout: >= 7 stockout days in month
# (SQL uses COUNT(*) FILTER(WHERE qty_on_hand <= 0) >= 7)
# ---------------------------------------------------------------------------

def test_had_extended_stockout_true():
    """7 or more stockout days → extended stockout."""
    stockout_days = 7
    had_extended = stockout_days >= 7
    assert had_extended is True


def test_had_extended_stockout_exactly_seven():
    """Exactly 7 days is the threshold → extended stockout."""
    had_extended = 7 >= 7
    assert had_extended is True


def test_had_extended_stockout_false_six_days():
    """6 days does not reach the 7-day threshold."""
    stockout_days = 6
    had_extended = stockout_days >= 7
    assert had_extended is False


def test_had_extended_stockout_false_zero():
    """Zero stockout days → not extended."""
    had_extended = 0 >= 7
    assert had_extended is False


# ---------------------------------------------------------------------------
# est_lost_sales: SUM(daily_sls) FILTER(WHERE qty_on_hand <= 0)
# Note: this sums actual daily_sls on stockout days (days with 0 on-hand),
# not avg_daily * stockout_days.
# ---------------------------------------------------------------------------

def test_estimated_lost_sales_from_daily_sls():
    """Lost sales = sum of daily_sls values observed on stockout days."""
    # Simulate 5 days: on-hand quantities and corresponding daily_sls
    days = [
        {"qty_on_hand": 0.0,  "daily_sls": 12.0},   # stockout → lost
        {"qty_on_hand": 5.0,  "daily_sls": 8.0},    # not stockout
        {"qty_on_hand": 0.0,  "daily_sls": 10.0},   # stockout → lost
        {"qty_on_hand": 10.0, "daily_sls": 9.0},    # not stockout
        {"qty_on_hand": 0.0,  "daily_sls": 0.0},    # stockout but 0 sales recorded
    ]
    est_lost = sum(d["daily_sls"] for d in days if d["qty_on_hand"] <= 0)
    assert est_lost == 22.0


def test_estimated_lost_sales_zero_when_no_stockout():
    """No stockout days → zero estimated lost sales."""
    days = [
        {"qty_on_hand": 5.0,  "daily_sls": 10.0},
        {"qty_on_hand": 10.0, "daily_sls": 12.0},
    ]
    est_lost = sum(d["daily_sls"] for d in days if d["qty_on_hand"] <= 0)
    assert est_lost == 0.0


def test_estimated_lost_sales_all_days_stockout():
    """All days stockout → lost sales equals total of all daily_sls."""
    days = [
        {"qty_on_hand": 0.0, "daily_sls": 5.0},
        {"qty_on_hand": 0.0, "daily_sls": 8.0},
        {"qty_on_hand": 0.0, "daily_sls": 3.0},
    ]
    est_lost = sum(d["daily_sls"] for d in days if d["qty_on_hand"] <= 0)
    assert est_lost == 16.0


# ---------------------------------------------------------------------------
# stockout_day_rate threshold filtering
# (idx_intramonth_high_rate partial index: stockout_day_rate > 0.10)
# ---------------------------------------------------------------------------

def test_high_rate_threshold_above():
    """stockout_day_rate > 0.10 qualifies for high-rate partial index."""
    rate = 0.15
    assert rate > 0.10


def test_high_rate_threshold_below():
    """stockout_day_rate <= 0.10 does NOT qualify for partial index."""
    rate = 0.05
    assert not (rate > 0.10)


def test_high_rate_threshold_exactly_10_pct():
    """Exactly 10% does NOT qualify (index uses strictly greater than)."""
    rate = 0.10
    assert not (rate > 0.10)


# ---------------------------------------------------------------------------
# _safe_refresh fallback logic (mirrored as pure Python state machine)
# ---------------------------------------------------------------------------

def test_safe_refresh_uses_concurrent_first():
    """_safe_refresh() should attempt CONCURRENT refresh before fallback."""
    calls = []

    class FakeConn:
        autocommit = False

        def execute(self, sql):
            calls.append(sql)
            if "CONCURRENTLY" in sql:
                raise Exception("FeatureNotSupported")

    conn = FakeConn()

    def safe_refresh_sim(conn, view):
        conn.autocommit = True
        try:
            conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
        except Exception:
            conn.execute(f"REFRESH MATERIALIZED VIEW {view}")

    safe_refresh_sim(conn, "mv_intramonth_stockout")

    assert calls[0] == "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_intramonth_stockout"
    assert calls[1] == "REFRESH MATERIALIZED VIEW mv_intramonth_stockout"


def test_safe_refresh_concurrent_succeeds_no_fallback():
    """When CONCURRENT succeeds, fallback is NOT called."""
    calls = []

    class FakeConn:
        autocommit = False

        def execute(self, sql):
            calls.append(sql)

    conn = FakeConn()

    def safe_refresh_sim(conn, view):
        conn.autocommit = True
        try:
            conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
        except Exception:
            conn.execute(f"REFRESH MATERIALIZED VIEW {view}")

    safe_refresh_sim(conn, "mv_intramonth_stockout")

    assert len(calls) == 1
    assert "CONCURRENTLY" in calls[0]
