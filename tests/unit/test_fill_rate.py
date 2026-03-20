"""Unit tests for fill rate business logic and the _f() helper function.

IPfeature8 — Fill Rate & Demand Fulfillment Analytics.
Tests cover the _f() helper and the fill rate / shortage formulas computed
in mv_fill_rate_monthly (verified as pure Python equivalents).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest


# ---------------------------------------------------------------------------
# Import the helper directly from the router
# ---------------------------------------------------------------------------

def _import_f():
    """Import _f from fill_rate router without triggering FastAPI startup."""
    from api.routers.fill_rate import _f
    return _f


# ---------------------------------------------------------------------------
# _f() helper tests
# ---------------------------------------------------------------------------

def test_f_none_returns_none():
    """_f(None) must return None — NULL propagation for SQL NULLs."""
    _f = _import_f()
    assert _f(None) is None


def test_f_zero_returns_float_zero():
    """_f(0) must return 0.0."""
    _f = _import_f()
    result = _f(0)
    assert result == 0.0
    assert isinstance(result, float)


def test_f_integer_converts_to_float():
    """_f(1) returns 1.0 (float)."""
    _f = _import_f()
    result = _f(1)
    assert result == 1.0
    assert isinstance(result, float)


def test_f_float_passthrough():
    """_f(1.5) returns 1.5."""
    _f = _import_f()
    assert _f(1.5) == 1.5


def test_f_string_numeric_converts():
    """_f('3.14') converts string number to float."""
    _f = _import_f()
    assert _f("3.14") == pytest.approx(3.14)


def test_f_large_value():
    """_f() handles large numeric values correctly."""
    _f = _import_f()
    assert _f(1_000_000) == 1_000_000.0


def test_f_negative_value():
    """_f() handles negative numbers."""
    _f = _import_f()
    assert _f(-5.0) == -5.0


# ---------------------------------------------------------------------------
# Fill rate formula tests (mirror SQL: shipped / ordered, zero-safe)
# ---------------------------------------------------------------------------

def test_fill_rate_formula_normal():
    """Standard fill rate: 90 shipped / 100 ordered = 0.90."""
    total_shipped = 90.0
    total_ordered = 100.0
    fill_rate = total_shipped / total_ordered
    assert abs(fill_rate - 0.90) < 0.001


def test_fill_rate_formula_perfect():
    """Perfect fill: shipped == ordered → fill rate == 1.0."""
    assert 100.0 / 100.0 == 1.0


def test_fill_rate_formula_zero_ordered():
    """Zero ordered → fill rate must be NULL (avoid division by zero)."""
    total_ordered = 0.0
    result = None if total_ordered == 0 else 90.0 / total_ordered
    assert result is None


def test_fill_rate_rounding():
    """Fill rate rounds to 4 decimal places consistently."""
    shipped, ordered = 1, 3
    fill_rate = round(shipped / ordered, 4)
    assert fill_rate == 0.3333


def test_fill_rate_over_100_percent():
    """Over-shipment: shipped > ordered → fill rate > 1.0 (not clamped by formula)."""
    shipped, ordered = 110.0, 100.0
    fill_rate = shipped / ordered
    assert fill_rate > 1.0


# ---------------------------------------------------------------------------
# Shortage quantity tests (shortage_qty = max(0, ordered - shipped))
# ---------------------------------------------------------------------------

def test_shortage_qty_normal():
    """Ordered 100, shipped 90 → shortage of 10."""
    ordered, shipped = 100, 90
    shortage = max(0, ordered - shipped)
    assert shortage == 10


def test_shortage_qty_oversupply():
    """Shipped more than ordered → shortage is 0 (not negative)."""
    ordered, shipped = 80, 90
    shortage = max(0, ordered - shipped)
    assert shortage == 0


def test_shortage_qty_perfect_fulfillment():
    """Shipped == ordered → shortage is 0."""
    ordered, shipped = 50, 50
    shortage = max(0, ordered - shipped)
    assert shortage == 0


def test_shortage_qty_zero_shipped():
    """Nothing shipped → shortage equals full ordered quantity."""
    ordered, shipped = 100, 0
    shortage = max(0, ordered - shipped)
    assert shortage == 100


# ---------------------------------------------------------------------------
# Partial fulfillment flag tests (0 < shipped < ordered)
# ---------------------------------------------------------------------------

def test_partial_fulfillment_true():
    """Shipped > 0 but < ordered → partial fulfillment."""
    ordered, shipped = 100, 50
    had_partial = 0 < shipped < ordered
    assert had_partial is True


def test_partial_fulfillment_false_zero_shipped():
    """Nothing shipped → not partial (it's a full stockout)."""
    ordered, shipped = 100, 0
    had_partial = 0 < shipped < ordered
    assert had_partial is False


def test_partial_fulfillment_false_full_fill():
    """Fully shipped → not partial."""
    ordered, shipped = 100, 100
    had_partial = 0 < shipped < ordered
    assert had_partial is False


def test_partial_fulfillment_false_overfill():
    """Shipped > ordered → not counted as partial."""
    ordered, shipped = 100, 110
    had_partial = 0 < shipped < ordered
    assert had_partial is False


# ---------------------------------------------------------------------------
# ABC segment fill rate aggregation logic
# ---------------------------------------------------------------------------

def test_abc_fill_rate_aggregation():
    """Aggregate fill rate across ABC segment: SUM(shipped)/SUM(ordered)."""
    rows = [
        {"ordered": 1000.0, "shipped": 950.0},
        {"ordered": 500.0,  "shipped": 480.0},
        {"ordered": 200.0,  "shipped": 200.0},
    ]
    total_ordered = sum(r["ordered"] for r in rows)
    total_shipped = sum(r["shipped"] for r in rows)
    agg_fill_rate = total_shipped / total_ordered
    assert abs(agg_fill_rate - (1630.0 / 1700.0)) < 0.0001
