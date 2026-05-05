"""Unit tests for ``common.core.domain_partition``.

Pure-function tests — no DB, no mocks needed.
"""
from __future__ import annotations

from datetime import date

import pytest

from common.core.domain_partition import (
    PARTITION_SPECS,
    DomainPartition,
    get_partition,
    is_partitioned,
    slice_to_date_range,
)


# ---------------------------------------------------------------------------
# PARTITION_SPECS registry
# ---------------------------------------------------------------------------
def test_partition_specs_has_expected_domains():
    """All four partitioned domains must be registered."""
    assert set(PARTITION_SPECS.keys()) == {
        "customer_demand",
        "inventory",
        "forecast",
        "sales",
    }


def test_inventory_partition_has_file_glob():
    """Inventory uses a file_glob for per-file partition discovery."""
    spec = PARTITION_SPECS["inventory"]
    assert isinstance(spec, DomainPartition)
    assert spec.file_glob == "Inventory_Snapshot_*.csv"
    assert spec.field == "snapshot_date"
    assert spec.format == "YYYY_MM"


def test_customer_demand_partition_fields():
    spec = PARTITION_SPECS["customer_demand"]
    assert spec.field == "startdate"
    assert spec.format == "YYYY-MM"
    assert spec.file_glob is None


# ---------------------------------------------------------------------------
# get_partition / is_partitioned
# ---------------------------------------------------------------------------
def test_get_partition_returns_none_for_unpartitioned():
    """Unpartitioned domains (e.g. dimension tables) return None."""
    assert get_partition("item") is None
    assert get_partition("location") is None
    assert get_partition("nonexistent_domain") is None


def test_get_partition_returns_dataclass_for_partitioned():
    spec = get_partition("sales")
    assert isinstance(spec, DomainPartition)
    assert spec.field == "startdate"


@pytest.mark.parametrize(
    "domain,expected",
    [
        ("customer_demand", True),
        ("inventory", True),
        ("forecast", True),
        ("sales", True),
        ("item", False),
        ("location", False),
        ("customer", False),
        ("time", False),
        ("sku", False),
        ("sourcing", False),
        ("purchase_order", False),
    ],
)
def test_is_partitioned_true_false(domain, expected):
    assert is_partitioned(domain) is expected


# ---------------------------------------------------------------------------
# slice_to_date_range — happy paths
# ---------------------------------------------------------------------------
def test_slice_to_date_range_yyyy_mm():
    start, end = slice_to_date_range("2026-03", "YYYY-MM")
    assert start == date(2026, 3, 1)
    assert end == date(2026, 4, 1)


def test_slice_to_date_range_yyyy_mm_underscore():
    start, end = slice_to_date_range("2026_03", "YYYY_MM")
    assert start == date(2026, 3, 1)
    assert end == date(2026, 4, 1)


def test_slice_to_date_range_yyyymm():
    start, end = slice_to_date_range("202603", "YYYYMM")
    assert start == date(2026, 3, 1)
    assert end == date(2026, 4, 1)


def test_slice_to_date_range_yyyy_mm_dd():
    start, end = slice_to_date_range("2026-03-15", "YYYY-MM-DD")
    assert start == date(2026, 3, 15)
    assert end == date(2026, 3, 16)


def test_slice_to_date_range_year_rollover():
    """December rolls over to January of the next year."""
    start, end = slice_to_date_range("2026-12", "YYYY-MM")
    assert start == date(2026, 12, 1)
    assert end == date(2027, 1, 1)


def test_slice_to_date_range_year_rollover_underscore():
    start, end = slice_to_date_range("2026_12", "YYYY_MM")
    assert start == date(2026, 12, 1)
    assert end == date(2027, 1, 1)


def test_slice_to_date_range_yyyymm_year_rollover():
    start, end = slice_to_date_range("202612", "YYYYMM")
    assert end == date(2027, 1, 1)


def test_slice_to_date_range_day_rollover_end_of_month():
    """A day-grain slice on the last day of a month rolls into the next month."""
    start, end = slice_to_date_range("2026-01-31", "YYYY-MM-DD")
    assert start == date(2026, 1, 31)
    assert end == date(2026, 2, 1)


def test_slice_to_date_range_strips_whitespace():
    start, end = slice_to_date_range("  2026-03  ", "YYYY-MM")
    assert start == date(2026, 3, 1)


# ---------------------------------------------------------------------------
# slice_to_date_range — error cases
# ---------------------------------------------------------------------------
def test_slice_invalid_raises():
    with pytest.raises(ValueError):
        slice_to_date_range("badformat", "YYYY-MM")


def test_slice_unknown_format_raises():
    with pytest.raises(ValueError):
        slice_to_date_range("2026-03", "WHATEVER")


@pytest.mark.parametrize(
    "bad_input,fmt",
    [
        ("2026/03", "YYYY-MM"),       # wrong separator
        ("26-03", "YYYY-MM"),         # too short
        ("2026-3", "YYYY-MM"),        # missing zero pad
        ("2026-13", "YYYY-MM"),       # invalid month
        ("2026-00", "YYYY-MM"),       # month 0
        # Note: "2026-03" with format "YYYY_MM" is now ACCEPTED (lenient
        # separator normalization for common user typo). Removed from this list.
        ("2026_03_15", "YYYY_MM"),    # too long
        ("20263", "YYYYMM"),          # too short
        ("2026-03-1", "YYYY-MM-DD"),  # missing day pad
        ("abcdef", "YYYYMM"),         # non-numeric
    ],
)
def test_slice_malformed_inputs_raise(bad_input, fmt):
    with pytest.raises(ValueError):
        slice_to_date_range(bad_input, fmt)


@pytest.mark.parametrize(
    "user_input,fmt,expected_start",
    [
        # Hyphen typed when underscore expected (and vice versa) — auto-normalized.
        ("2026-04", "YYYY_MM", date(2026, 4, 1)),
        ("2026_04", "YYYY-MM", date(2026, 4, 1)),
    ],
)
def test_slice_separator_leniency(user_input, fmt, expected_start):
    start, end = slice_to_date_range(user_input, fmt)
    assert start == expected_start
    assert end == date(2026, 5, 1)


def test_slice_to_date_range_returns_dates_not_datetimes():
    """Both ends must be ``datetime.date`` instances (not datetime)."""
    start, end = slice_to_date_range("2026-03", "YYYY-MM")
    assert type(start) is date
    assert type(end) is date


def test_slice_to_date_range_end_is_exclusive():
    """End is the first day of the *next* period (exclusive)."""
    _, end = slice_to_date_range("2026-03", "YYYY-MM")
    # End belongs to April, not March.
    assert end.month == 4
    assert end.day == 1
