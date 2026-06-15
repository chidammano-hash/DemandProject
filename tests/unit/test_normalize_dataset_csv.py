"""Characterization tests for scripts/etl/normalize_dataset_csv.py.

US1 (data-ingestion streamlining): pins the behavior of the normalize
helper functions BEFORE the Phase 1-4 refactors touch this file. These
assert current behavior exactly so a future change cannot drift silently.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl.normalize_dataset_csv import (
    dedupe_headers,
    month_diff,
    month_end,
    quarter_end,
    quarter_start,
    to_int_string,
    to_iso_date_yyyymmdd,
    to_iso_month_start,
)


class TestDedupeHeaders:
    def test_unique_headers_unchanged(self):
        assert dedupe_headers(["a", "b", "c"]) == ["a", "b", "c"]

    def test_duplicates_suffixed_from_second_occurrence(self):
        assert dedupe_headers(["a", "a", "a"]) == ["a", "a_2", "a_3"]

    def test_strips_whitespace(self):
        assert dedupe_headers([" a ", "a"]) == ["a", "a_2"]


class TestMonthEnd:
    def test_february_leap_year(self):
        assert month_end(date(2024, 2, 15)) == date(2024, 2, 29)

    def test_december_caps_at_31(self):
        assert month_end(date(2024, 12, 5)) == date(2024, 12, 31)

    def test_april_30(self):
        assert month_end(date(2025, 4, 10)) == date(2025, 4, 30)


class TestQuarterBoundaries:
    def test_quarter_start_q2(self):
        assert quarter_start(date(2024, 5, 3)) == date(2024, 4, 1)

    def test_quarter_start_q1(self):
        assert quarter_start(date(2024, 1, 31)) == date(2024, 1, 1)

    def test_quarter_end_q2(self):
        assert quarter_end(date(2024, 5, 3)) == date(2024, 6, 30)

    def test_quarter_end_q4_caps_at_dec31(self):
        assert quarter_end(date(2024, 11, 20)) == date(2024, 12, 31)


class TestToIsoDateYyyymmdd:
    def test_compact_format(self):
        assert to_iso_date_yyyymmdd("20240115") == "2024-01-15"

    def test_iso_format_passthrough(self):
        assert to_iso_date_yyyymmdd("2024-01-15") == "2024-01-15"

    def test_empty_returns_empty(self):
        assert to_iso_date_yyyymmdd("") == ""

    def test_invalid_returns_empty(self):
        assert to_iso_date_yyyymmdd("abc") == ""
        assert to_iso_date_yyyymmdd("20241301") == ""

    def test_require_month_start_compact(self):
        assert to_iso_date_yyyymmdd("20240201", require_month_start=True) == "2024-02-01"
        assert to_iso_date_yyyymmdd("20240215", require_month_start=True) == ""

    def test_require_month_start_iso(self):
        assert to_iso_date_yyyymmdd("2024-02-01", require_month_start=True) == "2024-02-01"
        assert to_iso_date_yyyymmdd("2024-02-15", require_month_start=True) == ""


class TestToIsoMonthStart:
    def test_iso_month_start(self):
        assert to_iso_month_start("2024-03-01") == "2024-03-01"

    def test_iso_non_month_start_rejected(self):
        assert to_iso_month_start("2024-03-15") == ""

    def test_compact_month_start(self):
        assert to_iso_month_start("20240301") == "2024-03-01"

    def test_compact_non_month_start_rejected(self):
        assert to_iso_month_start("20240315") == ""

    def test_empty(self):
        assert to_iso_month_start("") == ""


class TestMonthDiff:
    def test_positive_diff(self):
        assert month_diff("2024-03-01", "2024-01-01") == 2

    def test_cross_year_diff(self):
        assert month_diff("2025-01-01", "2024-11-01") == 2

    def test_zero_diff(self):
        assert month_diff("2024-06-01", "2024-06-01") == 0

    def test_invalid_returns_none(self):
        assert month_diff("not-a-date", "2024-01-01") is None


class TestToIntString:
    def test_float_string_truncated(self):
        assert to_int_string("5.0") == "5"

    def test_plain_int(self):
        assert to_int_string("3") == "3"

    def test_empty(self):
        assert to_int_string("") == ""

    def test_invalid(self):
        assert to_int_string("abc") == ""
