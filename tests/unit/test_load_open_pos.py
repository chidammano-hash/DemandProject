"""Characterization tests for scripts/etl/load_open_pos.py.

US1 (data-ingestion streamlining): pins the validation/date-parse behavior
BEFORE US11 (COPY/executemany) and US13 (transaction isolation) rewrite the
loader internals. These guard parity through that refactor.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl.load_open_pos import _parse_date, validate_po_row

_CONFIG = {"validation": {"reject_pos_past_due_days": 30}}


class TestParseDate:
    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_date_passthrough(self):
        d = date(2024, 1, 15)
        assert _parse_date(d) == d

    def test_iso_string_parsed(self):
        assert _parse_date("2024-01-15") == date(2024, 1, 15)

    def test_garbage_returns_none(self):
        assert _parse_date("not-a-date") is None


class TestValidatePoRow:
    def test_no_delivery_date_rejected(self):
        row = {"ordered_qty": 10, "received_qty": 0}
        ok, reason = validate_po_row(row, _CONFIG)
        assert ok is False
        assert reason == "no_delivery_date"

    def test_closed_line_rejected(self):
        row = {
            "promised_delivery_date": date(2099, 1, 1),
            "line_status": "closed",
            "ordered_qty": 10,
        }
        ok, reason = validate_po_row(row, _CONFIG)
        assert ok is False
        assert reason == "line_closed_or_cancelled"

    def test_cancelled_line_rejected(self):
        row = {
            "promised_delivery_date": date(2099, 1, 1),
            "line_status": "cancelled",
            "ordered_qty": 10,
        }
        ok, reason = validate_po_row(row, _CONFIG)
        assert ok is False
        assert reason == "line_closed_or_cancelled"

    def test_zero_open_qty_rejected(self):
        row = {
            "promised_delivery_date": date(2099, 1, 1),
            "ordered_qty": 5,
            "received_qty": 5,
        }
        ok, reason = validate_po_row(row, _CONFIG)
        assert ok is False
        assert reason == "open_qty_zero_or_negative"

    def test_valid_row_accepted(self):
        row = {
            "promised_delivery_date": date(2099, 1, 1),
            "line_status": "open",
            "ordered_qty": 10,
            "received_qty": 2,
        }
        ok, reason = validate_po_row(row, _CONFIG)
        assert ok is True
        assert reason == "ok"

    def test_confirmed_qty_preferred_over_ordered(self):
        # confirmed_qty(8) - received(8) == 0 -> rejected even though ordered is large
        row = {
            "promised_delivery_date": date(2099, 1, 1),
            "confirmed_qty": 8,
            "ordered_qty": 100,
            "received_qty": 8,
        }
        ok, reason = validate_po_row(row, _CONFIG)
        assert ok is False
        assert reason == "open_qty_zero_or_negative"
