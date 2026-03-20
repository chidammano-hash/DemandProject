"""Unit tests for load_open_pos.py — F1.3 Open PO Integration."""

from __future__ import annotations

import sys
import os
from datetime import date, timedelta

from common.planning_date import get_planning_date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(past_due_days: int = 180, require_confirmed: bool = False) -> dict:
    return {
        "validation": {
            "reject_pos_past_due_days": past_due_days,
            "require_confirmed_delivery_date": require_confirmed,
            "max_open_qty_ratio": 10.0,
        },
        "data_quality": {
            "past_due_threshold_days": 7,
            "delivery_date_tolerance_days": 2,
        },
    }


def _future(days: int = 30) -> date:
    return get_planning_date() + timedelta(days=days)


def _past(days: int = 30) -> date:
    return get_planning_date() - timedelta(days=days)


# ---------------------------------------------------------------------------
# validate_po_row — pure function tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_config(tmp_path, monkeypatch):
    """Patch open() for CONFIG_PATH so validate_po_row doesn't need the file."""
    # validate_po_row receives config as a param, no file needed
    yield


def _validate(row: dict, past_due_days: int = 180):
    from scripts.load_open_pos import validate_po_row
    return validate_po_row(row, _make_config(past_due_days=past_due_days))


class TestValidatePoRow:
    def test_valid_open_po(self):
        row = {
            "promised_delivery_date": _future(30),
            "confirmed_delivery_date": _future(30),
            "revised_delivery_date": None,
            "line_status": "open",
            "ordered_qty": 150.0,
            "confirmed_qty": 150.0,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row)
        assert valid is True
        assert reason == "ok"

    def test_no_delivery_date_rejected(self):
        row = {
            "promised_delivery_date": None,
            "confirmed_delivery_date": None,
            "revised_delivery_date": None,
            "line_status": "open",
            "ordered_qty": 100.0,
            "confirmed_qty": 100.0,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row)
        assert valid is False
        assert reason == "no_delivery_date"

    def test_closed_line_rejected(self):
        row = {
            "promised_delivery_date": _future(10),
            "confirmed_delivery_date": _future(10),
            "revised_delivery_date": None,
            "line_status": "closed",
            "ordered_qty": 100.0,
            "confirmed_qty": 100.0,
            "received_qty": 100.0,
        }
        valid, reason = _validate(row)
        assert valid is False
        assert reason == "line_closed_or_cancelled"

    def test_cancelled_line_rejected(self):
        row = {
            "promised_delivery_date": _future(10),
            "confirmed_delivery_date": None,
            "revised_delivery_date": None,
            "line_status": "cancelled",
            "ordered_qty": 100.0,
            "confirmed_qty": None,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row)
        assert valid is False
        assert reason == "line_closed_or_cancelled"

    def test_too_far_past_due_rejected(self):
        row = {
            "promised_delivery_date": _past(200),
            "confirmed_delivery_date": _past(200),
            "revised_delivery_date": None,
            "line_status": "open",
            "ordered_qty": 100.0,
            "confirmed_qty": 100.0,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row, past_due_days=180)
        assert valid is False
        assert "past_due_exceeds" in reason

    def test_past_due_within_threshold_accepted(self):
        row = {
            "promised_delivery_date": _past(10),
            "confirmed_delivery_date": _past(10),
            "revised_delivery_date": None,
            "line_status": "open",
            "ordered_qty": 100.0,
            "confirmed_qty": 100.0,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row, past_due_days=180)
        assert valid is True

    def test_zero_open_qty_rejected(self):
        row = {
            "promised_delivery_date": _future(10),
            "confirmed_delivery_date": _future(10),
            "revised_delivery_date": None,
            "line_status": "open",
            "ordered_qty": 100.0,
            "confirmed_qty": 100.0,
            "received_qty": 100.0,  # fully received → open_qty = 0
        }
        valid, reason = _validate(row)
        assert valid is False
        assert reason == "open_qty_zero_or_negative"

    def test_negative_open_qty_rejected(self):
        row = {
            "promised_delivery_date": _future(10),
            "confirmed_delivery_date": _future(10),
            "revised_delivery_date": None,
            "line_status": "partially_received",
            "ordered_qty": 100.0,
            "confirmed_qty": None,
            "received_qty": 110.0,  # over-received
        }
        valid, reason = _validate(row)
        assert valid is False
        assert reason == "open_qty_zero_or_negative"

    def test_partially_received_valid(self):
        row = {
            "promised_delivery_date": _future(5),
            "confirmed_delivery_date": _future(5),
            "revised_delivery_date": None,
            "line_status": "partially_received",
            "ordered_qty": 1000.0,
            "confirmed_qty": 900.0,
            "received_qty": 100.0,  # open_qty = 800
        }
        valid, reason = _validate(row)
        assert valid is True

    def test_only_promised_date_is_enough(self):
        """A PO with only promised_delivery_date (no confirmed) should be accepted."""
        row = {
            "promised_delivery_date": _future(14),
            "confirmed_delivery_date": None,
            "revised_delivery_date": None,
            "line_status": "open",
            "ordered_qty": 50.0,
            "confirmed_qty": None,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row)
        assert valid is True

    def test_revised_date_used_for_past_due_check(self):
        """When revised_delivery_date is set, it should be used for the past-due check."""
        row = {
            "promised_delivery_date": _past(200),   # very old promised date
            "confirmed_delivery_date": _past(200),
            "revised_delivery_date": _future(7),    # revised to future → not past due
            "line_status": "open",
            "ordered_qty": 100.0,
            "confirmed_qty": 100.0,
            "received_qty": 0.0,
        }
        valid, reason = _validate(row, past_due_days=180)
        assert valid is True


# ---------------------------------------------------------------------------
# _parse_date helper
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_valid_string(self):
        from scripts.load_open_pos import _parse_date
        result = _parse_date("2026-03-14")
        assert result == date(2026, 3, 14)

    def test_none_returns_none(self):
        from scripts.load_open_pos import _parse_date
        assert _parse_date(None) is None

    def test_empty_string_returns_none(self):
        from scripts.load_open_pos import _parse_date
        assert _parse_date("") is None

    def test_date_object_passthrough(self):
        from scripts.load_open_pos import _parse_date
        d = date(2026, 3, 1)
        assert _parse_date(d) == d

    def test_invalid_string_returns_none(self):
        from scripts.load_open_pos import _parse_date
        assert _parse_date("not-a-date") is None
