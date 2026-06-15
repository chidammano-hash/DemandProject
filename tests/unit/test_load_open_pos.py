"""Characterization tests for scripts/etl/load_open_pos.py.

US1 (data-ingestion streamlining): pins the validation/date-parse behavior
BEFORE US11 (COPY/executemany) and US13 (transaction isolation) rewrite the
loader internals. These guard parity through that refactor.
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import scripts.etl.load_open_pos as lop
from scripts.etl.load_open_pos import (
    _parse_date,
    load_pos,
    load_receipts,
    load_suppliers,
    reconcile_received_qty,
    validate_po_row,
)

_CONFIG = {"validation": {"reject_pos_past_due_days": 30}}


def _conn_with_cursor():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    return conn, cur


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


# ---------------------------------------------------------------------------
# US11: batched (executemany) inserts replace row-by-row iterrows()
# ---------------------------------------------------------------------------

def _supplier_df(n=2):
    return pd.DataFrame([
        {
            "supplier_id": f"S{i}", "supplier_name": f"Name{i}",
            "country_code": "US", "address_line1": "1 St", "city": "C",
            "state_province": "CA", "postal_code": "90001", "payment_terms": "NET30",
            "default_lead_time_days": "5", "reliability_score": "0.9",
            "on_time_pct": "95", "is_active": "true",
        }
        for i in range(n)
    ])


class TestLoadSuppliersBatched:
    def test_executemany_single_call_with_all_rows(self):
        conn, cur = _conn_with_cursor()
        with patch("scripts.etl.load_open_pos.pd.read_csv", return_value=_supplier_df(3)):
            n = load_suppliers("suppliers.csv", conn, dry_run=False)
        assert n == 3
        cur.executemany.assert_called_once()
        sql, params = cur.executemany.call_args.args
        assert "ON CONFLICT (supplier_id)" in sql
        assert len(params) == 3
        # US13: loaders no longer own the transaction — caller (main) commits.
        conn.commit.assert_not_called()

    def test_dry_run_no_writes(self):
        conn, cur = _conn_with_cursor()
        with patch("scripts.etl.load_open_pos.pd.read_csv", return_value=_supplier_df(2)):
            n = load_suppliers("suppliers.csv", conn, dry_run=True)
        assert n == 2
        cur.executemany.assert_not_called()
        cur.execute.assert_not_called()
        conn.commit.assert_not_called()


class TestLoadPosBatched:
    def _pos_df(self):
        return pd.DataFrame([
            {  # valid
                "po_number": "PO1", "po_line_number": "1", "item_id": "I1", "loc": "L1",
                "supplier_id": "S1", "po_date": "2026-01-01", "ordered_qty": "10",
                "confirmed_qty": "10", "received_qty": "2", "unit_cost": "5",
                "currency": "USD", "promised_delivery_date": "2099-01-01",
                "confirmed_delivery_date": "", "revised_delivery_date": "",
                "po_status": "open", "line_status": "open",
            },
            {  # invalid: open_qty zero -> skipped
                "po_number": "PO2", "po_line_number": "1", "item_id": "I2", "loc": "L1",
                "supplier_id": "S1", "po_date": "2026-01-01", "ordered_qty": "5",
                "confirmed_qty": "5", "received_qty": "5", "unit_cost": "5",
                "currency": "USD", "promised_delivery_date": "2099-01-01",
                "confirmed_delivery_date": "", "revised_delivery_date": "",
                "po_status": "open", "line_status": "open",
            },
        ])

    def test_batches_valid_skips_invalid(self):
        conn, cur = _conn_with_cursor()
        with patch("scripts.etl.load_open_pos.pd.read_csv", return_value=self._pos_df()):
            loaded, skipped, reasons = load_pos("po.csv", conn, dry_run=False, config=_CONFIG)
        assert loaded == 1
        assert skipped == 1
        assert reasons.get("open_qty_zero_or_negative") == 1
        cur.executemany.assert_called_once()
        sql, params = cur.executemany.call_args.args
        assert "ON CONFLICT (po_number, po_line_number)" in sql
        assert len(params) == 1


class TestLoadReceiptsBatched:
    def _receipt_df(self):
        return pd.DataFrame([
            {
                "receipt_number": "R1", "po_number": "PO1", "po_line_number": "1",
                "item_id": "I1", "loc": "L1", "received_qty": "2", "unit_cost": "5",
                "actual_receipt_date": "2026-01-05", "receipt_status": "posted",
            },
            {  # no receipt date -> skipped
                "receipt_number": "R2", "po_number": "PO1", "po_line_number": "2",
                "item_id": "I1", "loc": "L1", "received_qty": "1", "unit_cost": "5",
                "actual_receipt_date": "", "receipt_status": "posted",
            },
        ])

    def test_batches_rows_with_receipt_date(self):
        conn, cur = _conn_with_cursor()
        with patch("scripts.etl.load_open_pos.pd.read_csv", return_value=self._receipt_df()):
            n = load_receipts("rec.csv", conn, dry_run=False)
        assert n == 1
        cur.executemany.assert_called_once()
        sql, params = cur.executemany.call_args.args
        assert "ON CONFLICT (receipt_number, po_number, po_line_number)" in sql
        assert len(params) == 1


# ---------------------------------------------------------------------------
# US13: transaction isolation — loaders don't commit; main() is atomic
# ---------------------------------------------------------------------------


class TestTransactionIsolation:
    def test_loaders_do_not_commit(self):
        # Each loader must leave the commit to the caller's transaction.
        conn, _cur = _conn_with_cursor()
        with patch("scripts.etl.load_open_pos.pd.read_csv", return_value=_supplier_df(1)):
            load_suppliers("s.csv", conn, dry_run=False)
        conn.commit.assert_not_called()

    def test_reconcile_does_not_commit(self):
        conn, cur = _conn_with_cursor()
        cur.rowcount = 3
        reconcile_received_qty(conn)
        conn.commit.assert_not_called()

    def test_main_wraps_steps_in_single_transaction(self):
        conn = MagicMock()
        txn = MagicMock()
        conn.transaction.return_value.__enter__ = MagicMock(return_value=txn)
        conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
        with patch("sys.argv", ["load_open_pos.py"]), \
             patch("scripts.etl.load_open_pos.load_config", return_value={}), \
             patch("scripts.etl.load_open_pos.get_db_params", return_value={}), \
             patch("psycopg.connect") as mock_connect, \
             patch("scripts.etl.load_open_pos._execute_load", return_value=5) as mock_exec, \
             patch("scripts.etl.load_open_pos._record_open_po_audit") as mock_audit:
            mock_connect.return_value.__enter__ = MagicMock(return_value=conn)
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            lop.main()
        conn.transaction.assert_called_once()
        mock_exec.assert_called_once()
        mock_audit.assert_called_once_with("completed", 5)

    def test_main_records_failed_batch_and_reraises_on_error(self):
        conn = MagicMock()
        conn.transaction.return_value.__enter__ = MagicMock(return_value=MagicMock())
        conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
        with patch("sys.argv", ["load_open_pos.py"]), \
             patch("scripts.etl.load_open_pos.load_config", return_value={}), \
             patch("scripts.etl.load_open_pos.get_db_params", return_value={}), \
             patch("psycopg.connect") as mock_connect, \
             patch("scripts.etl.load_open_pos._execute_load", side_effect=ValueError("boom")), \
             patch("scripts.etl.load_open_pos._record_open_po_audit") as mock_audit:
            mock_connect.return_value.__enter__ = MagicMock(return_value=conn)
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(ValueError):
                lop.main()
        # failed batch recorded with status "failed"
        assert mock_audit.call_args.args[0] == "failed"
