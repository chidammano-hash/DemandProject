"""
F2.4 — Unit tests for release_planned_orders.py

Tests pure functions: _map_fields, generate_po_number_format, export_pos_to_csv column order.
"""

import sys
import os
import csv
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import scripts.inventory.release_planned_orders as rpo


# ---------------------------------------------------------------------------
# _map_fields
# ---------------------------------------------------------------------------

class TestMapFields:
    def _po_tuple(self):
        return (
            "DS-2026-04-001",  # po_number
            "100320",           # item_id
            "1401-BULK",        # loc
            "SUP-4821",         # supplier_id
            316.0,              # ordered_qty
            "EA",               # unit_of_measure
            24.00,              # unit_cost
            "USD",              # currency
            date(2026, 4, 28),  # requested_delivery_date
            date(2026, 4, 15),  # po_date
            "JSMITH",           # buyer_code
            "COMP001",          # company_code
            "PLT01",            # plant_code
        )

    def test_identity_mapping(self):
        """No field_mapping → DS field names preserved."""
        result = rpo._map_fields(self._po_tuple(), {})
        assert result["po_number"] == "DS-2026-04-001"
        assert result["item_id"] == "100320"
        assert result["ordered_qty"] == 316.0

    def test_sap_field_mapping(self):
        """SAP BAPI field names applied correctly."""
        mapping = {
            "item_id": "MATNR",
            "loc": "WERKS",
            "supplier_id": "LIFNR",
            "ordered_qty": "MENGE",
            "unit_of_measure": "MEINS",
            "unit_cost": "NETPR",
            "currency": "WAERS",
            "requested_delivery_date": "EINDT",
            "po_date": "BEDAT",
            "buyer_code": "EKGRP",
            "company_code": "BUKRS",
        }
        result = rpo._map_fields(self._po_tuple(), mapping)
        assert result["MATNR"] == "100320"
        assert result["WERKS"] == "1401-BULK"
        assert result["LIFNR"] == "SUP-4821"
        assert result["MENGE"] == 316.0
        assert result["WAERS"] == "USD"

    def test_date_fields_serialized_to_string(self):
        """Date fields should be converted to string."""
        result = rpo._map_fields(self._po_tuple(), {})
        assert isinstance(result["requested_delivery_date"], str)
        assert result["requested_delivery_date"] == "2026-04-28"

    def test_none_fields_excluded(self):
        """None values should be excluded from the mapped payload."""
        # Plant code is None in this tuple (replace last field)
        po = list(self._po_tuple())
        po[-1] = None  # plant_code = None
        result = rpo._map_fields(tuple(po), {})
        assert "plant_code" not in result

    def test_partial_mapping(self):
        """Only mapped fields are renamed; unmapped fields keep DS names."""
        mapping = {"item_id": "MATNR"}
        result = rpo._map_fields(self._po_tuple(), mapping)
        assert "MATNR" in result
        assert "item_id" not in result
        assert "loc" in result  # unmapped field kept


# ---------------------------------------------------------------------------
# generate_po_number format validation (no actual DB needed)
# ---------------------------------------------------------------------------

class TestGeneratePoNumberFormat:
    def test_format_matches_pattern(self):
        """
        Validate format DS-{YYYY}-{MM}-{NNN} without a real DB call.
        We test by inspecting the return value from a mocked connection.
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (42,)

        po_number = rpo.generate_po_number(mock_conn)

        # Format: DS-{4digit_year}-{2digit_month}-{3digit_seq}
        parts = po_number.split("-")
        assert len(parts) == 4
        assert parts[0] == "DS"
        assert len(parts[1]) == 4 and parts[1].isdigit()   # YYYY
        assert len(parts[2]) == 2 and parts[2].isdigit()   # MM
        assert len(parts[3]) == 3 and parts[3].isdigit()   # NNN

    def test_sequence_value_embedded(self):
        """The sequence value is correctly zero-padded in the PO number."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (7,)

        po_number = rpo.generate_po_number(mock_conn)
        assert po_number.endswith("-007")

    def test_sequence_100_padded_correctly(self):
        """Sequence value >= 100 should still be 3 digits."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (100,)

        po_number = rpo.generate_po_number(mock_conn)
        assert po_number.endswith("-100")


# ---------------------------------------------------------------------------
# FIELDNAMES column order (ERP import format)
# ---------------------------------------------------------------------------

class TestFieldnamesOrder:
    def test_po_number_is_first_field(self):
        assert rpo.FIELDNAMES[0] == "PO_NUMBER"

    def test_line_no_is_second_field(self):
        assert rpo.FIELDNAMES[1] == "LINE_NO"

    def test_demand_studio_exception_id_is_second_to_last(self):
        assert rpo.FIELDNAMES[-2] == "DEMAND_STUDIO_EXCEPTION_ID"

    def test_notes_is_last_field(self):
        assert rpo.FIELDNAMES[-1] == "NOTES"

    def test_all_required_erp_fields_present(self):
        """ERP import format must include all required fields."""
        required = {
            "PO_NUMBER", "LINE_NO", "ITEM_NUMBER", "LOCATION",
            "SUPPLIER_ID", "ORDERED_QTY", "CURRENCY",
            "REQUESTED_DELIVERY_DATE", "PO_DATE",
        }
        assert required.issubset(set(rpo.FIELDNAMES))


# ---------------------------------------------------------------------------
# export_pos_to_csv
# ---------------------------------------------------------------------------

class TestExportPosToCsv:
    def _make_rows(self, n: int = 4) -> list:
        return [
            (f"DS-2026-04-001", i + 1, f"ITEM{i:03d}", f"Desc {i}",
             "LOC1", "SUP-001", "Supplier A", 100.0 * (i + 1),
             "EA", 10.0, 1000.0 * (i + 1), "USD",
             date(2026, 4, 28), date(2026, 4, 15),
             "BOB", "COMP001", "PLT01", 7834 + i, f"Note {i}")
            for i in range(n)
        ]

    def test_csv_line_count(self, tmp_path):
        """4 exception rows → 4 CSV data lines (plus header)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = self._make_rows(4)

        output_path = tmp_path / "test_export.csv"
        n = rpo.export_pos_to_csv(["DS-2026-04-001"], output_path, mock_conn)

        assert n == 4
        assert output_path.exists()

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4

    def test_csv_headers_match_fieldnames(self, tmp_path):
        """CSV headers must match FIELDNAMES exactly."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = self._make_rows(1)

        output_path = tmp_path / "test_export.csv"
        rpo.export_pos_to_csv(["DS-2026-04-001"], output_path, mock_conn)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert headers == rpo.FIELDNAMES

    def test_export_empty_result(self, tmp_path):
        """Empty result → 0 lines, empty CSV (just header)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        output_path = tmp_path / "empty_export.csv"
        n = rpo.export_pos_to_csv(["NONESUCH"], output_path, mock_conn)

        assert n == 0
