"""Tests for scripts/populate_dq_checks.py — DQ check catalog population."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.populate_dq_checks import (
    _completeness_template,
    _domain_from_table,
    _range_template,
    _referential_integrity_template,
    _uniqueness_template,
    _volume_delta_template,
    parse_checks,
    upsert_checks,
)


# ---------------------------------------------------------------------------
# SQL template tests
# ---------------------------------------------------------------------------

class TestCompletenessTemplate:
    def test_basic(self):
        sql = _completeness_template("dim_item", "brand")
        assert "dim_item" in sql
        assert "brand IS NULL" in sql
        assert "null_pct" in sql

    def test_different_column(self):
        sql = _completeness_template("fact_sales_monthly", "qty")
        assert "qty IS NULL" in sql


class TestUniquenessTemplate:
    def test_single_key(self):
        sql = _uniqueness_template("dim_item", ["item_id"])
        assert "item_id" in sql
        assert "HAVING COUNT(*) > 1" in sql

    def test_composite_key(self):
        sql = _uniqueness_template("dim_sku", ["item_id", "loc"])
        assert "item_id, loc" in sql
        assert "GROUP BY item_id, loc" in sql


class TestRangeTemplate:
    def test_basic(self):
        sql = _range_template("fact_sales_monthly", "qty", 0, 10000000)
        assert "qty < 0" in sql
        assert "qty > 10000000" in sql
        assert "out_of_range" in sql


class TestVolumeDeltaTemplate:
    def test_basic(self):
        sql = _volume_delta_template("fact_sales_monthly")
        assert "COUNT(*)" in sql
        assert "fact_sales_monthly" in sql


class TestReferentialIntegrityTemplate:
    def test_single_column(self):
        sql = _referential_integrity_template(
            "fact_inventory_snapshot", ["item_id"],
            "dim_item", ["item_id"],
        )
        assert "fact_inventory_snapshot s" in sql
        assert "dim_item t" in sql
        assert "s.item_id = t.item_id" in sql
        assert "t.item_id IS NULL" in sql

    def test_composite_columns(self):
        sql = _referential_integrity_template(
            "fact_sales_monthly", ["item_id", "loc"],
            "dim_sku", ["item_id", "loc"],
        )
        assert "s.item_id = t.item_id" in sql
        assert "s.loc = t.loc" in sql


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

class TestDomainFromTable:
    def test_sales(self):
        assert _domain_from_table("fact_sales_monthly") == "sales"

    def test_forecast(self):
        assert _domain_from_table("fact_external_forecast_monthly") == "forecast"

    def test_inventory(self):
        assert _domain_from_table("fact_inventory_snapshot") == "inventory"

    def test_dim_sku(self):
        assert _domain_from_table("dim_sku") == "sku"

    def test_dim_item(self):
        assert _domain_from_table("dim_item") == "item"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "global_defaults": {"severity": "warning", "enabled": True},
    "checks": {
        "completeness": {
            "item": {
                "table": "dim_item",
                "columns": [
                    {"column": "item_id", "null_pct_threshold": 0.0, "severity": "critical"},
                    {"column": "brand", "null_pct_threshold": 10.0, "severity": "warning"},
                ],
            },
        },
        "uniqueness": {
            "sku": {"table": "dim_sku", "key_columns": ["item_id", "loc"], "severity": "critical"},
        },
        "range": {
            "sales": {
                "table": "fact_sales_monthly",
                "columns": [
                    {"column": "qty", "min": 0, "max": 10000000, "severity": "warning"},
                ],
            },
        },
        "volume_delta": {
            "sales": {"table": "fact_sales_monthly", "max_pct_change": 50.0, "severity": "warning"},
        },
        "referential_integrity": {
            "sales_to_dfu": {
                "source_table": "fact_sales_monthly",
                "source_columns": ["item_id", "loc"],
                "target_table": "dim_sku",
                "target_columns": ["item_id", "loc"],
                "severity": "warning",
            },
        },
    },
}


class TestParseChecks:
    def test_total_count(self):
        checks = parse_checks(MINIMAL_CONFIG)
        # 2 completeness + 1 uniqueness + 1 range + 1 volume_delta + 1 RI = 6
        assert len(checks) == 6

    def test_table_name_included(self):
        checks = parse_checks(MINIMAL_CONFIG)
        for chk in checks:
            assert "table_name" in chk, f"Missing table_name in {chk['check_name']}"
            assert chk["table_name"] is not None

    def test_completeness_checks(self):
        checks = parse_checks(MINIMAL_CONFIG)
        comp = [c for c in checks if c["check_type"] == "completeness"]
        assert len(comp) == 2
        names = {c["check_name"] for c in comp}
        assert names == {"completeness_item_item_id", "completeness_item_brand"}

    def test_uniqueness_check(self):
        checks = parse_checks(MINIMAL_CONFIG)
        uniq = [c for c in checks if c["check_type"] == "uniqueness"]
        assert len(uniq) == 1
        assert uniq[0]["check_name"] == "uniqueness_sku"
        assert uniq[0]["threshold"] == 0

    def test_range_check(self):
        checks = parse_checks(MINIMAL_CONFIG)
        rng = [c for c in checks if c["check_type"] == "range"]
        assert len(rng) == 1
        assert rng[0]["check_name"] == "range_sales_qty"

    def test_volume_delta_check(self):
        checks = parse_checks(MINIMAL_CONFIG)
        vol = [c for c in checks if c["check_type"] == "volume_delta"]
        assert len(vol) == 1
        assert vol[0]["check_name"] == "volume_delta_sales"
        assert vol[0]["threshold"] == 50.0

    def test_ri_check(self):
        checks = parse_checks(MINIMAL_CONFIG)
        ri = [c for c in checks if c["check_type"] == "referential_integrity"]
        assert len(ri) == 1
        assert ri[0]["check_name"] == "ri_sales_to_dfu"
        assert ri[0]["domain"] == "sales"

    def test_severity_preserved(self):
        checks = parse_checks(MINIMAL_CONFIG)
        comp_id = [c for c in checks if c["check_name"] == "completeness_item_item_id"][0]
        assert comp_id["severity"] == "critical"

    def test_enabled_default(self):
        checks = parse_checks(MINIMAL_CONFIG)
        for chk in checks:
            assert chk["enabled"] is True

    def test_empty_config(self):
        checks = parse_checks({})
        assert checks == []

    def test_full_config_parse(self):
        """Parse the real config file and verify all 5 check types present."""
        from common.utils import load_config, reset_config
        reset_config(None)
        config = load_config("data_quality_config.yaml")
        checks = parse_checks(config)
        types_found = {c["check_type"] for c in checks}
        assert types_found == {
            "completeness", "uniqueness",
            "range", "volume_delta", "referential_integrity",
        }
        # Verify all check names are unique
        names = [c["check_name"] for c in checks]
        assert len(names) == len(set(names)), "Duplicate check_name found"


# ---------------------------------------------------------------------------
# Upsert logic
# ---------------------------------------------------------------------------

class TestUpsertChecks:
    def _make_conn(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return conn, cursor

    def test_upsert_executes_sql(self):
        conn, cursor = self._make_conn()
        checks = [
            {
                "check_name": "completeness_item_item_id",
                "check_type": "completeness",
                "domain": "item",
                "sql_template": "SELECT 1",
                "threshold": 0.0,
                "severity": "warning",
                "enabled": True,
            },
        ]
        count = upsert_checks(conn, checks, dry_run=False)
        assert count == 1
        cursor.executemany.assert_called_once()
        conn.commit.assert_called_once()

    def test_dry_run_skips_execute(self):
        conn, cursor = self._make_conn()
        checks = [
            {
                "check_name": "completeness_item_item_id",
                "check_type": "completeness",
                "domain": "item",
                "sql_template": "SELECT 1",
                "threshold": 0.0,
                "severity": "warning",
                "enabled": True,
            },
        ]
        count = upsert_checks(conn, checks, dry_run=True)
        assert count == 1
        cursor.execute.assert_not_called()
        conn.commit.assert_not_called()

    def test_multiple_checks(self):
        conn, cursor = self._make_conn()
        checks = [
            {"check_name": f"check_{i}", "check_type": "completeness", "domain": "item",
             "sql_template": "SELECT 1", "threshold": 0.0, "severity": "warning", "enabled": True}
            for i in range(5)
        ]
        count = upsert_checks(conn, checks, dry_run=False)
        assert count == 5
        cursor.executemany.assert_called_once()
        # All 5 checks passed as a single batch
        assert len(cursor.executemany.call_args[0][1]) == 5

    def test_empty_checks_list(self):
        conn, cursor = self._make_conn()
        count = upsert_checks(conn, [], dry_run=False)
        assert count == 0
        cursor.execute.assert_not_called()


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------

class TestRun:
    @patch.dict("sys.modules", {"dotenv": MagicMock()})
    @patch("scripts.ops.populate_dq_checks.load_config")
    def test_run_dry_run(self, mock_load_config):
        mock_load_config.return_value = MINIMAL_CONFIG

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect") as mock_connect:
            mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            from scripts.ops.populate_dq_checks import run
            result = run(dry_run=True)

        assert result["dry_run"] is True
        assert result["total"] == 6
        assert "completeness" in result["by_type"]
        # dry_run should NOT execute any SQL
        mock_cursor.execute.assert_not_called()

    @patch.dict("sys.modules", {"dotenv": MagicMock()})
    @patch("scripts.ops.populate_dq_checks.load_config")
    def test_run_writes(self, mock_load_config):
        mock_load_config.return_value = MINIMAL_CONFIG

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect") as mock_connect:
            mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)

            from scripts.ops.populate_dq_checks import run
            result = run(dry_run=False)

        assert result["dry_run"] is False
        assert result["total"] == 6
        mock_cursor.executemany.assert_called_once()
        assert len(mock_cursor.executemany.call_args[0][1]) == 6
        mock_conn.commit.assert_called_once()
