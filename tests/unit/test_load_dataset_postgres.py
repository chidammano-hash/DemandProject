"""Tests for scripts/load_dataset_postgres.py — simplified loader."""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl.load_dataset_postgres import (
    _resolve_forecast_execution_lag,
    _load_forecast_archive,
    _is_partitioned,
    _ensure_partition_exists,
    load_domain,
)

# Index/constraint helpers (US3) and DFU/FK filters (US5) moved to
# common/core/etl_helpers.py; their tests live in tests/unit/test_etl_helpers.py.


# ---------- Helpers ----------

def _cycling_property(values):
    it = iter(values)
    return property(lambda self: next(it))


def _cursor_with_rowcounts(rowcounts):
    """Build a MagicMock cursor that cycles through rowcount values on execute."""
    cur = MagicMock()
    rc_iter = iter(rowcounts)

    def _on_execute(*a, **kw):
        cur.rowcount = next(rc_iter)

    cur.execute.side_effect = _on_execute
    return cur


# ---------- TestResolveForecastExecutionLag ----------

class TestResolveForecastExecutionLag:
    def test_dim_sku_exists_updates_lag_and_execution_lag(self):
        cur = _cursor_with_rowcounts([None, 500])
        cur.fetchone.side_effect = [(True,)]

        matched = _resolve_forecast_execution_lag(cur, "stg_fcst")

        assert matched == 500
        assert cur.execute.call_count == 2  # EXISTS + UPDATE
        update_sql = cur.execute.call_args_list[1][0][0]
        assert "dim_sku" in update_sql
        assert '"lag"' in update_sql
        assert '"execution_lag"' in update_sql

    def test_dim_sku_missing_returns_zero(self):
        cur = MagicMock()
        cur.fetchone.return_value = (False,)

        matched = _resolve_forecast_execution_lag(cur, "stg_fcst")

        assert matched == 0
        assert cur.execute.call_count == 1  # EXISTS only


# ---------- TestLoadForecastArchive ----------

class TestLoadForecastArchive:
    def test_deletes_old_and_inserts_new(self):
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([50, 8000])

        count = _load_forecast_archive(cur, "stg_fcst", "d")

        assert count == 8000
        # First call: DELETE
        del_sql = cur.execute.call_args_list[0][0][0]
        assert "DELETE" in del_sql
        assert "backtest_lag_archive" in del_sql
        # Second call: SELECT EXISTS (check for non-external rows)
        exists_sql = cur.execute.call_args_list[1][0][0]
        assert "SELECT EXISTS" in exists_sql
        # Third call: INSERT (with or without ON CONFLICT depending on EXISTS result)
        ins_sql = cur.execute.call_args_list[2][0][0]
        assert "INSERT INTO" in ins_sql

    def test_returns_inserted_row_count(self):
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 12345])

        assert _load_forecast_archive(cur, "stg", "s") == 12345

    def test_uses_staging_alias_in_sql(self):
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 100])

        _load_forecast_archive(cur, "stg", "myalias")

        ins_sql = cur.execute.call_args_list[2][0][0]
        assert 'myalias."item_id"' in ins_sql


# ---------- TestPartitionHelpers ----------

class TestPartitionHelpers:
    def test_is_partitioned_true(self):
        cur = MagicMock()
        cur.fetchone.return_value = (True,)

        assert _is_partitioned(cur, "fact_inventory_snapshot") is True
        sql = cur.execute.call_args[0][0]
        assert "relkind" in sql

    def test_is_partitioned_false(self):
        cur = MagicMock()
        cur.fetchone.return_value = (False,)

        assert _is_partitioned(cur, "fact_sales_monthly") is False

    def test_is_partitioned_table_missing(self):
        cur = MagicMock()
        cur.fetchone.return_value = None

        assert _is_partitioned(cur, "nonexistent") is False

    def test_ensure_partition_exists_creates_when_missing(self):
        cur = MagicMock()
        cur.fetchone.return_value = None  # partition doesn't exist

        name = _ensure_partition_exists(cur, "fact_inventory_snapshot", "2027-07-01", "2027-08-01")

        assert name == "fact_inventory_snapshot_2027_07"
        assert cur.execute.call_count == 2  # SELECT + CREATE
        create_sql = cur.execute.call_args_list[1][0][0]
        assert "PARTITION OF" in create_sql

    def test_ensure_partition_exists_skips_when_present(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1,)  # partition exists

        name = _ensure_partition_exists(cur, "fact_inventory_snapshot", "2025-01-01", "2025-02-01")

        assert name == "fact_inventory_snapshot_2025_01"
        assert cur.execute.call_count == 1  # Only SELECT, no CREATE


# ---------- TestLoadDomain ----------

class TestLoadDomain:
    @patch("scripts.etl.load_dataset_postgres.complete_batch")
    @patch("scripts.etl.load_dataset_postgres.create_batch", return_value=1)
    @patch("scripts.etl.load_dataset_postgres.file_hash", return_value="abc123")
    @patch("scripts.etl.load_dataset_postgres.get_db_params", return_value={"dbname": "test"})
    @patch("scripts.etl.load_dataset_postgres.psycopg")
    def test_load_returns_summary(self, mock_pg, mock_db, mock_hash, mock_batch, mock_complete, tmp_path):
        csv_file = tmp_path / "clean_sales.csv"
        csv_file.write_text("col1,col2\na,b\n")

        cur = MagicMock()
        # fetchone calls: staging count, _is_partitioned (False),
        # DFU filter EXISTS check (False=skip),
        # FK orphan filter: dim_location EXISTS (False), dim_item EXISTS (False)
        cur.fetchone.side_effect = [(1,), (False,), (False,), (False,), (False,)]
        cur.fetchall.return_value = []    # no indexes/constraints
        cur.rowcount = 1

        copy_ctx = MagicMock()
        cur.copy.return_value.__enter__ = MagicMock(return_value=copy_ctx)
        cur.copy.return_value.__exit__ = MagicMock(return_value=False)

        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.connect.return_value.__enter__ = MagicMock(return_value=conn)
        mock_pg.connect.return_value.__exit__ = MagicMock(return_value=False)

        spec = MagicMock()
        spec.name = "sales"
        spec.table = "fact_sales_monthly"
        spec.columns = ["col1", "col2"]
        spec.ck_field = "sales_ck"
        spec.key_fields = ["col1"]
        spec.int_fields = set()
        spec.float_fields = set()
        spec.date_fields = set()
        spec.business_key_separator = "_"
        spec.clean_file = "clean_sales.csv"

        result = load_domain(spec, csv_file)

        assert result["domain"] == "sales"
        assert "rows_in" in result
        assert "rows_loaded" in result
        assert "elapsed" in result
        # Verify staging table created and TRUNCATE issued
        executed_sqls = [c[0][0] for c in cur.execute.call_args_list]
        assert any("CREATE TEMP TABLE" in s for s in executed_sqls)
        assert any("TRUNCATE" in s for s in executed_sqls)

    @patch("scripts.etl.load_dataset_postgres.get_db_params", return_value={"dbname": "test"})
    @patch("scripts.etl.load_dataset_postgres.psycopg")
    def test_load_skips_missing_csv(self, mock_pg, mock_db, tmp_path):
        spec = MagicMock()
        spec.name = "sales"
        missing = tmp_path / "does_not_exist.csv"

        result = load_domain(spec, missing)

        assert result["skipped"] is True
        mock_pg.connect.assert_not_called()
