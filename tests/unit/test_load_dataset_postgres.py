"""Tests for scripts/load_dataset_postgres.py — simplified loader."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl.load_dataset_postgres import (
    _ensure_partition_exists,
    _is_partitioned,
    _refresh_original_sales,
    _resolve_forecast_execution_lag,
    load_domain,
)

# Index/constraint helpers (US3) and DFU/FK filters (US5) moved to
# common/core/etl_helpers.py; their tests live in tests/unit/test_etl_helpers.py.


# ---------- Helpers ----------

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


# _load_forecast_archive removed (US9): it was dead code — external forecasts
# skip the archive (owned by load_backtest_forecasts.py).


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
    def test_original_sales_mirror_requires_exact_row_parity(self):
        cur = MagicMock()
        cur.rowcount = 9

        with pytest.raises(RuntimeError, match="mirror row count"):
            _refresh_original_sales(
                cur,
                source_table="fact_sales_monthly",
                columns=["sales_ck", "item_id"],
                expected_rows=10,
            )

        sqls = [call.args[0] for call in cur.execute.call_args_list]
        assert sqls[0] == "TRUNCATE TABLE fact_sales_monthly_original"
        assert "INSERT INTO fact_sales_monthly_original" in sqls[1]

    @patch(
        "scripts.etl.load_dataset_postgres._refresh_original_sales",
        side_effect=RuntimeError("mirror mismatch"),
    )
    @patch("scripts.etl.load_dataset_postgres.complete_batch")
    @patch("scripts.etl.load_dataset_postgres.fail_batch")
    @patch("scripts.etl.load_dataset_postgres.create_batch", return_value=1)
    @patch("scripts.etl.load_dataset_postgres.file_hash", return_value="abc123")
    @patch("scripts.etl.load_dataset_postgres.get_db_params", return_value={"dbname": "test"})
    @patch("scripts.etl.load_dataset_postgres.psycopg.connect")
    def test_original_sales_mirror_failure_rolls_back_batch(
        self,
        mock_connect,
        _mock_db,
        _mock_hash,
        _mock_batch,
        mock_fail_batch,
        mock_complete_batch,
        _mock_refresh,
        tmp_path,
    ):
        csv_file = tmp_path / "clean_sales.csv"
        csv_file.write_text("col1,col2\na,b\n")
        cur = MagicMock()
        cur.fetchone.side_effect = [
            (1,),
            (False,),
            (1_000_000,),
            (False,),
            (False,),
            (False,),
        ]
        cur.fetchall.return_value = []
        cur.rowcount = 1
        copy_ctx = MagicMock()
        cur.copy.return_value.__enter__.return_value = copy_ctx
        cur.copy.return_value.__exit__.return_value = False
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False
        connection_context = MagicMock()
        connection_context.__enter__.return_value = conn
        connection_context.__exit__.return_value = False
        mock_connect.return_value = connection_context
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

        with pytest.raises(RuntimeError, match="mirror mismatch"):
            load_domain(spec, csv_file)

        conn.rollback.assert_called_once()
        mock_complete_batch.assert_not_called()
        mock_fail_batch.assert_called_once()

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
        # should_drop_indexes reltuples estimate (large -> drop path),
        # DFU filter EXISTS check (False=skip),
        # FK orphan filter: dim_location EXISTS (False), dim_item EXISTS (False)
        cur.fetchone.side_effect = [(1,), (False,), (1_000_000,), (False,), (False,), (False,)]
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
        assert any("TRUNCATE TABLE fact_sales_monthly_original" in s for s in executed_sqls)
        assert any("INSERT INTO fact_sales_monthly_original" in s for s in executed_sqls)

    @patch("scripts.etl.load_dataset_postgres.get_db_params", return_value={"dbname": "test"})
    @patch("scripts.etl.load_dataset_postgres.psycopg")
    def test_load_skips_missing_csv(self, mock_pg, mock_db, tmp_path):
        spec = MagicMock()
        spec.name = "sales"
        missing = tmp_path / "does_not_exist.csv"

        result = load_domain(spec, missing)

        assert result["skipped"] is True
        mock_pg.connect.assert_not_called()
