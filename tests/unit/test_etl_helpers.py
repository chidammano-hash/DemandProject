"""Tests for common/core/etl_helpers.py (US3 — shared index/constraint helper).

Covers the generic introspection-based helpers and the forecast/archive
index+constraint management consolidated from the three ETL loaders. The
forecast assertions preserve the exact SQL the loaders emitted before the
consolidation (parity with US1 characterization).
"""

from datetime import date
from unittest.mock import MagicMock, patch

from common.core import etl_helpers as eh


def _executed_sql(cur) -> str:
    return " ; ".join(str(c.args[0]) for c in cur.execute.call_args_list)


# ---------------------------------------------------------------------------
# Generic introspection helpers
# ---------------------------------------------------------------------------

class TestGetSecondaryIndexes:
    def test_queries_pg_indexes_and_returns_rows(self):
        cur = MagicMock()
        cur.fetchall.return_value = [("idx1", "CREATE INDEX idx1 ON t (a)")]
        result = eh.get_secondary_indexes(cur, "my_table")
        assert result == [("idx1", "CREATE INDEX idx1 ON t (a)")]
        sql = _executed_sql(cur)
        assert "pg_indexes" in sql
        assert cur.execute.call_args_list[0].args[1] == ("my_table",)


class TestGetUniqueConstraints:
    def test_maps_rows(self):
        cur = MagicMock()
        cur.fetchall.return_value = [("c1", "u", ["a", "b"])]
        result = eh.get_unique_constraints(cur, "my_table")
        assert result == [("c1", "u", ["a", "b"])]
        assert "pg_constraint" in _executed_sql(cur)


class TestPerfSetting:
    def test_reads_value_from_config(self):
        with patch("common.core.utils.load_config",
                   return_value={"performance": {"batch_size": 500_000}}):
            assert eh.perf_setting("batch_size", 2_000_000) == 500_000

    def test_default_when_key_absent(self):
        with patch("common.core.utils.load_config", return_value={"performance": {}}):
            assert eh.perf_setting("batch_size", 2_000_000) == 2_000_000

    def test_default_when_config_missing(self):
        with patch("common.core.utils.load_config", side_effect=FileNotFoundError):
            assert eh.perf_setting("pg_work_mem", "256MB") == "256MB"

    def test_real_config_documents_perf_keys(self):
        # AC: the keys the loaders read are present in the real etl_config.yaml.
        from common.core.utils import load_config
        perf = (load_config("etl/etl_config.yaml") or {}).get("performance") or {}
        for key in (
            "batch_size", "pg_work_mem", "pg_maintenance_work_mem",
            "customer_demand_work_mem", "customer_demand_maintenance_work_mem",
            "customer_demand_max_workers", "index_drop_row_threshold",
        ):
            assert key in perf, f"missing performance.{key} in etl_config.yaml"


class TestSizeBasedIndexDrop:
    def test_estimate_row_count_reads_reltuples(self):
        cur = MagicMock()
        cur.fetchone.return_value = (12345,)
        assert eh.estimate_row_count(cur, "fact_x") == 12345
        assert "reltuples" in _executed_sql(cur)

    def test_estimate_row_count_missing_table_zero(self):
        cur = MagicMock()
        cur.fetchone.return_value = None
        assert eh.estimate_row_count(cur, "nope") == 0

    def test_should_drop_above_threshold(self):
        cur = MagicMock()
        cur.fetchone.return_value = (100_000,)
        assert eh.should_drop_indexes(cur, "fact_x", threshold=50_000) is True

    def test_should_keep_below_threshold(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1_000,)
        assert eh.should_drop_indexes(cur, "dim_x", threshold=50_000) is False

    def test_never_analyzed_treated_as_large(self):
        cur = MagicMock()
        cur.fetchone.return_value = (-1,)  # PG "unknown"
        assert eh.should_drop_indexes(cur, "fact_new", threshold=50_000) is True

    def test_threshold_from_config(self):
        cur = MagicMock()
        cur.fetchone.return_value = (5_000,)
        with patch("common.core.utils.load_config",
                   return_value={"performance": {"index_drop_row_threshold": 1_000}}):
            assert eh.should_drop_indexes(cur, "t") is True  # 5000 >= 1000

    def test_threshold_default_when_config_missing(self):
        with patch("common.core.utils.load_config", side_effect=FileNotFoundError):
            assert eh.index_drop_row_threshold() == eh._DEFAULT_INDEX_DROP_ROW_THRESHOLD


class TestDropAndRecreate:
    def test_drop_indexes_quotes_identifiers(self):
        cur = MagicMock()
        eh.drop_indexes(cur, [("idx1", "d1"), ("idx2", "d2")])
        sql = _executed_sql(cur)
        assert 'DROP INDEX IF EXISTS "idx1"' in sql
        assert 'DROP INDEX IF EXISTS "idx2"' in sql

    def test_recreate_indexes_replays_ddl(self):
        cur = MagicMock()
        eh.recreate_indexes(cur, [("idx1", "CREATE INDEX idx1 ON t (a)")])
        assert "CREATE INDEX idx1 ON t (a);" in _executed_sql(cur)

    def test_drop_unique_constraints(self):
        cur = MagicMock()
        eh.drop_unique_constraints(cur, "my_table", [("uc", "u", ["a"])])
        assert 'ALTER TABLE "my_table" DROP CONSTRAINT IF EXISTS "uc"' in _executed_sql(cur)

    def test_recreate_unique_constraints(self):
        cur = MagicMock()
        eh.recreate_unique_constraints(cur, "my_table", [("uc", "u", ["a", "b"])])
        assert 'ADD CONSTRAINT "uc" UNIQUE ("a", "b")' in _executed_sql(cur)


# ---------------------------------------------------------------------------
# Forecast / archive specs + functions (parity with prior loader behavior)
# ---------------------------------------------------------------------------

class TestStagingTableName:
    def test_deterministic_convention(self):
        assert eh.staging_table_name("sales") == "stg_sales"
        assert eh.staging_table_name("customer_demand") == "stg_customer_demand"


class TestIsPgPartitioned:
    def test_true(self):
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        assert eh.is_pg_partitioned(cur, "fact_inventory_snapshot") is True
        assert "relkind" in _executed_sql(cur)

    def test_false(self):
        cur = MagicMock()
        cur.fetchone.return_value = (False,)
        assert eh.is_pg_partitioned(cur, "fact_sales_monthly") is False

    def test_missing(self):
        cur = MagicMock()
        cur.fetchone.return_value = None
        assert eh.is_pg_partitioned(cur, "nope") is False


class TestMonthlyPartitions:
    def test_partition_name(self):
        assert eh.monthly_partition_name("fact_x", date(2024, 1, 1)) == "fact_x_2024_01"

    def test_create_monthly_partition_unconditional(self):
        cur = MagicMock()
        name = eh.create_monthly_partition(cur, "fact_x", date(2024, 5, 1))
        assert name == "fact_x_2024_05"
        sql = _executed_sql(cur)
        assert "CREATE TABLE" in sql and "PARTITION OF" in sql
        assert "FOR VALUES FROM ('2024-05-01') TO ('2024-06-01')" in sql
        # no existence SELECT — unconditional
        assert cur.execute.call_count == 1

    def test_month_bounds_mid_year(self):
        assert eh.month_bounds(date(2024, 3, 1)) == ("2024-03-01", "2024-04-01")

    def test_month_bounds_december_wraps(self):
        assert eh.month_bounds(date(2024, 12, 1)) == ("2024-12-01", "2025-01-01")

    def test_ensure_creates_when_absent(self):
        cur = MagicMock()
        cur.fetchone.return_value = None
        name = eh.ensure_monthly_partition(cur, "fact_x", date(2024, 5, 1))
        assert name == "fact_x_2024_05"
        assert cur.execute.call_count == 2  # SELECT + CREATE
        sql = _executed_sql(cur)
        assert "PARTITION OF" in sql
        assert "FOR VALUES FROM ('2024-05-01') TO ('2024-06-01')" in sql

    def test_ensure_skips_when_present(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        eh.ensure_monthly_partition(cur, "fact_x", date(2024, 5, 1))
        assert cur.execute.call_count == 1  # SELECT only
        assert "CREATE TABLE" not in _executed_sql(cur)

    def test_drop_partition(self):
        cur = MagicMock()
        eh.drop_monthly_partition(cur, "fact_x", date(2024, 7, 1))
        assert 'DROP TABLE IF EXISTS "fact_x_2024_07"' in _executed_sql(cur)


class TestSliceDeleteMetadataParity:
    """The removed load.py _SLICE_DELETE_TABLES map is now derived from
    DomainSpec.table + DomainPartition.field — verify they still agree."""

    def test_derived_table_and_column_match_legacy_map(self):
        from common.core.domain_partition import get_partition
        from common.core.domain_specs import get_spec

        legacy = {
            "inventory": ("fact_inventory_snapshot", "snapshot_date"),
            "forecast": ("fact_external_forecast_monthly", "fcstdate"),
            "sales": ("fact_sales_monthly", "startdate"),
        }
        for domain, (table, col) in legacy.items():
            assert get_spec(domain).table == table
            assert get_partition(domain).field == col


class TestDeletePartitionRange:
    def test_half_open_interval(self):
        cur = MagicMock()
        cur.rowcount = 42
        deleted = eh.delete_partition_range(cur, "fact_x", "startdate", "2024-03-01", "2024-04-01")
        assert deleted == 42
        call = cur.execute.call_args
        assert '>= %s AND "startdate" < %s' in call.args[0]
        assert call.args[1] == ("2024-03-01", "2024-04-01")


class TestForecastConstants:
    def test_table_and_unique_constraint_names(self):
        assert eh.FORECAST_TABLE == "fact_external_forecast_monthly"
        assert eh.FORECAST_UNIQUE_CONSTRAINT == "uq_forecast_ck_model"
        assert eh.FORECAST_ARCHIVE_TABLE == "backtest_lag_archive"
        assert eh.FORECAST_ARCHIVE_UNIQUE_CONSTRAINT == "uq_backtest_lag_archive_ck"

    def test_index_counts(self):
        assert len(eh.FORECAST_SECONDARY_INDEXES) == 6
        assert len(eh.FORECAST_INDEX_DDL) == 6
        assert len(eh.FORECAST_ARCHIVE_SECONDARY_INDEXES) == 4
        assert len(eh.FORECAST_ARCHIVE_INDEX_DDL) == 4


class TestForecastDropRecreate:
    def test_drop_main_counts_and_sql(self):
        cur = MagicMock()
        eh.drop_forecast_indexes_and_constraints(cur)
        sql = _executed_sql(cur)
        for idx in eh.FORECAST_SECONDARY_INDEXES:
            assert f"DROP INDEX IF EXISTS {idx}" in sql
        assert f"DROP CONSTRAINT IF EXISTS {eh.FORECAST_UNIQUE_CONSTRAINT}" in sql
        assert cur.execute.call_count == 10  # 6 idx + 1 unique + 3 checks

    def test_recreate_main_counts_and_sql(self):
        cur = MagicMock()
        eh.recreate_forecast_indexes_and_constraints(cur)
        sql = _executed_sql(cur)
        assert f"ADD CONSTRAINT {eh.FORECAST_UNIQUE_CONSTRAINT} UNIQUE (forecast_ck, model_id)" in sql
        assert "CHECK (lag BETWEEN 0 AND 4)" in sql
        assert cur.execute.call_count == 8  # 1 unique + 6 idx + 1 combined check

    def test_drop_archive_counts(self):
        cur = MagicMock()
        eh.drop_forecast_archive_indexes_and_constraints(cur)
        assert cur.execute.call_count == 8  # 4 idx + 1 unique + 3 checks

    def test_recreate_archive_counts_and_unique_key(self):
        cur = MagicMock()
        eh.recreate_forecast_archive_indexes_and_constraints(cur)
        sql = _executed_sql(cur)
        assert (
            f"ADD CONSTRAINT {eh.FORECAST_ARCHIVE_UNIQUE_CONSTRAINT} "
            "UNIQUE (forecast_ck, model_id, lag)"
        ) in sql
        assert cur.execute.call_count == 6  # 1 unique + 4 idx + 1 combined check


# ---------------------------------------------------------------------------
# DFU / FK filters (US5, relocated from test_load_dataset_postgres)
# ---------------------------------------------------------------------------

def _cursor_with_rowcounts(rowcounts):
    """MagicMock cursor cycling rowcount values on execute."""
    cur = MagicMock()
    rc_iter = iter(rowcounts)

    def _on_execute(*a, **kw):
        try:
            cur.rowcount = next(rc_iter)
        except StopIteration:
            pass

    cur.execute.side_effect = _on_execute
    return cur


class TestFilterUnmatchedDfus:
    def test_dim_sku_missing_skips_filter(self):
        cur = MagicMock()
        cur.fetchone.return_value = (False,)
        assert eh.filter_unmatched_dfus(cur, "stg_sales", "sales") == 0
        assert cur.execute.call_count == 1  # EXISTS only

    def test_sales_deletes_by_sku_ck(self):
        cur = _cursor_with_rowcounts([None, 200])
        cur.fetchone.side_effect = [(True,)]
        assert eh.filter_unmatched_dfus(cur, "stg_sales", "sales") == 200
        delete_sql = cur.execute.call_args_list[1][0][0]
        assert "NOT EXISTS" in delete_sql and "sku_ck" in delete_sql

    def test_inventory_matches_on_item_id_loc(self):
        cur = _cursor_with_rowcounts([None, 50])
        cur.fetchone.side_effect = [(True,)]
        assert eh.filter_unmatched_dfus(cur, "stg_inv", "inventory") == 50
        delete_sql = cur.execute.call_args_list[1][0][0]
        assert "item_id" in delete_sql and "loc" in delete_sql
        assert "sku_ck" not in delete_sql

    def test_no_unmatched_returns_zero(self):
        cur = _cursor_with_rowcounts([None, 0])
        cur.fetchone.side_effect = [(True,)]
        assert eh.filter_unmatched_dfus(cur, "stg_sales", "sales") == 0


class TestFilterFkOrphans:
    def test_unconfigured_domain_returns_zero(self):
        cur = MagicMock()
        assert eh.filter_fk_orphans(cur, "stg_item", "item") == 0
        cur.execute.assert_not_called()

    def test_removes_orphans_when_dim_and_col_present(self):
        # For each of the 2 FK checks: EXISTS(dim)=True, EXISTS(col)=True, then DELETE
        cur = _cursor_with_rowcounts([None, None, 7, None, None, 3])
        cur.fetchone.side_effect = [(True,), (True,), (True,), (True,)]
        assert eh.filter_fk_orphans(cur, "stg_sales", "sales") == 10


class TestUnmatchedWarnPct:
    def test_reads_from_config(self):
        # load_config is imported lazily inside the helper from common.core.utils.
        with patch("common.core.utils.load_config",
                   return_value={"filters": {"unmatched_dfu_warn_pct": 25.0}}):
            assert eh.unmatched_warn_pct() == 25.0

    def test_default_when_config_missing(self):
        with patch("common.core.utils.load_config", side_effect=FileNotFoundError):
            assert eh.unmatched_warn_pct() == eh._DEFAULT_UNMATCHED_WARN_PCT


class TestDfuKeyForRow:
    def test_sales_forecast_key_is_sku_ck_shape(self):
        row = {"item_id": " A ", "customer_group": " CG ", "loc": " L "}
        assert eh.dfu_key_for_row(row, "sales") == "A_CG_L"
        assert eh.dfu_key_for_row(row, "forecast") == "A_CG_L"

    def test_inventory_key_is_item_loc_only(self):
        row = {"item_id": "A", "customer_group": "ignored", "loc": "L"}
        assert eh.dfu_key_for_row(row, "inventory") == "A\tL"

    def test_missing_fields_default_empty(self):
        assert eh.dfu_key_for_row({}, "sales") == "__"


class TestLoadValidDfuKeys:
    def test_none_when_dim_sku_missing(self):
        cur = MagicMock()
        cur.fetchone.return_value = (False,)  # dim_sku does not exist
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("common.core.db.get_db_params", return_value={}), \
             patch("psycopg.connect") as mock_connect:
            mock_connect.return_value.__enter__.return_value = conn
            assert eh.load_valid_dfu_keys("sales") is None

    def test_returns_sku_ck_set(self):
        cur = MagicMock()
        cur.fetchone.return_value = (True,)  # dim_sku exists
        cur.fetchall.return_value = [("A_CG_L",), ("B_CG_L",)]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("common.core.db.get_db_params", return_value={}), \
             patch("psycopg.connect") as mock_connect:
            mock_connect.return_value.__enter__.return_value = conn
            assert eh.load_valid_dfu_keys("forecast") == {"A_CG_L", "B_CG_L"}


class TestRecordLoadBatch:
    def test_completed_path_creates_and_completes(self):
        cur = MagicMock()
        cur.fetchone.return_value = (99,)  # batch_id from create_batch RETURNING
        bid = eh.record_load_batch(
            cur, "customer_demand", source_file="cd.csv", source_hash="h",
            rows_in=10, rows_out=9,
        )
        assert bid == 99
        sql = " ; ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        assert "INSERT INTO audit_load_batch" in sql
        assert "status = 'completed'" in sql

    def test_failed_path_marks_failed(self):
        cur = MagicMock()
        cur.fetchone.return_value = (7,)
        eh.record_load_batch(cur, "sales", status="failed", error="boom")
        sql = " ; ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        assert "status = 'failed'" in sql
