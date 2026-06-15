"""Tests for common/core/etl_helpers.py (US3 — shared index/constraint helper).

Covers the generic introspection-based helpers and the forecast/archive
index+constraint management consolidated from the three ETL loaders. The
forecast assertions preserve the exact SQL the loaders emitted before the
consolidation (parity with US1 characterization).
"""

from unittest.mock import MagicMock

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
