"""Characterization tests for scripts/etl/load_backtest_forecasts.py.

US1 (data-ingestion streamlining): pins the index/constraint drop+recreate
behavior and column/constant definitions BEFORE US3 (shared index helper)
consolidates the three duplicated implementations. These guard parity.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl import load_backtest_forecasts as bt


def _executed_sql(cur) -> str:
    return " ; ".join(str(c.args[0]) for c in cur.execute.call_args_list)


class TestConstants:
    def test_main_table_and_unique_constraint(self):
        assert bt._TABLE == "fact_external_forecast_monthly"
        assert bt._UNIQUE_CONSTRAINT == "uq_forecast_ck_model"

    def test_archive_table_and_unique_constraint(self):
        assert bt._ARCHIVE_TABLE == "backtest_lag_archive"
        assert bt._ARCHIVE_UNIQUE_CONSTRAINT == "uq_backtest_lag_archive_ck"

    def test_secondary_index_counts(self):
        assert len(bt._SECONDARY_INDEXES) == 6
        assert len(bt._INDEX_DDL) == 6
        assert len(bt._ARCHIVE_SECONDARY_INDEXES) == 4
        assert len(bt._ARCHIVE_INDEX_DDL) == 4

    def test_load_cols_lead_with_forecast_ck(self):
        assert bt.LOAD_COLS[0] == "forecast_ck"
        assert "model_id" in bt.LOAD_COLS
        assert bt.ARCHIVE_COLS[-1] == "timeframe"


class TestDropIndexesAndConstraints:
    def test_drops_all_indexes_unique_and_checks(self):
        cur = MagicMock()
        bt._drop_indexes_and_constraints(cur)
        sql = _executed_sql(cur)
        for idx in bt._SECONDARY_INDEXES:
            assert f"DROP INDEX IF EXISTS {idx}" in sql
        assert f"DROP CONSTRAINT IF EXISTS {bt._UNIQUE_CONSTRAINT}" in sql
        for ck in bt._CHECK_CONSTRAINTS:
            assert f"DROP CONSTRAINT IF EXISTS {ck}" in sql
        # 6 indexes + 1 unique + 3 checks
        assert cur.execute.call_count == 10


class TestRecreateIndexesAndConstraints:
    def test_recreates_unique_indexes_and_checks(self):
        cur = MagicMock()
        bt._recreate_indexes_and_constraints(cur)
        sql = _executed_sql(cur)
        assert f"ADD CONSTRAINT {bt._UNIQUE_CONSTRAINT} UNIQUE (forecast_ck, model_id)" in sql
        for idx in bt._SECONDARY_INDEXES:
            assert idx in sql
        assert "CHECK (lag BETWEEN 0 AND 4)" in sql
        # 1 unique + 6 indexes + 1 combined check-constraint statement
        assert cur.execute.call_count == 8


class TestArchiveDropAndRecreate:
    def test_archive_drop_counts(self):
        cur = MagicMock()
        bt._drop_archive_indexes_and_constraints(cur)
        # 4 indexes + 1 unique + 3 checks
        assert cur.execute.call_count == 8

    def test_archive_recreate_counts_and_unique_key(self):
        cur = MagicMock()
        bt._recreate_archive_indexes_and_constraints(cur)
        sql = _executed_sql(cur)
        assert f"ADD CONSTRAINT {bt._ARCHIVE_UNIQUE_CONSTRAINT} UNIQUE (forecast_ck, model_id, lag)" in sql
        # 1 unique + 4 indexes + 1 combined check-constraint statement
        assert cur.execute.call_count == 6
