"""Tests for scripts/clean_forecasts_by_date.py — date-based forecast cleanup."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from scripts.ml.clean_forecasts_by_date import (
    TABLES_ALL,
    TABLES_ARCHIVE,
    TABLES_FORECAST,
    VALID_DATE_COLUMNS,
    build_where_clause,
    clean_by_date,
    parse_date,
)


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------
class TestParseDate:
    """Test date parsing from various string formats."""

    def test_iso_full_date(self):
        assert parse_date("2025-04-01") == date(2025, 4, 1)

    def test_iso_mid_month_normalizes_to_first(self):
        assert parse_date("2025-04-15") == date(2025, 4, 1)

    def test_year_month_format(self):
        assert parse_date("2025-04") == date(2025, 4, 1)

    def test_us_date_format(self):
        assert parse_date("04/01/2025") == date(2025, 4, 1)

    def test_us_date_mid_month_normalizes(self):
        assert parse_date("04/15/2025") == date(2025, 4, 1)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            parse_date("April 2025")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            parse_date("not-a-date")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            parse_date("")


# ---------------------------------------------------------------------------
# build_where_clause
# ---------------------------------------------------------------------------
class TestBuildWhereClause:
    """Test WHERE clause construction for different filter combinations."""

    def test_before_only(self):
        sql, params = build_where_clause("startdate", before=date(2025, 4, 1))
        assert sql == "WHERE startdate < %s"
        assert params == [date(2025, 4, 1)]

    def test_after_only(self):
        sql, params = build_where_clause("startdate", after=date(2025, 6, 1))
        assert sql == "WHERE startdate >= %s"
        assert params == [date(2025, 6, 1)]

    def test_between(self):
        sql, params = build_where_clause(
            "startdate", between=(date(2024, 1, 1), date(2024, 7, 1))
        )
        assert sql == "WHERE startdate >= %s AND startdate < %s"
        assert params == [date(2024, 1, 1), date(2024, 7, 1)]

    def test_before_with_model(self):
        sql, params = build_where_clause(
            "startdate", before=date(2025, 4, 1), model="external"
        )
        assert sql == "WHERE startdate < %s AND model_id = %s"
        assert params == [date(2025, 4, 1), "external"]

    def test_after_with_model(self):
        sql, params = build_where_clause(
            "fcstdate", after=date(2025, 1, 1), model="lgbm_cluster"
        )
        assert sql == "WHERE fcstdate >= %s AND model_id = %s"
        assert params == [date(2025, 1, 1), "lgbm_cluster"]

    def test_between_with_model(self):
        sql, params = build_where_clause(
            "startdate",
            between=(date(2024, 1, 1), date(2024, 7, 1)),
            model="mstl",
        )
        assert "startdate >= %s AND startdate < %s" in sql
        assert "model_id = %s" in sql
        assert params == [date(2024, 1, 1), date(2024, 7, 1), "mstl"]

    def test_single_month(self):
        sql, params = build_where_clause(
            "startdate", months=[date(2024, 3, 1)]
        )
        assert sql == "WHERE startdate IN (%s)"
        assert params == [date(2024, 3, 1)]

    def test_multiple_months(self):
        sql, params = build_where_clause(
            "startdate",
            months=[date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)],
        )
        assert sql == "WHERE startdate IN (%s, %s, %s)"
        assert params == [date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)]

    def test_months_with_model(self):
        sql, params = build_where_clause(
            "startdate",
            months=[date(2024, 3, 1), date(2024, 6, 1)],
            model="external",
        )
        assert "startdate IN (%s, %s)" in sql
        assert "model_id = %s" in sql
        assert params == [date(2024, 3, 1), date(2024, 6, 1), "external"]

    def test_months_with_fcstdate(self):
        sql, _params = build_where_clause(
            "fcstdate", months=[date(2025, 1, 1)]
        )
        assert "fcstdate IN (%s)" in sql

    def test_fcstdate_column(self):
        sql, _params = build_where_clause("fcstdate", before=date(2025, 1, 1))
        assert "fcstdate < %s" in sql

    def test_invalid_date_column_raises(self):
        with pytest.raises(ValueError, match="Invalid date column"):
            build_where_clause("bad_column", before=date(2025, 1, 1))

    def test_no_date_filter_raises(self):
        with pytest.raises(ValueError, match="Must specify one of"):
            build_where_clause("startdate")

    def test_model_only_without_date_raises(self):
        with pytest.raises(ValueError, match="Must specify one of"):
            build_where_clause("startdate", model="external")


# ---------------------------------------------------------------------------
# clean_by_date
# ---------------------------------------------------------------------------
class TestCleanByDate:
    """Test the clean_by_date function with mocked DB connection."""

    def _mock_conn(self, counts: dict[str, int]):
        """Build a mock connection returning row counts per table."""
        conn = MagicMock()
        fetchone_map = {}
        for table, cnt in counts.items():
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (cnt,)
            fetchone_map[table] = mock_result

        def execute_side_effect(sql, params=None):
            if sql.startswith("SELECT COUNT"):
                for table in counts:
                    if table in sql:
                        return fetchone_map[table]
            return MagicMock()

        conn.execute = MagicMock(side_effect=execute_side_effect)
        conn.commit = MagicMock()
        conn.rollback = MagicMock()
        return conn

    def test_dry_run_does_not_delete(self):
        conn = self._mock_conn({
            "fact_external_forecast_monthly": 100,
            "backtest_lag_archive": 50,
        })
        clean_by_date(
            conn, "WHERE startdate < %s", [date(2025, 4, 1)],
            TABLES_ALL, dry_run=True,
        )
        for call_args in conn.execute.call_args_list:
            sql = call_args[0][0]
            assert not sql.startswith("DELETE"), "DELETE should not run in dry-run"

    def test_actual_delete_calls_delete_and_commit(self):
        conn = self._mock_conn({
            "fact_external_forecast_monthly": 100,
            "backtest_lag_archive": 50,
        })
        clean_by_date(
            conn, "WHERE startdate < %s", [date(2025, 4, 1)],
            TABLES_ALL, dry_run=False,
        )
        delete_calls = [
            c for c in conn.execute.call_args_list
            if c[0][0].startswith("DELETE")
        ]
        assert len(delete_calls) == 2

    def test_no_matching_rows_skips(self):
        conn = self._mock_conn({
            "fact_external_forecast_monthly": 0,
            "backtest_lag_archive": 0,
        })
        clean_by_date(
            conn, "WHERE startdate < %s", [date(2020, 1, 1)],
            TABLES_ALL, dry_run=False,
        )
        delete_calls = [
            c for c in conn.execute.call_args_list
            if c[0][0].startswith("DELETE")
        ]
        assert len(delete_calls) == 0

    def test_forecast_only_scope(self):
        conn = self._mock_conn({"fact_external_forecast_monthly": 100})
        clean_by_date(
            conn, "WHERE startdate < %s", [date(2025, 4, 1)],
            TABLES_FORECAST, dry_run=False,
        )
        delete_calls = [
            c for c in conn.execute.call_args_list
            if c[0][0].startswith("DELETE")
        ]
        assert len(delete_calls) == 1
        assert "fact_external_forecast_monthly" in delete_calls[0][0][0]

    def test_archive_only_scope(self):
        conn = self._mock_conn({"backtest_lag_archive": 50})
        clean_by_date(
            conn, "WHERE startdate < %s", [date(2025, 4, 1)],
            TABLES_ARCHIVE, dry_run=False,
        )
        delete_calls = [
            c for c in conn.execute.call_args_list
            if c[0][0].startswith("DELETE")
        ]
        assert len(delete_calls) == 1
        assert "backtest_lag_archive" in delete_calls[0][0][0]

    def test_view_refresh_called_after_delete(self):
        conn = self._mock_conn({
            "fact_external_forecast_monthly": 100,
            "backtest_lag_archive": 50,
        })
        with patch("scripts.ml.clean_forecasts_by_date.refresh_for_tables") as refresh:
            clean_by_date(
                conn, "WHERE startdate < %s", [date(2025, 4, 1)],
                TABLES_ALL, dry_run=False,
            )
        refresh.assert_called_once_with(TABLES_ALL)

    def test_no_view_refresh_on_dry_run(self):
        conn = self._mock_conn({
            "fact_external_forecast_monthly": 100,
            "backtest_lag_archive": 50,
        })
        with patch("scripts.ml.clean_forecasts_by_date.refresh_for_tables") as refresh:
            clean_by_date(
                conn, "WHERE startdate < %s", [date(2025, 4, 1)],
                TABLES_ALL, dry_run=True,
            )
        refresh.assert_not_called()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class TestConstants:
    """Verify module-level constants."""

    def test_valid_date_columns(self):
        assert "startdate" in VALID_DATE_COLUMNS
        assert "fcstdate" in VALID_DATE_COLUMNS
        assert len(VALID_DATE_COLUMNS) == 2

    def test_tables_all_contains_both(self):
        assert "fact_external_forecast_monthly" in TABLES_ALL
        assert "backtest_lag_archive" in TABLES_ALL

    def test_central_map_covers_cleaned_tables(self):
        # The refresh set now comes from the central dependency map — assert
        # the tables this script deletes from actually have registered
        # dependents there (else a cleanup would silently refresh nothing).
        from common.core.mv_refresh import mvs_for_tables

        dependents = mvs_for_tables(TABLES_ALL)
        assert "agg_forecast_monthly" in dependents
        assert "agg_accuracy_lag_archive" in dependents
