"""Tests for load_monthly_errors_df lag_mode='all' and champion per-lag breakdown.

Verifies that:
  - lag_mode="all" queries backtest_lag_archive (not fact_external_forecast_monthly)
  - lag_mode="execution" queries fact_external_forecast_monthly with lag=execution_lag
  - lag_mode="2" queries fact_external_forecast_monthly with lag=2
"""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest


def _mock_read_sql(expected_table: str):
    """Return a mock pd.read_sql that asserts the correct table is queried."""
    def _read_sql(sql, conn, params=None):
        assert expected_table in sql, (
            f"Expected query against '{expected_table}' but got:\n{sql}"
        )
        return pd.DataFrame({
            "item_id": ["A"],
            "customer_group": ["G"],
            "loc": ["L"],
            "startdate": ["2025-01-01"],
            "fcstdate": ["2024-12-01"],
            "execution_lag": [1],
            "model_id": ["lgbm_cluster"],
            "basefcst_pref": [100.0],
            "tothist_dmd": [90.0],
            "abs_err": [10.0],
            **({"lag": [0]} if expected_table == "backtest_lag_archive" else {}),
        })
    return _read_sql


class TestLoadMonthlyErrorsLagMode:
    """Verify load_monthly_errors_df routes to the correct table per lag_mode."""

    @patch("psycopg.connect")
    @patch("pandas.read_sql")
    def test_lag_mode_all_queries_archive(self, mock_rsql, mock_connect):
        mock_rsql.side_effect = _mock_read_sql("backtest_lag_archive")
        mock_connect.return_value.__enter__ = MagicMock()
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        from scripts.run_champion_selection import load_monthly_errors_df
        df = load_monthly_errors_df(
            db={"host": "x", "dbname": "x", "user": "x", "password": "x"},
            models=["lgbm_cluster", "catboost_cluster"],
            lag_mode="all",
        )
        assert not df.empty
        assert "lag" in df.columns

    @patch("psycopg.connect")
    @patch("pandas.read_sql")
    def test_lag_mode_execution_queries_main_table(self, mock_rsql, mock_connect):
        mock_rsql.side_effect = _mock_read_sql("fact_external_forecast_monthly")
        mock_connect.return_value.__enter__ = MagicMock()
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        from scripts.run_champion_selection import load_monthly_errors_df
        df = load_monthly_errors_df(
            db={"host": "x", "dbname": "x", "user": "x", "password": "x"},
            models=["lgbm_cluster"],
            lag_mode="execution",
        )
        assert not df.empty

    @patch("psycopg.connect")
    @patch("pandas.read_sql")
    def test_lag_mode_specific_queries_main_table(self, mock_rsql, mock_connect):
        mock_rsql.side_effect = _mock_read_sql("fact_external_forecast_monthly")
        mock_connect.return_value.__enter__ = MagicMock()
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        from scripts.run_champion_selection import load_monthly_errors_df
        df = load_monthly_errors_df(
            db={"host": "x", "dbname": "x", "user": "x", "password": "x"},
            models=["lgbm_cluster"],
            lag_mode="2",
        )
        assert not df.empty

    @patch("psycopg.connect")
    @patch("pandas.read_sql")
    def test_lag_mode_all_sql_has_no_lag_filter(self, mock_rsql, mock_connect):
        """lag_mode='all' should NOT have a WHERE lag = ... condition."""
        captured_sql = []

        def _capture(sql, conn, params=None):
            captured_sql.append(sql)
            return pd.DataFrame({
                "item_id": [], "customer_group": [], "loc": [],
                "startdate": [], "fcstdate": [], "execution_lag": [],
                "lag": [], "model_id": [], "basefcst_pref": [],
                "tothist_dmd": [], "abs_err": [],
            })

        mock_rsql.side_effect = _capture
        mock_connect.return_value.__enter__ = MagicMock()
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        from scripts.run_champion_selection import load_monthly_errors_df
        load_monthly_errors_df(
            db={"host": "x", "dbname": "x", "user": "x", "password": "x"},
            models=["lgbm_cluster"],
            lag_mode="all",
        )
        sql = captured_sql[0]
        # Should NOT filter by specific lag
        assert "lag::text = execution_lag::text" not in sql
        assert "lag = %s" not in sql

    @patch("psycopg.connect")
    @patch("pandas.read_sql")
    def test_lag_mode_execution_sql_has_exec_lag_filter(self, mock_rsql, mock_connect):
        captured_sql = []

        def _capture(sql, conn, params=None):
            captured_sql.append(sql)
            return pd.DataFrame({
                "item_id": [], "customer_group": [], "loc": [],
                "startdate": [], "fcstdate": [], "execution_lag": [],
                "model_id": [], "basefcst_pref": [],
                "tothist_dmd": [], "abs_err": [],
            })

        mock_rsql.side_effect = _capture
        mock_connect.return_value.__enter__ = MagicMock()
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        from scripts.run_champion_selection import load_monthly_errors_df
        load_monthly_errors_df(
            db={"host": "x", "dbname": "x", "user": "x", "password": "x"},
            models=["lgbm_cluster"],
            lag_mode="execution",
        )
        sql = captured_sql[0]
        assert "lag::text = execution_lag::text" in sql
