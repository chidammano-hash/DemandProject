"""Tests for scripts/etl/load_ext_ml_forecasts.py — external ML forecast loader."""

import pytest

from scripts.etl.load_ext_ml_forecasts import (
    _normalize_date,
    _build_stg_insert_sql,
    _build_archive_insert_sql,
)


# ---------------------------------------------------------------------------
# _normalize_date
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    """Verify date normalization to YYYY-MM-01 month-start ISO format."""

    def test_already_month_start_unchanged(self):
        assert _normalize_date("2025-05-01") == "2025-05-01"

    def test_mid_month_normalized_to_first(self):
        assert _normalize_date("2025-05-15") == "2025-05-01"

    def test_end_of_month_normalized_to_first(self):
        assert _normalize_date("2025-01-31") == "2025-01-01"

    def test_year_month_only_returns_none(self):
        # Only YYYY-MM-DD is supported; YYYY-MM (no day) cannot split into 3 parts
        assert _normalize_date("2025-05") is None

    def test_slash_date_format_returns_none(self):
        # Slash-separated dates are not supported; input CSV uses YYYY-MM-DD only
        assert _normalize_date("2024/03/15") is None

    def test_us_date_format_returns_none(self):
        # MM/DD/YYYY format is not supported; input CSV uses YYYY-MM-DD only
        assert _normalize_date("06/20/2023") is None

    def test_empty_string_returns_none(self):
        assert _normalize_date("") is None

    def test_whitespace_only_returns_none(self):
        assert _normalize_date("   ") is None

    def test_null_string_returns_none(self):
        assert _normalize_date("null") is None

    def test_none_string_case_insensitive_returns_none(self):
        assert _normalize_date("None") is None

    def test_none_lowercase_returns_none(self):
        assert _normalize_date("none") is None

    def test_na_string_returns_none(self):
        assert _normalize_date("NA") is None

    def test_na_lowercase_returns_none(self):
        assert _normalize_date("na") is None

    def test_n_slash_a_returns_none(self):
        assert _normalize_date("n/a") is None

    def test_nan_returns_none(self):
        assert _normalize_date("nan") is None

    def test_completely_invalid_returns_none(self):
        assert _normalize_date("not-a-date") is None

    def test_partial_garbage_returns_none(self):
        assert _normalize_date("2025-99-99") is None

    def test_leading_trailing_whitespace_stripped(self):
        assert _normalize_date("  2025-07-10  ") == "2025-07-01"

    def test_different_year(self):
        assert _normalize_date("2020-12-31") == "2020-12-01"

    def test_february_leap_year_normalized(self):
        assert _normalize_date("2024-02-29") == "2024-02-01"

    def test_result_always_day_01(self):
        result = _normalize_date("2026-03-29")
        assert result is not None
        assert result.endswith("-01")

    def test_returns_string_not_date_object(self):
        result = _normalize_date("2025-05-01")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _build_stg_insert_sql
# ---------------------------------------------------------------------------


class TestBuildStgInsertSql:
    """Verify SQL for staging → fact_external_forecast_monthly insert."""

    def test_bulk_fast_omits_on_conflict(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "ON CONFLICT" not in sql

    def test_not_bulk_fast_has_on_conflict(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=False)
        assert "ON CONFLICT" in sql

    def test_not_bulk_fast_has_do_update_set(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=False)
        assert "DO UPDATE SET" in sql

    def test_targets_fact_external_forecast_monthly(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "fact_external_forecast_monthly" in sql

    def test_joins_dim_sku_for_dfu_resolution(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "dim_sku" in sql

    def test_uses_sku_ck_join_column(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "sku_ck" in sql

    def test_filters_to_execution_lag_only(self):
        # main table must only load the row where lag = dim_sku.execution_lag
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "d.execution_lag" in sql
        assert "= d.execution_lag" in sql

    def test_uses_dim_sku_execution_lag_column(self):
        # execution_lag in SELECT must come from dim_sku, not NULL or CSV
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "d.execution_lag" in sql

    def test_row_id_batching_present(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "_row_id" in sql

    def test_references_staging_table(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "_stg_ext_ml" in sql

    def test_returns_non_empty_string(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert isinstance(sql, str)
        assert len(sql.strip()) > 0

    def test_bulk_fast_false_still_targets_correct_table(self):
        sql = _build_stg_insert_sql("ext_catboost", bulk_fast=False)
        assert "fact_external_forecast_monthly" in sql

    def test_sql_is_insert_statement(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "INSERT" in sql.upper()

    def test_bulk_fast_true_no_upsert_keywords(self):
        sql = _build_stg_insert_sql("ext_lgbm", bulk_fast=True)
        assert "DO NOTHING" not in sql
        assert "DO UPDATE" not in sql


# ---------------------------------------------------------------------------
# _build_archive_insert_sql
# ---------------------------------------------------------------------------


class TestBuildArchiveInsertSql:
    """Verify SQL for staging → backtest_lag_archive insert."""

    def test_bulk_fast_omits_on_conflict(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert "ON CONFLICT" not in sql

    def test_not_bulk_fast_has_on_conflict(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=False)
        assert "ON CONFLICT" in sql

    def test_not_bulk_fast_conflict_target_includes_forecast_ck_model_id_lag(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=False)
        assert "forecast_ck, model_id, lag" in sql

    def test_targets_backtest_lag_archive(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert "backtest_lag_archive" in sql

    def test_timeframe_is_null(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert "NULL" in sql

    def test_references_staging_table(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert "_stg_ext_ml" in sql

    def test_returns_non_empty_string(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert isinstance(sql, str)
        assert len(sql.strip()) > 0

    def test_sql_is_insert_statement(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert "INSERT" in sql.upper()

    def test_bulk_fast_false_still_targets_archive_table(self):
        sql = _build_archive_insert_sql("ext_catboost", bulk_fast=False)
        assert "backtest_lag_archive" in sql

    def test_bulk_fast_true_no_upsert_keywords(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=True)
        assert "DO NOTHING" not in sql
        assert "DO UPDATE" not in sql

    def test_not_bulk_fast_has_do_update_set(self):
        sql = _build_archive_insert_sql("ext_lgbm", bulk_fast=False)
        assert "DO UPDATE SET" in sql
