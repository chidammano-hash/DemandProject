"""Tests for scripts/load_dataset_postgres.py — forecast execution-lag loading."""

from unittest.mock import MagicMock, call

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.load_dataset_postgres import (
    _resolve_forecast_execution_lag,
    _load_forecast_archive,
    qident,
    business_key_expr,
    typed_expr,
)


# ---------- Helpers ----------

def _make_cursor(fetchone_values=None, rowcount_values=None):
    """Build a MagicMock cursor with configurable fetchone/rowcount sequences."""
    cur = MagicMock()
    if fetchone_values:
        cur.fetchone.side_effect = fetchone_values
    if rowcount_values:
        # rowcount is a property, but MagicMock handles repeated attribute reads;
        # use a side_effect on a function to cycle through values
        type(cur).rowcount = _cycling_property(rowcount_values)
    return cur


def _cycling_property(values):
    """Return a property descriptor that cycles through *values* on each get."""
    it = iter(values)
    return property(lambda self: next(it))


# ---------- qident ----------

class TestQident:
    def test_simple_name(self):
        assert qident("my_table") == '"my_table"'

    def test_name_with_quotes(self):
        assert qident('table"name') == '"table""name"'


# ---------- typed_expr ----------

class TestTypedExpr:
    def test_int_field(self):
        result = typed_expr("qty", {"qty"}, set(), set(), "s")
        assert "::integer" in result
        assert "CASE WHEN" in result

    def test_float_field(self):
        result = typed_expr("amount", set(), {"amount"}, set(), "s")
        assert "::numeric" in result

    def test_date_field(self):
        result = typed_expr("startdate", set(), set(), {"startdate"}, "s")
        assert "::date" in result

    def test_text_field(self):
        result = typed_expr("name", set(), set(), set(), "s")
        assert "::integer" not in result
        assert "::numeric" not in result
        assert "::date" not in result
        assert 's."name"' in result


# ---------- _resolve_forecast_execution_lag ----------

class TestResolveForecastExecutionLag:
    def test_no_dim_dfu_defaults_to_zero(self):
        """When dim_dfu doesn't exist, all rows should default to execution_lag=0."""
        cur = MagicMock()
        cur.fetchone.return_value = (False,)  # dim_dfu doesn't exist
        cur.rowcount = 500

        matched, unmatched = _resolve_forecast_execution_lag(cur, "stg_test")

        assert matched == 0
        assert unmatched == 500
        # Should have checked for dim_dfu existence
        assert cur.execute.call_count == 2  # EXISTS check + UPDATE
        # Second call should be the blanket UPDATE to '0'
        second_call_sql = cur.execute.call_args_list[1][0][0]
        assert '"execution_lag"' in second_call_sql
        assert "'0'" in second_call_sql

    def test_with_dim_dfu_all_matched(self):
        """When dim_dfu exists and all DFUs match, should return (matched, 0)."""
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        # Track execute calls and set rowcount accordingly
        rowcounts = iter([None, 1000, 0])  # EXISTS, matched UPDATE, unmatched UPDATE

        def _set_rowcount(*args, **kwargs):
            cur.rowcount = next(rowcounts)

        cur.execute.side_effect = _set_rowcount

        matched, unmatched = _resolve_forecast_execution_lag(cur, "stg_test")

        assert matched == 1000
        assert unmatched == 0
        assert cur.execute.call_count == 3  # EXISTS + matched UPDATE + unmatched UPDATE

    def test_with_dim_dfu_partial_match(self):
        """Some DFUs match dim_dfu, others don't → both counts > 0."""
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        rowcounts = iter([None, 800, 200])

        def _set_rowcount(*args, **kwargs):
            cur.rowcount = next(rowcounts)

        cur.execute.side_effect = _set_rowcount

        matched, unmatched = _resolve_forecast_execution_lag(cur, "stg_test")

        assert matched == 800
        assert unmatched == 200

    def test_update_sql_joins_on_dfu_ck(self):
        """The UPDATE SQL should join staging with dim_dfu on dfu_ck."""
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        type(cur).rowcount = _cycling_property([None, 100, 0])

        _resolve_forecast_execution_lag(cur, "stg_forecast")

        # Check the matched UPDATE SQL references dim_dfu join
        matched_sql = cur.execute.call_args_list[1][0][0]
        assert "dim_dfu" in matched_sql
        assert "dfu_ck" in matched_sql
        assert "COALESCE" in matched_sql
        assert "execution_lag" in matched_sql

    def test_unmatched_update_defaults_to_zero(self):
        """Unmatched DFUs should have execution_lag set to '0'."""
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        type(cur).rowcount = _cycling_property([None, 50, 50])

        _resolve_forecast_execution_lag(cur, "stg_test")

        # Third execute call is the unmatched UPDATE
        unmatched_sql = cur.execute.call_args_list[2][0][0]
        assert "NOT EXISTS" in unmatched_sql
        assert "'0'" in unmatched_sql

    def test_staging_table_name_is_quoted(self):
        """The staging table name should be properly quoted in SQL."""
        cur = MagicMock()
        cur.fetchone.return_value = (False,)
        cur.rowcount = 0

        _resolve_forecast_execution_lag(cur, "stg_with_special")

        update_sql = cur.execute.call_args_list[1][0][0]
        assert '"stg_with_special"' in update_sql


# ---------- Phase ordering contract ----------

class TestPhaseOrdering:
    """The archive must be loaded BEFORE _resolve_forecast_execution_lag
    mutates the staging table, so the archive preserves each row's
    original lag as execution_lag."""

    def test_archive_load_reads_original_execution_lag(self):
        """Archive INSERT should read execution_lag from staging as-is
        (not the DFU-level value from dim_dfu)."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 5000])

        _load_forecast_archive(cur, "stg_test", "d")

        insert_sql = cur.execute.call_args_list[1][0][0]
        # Archive reads execution_lag directly from staging (stg_alias."execution_lag")
        assert '"execution_lag"' in insert_sql
        # It should NOT reference dim_dfu (that's the resolve function's job)
        assert "dim_dfu" not in insert_sql

    def test_resolve_mutates_staging(self):
        """_resolve_forecast_execution_lag UPDATEs the staging table —
        confirming it must run AFTER archive load."""
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        type(cur).rowcount = _cycling_property([None, 100, 50])

        _resolve_forecast_execution_lag(cur, "stg_test")

        update_sqls = [c[0][0] for c in cur.execute.call_args_list
                       if "UPDATE" in c[0][0]]
        assert len(update_sqls) >= 1, "Resolve must UPDATE staging"


# ---------- _load_forecast_archive ----------

class TestLoadForecastArchive:
    def test_deletes_existing_external_rows(self):
        """Should DELETE existing 'external' rows from archive before inserting."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([100, 5000])

        _load_forecast_archive(cur, "stg_test", "d")

        # First execute should be DELETE
        delete_sql = cur.execute.call_args_list[0][0][0]
        assert "DELETE" in delete_sql
        assert "backtest_lag_archive" in delete_sql
        assert "external" in delete_sql

    def test_inserts_all_lags_into_archive(self):
        """Should INSERT all staging rows (all lags) into backtest_lag_archive."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 45000])

        count = _load_forecast_archive(cur, "stg_test", "d")

        assert count == 45000
        # Second execute should be INSERT
        insert_sql = cur.execute.call_args_list[1][0][0]
        assert "INSERT INTO" in insert_sql
        assert "backtest_lag_archive" in insert_sql

    def test_uses_on_conflict_upsert(self):
        """Should use ON CONFLICT (forecast_ck, model_id, lag) DO UPDATE."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 1000])

        _load_forecast_archive(cur, "stg_test", "d")

        insert_sql = cur.execute.call_args_list[1][0][0]
        assert "ON CONFLICT" in insert_sql
        assert "forecast_ck" in insert_sql
        assert "model_id" in insert_sql
        assert "DO UPDATE" in insert_sql

    def test_timeframe_is_null(self):
        """External forecast archive rows should have timeframe=NULL."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 1000])

        _load_forecast_archive(cur, "stg_test", "d")

        insert_sql = cur.execute.call_args_list[1][0][0]
        # The SELECT should include NULL for timeframe
        assert "NULL" in insert_sql

    def test_builds_forecast_ck_from_key_fields(self):
        """forecast_ck should be built from dmdunit_dmdgroup_loc_fcstdate_startdate."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 1000])

        _load_forecast_archive(cur, "stg_test", "d")

        insert_sql = cur.execute.call_args_list[1][0][0]
        assert '"dmdunit"' in insert_sql
        assert '"dmdgroup"' in insert_sql
        assert '"loc"' in insert_sql
        assert '"fcstdate"' in insert_sql
        assert '"startdate"' in insert_sql

    def test_casts_numeric_fields(self):
        """lag, execution_lag should be cast to integer; basefcst_pref, tothist_dmd to numeric."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 1000])

        _load_forecast_archive(cur, "stg_test", "d")

        insert_sql = cur.execute.call_args_list[1][0][0]
        assert "::integer" in insert_sql
        assert "::numeric" in insert_sql
        assert "::date" in insert_sql

    def test_returns_inserted_row_count(self):
        """Should return the number of rows inserted into the archive."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([50, 12345])

        count = _load_forecast_archive(cur, "stg_test", "d")

        assert count == 12345

    def test_stg_alias_used_in_sql(self):
        """The staging alias parameter should be used in the SQL."""
        cur = MagicMock()
        type(cur).rowcount = _cycling_property([0, 100])

        _load_forecast_archive(cur, "stg_test", "myalias")

        insert_sql = cur.execute.call_args_list[1][0][0]
        assert 'myalias."dmdunit"' in insert_sql


# ---------- business_key_expr ----------

class TestBusinessKeyExpr:
    def test_single_key_field(self):
        spec = MagicMock()
        spec.key_fields = ["item_no"]
        spec.business_key_separator = "_"
        result = business_key_expr(spec, "s")
        assert 'trim(s."item_no")' in result

    def test_multi_key_field(self):
        spec = MagicMock()
        spec.key_fields = ["dmdunit", "dmdgroup", "loc"]
        spec.business_key_separator = "_"
        result = business_key_expr(spec, "s")
        assert "'_'" in result
        assert 'trim(s."dmdunit")' in result
        assert 'trim(s."loc")' in result
