"""Unit tests for common/dq_engine.py — Data Quality Engine (Spec 08-01).

Tests all check functions, DQEngine class methods, scoring, and error handling
using mocked database connections (no real DB required).
"""
from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

from common.dq_engine import (
    _check_freshness,
    _check_completeness,
    _check_row_count,
    _check_uniqueness,
    _check_range,
    _check_volume_delta,
    _check_referential_integrity,
    _check_statistical_outlier,
    _check_distribution_drift,
    _check_temporal_gaps,
    _check_cross_column,
    _check_cardinality_anomaly,
    CHECK_FUNCTIONS,
    DQEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_conn(fetchone_return=None, fetchall_return=None):
    """Build a mock psycopg connection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    if fetchone_return is not None:
        cursor.fetchone.return_value = fetchone_return
    if fetchall_return is not None:
        cursor.fetchall.return_value = fetchall_return
    return conn, cursor


# ---------------------------------------------------------------------------
# _check_freshness
# ---------------------------------------------------------------------------

class TestCheckFreshness:
    """Freshness checks compare against the planning date (2026-02-24), not wall-clock time."""

    _PLANNING_DATE = datetime(2026, 2, 24).date()

    def test_pass_recent_load(self):
        """Table loaded 1 hour before planning date passes a 48-hour threshold."""
        recent = datetime(2026, 2, 23, 23, 0, tzinfo=timezone.utc)
        conn, _ = _mock_conn(fetchone_return=(recent,))
        with patch("common.dq_engine.get_planning_date", return_value=self._PLANNING_DATE):
            result = _check_freshness(conn, "dim_item", max_hours=48)
        assert result["status"] == "pass"
        assert result["metric_value"] is not None
        assert result["metric_value"] < 2

    def test_fail_stale_load(self):
        """Table loaded 72 hours before planning date fails a 48-hour threshold."""
        stale = datetime(2026, 2, 21, 0, 0, tzinfo=timezone.utc)
        conn, _ = _mock_conn(fetchone_return=(stale,))
        with patch("common.dq_engine.get_planning_date", return_value=self._PLANNING_DATE):
            result = _check_freshness(conn, "fact_sales_monthly", max_hours=48)
        assert result["status"] == "fail"
        assert result["metric_value"] > 48

    def test_fail_no_load_ts(self):
        """No load_ts found returns fail with None metric."""
        conn, _ = _mock_conn(fetchone_return=(None,))
        with patch("common.dq_engine.get_planning_date", return_value=self._PLANNING_DATE):
            result = _check_freshness(conn, "dim_item", max_hours=48)
        assert result["status"] == "fail"
        assert result["metric_value"] is None
        assert "No load_ts" in result["details"]["message"]

    def test_fail_empty_row(self):
        """Empty fetchone result returns fail."""
        conn, _ = _mock_conn(fetchone_return=(None,))
        with patch("common.dq_engine.get_planning_date", return_value=self._PLANNING_DATE):
            result = _check_freshness(conn, "dim_item", max_hours=48)
        assert result["status"] == "fail"

    def test_pass_exactly_at_threshold(self):
        """Load_ts exactly at the threshold boundary passes."""
        boundary = datetime(2026, 2, 22, 0, 1, tzinfo=timezone.utc)  # ~47h59m before planning date
        conn, _ = _mock_conn(fetchone_return=(boundary,))
        with patch("common.dq_engine.get_planning_date", return_value=self._PLANNING_DATE):
            result = _check_freshness(conn, "dim_item", max_hours=48)
        assert result["status"] == "pass"

    def test_details_contain_hours_since_load_and_planning_date(self):
        """Details dict includes hours_since_load and planning_date."""
        recent = datetime(2026, 2, 23, 19, 0, tzinfo=timezone.utc)
        conn, _ = _mock_conn(fetchone_return=(recent,))
        with patch("common.dq_engine.get_planning_date", return_value=self._PLANNING_DATE):
            result = _check_freshness(conn, "dim_item", max_hours=48)
        assert "hours_since_load" in result["details"]
        assert isinstance(result["details"]["hours_since_load"], float)
        assert result["details"]["planning_date"] == "2026-02-24"


# ---------------------------------------------------------------------------
# _check_completeness
# ---------------------------------------------------------------------------

class TestCheckCompleteness:
    def test_pass_low_null_pct(self):
        """Column with 1% nulls passes a 5% threshold."""
        conn, _ = _mock_conn(fetchone_return=(1000, 10))
        result = _check_completeness(conn, "dim_item", "description", max_null_pct=5.0)
        assert result["status"] == "pass"
        assert result["metric_value"] == 1.0  # 10/1000 * 100

    def test_fail_high_null_pct(self):
        """Column with 20% nulls fails a 5% threshold."""
        conn, _ = _mock_conn(fetchone_return=(100, 20))
        result = _check_completeness(conn, "dim_item", "brand", max_null_pct=5.0)
        assert result["status"] == "fail"
        assert result["metric_value"] == 20.0

    def test_warn_empty_table(self):
        """Empty table returns warn status."""
        conn, _ = _mock_conn(fetchone_return=(0, 0))
        result = _check_completeness(conn, "dim_item", "item_no", max_null_pct=0.0)
        assert result["status"] == "warn"
        assert result["details"]["message"] == "Empty table"

    def test_pass_zero_nulls(self):
        """Column with 0 nulls passes any threshold."""
        conn, _ = _mock_conn(fetchone_return=(5000, 0))
        result = _check_completeness(conn, "fact_sales_monthly", "qty", max_null_pct=0.0)
        assert result["status"] == "pass"
        assert result["metric_value"] == 0.0

    def test_details_contain_total_and_nulls(self):
        """Details include total, nulls, and null_pct."""
        conn, _ = _mock_conn(fetchone_return=(200, 6))
        result = _check_completeness(conn, "dim_item", "brand", max_null_pct=5.0)
        assert result["details"]["total"] == 200
        assert result["details"]["nulls"] == 6
        assert result["details"]["null_pct"] == 3.0

    @pytest.mark.parametrize("total,nulls,threshold,expected_status", [
        (100, 0, 0.0, "pass"),
        (100, 5, 5.0, "pass"),     # exactly at threshold
        (100, 6, 5.0, "fail"),     # just over
        (100, 100, 50.0, "fail"),  # all nulls
    ])
    def test_boundary_cases(self, total, nulls, threshold, expected_status):
        conn, _ = _mock_conn(fetchone_return=(total, nulls))
        result = _check_completeness(conn, "t", "c", max_null_pct=threshold)
        assert result["status"] == expected_status


# ---------------------------------------------------------------------------
# _check_row_count
# ---------------------------------------------------------------------------

class TestCheckRowCount:
    def test_pass_above_min(self):
        """Table with 10k rows passes min_rows=1000."""
        conn, _ = _mock_conn(fetchone_return=(10000,))
        result = _check_row_count(conn, "fact_sales_monthly", min_rows=1000)
        assert result["status"] == "pass"
        assert result["metric_value"] == 10000

    def test_fail_below_min(self):
        """Empty table fails min_rows=1."""
        conn, _ = _mock_conn(fetchone_return=(0,))
        result = _check_row_count(conn, "dim_item", min_rows=1)
        assert result["status"] == "fail"
        assert result["metric_value"] == 0

    def test_pass_exactly_at_min(self):
        """Count exactly at min_rows passes."""
        conn, _ = _mock_conn(fetchone_return=(100,))
        result = _check_row_count(conn, "dim_item", min_rows=100)
        assert result["status"] == "pass"

    def test_default_min_rows_is_one(self):
        """Default min_rows parameter is 1."""
        conn, _ = _mock_conn(fetchone_return=(1,))
        result = _check_row_count(conn, "dim_item")
        assert result["status"] == "pass"

    def test_details_contain_row_count_and_min(self):
        conn, _ = _mock_conn(fetchone_return=(42,))
        result = _check_row_count(conn, "dim_item", min_rows=50)
        assert result["details"]["row_count"] == 42
        assert result["details"]["min_expected"] == 50


# ---------------------------------------------------------------------------
# _check_uniqueness
# ---------------------------------------------------------------------------

class TestCheckUniqueness:
    def test_pass_no_dupes(self):
        """No duplicate groups → pass."""
        conn, _ = _mock_conn(fetchone_return=(0,))
        result = _check_uniqueness(conn, "dim_item", key_columns=["item_no"])
        assert result["status"] == "pass"
        assert result["metric_value"] == 0

    def test_fail_has_dupes(self):
        """5 duplicate groups → fail."""
        conn, _ = _mock_conn(fetchone_return=(5,))
        result = _check_uniqueness(conn, "dim_dfu", key_columns=["dmdunit", "loc"])
        assert result["status"] == "fail"
        assert result["metric_value"] == 5

    def test_details_contain_duplicate_groups(self):
        conn, _ = _mock_conn(fetchone_return=(3,))
        result = _check_uniqueness(conn, "dim_item", key_columns=["item_no"])
        assert result["details"]["duplicate_groups"] == 3

    def test_sql_uses_correct_columns(self):
        """Verify the SQL groups by the requested key columns."""
        conn, cursor = _mock_conn(fetchone_return=(0,))
        _check_uniqueness(conn, "fact_sales_monthly", key_columns=["dmdunit", "loc", "startdate"])
        sql = cursor.execute.call_args[0][0]
        assert "dmdunit, loc, startdate" in sql


# ---------------------------------------------------------------------------
# CHECK_FUNCTIONS registry
# ---------------------------------------------------------------------------

class TestCheckFunctionsRegistry:
    def test_all_twelve_types_registered(self):
        assert set(CHECK_FUNCTIONS.keys()) == {
            "freshness", "completeness", "row_count", "uniqueness",
            "range", "volume_delta", "referential_integrity",
            "statistical_outlier", "distribution_drift", "temporal_gaps",
            "cross_column", "cardinality_anomaly",
        }

    def test_functions_are_callable(self):
        for fn in CHECK_FUNCTIONS.values():
            assert callable(fn)


# ---------------------------------------------------------------------------
# DQEngine._run_single
# ---------------------------------------------------------------------------

class TestRunSingle:
    def _make_engine(self, checks=None):
        """Create DQEngine with a mocked config."""
        with patch("common.dq_engine._load_config", return_value={"checks": checks or []}):
            return DQEngine()

    def test_freshness_check(self):
        engine = self._make_engine()
        # 2 hours before the planning date
        recent = datetime(2026, 2, 23, 22, 0, tzinfo=timezone.utc)
        conn, _ = _mock_conn(fetchone_return=(recent,))
        check = {
            "check_name": "test_fresh",
            "check_type": "freshness",
            "table_name": "dim_item",
            "max_hours": 48,
            "domain": "item",
            "severity": "warning",
        }
        with patch("common.dq_engine.get_planning_date", return_value=datetime(2026, 2, 24).date()):
            result = engine._run_single(conn, check)
        assert result["check_name"] == "test_fresh"
        assert result["status"] == "pass"
        assert result["domain"] == "item"
        assert result["severity"] == "warning"

    def test_unknown_check_type_returns_skip(self):
        engine = self._make_engine()
        conn, _ = _mock_conn()
        check = {"check_name": "x", "check_type": "nonexistent", "table_name": "t"}
        result = engine._run_single(conn, check)
        assert result["status"] == "skip"
        assert "Unknown check type" in result["details"]["message"]

    def test_sql_error_returns_error_status(self):
        engine = self._make_engine()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("relation does not exist")
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        check = {"check_name": "bad", "check_type": "row_count", "table_name": "missing_table"}
        result = engine._run_single(conn, check)
        assert result["status"] == "error"
        assert "relation does not exist" in result["details"]["error"]

    def test_completeness_check_via_run_single(self):
        engine = self._make_engine()
        conn, _ = _mock_conn(fetchone_return=(1000, 50))
        check = {
            "check_name": "null_check",
            "check_type": "completeness",
            "table_name": "dim_item",
            "column": "brand",
            "max_null_pct": 10.0,
            "domain": "item",
            "severity": "warning",
        }
        result = engine._run_single(conn, check)
        assert result["status"] == "pass"
        assert result["metric_value"] == 5.0

    def test_uniqueness_check_via_run_single(self):
        engine = self._make_engine()
        conn, _ = _mock_conn(fetchone_return=(2,))
        check = {
            "check_name": "dupe_check",
            "check_type": "uniqueness",
            "table_name": "dim_dfu",
            "key_columns": ["dmdunit", "loc"],
            "domain": "dfu",
            "severity": "critical",
        }
        result = engine._run_single(conn, check)
        assert result["status"] == "fail"
        assert result["metric_value"] == 2


# ---------------------------------------------------------------------------
# DQEngine.run_all_checks
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_runs_enabled_checks_only(self):
        checks = [
            {"check_name": "c1", "check_type": "row_count", "table_name": "t1", "enabled": True, "domain": "d"},
            {"check_name": "c2", "check_type": "row_count", "table_name": "t2", "enabled": False, "domain": "d"},
        ]
        with patch("common.dq_engine._load_config", return_value={"checks": checks}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            results = engine.run_all_checks()

        # Only c1 should run (c2 disabled)
        assert len(results) == 1
        assert results[0]["check_name"] == "c1"

    def test_filters_by_domain(self):
        checks = [
            {"check_name": "c1", "check_type": "row_count", "table_name": "t1", "domain": "sales"},
            {"check_name": "c2", "check_type": "row_count", "table_name": "t2", "domain": "item"},
        ]
        with patch("common.dq_engine._load_config", return_value={"checks": checks}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            results = engine.run_all_checks(domain="item")

        assert len(results) == 1
        assert results[0]["domain"] == "item"

    def test_records_results_to_db(self):
        checks = [
            {"check_name": "c1", "check_type": "row_count", "table_name": "t1", "domain": "d"},
        ]
        with patch("common.dq_engine._load_config", return_value={"checks": checks}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # psycopg.connect() is used as context manager: `with connect() as conn:`
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            engine.run_all_checks()

        # autocommit=True so no explicit commit needed
        # cursor.execute called at least twice: the check + the record INSERT
        assert cursor.execute.call_count >= 2

    def test_empty_checks_list(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        mock_conn = MagicMock()
        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            results = engine.run_all_checks()

        assert results == []


# ---------------------------------------------------------------------------
# DQEngine._record_result (best-effort)
# ---------------------------------------------------------------------------

class TestRecordResult:
    def test_record_result_inserts_row(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()
        conn, cursor = _mock_conn()
        result = {
            "check_name": "test",
            "domain": "item",
            "table_name": "dim_item",
            "severity": "warning",
            "status": "pass",
            "metric_value": 42.0,
            "details": {"foo": "bar"},
        }
        engine._record_result(conn, result)
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO fact_dq_check_results" in sql

    def test_record_result_swallows_exceptions(self):
        """Best-effort recording should not raise."""
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("table not found")
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = {"check_name": "x", "status": "pass", "metric_value": 1}
        # Should not raise
        engine._record_result(conn, result)


# ---------------------------------------------------------------------------
# DQEngine.get_domain_score
# ---------------------------------------------------------------------------

class TestGetDomainScore:
    def test_computes_score_from_results(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        # 8 pass, 2 fail
        cursor.fetchall.return_value = [("pass", 8), ("fail", 2)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            score = engine.get_domain_score("item")

        assert score["domain"] == "item"
        assert score["score"] == 80.0
        assert score["total_checks"] == 10
        assert score["passed"] == 8
        assert score["failed"] == 2

    def test_perfect_score_all_pass(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [("pass", 5)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            score = engine.get_domain_score("sales")

        assert score["score"] == 100.0
        assert score["failed"] == 0

    def test_no_checks_returns_100(self):
        """No recent checks defaults to 100.0."""
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            score = engine.get_domain_score("unknown")

        assert score["score"] == 100.0
        assert score["total_checks"] == 0

    def test_includes_warn_in_total(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [("pass", 3), ("fail", 1), ("warn", 2)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}):
            score = engine.get_domain_score("item")

        assert score["total_checks"] == 6
        assert score["score"] == 50.0  # 3/6 * 100
        assert score["warnings"] == 2


# ---------------------------------------------------------------------------
# DQEngine.get_pipeline_health
# ---------------------------------------------------------------------------

class TestGetPipelineHealth:
    def test_returns_all_five_tables(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        recent = datetime(2026, 2, 23, 23, 0, tzinfo=timezone.utc)
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (recent,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}), \
             patch("common.dq_engine.get_planning_date", return_value=datetime(2026, 2, 24).date()):
            health = engine.get_pipeline_health()

        assert len(health["tables"]) == 5
        expected_tables = {"dim_item", "dim_location", "dim_dfu", "fact_sales_monthly", "fact_external_forecast_monthly"}
        actual_tables = {t["table"] for t in health["tables"]}
        assert actual_tables == expected_tables

    def test_handles_table_error_gracefully(self):
        """If a table check raises, that table gets error status."""
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        call_count = 0

        def side_effect_execute(sql):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("permission denied")

        mock_conn = MagicMock()
        cursor = MagicMock()
        recent = datetime(2026, 2, 23, 23, 0, tzinfo=timezone.utc)
        cursor.execute.side_effect = side_effect_execute
        cursor.fetchone.return_value = (recent,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn), \
             patch("common.dq_engine.get_db_params", return_value={}), \
             patch("common.dq_engine.get_planning_date", return_value=datetime(2026, 2, 24).date()):
            health = engine.get_pipeline_health()

        # First table should have error, rest should succeed
        assert health["tables"][0]["status"] == "error"
        assert len(health["tables"]) == 5


# ---------------------------------------------------------------------------
# Connection / initialization errors
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_run_all_checks_connection_failure(self):
        """Connection error propagates from run_all_checks."""
        checks = [{"check_name": "c1", "check_type": "row_count", "table_name": "t", "domain": "d"}]
        with patch("common.dq_engine._load_config", return_value={"checks": checks}):
            engine = DQEngine()

        with patch("psycopg.connect", side_effect=Exception("connection refused")), \
             patch("common.dq_engine.get_db_params", return_value={}):
            with pytest.raises(Exception, match="connection refused"):
                engine.run_all_checks()

    def test_get_domain_score_connection_failure(self):
        with patch("common.dq_engine._load_config", return_value={"checks": []}):
            engine = DQEngine()

        with patch("psycopg.connect", side_effect=Exception("timeout")), \
             patch("common.dq_engine.get_db_params", return_value={}):
            with pytest.raises(Exception, match="timeout"):
                engine.get_domain_score("item")


# ---------------------------------------------------------------------------
# _check_range
# ---------------------------------------------------------------------------

class TestCheckRange:
    def test_pass_no_outliers(self):
        conn, _ = _mock_conn(fetchone_return=(1000, 0))
        result = _check_range(conn, "fact_sales_monthly", "qty", 0, 10000000)
        assert result["status"] == "pass"
        assert result["metric_value"] == 0

    def test_fail_with_outliers(self):
        conn, _ = _mock_conn(fetchone_return=(1000, 5))
        result = _check_range(conn, "fact_sales_monthly", "qty", 0, 10000000)
        assert result["status"] == "fail"
        assert result["metric_value"] == 5
        assert result["details"]["outliers"] == 5
        assert result["details"]["outlier_pct"] == 0.5

    def test_skip_no_bounds(self):
        conn, _ = _mock_conn()
        result = _check_range(conn, "t", "c", None, None)
        assert result["status"] == "skip"

    def test_warn_empty_table(self):
        conn, _ = _mock_conn(fetchone_return=(0, 0))
        result = _check_range(conn, "t", "c", 0, 100)
        assert result["status"] == "warn"

    def test_min_only(self):
        conn, _ = _mock_conn(fetchone_return=(500, 3))
        result = _check_range(conn, "t", "c", 0, None)
        assert result["status"] == "fail"
        assert result["details"]["min"] == 0
        assert result["details"]["max"] is None

    def test_max_only(self):
        conn, _ = _mock_conn(fetchone_return=(500, 0))
        result = _check_range(conn, "t", "c", None, 1000)
        assert result["status"] == "pass"

    def test_details_structure(self):
        conn, _ = _mock_conn(fetchone_return=(200, 10))
        result = _check_range(conn, "t", "c", 0, 100)
        d = result["details"]
        assert d["total"] == 200
        assert d["outliers"] == 10
        assert d["outlier_pct"] == 5.0
        assert d["min"] == 0
        assert d["max"] == 100


# ---------------------------------------------------------------------------
# _check_volume_delta
# ---------------------------------------------------------------------------

class TestCheckVolumeDelta:
    def test_pass_small_change(self):
        conn, _ = _mock_conn(fetchall_return=[
            (datetime(2026, 3, 1).date(), 1000),
            (datetime(2026, 2, 1).date(), 950),
        ])
        result = _check_volume_delta(conn, "fact_sales_monthly", 50.0)
        assert result["status"] == "pass"
        assert result["metric_value"] < 50.0

    def test_fail_large_change(self):
        conn, _ = _mock_conn(fetchall_return=[
            (datetime(2026, 3, 1).date(), 2000),
            (datetime(2026, 2, 1).date(), 1000),
        ])
        result = _check_volume_delta(conn, "t", 50.0)
        assert result["status"] == "fail"
        assert result["metric_value"] == 100.0

    def test_skip_single_load(self):
        conn, _ = _mock_conn(fetchall_return=[
            (datetime(2026, 3, 1).date(), 1000),
        ])
        result = _check_volume_delta(conn, "t", 50.0)
        assert result["status"] == "skip"

    def test_skip_no_loads(self):
        conn, _ = _mock_conn(fetchall_return=[])
        result = _check_volume_delta(conn, "t", 50.0)
        assert result["status"] == "skip"

    def test_warn_prev_zero(self):
        conn, _ = _mock_conn(fetchall_return=[
            (datetime(2026, 3, 1).date(), 100),
            (datetime(2026, 2, 1).date(), 0),
        ])
        result = _check_volume_delta(conn, "t", 50.0)
        assert result["status"] == "warn"

    def test_details_contain_counts(self):
        conn, _ = _mock_conn(fetchall_return=[
            (datetime(2026, 3, 1).date(), 1200),
            (datetime(2026, 2, 1).date(), 1000),
        ])
        result = _check_volume_delta(conn, "t", 50.0)
        d = result["details"]
        assert d["latest_count"] == 1200
        assert d["prev_count"] == 1000
        assert d["pct_change"] == 20.0
        assert d["max_pct_change"] == 50.0


# ---------------------------------------------------------------------------
# _check_referential_integrity
# ---------------------------------------------------------------------------

class TestCheckReferentialIntegrity:
    def test_pass_no_orphans(self):
        conn, _ = _mock_conn(fetchone_return=(0,))
        result = _check_referential_integrity(
            conn, "fact_sales_monthly", ["dmdunit", "loc"],
            "dim_dfu", ["dmdunit", "loc"],
        )
        assert result["status"] == "pass"
        assert result["metric_value"] == 0

    def test_fail_with_orphans(self):
        conn, _ = _mock_conn(fetchone_return=(42,))
        result = _check_referential_integrity(
            conn, "fact_sales_monthly", ["dmdunit", "loc"],
            "dim_dfu", ["dmdunit", "loc"],
        )
        assert result["status"] == "fail"
        assert result["metric_value"] == 42
        assert result["details"]["orphan_keys"] == 42

    def test_single_column_fk(self):
        conn, _ = _mock_conn(fetchone_return=(0,))
        result = _check_referential_integrity(
            conn, "fact_inventory_snapshot", ["item_no"],
            "dim_item", ["item_no"],
        )
        assert result["status"] == "pass"

    def test_details_contain_table_names(self):
        conn, _ = _mock_conn(fetchone_return=(5,))
        result = _check_referential_integrity(
            conn, "src_table", ["col_a"],
            "tgt_table", ["col_b"],
        )
        assert result["details"]["source_table"] == "src_table"
        assert result["details"]["target_table"] == "tgt_table"


# ---------------------------------------------------------------------------
# _run_single dispatch for new check types
# ---------------------------------------------------------------------------

class TestRunSingleNewTypes:
    def _make_engine(self, checks=None):
        with patch("common.dq_engine._load_config", return_value={"checks": checks or []}):
            return DQEngine()

    def test_range_dispatch(self):
        engine = self._make_engine()
        conn, _ = _mock_conn(fetchone_return=(100, 0))
        result = engine._run_single(conn, {
            "check_type": "range",
            "check_name": "range_sales_qty",
            "table_name": "fact_sales_monthly",
            "column": "qty",
            "min": 0,
            "max": 10000000,
        })
        assert result["status"] == "pass"
        assert result["check_name"] == "range_sales_qty"

    def test_volume_delta_dispatch(self):
        engine = self._make_engine()
        conn, _ = _mock_conn(fetchall_return=[
            (datetime(2026, 3, 1).date(), 1000),
            (datetime(2026, 2, 1).date(), 950),
        ])
        result = engine._run_single(conn, {
            "check_type": "volume_delta",
            "check_name": "volume_delta_sales",
            "table_name": "fact_sales_monthly",
            "max_pct_change": 50.0,
        })
        assert result["status"] == "pass"

    def test_referential_integrity_dispatch(self):
        engine = self._make_engine()
        conn, _ = _mock_conn(fetchone_return=(0,))
        result = engine._run_single(conn, {
            "check_type": "referential_integrity",
            "check_name": "ri_sales_to_dfu",
            "table_name": "fact_sales_monthly",
            "source_table": "fact_sales_monthly",
            "source_columns": ["dmdunit", "loc"],
            "target_table": "dim_dfu",
            "target_columns": ["dmdunit", "loc"],
        })
        assert result["status"] == "pass"


# ---------------------------------------------------------------------------
# _flatten_checks — config parsing
# ---------------------------------------------------------------------------

class TestFlattenChecks:
    def _make_engine(self, config: dict):
        with patch("common.dq_engine._load_config", return_value=config):
            return DQEngine()

    def test_flat_list_passthrough(self):
        """If checks is already a flat list, return it as-is."""
        flat_checks = [{"check_type": "row_count", "table_name": "t"}]
        engine = self._make_engine({"checks": flat_checks})
        result = engine._flatten_checks()
        assert result == flat_checks

    def test_freshness_flattened(self):
        config = {
            "checks": {
                "freshness": {
                    "sales": {
                        "table": "fact_sales_monthly",
                        "max_hours_since_load": 48,
                        "severity": "critical",
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        c = flat[0]
        assert c["check_type"] == "freshness"
        assert c["check_name"] == "freshness_sales"
        assert c["domain"] == "sales"
        assert c["table_name"] == "fact_sales_monthly"
        assert c["max_hours"] == 48
        assert c["severity"] == "critical"

    def test_completeness_per_column(self):
        config = {
            "checks": {
                "completeness": {
                    "item": {
                        "table": "dim_item",
                        "columns": [
                            {"column": "item_no", "null_pct_threshold": 0.0, "severity": "critical"},
                            {"column": "brand", "null_pct_threshold": 10.0, "severity": "warning"},
                        ],
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 2
        assert flat[0]["check_name"] == "completeness_item_item_no"
        assert flat[0]["column"] == "item_no"
        assert flat[0]["max_null_pct"] == 0.0
        assert flat[1]["check_name"] == "completeness_item_brand"
        assert flat[1]["max_null_pct"] == 10.0

    def test_uniqueness_flattened(self):
        config = {
            "checks": {
                "uniqueness": {
                    "dfu": {
                        "table": "dim_dfu",
                        "key_columns": ["dmdunit", "loc"],
                        "severity": "critical",
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        assert flat[0]["key_columns"] == ["dmdunit", "loc"]

    def test_range_per_column(self):
        config = {
            "checks": {
                "range": {
                    "sales": {
                        "table": "fact_sales_monthly",
                        "columns": [
                            {"column": "qty", "min": 0, "max": 10000000, "severity": "warning"},
                        ],
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        assert flat[0]["check_type"] == "range"
        assert flat[0]["column"] == "qty"
        assert flat[0]["min"] == 0
        assert flat[0]["max"] == 10000000

    def test_volume_delta_flattened(self):
        config = {
            "checks": {
                "volume_delta": {
                    "sales": {
                        "table": "fact_sales_monthly",
                        "max_pct_change": 50.0,
                        "severity": "warning",
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        assert flat[0]["check_type"] == "volume_delta"
        assert flat[0]["max_pct_change"] == 50.0

    def test_referential_integrity_flattened(self):
        config = {
            "checks": {
                "referential_integrity": {
                    "sales_to_dfu": {
                        "source_table": "fact_sales_monthly",
                        "source_columns": ["dmdunit", "loc"],
                        "target_table": "dim_dfu",
                        "target_columns": ["dmdunit", "loc"],
                        "severity": "warning",
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        c = flat[0]
        assert c["check_type"] == "referential_integrity"
        assert c["source_table"] == "fact_sales_monthly"
        assert c["target_table"] == "dim_dfu"
        assert c["source_columns"] == ["dmdunit", "loc"]
        assert c["target_columns"] == ["dmdunit", "loc"]

    def test_global_defaults_applied(self):
        config = {
            "global_defaults": {"severity": "info", "enabled": True},
            "checks": {
                "freshness": {
                    "x": {"table": "t", "max_hours_since_load": 24}
                }
            },
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert flat[0]["severity"] == "info"

    def test_empty_nested_checks(self):
        engine = self._make_engine({"checks": {}})
        flat = engine._flatten_checks()
        assert flat == []

    def test_mixed_check_types(self):
        """Multiple check types in one config produce correct total."""
        config = {
            "checks": {
                "freshness": {
                    "a": {"table": "t1", "max_hours_since_load": 48},
                },
                "uniqueness": {
                    "b": {"table": "t2", "key_columns": ["id"]},
                },
                "volume_delta": {
                    "c": {"table": "t3", "max_pct_change": 25.0},
                },
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 3
        types = {c["check_type"] for c in flat}
        assert types == {"freshness", "uniqueness", "volume_delta"}

    def test_statistical_outlier_per_column(self):
        config = {
            "checks": {
                "statistical_outlier": {
                    "sales": {
                        "table": "fact_sales_monthly",
                        "columns": [
                            {"column": "qty", "method": "iqr", "threshold": 1.5},
                            {"column": "qty_shipped", "method": "zscore", "threshold": 3.0},
                        ],
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 2
        assert flat[0]["check_name"] == "statistical_outlier_sales_qty"
        assert flat[0]["method"] == "iqr"
        assert flat[1]["check_name"] == "statistical_outlier_sales_qty_shipped"
        assert flat[1]["method"] == "zscore"
        assert flat[1]["threshold"] == 3.0

    def test_distribution_drift_per_column(self):
        config = {
            "checks": {
                "distribution_drift": {
                    "sales": {
                        "table": "fact_sales_monthly",
                        "columns": [{"column": "qty", "max_drift": 0.15}],
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        assert flat[0]["check_type"] == "distribution_drift"
        assert flat[0]["max_drift"] == 0.15

    def test_temporal_gaps_flattened(self):
        config = {
            "checks": {
                "temporal_gaps": {
                    "sales": {"table": "fact_sales_monthly", "date_column": "startdate", "grain": "month"},
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        assert flat[0]["check_type"] == "temporal_gaps"
        assert flat[0]["date_column"] == "startdate"
        assert flat[0]["grain"] == "month"

    def test_cross_column_per_rule(self):
        config = {
            "checks": {
                "cross_column": {
                    "sales": {
                        "table": "fact_sales_monthly",
                        "rules": [
                            {"name": "qty_pos", "expression": "qty >= 0", "description": "Qty non-negative"},
                            {"name": "shipped_ok", "expression": "qty_shipped <= qty_ordered OR qty_ordered IS NULL"},
                        ],
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 2
        assert flat[0]["check_name"] == "cross_column_sales_qty_pos"
        assert flat[0]["rule"] == "qty >= 0"
        assert flat[1]["check_name"] == "cross_column_sales_shipped_ok"

    def test_cardinality_anomaly_per_column(self):
        config = {
            "checks": {
                "cardinality_anomaly": {
                    "item": {
                        "table": "dim_item",
                        "columns": [
                            {"column": "category", "max_change_pct": 15.0},
                        ],
                    }
                }
            }
        }
        engine = self._make_engine(config)
        flat = engine._flatten_checks()
        assert len(flat) == 1
        assert flat[0]["check_type"] == "cardinality_anomaly"
        assert flat[0]["column"] == "category"
        assert flat[0]["max_change_pct"] == 15.0


# ---------------------------------------------------------------------------
# _check_statistical_outlier
# ---------------------------------------------------------------------------

class TestCheckStatisticalOutlier:
    def test_iqr_pass_no_outliers(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # First call: stats; second call: count outliers
        cursor.fetchone.side_effect = [
            (1000, 50.0, 10.0, 40.0, 50.0, 60.0, 0.0, 100.0),  # stats
            (0,),  # outlier count
        ]
        result = _check_statistical_outlier(conn, "t", "c", method="iqr", threshold=1.5)
        assert result["status"] == "pass"
        assert result["metric_value"] == 0
        assert result["details"]["method"] == "iqr"

    def test_iqr_fail_many_outliers(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            (1000, 50.0, 10.0, 40.0, 50.0, 60.0, 0.0, 100.0),
            (60,),  # 6% outliers → fail
        ]
        result = _check_statistical_outlier(conn, "t", "c", method="iqr", threshold=1.5)
        assert result["status"] == "fail"
        assert result["details"]["outlier_pct"] == 6.0

    def test_zscore_method(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            (1000, 100.0, 20.0, 80.0, 100.0, 120.0, 10.0, 200.0),
            (5,),  # 0.5% → pass
        ]
        result = _check_statistical_outlier(conn, "t", "c", method="zscore", threshold=3.0)
        assert result["status"] == "pass"
        assert result["details"]["method"] == "zscore"

    def test_warn_empty_table(self):
        conn, _ = _mock_conn(fetchone_return=(0, None, None, None, None, None, None, None))
        result = _check_statistical_outlier(conn, "t", "c")
        assert result["status"] == "warn"

    def test_warn_moderate_outliers(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            (1000, 50.0, 10.0, 40.0, 50.0, 60.0, 0.0, 100.0),
            (20,),  # 2% → warn (between 1% and 5%)
        ]
        result = _check_statistical_outlier(conn, "t", "c")
        assert result["status"] == "warn"


# ---------------------------------------------------------------------------
# _check_distribution_drift
# ---------------------------------------------------------------------------

class TestCheckDistributionDrift:
    def test_pass_no_drift(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        d1 = datetime(2026, 3, 2).date()
        d2 = datetime(2026, 3, 1).date()
        cursor.fetchall.return_value = [(d1,), (d2,)]
        cursor.fetchone.side_effect = [
            (500, 100.0, 10.0, 100.0),  # latest
            (500, 99.0, 10.0, 99.0),    # previous — tiny shift
        ]
        result = _check_distribution_drift(conn, "t", "c", max_drift=0.1)
        assert result["status"] == "pass"
        assert result["details"]["drift_score"] < 0.1

    def test_fail_large_drift(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        d1 = datetime(2026, 3, 2).date()
        d2 = datetime(2026, 3, 1).date()
        cursor.fetchall.return_value = [(d1,), (d2,)]
        cursor.fetchone.side_effect = [
            (500, 200.0, 30.0, 200.0),  # latest — big shift
            (500, 100.0, 10.0, 100.0),  # previous
        ]
        result = _check_distribution_drift(conn, "t", "c", max_drift=0.1)
        assert result["status"] == "fail"
        assert result["details"]["drift_score"] > 0.1

    def test_skip_single_batch(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = [(datetime(2026, 3, 1).date(),)]
        result = _check_distribution_drift(conn, "t", "c")
        assert result["status"] == "skip"


# ---------------------------------------------------------------------------
# _check_temporal_gaps
# ---------------------------------------------------------------------------

class TestCheckTemporalGaps:
    def test_pass_no_gaps(self):
        from datetime import date
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = [
            (date(2026, 1, 1),), (date(2026, 2, 1),), (date(2026, 3, 1),),
        ]
        result = _check_temporal_gaps(conn, "t", "startdate", grain="month")
        assert result["status"] == "pass"
        assert result["metric_value"] == 0

    def test_fail_with_gaps(self):
        from datetime import date
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # Missing Feb-May 2026 (4 gaps > 3 threshold → fail)
        cursor.fetchall.return_value = [
            (date(2026, 1, 1),), (date(2026, 6, 1),),
        ]
        result = _check_temporal_gaps(conn, "t", "startdate", grain="month")
        assert result["status"] == "fail"
        assert result["metric_value"] == 4
        assert "2026-02-01" in result["details"]["missing_periods"]
        assert "2026-03-01" in result["details"]["missing_periods"]

    def test_skip_single_period(self):
        from datetime import date
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = [(date(2026, 1, 1),)]
        result = _check_temporal_gaps(conn, "t", "startdate")
        assert result["status"] == "skip"

    def test_warn_small_gap(self):
        from datetime import date
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # Missing just Feb
        cursor.fetchall.return_value = [
            (date(2026, 1, 1),), (date(2026, 3, 1),),
        ]
        result = _check_temporal_gaps(conn, "t", "startdate", grain="month")
        assert result["status"] == "warn"
        assert result["metric_value"] == 1


# ---------------------------------------------------------------------------
# _check_cross_column
# ---------------------------------------------------------------------------

class TestCheckCrossColumn:
    def test_pass_no_violations(self):
        conn, _ = _mock_conn(fetchone_return=(1000, 0))
        result = _check_cross_column(conn, "fact_sales_monthly", "qty >= 0", "Qty non-negative")
        assert result["status"] == "pass"
        assert result["metric_value"] == 0

    def test_fail_many_violations(self):
        conn, _ = _mock_conn(fetchone_return=(1000, 50))
        result = _check_cross_column(conn, "t", "qty >= 0")
        assert result["status"] == "fail"
        assert result["metric_value"] == 50
        assert result["details"]["violation_pct"] == 5.0

    def test_warn_few_violations(self):
        conn, _ = _mock_conn(fetchone_return=(10000, 5))
        result = _check_cross_column(conn, "t", "qty >= 0")
        assert result["status"] == "warn"  # 0.05% < 1%

    def test_warn_empty_table(self):
        conn, _ = _mock_conn(fetchone_return=(0, 0))
        result = _check_cross_column(conn, "t", "TRUE")
        assert result["status"] == "warn"

    def test_details_contain_rule(self):
        conn, _ = _mock_conn(fetchone_return=(100, 2))
        result = _check_cross_column(conn, "t", "a <= b", "A should be <= B")
        assert result["details"]["rule"] == "a <= b"
        assert result["details"]["description"] == "A should be <= B"


# ---------------------------------------------------------------------------
# _check_cardinality_anomaly
# ---------------------------------------------------------------------------

class TestCheckCardinalityAnomaly:
    def test_pass_small_change(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        d1 = datetime(2026, 3, 2).date()
        d2 = datetime(2026, 3, 1).date()
        cursor.fetchall.return_value = [(d1,), (d2,)]
        cursor.fetchone.side_effect = [
            (100,),  # latest distinct
            (98,),   # prev distinct
            (2,),    # new values
            (0,),    # dropped values
        ]
        result = _check_cardinality_anomaly(conn, "dim_item", "category", max_change_pct=10.0)
        assert result["status"] == "pass"
        assert result["details"]["new_values"] == 2
        assert result["details"]["dropped_values"] == 0

    def test_fail_large_change(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        d1 = datetime(2026, 3, 2).date()
        d2 = datetime(2026, 3, 1).date()
        cursor.fetchall.return_value = [(d1,), (d2,)]
        cursor.fetchone.side_effect = [
            (50,),   # latest distinct
            (100,),  # prev distinct
            (10,),   # new values
            (60,),   # dropped values — 70% change
        ]
        result = _check_cardinality_anomaly(conn, "dim_item", "category", max_change_pct=10.0)
        assert result["status"] == "fail"
        assert result["details"]["change_pct"] == 70.0

    def test_skip_single_batch(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = [(datetime(2026, 3, 1).date(),)]
        result = _check_cardinality_anomaly(conn, "t", "c")
        assert result["status"] == "skip"


# ---------------------------------------------------------------------------
# _run_single dispatch for new statistical check types
# ---------------------------------------------------------------------------

class TestRunSingleStatisticalTypes:
    def _make_engine(self, checks=None):
        with patch("common.dq_engine._load_config", return_value={"checks": checks or []}):
            return DQEngine()

    def test_statistical_outlier_dispatch(self):
        engine = self._make_engine()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.side_effect = [
            (1000, 50.0, 10.0, 40.0, 50.0, 60.0, 0.0, 100.0),
            (0,),
        ]
        result = engine._run_single(conn, {
            "check_type": "statistical_outlier",
            "check_name": "stat_sales_qty",
            "table_name": "fact_sales_monthly",
            "column": "qty",
            "method": "iqr",
            "threshold": 1.5,
        })
        assert result["status"] == "pass"
        assert result["check_name"] == "stat_sales_qty"

    def test_cross_column_dispatch(self):
        engine = self._make_engine()
        conn, _ = _mock_conn(fetchone_return=(100, 0))
        result = engine._run_single(conn, {
            "check_type": "cross_column",
            "check_name": "cc_test",
            "table_name": "t",
            "rule": "qty >= 0",
            "description": "test",
        })
        assert result["status"] == "pass"

    def test_temporal_gaps_dispatch(self):
        from datetime import date
        engine = self._make_engine()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = [
            (date(2026, 1, 1),), (date(2026, 2, 1),), (date(2026, 3, 1),),
        ]
        result = engine._run_single(conn, {
            "check_type": "temporal_gaps",
            "check_name": "tg_sales",
            "table_name": "fact_sales_monthly",
            "date_column": "startdate",
            "grain": "month",
        })
        assert result["status"] == "pass"
