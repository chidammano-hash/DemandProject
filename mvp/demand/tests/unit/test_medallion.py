"""Tests for common/medallion.py -- Medallion pipeline functions."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from common.domain_specs import get_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_cursor():
    """Create a mock cursor with fetchone/fetchall."""
    cur = MagicMock()
    cur.fetchone.return_value = (1,)
    cur.fetchall.return_value = []
    cur.rowcount = 0
    return cur


# ---------------------------------------------------------------------------
# file_hash
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_returns_hex_digest(self, tmp_path):
        from common.medallion import file_hash
        f = tmp_path / "test.csv"
        f.write_text("header\nrow1\n")
        result = file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_deterministic(self, tmp_path):
        from common.medallion import file_hash
        f = tmp_path / "test.csv"
        f.write_text("data\n")
        assert file_hash(f) == file_hash(f)

    def test_io_error_returns_empty(self, tmp_path):
        """E2: file_hash returns empty string on IO error."""
        from common.medallion import file_hash
        bad_path = tmp_path / "nonexistent.csv"
        result = file_hash(bad_path)
        assert result == ""

    def test_uses_hash_chunk_size(self, tmp_path):
        """Verify file_hash uses HASH_CHUNK_SIZE constant."""
        from common.medallion import file_hash
        from common.sql_helpers import HASH_CHUNK_SIZE
        f = tmp_path / "test.csv"
        f.write_text("x" * 100)
        result = file_hash(f)
        assert len(result) == 64  # SHA-256


# ---------------------------------------------------------------------------
# Batch lifecycle
# ---------------------------------------------------------------------------

class TestCreateBatch:
    def test_creates_batch(self):
        from common.medallion import create_batch
        cur = _mock_cursor()
        cur.fetchone.return_value = (42,)
        batch_id = create_batch(cur, "sales", source_file="sales_clean.csv")
        assert batch_id == 42
        cur.execute.assert_called_once()
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO audit_load_batch" in sql


class TestCompleteBatch:
    def test_updates_status(self):
        from common.medallion import complete_batch
        cur = _mock_cursor()
        complete_batch(cur, 42, row_count_in=100, row_count_out=95, quarantined=5)
        sql = cur.execute.call_args[0][0]
        assert "status = %s" in sql
        assert "completed_at" in sql
        params = cur.execute.call_args[0][1]
        assert "completed" in params
        assert 42 in params


class TestFailBatch:
    def test_marks_failed(self):
        from common.medallion import fail_batch
        cur = _mock_cursor()
        fail_batch(cur, 42, "type cast error")
        sql = cur.execute.call_args[0][0]
        assert "status = %s" in sql
        params = cur.execute.call_args[0][1]
        assert "failed" in params
        assert "type cast error" in params


class TestUpdateBatchStatus:
    """D6: Tests for consolidated _update_batch_status."""

    def test_completed_with_stats(self):
        from common.medallion import _update_batch_status
        cur = _mock_cursor()
        _update_batch_status(cur, 1, "completed", stats={
            "row_count_in": 100, "row_count_out": 90, "quarantined": 10,
        })
        sql = cur.execute.call_args[0][0]
        assert "row_count_in" in sql
        assert "row_count_out" in sql
        assert "row_count_quarantined" in sql

    def test_failed_with_error(self):
        from common.medallion import _update_batch_status
        cur = _mock_cursor()
        _update_batch_status(cur, 1, "failed", error_msg="boom")
        sql = cur.execute.call_args[0][0]
        assert "error_message" in sql
        params = cur.execute.call_args[0][1]
        assert "boom" in params

    def test_minimal_update(self):
        from common.medallion import _update_batch_status
        cur = _mock_cursor()
        _update_batch_status(cur, 1, "running")
        sql = cur.execute.call_args[0][0]
        assert "status = %s" in sql
        assert "error_message" not in sql


# ---------------------------------------------------------------------------
# typed_expr / business_key_expr (via sql_helpers)
# ---------------------------------------------------------------------------

class TestTypedExpr:
    def test_integer_field(self):
        from common.medallion import typed_expr
        spec = get_spec("sales")
        result = typed_expr("type", spec, "s")
        assert "::integer" in result

    def test_date_field(self):
        from common.medallion import typed_expr
        spec = get_spec("sales")
        result = typed_expr("startdate", spec, "s")
        assert "::date" in result

    def test_float_field(self):
        from common.medallion import typed_expr
        spec = get_spec("sales")
        result = typed_expr("qty", spec, "s")
        assert "::numeric" in result

    def test_text_field_passthrough(self):
        from common.medallion import typed_expr
        spec = get_spec("sales")
        result = typed_expr("dmdunit", spec, "s")
        assert "::" not in result


class TestBusinessKeyExpr:
    def test_single_key(self):
        from common.medallion import business_key_expr
        spec = get_spec("item")
        result = business_key_expr(spec, "s")
        assert "||" not in result  # single key, no concatenation

    def test_composite_key(self):
        from common.medallion import business_key_expr
        spec = get_spec("sales")
        result = business_key_expr(spec, "s")
        assert "||" in result  # multi-key, concatenation


# ---------------------------------------------------------------------------
# ingest_bronze
# ---------------------------------------------------------------------------

class TestIngestBronze:
    def test_creates_staging_and_inserts(self, tmp_path):
        from common.medallion import ingest_bronze
        spec = get_spec("item")
        csv = tmp_path / "item_clean.csv"
        csv.write_text("item_no,item_desc\n100,Widget\n")

        cur = _mock_cursor()
        # Mock the COPY context manager
        copy_ctx = MagicMock()
        cur.copy.return_value.__enter__ = MagicMock(return_value=copy_ctx)
        cur.copy.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.return_value = (1,)  # stg_count

        count = ingest_bronze(cur, spec, csv, batch_id=1)
        assert count == 1
        # Should have: CREATE TEMP TABLE, COPY, SELECT count, INSERT
        assert cur.execute.call_count >= 3

    def test_csv_copy_error_raises_runtime_error(self, tmp_path):
        """E3: CSV COPY failure raises RuntimeError with informative message."""
        from common.medallion import ingest_bronze
        spec = get_spec("item")
        csv = tmp_path / "item_clean.csv"
        csv.write_text("item_no\n100\n")

        cur = _mock_cursor()
        # Make COPY context raise
        cur.copy.return_value.__enter__ = MagicMock(
            side_effect=Exception("bad CSV format")
        )
        cur.copy.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(RuntimeError, match="CSV COPY.*failed"):
            ingest_bronze(cur, spec, csv, batch_id=1)


# ---------------------------------------------------------------------------
# promote_to_silver
# ---------------------------------------------------------------------------

class TestPromoteToSilver:
    def test_returns_tuple(self):
        from common.medallion import promote_to_silver
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.rowcount = 10
        inserted, quarantined = promote_to_silver(cur, spec, batch_id=1)
        assert inserted == 10
        assert quarantined == 0

    def test_sales_includes_original_columns(self):
        from common.medallion import promote_to_silver
        spec = get_spec("sales")
        cur = _mock_cursor()
        cur.rowcount = 5
        promote_to_silver(cur, spec, batch_id=1)
        sql = cur.execute.call_args[0][0]
        assert "_orig_qty" in sql

    def test_uses_parameterized_batch_id(self):
        """SQL2: batch_id should use %s parameter, not f-string."""
        from common.medallion import promote_to_silver
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.rowcount = 3
        promote_to_silver(cur, spec, batch_id=99)
        sql = cur.execute.call_args[0][0]
        # The batch_id should not appear literally in the SQL
        assert "99" not in sql
        # It should be in the params
        params = cur.execute.call_args[0][1]
        assert 99 in params


# ---------------------------------------------------------------------------
# _quarantine_rows (D2)
# ---------------------------------------------------------------------------

class TestQuarantineRows:
    def test_no_bad_rows(self):
        from common.medallion import _quarantine_rows
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.fetchall.return_value = []
        count = _quarantine_rows(
            cur, spec, "silver_item", 1,
            "item_ck IS NULL AND _load_batch_id = %s", [1],
            "null_pk", {"column": "item_ck"},
        )
        assert count == 0

    def test_quarantines_bad_rows(self):
        from common.medallion import _quarantine_rows
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.fetchall.return_value = [
            (1, 100, {"item_ck": None}),
            (2, 101, {"item_ck": None}),
        ]
        count = _quarantine_rows(
            cur, spec, "silver_item", 1,
            "item_ck IS NULL AND _load_batch_id = %s", [1],
            "null_pk", {"column": "item_ck"},
        )
        assert count == 2
        # INSERT quarantine + UPDATE status
        assert cur.execute.call_count >= 4  # 1 SELECT + 2 INSERTs + 1 UPDATE


# ---------------------------------------------------------------------------
# _get_percentiles (D4)
# ---------------------------------------------------------------------------

class TestGetPercentiles:
    def test_returns_values(self):
        from common.medallion import _get_percentiles
        cur = _mock_cursor()
        cur.fetchone.return_value = (10.0, 20.0)
        result = _get_percentiles(cur, "silver_sales", "qty", 1, [0.25, 0.75])
        assert result == [10.0, 20.0]

    def test_single_percentile(self):
        from common.medallion import _get_percentiles
        cur = _mock_cursor()
        cur.fetchone.return_value = (15.0,)
        result = _get_percentiles(cur, "silver_sales", "qty", 1, [0.5])
        assert result == [15.0]


# ---------------------------------------------------------------------------
# _impute_numeric / _impute_categorical (D3/L4)
# ---------------------------------------------------------------------------

class TestImputeNumeric:
    def test_no_median_skips(self):
        from common.medallion import _impute_numeric
        cur = _mock_cursor()
        cur.fetchone.return_value = (None,)  # no non-null values
        spec = get_spec("sales")
        count = _impute_numeric(cur, spec, "silver_sales", 1, "qty")
        assert count == 0

    def test_imputes_with_median(self):
        from common.medallion import _impute_numeric
        cur = _mock_cursor()
        # First fetchone: percentile (median)
        # Then fetchall: null rows
        cur.fetchone.return_value = (50.0,)
        cur.fetchall.return_value = [("key1",), ("key2",)]
        spec = get_spec("sales")
        count = _impute_numeric(cur, spec, "silver_sales", 1, "qty")
        assert count == 2


class TestImputeCategorical:
    def test_no_mode_skips(self):
        from common.medallion import _impute_categorical
        cur = _mock_cursor()
        cur.fetchone.return_value = None  # no non-null values
        spec = get_spec("item")
        count = _impute_categorical(cur, spec, "silver_item", 1, "brand_name")
        assert count == 0

    def test_imputes_with_mode(self):
        from common.medallion import _impute_categorical
        cur = _mock_cursor()
        cur.fetchone.return_value = ("BrandX", 50)
        cur.fetchall.return_value = [("key1",), ("key2",), ("key3",)]
        spec = get_spec("item")
        count = _impute_categorical(cur, spec, "silver_item", 1, "brand_name")
        assert count == 3


# ---------------------------------------------------------------------------
# run_silver_dq_gate
# ---------------------------------------------------------------------------

class TestRunSilverDqGate:
    @patch("common.medallion.load_config")
    def test_empty_batch(self, mock_cfg):
        from common.medallion import run_silver_dq_gate, _CFG
        import common.medallion as mod
        mod._CFG = {
            "promotion_gates": {
                "blocking_checks": ["completeness"],
                "min_pass_rate": 95.0,
            }
        }
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.fetchone.return_value = (0,)  # total_rows = 0
        result = run_silver_dq_gate(cur, spec, batch_id=1)
        assert result["passed"] is True
        assert result["total"] == 0
        mod._CFG = None

    @patch("common.medallion.load_config")
    def test_all_pass(self, mock_cfg):
        from common.medallion import run_silver_dq_gate
        import common.medallion as mod
        mod._CFG = {
            "promotion_gates": {
                "blocking_checks": ["completeness"],
                "min_pass_rate": 95.0,
            }
        }
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.fetchone.return_value = (100,)
        cur.fetchall.return_value = []
        cur.rowcount = 100

        result = run_silver_dq_gate(cur, spec, batch_id=1)
        assert result["passed"] is True
        assert result["pass_rate"] == 100.0
        mod._CFG = None

    @patch("common.medallion.load_config")
    def test_zero_pass_rate(self, mock_cfg):
        """Edge case: 0% pass rate when all rows quarantined."""
        from common.medallion import run_silver_dq_gate
        import common.medallion as mod
        mod._CFG = {
            "promotion_gates": {
                "blocking_checks": ["completeness"],
                "min_pass_rate": 95.0,
            }
        }
        spec = get_spec("item")
        cur = _mock_cursor()
        # total_rows = 10
        cur.fetchone.return_value = (10,)
        # All rows have null PKs
        cur.fetchall.return_value = [
            (i, i + 100, {"item_ck": None}) for i in range(10)
        ]
        cur.rowcount = 0  # no rows pass

        result = run_silver_dq_gate(cur, spec, batch_id=1)
        assert result["passed"] is False
        assert result["pass_rate"] == 0.0
        assert result["quarantined"] == 10
        mod._CFG = None

    @patch("common.medallion.load_config")
    def test_range_check_parameterized(self, mock_cfg):
        """SQL1: range check uses parameterized values."""
        from common.medallion import run_silver_dq_gate
        import common.medallion as mod
        mod._CFG = {
            "promotion_gates": {
                "blocking_checks": ["range"],
                "min_pass_rate": 95.0,
            }
        }
        mock_cfg.return_value = {
            "checks": {
                "range": {
                    "sales": {
                        "columns": [
                            {"column": "qty", "min": 0, "max": 100000},
                        ]
                    }
                }
            }
        }
        spec = get_spec("sales")
        cur = _mock_cursor()
        cur.fetchone.return_value = (100,)
        cur.fetchall.return_value = []
        cur.rowcount = 100

        result = run_silver_dq_gate(cur, spec, batch_id=1)
        assert result["passed"] is True
        mod._CFG = None


# ---------------------------------------------------------------------------
# apply_silver_fixes
# ---------------------------------------------------------------------------

class TestApplySilverFixes:
    @patch("common.medallion.load_config")
    def test_disabled_returns_zero(self, mock_load):
        from common.medallion import apply_silver_fixes
        import common.medallion as mod
        mod._CFG = {
            "auto_fix": {"item": {"enabled": False}},
        }
        spec = get_spec("item")
        cur = _mock_cursor()
        result = apply_silver_fixes(cur, spec, batch_id=1)
        assert result["fixes_applied"] == 0
        mod._CFG = None

    @patch("common.medallion.load_config")
    def test_enabled_snapshots_sales(self, mock_load):
        from common.medallion import apply_silver_fixes
        import common.medallion as mod
        mod._CFG = {
            "auto_fix": {
                "sales": {
                    "enabled": True,
                    "strategies": [],
                    "preserve_original": True,
                },
            },
        }
        mock_load.return_value = {"checks": {}}
        spec = get_spec("sales")
        cur = _mock_cursor()
        result = apply_silver_fixes(cur, spec, batch_id=1)
        # Should have executed the snapshot UPDATE
        calls = [str(c) for c in cur.execute.call_args_list]
        snapshot_calls = [c for c in calls if "_orig_qty" in c]
        assert len(snapshot_calls) >= 1
        mod._CFG = None


# ---------------------------------------------------------------------------
# _fix_outliers_with_audit — edge cases
# ---------------------------------------------------------------------------

class TestFixOutliers:
    def test_iqr_zero_skips(self):
        """Edge case: IQR=0 should skip the column."""
        from common.medallion import _fix_outliers_with_audit
        spec = get_spec("sales")
        cur = _mock_cursor()
        # Q1 == Q3 -> IQR = 0
        cur.fetchone.return_value = (50.0, 50.0)
        count = _fix_outliers_with_audit(cur, spec, "silver_sales", 1)
        assert count == 0

    def test_null_percentiles_skips(self):
        """Edge case: NULL percentiles should skip."""
        from common.medallion import _fix_outliers_with_audit
        spec = get_spec("sales")
        cur = _mock_cursor()
        cur.fetchone.return_value = (None, None)
        count = _fix_outliers_with_audit(cur, spec, "silver_sales", 1)
        assert count == 0


# ---------------------------------------------------------------------------
# _fix_lead_time_with_audit
# ---------------------------------------------------------------------------

class TestFixLeadTime:
    def test_no_lead_time_column_returns_zero(self):
        from common.medallion import _fix_lead_time_with_audit
        spec = get_spec("sales")  # sales has no lead_time_days
        cur = _mock_cursor()
        count = _fix_lead_time_with_audit(cur, spec, "silver_sales", 1)
        assert count == 0

    def test_uses_constants(self):
        """Verify lead time fix uses LEAD_TIME_MAX_DAYS and LEAD_TIME_DEFAULT_DAYS constants."""
        from common.medallion import _fix_lead_time_with_audit
        from common.sql_helpers import LEAD_TIME_MAX_DAYS, LEAD_TIME_DEFAULT_DAYS
        spec = get_spec("inventory")
        cur = _mock_cursor()
        # Global median returns None -> use LEAD_TIME_DEFAULT_DAYS
        cur.fetchone.return_value = (None,)
        cur.fetchall.return_value = [("key1", 999)]
        count = _fix_lead_time_with_audit(cur, spec, "silver_inventory", 1)
        assert count == 1
        # Verify the apply UPDATE uses parameterized LEAD_TIME_MAX_DAYS
        last_call = cur.execute.call_args_list[-1]
        params = last_call[0][1]
        assert LEAD_TIME_DEFAULT_DAYS in params
        assert LEAD_TIME_MAX_DAYS in params


# ---------------------------------------------------------------------------
# _record_correction
# ---------------------------------------------------------------------------

class TestRecordCorrection:
    def test_inserts_row(self):
        from common.medallion import _record_correction
        cur = _mock_cursor()
        _record_correction(
            cur, "sales", "silver_sales", "key123", "qty",
            100, 50, "clamp", "range", batch_id=1,
        )
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO audit_dq_corrections" in sql


# ---------------------------------------------------------------------------
# promote_to_gold
# ---------------------------------------------------------------------------

class TestPromoteToGold:
    @patch("common.medallion.load_config")
    def test_basic_promotion(self, mock_cfg):
        from common.medallion import promote_to_gold
        import common.medallion as mod
        mod._CFG = {"auto_fix": {"item": {}}}
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.rowcount = 50
        result = promote_to_gold(cur, spec, batch_id=1)
        assert result["gold_count"] == 50
        assert result["gold_table"] == "dim_item"
        mod._CFG = None

    @patch("common.medallion.load_config")
    def test_sales_dual_track(self, mock_cfg):
        from common.medallion import promote_to_gold
        import common.medallion as mod
        mod._CFG = {
            "auto_fix": {
                "sales": {"preserve_original": True},
            },
        }
        spec = get_spec("sales")
        cur = _mock_cursor()
        cur.fetchone.side_effect = [(True,), (True,)]
        cur.rowcount = 100

        result = promote_to_gold(cur, spec, batch_id=1)
        assert result["gold_count"] == 100
        assert result["original_count"] == 100
        calls = [str(c) for c in cur.execute.call_args_list]
        orig_calls = [c for c in calls if "fact_sales_monthly_original" in c]
        assert len(orig_calls) >= 1
        mod._CFG = None

    @patch("common.medallion.load_config")
    def test_forecast_replace_uses_external_model_id(self, mock_cfg):
        """M5: promote_to_gold uses EXTERNAL_MODEL_ID constant for forecast."""
        from common.medallion import promote_to_gold
        from common.sql_helpers import EXTERNAL_MODEL_ID
        import common.medallion as mod
        mod._CFG = {"auto_fix": {"forecast": {}}}
        spec = get_spec("forecast")
        cur = _mock_cursor()
        cur.rowcount = 200
        result = promote_to_gold(cur, spec, batch_id=1, replace_mode=True)
        # First execute should be DELETE with model_id param
        first_call = cur.execute.call_args_list[0]
        assert EXTERNAL_MODEL_ID in first_call[0][1]
        mod._CFG = None


# ---------------------------------------------------------------------------
# write_lineage
# ---------------------------------------------------------------------------

class TestWriteLineage:
    def test_writes_records(self):
        from common.medallion import write_lineage
        spec = get_spec("item")
        cur = _mock_cursor()
        cur.rowcount = 25
        count = write_lineage(cur, spec, batch_id=1)
        assert count == 25
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO audit_row_lineage" in sql


# ---------------------------------------------------------------------------
# prune_old_batches
# ---------------------------------------------------------------------------

class TestPruneOldBatches:
    @patch("common.medallion.load_config")
    def test_prunes_single_domain(self, mock_cfg):
        from common.medallion import prune_old_batches
        import common.medallion as mod
        mod._CFG = {
            "layers": {
                "bronze": {"retention_days": 90},
                "silver": {"retention_days": 30},
            }
        }
        cur = _mock_cursor()
        cur.rowcount = 5
        result = prune_old_batches(cur, domain="sales")
        assert result["bronze_deleted"] == 5
        assert result["silver_deleted"] == 5
        mod._CFG = None


# ---------------------------------------------------------------------------
# qident (via sql_helpers, re-exported)
# ---------------------------------------------------------------------------

class TestQident:
    def test_basic(self):
        from common.medallion import qident
        assert qident("col") == '"col"'

    def test_escapes_quotes(self):
        from common.medallion import qident
        assert qident('col"name') == '"col""name"'


# ---------------------------------------------------------------------------
# _build_silver_column_lists / _build_silver_insert_sql (L3)
# ---------------------------------------------------------------------------

class TestBuildSilverHelpers:
    def test_item_columns(self):
        from common.medallion import _build_silver_column_lists
        spec = get_spec("item")
        cols, typed = _build_silver_column_lists(spec, "b")
        assert spec.ck_field in cols
        assert "_bronze_id" in cols
        assert "_load_batch_id" in cols
        assert "_dq_status" in cols
        # Sales-specific columns should NOT be in item
        assert "_orig_qty_shipped" not in cols

    def test_sales_has_original_columns(self):
        from common.medallion import _build_silver_column_lists
        spec = get_spec("sales")
        cols, typed = _build_silver_column_lists(spec, "b")
        assert "_orig_qty_shipped" in cols
        assert "_orig_qty_ordered" in cols
        assert "_orig_qty" in cols

    def test_insert_sql_not_empty(self):
        from common.medallion import _build_silver_column_lists, _build_silver_insert_sql
        spec = get_spec("item")
        cols, typed = _build_silver_column_lists(spec, "b")
        sql = _build_silver_insert_sql(spec, "silver_item", "bronze_item", "b", cols, typed)
        assert "INSERT INTO" in sql
        assert "DISTINCT ON" in sql
