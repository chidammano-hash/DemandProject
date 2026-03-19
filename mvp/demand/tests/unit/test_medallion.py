"""Tests for common/medallion.py — Medallion pipeline functions."""

import json
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
        assert "status = 'completed'" in sql
        params = cur.execute.call_args[0][1]
        assert params == [100, 95, 5, 42]


class TestFailBatch:
    def test_marks_failed(self):
        from common.medallion import fail_batch
        cur = _mock_cursor()
        fail_batch(cur, 42, "type cast error")
        sql = cur.execute.call_args[0][0]
        assert "status = 'failed'" in sql


# ---------------------------------------------------------------------------
# typed_expr / business_key_expr
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
        # First call: count total rows
        # Second call: fetchall for null PK check (no bad rows)
        # Third call: UPDATE set passed
        cur.fetchone.return_value = (100,)
        cur.fetchall.return_value = []
        cur.rowcount = 100

        result = run_silver_dq_gate(cur, spec, batch_id=1)
        assert result["passed"] is True
        assert result["pass_rate"] == 100.0
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
        # Sequence: TRUNCATE, INSERT gold, check table exists, TRUNCATE original,
        #           INSERT original, UPDATE promoted_at
        cur.fetchone.side_effect = [(True,), (True,)]
        cur.rowcount = 100

        result = promote_to_gold(cur, spec, batch_id=1)
        assert result["gold_count"] == 100
        assert result["original_count"] == 100
        # Verify fact_sales_monthly_original INSERT happened
        calls = [str(c) for c in cur.execute.call_args_list]
        orig_calls = [c for c in calls if "fact_sales_monthly_original" in c]
        assert len(orig_calls) >= 1
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
# qident
# ---------------------------------------------------------------------------

class TestQident:
    def test_basic(self):
        from common.medallion import qident
        assert qident("col") == '"col"'

    def test_escapes_quotes(self):
        from common.medallion import qident
        assert qident('col"name') == '"col""name"'
