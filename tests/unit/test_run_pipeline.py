"""Tests for scripts/etl/run_pipeline.py — unified pipeline orchestrator."""

import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Import the module under test
from scripts.etl.run_pipeline import (
    ALL_DOMAINS,
    _month_range_from_filename,
    build_incremental_delete,
    detect_changes,
    detect_inventory_changes,
    get_mvs_for_domains,
    normalize_domain,
    normalize_inventory,
    print_summary,
    run_full,
    run_refresh,
)


# ---------------------------------------------------------------------------
# _month_range_from_filename
# ---------------------------------------------------------------------------

class TestMonthRange:
    def test_standard_filename(self):
        result = _month_range_from_filename("Inventory_Snapshot_2026_03.csv")
        assert result == ("2026-03-01", "2026-04-01")

    def test_december_wraps_year(self):
        result = _month_range_from_filename("Inventory_Snapshot_2025_12.csv")
        assert result == ("2025-12-01", "2026-01-01")

    def test_january(self):
        result = _month_range_from_filename("Inventory_Snapshot_2026_01.csv")
        assert result == ("2026-01-01", "2026-02-01")

    def test_unparseable_returns_none(self):
        assert _month_range_from_filename("random_file.csv") is None


# ---------------------------------------------------------------------------
# build_incremental_delete
# ---------------------------------------------------------------------------

class TestBuildIncrementalDelete:
    def test_single_file(self):
        files = [Path("Inventory_Snapshot_2026_02.csv")]
        clause = build_incremental_delete(files)
        assert "snapshot_date >= '2026-02-01'" in clause
        assert "snapshot_date < '2026-03-01'" in clause

    def test_multiple_files(self):
        files = [
            Path("Inventory_Snapshot_2026_02.csv"),
            Path("Inventory_Snapshot_2026_03.csv"),
        ]
        clause = build_incremental_delete(files)
        assert "2026-02-01" in clause
        assert "2026-03-01" in clause
        assert " OR " in clause

    def test_empty_list(self):
        assert build_incremental_delete([]) == ""

    def test_unparseable_files_ignored(self):
        files = [Path("unknown.csv")]
        assert build_incremental_delete(files) == ""


# ---------------------------------------------------------------------------
# detect_changes
# ---------------------------------------------------------------------------

class TestDetectChanges:
    @patch("scripts.etl.run_pipeline.file_hash", return_value="abc123")
    @patch("scripts.etl.run_pipeline.get_spec")
    def test_no_previous_batch_marks_changed(self, mock_spec, mock_hash):
        spec = MagicMock()
        spec.clean_file = "item_clean.csv"
        mock_spec.return_value = spec

        cur = MagicMock()
        cur.fetchone.return_value = None  # no previous batch

        with patch("scripts.etl.run_pipeline.ROOT", Path("/fake")):
            with patch.object(Path, "exists", return_value=True):
                result = detect_changes(["item"], cur)

        assert result["item"] is True

    @patch("scripts.etl.run_pipeline.file_hash", return_value="abc123")
    @patch("scripts.etl.run_pipeline.get_spec")
    def test_matching_hash_marks_unchanged(self, mock_spec, mock_hash):
        spec = MagicMock()
        spec.clean_file = "item_clean.csv"
        mock_spec.return_value = spec

        cur = MagicMock()
        cur.fetchone.return_value = ("abc123",)  # same hash

        with patch("scripts.etl.run_pipeline.ROOT", Path("/fake")):
            with patch.object(Path, "exists", return_value=True):
                result = detect_changes(["item"], cur)

        assert result["item"] is False

    @patch("scripts.etl.run_pipeline.file_hash", return_value="new_hash")
    @patch("scripts.etl.run_pipeline.get_spec")
    def test_different_hash_marks_changed(self, mock_spec, mock_hash):
        spec = MagicMock()
        spec.clean_file = "item_clean.csv"
        mock_spec.return_value = spec

        cur = MagicMock()
        cur.fetchone.return_value = ("old_hash",)

        with patch("scripts.etl.run_pipeline.ROOT", Path("/fake")):
            with patch.object(Path, "exists", return_value=True):
                result = detect_changes(["item"], cur)

        assert result["item"] is True

    def test_missing_csv_marks_changed(self):
        spec = MagicMock()
        spec.clean_file = "nonexistent.csv"
        cur = MagicMock()

        with patch("scripts.etl.run_pipeline.get_spec", return_value=spec):
            with patch("scripts.etl.run_pipeline.ROOT", Path("/fake")):
                result = detect_changes(["item"], cur)

        assert result["item"] is True

    def test_inventory_skipped(self):
        cur = MagicMock()
        result = detect_changes(["inventory"], cur)
        assert "inventory" not in result


# ---------------------------------------------------------------------------
# detect_inventory_changes
# ---------------------------------------------------------------------------

class TestDetectInventoryChanges:
    @patch("scripts.etl.run_pipeline.file_hash", return_value="new_hash")
    def test_new_file_detected(self, mock_hash, tmp_path):
        f = tmp_path / "Inventory_Snapshot_2026_03.csv"
        f.write_text("header\nrow1\n")

        cur = MagicMock()
        cur.fetchone.return_value = None  # no previous batch

        result = detect_inventory_changes(tmp_path, cur)
        assert len(result) == 1
        assert result[0].name == "Inventory_Snapshot_2026_03.csv"

    @patch("scripts.etl.run_pipeline.file_hash", return_value="same_hash")
    def test_unchanged_file_skipped(self, mock_hash, tmp_path):
        f = tmp_path / "Inventory_Snapshot_2026_03.csv"
        f.write_text("header\nrow1\n")

        cur = MagicMock()
        cur.fetchone.return_value = ("same_hash",)  # matching hash

        result = detect_inventory_changes(tmp_path, cur)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# get_mvs_for_domains
# ---------------------------------------------------------------------------

class TestGetMvsForDomains:
    @patch("scripts.etl.run_pipeline._cfg")
    def test_deduplicates_mvs(self, mock_cfg):
        mock_cfg.return_value = {
            "mv_refresh": {
                "sales": ["agg_sales_monthly"],
                "forecast": ["agg_forecast_monthly"],
            },
            "always_refresh": ["mv_dq_dashboard", "agg_sales_monthly"],
        }
        result = get_mvs_for_domains(["sales", "forecast"])
        # agg_sales_monthly appears in both always_refresh and sales but should be deduplicated
        assert result.count("agg_sales_monthly") == 1
        assert "mv_dq_dashboard" in result
        assert "agg_forecast_monthly" in result

    @patch("scripts.etl.run_pipeline._cfg")
    def test_empty_domains(self, mock_cfg):
        mock_cfg.return_value = {
            "mv_refresh": {"item": []},
            "always_refresh": ["mv_dq_dashboard"],
        }
        result = get_mvs_for_domains(["item"])
        assert result == ["mv_dq_dashboard"]


# ---------------------------------------------------------------------------
# normalize_domain
# ---------------------------------------------------------------------------

class TestNormalizeDomain:
    @patch("scripts.etl.run_pipeline.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert normalize_domain("item", Path("/data/input")) is True

    @patch("scripts.etl.run_pipeline.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        assert normalize_domain("item", Path("/data/input")) is False


# ---------------------------------------------------------------------------
# run_full (dry-run)
# ---------------------------------------------------------------------------

class TestRunFull:
    @patch("scripts.etl.run_pipeline.load_domain")
    @patch("scripts.etl.run_pipeline.normalize_domain")
    @patch("scripts.etl.run_pipeline.normalize_inventory")
    @patch("scripts.etl.run_pipeline.refresh_views")
    @patch("scripts.etl.run_pipeline.get_mvs_for_domains", return_value=[])
    @patch("scripts.etl.run_pipeline._cfg", return_value={"parallel": {"max_workers": 2}})
    def test_dry_run_no_side_effects(self, mock_cfg, mock_mvs, mock_refresh,
                                     mock_norm_inv, mock_norm, mock_load):
        results = run_full(["item", "sales"], Path("/data/input"), dry_run=True)
        mock_norm.assert_not_called()
        mock_norm_inv.assert_not_called()
        mock_load.assert_not_called()
        assert len(results) == 2
        assert all(r.get("dry_run") for r in results)

    @patch("scripts.etl.run_pipeline.load_domain")
    @patch("scripts.etl.run_pipeline.normalize_domain")
    @patch("scripts.etl.run_pipeline.normalize_inventory")
    @patch("scripts.etl.run_pipeline.refresh_views")
    @patch("scripts.etl.run_pipeline.get_mvs_for_domains", return_value=[])
    @patch("scripts.etl.run_pipeline.get_db_params", return_value={})
    @patch("scripts.etl.run_pipeline._cfg", return_value={"parallel": {"max_workers": 2}})
    @patch("scripts.etl.run_pipeline.psycopg")
    def test_full_calls_cleanup_script(self, mock_psycopg, mock_cfg, mock_db,
                                       mock_mvs, mock_refresh, mock_norm_inv,
                                       mock_norm, mock_load, tmp_path):
        cleanup = tmp_path / "cleanup_input.py"
        cleanup.write_text("# cleanup script")
        mock_load.return_value = {"domain": "item", "rows_in": 10,
                                  "rows_loaded": 10}

        with patch("scripts.etl.run_pipeline.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0)
            run_full(["item"], tmp_path)
            # First subprocess call should be the cleanup script
            assert mock_sub.called


# ---------------------------------------------------------------------------
# run_refresh (dry-run)
# ---------------------------------------------------------------------------

class TestRunRefresh:
    @patch("scripts.etl.run_pipeline.psycopg")
    @patch("scripts.etl.run_pipeline.get_db_params", return_value={})
    @patch("scripts.etl.run_pipeline.detect_changes", return_value={"item": True})
    @patch("scripts.etl.run_pipeline.normalize_domain")
    @patch("scripts.etl.run_pipeline.load_domain")
    @patch("scripts.etl.run_pipeline.refresh_views")
    @patch("scripts.etl.run_pipeline.get_mvs_for_domains", return_value=[])
    def test_dry_run_no_side_effects(self, mock_mvs, mock_refresh, mock_load,
                                     mock_norm, mock_detect, mock_db, mock_psycopg):
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        results = run_refresh(["item"], Path("/data/input"), dry_run=True)
        mock_norm.assert_not_called()
        mock_load.assert_not_called()

    @patch("scripts.etl.run_pipeline.psycopg")
    @patch("scripts.etl.run_pipeline.get_db_params", return_value={})
    @patch("scripts.etl.run_pipeline.detect_changes", return_value={"item": False, "sales": False})
    def test_no_changes_returns_empty(self, mock_detect, mock_db, mock_psycopg):
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        results = run_refresh(["item", "sales"], Path("/data/input"))
        assert results == []


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------

class TestDomainFiltering:
    def test_all_domains_list(self):
        assert len(ALL_DOMAINS) == 10
        assert "item" in ALL_DOMAINS
        assert "inventory" in ALL_DOMAINS
        assert "sku" in ALL_DOMAINS
        assert "sourcing" in ALL_DOMAINS
        assert "purchase_order" in ALL_DOMAINS


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_empty_results(self, capsys):
        print_summary([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_dry_run_results(self, capsys):
        print_summary([{"domain": "item", "dry_run": True}])
        captured = capsys.readouterr()
        assert "item" in captured.out
        assert "dry-run" in captured.out

    def test_skipped_results(self, capsys):
        print_summary([{"domain": "sales", "skipped": True}])
        captured = capsys.readouterr()
        assert "sales" in captured.out
        assert "skipped" in captured.out

    def test_full_results(self, capsys):
        print_summary([{
            "domain": "item",
            "rows_in": 1000,
            "rows_loaded": 950,
            "elapsed": "2.3s",
        }])
        captured = capsys.readouterr()
        assert "item" in captured.out
        assert "1,000" in captured.out
        assert "950" in captured.out
