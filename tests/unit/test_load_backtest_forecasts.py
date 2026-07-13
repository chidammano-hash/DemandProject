"""Characterization tests for scripts/etl/load_backtest_forecasts.py.

US1 pinned this loader's behavior; US3 moved its index/constraint management
into common/core/etl_helpers.py (covered by tests/unit/test_etl_helpers.py).
What remains here are the loader's own column/table constants.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl import load_backtest_forecasts as bt


def _make_backtest_dir(tmp_path, model_dirs, *, with_predictions=True):
    """Create a fake data/backtest/ tree with <model>/backtest_predictions.csv files."""
    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    for name in model_dirs:
        d = backtest_dir / name
        d.mkdir()
        if with_predictions:
            (d / "backtest_predictions.csv").write_text(
                "forecast_ck,model_id\n1,placeholder\n", encoding="utf-8"
            )
    return backtest_dir


class TestResolveInputFilesAll:
    """The --all branch must derive its include set from the config roster
    (forecastable union competing), not a hardcoded 4-name allowlist. Regression
    guard for the 'data for algorithms not copied to champion' bug where
    configured model directories were silently skipped.
    """

    def test_resolve_all_loads_every_roster_dir_with_predictions(self, tmp_path, monkeypatch):
        roster = {"lgbm_cluster", "nhits", "nbeats", "mstl", "chronos2_enriched"}
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: sorted(roster))
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: sorted(roster))

        backtest_dir = _make_backtest_dir(tmp_path, roster)

        resolved = bt._resolve_input_files(None, None, True, backtest_dir, None)
        resolved_models = {p.parent.name for p in resolved}

        assert resolved_models == roster

    def test_resolve_all_excludes_auxiliary_dirs(self, tmp_path, monkeypatch):
        roster = {"lgbm_cluster", "mstl"}
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: sorted(roster))
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: sorted(roster))

        backtest_dir = _make_backtest_dir(
            tmp_path,
            {"lgbm_cluster", "logs", "tuning_archive", "lgbm_cluster_baseline_20260322"},
        )

        resolved = bt._resolve_input_files(None, None, True, backtest_dir, None)
        resolved_models = {p.parent.name for p in resolved}

        assert resolved_models == {"lgbm_cluster"}
        assert "logs" not in resolved_models
        assert "tuning_archive" not in resolved_models
        assert "lgbm_cluster_baseline_20260322" not in resolved_models

    def test_resolve_all_warns_on_skipped_dir(self, tmp_path, monkeypatch, caplog):
        roster = {"lgbm_cluster"}
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: sorted(roster))
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: sorted(roster))

        # 'mystery_model' is not in the roster and not an aux dir → must warn.
        backtest_dir = _make_backtest_dir(tmp_path, {"lgbm_cluster", "mystery_model"})

        with caplog.at_level(logging.WARNING, logger=bt.logger.name):
            resolved = bt._resolve_input_files(None, None, True, backtest_dir, None)

        resolved_models = {p.parent.name for p in resolved}
        assert resolved_models == {"lgbm_cluster"}
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("mystery_model" in msg for msg in warnings)

    def test_resolve_all_includes_chronos2_enriched(self, tmp_path, monkeypatch):
        # Regression for the inverted allowlist: Chronos 2 Enriched is forecastable but
        # NOT competing, so the union must still keep it loadable.
        forecastable = {"lgbm_cluster", "chronos2_enriched"}
        competing = {"lgbm_cluster"}  # Chronos 2 Enriched intentionally absent here
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: sorted(forecastable))
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: sorted(competing))

        backtest_dir = _make_backtest_dir(tmp_path, {"lgbm_cluster", "chronos2_enriched"})

        resolved = bt._resolve_input_files(None, None, True, backtest_dir, None)
        resolved_models = {p.parent.name for p in resolved}
        assert "chronos2_enriched" in resolved_models


class TestCanonicalModelValidation:
    def test_explicit_retired_model_is_rejected_before_path_lookup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: ["lgbm_cluster", "mstl"])
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: ["lgbm_cluster", "mstl"])

        with pytest.raises(ValueError, match="catboost_cluster"):
            bt._resolve_input_files(None, "catboost_cluster", False, tmp_path, None)

    def test_csv_scan_rejects_retired_model_after_first_hundred_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: ["lgbm_cluster"])
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: ["lgbm_cluster"])
        csv_path = tmp_path / "backtest_predictions.csv"
        csv_path.write_text(
            "model_id\n" + "lgbm_cluster\n" * 101 + "xgboost_cluster\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="xgboost_cluster"):
            bt._read_csv_model_ids(csv_path)

    def test_csv_filter_must_exist_in_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            bt,
            "get_forecastable_model_ids",
            lambda: ["lgbm_cluster", "mstl"],
        )
        monkeypatch.setattr(
            bt,
            "get_competing_model_ids",
            lambda: ["lgbm_cluster", "mstl"],
        )
        csv_path = tmp_path / "backtest_predictions.csv"
        csv_path.write_text("model_id\nlgbm_cluster\n", encoding="utf-8")

        with pytest.raises(ValueError, match="does not contain requested model_id 'mstl'"):
            bt._read_csv_model_ids(csv_path, model_id_filter="mstl")

    def test_csv_model_ids_must_match_canonical_spelling_exactly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "get_forecastable_model_ids", lambda: ["lgbm_cluster"])
        monkeypatch.setattr(bt, "get_competing_model_ids", lambda: ["lgbm_cluster"])
        csv_path = tmp_path / "backtest_predictions.csv"
        csv_path.write_text("model_id\n lgbm_cluster\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unsupported backtest model"):
            bt._read_csv_model_ids(csv_path)


class TestArchiveStreaming:
    """US9: the real archive owner streams in BATCH_SIZE chunks (not one giant
    INSERT). External forecasts skip the archive entirely (see load_dataset_postgres)."""

    def test_batch_size_is_positive_int(self):
        assert isinstance(bt.BATCH_SIZE, int)
        assert bt.BATCH_SIZE > 0

    def test_load_archive_loops_in_batches(self):
        import inspect
        src = inspect.getsource(bt._load_archive)
        assert "BATCH_SIZE" in src
        assert "range(" in src  # batched loop over staged rows


class TestConstants:
    def test_main_and_archive_tables(self):
        assert bt._TABLE == "fact_external_forecast_monthly"
        assert bt._ARCHIVE_TABLE == "backtest_lag_archive"

    def test_load_cols_lead_with_forecast_ck(self):
        assert bt.LOAD_COLS[0] == "forecast_ck"
        assert "model_id" in bt.LOAD_COLS

    def test_archive_cols_end_with_timeframe(self):
        assert bt.ARCHIVE_COLS[-1] == "timeframe"
        assert bt.ARCHIVE_COLS[0] == "forecast_ck"
