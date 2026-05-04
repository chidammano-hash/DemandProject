"""Tests for scripts/run_clustering_scenario.py — scenario runner logic."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ml.run_clustering_scenario import (
    generate_scenario_id,
    get_scenario_result,
    SCENARIO_BASE,
)


class TestGenerateScenarioId:
    def test_format(self):
        sid = generate_scenario_id()
        assert sid.startswith("sc_")
        parts = sid.split("_")
        assert len(parts) == 4  # sc, date, time, uuid

    def test_unique(self):
        ids = {generate_scenario_id() for _ in range(10)}
        assert len(ids) == 10


class TestGetScenarioResult:
    def test_returns_none_for_missing(self):
        result = get_scenario_result("nonexistent_scenario_id_xyz")
        assert result is None

    def test_returns_saved_result(self, tmp_path):
        scenario_id = "sc_20250101_120000_ab34"
        scenario_dir = tmp_path / scenario_id
        scenario_dir.mkdir()
        expected = {"scenario_id": scenario_id, "status": "completed", "result": {"optimal_k": 5}}
        with open(scenario_dir / "scenario_result.json", "w") as f:
            json.dump(expected, f)

        with patch("common.ml.clustering.scenario.SCENARIO_BASE", tmp_path):
            result = get_scenario_result(scenario_id)
            assert result is not None
            assert result["scenario_id"] == scenario_id
            assert result["status"] == "completed"


class TestRunScenarioErrorHandling:
    def test_returns_failed_on_exception(self):
        """run_scenario should catch exceptions and return failed status."""
        from scripts.ml.run_clustering_scenario import run_scenario

        with patch("scripts.ml.run_clustering_scenario._run_full_pipeline", side_effect=ValueError("test error")), \
             patch("common.ml.clustering.scenario.SCENARIO_BASE", Path(tempfile.mkdtemp())):
            result = run_scenario(
                feature_params={"time_window_months": 24, "min_months_history": 1},
                model_params={"k_range": [3, 5]},
                label_params={"volume_high": 0.75},
            )
            assert result["status"] == "failed"
            assert "test error" in result["error"]

    def test_merges_default_params(self):
        """run_scenario should merge user params with config-driven defaults."""
        from scripts.ml.run_clustering_scenario import run_scenario

        with patch("scripts.ml.run_clustering_scenario._run_full_pipeline", side_effect=ValueError("skip")), \
             patch("common.ml.clustering.scenario.SCENARIO_BASE", Path(tempfile.mkdtemp())), \
             patch("scripts.ml.run_clustering_scenario._load_config_defaults", return_value={
                 "time_window_months": 36, "min_months_history": 12,
                 "k_range": [9, 18], "min_cluster_size_pct": 2.0,
                 "use_pca": False, "pca_components": None,
                 "labeling": {
                     "volume_thresholds": {"high": 0.75, "low": 0.25},
                     "cv_thresholds": {"steady": 0.4, "volatile": 0.8},
                     "seasonality_threshold": 0.3, "zero_demand_threshold": 0.15,
                 },
             }):
            result = run_scenario(
                model_params={"k_range": [4, 8]},
            )
            params = result["params"]
            # User-specified k_range should be present
            assert params["model_params"]["k_range"] == [4, 8]
            # Defaults from promoted experiment or hardcoded fallbacks
            assert params["feature_params"]["time_window_months"] == 36
            # Label params filled from config
            assert "volume_high" in params["label_params"]
            assert "cv_steady" in params["label_params"]


class TestRelabelOnly:
    def test_relabel_requires_previous_scenario(self):
        """relabel_only should fail gracefully when previous scenario doesn't exist."""
        from scripts.ml.run_clustering_scenario import run_scenario

        with patch("scripts.ml.run_clustering_scenario.SCENARIO_BASE", Path(tempfile.mkdtemp())):
            result = run_scenario(
                label_params={"volume_high": 0.8},
                relabel_only=True,
                previous_scenario_id="nonexistent",
            )
            assert result["status"] == "failed"
