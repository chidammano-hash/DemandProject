"""Tests for scripts/ml/auto_tune.py helpers."""

import json
import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from scripts.ml.auto_tune import (
    apply_overrides,
    export_best_params,
    format_duration,
    load_strategies,
    print_leaderboard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config():
    """Minimal forecast_pipeline_config.yaml structure."""
    return {
        "backtest": {"n_timeframes": 10, "output_dir": "data/backtest"},
        "algorithms": {
            "lgbm_cluster": {
                "type": "tree",
                "enabled": True,
                "cluster_strategy": "per_cluster",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.02,
                    "num_leaves": 63,
                    "min_child_samples": 20,
                    "max_depth": 10,
                    "subsample": 0.80,
                    "colsample_bytree": 0.80,
                    "reg_lambda": 1.0,
                    "recursive": True,
                },
            },
        },
    }


@pytest.fixture()
def strategies_file(tmp_path):
    """Write a small strategies YAML for testing."""
    content = {
        "strategies": [
            {
                "label": "strat_a",
                "description": "Test strategy A",
                "overrides": {"learning_rate": 0.05},
            },
            {
                "label": "strat_b",
                "description": "Test strategy B",
                "overrides": {"max_depth": 14, "num_leaves": 127},
            },
            {
                "label": "strat_c",
                "description": "Test strategy C",
                "overrides": {"subsample": 0.60, "colsample_bytree": 0.60},
            },
        ],
    }
    p = tmp_path / "strategies.yaml"
    with open(p, "w") as f:
        yaml.dump(content, f)
    return p


# ---------------------------------------------------------------------------
# Tests: load_strategies
# ---------------------------------------------------------------------------


class TestLoadStrategies:
    def test_loads_from_file(self, strategies_file):
        strategies = load_strategies(strategies_file)
        assert len(strategies) == 3
        assert strategies[0]["label"] == "strat_a"
        assert strategies[1]["overrides"]["max_depth"] == 14

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_strategies(tmp_path / "nonexistent.yaml")

    def test_raises_on_empty_strategies(self, tmp_path):
        p = tmp_path / "empty.yaml"
        with open(p, "w") as f:
            yaml.dump({"strategies": []}, f)
        with pytest.raises(ValueError, match="No strategies"):
            load_strategies(p)


# ---------------------------------------------------------------------------
# Tests: apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_overrides_learning_rate(self, base_config):
        result = apply_overrides(base_config, {"learning_rate": 0.05})
        assert result["algorithms"]["lgbm_cluster"]["params"]["learning_rate"] == 0.05
        # Original unchanged
        assert base_config["algorithms"]["lgbm_cluster"]["params"]["learning_rate"] == 0.02

    def test_overrides_multiple_params(self, base_config):
        result = apply_overrides(
            base_config,
            {"max_depth": 14, "num_leaves": 127, "reg_lambda": 5.0},
        )
        lgbm = result["algorithms"]["lgbm_cluster"]["params"]
        assert lgbm["max_depth"] == 14
        assert lgbm["num_leaves"] == 127
        assert lgbm["reg_lambda"] == 5.0
        # Unchanged params preserved
        assert lgbm["learning_rate"] == 0.02
        assert lgbm["n_estimators"] == 1500

    def test_does_not_mutate_original(self, base_config):
        original_lr = base_config["algorithms"]["lgbm_cluster"]["params"]["learning_rate"]
        apply_overrides(base_config, {"learning_rate": 0.99})
        assert base_config["algorithms"]["lgbm_cluster"]["params"]["learning_rate"] == original_lr

    def test_empty_overrides(self, base_config):
        result = apply_overrides(base_config, {})
        assert result["algorithms"]["lgbm_cluster"] == base_config["algorithms"]["lgbm_cluster"]


# ---------------------------------------------------------------------------
# Tests: format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45) == "0m 45s"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "2m 5s"

    def test_hours(self):
        assert format_duration(3661) == "1h 1m 1s"

    def test_zero(self):
        assert format_duration(0) == "0m 0s"


# ---------------------------------------------------------------------------
# Tests: export_best_params
# ---------------------------------------------------------------------------


class TestExportBestParams:
    def test_exports_json(self, base_config, tmp_path):
        run_result = {
            "run_id": 7,
            "label": "strat_a",
            "accuracy_pct": 72.5,
            "wape": 27.5,
        }
        overrides = {"learning_rate": 0.05, "max_depth": 14}

        # Monkey-patch ROOT so it writes to tmp_path
        import scripts.ml.auto_tune as mod
        original_root = mod.ROOT
        mod.ROOT = tmp_path
        try:
            path = export_best_params(run_result, base_config, overrides)
        finally:
            mod.ROOT = original_root

        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert data["source"] == "auto_tune"
        assert data["run_id"] == 7
        assert data["accuracy_pct"] == 72.5
        assert data["best_params"]["learning_rate"] == 0.05
        assert data["best_params"]["max_depth"] == 14
        # n_estimators goes to top-level key, not inside best_params
        assert data["best_n_estimators"] == 1500
        assert "n_estimators" not in data["best_params"]

    def test_overrides_merged_with_base(self, base_config, tmp_path):
        run_result = {"run_id": 1, "label": "test", "accuracy_pct": 70.0, "wape": 30.0}
        overrides = {"learning_rate": 0.08, "n_estimators": 800}

        import scripts.ml.auto_tune as mod
        original_root = mod.ROOT
        mod.ROOT = tmp_path
        try:
            path = export_best_params(run_result, base_config, overrides)
        finally:
            mod.ROOT = original_root

        with open(path) as f:
            data = json.load(f)

        # n_estimators overridden to 800
        assert data["best_n_estimators"] == 800
        # learning_rate overridden
        assert data["best_params"]["learning_rate"] == 0.08
        # Base params preserved
        assert data["best_params"]["num_leaves"] == 63
        assert data["best_params"]["subsample"] == 0.80


# ---------------------------------------------------------------------------
# Tests: print_leaderboard (smoke test — just ensure no crash)
# ---------------------------------------------------------------------------


class TestPrintLeaderboard:
    def test_prints_without_error(self, capsys):
        results = [
            {"run_id": 1, "label": "strat_a", "status": "completed",
             "accuracy_pct": 72.0, "wape": 28.0, "bias": -0.01, "duration": 120},
            {"run_id": 2, "label": "strat_b", "status": "completed",
             "accuracy_pct": 70.5, "wape": 29.5, "bias": -0.02, "duration": 130},
            {"run_id": 3, "label": "strat_c", "status": "failed",
             "accuracy_pct": None, "wape": None, "bias": None, "duration": 60},
        ]
        print_leaderboard(results, baseline_accuracy=69.0)
        captured = capsys.readouterr()
        assert "strat_a" in captured.out
        assert "strat_b" in captured.out
        assert "72.00" in captured.out

    def test_prints_empty(self, capsys):
        print_leaderboard([], baseline_accuracy=69.0)
        captured = capsys.readouterr()
        assert "No completed runs" in captured.out

    def test_prints_without_baseline(self, capsys):
        results = [
            {"run_id": 1, "label": "strat_a", "status": "completed",
             "accuracy_pct": 72.0, "wape": 28.0, "bias": -0.01, "duration": 120},
        ]
        print_leaderboard(results, baseline_accuracy=None)
        captured = capsys.readouterr()
        assert "strat_a" in captured.out


# ---------------------------------------------------------------------------
# Tests: production config strategies file
# ---------------------------------------------------------------------------


class TestProductionStrategies:
    """Validate the actual config/forecasting/tune_strategies.yaml file."""

    def test_strategies_file_exists(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        assert STRATEGIES_FILE.exists()

    def test_has_13_lgbm_strategies(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        strategies = load_strategies(STRATEGIES_FILE, model="lgbm")
        assert len(strategies) == 13

    def test_has_15_catboost_strategies(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        strategies = load_strategies(STRATEGIES_FILE, model="catboost")
        assert len(strategies) == 15

    def test_has_15_xgboost_strategies(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        strategies = load_strategies(STRATEGIES_FILE, model="xgboost")
        assert len(strategies) == 15

    def test_all_have_required_fields(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        for model in ("lgbm", "catboost", "xgboost"):
            strategies = load_strategies(STRATEGIES_FILE, model=model)
            for s in strategies:
                assert "label" in s, f"Strategy missing label: {s}"
                assert "description" in s, f"Strategy missing description: {s}"
                assert "overrides" in s, f"Strategy missing overrides: {s}"
                assert isinstance(s["overrides"], dict)
                assert len(s["overrides"]) > 0, f"Strategy has empty overrides: {s['label']}"

    def test_all_labels_unique_per_model(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        for model in ("lgbm", "catboost", "xgboost"):
            strategies = load_strategies(STRATEGIES_FILE, model=model)
            labels = [s["label"] for s in strategies]
            assert len(labels) == len(set(labels)), f"Duplicate labels for {model}: {labels}"

    def test_overrides_are_valid_lgbm_params(self):
        from scripts.ml.auto_tune import STRATEGIES_FILE
        valid_keys = {
            "n_estimators", "learning_rate", "num_leaves", "min_child_samples",
            "max_depth", "subsample", "colsample_bytree", "reg_lambda",
            "reg_alpha", "path_smooth", "feature_fraction_bynode", "min_gain_to_split",
            "cluster_strategy", "recursive", "shap_select", "shap_threshold",
        }
        strategies = load_strategies(STRATEGIES_FILE, model="lgbm")
        for s in strategies:
            for key in s["overrides"]:
                assert key in valid_keys, (
                    f"Strategy '{s['label']}' has invalid override key: {key}"
                )
