"""Tests for multi-seed evaluation with variance estimation in run_backtest.py.

Validates:
- default_params lambdas accept a seed parameter for all models
- Seeds are correctly passed through to model params
- Backward compatibility: --n-seeds 1 produces same behavior as before
- Multi-seed metadata includes mean/std statistics
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import MODULE_REGISTRY from the script
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.run_backtest import MODEL_REGISTRY


def _complete_tree_params() -> dict[str, Any]:
    """Union of required tree params for seed/default-param unit tests."""
    params = _complete_lgbm_params()
    params.update(_complete_catboost_params())
    params.update(_complete_xgboost_params())
    return params


def _complete_lgbm_params(**overrides) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_id": "lgbm_cluster",
        "cluster_strategy": "per_cluster",
        "recursive": False,
        "shap_select": False,
        "tune_inline": False,
        "params_file": None,
        "objective": "regression_l1",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "max_depth": 6,
        "min_gain_to_split": 0.01,
        "subsample": 0.8,
        "bagging_freq": 1,
        "colsample_bytree": 0.8,
        "feature_fraction_bynode": 0.8,
        "reg_lambda": 0.8,
        "reg_alpha": 0.1,
        "path_smooth": 0.5,
        "max_bin": 255,
    }
    params.update(overrides)
    return params


def _complete_catboost_params(**overrides) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_id": "catboost_cluster",
        "cluster_strategy": "per_cluster",
        "recursive": False,
        "shap_select": False,
        "tune_inline": False,
        "params_file": None,
        "loss_function": "RMSE",
        "iterations": 100,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "border_count": 32,
        "max_ctr_complexity": 1,
    }
    params.update(overrides)
    return params


def _complete_xgboost_params(**overrides) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_id": "xgboost_cluster",
        "cluster_strategy": "per_cluster",
        "recursive": False,
        "shap_select": False,
        "tune_inline": False,
        "params_file": None,
        "objective": "reg:absoluteerror",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    }
    params.update(overrides)
    return params


# ── default_params lambda tests ───────────────────────────────────────────────


class TestDefaultParamsLambda:
    """Verify that MODEL_REGISTRY default_params lambdas accept a seed parameter."""

    @pytest.fixture
    def algo_config(self) -> dict[str, Any]:
        """Minimal algo config dict matching what algorithm_config.yaml provides."""
        return _complete_tree_params()

    def test_lgbm_default_params_accepts_seed(self, algo_config: dict) -> None:
        """LGBM default_params lambda accepts seed parameter."""
        params = MODEL_REGISTRY["lgbm"]["default_params"](algo_config, seed=99)
        assert params["random_state"] == 99

    def test_catboost_default_params_accepts_seed(self, algo_config: dict) -> None:
        """CatBoost default_params lambda accepts seed parameter."""
        params = MODEL_REGISTRY["catboost"]["default_params"](algo_config, seed=99)
        assert params["random_seed"] == 99

    def test_xgboost_default_params_accepts_seed(self, algo_config: dict) -> None:
        """XGBoost default_params lambda accepts seed parameter."""
        params = MODEL_REGISTRY["xgboost"]["default_params"](algo_config, seed=99)
        assert params["random_state"] == 99

    def test_lgbm_default_seed_is_42(self, algo_config: dict) -> None:
        """Without seed arg, LGBM defaults to 42 (backward compatible)."""
        params = MODEL_REGISTRY["lgbm"]["default_params"](algo_config)
        assert params["random_state"] == 42

    def test_catboost_default_seed_is_42(self, algo_config: dict) -> None:
        """Without seed arg, CatBoost defaults to 42 (backward compatible)."""
        params = MODEL_REGISTRY["catboost"]["default_params"](algo_config)
        assert params["random_seed"] == 42

    def test_xgboost_default_seed_is_42(self, algo_config: dict) -> None:
        """Without seed arg, XGBoost defaults to 42 (backward compatible)."""
        params = MODEL_REGISTRY["xgboost"]["default_params"](algo_config)
        assert params["random_state"] == 42


# ── Seed values produce valid param dicts ──────────────────────────────────────


class TestSeedValues:
    """Verify that various seed values produce valid parameter dictionaries."""

    @pytest.fixture
    def algo_config(self) -> dict[str, Any]:
        return _complete_tree_params()

    @pytest.mark.parametrize("seed_value", [0, 1, 42, 100, 999])
    @pytest.mark.parametrize("model_name", ["lgbm", "catboost", "xgboost"])
    def test_seed_produces_valid_params(
        self, algo_config: dict, model_name: str, seed_value: int
    ) -> None:
        """Each seed value produces a valid dict with the correct seed key."""
        params = MODEL_REGISTRY[model_name]["default_params"](algo_config, seed=seed_value)
        assert isinstance(params, dict)
        # Verify param dict is non-empty and has expected keys
        assert len(params) > 5
        # Verify seed is set correctly
        seed_key = "random_seed" if model_name == "catboost" else "random_state"
        assert params[seed_key] == seed_value

    @pytest.mark.parametrize("model_name", ["lgbm", "catboost", "xgboost"])
    def test_different_seeds_produce_different_seed_values(
        self, algo_config: dict, model_name: str
    ) -> None:
        """Different seed arguments produce different random_state/random_seed values."""
        seed_key = "random_seed" if model_name == "catboost" else "random_state"
        params_0 = MODEL_REGISTRY[model_name]["default_params"](algo_config, seed=0)
        params_1 = MODEL_REGISTRY[model_name]["default_params"](algo_config, seed=1)
        params_42 = MODEL_REGISTRY[model_name]["default_params"](algo_config, seed=42)
        assert params_0[seed_key] != params_1[seed_key]
        assert params_1[seed_key] != params_42[seed_key]
        assert params_0[seed_key] == 0
        assert params_1[seed_key] == 1
        assert params_42[seed_key] == 42

    @pytest.mark.parametrize("model_name", ["lgbm", "catboost", "xgboost"])
    def test_non_seed_params_unchanged_across_seeds(
        self, algo_config: dict, model_name: str
    ) -> None:
        """All params except the seed key remain identical across different seeds."""
        seed_key = "random_seed" if model_name == "catboost" else "random_state"
        params_0 = MODEL_REGISTRY[model_name]["default_params"](algo_config, seed=0)
        params_42 = MODEL_REGISTRY[model_name]["default_params"](algo_config, seed=42)
        # Remove seed key and compare remaining params
        p0 = {k: v for k, v in params_0.items() if k != seed_key}
        p42 = {k: v for k, v in params_42.items() if k != seed_key}
        assert p0 == p42


# ── Multi-seed main() integration tests ───────────────────────────────────────


class TestMultiSeedIntegration:
    """Test the multi-seed loop logic in main()."""

    @pytest.fixture(autouse=True)
    def _disable_pipeline_config(self):
        """Disable forecast_pipeline_config.yaml so tests use their own config."""
        with patch(
            "scripts.ml.run_backtest.load_forecast_pipeline_config",
            side_effect=FileNotFoundError,
        ), patch(
            "scripts.ml.run_backtest.get_algorithm_roster",
            side_effect=FileNotFoundError,
        ):
            yield

    def _make_metadata(self, accuracy_pct: float) -> dict:
        """Build a minimal metadata dict matching backtest output format."""
        return {
            "model_id": "lgbm_cluster",
            "accuracy_at_execution_lag": {
                "n_rows": 1000,
                "wape": 100 - accuracy_pct,
                "bias": 0.01,
                "accuracy_pct": accuracy_pct,
            },
        }

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_single_seed_backward_compatible(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """With n_seeds=1, run_tree_backtest is called exactly once with default seed."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock
        mock_backtest.return_value = None

        algo_config = _complete_lgbm_params()
        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path), "n_seeds": 1},
            "algorithms": {"lgbm": algo_config},
        }
        config_file = tmp_path / "algo_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch("sys.argv", ["run_backtest.py", "--model", "lgbm", "--config", str(config_file)]):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        # Single seed means exactly one call to run_tree_backtest
        assert mock_backtest.call_count == 1
        call_kwargs = mock_backtest.call_args[1]
        # With n_seeds=1, seed 0 is used — random_state should be 0
        assert call_kwargs["model_params"]["random_state"] == 0

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_model_id_selects_matching_pipeline_config(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """--model-id must read that algorithm section, not the base tree section."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock
        mock_backtest.return_value = None

        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path), "n_seeds": 1},
            "algorithms": {
                "lgbm_cluster": {
                    "type": "tree",
                    "enabled": True,
                    "backtest": True,
                    "cluster_strategy": "per_cluster",
                    "params": _complete_lgbm_params(),
                },
                "lgbm_cust_enriched": {
                    "type": "tree",
                    "enabled": True,
                    "backtest": True,
                    "cluster_strategy": "per_cluster",
                    "params": _complete_lgbm_params(
                        model_id="lgbm_cust_enriched",
                        customer_features=True,
                        n_estimators=222,
                        learning_rate=0.012,
                    ),
                },
            },
        }
        config_file = tmp_path / "forecast_pipeline_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch(
            "sys.argv",
            [
                "run_backtest.py",
                "--model",
                "lgbm",
                "--model-id",
                "lgbm_cust_enriched",
                "--config",
                str(config_file),
            ],
        ):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        call_kwargs = mock_backtest.call_args[1]
        assert call_kwargs["model_id"] == "lgbm_cust_enriched"
        assert call_kwargs["model_params"]["n_estimators"] == 222
        assert call_kwargs["model_params"]["learning_rate"] == 0.012
        assert call_kwargs["algo_config"]["customer_features"] is True

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_multi_seed_runs_multiple_times(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """With --n-seeds 3, run_tree_backtest is called 3 times with seeds 0, 1, 2."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock
        mock_backtest.return_value = None

        algo_config = _complete_lgbm_params()
        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path)},
            "algorithms": {"lgbm": algo_config},
        }
        config_file = tmp_path / "algo_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Create fake metadata files for each seed run
        model_dir = tmp_path / "lgbm_cluster"
        model_dir.mkdir(parents=True, exist_ok=True)

        def write_meta_side_effect(**kwargs):
            # Write metadata with varying accuracy for each call
            call_num = mock_backtest.call_count
            acc = 70.0 + call_num * 0.5  # 70.5, 71.0, 71.5
            meta = self._make_metadata(acc)
            with open(model_dir / "backtest_metadata.json", "w") as mf:
                json.dump(meta, mf)

        mock_backtest.side_effect = write_meta_side_effect

        with patch("sys.argv", ["run_backtest.py", "--model", "lgbm", "--config", str(config_file), "--n-seeds", "3"]):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        assert mock_backtest.call_count == 3

        # Verify seeds were 0, 1, 2
        seeds_used = []
        for call in mock_backtest.call_args_list:
            seeds_used.append(call[1]["model_params"]["random_state"])
        assert seeds_used == [0, 1, 2]

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_multi_seed_writes_summary_metadata(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """Multi-seed run writes mean/std summary into metadata JSON."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock
        mock_backtest.return_value = None

        algo_config = _complete_lgbm_params()
        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path)},
            "algorithms": {"lgbm": algo_config},
        }
        config_file = tmp_path / "algo_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        model_dir = tmp_path / "lgbm_cluster"
        model_dir.mkdir(parents=True, exist_ok=True)

        accuracies = [70.0, 72.0, 71.0]

        def write_meta_side_effect(**kwargs):
            call_idx = mock_backtest.call_count - 1
            meta = self._make_metadata(accuracies[call_idx])
            with open(model_dir / "backtest_metadata.json", "w") as mf:
                json.dump(meta, mf)

        mock_backtest.side_effect = write_meta_side_effect

        with patch("sys.argv", ["run_backtest.py", "--model", "lgbm", "--config", str(config_file), "--n-seeds", "3"]):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        # Read final metadata and check multi-seed summary
        with open(model_dir / "backtest_metadata.json") as mf:
            final_meta = json.load(mf)

        assert "multi_seed_summary" in final_meta
        summary = final_meta["multi_seed_summary"]
        assert summary["n_seeds"] == 3
        assert summary["seed_accuracies"] == accuracies
        assert abs(summary["mean_accuracy_pct"] - np.mean(accuracies)) < 0.001
        assert abs(summary["std_accuracy_pct"] - np.std(accuracies)) < 0.001

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_multi_seed_extra_metadata_includes_seed_info(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """Each seed run passes seed and n_seeds in extra_metadata."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock

        algo_config = _complete_lgbm_params()
        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path)},
            "algorithms": {"lgbm": algo_config},
        }
        config_file = tmp_path / "algo_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        model_dir = tmp_path / "lgbm_cluster"
        model_dir.mkdir(parents=True, exist_ok=True)

        def write_meta_side_effect(**kwargs):
            meta = self._make_metadata(70.0)
            with open(model_dir / "backtest_metadata.json", "w") as mf:
                json.dump(meta, mf)

        mock_backtest.side_effect = write_meta_side_effect

        with patch("sys.argv", ["run_backtest.py", "--model", "lgbm", "--config", str(config_file), "--n-seeds", "2"]):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        # Check that each call included seed metadata
        for call_idx, call in enumerate(mock_backtest.call_args_list):
            extra = call[1]["extra_metadata"]
            assert extra["seed"] == call_idx
            assert extra["n_seeds"] == 2

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_n_seeds_from_config(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """n_seeds is read from config when not specified on CLI."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock
        mock_backtest.return_value = None

        algo_config = _complete_lgbm_params()
        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path), "n_seeds": 2},
            "algorithms": {"lgbm": algo_config},
        }
        config_file = tmp_path / "algo_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        model_dir = tmp_path / "lgbm_cluster"
        model_dir.mkdir(parents=True, exist_ok=True)

        def write_meta_side_effect(**kwargs):
            meta = self._make_metadata(70.0)
            with open(model_dir / "backtest_metadata.json", "w") as mf:
                json.dump(meta, mf)

        mock_backtest.side_effect = write_meta_side_effect

        # No --n-seeds flag => should pick up n_seeds: 2 from config
        with patch("sys.argv", ["run_backtest.py", "--model", "lgbm", "--config", str(config_file)]):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        assert mock_backtest.call_count == 2

    @patch("scripts.ml.run_backtest.run_tree_backtest")
    @patch("scripts.ml.run_backtest.profiled_section")
    @patch("scripts.ml.run_backtest._import_model_class")
    def test_catboost_seed_param_key(
        self, mock_import, mock_profiler, mock_backtest, tmp_path: Path
    ) -> None:
        """CatBoost uses random_seed (not random_state) as the seed parameter key."""
        mock_profiler.return_value.__enter__ = MagicMock()
        mock_profiler.return_value.__exit__ = MagicMock(return_value=False)
        mock_import.return_value = MagicMock
        mock_backtest.return_value = None

        algo_config = _complete_catboost_params()
        config_data = {
            "backtest": {"n_timeframes": 2, "output_dir": str(tmp_path)},
            "algorithms": {"catboost": algo_config},
        }
        config_file = tmp_path / "algo_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        model_dir = tmp_path / "catboost_cluster"
        model_dir.mkdir(parents=True, exist_ok=True)

        def write_meta_side_effect(**kwargs):
            meta = self._make_metadata(70.0)
            with open(model_dir / "backtest_metadata.json", "w") as mf:
                json.dump(meta, mf)

        mock_backtest.side_effect = write_meta_side_effect

        with patch("sys.argv", ["run_backtest.py", "--model", "catboost", "--config", str(config_file), "--n-seeds", "2"]):
            with patch("importlib.import_module") as mock_importlib:
                mock_importlib.return_value = MagicMock()
                with patch("scripts.ml.run_backtest.load_dotenv"):
                    from scripts.ml.run_backtest import main

                    main()

        # CatBoost should use random_seed key
        for call_idx, call in enumerate(mock_backtest.call_args_list):
            params = call[1]["model_params"]
            assert "random_seed" in params
            assert params["random_seed"] == call_idx
