"""Tests for production tree model training orchestration."""

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.core.constants import MIN_CLUSTER_ROWS
from common.core.utils import load_forecast_pipeline_config


def _make_train_df(cluster_label: str, n_rows: int) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="MS")
    return pd.DataFrame({
        "sku_ck": [f"SKU_{i:03d}" for i in range(n_rows)],
        "item_id": [f"ITEM_{i:03d}" for i in range(n_rows)],
        "customer_group": ["CG1"] * n_rows,
        "loc": ["L1"] * n_rows,
        "startdate": dates,
        "qty": [100.0 + (i % 7) for i in range(n_rows)],
        "ml_cluster": [cluster_label] * n_rows,
        "month": [d.month for d in dates],
    })


def test_train_cluster_builds_tree_model_through_registry():
    """Production training must construct estimators via model_registry.build_tree_model."""
    from scripts.ml.train_production_models import _train_cluster

    n = MIN_CLUSTER_ROWS
    train = _make_train_df("normal", n)
    n_val = max(1, int(len(train["startdate"].unique()) * 0.20))

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([100.0] * n_val)

    with patch(
        "scripts.ml.train_production_models.build_tree_model",
        return_value=mock_model,
    ) as build:
        with patch("scripts.ml.train_production_models.fit_model"):
            with patch("scripts.ml.train_production_models.get_best_iteration", return_value=100):
                with patch(
                    "scripts.ml.train_production_models.compute_cluster_demand_stats",
                    return_value={
                        "mean_demand": 100.0,
                        "cv_demand": 0.1,
                        "zero_demand_pct": 0.0,
                        "seasonal_amplitude": 0.0,
                    },
                ):
                    with patch(
                        "scripts.ml.train_production_models.resolve_cluster_params",
                        return_value=({"n_estimators": 100}, "default"),
                    ):
                        label, model, meta = _train_cluster(
                            "normal",
                            1,
                            1,
                            train,
                            ["month"],
                            [],
                            {"n_estimators": 100},
                            model_name="lgbm",
                            model_class=MagicMock,
                            lib_module=MagicMock(),
                            iter_param="n_estimators",
                            needs_cat_dtype_cast=False,
                            constant_target_guard=True,
                            backtest_cfg={},
                        )

    assert label == "normal"
    assert model is mock_model
    assert meta["n_estimators_used"] == 100
    build.assert_called_once_with("lgbm", {"n_estimators": 100})


def test_train_cluster_routes_intermittent_to_seasonal_naive_artifact():
    """Production training must deploy the same sparse-cluster fallback used in backtests."""
    from common.ml.seasonal_naive import SeasonalNaiveModel
    from scripts.ml.train_production_models import _train_cluster

    train = _make_train_df("sparse", 60)
    train["qty"] = [0.0 if i % 5 else 10.0 for i in range(len(train))]

    with patch(
        "scripts.ml.train_production_models.build_tree_model",
        side_effect=AssertionError("intermittent clusters should not fit a tree"),
    ) as build:
        with patch("scripts.ml.train_production_models.fit_model") as fit:
            with patch(
                "scripts.ml.train_production_models.compute_cluster_demand_stats",
                return_value={
                    "mean_demand": 2.0,
                    "cv_demand": 2.0,
                    "zero_demand_pct": 0.8,
                    "seasonal_amplitude": 0.0,
                },
            ):
                with patch(
                    "scripts.ml.train_production_models.resolve_cluster_params",
                    return_value=({"n_estimators": 100}, "default"),
                ):
                    label, model, meta = _train_cluster(
                        "sparse",
                        1,
                        1,
                        train,
                        ["month"],
                        [],
                        {"n_estimators": 100},
                        model_name="lgbm",
                        model_class=MagicMock,
                        lib_module=MagicMock(),
                        iter_param="n_estimators",
                        needs_cat_dtype_cast=False,
                        constant_target_guard=True,
                        backtest_cfg={
                            "baseline_intermittent": True,
                            "baseline_intermittent_window": 12,
                            "intermittent_threshold": 0.7,
                            "lumpy_threshold": 0.3,
                        },
                    )

    assert label == "sparse"
    assert isinstance(model, SeasonalNaiveModel)
    assert meta["cluster_profile"] == "seasonal_naive_baseline"
    assert meta["demand_pattern"] == "intermittent"
    assert meta["n_estimators_used"] == 0
    build.assert_not_called()
    fit.assert_not_called()


def test_production_and_backtest_share_tree_default_params():
    """Backtest and production training must resolve identical tree params."""
    from scripts.ml.run_backtest import MODEL_REGISTRY
    from scripts.ml.train_production_models import _MODEL_LIBRARY

    cfg = load_forecast_pipeline_config()
    model_ids = {
        "lgbm": "lgbm_cluster",
        "catboost": "catboost_cluster",
        "xgboost": "xgboost_cluster",
    }

    for model_name, model_id in model_ids.items():
        algo_params = cfg["algorithms"][model_id]["params"]
        assert _MODEL_LIBRARY[model_name]["default_params_fn"](algo_params, seed=7) == (
            MODEL_REGISTRY[model_name]["default_params"](algo_params, seed=7)
        )


def test_apply_tuned_params_file_overlays_best_params_and_iterations(tmp_path):
    """Production artifacts must use the same tuned params available to backtests."""
    from scripts.ml.train_production_models import _apply_tuned_params_file

    params_file = tmp_path / "best_params_lgbm_cluster.json"
    params_file.write_text(json.dumps({
        "model": "lgbm_cluster",
        "best_params": {
            "learning_rate": 0.02,
            "num_leaves": 31,
        },
        "best_n_estimators": 375,
    }))

    params, source = _apply_tuned_params_file(
        {"learning_rate": 0.1, "n_estimators": 2000, "num_leaves": 63},
        params_file=params_file,
        iter_param="n_estimators",
        model_id="lgbm_cluster",
        model_name="lgbm",
    )

    assert params["learning_rate"] == 0.02
    assert params["num_leaves"] == 31
    assert params["n_estimators"] == 375
    assert source == f"tuning_file:{params_file}"


def test_apply_tuned_params_file_accepts_legacy_base_model_name(tmp_path):
    """Older tuning artifacts stored the base library name rather than pipeline id."""
    from scripts.ml.train_production_models import _apply_tuned_params_file

    params_file = tmp_path / "best_params_lgbm.json"
    params_file.write_text(json.dumps({
        "model": "lgbm",
        "best_params": {"learning_rate": 0.03},
        "best_n_estimators": 250,
    }))

    params, _source = _apply_tuned_params_file(
        {"learning_rate": 0.1, "n_estimators": 2000},
        params_file=params_file,
        iter_param="n_estimators",
        model_id="lgbm_cluster",
        model_name="lgbm",
    )

    assert params == {"learning_rate": 0.03, "n_estimators": 250}


def test_apply_tuned_params_file_rejects_wrong_model_artifact(tmp_path):
    from scripts.ml.train_production_models import _apply_tuned_params_file

    params_file = tmp_path / "best_params_catboost.json"
    params_file.write_text(json.dumps({
        "model": "catboost_cluster",
        "best_params": {"learning_rate": 0.03},
        "best_n_estimators": 250,
    }))

    with pytest.raises(ValueError, match="not 'lgbm_cluster'"):
        _apply_tuned_params_file(
            {"learning_rate": 0.1, "n_estimators": 2000},
            params_file=params_file,
            iter_param="n_estimators",
            model_id="lgbm_cluster",
            model_name="lgbm",
        )


def test_save_training_metadata_records_params_source(tmp_path):
    from scripts.ml.train_production_models import _save_training_metadata

    _save_training_metadata(
        out_dir=tmp_path,
        model_id="xgboost_cluster",
        planning_date="2026-07-01",
        params_source="tuning_file:data/tuning/best_params_xgboost.json",
        cluster_results={"0": {"val_wape": 12.3}},
        feature_cols_per_cluster={"0": ["month"]},
        total_rows=10,
        total_dfus=2,
        elapsed_seconds=1.23,
    )

    metadata = json.loads((tmp_path / "training_metadata.json").read_text())
    assert metadata["params_source"] == "tuning_file:data/tuning/best_params_xgboost.json"


def test_main_all_exits_nonzero_when_any_tree_model_fails():
    """The all-model training job must not report success with missing artifacts."""
    from scripts.ml.train_production_models import main

    roster = {
        "lgbm_cluster": {"type": "tree"},
        "catboost_cluster": {"type": "tree"},
        "xgboost_cluster": {"type": "tree"},
        "rolling_mean": {"type": "statistical"},
    }

    def fake_train(model_id: str) -> None:
        if model_id == "catboost_cluster":
            raise RuntimeError("catboost failed")

    with patch.object(sys, "argv", ["train_production_models.py", "--all"]):
        with patch("scripts.ml.train_production_models.load_project_env"):
            with patch("scripts.ml.train_production_models.get_algorithm_roster", return_value=roster):
                with patch(
                    "scripts.ml.train_production_models.train_production_model",
                    side_effect=fake_train,
                ) as train:
                    with pytest.raises(SystemExit) as exc:
                        main()

    assert exc.value.code == 1
    assert train.call_count == 3
    assert [call.args[0] for call in train.call_args_list] == [
        "catboost_cluster",
        "lgbm_cluster",
        "xgboost_cluster",
    ]


def test_main_all_succeeds_when_all_tree_models_train():
    """All three tree families should be attempted and a clean run should exit normally."""
    from scripts.ml.train_production_models import main

    roster = {
        "lgbm_cluster": {"type": "tree"},
        "catboost_cluster": {"type": "tree"},
        "xgboost_cluster": {"type": "tree"},
        "seasonal_naive": {"type": "statistical"},
    }

    with patch.object(sys, "argv", ["train_production_models.py", "--all"]):
        with patch("scripts.ml.train_production_models.load_project_env"):
            with patch("scripts.ml.train_production_models.get_algorithm_roster", return_value=roster):
                with patch("scripts.ml.train_production_models.train_production_model") as train:
                    main()

    assert [call.args[0] for call in train.call_args_list] == [
        "catboost_cluster",
        "lgbm_cluster",
        "xgboost_cluster",
    ]
