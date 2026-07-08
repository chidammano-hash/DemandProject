"""Tests for production tree model training orchestration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

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
