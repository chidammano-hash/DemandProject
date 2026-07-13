"""Regression contracts for production LightGBM final-refit parity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import build_tree_artifact_spec


def _training_frame() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "sku_ck": [f"sku-{index}" for index in range(len(months))],
            "startdate": months,
            "qty": np.linspace(10.0, 21.0, len(months)),
            "ml_cluster": ["steady"] * len(months),
            "month": [month.month for month in months],
            "qty_lag_1": np.linspace(9.0, 20.0, len(months)),
            "qty_lag_2": np.linspace(8.0, 19.0, len(months)),
        }
    )


@pytest.mark.parametrize("value", [None, True, 0, 1, -0.1, 1.1, "0.2"])
def test_production_validation_fraction_is_required_and_bounded(value: object) -> None:
    from scripts.ml.train_production_models import _production_validation_fraction

    config = {"production_forecast": {"production_training": {}}}
    if value is not None:
        config["production_forecast"]["production_training"]["val_fraction"] = value

    with pytest.raises((KeyError, ValueError), match="val_fraction"):
        _production_validation_fraction(config)


def test_production_validation_fraction_reads_exact_yaml_path() -> None:
    from scripts.ml.train_production_models import _production_validation_fraction

    config = {
        "val_fraction": 0.9,
        "production_forecast": {
            "val_fraction": 0.8,
            "production_training": {"val_fraction": 0.25},
        },
    }

    assert _production_validation_fraction(config) == 0.25


def test_recursive_noise_is_deterministic_copy_on_write_and_lag_only() -> None:
    from scripts.ml.train_production_models import (
        _prepare_recursive_training_features,
        _recursive_noise_seed,
    )

    source = _training_frame()
    source_before = source.copy(deep=True)
    feature_cols = ["month", "qty_lag_1", "qty_lag_2"]
    config = {
        "recursive_noise_enabled": True,
        "recursive_noise_pct": 0.05,
    }

    first, first_meta = _prepare_recursive_training_features(
        source,
        feature_cols=feature_cols,
        cluster_label="steady",
        backtest_cfg=config,
        random_state=42,
    )
    second, second_meta = _prepare_recursive_training_features(
        source,
        feature_cols=feature_cols,
        cluster_label="steady",
        backtest_cfg=config,
        random_state=42,
    )

    pd.testing.assert_frame_equal(source, source_before)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_series_equal(first["month"], source["month"])
    assert not np.array_equal(first["qty_lag_1"], source["qty_lag_1"])
    assert not np.array_equal(first["qty_lag_2"], source["qty_lag_2"])
    assert first_meta == second_meta == {
        "enabled": True,
        "pct": 0.05,
        "random_state": 42,
        "lag_features": ["qty_lag_1", "qty_lag_2"],
        "feature_seeds": {
            feature: _recursive_noise_seed(
                random_state=42,
                cluster_label="steady",
                feature_name=feature,
            )
            for feature in ("qty_lag_1", "qty_lag_2")
        },
    }


def test_train_cluster_uses_same_noisy_history_for_early_stop_and_final_fit() -> None:
    from scripts.ml.train_production_models import _recursive_noise_seed, _train_cluster

    source = _training_frame()
    source_before = source.copy(deep=True)
    feature_cols = ["month", "qty_lag_1", "qty_lag_2"]
    eval_model = MagicMock()
    eval_model.predict.return_value = np.full(3, 15.0)
    final_model = MagicMock()

    with (
        patch(
            "scripts.ml.train_production_models.compute_min_cluster_rows",
            return_value=1,
        ),
        patch(
            "scripts.ml.train_production_models.build_tree_model",
            side_effect=[eval_model, final_model],
        ),
        patch("scripts.ml.train_production_models.fit_model") as early_fit,
        patch("scripts.ml.train_production_models.fit_final_model") as final_fit,
        patch(
            "scripts.ml.train_production_models.get_best_iteration",
            return_value=17,
        ),
        patch(
            "scripts.ml.train_production_models.compute_cluster_demand_stats",
            return_value={
                "mean_demand": 15.0,
                "cv_demand": 0.1,
                "zero_demand_pct": 0.0,
                "seasonal_amplitude": 0.0,
            },
        ),
        patch(
            "scripts.ml.train_production_models.resolve_cluster_params",
            return_value=(
                {"n_estimators": 100, "random_state": 42},
                "default",
            ),
        ),
    ):
        _label, model, meta = _train_cluster(
            "steady",
            1,
            1,
            source,
            feature_cols,
            [],
            {"n_estimators": 100, "random_state": 42},
            model_name="lgbm",
            model_class=MagicMock,
            lib_module=MagicMock(),
            iter_param="n_estimators",
            needs_cat_dtype_cast=False,
            constant_target_guard=True,
            backtest_cfg={
                "recursive_noise_enabled": True,
                "recursive_noise_pct": 0.05,
                "intermittent_threshold": 0.7,
                "lumpy_threshold": 0.3,
            },
            validation_fraction=0.25,
        )

    pd.testing.assert_frame_equal(source, source_before)
    assert model is final_model
    early_train = early_fit.call_args.args[2]
    persisted_train = final_fit.call_args.args[2]
    pd.testing.assert_frame_equal(
        early_train,
        persisted_train.loc[early_train.index],
    )
    assert not np.array_equal(persisted_train["qty_lag_1"], source["qty_lag_1"])
    assert meta["n_val_months"] == 3
    assert meta["validation_fraction"] == 0.25
    assert meta["recursive_noise"] == {
        "enabled": True,
        "pct": 0.05,
        "random_state": 42,
        "lag_features": ["qty_lag_1", "qty_lag_2"],
        "feature_seeds": {
            feature: _recursive_noise_seed(
                random_state=42,
                cluster_label="steady",
                feature_name=feature,
            )
            for feature in ("qty_lag_1", "qty_lag_2")
        },
    }


def test_production_shap_selection_uses_backtest_selector_config_and_threshold() -> None:
    from scripts.ml.train_production_models import _select_production_cluster_features

    models = {"steady": object(), "lumpy": object()}
    training = pd.concat(
        [
            _training_frame(),
            _training_frame().assign(ml_cluster="lumpy"),
        ],
        ignore_index=True,
    )
    features = ["month", "qty_lag_1", "qty_lag_2", "price"]
    selected = {
        "steady": ["qty_lag_1", "month"],
        "lumpy": ["qty_lag_1", "qty_lag_2", "month"],
    }
    report = pd.DataFrame({"cluster": ["steady", "lumpy"]})
    algo_params = {
        "shap_select": True,
        "shap_threshold": 0.9,
        "shap_top_n": None,
        "shap_sample_size": 500,
        "correlation_filter": True,
        "correlation_threshold": 0.98,
        "variance_filter": True,
        "variance_threshold": 0.01,
    }
    backtest_cfg = {
        "shap_retrain_threshold": 0.5,
        "shap_min_features": 2,
    }

    with patch(
        "scripts.ml.train_production_models.compute_timeframe_shap_per_cluster",
        return_value=(selected, report),
    ) as selector:
        effective, raw_selected, returned_report = _select_production_cluster_features(
            models=models,
            train_data=training,
            feature_cols=features,
            cat_cols=[],
            algo_params=algo_params,
            backtest_cfg=backtest_cfg,
            cutoff_date=pd.Timestamp("2024-12-01"),
        )

    assert effective == {
        "steady": ["qty_lag_1", "month"],
        # Dropping only 25% is below the 50% retrain threshold.
        "lumpy": features,
    }
    assert raw_selected == selected
    assert returned_report is report
    args = selector.call_args.args
    kwargs = selector.call_args.kwargs
    assert args[:4] == (models, training, features, [])
    assert args[4] == 0
    assert args[5] == pd.Timestamp("2024-12-01")
    assert kwargs == {
        "shap_extractor_fn": kwargs["shap_extractor_fn"],
        "sample_size": 500,
        "cumulative_threshold": 0.9,
        "top_n": None,
        "min_features": 2,
        "correlation_filter": True,
        "correlation_threshold": 0.98,
        "variance_filter": True,
        "variance_threshold": 0.01,
    }


def test_shap_retrain_persists_better_subset_and_reverts_worse_subset() -> None:
    from scripts.ml.train_production_models import _retrain_selected_cluster_models

    training = pd.concat(
        [
            _training_frame(),
            _training_frame().assign(ml_cluster="lumpy"),
        ],
        ignore_index=True,
    )
    full_features = ["month", "qty_lag_1", "qty_lag_2"]
    selected_features = {
        "steady": ["month", "qty_lag_1"],
        "lumpy": ["month", "qty_lag_1"],
    }
    original_models = {"steady": object(), "lumpy": object()}
    original_meta = {
        "steady": {"val_wape": 12.0},
        "lumpy": {"val_wape": 8.0},
    }
    better_model = object()
    worse_model = object()

    with patch(
        "scripts.ml.train_production_models._train_cluster",
        side_effect=[
            ("steady", better_model, {"val_wape": 7.0}),
            ("lumpy", worse_model, {"val_wape": 11.0}),
        ],
    ) as trainer:
        models, metadata, effective = _retrain_selected_cluster_models(
            train_data=training,
            clusters=["steady", "lumpy"],
            initial_models=original_models,
            initial_metadata=original_meta,
            selected_feature_cols=selected_features,
            full_feature_cols=full_features,
            cat_cols=[],
            params={"n_estimators": 100, "random_state": 42},
            model_name="lgbm",
            model_class=MagicMock,
            lib_module=MagicMock(),
            iter_param="n_estimators",
            needs_cat_dtype_cast=False,
            constant_target_guard=True,
            backtest_cfg={},
            validation_fraction=0.25,
        )

    assert models["steady"] is better_model
    assert effective["steady"] == ["month", "qty_lag_1"]
    assert metadata["steady"]["feature_selection"]["retrained"] is True
    assert metadata["steady"]["feature_selection"]["reverted"] is False
    assert models["lumpy"] is original_models["lumpy"]
    assert effective["lumpy"] == full_features
    assert metadata["lumpy"]["feature_selection"]["retrained"] is True
    assert metadata["lumpy"]["feature_selection"]["reverted"] is True
    assert [call.kwargs["feature_cols"] for call in trainer.call_args_list] == [
        ["month", "qty_lag_1"],
        ["month", "qty_lag_1"],
    ]


def test_tree_config_checksum_captures_noise_smoothing_and_shap_contract() -> None:
    from common.ml.tree_artifacts import build_tree_model_config_payload

    pipeline_config = {
        "algorithms": {
            "lgbm_cluster": {
                "params": {
                    "recursive": True,
                    "shap_select": True,
                    "shap_threshold": 0.9,
                    "shap_top_n": None,
                    "shap_sample_size": 500,
                    "correlation_filter": False,
                    "correlation_threshold": 0.98,
                    "variance_filter": False,
                    "variance_threshold": 0.01,
                    "params_file": None,
                }
            }
        },
        "clustering": {"enabled": True},
        "backtest": {
            "intermittent_threshold": 0.7,
            "lumpy_threshold": 0.3,
            "recursive_noise_enabled": True,
            "recursive_noise_pct": 0.05,
            "recursive_lag_smooth": 0.15,
            "shap_retrain_threshold": 0.5,
            "shap_min_features": 20,
        },
        "production_forecast": {
            "lookback_months": 60,
            "production_training": {"val_fraction": 0.25},
        },
    }

    with patch(
        "common.ml.tree_artifacts.load_config",
        return_value={"profiles": []},
    ):
        payload = build_tree_model_config_payload(pipeline_config)

    assert payload["recursive_training"] == {
        "enabled": True,
        "noise_enabled": True,
        "noise_pct": 0.05,
        "lag_smooth": 0.15,
    }
    assert payload["feature_selection"] == {
        "enabled": True,
        "cumulative_threshold": 0.9,
        "top_n": None,
        "sample_size": 500,
        "min_features": 20,
        "retrain_threshold": 0.5,
        "correlation_filter": False,
        "correlation_threshold": 0.98,
        "variance_filter": False,
        "variance_threshold": 0.01,
    }
    assert payload["production_training"] == {"val_fraction": 0.25}


def test_production_tree_config_includes_the_shared_backtest_contract() -> None:
    from common.ml.tree_artifacts import build_production_tree_model_config_payload

    pipeline_config = {
        "algorithms": {"lgbm_cluster": {"params": {"params_file": None}}},
        "clustering": {"enabled": True},
        "backtest": {},
        "production_forecast": {
            "lookback_months": 36,
            "production_training": {"val_fraction": 0.2},
        },
    }
    snapshot = MagicMock()
    snapshot.as_metadata.return_value = {"config_checksum": "same-at-train-and-inference"}
    with (
        patch("common.ml.tree_artifacts.load_config", return_value={"profiles": []}),
        patch(
            "common.ml.tree_artifacts.build_backtest_config_snapshot",
            return_value=snapshot,
        ),
    ):
        payload = build_production_tree_model_config_payload(pipeline_config)

    assert payload["backtest_config"] == {
        "lgbm_cluster": {"config_checksum": "same-at-train-and-inference"}
    }


def test_backtest_propagates_configured_shap_retrain_threshold() -> None:
    from scripts.ml.run_backtest import _merge_backtest_training_settings

    merged = _merge_backtest_training_settings(
        {"recursive": True},
        {
            "recursive_noise_enabled": True,
            "recursive_noise_pct": 0.05,
            "recursive_lag_smooth": 0.15,
            "shap_retrain_threshold": 0.5,
        },
        {},
    )

    assert merged["shap_retrain_threshold"] == 0.5
    assert _merge_backtest_training_settings(
        {"shap_retrain_threshold": 0.25},
        {"shap_retrain_threshold": 0.5},
        {},
    )["shap_retrain_threshold"] == 0.25


def test_cluster_artifact_records_exact_features_noise_and_smoothing() -> None:
    from scripts.ml.train_production_models import _build_cluster_artifact

    recursive_training = {
        "enabled": True,
        "noise_enabled": True,
        "noise_pct": 0.05,
        "lag_smooth": 0.15,
    }
    spec = build_tree_artifact_spec(
        model_id="lgbm_cluster",
        model_config={
            "algorithm": {"params": {"shap_select": True}},
            "recursive_training": recursive_training,
        },
        lineage=ProductionTreeArtifactLineage(
            source_sales_batch_id=12,
            data_checksum="a" * 64,
            history_end=pd.Timestamp("2026-06-01").date(),
            cluster_experiment_id=3,
            cluster_assignment_count=10,
            cluster_assignment_checksum="b" * 64,
        ),
        cluster_strategy="per_cluster",
        cluster_labels={"steady"},
    )
    model = MagicMock()
    model.feature_importances_ = np.array([0.8, 0.2])
    noise = {
        "enabled": True,
        "pct": 0.05,
        "random_state": 42,
        "lag_features": ["qty_lag_1"],
        "feature_seeds": {"qty_lag_1": 123},
    }
    selection = {
        "enabled": True,
        "shap_selected_features": ["qty_lag_1", "month"],
        "effective_features": ["qty_lag_1", "month"],
        "retrained": True,
        "reverted": False,
    }

    artifact = _build_cluster_artifact(
        cluster_label="steady",
        model=model,
        feature_cols=["qty_lag_1", "month"],
        model_id="lgbm_cluster",
        model_name="lgbm",
        meta={
            "n_estimators_used": 17,
            "train_rows": 100,
            "total_rows": 100,
            "val_wape": 7.0,
            "recursive_noise": noise,
            "feature_selection": selection,
        },
        tree_spec=spec,
    )

    assert artifact["feature_cols"] == ["qty_lag_1", "month"]
    assert artifact["recursive_noise"] == noise
    assert artifact["recursive_training"] == recursive_training
    assert artifact["feature_selection"] == selection
