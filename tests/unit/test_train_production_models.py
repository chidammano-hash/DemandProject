"""Tests for production tree model training orchestration."""

import json
import sys
from datetime import date
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from common.core.constants import MIN_CLUSTER_ROWS
from common.core.utils import load_forecast_pipeline_config
from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import build_tree_artifact_spec


class PicklableModelStub:
    """Small model double for artifact serialization tests."""

    feature_importances_ = [0.7, 0.3]


def _tree_spec(label: str):
    return build_tree_artifact_spec(
        model_id="lgbm_cluster",
        model_config={"algorithm": "lgbm", "clustering": {"enabled": True}},
        lineage=ProductionTreeArtifactLineage(
            source_sales_batch_id=91,
            data_checksum="a" * 64,
            history_end=date(2026, 6, 1),
            cluster_experiment_id=17,
            cluster_assignment_count=1,
            cluster_assignment_checksum="b" * 64,
        ),
        cluster_strategy="per_cluster",
        cluster_labels={label},
    )


def _make_train_df(cluster_label: str, n_rows: int) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="MS")
    return pd.DataFrame(
        {
            "sku_ck": [f"SKU_{i:03d}" for i in range(n_rows)],
            "item_id": [f"ITEM_{i:03d}" for i in range(n_rows)],
            "customer_group": ["CG1"] * n_rows,
            "loc": ["L1"] * n_rows,
            "startdate": dates,
            "qty": [100.0 + (i % 7) for i in range(n_rows)],
            "ml_cluster": [cluster_label] * n_rows,
            "month": [d.month for d in dates],
        }
    )


def test_train_cluster_builds_tree_model_through_registry():
    """Production training must construct estimators via model_registry.build_tree_model."""
    from scripts.ml.train_production_models import _train_cluster

    n = MIN_CLUSTER_ROWS
    train = _make_train_df("normal", n)
    n_val = max(1, int(len(train["startdate"].unique()) * 0.20))

    eval_model = MagicMock()
    eval_model.predict.return_value = np.array([100.0] * n_val)
    final_model = MagicMock()

    with patch(
        "scripts.ml.train_production_models.build_tree_model",
        side_effect=[eval_model, final_model],
    ) as build:
        with patch("scripts.ml.train_production_models.fit_model"):
            with patch("scripts.ml.train_production_models.fit_final_model") as final_fit:
                with patch(
                    "scripts.ml.train_production_models.get_best_iteration", return_value=100
                ):
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
                            return_value=({"n_estimators": 200}, "default"),
                        ):
                            label, model, meta = _train_cluster(
                                "normal",
                                1,
                                1,
                                train,
                                ["month"],
                                [],
                                {"n_estimators": 200},
                                model_name="lgbm",
                                model_class=MagicMock,
                                lib_module=MagicMock(),
                                iter_param="n_estimators",
                                needs_cat_dtype_cast=False,
                                constant_target_guard=True,
                                backtest_cfg={},
                                validation_fraction=0.20,
                            )

    assert label == "normal"
    assert model is final_model
    assert meta["n_estimators_used"] == 100
    assert meta["train_rows"] == len(train)
    assert meta["early_stop_train_rows"] == len(train) - n_val
    assert build.call_args_list[0].args == ("lgbm", {"n_estimators": 200})
    assert build.call_args_list[1].args == ("lgbm", {"n_estimators": 100})
    final_fit.assert_called_once()


def test_train_cluster_saved_model_refits_all_history_after_early_stopping():
    """Validation chooses n_estimators; the persisted model must then fit all rows."""
    from scripts.ml.train_production_models import _train_cluster

    train = _make_train_df("normal", MIN_CLUSTER_ROWS)
    n_val = max(1, int(len(train["startdate"].unique()) * 0.20))

    eval_model = MagicMock()
    eval_model.predict.return_value = np.array([100.0] * n_val)
    final_model = MagicMock()

    with patch(
        "scripts.ml.train_production_models.build_tree_model",
        side_effect=[eval_model, final_model],
    ):
        with patch("scripts.ml.train_production_models.fit_model"):
            with patch("scripts.ml.train_production_models.fit_final_model") as final_fit:
                with patch(
                    "scripts.ml.train_production_models.get_best_iteration", return_value=17
                ):
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
                            return_value=({"n_estimators": 200}, "default"),
                        ):
                            _label, model, meta = _train_cluster(
                                "normal",
                                1,
                                1,
                                train,
                                ["month"],
                                [],
                                {"n_estimators": 200},
                                model_name="lgbm",
                                model_class=MagicMock,
                                lib_module=MagicMock(),
                                iter_param="n_estimators",
                                needs_cat_dtype_cast=False,
                                constant_target_guard=True,
                                backtest_cfg={},
                                validation_fraction=0.20,
                            )

    assert model is final_model
    assert meta["train_rows"] == len(train)
    assert meta["early_stop_train_rows"] == len(train) - n_val
    X_all = final_fit.call_args.args[2]
    y_all = final_fit.call_args.args[3]
    assert len(X_all) == len(train)
    assert len(y_all) == len(train)


def test_production_and_backtest_share_tree_default_params():
    """Backtest and production training must resolve identical tree params."""
    from scripts.ml.run_backtest import MODEL_REGISTRY
    from scripts.ml.train_production_models import _MODEL_LIBRARY

    cfg = load_forecast_pipeline_config()
    algo_params = cfg["algorithms"]["lgbm_cluster"]["params"]
    assert _MODEL_LIBRARY["lgbm"]["default_params_fn"](algo_params, seed=7) == (
        MODEL_REGISTRY["lgbm"]["default_params"](algo_params, seed=7)
    )


def test_tree_final_refit_uses_latest_closed_month_and_excludes_record_month():
    from scripts.ml.train_production_models import _select_closed_training_history

    sales = pd.DataFrame(
        {
            "startdate": pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01"]),
            "qty": [10.0, 20.0, 999.0],
        }
    )

    closed, history_end = _select_closed_training_history(
        sales,
        planning_month=pd.Timestamp("2026-07-01"),
    )

    assert history_end == pd.Timestamp("2026-06-01")
    assert closed["startdate"].max() == pd.Timestamp("2026-06-01")
    assert 999.0 not in set(closed["qty"])


def test_tree_final_refit_fails_when_latest_closed_month_is_missing():
    from scripts.ml.train_production_models import _select_closed_training_history

    sales = pd.DataFrame(
        {
            "startdate": pd.to_datetime(["2026-04-01", "2026-05-01"]),
            "qty": [10.0, 20.0],
        }
    )

    with pytest.raises(RuntimeError, match=r"latest closed month 2026-06"):
        _select_closed_training_history(
            sales,
            planning_month=pd.Timestamp("2026-07-01"),
        )


def test_disabled_clustering_collapses_tree_final_refit_to_one_global_model():
    from scripts.ml.train_production_models import _apply_production_cluster_strategy

    training = pd.DataFrame(
        {
            "sku_ck": ["A", "B"],
            "ml_cluster": ["stable", "lumpy"],
            "qty": [10.0, 20.0],
        }
    )

    global_training = _apply_production_cluster_strategy(
        training,
        clustering_enabled=False,
    )

    assert set(global_training["ml_cluster"]) == {"global"}
    assert set(training["ml_cluster"]) == {"stable", "lumpy"}


def test_apply_tuned_params_file_overlays_best_params_and_iterations(tmp_path):
    """Production artifacts must use the same tuned params available to backtests."""
    from scripts.ml.train_production_models import _apply_tuned_params_file

    params_file = tmp_path / "best_params_lgbm_cluster.json"
    params_file.write_text(
        json.dumps(
            {
                "model": "lgbm_cluster",
                "best_params": {
                    "learning_rate": 0.02,
                    "num_leaves": 31,
                },
                "best_n_estimators": 375,
            }
        )
    )

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
    params_file.write_text(
        json.dumps(
            {
                "model": "lgbm",
                "best_params": {"learning_rate": 0.03},
                "best_n_estimators": 250,
            }
        )
    )

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

    params_file = tmp_path / "best_params_other_model.json"
    params_file.write_text(
        json.dumps(
            {
                "model": "other_model",
                "best_params": {"learning_rate": 0.03},
                "best_n_estimators": 250,
            }
        )
    )

    with pytest.raises(ValueError, match="not 'lgbm_cluster'"):
        _apply_tuned_params_file(
            {"learning_rate": 0.1, "n_estimators": 2000},
            params_file=params_file,
            iter_param="n_estimators",
            model_id="lgbm_cluster",
            model_name="lgbm",
        )


def test_build_training_metadata_records_params_source():
    from scripts.ml.train_production_models import _build_training_metadata

    metadata = _build_training_metadata(
        model_id="lgbm_cluster",
        planning_date="2026-07-01",
        params_source="tuning_file:data/tuning/best_params_lgbm.json",
        cluster_results={"0": {"val_wape": 12.3}},
        feature_cols_per_cluster={"0": ["month"]},
        total_rows=10,
        total_dfus=2,
        elapsed_seconds=1.23,
    )

    assert metadata["params_source"] == "tuning_file:data/tuning/best_params_lgbm.json"


def test_build_cluster_artifact_uses_meta_estimators_after_final_refit():
    """Final refit models may not expose best_iteration; persist selected round from meta."""
    from scripts.ml.train_production_models import _build_cluster_artifact

    model = PicklableModelStub()

    artifact = _build_cluster_artifact(
        cluster_label="normal",
        model=model,
        feature_cols=["month", "qty_lag_1"],
        model_id="lgbm_cluster",
        model_name="lgbm",
        meta={
            "n_estimators_used": 17,
            "train_rows": 100,
            "total_rows": 100,
            "val_wape": 12.3,
        },
        tree_spec=_tree_spec("normal"),
    )

    assert artifact["n_estimators_used"] == 17


def test_main_all_exits_nonzero_when_any_tree_model_fails():
    """The all-model training job must not report success with missing artifacts."""
    from scripts.ml.train_production_models import main

    roster = {"lgbm_cluster": {"type": "tree"}}

    with patch.object(sys, "argv", ["train_production_models.py", "--all"]):
        with patch("scripts.ml.train_production_models.load_project_env"):
            with patch(
                "scripts.ml.train_production_models.get_algorithm_roster", return_value=roster
            ):
                with patch(
                    "scripts.ml.train_production_models._train_model_in_subprocess",
                    return_value=1,
                ) as train:
                    with pytest.raises(SystemExit) as exc:
                        main()

    assert exc.value.code == 1
    train.assert_called_once_with("lgbm_cluster")


def test_main_all_trains_every_model_that_requires_a_persisted_artifact():
    """LightGBM and both neural models require final-refit artifacts."""
    from scripts.ml.train_production_models import main

    roster = {
        "lgbm_cluster": {"type": "tree"},
        "mstl": {"type": "statistical"},
        "nhits": {"type": "deep_learning"},
        "nbeats": {"type": "deep_learning"},
        "chronos2_enriched": {"type": "foundation"},
    }

    with patch.object(sys, "argv", ["train_production_models.py", "--all"]):
        with patch("scripts.ml.train_production_models.load_project_env"):
            with patch(
                "scripts.ml.train_production_models.get_algorithm_roster", return_value=roster
            ):
                with patch(
                    "scripts.ml.train_production_models._train_model_in_subprocess",
                    return_value=0,
                ) as train:
                    main()

    assert train.call_args_list == [call("lgbm_cluster"), call("nbeats"), call("nhits")]
