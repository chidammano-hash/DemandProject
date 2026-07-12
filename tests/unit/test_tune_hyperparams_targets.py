"""Tests for tree tuning target resolution."""

import inspect
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.ml import tune_cluster_hyperparams, tune_hyperparams
from scripts.ml.tune_hyperparams import _base_model_name, _resolve_tuning_target


def _pipeline_cfg() -> dict:
    return {
        "algorithms": {
            "lgbm_cluster": {
                "type": "tree",
                "params": {"learning_rate": 0.05},
            },
            "mstl": {
                "type": "statistical",
                "params": {},
            },
        }
    }


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("lgbm", "lgbm"),
        ("lgbm_cluster", "lgbm"),
    ],
)
def test_base_model_name_resolves_pipeline_tree_ids(model_id: str, expected: str) -> None:
    assert _base_model_name(model_id) == expected


def test_base_model_name_rejects_unknown_tree_prefix() -> None:
    with pytest.raises(ValueError, match="Cannot resolve tree backend"):
        _base_model_name("random_forest_cluster")


def test_default_tuning_target_uses_cluster_model_id() -> None:
    model_name, model_id, entry = _resolve_tuning_target("lgbm", None, _pipeline_cfg())

    assert model_name == "lgbm"
    assert model_id == "lgbm_cluster"
    assert entry["type"] == "tree"


def test_tuning_target_rejects_non_tree_pipeline_id() -> None:
    with pytest.raises(ValueError, match="not a tree algorithm"):
        _resolve_tuning_target("lgbm", "mstl", _pipeline_cfg())


def test_global_tuning_uses_configured_round_fallback() -> None:
    source = inspect.getsource(tune_hyperparams.main)

    assert 'trial_best_rounds_or_max(best_trial, t_cfg["n_estimators_max"])' in source
    assert 'best_trial.user_attrs.get("best_n_estimators", 500)' not in source


def test_cluster_tuning_writes_native_iteration_param() -> None:
    source = inspect.getsource(tune_cluster_hyperparams.main)

    assert 'trial_best_rounds_or_max(best_trial, t_cfg["n_estimators_max"])' in source
    assert "iter_param = iteration_param_for_model(model_name)" in source
    assert "best_params[iter_param] = best_n_estimators" in source
    assert 'best_params["n_estimators"] = best_n_estimators' not in source


def test_cluster_tuning_carries_actuals_through_index_resetting_masks(monkeypatch) -> None:
    jan = pd.Timestamp("2025-01-01")
    feb = pd.Timestamp("2025-02-01")
    grid = pd.DataFrame(
        {
            "startdate": [jan, feb],
            "qty": [10.0, 20.0],
            **{f"qty_lag_{lag}": [1.0, 2.0] for lag in range(1, 13)},
        }
    )
    captured: dict[str, np.ndarray] = {}

    def resetting_mask(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        masked = frame.iloc[::-1].reset_index(drop=True).copy()
        masked.loc[masked["startdate"] > cutoff, "qty"] = np.nan
        return masked

    def train_fold(_x_train, _y_train, _x_val, y_val, *_args):
        captured["actuals"] = np.asarray(y_val)
        return np.asarray(y_val), 1

    monkeypatch.setattr(tune_cluster_hyperparams, "mask_future_sales", resetting_mask)
    monkeypatch.setattr(tune_cluster_hyperparams, "suggest_model_params", lambda *_: {})
    monkeypatch.setitem(tune_cluster_hyperparams.TRAIN_FOLD_FNS, "lgbm", train_fold)
    trial = MagicMock()
    trial.should_prune.return_value = False
    objective = tune_cluster_hyperparams.make_cluster_objective(
        "lgbm",
        grid,
        ["qty_lag_1"],
        [],
        [([jan], [feb])],
        {"tuning": {"early_stopping_rounds": 2, "n_estimators_max": 5}},
    )

    assert objective(trial) == 0.0
    np.testing.assert_array_equal(captured["actuals"], [20.0])


def test_stale_cluster_filter_empty_intersection_exits_before_data_load(monkeypatch) -> None:
    load_data = MagicMock(side_effect=AssertionError("must not load all clusters"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_cluster_hyperparams.py",
            "--model",
            "lgbm",
            "--stale-only",
            "--clusters",
            "current_b",
        ],
    )
    monkeypatch.setattr(tune_cluster_hyperparams, "get_db_params", lambda: {})
    monkeypatch.setattr(
        tune_cluster_hyperparams, "load_forecast_pipeline_config", lambda: {"tuning": {}}
    )
    monkeypatch.setattr(tune_cluster_hyperparams, "fetch_stale_clusters", lambda _db: ["current_a"])
    monkeypatch.setattr(tune_cluster_hyperparams, "load_backtest_data", load_data)

    tune_cluster_hyperparams.main()

    load_data.assert_not_called()
