"""Tests for tree tuning target resolution."""

import inspect

import pytest

from common.core.paths import PROJECT_ROOT
from scripts.ml import tune_cluster_hyperparams, tune_hyperparams
from scripts.ml.tune_hyperparams import _base_model_name, _resolve_tuning_target


def _pipeline_cfg() -> dict:
    return {
        "algorithms": {
            "lgbm_cluster": {
                "type": "tree",
                "params": {"learning_rate": 0.05},
            },
            "lgbm_cust_enriched": {
                "type": "tree",
                "params": {"customer_features": True},
            },
            "catboost_cust_enriched": {
                "type": "tree",
                "params": {"customer_features": True},
            },
            "xgboost_cluster": {
                "type": "tree",
                "params": {},
            },
            "lgbm_statistical_shadow": {
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
        ("lgbm_cust_enriched", "lgbm"),
        ("catboost_cust_enriched", "catboost"),
        ("xgboost_cluster", "xgboost"),
    ],
)
def test_base_model_name_resolves_pipeline_tree_ids(model_id: str, expected: str) -> None:
    assert _base_model_name(model_id) == expected


def test_base_model_name_rejects_unknown_tree_prefix() -> None:
    with pytest.raises(ValueError, match="Cannot resolve tree backend"):
        _base_model_name("random_forest_cluster")


def test_default_tuning_target_uses_cluster_model_id() -> None:
    model_name, model_id, entry = _resolve_tuning_target("xgboost", None, _pipeline_cfg())

    assert model_name == "xgboost"
    assert model_id == "xgboost_cluster"
    assert entry["type"] == "tree"


def test_enriched_tuning_target_preserves_customer_feature_flag() -> None:
    model_name, model_id, entry = _resolve_tuning_target(
        "lgbm",
        "lgbm_cust_enriched",
        _pipeline_cfg(),
    )

    assert model_name == "lgbm"
    assert model_id == "lgbm_cust_enriched"
    assert entry["params"]["customer_features"] is True


def test_tuning_target_rejects_model_id_backend_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _resolve_tuning_target("catboost", "lgbm_cust_enriched", _pipeline_cfg())


def test_tuning_target_rejects_non_tree_pipeline_id() -> None:
    with pytest.raises(ValueError, match="not a tree algorithm"):
        _resolve_tuning_target("lgbm", "lgbm_statistical_shadow", _pipeline_cfg())


def test_makefile_exposes_customer_enriched_tuning_targets() -> None:
    text = (PROJECT_ROOT / "Makefile").read_text()

    assert "tune-lgbm-cust:" in text
    assert "tune-catboost-cust:" in text
    assert "tune-xgboost-cust:" in text
    assert "tune-cust-enriched-all:" in text


def test_tuning_main_threads_customer_features_into_matrix() -> None:
    source = inspect.getsource(tune_hyperparams.main)

    assert "include_customer_features=include_customer_features" in source
    assert "customer_features=customer_features" in source


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
