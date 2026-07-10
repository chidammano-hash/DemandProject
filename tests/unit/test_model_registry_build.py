"""Tests for ``common.ml.model_registry.build_model`` (Gen-4 Stream J)."""
from __future__ import annotations

import pytest

from common.ml.model_registry import (
    UnknownAlgorithm,
    _FoundationStub,
    _base_model_name,
    build_model,
)


def test_base_model_name_tree_variants():
    """_base_model_name maps tree variants to their canonical backend."""
    assert _base_model_name("lgbm_cluster", "tree") == "lgbm"


def test_base_model_name_non_tree_passthrough():
    """Non-tree families return their model_id as-is."""
    assert _base_model_name("chronos2_enriched", "foundation") == "chronos2_enriched"
    assert _base_model_name("mstl", "statistical") == "mstl"


def test_build_model_unknown_algorithm_raises():
    """Unknown algorithm ids raise UnknownAlgorithm (subclass of ValueError)."""
    with pytest.raises(UnknownAlgorithm):
        build_model("nope_not_a_real_algo")
    # Also catchable as ValueError for backwards compat.
    with pytest.raises(ValueError):
        build_model("definitely_not_in_config")


def test_build_model_lgbm_cluster_returns_lgbm_regressor():
    """``lgbm_cluster`` produces a configured LGBMRegressor."""
    lightgbm = pytest.importorskip("lightgbm")

    model = build_model("lgbm_cluster")
    assert isinstance(model, lightgbm.LGBMRegressor)

    # A few params from forecast_pipeline_config.yaml should be on the estimator.
    p = model.get_params()
    # The yaml ships ``n_estimators: 2000`` / ``learning_rate: 0.015`` for lgbm_cluster.
    assert p.get("n_estimators") == 2000
    assert pytest.approx(p.get("learning_rate"), rel=1e-6) == 0.015


def test_build_model_override_params():
    """Caller-provided params override the config defaults."""
    pytest.importorskip("lightgbm")

    model = build_model("lgbm_cluster", params={"n_estimators": 7})
    assert model.get_params().get("n_estimators") == 7


def test_build_model_foundation_returns_stub():
    """Foundation models return a _FoundationStub with algorithm metadata."""
    model = build_model("chronos2_enriched")
    assert isinstance(model, _FoundationStub)
    assert model.algorithm_id == "chronos2_enriched"
    assert model.algo_type == "foundation"
