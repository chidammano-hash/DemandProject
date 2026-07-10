"""LightGBM-only registry contract tests."""

from unittest.mock import MagicMock

import pytest

from common.ml.model_registry import (
    CANONICAL_TO_NATIVE,
    SPARSE_EARLY_STOP_FLOOR,
    UnknownAlgorithm,
    build_tree_model,
    compute_early_stop_patience,
    from_native_params,
    get_best_iteration,
    to_native_params,
)


def test_registry_exposes_only_lgbm_mapping():
    assert set(CANONICAL_TO_NATIVE) == {"lgbm"}


def test_lgbm_params_round_trip():
    canonical = {"estimators": 1500, "l2_reg": 1.0, "max_depth": 8}
    assert from_native_params("lgbm", to_native_params("lgbm", canonical)) == canonical


def test_unknown_tree_backend_is_rejected():
    with pytest.raises(UnknownAlgorithm, match="Expected 'lgbm'"):
        build_tree_model("retired", {})


def test_best_iteration_uses_lgbm_attribute():
    model = MagicMock()
    model.best_iteration_ = 42
    assert get_best_iteration(model, "lgbm") == 42


def test_sparse_patience_uses_higher_floor():
    assert compute_early_stop_patience(1500) == 75
    assert compute_early_stop_patience(1500, sparse=True) == 150
    assert compute_early_stop_patience(100, sparse=True) == SPARSE_EARLY_STOP_FLOOR
