"""Tests for common.ml.model_registry — canonical param mapping, fit abstraction, best_iteration."""

from unittest.mock import MagicMock, patch

import pytest

from common.ml.model_registry import (
    CANONICAL_TO_NATIVE,
    compute_early_stop_patience,
    fit_model,
    from_native_params,
    get_best_iteration,
    to_native_params,
)


# ---------------------------------------------------------------------------
# to_native_params
# ---------------------------------------------------------------------------


class TestToNativeParams:
    def test_lgbm_maps_estimators(self):
        result = to_native_params("lgbm", {"estimators": 1500})
        assert result == {"n_estimators": 1500}

    def test_catboost_maps_estimators_and_depth(self):
        result = to_native_params("catboost", {"estimators": 3000, "max_depth": 10})
        assert result == {"iterations": 3000, "depth": 10}

    def test_xgboost_maps_estimators(self):
        result = to_native_params("xgboost", {"estimators": 500})
        assert result == {"n_estimators": 500}

    def test_passthrough_unknown_keys(self):
        result = to_native_params("lgbm", {"num_leaves": 127, "path_smooth": 4})
        assert result == {"num_leaves": 127, "path_smooth": 4}

    def test_catboost_skips_none_mapping(self):
        """CatBoost has no L1 reg — l1_reg should be dropped."""
        result = to_native_params("catboost", {"l1_reg": 0.5, "l2_reg": 3.0})
        assert "l1_reg" not in result
        assert result == {"l2_leaf_reg": 3.0}

    def test_unknown_model_passes_through(self):
        result = to_native_params("prophet", {"estimators": 100})
        assert result == {"estimators": 100}

    def test_all_canonical_keys_mapped(self):
        """Every canonical key should map to a native key for lgbm and xgboost."""
        for model in ("lgbm", "xgboost"):
            mapping = CANONICAL_TO_NATIVE[model]
            for canonical, native in mapping.items():
                assert native is not None, f"{model}.{canonical} has None mapping"


# ---------------------------------------------------------------------------
# from_native_params
# ---------------------------------------------------------------------------


class TestFromNativeParams:
    def test_lgbm_roundtrip(self):
        canonical = {"estimators": 1500, "l2_reg": 1.0}
        native = to_native_params("lgbm", canonical)
        back = from_native_params("lgbm", native)
        assert back == canonical

    def test_catboost_roundtrip(self):
        canonical = {"estimators": 3000, "max_depth": 10, "l2_reg": 7.5}
        native = to_native_params("catboost", canonical)
        back = from_native_params("catboost", native)
        assert back == canonical

    def test_xgboost_roundtrip(self):
        canonical = {"estimators": 500, "min_leaf_samples": 5}
        native = to_native_params("xgboost", canonical)
        back = from_native_params("xgboost", native)
        assert back == canonical


# ---------------------------------------------------------------------------
# get_best_iteration
# ---------------------------------------------------------------------------


class TestGetBestIteration:
    def test_lgbm_trailing_underscore(self):
        model = MagicMock()
        model.best_iteration_ = 42
        assert get_best_iteration(model, "lgbm") == 42

    def test_catboost_trailing_underscore(self):
        model = MagicMock()
        model.best_iteration_ = 100
        assert get_best_iteration(model, "catboost") == 100

    def test_xgboost_no_underscore(self):
        model = MagicMock()
        model.best_iteration = 77
        assert get_best_iteration(model, "xgboost") == 77

    def test_missing_attribute_returns_none(self):
        model = MagicMock(spec=[])
        assert get_best_iteration(model, "lgbm") is None

    def test_falsy_value_returns_none(self):
        model = MagicMock()
        model.best_iteration_ = 0
        assert get_best_iteration(model, "lgbm") is None


# ---------------------------------------------------------------------------
# compute_early_stop_patience
# ---------------------------------------------------------------------------


class TestComputeEarlyStopPatience:
    def test_3pct_of_1500(self):
        assert compute_early_stop_patience(1500) == 45

    def test_3pct_of_3000(self):
        assert compute_early_stop_patience(3000) == 90

    def test_3pct_of_500(self):
        assert compute_early_stop_patience(500) == 15

    def test_floor_of_10(self):
        """For small iteration counts, patience should not go below 10."""
        assert compute_early_stop_patience(100) == 10
        assert compute_early_stop_patience(50) == 10

    def test_custom_pct(self):
        assert compute_early_stop_patience(1000, pct=0.05) == 50


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------


class TestFitModel:
    def test_lgbm_calls_fit_with_callbacks(self):
        model = MagicMock()
        lib_module = MagicMock()
        lib_module.early_stopping.return_value = "es_callback"
        lib_module.log_evaluation.return_value = "log_callback"

        fit_model(model, "lgbm", MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                  ["ml_cluster"], ["ml_cluster", "qty_lag_1"], lib_module, 1500)

        model.fit.assert_called_once()
        call_kwargs = model.fit.call_args
        assert "categorical_feature" in call_kwargs.kwargs
        assert "callbacks" in call_kwargs.kwargs

    def test_catboost_creates_pool(self):
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(model, "catboost", MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                  ["ml_cluster"], ["ml_cluster", "qty_lag_1"], lib_module, 3000)

        model.fit.assert_called_once()
        lib_module.Pool.assert_called_once()
        call_kwargs = model.fit.call_args
        assert "early_stopping_rounds" in call_kwargs.kwargs

    def test_xgboost_calls_fit(self):
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(model, "xgboost", MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                  [], ["qty_lag_1"], lib_module, 500)

        model.fit.assert_called_once()
        call_kwargs = model.fit.call_args
        assert "eval_set" in call_kwargs.kwargs

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_model(MagicMock(), "prophet", MagicMock(), MagicMock(), MagicMock(),
                      MagicMock(), [], [], MagicMock(), 100)

    def test_standardized_patience_lgbm(self):
        """LGBM early stopping should use 3% patience."""
        model = MagicMock()
        lib_module = MagicMock()
        lib_module.early_stopping.return_value = "es"
        lib_module.log_evaluation.return_value = "log"

        fit_model(model, "lgbm", MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                  [], ["f1"], lib_module, 1500)

        lib_module.early_stopping.assert_called_once_with(stopping_rounds=45, verbose=False)

    def test_standardized_patience_catboost(self):
        """CatBoost early stopping should use 3% patience."""
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(model, "catboost", MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                  [], ["f1"], lib_module, 3000)

        call_kwargs = model.fit.call_args.kwargs
        assert call_kwargs["early_stopping_rounds"] == 90
