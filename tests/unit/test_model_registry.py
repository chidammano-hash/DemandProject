"""Tests for common.ml.model_registry — canonical param mapping, fit abstraction, best_iteration, WAPE eval."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from common.ml.model_registry import (
    CANONICAL_TO_NATIVE,
    REQUIRED_TREE_PARAM_KEYS,
    SPARSE_EARLY_STOP_FLOOR,
    WapeMetric,
    _wape_lgbm,
    _wape_xgb,
    compute_early_stop_patience,
    fit_final_model,
    fit_model,
    fit_tree_classifier,
    from_native_params,
    get_best_iteration,
    get_tree_default_params,
    probe_tree_gpu_available,
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


class TestBuildTreeClassifier:
    def test_build_lgbm_classifier(self):
        lightgbm = pytest.importorskip("lightgbm")
        from common.ml.model_registry import build_tree_classifier

        model = build_tree_classifier("lgbm", {"n_estimators": 7})
        assert isinstance(model, lightgbm.LGBMClassifier)
        assert model.get_params()["n_estimators"] == 7

    def test_fit_lgbm_classifier_uses_registry_boundary(self):
        model = MagicMock()
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 1])

        fit_tree_classifier(model, "lgbm", X, y, categorical_feature=[0])

        model.fit.assert_called_once_with(X, y, categorical_feature=[0])

    def test_all_canonical_keys_mapped(self):
        """Every canonical key should map to a native key for lgbm and xgboost."""
        for model in ("lgbm", "xgboost"):
            mapping = CANONICAL_TO_NATIVE[model]
            for canonical, native in mapping.items():
                assert native is not None, f"{model}.{canonical} has None mapping"


class TestProbeTreeGpuAvailable:
    def test_probe_uses_registry_build_and_fit(self, monkeypatch):
        model = MagicMock()
        build_mock = MagicMock(return_value=model)
        fit_mock = MagicMock()
        monkeypatch.setattr("common.ml.model_registry.build_tree_model", build_mock)
        monkeypatch.setattr("common.ml.model_registry.fit_final_model", fit_mock)

        assert probe_tree_gpu_available("xgboost", {"device": "cuda"})

        build_mock.assert_called_once()
        model_name, params = build_mock.call_args.args
        assert model_name == "xgboost"
        assert params["device"] == "cuda"
        assert params["n_estimators"] == 1
        fit_mock.assert_called_once()


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
    def test_5pct_of_1500(self):
        assert compute_early_stop_patience(1500) == 75

    def test_5pct_of_3000(self):
        assert compute_early_stop_patience(3000) == 150

    def test_5pct_of_500(self):
        assert compute_early_stop_patience(500) == 25

    def test_floor_of_10(self):
        """For small iteration counts, patience should not go below 10."""
        assert compute_early_stop_patience(100) == 10
        assert compute_early_stop_patience(50) == 10

    def test_custom_pct(self):
        assert compute_early_stop_patience(1000, pct=0.05) == 50

    def test_sparse_uses_higher_pct(self):
        """Sparse clusters use 10% patience instead of 5%."""
        assert compute_early_stop_patience(1500, sparse=True) == 150  # 10% of 1500

    def test_sparse_floor_of_50(self):
        """Sparse clusters have a minimum patience of 50 rounds."""
        assert compute_early_stop_patience(100, sparse=True) == SPARSE_EARLY_STOP_FLOOR
        assert compute_early_stop_patience(200, sparse=True) == SPARSE_EARLY_STOP_FLOOR

    def test_sparse_with_custom_pct_uses_max(self):
        """When custom pct is higher than SPARSE_EARLY_STOP_PCT, custom wins."""
        # 15% of 1000 = 150, which is > 10% of 1000 = 100
        assert compute_early_stop_patience(1000, pct=0.15, sparse=True) == 150

    def test_sparse_pct_overrides_low_custom(self):
        """When custom pct is lower than SPARSE_EARLY_STOP_PCT, sparse pct wins."""
        # sparse=True forces max(0.03, 0.10) = 0.10; 10% of 1000 = 100
        assert compute_early_stop_patience(1000, pct=0.03, sparse=True) == 100

    def test_non_sparse_ignores_sparse_settings(self):
        """Non-sparse mode always uses the standard 5% default."""
        assert compute_early_stop_patience(1500) == 75
        assert compute_early_stop_patience(1500, sparse=False) == 75


# ---------------------------------------------------------------------------
# get_tree_default_params
# ---------------------------------------------------------------------------


class TestGetTreeDefaultParams:
    @pytest.mark.parametrize("model_name", ["lgbm", "catboost", "xgboost"])
    def test_missing_yaml_params_raise(self, model_name):
        """Tree params must come from YAML, not Python fallback defaults."""
        with pytest.raises(ValueError, match="missing required YAML keys"):
            get_tree_default_params(model_name, {})

    def test_live_config_covers_every_forecastable_tree_model(self):
        """Every enabled forecastable tree model has the required YAML params."""
        from common.core.utils import get_algorithm_roster

        tree_algos = {
            model_id: entry
            for model_id, entry in get_algorithm_roster(stage="forecast").items()
            if entry.get("type") == "tree"
        }
        assert tree_algos

        for model_id, entry in tree_algos.items():
            base = model_id.split("_", 1)[0]
            params = entry.get("params", {})
            missing = set(REQUIRED_TREE_PARAM_KEYS[base]) - set(params)
            assert not missing, f"{model_id} missing required tree params: {sorted(missing)}"
            resolved = get_tree_default_params(base, params, seed=13)
            assert resolved


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------


class TestFitModel:
    def test_lgbm_calls_fit_with_callbacks(self):
        model = MagicMock()
        lib_module = MagicMock()
        lib_module.early_stopping.return_value = "es_callback"
        lib_module.log_evaluation.return_value = "log_callback"

        fit_model(
            model,
            "lgbm",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            ["ml_cluster"],
            ["ml_cluster", "qty_lag_1"],
            lib_module,
            1500,
        )

        model.fit.assert_called_once()
        call_kwargs = model.fit.call_args
        assert "categorical_feature" in call_kwargs.kwargs
        assert "callbacks" in call_kwargs.kwargs

    def test_catboost_creates_pool(self):
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "catboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            ["ml_cluster"],
            ["ml_cluster", "qty_lag_1"],
            lib_module,
            3000,
        )

        model.fit.assert_called_once()
        lib_module.Pool.assert_called_once()
        call_kwargs = model.fit.call_args
        assert "early_stopping_rounds" in call_kwargs.kwargs

    def test_xgboost_calls_fit(self):
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "xgboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["qty_lag_1"],
            lib_module,
            500,
        )

        model.fit.assert_called_once()
        call_kwargs = model.fit.call_args
        assert "eval_set" in call_kwargs.kwargs

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_model(
                MagicMock(),
                "prophet",
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                [],
                [],
                MagicMock(),
                100,
            )

    def test_final_lgbm_fit_uses_full_data_without_eval_set(self):
        model = MagicMock()
        X = MagicMock()
        y = MagicMock()

        fit_final_model(model, "lgbm", X, y, ["brand"], ["brand", "qty_lag_1"])

        model.fit.assert_called_once_with(X, y, categorical_feature=["brand"])

    def test_final_catboost_fit_uses_cat_indices_without_eval_set(self):
        model = MagicMock()
        X = MagicMock()
        y = MagicMock()

        fit_final_model(model, "catboost", X, y, ["brand"], ["qty_lag_1", "brand"])

        model.fit.assert_called_once_with(X, y, cat_features=[1], verbose=False)

    def test_final_xgboost_fit_uses_full_data_without_eval_set(self):
        model = MagicMock()
        X = MagicMock()
        y = MagicMock()

        fit_final_model(model, "xgboost", X, y, [], ["qty_lag_1"])

        model.fit.assert_called_once_with(X, y, verbose=False)

    def test_final_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_final_model(MagicMock(), "prophet", MagicMock(), MagicMock(), [], [])

    def test_standardized_patience_lgbm(self):
        """LGBM early stopping should use 5% patience."""
        model = MagicMock()
        lib_module = MagicMock()
        lib_module.early_stopping.return_value = "es"
        lib_module.log_evaluation.return_value = "log"

        fit_model(
            model,
            "lgbm",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            1500,
        )

        lib_module.early_stopping.assert_called_once_with(stopping_rounds=75, verbose=False)

    def test_standardized_patience_catboost(self):
        """CatBoost early stopping should use 5% patience."""
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "catboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            3000,
        )

        call_kwargs = model.fit.call_args.kwargs
        assert call_kwargs["early_stopping_rounds"] == 150

    def test_xgboost_receives_early_stopping_rounds(self):
        """XGBoost fit must receive early_stopping_rounds parameter."""
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "xgboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            500,
        )

        sp_kwargs = model.set_params.call_args.kwargs
        assert "early_stopping_rounds" in sp_kwargs

    def test_xgboost_patience_matches_compute(self):
        """XGBoost early_stopping_rounds must equal compute_early_stop_patience output."""
        model = MagicMock()
        lib_module = MagicMock()
        max_iter = 500

        fit_model(
            model,
            "xgboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            max_iter,
        )

        expected_patience = compute_early_stop_patience(max_iter)
        sp_kwargs = model.set_params.call_args.kwargs
        assert sp_kwargs["early_stopping_rounds"] == expected_patience
        assert sp_kwargs["early_stopping_rounds"] == 25

    def test_all_models_receive_early_stopping(self):
        """All three models (LGBM, CatBoost, XGBoost) must receive early stopping config."""
        max_iter = 1000
        expected_patience = compute_early_stop_patience(max_iter)  # 30

        # LGBM
        lgbm_model = MagicMock()
        lgbm_lib = MagicMock()
        lgbm_lib.early_stopping.return_value = "es"
        lgbm_lib.log_evaluation.return_value = "log"
        fit_model(
            lgbm_model,
            "lgbm",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lgbm_lib,
            max_iter,
        )
        lgbm_lib.early_stopping.assert_called_once_with(
            stopping_rounds=expected_patience,
            verbose=False,
        )

        # CatBoost
        cat_model = MagicMock()
        cat_lib = MagicMock()
        fit_model(
            cat_model,
            "catboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            cat_lib,
            max_iter,
        )
        assert cat_model.fit.call_args.kwargs["early_stopping_rounds"] == expected_patience

        # XGBoost (early_stopping_rounds via set_params, not fit)
        xgb_model = MagicMock()
        xgb_lib = MagicMock()
        fit_model(
            xgb_model,
            "xgboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            xgb_lib,
            max_iter,
        )
        assert xgb_model.set_params.call_args.kwargs["early_stopping_rounds"] == expected_patience

    def test_xgboost_small_n_estimators_patience_floor(self):
        """With very small n_estimators, patience should hit the floor of 10."""
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "xgboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            100,
        )

        sp_kwargs = model.set_params.call_args.kwargs
        assert sp_kwargs["early_stopping_rounds"] == 10

    def test_compute_early_stop_patience_500_explicit_3pct(self):
        """compute_early_stop_patience(500, 0.03) must return 15 when explicit pct=0.03."""
        assert compute_early_stop_patience(500, 0.03) == 15

    def test_compute_early_stop_patience_500_default_5pct(self):
        """compute_early_stop_patience(500) with default 5% must return 25."""
        assert compute_early_stop_patience(500) == 25

    def test_lgbm_passes_wape_eval_metric(self):
        """LGBM fit must receive custom WAPE eval_metric function."""
        from common.ml.model_registry import _wape_lgbm

        model = MagicMock()
        lib_module = MagicMock()
        lib_module.early_stopping.return_value = "es"
        lib_module.log_evaluation.return_value = "log"

        fit_model(
            model,
            "lgbm",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            1500,
        )

        call_kwargs = model.fit.call_args.kwargs
        assert call_kwargs["eval_metric"] is _wape_lgbm

    def test_catboost_does_not_override_eval_metric(self):
        """CatBoost must NOT pass eval_metric or custom_metric in fit() kwargs."""
        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "catboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            3000,
        )

        fit_kwargs = model.fit.call_args.kwargs
        assert "custom_metric" not in fit_kwargs
        assert "eval_metric" not in fit_kwargs

    def test_xgboost_passes_wape_eval_metric(self):
        """XGBoost eval_metric set via set_params as custom WAPE function."""
        from common.ml.model_registry import _wape_xgb

        model = MagicMock()
        lib_module = MagicMock()

        fit_model(
            model,
            "xgboost",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            [],
            ["f1"],
            lib_module,
            500,
        )

        model.set_params.assert_called_once()
        sp_kwargs = model.set_params.call_args.kwargs
        assert sp_kwargs["eval_metric"] is _wape_xgb
        # custom_metric must NOT be in fit() kwargs
        assert "custom_metric" not in model.fit.call_args.kwargs


# ---------------------------------------------------------------------------
# WAPE eval callbacks
# ---------------------------------------------------------------------------


class TestWapeLgbm:
    def test_known_values(self):
        """WAPE = sum(|F-A|) / |sum(A)| with known inputs."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        # |110-100| + |190-200| + |310-300| = 10 + 10 + 10 = 30
        # |sum(A)| = |600| = 600
        # WAPE = 30 / 600 = 0.05
        name, value, is_higher_better = _wape_lgbm(y_true, y_pred)
        assert name == "wape"
        assert value == pytest.approx(0.05)
        assert is_higher_better is False

    def test_perfect_predictions(self):
        """WAPE = 0 when predictions exactly match actuals."""
        y_true = np.array([50.0, 100.0, 150.0])
        y_pred = np.array([50.0, 100.0, 150.0])
        name, value, is_higher_better = _wape_lgbm(y_true, y_pred)
        assert name == "wape"
        assert value == pytest.approx(0.0)
        assert is_higher_better is False

    def test_denominator_floor_prevents_division_by_zero(self):
        """When sum(actuals) = 0, denominator is floored to 1.0."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([5.0, 3.0, 2.0])
        # |sum(A)| = 0 → denom = max(0, 1.0) = 1.0
        # sum(|F-A|) = 5 + 3 + 2 = 10
        # WAPE = 10 / 1.0 = 10.0
        name, value, _ = _wape_lgbm(y_true, y_pred)
        assert name == "wape"
        assert value == pytest.approx(10.0)

    def test_negative_actuals_absolute_denominator(self):
        """WAPE denominator uses |sum(A)|, so negatives don't cancel out."""
        y_true = np.array([-100.0, -200.0])
        y_pred = np.array([-90.0, -210.0])
        # |F-A|: |-90 - (-100)| + |-210 - (-200)| = 10 + 10 = 20
        # |sum(A)| = |(-300)| = 300
        # WAPE = 20 / 300 = 0.0667
        _, value, _ = _wape_lgbm(y_true, y_pred)
        assert value == pytest.approx(20.0 / 300.0)


class TestWapeXgb:
    def test_known_values(self):
        """WAPE = sum(|F-A|) / |sum(A)| with known inputs."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        value = _wape_xgb(y_true, y_pred)
        assert value == pytest.approx(0.05)

    def test_perfect_predictions(self):
        """WAPE = 0 when predictions exactly match actuals."""
        y_true = np.array([50.0, 100.0, 150.0])
        y_pred = np.array([50.0, 100.0, 150.0])
        value = _wape_xgb(y_true, y_pred)
        assert value == pytest.approx(0.0)

    def test_denominator_floor_prevents_division_by_zero(self):
        """When sum(actuals) = 0, denominator is floored to 1.0."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([5.0, 3.0, 2.0])
        value = _wape_xgb(y_true, y_pred)
        assert value == pytest.approx(10.0)

    def test_returns_float(self):
        """XGBoost 3.x eval returns a float (func.__name__ used as metric name)."""
        result = _wape_xgb(np.array([1.0]), np.array([1.0]))
        assert isinstance(result, float)


class TestWapeMetric:
    """Tests for CatBoost WapeMetric custom metric class."""

    def test_evaluate_known_values(self):
        """WapeMetric.evaluate() returns correct WAPE for known inputs."""
        metric = WapeMetric()
        approxes = [[110.0, 190.0, 310.0]]
        target = [100.0, 200.0, 300.0]
        # WAPE = 30 / 600 = 0.05
        value, weight = metric.evaluate(approxes, target, None)
        assert value == pytest.approx(0.05)
        assert weight == 1.0

    def test_evaluate_perfect_predictions(self):
        """WAPE = 0 when predictions exactly match actuals."""
        metric = WapeMetric()
        approxes = [[50.0, 100.0, 150.0]]
        target = [50.0, 100.0, 150.0]
        value, weight = metric.evaluate(approxes, target, None)
        assert value == pytest.approx(0.0)
        assert weight == 1.0

    def test_evaluate_denominator_floor(self):
        """When sum(actuals) = 0, denominator is floored to 1.0."""
        metric = WapeMetric()
        approxes = [[5.0, 3.0, 2.0]]
        target = [0.0, 0.0, 0.0]
        value, _ = metric.evaluate(approxes, target, None)
        assert value == pytest.approx(10.0)

    def test_is_max_optimal_false(self):
        """WAPE is minimized, not maximized."""
        metric = WapeMetric()
        assert metric.is_max_optimal() is False

    def test_get_final_error_passthrough(self):
        """get_final_error returns error unchanged."""
        metric = WapeMetric()
        assert metric.get_final_error(0.42, 1.0) == 0.42

    def test_evaluate_matches_formula(self):
        """WAPE computation matches sum(|F-A|) / |sum(A)| exactly."""
        metric = WapeMetric()
        y_pred_list = [120.0, 80.0, 250.0, 300.0]
        y_true_list = [100.0, 100.0, 200.0, 350.0]
        approxes = [y_pred_list]
        y_pred = np.array(y_pred_list)
        y_true = np.array(y_true_list)
        # Manual: |20| + |20| + |50| + |50| = 140
        # |sum(A)| = |750| = 750
        # WAPE = 140 / 750
        expected = np.sum(np.abs(y_pred - y_true)) / abs(y_true.sum())
        value, _ = metric.evaluate(approxes, y_true_list, None)
        assert value == pytest.approx(expected)
        assert value == pytest.approx(140.0 / 750.0)


class TestWapeConsistency:
    """Cross-library consistency: all three WAPE implementations produce the same value."""

    def test_all_three_produce_same_wape(self):
        """LGBM, XGBoost, and CatBoost WAPE functions produce identical values."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([120.0, 180.0, 310.0, 390.0])

        # LGBM (sklearn API signature: y_true, y_pred)
        _, lgbm_val, _ = _wape_lgbm(y_true, y_pred)

        # XGBoost (sklearn API signature: y_true, y_pred -> float)
        xgb_val = _wape_xgb(y_true, y_pred)

        # CatBoost
        cat_metric = WapeMetric()
        cat_val, _ = cat_metric.evaluate([y_pred.tolist()], y_true.tolist(), None)

        assert lgbm_val == pytest.approx(xgb_val)
        assert lgbm_val == pytest.approx(cat_val)

        # Verify the value: |20|+|20|+|10|+|10| = 60, |sum(A)| = 1000
        assert lgbm_val == pytest.approx(60.0 / 1000.0)
