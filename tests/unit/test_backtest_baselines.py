"""Tests for baseline benchmark models (seasonal naive, rolling mean) in run_backtest.py."""

import pandas as pd
import numpy as np
import pytest

from scripts.run_backtest import (
    MODEL_REGISTRY,
    _BASELINE_PREDICT_FNS,
    _RollingMeanModel,
    _SeasonalNaiveModel,
    _predict_rolling_mean,
    _predict_seasonal_naive,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_META_COLS = ["sku_ck", "item_id", "customer_group", "loc", "startdate"]


def _make_train_df(rows: list[dict]) -> pd.DataFrame:
    """Build a training DataFrame with required columns."""
    df = pd.DataFrame(rows)
    for col in _META_COLS + ["qty"]:
        if col not in df.columns:
            df[col] = 0
    return df


def _make_pred_df(rows: list[dict]) -> pd.DataFrame:
    """Build a prediction DataFrame with required meta columns."""
    df = pd.DataFrame(rows)
    for col in _META_COLS:
        if col not in df.columns:
            df[col] = 0
    return df


# ---------------------------------------------------------------------------
# Seasonal naive tests
# ---------------------------------------------------------------------------


class TestSeasonalNaive:
    """Tests for _predict_seasonal_naive."""

    def test_prediction_matches_prior_year_same_month(self):
        """Jan prediction should match the prior January's actual."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 100, "sku_ck": "ck1"},
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-02-01", "qty": 200, "sku_ck": "ck1"},
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-03-01", "qty": 150, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_seasonal_naive(train, pred)
        assert result["basefcst_pref"].iloc[0] == 100.0

    def test_uses_most_recent_year_when_multiple_years(self):
        """When multiple years have the same month, use the most recent."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2022-01-01", "qty": 50, "sku_ck": "ck1"},
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 100, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_seasonal_naive(train, pred)
        assert result["basefcst_pref"].iloc[0] == 100.0

    def test_missing_month_falls_back_to_dfu_mean(self):
        """When no matching month exists in training, fall back to DFU mean."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-02-01", "qty": 200, "sku_ck": "ck1"},
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-03-01", "qty": 100, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-06-01", "sku_ck": "ck1"},  # June not in training
        ])

        result = _predict_seasonal_naive(train, pred)
        expected_mean = (200.0 + 100.0) / 2
        assert result["basefcst_pref"].iloc[0] == expected_mean

    def test_no_training_data_returns_zero(self):
        """When DFU has zero training rows, predict 0."""
        train = _make_train_df([])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_seasonal_naive(train, pred)
        assert result["basefcst_pref"].iloc[0] == 0.0

    def test_unknown_dfu_returns_zero(self):
        """When prediction DFU does not exist in training at all, predict 0."""
        train = _make_train_df([
            {"item_id": "B", "customer_group": "G2", "loc": "L2",
             "startdate": "2023-01-01", "qty": 999, "sku_ck": "ck2"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_seasonal_naive(train, pred)
        assert result["basefcst_pref"].iloc[0] == 0.0

    def test_output_has_correct_columns(self):
        """Output should have all meta columns plus basefcst_pref."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 100, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_seasonal_naive(train, pred)
        expected_cols = {"sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"}
        assert set(result.columns) == expected_cols

    def test_predictions_non_negative(self):
        """Predictions should never be negative."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": -10, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_seasonal_naive(train, pred)
        assert (result["basefcst_pref"] >= 0).all()

    def test_multiple_dfus(self):
        """Correctly handles multiple DFUs in one call."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 100, "sku_ck": "ck1"},
            {"item_id": "B", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 300, "sku_ck": "ck2"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
            {"item_id": "B", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck2"},
        ])

        result = _predict_seasonal_naive(train, pred)
        assert len(result) == 2
        assert result["basefcst_pref"].iloc[0] == 100.0
        assert result["basefcst_pref"].iloc[1] == 300.0


# ---------------------------------------------------------------------------
# Rolling mean tests
# ---------------------------------------------------------------------------


class TestRollingMean:
    """Tests for _predict_rolling_mean."""

    def test_prediction_matches_mean_of_last_n_months(self):
        """With default window=6 and 6 months of data, mean of all 6."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": f"2023-{m:02d}-01", "qty": v, "sku_ck": "ck1"}
            for m, v in [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred, params={"window": 6})
        expected = (10 + 20 + 30 + 40 + 50 + 60) / 6
        assert abs(result["basefcst_pref"].iloc[0] - expected) < 0.01

    def test_uses_last_n_months_not_first(self):
        """Window should select the MOST RECENT months."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": f"2023-{m:02d}-01", "qty": v, "sku_ck": "ck1"}
            for m, v in [(1, 100), (2, 100), (3, 100), (4, 100), (5, 100), (6, 100),
                         (7, 200), (8, 200), (9, 200)]
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred, params={"window": 3})
        expected = (200 + 200 + 200) / 3
        assert abs(result["basefcst_pref"].iloc[0] - expected) < 0.01

    def test_fewer_months_than_window(self):
        """When fewer months exist than window, use all available months."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 100, "sku_ck": "ck1"},
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-02-01", "qty": 200, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        # Window=6 but only 2 months available -> mean of 2
        result = _predict_rolling_mean(train, pred, params={"window": 6})
        expected = (100 + 200) / 2
        assert abs(result["basefcst_pref"].iloc[0] - expected) < 0.01

    def test_no_training_data_returns_zero(self):
        """When DFU has zero training rows, predict 0."""
        train = _make_train_df([])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred)
        assert result["basefcst_pref"].iloc[0] == 0.0

    def test_unknown_dfu_returns_zero(self):
        """When prediction DFU does not exist in training, predict 0."""
        train = _make_train_df([
            {"item_id": "B", "customer_group": "G2", "loc": "L2",
             "startdate": "2023-01-01", "qty": 999, "sku_ck": "ck2"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred)
        assert result["basefcst_pref"].iloc[0] == 0.0

    def test_output_has_correct_columns(self):
        """Output should have all meta columns plus basefcst_pref."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": 100, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred)
        expected_cols = {"sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"}
        assert set(result.columns) == expected_cols

    def test_default_window_is_6(self):
        """When no params are passed, window defaults to 6."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": f"2023-{m:02d}-01", "qty": m * 10, "sku_ck": "ck1"}
            for m in range(1, 13)
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result_default = _predict_rolling_mean(train, pred)
        result_explicit = _predict_rolling_mean(train, pred, params={"window": 6})
        assert result_default["basefcst_pref"].iloc[0] == result_explicit["basefcst_pref"].iloc[0]

    def test_predictions_non_negative(self):
        """Predictions should never be negative."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2023-01-01", "qty": -50, "sku_ck": "ck1"},
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred)
        assert (result["basefcst_pref"] >= 0).all()

    def test_window_12_uses_full_year(self):
        """Window=12 should include all 12 months."""
        train = _make_train_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": f"2023-{m:02d}-01", "qty": 100, "sku_ck": "ck1"}
            for m in range(1, 13)
        ])
        pred = _make_pred_df([
            {"item_id": "A", "customer_group": "G1", "loc": "L1",
             "startdate": "2024-01-01", "sku_ck": "ck1"},
        ])

        result = _predict_rolling_mean(train, pred, params={"window": 12})
        assert abs(result["basefcst_pref"].iloc[0] - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestBaselineRegistry:
    """Tests for baseline model registration in MODEL_REGISTRY."""

    def test_seasonal_naive_registered(self):
        """seasonal_naive should be in MODEL_REGISTRY."""
        assert "seasonal_naive" in MODEL_REGISTRY

    def test_rolling_mean_registered(self):
        """rolling_mean should be in MODEL_REGISTRY."""
        assert "rolling_mean" in MODEL_REGISTRY

    def test_seasonal_naive_baseline_flag(self):
        """seasonal_naive must have baseline=True."""
        assert MODEL_REGISTRY["seasonal_naive"]["baseline"] is True

    def test_rolling_mean_baseline_flag(self):
        """rolling_mean must have baseline=True."""
        assert MODEL_REGISTRY["rolling_mean"]["baseline"] is True

    def test_tree_models_not_baseline(self):
        """Tree models (lgbm, catboost, xgboost) should not have baseline flag."""
        for name in ("lgbm", "catboost", "xgboost"):
            assert MODEL_REGISTRY[name].get("baseline", False) is False

    def test_baseline_predict_fns_resolved(self):
        """The predict_fn string references should map to actual callables."""
        for name in ("seasonal_naive", "rolling_mean"):
            fn_ref = MODEL_REGISTRY[name]["predict_fn"]
            assert fn_ref in _BASELINE_PREDICT_FNS
            assert callable(_BASELINE_PREDICT_FNS[fn_ref])

    def test_baseline_registry_has_required_keys(self):
        """Baseline registry entries must have all keys needed by run_tree_backtest."""
        required = {"config_key", "default_params", "cat_dtype",
                     "model_params_key", "model_type_tag"}
        for name in ("seasonal_naive", "rolling_mean"):
            entry = MODEL_REGISTRY[name]
            missing = required - set(entry.keys())
            assert not missing, f"{name} missing registry keys: {missing}"

    def test_all_models_in_registry(self):
        """All 5 model types should be registered."""
        expected = {"lgbm", "catboost", "xgboost", "seasonal_naive", "rolling_mean"}
        assert expected.issubset(set(MODEL_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# _SeasonalNaiveModel tests (used for recursive intermittent prediction)
# ---------------------------------------------------------------------------


class TestSeasonalNaiveModel:
    """Tests for _SeasonalNaiveModel predict behaviour."""

    def test_predict_with_sku_ck_and_startdate_columns(self):
        """When X has sku_ck and startdate, use seasonal map directly."""
        seasonal_map = {("ck1", 1): 100.0, ("ck1", 6): 200.0}
        fallback = {"ck1": 50.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)

        X = pd.DataFrame({
            "sku_ck": ["ck1", "ck1"],
            "startdate": ["2024-01-01", "2024-06-01"],
            "feat1": [0.5, 0.3],
        })
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [100.0, 200.0])

    def test_predict_falls_back_to_rolling_mean_for_missing_month(self):
        """When seasonal map lacks a month, fall back to rolling mean."""
        seasonal_map = {("ck1", 1): 100.0}  # only January
        fallback = {"ck1": 42.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)

        X = pd.DataFrame({
            "sku_ck": ["ck1", "ck1"],
            "startdate": ["2024-01-01", "2024-07-01"],  # July not in seasonal
            "feat1": [0.5, 0.3],
        })
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [100.0, 42.0])

    def test_predict_falls_back_to_zero_for_unknown_sku(self):
        """When both seasonal and fallback miss a SKU, predict 0."""
        seasonal_map = {("ck1", 1): 100.0}
        fallback = {"ck1": 50.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)

        X = pd.DataFrame({
            "sku_ck": ["ck_unknown"],
            "startdate": ["2024-01-01"],
            "feat1": [0.5],
        })
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [0.0])

    def test_predict_with_sku_cks_and_months_attrs(self):
        """In recursive mode, _sku_cks and _months are set externally."""
        seasonal_map = {("ck1", 3): 300.0, ("ck2", 3): 150.0}
        fallback = {"ck1": 10.0, "ck2": 20.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)
        model._sku_cks = ["ck1", "ck2"]
        model._months = [3, 3]

        # X has only feature columns (no sku_ck or startdate)
        X = pd.DataFrame({"feat1": [0.5, 0.3], "feat2": [1.0, 2.0]})
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [300.0, 150.0])

    def test_predict_with_sku_cks_and_months_mixed_fallback(self):
        """Recursive mode: some months found in seasonal, some fall back."""
        seasonal_map = {("ck1", 1): 100.0}
        fallback = {"ck1": 25.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)
        model._sku_cks = ["ck1", "ck1"]
        model._months = [1, 8]  # Jan in seasonal, Aug not

        X = pd.DataFrame({"feat1": [0.5, 0.3]})
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [100.0, 25.0])

    def test_predict_with_only_sku_cks_no_months(self):
        """When _sku_cks is set but _months is not, fall back to rolling mean."""
        seasonal_map = {("ck1", 1): 999.0}
        fallback = {"ck1": 42.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)
        model._sku_cks = ["ck1"]
        # _months not set

        X = pd.DataFrame({"feat1": [0.5]})
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [42.0])

    def test_predict_no_sku_cks_returns_zeros(self):
        """When neither sku_ck column nor _sku_cks attr exists, return zeros."""
        seasonal_map = {("ck1", 1): 100.0}
        fallback = {"ck1": 50.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)

        X = pd.DataFrame({"feat1": [0.5, 0.3]})
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, [0.0, 0.0])

    def test_predictions_non_negative(self):
        """Negative values in seasonal map are clipped to 0."""
        seasonal_map = {("ck1", 1): -50.0}
        fallback = {"ck1": -10.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)

        X = pd.DataFrame({
            "sku_ck": ["ck1", "ck1"],
            "startdate": ["2024-01-01", "2024-06-01"],  # June falls back
            "feat1": [0.5, 0.3],
        })
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_different_months_get_different_predictions(self):
        """Core value prop: different months produce different forecasts."""
        seasonal_map = {
            ("ck1", 1): 10.0,
            ("ck1", 2): 0.0,
            ("ck1", 3): 50.0,
            ("ck1", 4): 0.0,
            ("ck1", 5): 30.0,
            ("ck1", 6): 0.0,
        }
        fallback = {"ck1": 15.0}
        model = _SeasonalNaiveModel(seasonal_map, fallback)
        model._sku_cks = ["ck1"] * 6
        model._months = [1, 2, 3, 4, 5, 6]

        X = pd.DataFrame({"feat1": [0.0] * 6})
        preds = model.predict(X)
        # Non-constant predictions (unlike _RollingMeanModel)
        assert len(set(preds)) > 1, "Seasonal naive should produce varying predictions"
        np.testing.assert_array_equal(preds, [10.0, 0.0, 50.0, 0.0, 30.0, 0.0])

    def test_rolling_mean_model_gives_constant(self):
        """Contrast: _RollingMeanModel gives the same value for every month."""
        rm = _RollingMeanModel({"ck1": 15.0})
        rm._sku_cks = ["ck1"] * 6

        X = pd.DataFrame({"feat1": [0.0] * 6})
        preds = rm.predict(X)
        assert len(set(preds)) == 1, "Rolling mean should produce constant predictions"
