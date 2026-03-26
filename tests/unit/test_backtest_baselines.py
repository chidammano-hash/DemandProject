"""Tests for baseline benchmark models (seasonal naive, rolling mean) in run_backtest.py."""

import pandas as pd
import numpy as np
import pytest

from scripts.run_backtest import (
    MODEL_REGISTRY,
    _BASELINE_PREDICT_FNS,
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
