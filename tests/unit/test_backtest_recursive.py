"""Unit tests for recursive multi-step inference helpers (Feature 43).

Tests cover:
- _fill_predict_nans: NaN filling for numeric feature columns
- _predict_single_month: per-cluster model routing
- Integration: recursive loop produces non-zero lag_1 for month 2
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from common.backtest_framework import _fill_predict_nans, _predict_single_month
from common.feature_engineering import (
    build_feature_matrix,
    mask_future_sales,
    update_grid_with_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid():
    """Tiny 2-DFU × 4-month grid masked at 2024-02-01."""
    months = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    sales_df = pd.DataFrame({
        "sku_ck": ["A"] * 4 + ["B"] * 4,
        "startdate": list(months) * 2,
        "qty": [100, 200, 0, 0, 50, 80, 0, 0],
    })
    dfu_attrs = pd.DataFrame({
        "sku_ck": ["A", "B"],
        "item_id": ["I1", "I2"],
        "customer_group": ["G", "G"],
        "loc": ["L1", "L2"],
    })
    item_attrs = pd.DataFrame({"item_id": ["I1", "I2"]})
    grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
    return mask_future_sales(grid, pd.Timestamp("2024-02-01"))


@pytest.fixture
def predict_data_one_month(simple_grid):
    """Predict-data slice for 2024-03-01 (single month, 2 DFUs)."""
    month = pd.Timestamp("2024-03-01")
    data = simple_grid[simple_grid["startdate"] == month].copy()
    data["ml_cluster"] = ["cluster_A", "cluster_B"]
    return data


# ---------------------------------------------------------------------------
# _fill_predict_nans
# ---------------------------------------------------------------------------

class TestFillPredictNans:
    def test_fills_numeric_nans(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
        result = _fill_predict_nans(df.copy(), feature_cols=["a", "b"], cat_cols=[])
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0

    def test_skips_categorical_columns(self):
        df = pd.DataFrame({"a": [None, None], "cat": [None, None]})
        result = _fill_predict_nans(df.copy(), feature_cols=["a", "cat"], cat_cols=["cat"])
        assert result["a"].isna().sum() == 0
        assert result["cat"].isna().sum() == 2

    def test_skips_columns_not_in_feature_cols(self):
        df = pd.DataFrame({"a": [1.0, None], "extra": [None, None]})
        result = _fill_predict_nans(df.copy(), feature_cols=["a"], cat_cols=[])
        assert result["extra"].isna().sum() == 2

    def test_preserves_non_nan_values(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = _fill_predict_nans(df.copy(), feature_cols=["a"], cat_cols=[])
        assert list(result["a"]) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# _predict_single_month — per_cluster routing
# ---------------------------------------------------------------------------

class TestPredictSingleMonth:
    def _make_cluster_models(self, val_a=10.0, val_b=20.0):
        m_a = MagicMock()
        m_a.predict.return_value = np.array([val_a])
        m_b = MagicMock()
        m_b.predict.return_value = np.array([val_b])
        return {"cluster_A": m_a, "cluster_B": m_b}

    def test_routes_by_ml_cluster(self, predict_data_one_month):
        """Each DFU is routed to its cluster's model."""
        feature_cols = ["qty_lag_1"]
        models = self._make_cluster_models(val_a=55.0, val_b=77.0)
        result = _predict_single_month(models, predict_data_one_month, feature_cols)
        a_pred = result[result["sku_ck"] == "A"]["basefcst_pref"].iloc[0]
        b_pred = result[result["sku_ck"] == "B"]["basefcst_pref"].iloc[0]
        assert a_pred == pytest.approx(55.0)
        assert b_pred == pytest.approx(77.0)

    def test_clips_negative_predictions_to_zero(self, predict_data_one_month):
        """Negative predictions are clipped to 0."""
        feature_cols = ["qty_lag_1"]
        m_a = MagicMock()
        m_a.predict.return_value = np.array([-5.0])
        m_b = MagicMock()
        m_b.predict.return_value = np.array([10.0])
        models = {"cluster_A": m_a, "cluster_B": m_b}
        result = _predict_single_month(models, predict_data_one_month, feature_cols)
        assert result[result["sku_ck"] == "A"]["basefcst_pref"].iloc[0] == 0.0
        assert result[result["sku_ck"] == "B"]["basefcst_pref"].iloc[0] == 10.0

    def test_returns_required_columns(self, predict_data_one_month):
        """Output contains all required metadata columns + basefcst_pref."""
        feature_cols = ["qty_lag_1"]
        models = self._make_cluster_models()
        result = _predict_single_month(models, predict_data_one_month, feature_cols)
        for col in ["sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"]:
            assert col in result.columns

    def test_skips_unknown_cluster(self, predict_data_one_month):
        """DFUs whose cluster has no model are omitted from output."""
        feature_cols = ["qty_lag_1"]
        data = predict_data_one_month.copy()
        data["ml_cluster"] = ["unknown_cluster", "cluster_B"]
        m_b = MagicMock()
        m_b.predict.return_value = np.array([20.0])
        models = {"cluster_B": m_b}
        result = _predict_single_month(models, data, feature_cols)
        assert len(result) == 1
        assert result["sku_ck"].iloc[0] == "B"

    def test_empty_predict_data_returns_empty_df(self):
        """Empty predict_data returns empty DataFrame with correct columns."""
        data = pd.DataFrame(
            columns=["sku_ck", "item_id", "customer_group", "loc", "startdate", "ml_cluster", "qty_lag_1"]
        )
        models = {"cluster_A": MagicMock()}
        result = _predict_single_month(models, data, ["qty_lag_1"])
        assert len(result) == 0
        assert "basefcst_pref" in result.columns

    def test_passes_all_feature_cols_including_ml_cluster(self, predict_data_one_month):
        """ml_cluster is included in X passed to per-cluster models (trained with it)."""
        feature_cols = ["qty_lag_1", "ml_cluster"]
        models = self._make_cluster_models()
        _predict_single_month(models, predict_data_one_month, feature_cols)
        call_arg = models["cluster_A"].predict.call_args[0][0]
        assert "ml_cluster" in call_arg.columns
        assert list(call_arg.columns) == feature_cols


# ---------------------------------------------------------------------------
# Integration: recursive loop produces non-zero lag_1 for month 2
# ---------------------------------------------------------------------------

class TestRecursiveLoopIntegration:
    def test_lag1_for_month2_is_prediction_not_zero(self, simple_grid):
        """Verify the recursive update produces a non-zero lag_1 for month 2 of predict window."""
        month1 = pd.Timestamp("2024-03-01")
        month2 = pd.Timestamp("2024-04-01")

        m2_before = simple_grid[
            (simple_grid["sku_ck"] == "A") & (simple_grid["startdate"] == month2)
        ]["qty_lag_1"].iloc[0]
        # After NaN masking, feature columns are filled to 0 for model consumption
        assert m2_before == 0.0

        preds_m1 = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [123.0, 456.0]})
        updated = update_grid_with_predictions(simple_grid, month1, preds_m1)

        m2_after = updated[
            (updated["sku_ck"] == "A") & (updated["startdate"] == month2)
        ]["qty_lag_1"].iloc[0]
        assert m2_after == pytest.approx(123.0)
        assert m2_after != 0.0

    def test_two_month_recursive_chain(self, simple_grid):
        """Two recursive updates: qty for month 2 equals prediction for month 2."""
        month1 = pd.Timestamp("2024-03-01")
        month2 = pd.Timestamp("2024-04-01")

        preds_m1 = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [100.0, 200.0]})
        grid1 = update_grid_with_predictions(simple_grid, month1, preds_m1)

        preds_m2 = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [75.0, 150.0]})
        grid2 = update_grid_with_predictions(grid1, month2, preds_m2)

        a_m2 = grid2[(grid2["sku_ck"] == "A") & (grid2["startdate"] == month2)]["qty"].iloc[0]
        assert a_m2 == pytest.approx(75.0)

    def test_direct_and_recursive_differ(self, simple_grid):
        """Recursive inference produces different lag values than direct inference."""
        month1 = pd.Timestamp("2024-03-01")
        month2 = pd.Timestamp("2024-04-01")

        direct_lag = simple_grid[
            (simple_grid["sku_ck"] == "A") & (simple_grid["startdate"] == month2)
        ]["qty_lag_1"].iloc[0]

        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [150.0, 50.0]})
        recursive_grid = update_grid_with_predictions(simple_grid, month1, preds)
        recursive_lag = recursive_grid[
            (recursive_grid["sku_ck"] == "A") & (recursive_grid["startdate"] == month2)
        ]["qty_lag_1"].iloc[0]

        # After NaN masking, feature columns are filled to 0 for model consumption
        assert direct_lag == 0.0
        assert recursive_lag == pytest.approx(150.0)
        assert recursive_lag != direct_lag
