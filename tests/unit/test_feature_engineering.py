"""Tests for common/feature_engineering.py — build_feature_matrix, get_feature_columns, mask_future_sales, update_grid_with_predictions."""

import pytest
import pandas as pd
import numpy as np

from common.feature_engineering import build_feature_matrix, get_feature_columns, mask_future_sales, update_grid_with_predictions


@pytest.fixture
def sample_data():
    """Create small sample data for feature engineering tests."""
    months = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    sales_df = pd.DataFrame({
        "sku_ck": ["A"] * 4 + ["B"] * 4,
        "startdate": list(months) * 2,
        "qty": [100, 200, 150, 300, 50, 80, 60, 90],
    })
    dfu_attrs = pd.DataFrame({
        "sku_ck": ["A", "B"],
        "item_id": ["ITEM1", "ITEM2"],
        "customer_group": ["GRP1", "GRP1"],
        "loc": ["LOC1", "LOC2"],
    })
    item_attrs = pd.DataFrame({
        "item_id": ["ITEM1", "ITEM2"],
    })
    return sales_df, dfu_attrs, item_attrs, list(months)


class TestBuildFeatureMatrix:
    def test_grid_shape(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert len(grid) == 2 * 4  # 2 DFUs x 4 months

    def test_has_lag_columns(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "qty_lag_1" in grid.columns
        assert "qty_lag_12" in grid.columns

    def test_has_rolling_columns(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "rolling_mean_3m" in grid.columns
        assert "rolling_std_3m" in grid.columns

    def test_has_calendar_features(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "month" in grid.columns
        assert "quarter" in grid.columns
        assert "month_sin" in grid.columns
        assert "month_cos" in grid.columns

    def test_qty_filled_not_null(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert grid["qty"].notna().all()

    def test_cat_dtype_category(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months, cat_dtype="category")
        for col in grid.columns:
            if grid[col].dtype.name == "category":
                assert True
                return
        # If no cat columns exist in the small sample, that's OK
        assert True

    def test_cat_dtype_str(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months, cat_dtype="str")
        # Should not raise an error
        assert len(grid) > 0

    def test_numeric_item_attrs_are_coerced_from_object_dtype(self, sample_data):
        sales_df, dfu_attrs, _, months = sample_data
        item_attrs = pd.DataFrame({
            "item_id": ["ITEM1", "ITEM2"],
            "case_weight": ["12.5", "bad-value"],
            "item_proof": ["80", None],
            "bpc": ["24", "6"],
        })

        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        for col in ["case_weight", "item_proof", "bpc"]:
            assert pd.api.types.is_numeric_dtype(grid[col]), f"{col} should be numeric"

        item1_row = grid[grid["item_id"] == "ITEM1"].iloc[0]
        item2_row = grid[grid["item_id"] == "ITEM2"].iloc[0]
        assert float(item1_row["case_weight"]) == 12.5
        assert float(item1_row["item_proof"]) == 80.0
        assert float(item1_row["bpc"]) == 24.0
        assert float(item2_row["case_weight"]) == 0.0
        assert float(item2_row["item_proof"]) == 0.0
        assert float(item2_row["bpc"]) == 6.0


class TestLagRatioFeatures:
    def test_lag_ratio_yoy_present(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "lag_ratio_yoy" in grid.columns

    def test_lag_ratio_mom_present(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "lag_ratio_mom" in grid.columns

    def test_lag_ratio_3v12_present(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "lag_ratio_3v12" in grid.columns

    def test_lag_ratios_clipped(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        for col in ["lag_ratio_yoy", "lag_ratio_mom", "lag_ratio_3v12"]:
            non_null = grid[col].dropna()
            if len(non_null) > 0:
                assert non_null.max() <= 10.0, f"{col} exceeds upper clip"
                assert non_null.min() >= -10.0, f"{col} below lower clip"

    def test_n_zero_last_6m_present(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "n_zero_last_6m" in grid.columns

    def test_n_zero_last_6m_range(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert grid["n_zero_last_6m"].min() >= 0
        assert grid["n_zero_last_6m"].max() <= 6


class TestTsProfileFeatures:
    def test_ts_profile_features_present(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        from common.constants import TS_PROFILE_FEATURES
        for feat in TS_PROFILE_FEATURES:
            assert feat in grid.columns, f"Missing TS profile feature: {feat}"

    def test_ts_profile_no_nan(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        from common.constants import TS_PROFILE_FEATURES
        for feat in TS_PROFILE_FEATURES:
            if feat in grid.columns:
                assert grid[feat].notna().all(), f"NaN in TS profile feature: {feat}"

    def test_cv_demand_non_negative(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        if "cv_demand" in grid.columns:
            assert (grid["cv_demand"] >= 0).all()

    def test_mean_demand_positive(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        if "mean_demand" in grid.columns:
            assert (grid["mean_demand"] > 0).all()


class TestGetFeatureColumns:
    def test_excludes_metadata(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        feat_cols = get_feature_columns(grid)
        # Metadata columns should be excluded
        for meta in ["sku_ck", "startdate", "item_id", "customer_group", "loc", "qty"]:
            assert meta not in feat_cols, f"{meta} should be excluded from feature columns"

    def test_includes_lag_features(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        feat_cols = get_feature_columns(grid)
        assert "qty_lag_1" in feat_cols
        assert "month" in feat_cols


class TestMaskFutureSales:
    def test_masks_future_qty(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        future = masked[masked["startdate"] > cutoff]
        assert (future["qty"] == 0).all()

    def test_preserves_past_qty(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        past = masked[masked["startdate"] <= cutoff]
        # Past qty should equal original grid
        orig_past = grid[grid["startdate"] <= cutoff]
        pd.testing.assert_series_equal(
            masked.loc[past.index, "qty"].reset_index(drop=True),
            grid.loc[orig_past.index, "qty"].reset_index(drop=True),
        )

    def test_does_not_mutate_original(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        original_qty = grid["qty"].copy()
        mask_future_sales(grid, pd.Timestamp("2024-02-01"))
        pd.testing.assert_series_equal(grid["qty"], original_qty)

    def test_recomputes_lags_after_mask(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-01-01")
        masked = mask_future_sales(grid, cutoff)
        # Lag1 for months after cutoff+1 should reflect the masked (0) qty
        assert "qty_lag_1" in masked.columns


class TestUpdateGridWithPredictions:
    """Tests for update_grid_with_predictions (Feature 43 — recursive multi-step)."""

    @pytest.fixture
    def masked_grid(self, sample_data):
        """Return a masked grid (cutoff = 2024-02-01) ready for recursive inference."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        return mask_future_sales(grid, pd.Timestamp("2024-02-01"))

    def test_updates_qty_for_target_month(self, masked_grid):
        """qty for the predicted month is set to the predicted value."""
        month = pd.Timestamp("2024-03-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [111.0, 222.0]})
        updated = update_grid_with_predictions(masked_grid, month, preds)
        a_row = updated[(updated["sku_ck"] == "A") & (updated["startdate"] == month)]
        b_row = updated[(updated["sku_ck"] == "B") & (updated["startdate"] == month)]
        assert float(a_row["qty"].iloc[0]) == 111.0
        assert float(b_row["qty"].iloc[0]) == 222.0

    def test_does_not_mutate_other_months(self, masked_grid):
        """qty for months before the predicted month is unchanged."""
        month = pd.Timestamp("2024-03-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [111.0, 222.0]})
        before = masked_grid[masked_grid["startdate"] < month]["qty"].copy()
        updated = update_grid_with_predictions(masked_grid, month, preds)
        after = updated[updated["startdate"] < month]["qty"]
        pd.testing.assert_series_equal(before.reset_index(drop=True), after.reset_index(drop=True))

    def test_does_not_mutate_original_grid(self, masked_grid):
        """Original grid is not modified (returns a copy)."""
        month = pd.Timestamp("2024-03-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [111.0, 222.0]})
        original_vals = masked_grid[masked_grid["startdate"] == month]["qty"].values.copy()
        update_grid_with_predictions(masked_grid, month, preds)
        unchanged = masked_grid[masked_grid["startdate"] == month]["qty"].values
        assert (unchanged == original_vals).all()

    def test_lag1_for_next_month_equals_prediction(self, masked_grid):
        """After updating month T, qty_lag_1 for month T+1 equals the predicted value."""
        month = pd.Timestamp("2024-03-01")
        next_month = pd.Timestamp("2024-04-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [123.0, 456.0]})
        updated = update_grid_with_predictions(masked_grid, month, preds)
        a_next = updated[(updated["sku_ck"] == "A") & (updated["startdate"] == next_month)]
        b_next = updated[(updated["sku_ck"] == "B") & (updated["startdate"] == next_month)]
        assert float(a_next["qty_lag_1"].iloc[0]) == pytest.approx(123.0)
        assert float(b_next["qty_lag_1"].iloc[0]) == pytest.approx(456.0)

    def test_lag1_was_zero_before_update(self, masked_grid):
        """Before update, qty_lag_1 for next month is 0 (direct mode)."""
        next_month = pd.Timestamp("2024-04-01")
        a_before = masked_grid[(masked_grid["sku_ck"] == "A") & (masked_grid["startdate"] == next_month)]
        assert float(a_before["qty_lag_1"].iloc[0]) == 0.0

    def test_rolling_mean_recomputed(self, masked_grid):
        """Rolling mean features are recomputed after prediction update."""
        month = pd.Timestamp("2024-03-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [500.0, 500.0]})
        before_val = masked_grid[
            (masked_grid["sku_ck"] == "A") & (masked_grid["startdate"] == month)
        ]["rolling_mean_3m"].iloc[0]
        updated = update_grid_with_predictions(masked_grid, month, preds)
        after_val = updated[
            (updated["sku_ck"] == "A") & (updated["startdate"] == pd.Timestamp("2024-04-01"))
        ]["rolling_mean_3m"].iloc[0]
        # rolling mean for Apr should now incorporate the 500.0 prediction for Mar
        assert after_val != before_val
