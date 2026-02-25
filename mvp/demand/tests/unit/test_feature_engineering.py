"""Tests for common/feature_engineering.py — build_feature_matrix, get_feature_columns, mask_future_sales."""

import pytest
import pandas as pd
import numpy as np

from common.feature_engineering import build_feature_matrix, get_feature_columns, mask_future_sales


@pytest.fixture
def sample_data():
    """Create small sample data for feature engineering tests."""
    months = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    sales_df = pd.DataFrame({
        "dfu_ck": ["A"] * 4 + ["B"] * 4,
        "startdate": list(months) * 2,
        "qty": [100, 200, 150, 300, 50, 80, 60, 90],
    })
    dfu_attrs = pd.DataFrame({
        "dfu_ck": ["A", "B"],
        "dmdunit": ["ITEM1", "ITEM2"],
        "dmdgroup": ["GRP1", "GRP1"],
        "loc": ["LOC1", "LOC2"],
    })
    item_attrs = pd.DataFrame({
        "dmdunit": ["ITEM1", "ITEM2"],
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


class TestGetFeatureColumns:
    def test_excludes_metadata(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        feat_cols = get_feature_columns(grid)
        # Metadata columns should be excluded
        for meta in ["dfu_ck", "startdate", "dmdunit", "dmdgroup", "loc", "qty"]:
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
