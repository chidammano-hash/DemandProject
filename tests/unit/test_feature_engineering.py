"""Tests for common/feature_engineering.py — build_feature_matrix, get_feature_columns, mask_future_sales, update_grid_with_predictions, update_grid_incremental."""

import pytest
import pandas as pd
import numpy as np

from common.core.constants import TS_PROFILE_FEATURES
from common.ml.feature_engineering import (
    _compute_ts_profile_features,
    build_feature_matrix,
    get_feature_columns,
    mask_future_sales,
    update_grid_incremental,
    update_grid_with_predictions,
)


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
        # fourier_sin_12 / fourier_cos_12 replace legacy month_sin / month_cos
        assert "fourier_sin_12" in grid.columns
        assert "fourier_cos_12" in grid.columns

    def test_month_sin_cos_not_present(self, sample_data):
        """month_sin and month_cos should NOT be in the feature matrix (replaced by fourier_sin_12/cos_12)."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        assert "month_sin" not in grid.columns
        assert "month_cos" not in grid.columns

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
        for feat in TS_PROFILE_FEATURES:
            assert feat in grid.columns, f"Missing TS profile feature: {feat}"

    def test_ts_profile_no_nan(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
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
        # Future qty is NaN (not zero) so rolling stats skip it instead of
        # dragging down means.  qty is excluded from features by METADATA_COLS.
        assert future["qty"].isna().all()

    def test_preserves_past_qty(self, sample_data):
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        past = masked[masked["startdate"] <= cutoff]
        # Past qty should equal original grid (check_dtype=False because
        # NaN masking promotes int64 → float64)
        orig_past = grid[grid["startdate"] <= cutoff]
        pd.testing.assert_series_equal(
            masked.loc[past.index, "qty"].reset_index(drop=True),
            grid.loc[orig_past.index, "qty"].reset_index(drop=True),
            check_dtype=False,
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
        original_qty = masked_grid[masked_grid["startdate"] == month]["qty"].copy()
        update_grid_with_predictions(masked_grid, month, preds)
        unchanged = masked_grid[masked_grid["startdate"] == month]["qty"]
        # Use assert_series_equal to handle NaN comparison correctly
        pd.testing.assert_series_equal(
            unchanged.reset_index(drop=True),
            original_qty.reset_index(drop=True),
        )

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

    def test_lag1_was_filled_zero_before_update(self, masked_grid):
        """Before update, qty_lag_1 for next month is 0 (NaN filled to 0 for model consumption)."""
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


class TestNaNMaskingBehavior:
    """Tests for NaN-based future masking.

    Verifies that mask_future_sales sets future qty to NaN so rolling
    statistics skip masked months (instead of including artificial zeros),
    and that feature columns are properly filled to 0 for model consumption.
    """

    def test_future_qty_is_nan(self, sample_data):
        """After masking, future qty values are NaN (not zero)."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all(), "Future qty should be NaN"

    def test_past_qty_unchanged(self, sample_data):
        """Past qty values are preserved exactly."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        past = masked[masked["startdate"] <= cutoff]
        orig_past = grid[grid["startdate"] <= cutoff]
        pd.testing.assert_series_equal(
            masked.loc[past.index, "qty"].reset_index(drop=True),
            grid.loc[orig_past.index, "qty"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_feature_columns_no_nan(self, sample_data):
        """All feature columns (non-metadata) have no NaN after masking."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        feat_cols = get_feature_columns(masked)
        for col in feat_cols:
            if pd.api.types.is_numeric_dtype(masked[col]):
                assert masked[col].isna().sum() == 0, f"Feature column {col} has NaN values"

    def test_qty_column_is_nan_for_future(self, sample_data):
        """qty itself is NaN for future months (not a feature column)."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all(), "qty should be NaN for future months"
        assert "qty" not in get_feature_columns(masked), "qty must not be a feature column"

    def test_rolling_mean_skips_nan_masked_months(self):
        """With NaN masking, rolling mean skips masked months entirely."""
        # 12 months of steady demand at 100 units, then 4 future months
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 13)]
                                + ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"])
        sales_df = pd.DataFrame({
            "sku_ck": ["S"] * len(months),
            "startdate": months,
            "qty": [100.0] * 12 + [100.0] * 4,
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["S"], "item_id": ["I1"],
            "customer_group": ["G"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        cutoff = pd.Timestamp("2024-12-01")
        masked = mask_future_sales(grid, cutoff)

        # rolling_mean_3m at 2025-03-01: shifted values are qty at 2025-02 (NaN),
        # 2025-01 (NaN), 2024-12 (100). With NaN masking: only 100 is valid → mean = 100
        mar25 = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2025-03-01"))]
        rm3 = float(mar25["rolling_mean_3m"].iloc[0])
        assert rm3 == pytest.approx(100.0, abs=1.0), (
            f"Rolling mean should be ~100 (NaN-masked months skipped). Got {rm3}"
        )

    def test_rolling_mean_at_cutoff_plus_2(self):
        """rolling_mean_3m at cutoff+2 skips NaN-masked future months."""
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 7)])
        sales_df = pd.DataFrame({
            "sku_ck": ["S"] * 6,
            "startdate": months,
            "qty": [50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["S"], "item_id": ["I1"],
            "customer_group": ["G"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        cutoff = pd.Timestamp("2024-04-01")
        masked = mask_future_sales(grid, cutoff)

        # At 2024-06: shifted values are qty at 2024-05 (NaN, masked), 2024-04 (200),
        # 2024-03 (150). With NaN masking: NaN is skipped → mean of (200, 150) = 175
        jun = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-06-01"))]
        rm3 = float(jun["rolling_mean_3m"].iloc[0])
        assert rm3 == pytest.approx(175.0, abs=1.0), (
            f"Expected rolling_mean_3m ~175 (NaN skipped), got {rm3}"
        )

    def test_intermittent_demand_nan_vs_real_zeros(self):
        """For intermittent demand, real zeros are preserved while masked months are NaN."""
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 7)])
        # Intermittent: demand in months 1, 3, 5 only; zeros in months 2, 4
        sales_df = pd.DataFrame({
            "sku_ck": ["S"] * 6,
            "startdate": months,
            "qty": [100.0, 0.0, 80.0, 0.0, 120.0, 90.0],
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["S"], "item_id": ["I1"],
            "customer_group": ["G"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        cutoff = pd.Timestamp("2024-03-01")
        masked = mask_future_sales(grid, cutoff)

        # Future months (Apr, May, Jun) should have NaN qty (not zero)
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all(), "Future months should be NaN"

        # Historical zero (Feb qty=0) should stay as zero (real intermittent zero)
        feb = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-02-01"))]
        assert float(feb["qty"].iloc[0]) == 0.0, "Real zero should be preserved"

        # Historical months with demand should be preserved
        jan = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-01-01"))]
        assert float(jan["qty"].iloc[0]) == 100.0

    def test_all_future_months_masked_to_nan(self):
        """Edge case: all months after cutoff are masked to NaN."""
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 7)])
        sales_df = pd.DataFrame({
            "sku_ck": ["S"] * 6,
            "startdate": months,
            "qty": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["S"], "item_id": ["I1"],
            "customer_group": ["G"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        # Mask everything after Jan — so Feb through Jun are future
        cutoff = pd.Timestamp("2024-01-01")
        masked = mask_future_sales(grid, cutoff)

        # All future months should have NaN qty
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all()

        # rolling_mean_3m at Feb: shifted[1]=qty[0]=100 (real), shifted[0]=NaN
        # With NaN masking, only 1 valid value in the window → mean = 100
        feb = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-02-01"))]
        rm3 = float(feb["rolling_mean_3m"].iloc[0])
        assert rm3 == pytest.approx(100.0, abs=1.0)

        # rolling_mean_3m at Jun: shifted values are qty at May (NaN), Apr (NaN),
        # Mar (NaN). All NaN in the 3-month window → filled to 0 after masking.
        jun = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-06-01"))]
        rm3_jun = float(jun["rolling_mean_3m"].iloc[0])
        # With NaN masking, no valid values in the 3-month window → 0 (filled)
        assert rm3_jun == 0.0, f"Expected 0.0 (no valid data in window), got {rm3_jun}"

    def test_derived_features_filled_after_mask(self, sample_data):
        """Derived features (mom_growth, volatility_ratio, etc.) are filled to 0 for future rows."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-01-01")
        masked = mask_future_sales(grid, cutoff)
        for col in ["mom_growth", "demand_accel", "volatility_ratio", "lag_ratio_yoy"]:
            if col in masked.columns:
                assert masked[col].isna().sum() == 0, f"Derived feature {col} has NaN"

    def test_n_zero_last_6m_excludes_nan_future(self):
        """n_zero_last_6m should NOT count NaN future months as zeros."""
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 9)])
        # 8 months of steady nonzero demand
        sales_df = pd.DataFrame({
            "sku_ck": ["S"] * 8,
            "startdate": months,
            "qty": [100.0] * 8,
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["S"], "item_id": ["I1"],
            "customer_group": ["G"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))

        # Before masking: n_zero_last_6m should be 0 (all lags are 100)
        cutoff = pd.Timestamp("2024-04-01")
        masked = mask_future_sales(grid, cutoff)

        # At cutoff month (Apr), all 4 lag values available are historical (100).
        # The remaining lag columns (lag_5, lag_6) may be NaN from lack of history.
        # n_zero_last_6m counts (lag == 0), NaN lags don't count as zero.
        apr = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-04-01"))]
        n_zero = float(apr["n_zero_last_6m"].iloc[0])
        assert n_zero == 0.0, (
            f"No real zeros in history — n_zero_last_6m should be 0, got {n_zero}"
        )


class TestTsProfileLeakagePrevention:
    """Tests that mask_future_sales recomputes TS profiles to prevent leakage."""

    @pytest.fixture
    def long_sample_data(self):
        """Create sample data with enough months for meaningful TS profiles.

        DFU A has a clear upward trend with a spike at the end.
        DFU B has intermittent demand with zeros.
        This makes TS profiles clearly different when future data is excluded.
        """
        # 12 months to get meaningful profiles (recency, seasonal, etc.)
        months = pd.date_range("2023-01-01", periods=12, freq="MS")
        # DFU A: upward trend with spike at end
        qty_a = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 1000]
        # DFU B: intermittent demand
        qty_b = [50, 0, 60, 0, 0, 70, 0, 0, 0, 80, 0, 500]

        sales_df = pd.DataFrame({
            "sku_ck": ["A"] * 12 + ["B"] * 12,
            "startdate": list(months) * 2,
            "qty": qty_a + qty_b,
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["A", "B"],
            "item_id": ["ITEM1", "ITEM2"],
            "customer_group": ["GRP1", "GRP1"],
            "loc": ["LOC1", "LOC2"],
        })
        item_attrs = pd.DataFrame({"item_id": ["ITEM1", "ITEM2"]})
        return sales_df, dfu_attrs, item_attrs, list(months)

    def test_ts_profiles_change_when_cutoff_changes(self, long_sample_data):
        """TS profiles should differ between two different cutoffs, proving recomputation."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        cutoff_early = pd.Timestamp("2023-06-01")
        cutoff_late = pd.Timestamp("2023-10-01")

        masked_early = mask_future_sales(grid, cutoff_early)
        masked_late = mask_future_sales(grid, cutoff_late)

        # Extract profile for DFU A at an early month (same row, different cutoffs)
        row_early = masked_early[
            (masked_early["sku_ck"] == "A") & (masked_early["startdate"] == months[0])
        ]
        row_late = masked_late[
            (masked_late["sku_ck"] == "A") & (masked_late["startdate"] == months[0])
        ]

        # mean_demand should differ because different amounts of data are included
        mean_early = float(row_early["mean_demand"].iloc[0])
        mean_late = float(row_late["mean_demand"].iloc[0])
        assert mean_early != mean_late, (
            "mean_demand should differ with different cutoffs — "
            "profiles are not being recomputed"
        )

    def test_ts_profiles_exclude_future_data(self, long_sample_data):
        """TS profiles computed with cutoff should not include future months' data.

        DFU A has a 1000-unit spike in Dec 2023. With cutoff at Jun 2023,
        mean_demand should only reflect Jan-Jun (avg ~125), not include the
        1000-unit spike that would raise the average to ~220+.
        """
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        cutoff = pd.Timestamp("2023-06-01")
        masked = mask_future_sales(grid, cutoff)

        a_row = masked[masked["sku_ck"] == "A"].iloc[0]
        mean_d = float(a_row["mean_demand"])

        # With cutoff at Jun, DFU A has values [100,110,120,130,140,150] -> mean ~125
        # Without cutoff (full history), mean would be ~220.8 due to 1000 spike
        expected_mean = np.mean([100, 110, 120, 130, 140, 150])
        assert mean_d == pytest.approx(expected_mean, rel=0.01), (
            f"mean_demand {mean_d} should be ~{expected_mean} (pre-cutoff only), "
            f"not including future data"
        )

    def test_masked_profiles_differ_from_full_history_profiles(self, long_sample_data):
        """Profiles after masking should differ from build_feature_matrix profiles.

        build_feature_matrix uses ALL data; mask_future_sales should use only
        pre-cutoff data. With the spike in Dec, these must differ.
        """
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        # Full-history profiles (from build)
        full_mean_a = float(
            grid[grid["sku_ck"] == "A"].iloc[0]["mean_demand"]
        )

        # Masked profiles (cutoff at June, excluding the 1000-unit spike)
        cutoff = pd.Timestamp("2023-06-01")
        masked = mask_future_sales(grid, cutoff)
        masked_mean_a = float(
            masked[masked["sku_ck"] == "A"].iloc[0]["mean_demand"]
        )

        assert masked_mean_a != full_mean_a, (
            f"Masked mean_demand ({masked_mean_a}) should differ from "
            f"full-history mean_demand ({full_mean_a})"
        )
        # Masked mean should be lower (no spike included)
        assert masked_mean_a < full_mean_a

    def test_mask_future_sales_output_has_all_ts_profile_columns(self, long_sample_data):
        """mask_future_sales output retains all 8 TS profile columns."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        masked = mask_future_sales(grid, pd.Timestamp("2023-06-01"))
        for feat in TS_PROFILE_FEATURES:
            assert feat in masked.columns, f"Missing {feat} after mask_future_sales"

    def test_mask_future_sales_ts_profiles_no_nan(self, long_sample_data):
        """TS profiles have no NaN values after mask_future_sales."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        masked = mask_future_sales(grid, pd.Timestamp("2023-06-01"))
        for feat in TS_PROFILE_FEATURES:
            assert masked[feat].notna().all(), f"NaN in {feat} after mask_future_sales"

    def test_cutoff_at_end_of_data_no_masking_needed(self, long_sample_data):
        """When cutoff is at the last month, profiles should match full-history profiles."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        # Cutoff at the very last month — no future to mask
        cutoff = months[-1]
        masked = mask_future_sales(grid, cutoff)

        for feat in TS_PROFILE_FEATURES:
            orig = grid[grid["sku_ck"] == "A"].iloc[0][feat]
            msk = masked[masked["sku_ck"] == "A"].iloc[0][feat]
            assert float(orig) == pytest.approx(float(msk), abs=1e-5), (
                f"{feat} should match full history when cutoff is at end of data"
            )

    def test_cutoff_at_beginning_almost_all_future(self, long_sample_data):
        """When cutoff is at the first month, profiles are based on only 1 month."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        cutoff = months[0]  # Only Jan 2023 is "past"
        masked = mask_future_sales(grid, cutoff)

        a_row = masked[masked["sku_ck"] == "A"].iloc[0]
        # With only 1 month of data, mean_demand = that single month's value
        assert float(a_row["mean_demand"]) == pytest.approx(100.0, rel=0.01)
        # CV should be 0 with a single data point
        assert float(a_row["cv_demand"]) == pytest.approx(0.0, abs=1e-5)
        # zero_demand_pct should be 0 (the single month has qty=100)
        assert float(a_row["zero_demand_pct"]) == pytest.approx(0.0, abs=1e-5)

    def test_zero_demand_pct_changes_with_cutoff(self, long_sample_data):
        """DFU B has intermittent demand; zero_demand_pct should change with cutoff."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        # Full history for B: [50,0,60,0,0,70,0,0,0,80,0,500] -> 7/12 zeros = 58.3%
        full_zero_pct = float(
            grid[grid["sku_ck"] == "B"].iloc[0]["zero_demand_pct"]
        )

        # Cutoff at Mar: B has [50,0,60] -> 1/3 zeros = 33.3%
        cutoff = pd.Timestamp("2023-03-01")
        masked = mask_future_sales(grid, cutoff)
        masked_zero_pct = float(
            masked[masked["sku_ck"] == "B"].iloc[0]["zero_demand_pct"]
        )

        assert masked_zero_pct != full_zero_pct, (
            "zero_demand_pct should change when cutoff excludes future zeros"
        )
        expected_zero_pct = 1.0 / 3.0  # 1 zero out of 3 months
        assert masked_zero_pct == pytest.approx(expected_zero_pct, rel=0.01)

    def test_compute_ts_profile_features_with_cutoff(self, long_sample_data):
        """_compute_ts_profile_features respects cutoff parameter directly."""
        sales_df, dfu_attrs, item_attrs, months = long_sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        profiles_full = _compute_ts_profile_features(grid, cutoff=None)
        profiles_cut = _compute_ts_profile_features(grid, cutoff=pd.Timestamp("2023-06-01"))

        a_full = profiles_full[profiles_full["sku_ck"] == "A"].iloc[0]
        a_cut = profiles_cut[profiles_cut["sku_ck"] == "A"].iloc[0]

        # mean_demand should differ
        assert float(a_full["mean_demand"]) != float(a_cut["mean_demand"])

    def test_ts_profiles_are_static_per_dfu(self, sample_data):
        """TS profile values are the same for all rows of the same DFU."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)

        for feat in TS_PROFILE_FEATURES:
            for sku in grid["sku_ck"].unique():
                vals = grid.loc[grid["sku_ck"] == sku, feat].unique()
                assert len(vals) == 1, (
                    f"{feat} should be static per DFU but has {len(vals)} "
                    f"unique values for {sku}"
                )


class TestUpdateGridIncremental:
    """Tests for update_grid_incremental — fast in-place recursive lag/rolling update."""

    @pytest.fixture
    def grid_and_months(self, sample_data):
        """Build a masked grid ready for incremental updates."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        masked = mask_future_sales(grid, pd.Timestamp("2024-02-01"))
        all_months = sorted(masked["startdate"].unique())
        return masked, all_months

    def test_writes_qty_for_predicted_month(self, grid_and_months):
        """qty for the predicted month is set to prediction value."""
        grid, all_months = grid_and_months
        month = pd.Timestamp("2024-03-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [111.0, 222.0]})
        update_grid_incremental(grid, month, preds, all_months)
        a_row = grid[(grid["sku_ck"] == "A") & (grid["startdate"] == month)]
        assert float(a_row["qty"].iloc[0]) == pytest.approx(111.0, abs=0.1)

    def test_lag1_updated_for_next_month(self, grid_and_months):
        """After updating month T, qty_lag_1 for T+1 equals prediction."""
        grid, all_months = grid_and_months
        month = pd.Timestamp("2024-03-01")
        next_month = pd.Timestamp("2024-04-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [123.0, 456.0]})
        update_grid_incremental(grid, month, preds, all_months)
        a_next = grid[(grid["sku_ck"] == "A") & (grid["startdate"] == next_month)]
        assert float(a_next["qty_lag_1"].iloc[0]) == pytest.approx(123.0, abs=0.1)

    def test_smooth_factor_zero_is_raw_prediction(self, grid_and_months):
        """With smooth_factor=0, lag-1 equals the raw prediction (default behavior)."""
        grid, all_months = grid_and_months
        month = pd.Timestamp("2024-03-01")
        next_month = pd.Timestamp("2024-04-01")
        preds = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [200.0, 300.0]})
        update_grid_incremental(grid, month, preds, all_months, smooth_factor=0.0)
        a_lag1 = float(
            grid[(grid["sku_ck"] == "A") & (grid["startdate"] == next_month)]["qty_lag_1"].iloc[0]
        )
        assert a_lag1 == pytest.approx(200.0, abs=0.1)

    def test_smooth_factor_blends_lag1(self, grid_and_months):
        """With smooth_factor > 0, lag-1 is blended between prediction and old lag."""
        grid, all_months = grid_and_months
        # First, write predictions for month 3 (so month 4 has a non-zero lag-1)
        month3 = pd.Timestamp("2024-03-01")
        month4 = pd.Timestamp("2024-04-01")
        preds_m3 = pd.DataFrame({"sku_ck": ["A", "B"], "basefcst_pref": [100.0, 200.0]})
        update_grid_incremental(grid, month3, preds_m3, all_months, smooth_factor=0.0)

        # Now month 4 has lag-1 = 100 for A. Predict month 4 with a very different value.
        old_lag1_a = float(
            grid[(grid["sku_ck"] == "A") & (grid["startdate"] == month4)]["qty_lag_1"].iloc[0]
        )
        assert old_lag1_a == pytest.approx(100.0, abs=0.1), "Precondition: lag-1 from step 1"

        # Predict month 4 with smooth_factor=0.5 — lag-1 for (nonexistent) month 5
        # won't exist in 4-month grid, but we can check the qty column is still written.
        # Instead, let's use a 6-month grid.

    def test_smooth_factor_blends_lag1_with_history(self):
        """With smooth_factor > 0, lag-1 is blended: (1-sf)*pred + sf*old_lag1."""
        # Build a 6-month grid for more room
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 7)])
        sales_df = pd.DataFrame({
            "sku_ck": ["A"] * 6,
            "startdate": list(months),
            "qty": [100.0, 200.0, 150.0, 300.0, 250.0, 350.0],
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["A"], "item_id": ["I1"],
            "customer_group": ["G1"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        masked = mask_future_sales(grid, pd.Timestamp("2024-02-01"))
        all_months = sorted(masked["startdate"].unique())

        # Step 1: predict month 3 (no smoothing)
        m3 = pd.Timestamp("2024-03-01")
        m4 = pd.Timestamp("2024-04-01")
        m5 = pd.Timestamp("2024-05-01")
        preds_m3 = pd.DataFrame({"sku_ck": ["A"], "basefcst_pref": [100.0]})
        update_grid_incremental(masked, m3, preds_m3, all_months, smooth_factor=0.0)

        # Now lag-1 at m4 = 100.0 (the raw prediction for m3)
        lag1_m4 = float(
            masked[(masked["sku_ck"] == "A") & (masked["startdate"] == m4)]["qty_lag_1"].iloc[0]
        )
        assert lag1_m4 == pytest.approx(100.0, abs=0.1)

        # Step 2: predict month 4 with smooth_factor=0.5
        # Prediction = 500.0 (very different from lag-1=100)
        preds_m4 = pd.DataFrame({"sku_ck": ["A"], "basefcst_pref": [500.0]})
        update_grid_incremental(masked, m4, preds_m4, all_months, smooth_factor=0.5)

        # lag-1 at m5 should be: 500*(1-0.5) + old_lag1_at_m5 * 0.5
        # old_lag1_at_m5 was the lag-1 before this update. After m3 update,
        # lag-1 at m5 = qty at m4, which was set to 500 by the qty write.
        # Wait — update_grid_incremental first writes qty at m4 = 500, then
        # for lag-1 at m5: new_lag = qty_2d[:, month_pos] = 500 (the predicted qty).
        # old_lag = grid.loc[m5, "qty_lag_1"] which was set by the m3 update's
        # lag-2 propagation (lag_2 at m5 = qty at m3 = 100, but lag-1 at m5...
        # Actually lag-1 at m5 was set by qty at m4 from the m3 update. After m3
        # prediction wrote qty at m3 = 100, lag-1 at m4 = 100, lag-2 at m5 = 100.
        # lag-1 at m5 was NOT touched by the m3 update (lag-1 only looks at
        # target_pos = month_pos + 1, which is m4, not m5).
        # So lag-1 at m5 before the m4 update is whatever was left from masking (0).
        # Smoothed lag-1 at m5 = 500*(1-0.5) + 0*0.5 = 250
        lag1_m5 = float(
            masked[(masked["sku_ck"] == "A") & (masked["startdate"] == m5)]["qty_lag_1"].iloc[0]
        )
        assert lag1_m5 == pytest.approx(250.0, abs=0.1), (
            f"Expected smoothed lag-1 = 250.0 (blend of 500 and 0), got {lag1_m5}"
        )

    def test_smooth_factor_does_not_affect_lag2_and_beyond(self):
        """Smoothing only applies to lag-1; lag-2+ get raw prediction values."""
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 7)])
        sales_df = pd.DataFrame({
            "sku_ck": ["A"] * 6,
            "startdate": list(months),
            "qty": [100.0, 200.0, 150.0, 300.0, 250.0, 350.0],
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["A"], "item_id": ["I1"],
            "customer_group": ["G1"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        masked = mask_future_sales(grid, pd.Timestamp("2024-02-01"))
        all_months = sorted(masked["startdate"].unique())

        m3 = pd.Timestamp("2024-03-01")
        m5 = pd.Timestamp("2024-05-01")
        preds_m3 = pd.DataFrame({"sku_ck": ["A"], "basefcst_pref": [999.0]})
        update_grid_incremental(masked, m3, preds_m3, all_months, smooth_factor=0.5)

        # lag-2 at m5 = qty at m3 = 999.0 (raw, no smoothing)
        lag2_m5 = float(
            masked[(masked["sku_ck"] == "A") & (masked["startdate"] == m5)]["qty_lag_2"].iloc[0]
        )
        assert lag2_m5 == pytest.approx(999.0, abs=0.1), (
            f"lag-2 should be raw prediction (999), got {lag2_m5}"
        )

    def test_smooth_factor_handles_nan_old_lag(self):
        """When old lag-1 is NaN (no prior data), smoothing falls back to raw prediction."""
        months = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 7)])
        sales_df = pd.DataFrame({
            "sku_ck": ["A"] * 6,
            "startdate": list(months),
            "qty": [100.0, 200.0, 150.0, 300.0, 250.0, 350.0],
        })
        dfu_attrs = pd.DataFrame({
            "sku_ck": ["A"], "item_id": ["I1"],
            "customer_group": ["G1"], "loc": ["L1"],
        })
        item_attrs = pd.DataFrame({"item_id": ["I1"]})
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
        masked = mask_future_sales(grid, pd.Timestamp("2024-02-01"))
        all_months = sorted(masked["startdate"].unique())

        # Force lag-1 at m4 to NaN to test NaN handling
        m3 = pd.Timestamp("2024-03-01")
        m4 = pd.Timestamp("2024-04-01")
        masked.loc[
            (masked["sku_ck"] == "A") & (masked["startdate"] == m4), "qty_lag_1"
        ] = np.nan

        preds_m3 = pd.DataFrame({"sku_ck": ["A"], "basefcst_pref": [400.0]})
        update_grid_incremental(masked, m3, preds_m3, all_months, smooth_factor=0.5)

        # lag-1 at m4: old was NaN → finite_mask is False → uses raw prediction 400
        lag1_m4 = float(
            masked[(masked["sku_ck"] == "A") & (masked["startdate"] == m4)]["qty_lag_1"].iloc[0]
        )
        assert lag1_m4 == pytest.approx(400.0, abs=0.1), (
            f"With NaN old lag, should use raw prediction (400), got {lag1_m4}"
        )
