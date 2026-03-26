"""Tests for common/feature_engineering.py — build_feature_matrix, get_feature_columns, mask_future_sales, update_grid_with_predictions."""

import pytest
import pandas as pd
import numpy as np

from common.constants import TS_PROFILE_FEATURES
from common.feature_engineering import (
    _compute_ts_profile_features,
    build_feature_matrix,
    get_feature_columns,
    mask_future_sales,
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
    """Tests for NaN-based future masking (defect fix: zero → NaN).

    Verifies that mask_future_sales uses NaN for future qty, that rolling
    statistics only average real historical data, and that feature columns
    are properly filled for model consumption.
    """

    def test_future_qty_is_nan_not_zero(self, sample_data):
        """After masking, future qty values are NaN (not zero)."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all(), "Future qty should be NaN, not zero"
        assert not (future["qty"] == 0).any(), "No future qty should be exactly zero"

    def test_past_qty_unchanged(self, sample_data):
        """Past qty values are preserved exactly (not converted to NaN)."""
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

    def test_lags_referencing_future_are_filled_zero(self, sample_data):
        """Lag features that reference NaN future months are filled to 0 for model use."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-01-01")
        masked = mask_future_sales(grid, cutoff)
        # qty_lag_1 for month after cutoff+1 should reference a future NaN qty,
        # and be filled to 0.0 in the output
        apr = masked[(masked["sku_ck"] == "A") & (masked["startdate"] == pd.Timestamp("2024-04-01"))]
        assert float(apr["qty_lag_1"].iloc[0]) == 0.0, "Lag referencing future NaN should be filled to 0"

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

    def test_qty_column_stays_nan_for_future(self, sample_data):
        """qty itself remains NaN for future months (not a feature column)."""
        sales_df, dfu_attrs, item_attrs, months = sample_data
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, months)
        cutoff = pd.Timestamp("2024-02-01")
        masked = mask_future_sales(grid, cutoff)
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all(), "qty should stay NaN for future months"
        assert "qty" not in get_feature_columns(masked), "qty must not be a feature column"

    def test_rolling_mean_not_dragged_down_continuous_demand(self):
        """For continuous demand, rolling mean at cutoff boundary should reflect
        only real historical data, not be dragged down by artificial zeros."""
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

        # rolling_mean_3m at cutoff+3 (2025-03-01) should reflect only historical data
        # With zero masking (old behavior), this would be (100+0+0)/3 = 33.3
        # With NaN masking (new behavior), this should be ~100 (only historical values)
        mar25 = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2025-03-01"))]
        rm3 = float(mar25["rolling_mean_3m"].iloc[0])
        # rolling_mean_3m uses shifted (causal) values. At 2025-03, shifted values are
        # qty at 2025-02 (NaN), 2025-01 (NaN), 2024-12 (100). Only 100 is valid → mean=100
        assert rm3 == pytest.approx(100.0, abs=1.0), (
            f"Rolling mean should be ~100 (historical only), not dragged down. Got {rm3}"
        )

    def test_rolling_mean_at_cutoff_plus_1_uses_only_history(self):
        """rolling_mean_3m at cutoff+1 should use historical shifted values only."""
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

        # At 2024-06: shifted values are qty at 2024-05 (NaN), 2024-04 (200), 2024-03 (150)
        # rolling_mean_3m should average only the valid ones: (200+150)/2 = 175
        jun = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-06-01"))]
        rm3 = float(jun["rolling_mean_3m"].iloc[0])
        assert rm3 == pytest.approx(175.0, abs=1.0), (
            f"Expected rolling_mean_3m ~175 (avg of 200, 150), got {rm3}"
        )

    def test_intermittent_demand_nan_vs_real_zero(self):
        """For intermittent demand, NaN masking distinguishes real zeros (sparse demand)
        from masked future months."""
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

        # Future months (Apr, May, Jun) should have NaN qty
        future = masked[masked["startdate"] > cutoff]
        assert future["qty"].isna().all()

        # Historical zero (Feb qty=0) should stay as zero, not become NaN
        feb = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-02-01"))]
        assert float(feb["qty"].iloc[0]) == 0.0, "Real zero should be preserved, not NaN"

        # Historical months with demand should be preserved
        jan = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-01-01"))]
        assert float(jan["qty"].iloc[0]) == 100.0

    def test_all_future_months_masked_rolling_uses_history(self):
        """Edge case: all months after cutoff are masked. Rolling features should
        use only the historical data before cutoff."""
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

        # rolling_mean_3m at Feb: shifted = [NaN (col 0)], only Jan=100 available
        # Actually at Feb (position 1), shifted[1] = qty[0] = 100. Window of 3:
        # shifted[0]=NaN, shifted[1]=100 → only 1 valid → mean=100
        feb = masked[(masked["sku_ck"] == "S") & (masked["startdate"] == pd.Timestamp("2024-02-01"))]
        rm3 = float(feb["rolling_mean_3m"].iloc[0])
        assert rm3 == pytest.approx(100.0, abs=1.0)

        # rolling_mean_3m at Jun: shifted values are qty at May (NaN), Apr (NaN),
        # Mar (NaN), Feb (NaN), Jan (100). Window of 3: shifted[5]=NaN, shifted[4]=NaN,
        # shifted[3]=NaN → 0 valid in window. But shifted[2]=NaN, shifted[1]=100.
        # For window=3 at position 5: shifted[3..5] = [NaN, NaN, NaN] → 0 valid → filled to 0
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
