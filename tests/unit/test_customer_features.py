"""Tests for customer-derived feature generation."""

import pytest
import numpy as np
import pandas as pd

from scripts.ml.generate_customer_features import compute_features


@pytest.fixture
def sample_demand():
    """Build sample customer demand data (3 customers, 2 items, 1 location, 6 months)."""
    months = pd.date_range("2025-07-01", periods=6, freq="MS")
    rows = []
    # Item A, loc L1: 3 customers
    for m in months:
        rows.append({"item_id": "A", "loc": "L1", "customer_no": "C1", "startdate": m,
                      "demand_qty": 100, "sales_qty": 90, "oos_qty": 10, "channel": "On Premise"})
        rows.append({"item_id": "A", "loc": "L1", "customer_no": "C2", "startdate": m,
                      "demand_qty": 50, "sales_qty": 50, "oos_qty": 0, "channel": "Off Premise"})
        rows.append({"item_id": "A", "loc": "L1", "customer_no": "C3", "startdate": m,
                      "demand_qty": 50, "sales_qty": 50, "oos_qty": 0, "channel": "On Premise"})
    # Item B, loc L1: 1 customer (monopoly)
    for m in months:
        rows.append({"item_id": "B", "loc": "L1", "customer_no": "C1", "startdate": m,
                      "demand_qty": 200, "sales_qty": 150, "oos_qty": 50, "channel": "On Premise"})
    return pd.DataFrame(rows)


class TestCustomerFeatureGrain:
    def test_output_grain_is_item_loc_month(self, sample_demand):
        features = compute_features(sample_demand)
        assert not features.empty
        # Should have one row per (item_id, loc, startdate) — no duplicates
        dup = features.duplicated(subset=["item_id", "loc", "startdate"])
        assert dup.sum() == 0

    def test_correct_item_loc_count(self, sample_demand):
        features = compute_features(sample_demand)
        # 2 items × 6 months = 12 rows
        assert len(features) == 12


class TestConcentrationFeatures:
    def test_hhi_single_customer(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        # Single customer → HHI = 1.0
        for _, row in item_b.iterrows():
            assert row["hhi_demand"] == pytest.approx(1.0, abs=0.01)

    def test_hhi_diversified_lower(self, sample_demand):
        features = compute_features(sample_demand)
        item_a = features[features["item_id"] == "A"]
        item_b = features[features["item_id"] == "B"]
        # 3 customers should have lower HHI than 1 customer
        assert item_a.iloc[-1]["hhi_demand"] < item_b.iloc[-1]["hhi_demand"]

    def test_top1_share_single_customer(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        assert item_b.iloc[-1]["top1_cust_share"] == pytest.approx(1.0, abs=0.01)

    def test_n_active_cust(self, sample_demand):
        features = compute_features(sample_demand)
        item_a = features[(features["item_id"] == "A") & (features["startdate"] == features["startdate"].max())]
        assert item_a.iloc[0]["n_active_cust"] == 3

    def test_gini_single_customer_zero(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        # Single customer → Gini = 0
        assert item_b.iloc[-1]["cust_gini"] == pytest.approx(0.0, abs=0.01)


class TestTrueDemandFeatures:
    def test_true_demand_ratio_with_stockout(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        # demand=200, sales=150 → ratio = 200/150 ≈ 1.333
        assert item_b.iloc[-1]["true_demand_ratio"] > 1.0

    def test_oos_rate_positive(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        # oos_qty=50, demand_qty=200 → oos_rate = 0.25
        assert item_b.iloc[-1]["oos_rate"] > 0

    def test_no_stockout_ratio_near_one(self, sample_demand):
        # Item A, customer C2 has no OOS, but overall item A has C1 with OOS
        features = compute_features(sample_demand)
        # Overall item A: demand=200, sales=190, oos=10 → ratio ≈ 1.05
        item_a = features[(features["item_id"] == "A") & (features["startdate"] == features["startdate"].max())]
        assert item_a.iloc[0]["true_demand_ratio"] >= 1.0

    def test_demand_sales_gap_positive_with_oos(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        assert item_b.iloc[-1]["demand_sales_gap_3m"] > 0


class TestChannelMixFeatures:
    def test_channel_entropy_nonzero_multiple_channels(self, sample_demand):
        features = compute_features(sample_demand)
        item_a = features[(features["item_id"] == "A") & (features["startdate"] == features["startdate"].max())]
        # Item A has On Premise + Off Premise → entropy > 0
        assert item_a.iloc[0]["channel_entropy"] > 0

    def test_single_channel_entropy_zero(self, sample_demand):
        features = compute_features(sample_demand)
        item_b = features[features["item_id"] == "B"]
        # Item B has only On Premise → entropy = 0
        assert item_b.iloc[-1]["channel_entropy"] == pytest.approx(0.0, abs=0.01)


class TestEdgeCases:
    def test_empty_input(self):
        empty_df = pd.DataFrame(columns=["item_id", "loc", "startdate", "customer_no",
                                          "demand_qty", "sales_qty", "oos_qty", "channel"])
        result = compute_features(empty_df)
        assert result.empty

    def test_all_features_are_numeric(self, sample_demand):
        features = compute_features(sample_demand)
        numeric_cols = [c for c in features.columns if c not in ("item_id", "loc", "startdate")]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(features[col]), f"{col} is not numeric"

    def test_no_nan_in_output(self, sample_demand):
        features = compute_features(sample_demand)
        numeric_cols = [c for c in features.columns if c not in ("item_id", "loc", "startdate")]
        for col in numeric_cols:
            assert features[col].isna().sum() == 0, f"NaN found in {col}"
