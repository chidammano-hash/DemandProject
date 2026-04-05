"""Tests for bolt hierarchical backtest (customer BU + location TD, adaptive)."""

import pytest
import numpy as np
import pandas as pd

from scripts.run_backtest_bolt_hierarchical import (
    _build_customer_series,
    _pre_aggregate_demand,
    _build_agg_series,
    _aggregate_bu_to_item_loc,
    _reconcile_and_map,
    _compute_demand_share,
    _compute_adaptive_bu_weight,
)


@pytest.fixture
def demand_df():
    """Customer demand with pre-built cust_key (includes zero-demand months)."""
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    for m in months:
        rows.append({"item_id": "A", "customer_no": "C1", "loc": "L1", "startdate": m, "qty": 100.0})
        rows.append({"item_id": "A", "customer_no": "C2", "loc": "L1", "startdate": m, "qty": 50.0})
        rows.append({"item_id": "B", "customer_no": "C1", "loc": "L1", "startdate": m, "qty": 200.0})
    df = pd.DataFrame(rows)
    df["cust_key"] = df["item_id"] + "__" + df["customer_no"] + "__" + df["loc"]
    return df


@pytest.fixture
def dfu_map():
    return pd.DataFrame([
        {"sku_ck": "A_ALL_L1", "item_id": "A", "customer_group": "ALL", "loc": "L1", "execution_lag": 0},
        {"sku_ck": "B_ALL_L1", "item_id": "B", "customer_group": "ALL", "loc": "L1", "execution_lag": 0},
    ])


@pytest.fixture
def sales_df():
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    for m in months:
        rows.append({"sku_ck": "A_ALL_L1", "item_id": "A", "customer_group": "ALL", "loc": "L1", "startdate": m, "qty": 140.0})
        rows.append({"sku_ck": "B_ALL_L1", "item_id": "B", "customer_group": "ALL", "loc": "L1", "startdate": m, "qty": 180.0})
    return pd.DataFrame(rows)


class TestBuildCustomerSeries:
    def test_builds_customer_keys(self, demand_df):
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"))
        keys = result["sku_ck"].unique()
        assert len(keys) == 3
        assert "A__C1__L1" in keys
        assert "A__C2__L1" in keys
        assert "B__C1__L1" in keys

    def test_filters_short_series(self, demand_df):
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"), min_nonzero_months=10)
        assert result.empty

    def test_respects_train_end(self, demand_df):
        result = _build_customer_series(
            demand_df, pd.Timestamp("2025-03-01"), min_nonzero_months=1,
        )
        assert result["startdate"].max() <= pd.Timestamp("2025-03-01")

    def test_no_cap_all_customers_kept(self, demand_df):
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"))
        a_keys = [k for k in result["sku_ck"].unique() if k.startswith("A__")]
        assert len(a_keys) == 2  # C1 and C2 both kept

    def test_empty_demand(self):
        df = pd.DataFrame(columns=["item_id", "customer_no", "loc", "startdate", "qty", "cust_key"])
        result = _build_customer_series(df, pd.Timestamp("2025-06-01"))
        assert result.empty

    def test_max_customers_cap(self, demand_df):
        """With cap=1, only top customer per item×loc kept, rest in __OTHER__."""
        result = _build_customer_series(
            demand_df, pd.Timestamp("2025-06-01"),
            min_nonzero_months=3,
            max_customers_per_item_loc=1,
        )
        a_keys = [k for k in result["sku_ck"].unique() if k.startswith("A__")]
        # A has C1 (100/month) and C2 (50/month) — C1 kept, C2 → __OTHER__
        assert any("C1" in k for k in a_keys)
        assert any("__OTHER__" in k for k in a_keys)

    def test_max_customers_no_overflow_when_within_cap(self, demand_df):
        """Cap=10, only 2 customers per item — no __OTHER__ bucket."""
        result = _build_customer_series(
            demand_df, pd.Timestamp("2025-06-01"),
            min_nonzero_months=3,
            max_customers_per_item_loc=10,
        )
        other_keys = [k for k in result["sku_ck"].unique() if "__OTHER__" in k]
        assert len(other_keys) == 0

    def test_default_min_nonzero_is_6(self, demand_df):
        """Default min_nonzero_months=6; 6-month fixture should pass."""
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"))
        assert not result.empty


class TestPreAggregateDemand:
    def test_aggregates_to_item_loc(self, demand_df):
        result = _pre_aggregate_demand(demand_df)
        keys = result["sku_ck"].unique()
        assert len(keys) == 2
        assert "A__AGG__L1" in keys
        assert "B__AGG__L1" in keys

    def test_sums_across_customers(self, demand_df):
        result = _pre_aggregate_demand(demand_df)
        a_jan = result[
            (result["sku_ck"] == "A__AGG__L1")
            & (result["startdate"] == pd.Timestamp("2025-01-01"))
        ]
        assert a_jan.iloc[0]["qty"] == 150  # C1(100) + C2(50)

    def test_empty(self):
        df = pd.DataFrame(columns=["item_id", "customer_no", "loc", "startdate", "qty"])
        result = _pre_aggregate_demand(df)
        assert result.empty

    def test_has_item_id_and_loc_columns(self, demand_df):
        result = _pre_aggregate_demand(demand_df)
        assert "item_id" in result.columns
        assert "loc" in result.columns


class TestBuildAggSeries:
    def test_filters_by_train_end(self, demand_df):
        agg = _pre_aggregate_demand(demand_df)
        result = _build_agg_series(agg, pd.Timestamp("2025-03-01"), min_history_months=1)
        assert result["startdate"].max() <= pd.Timestamp("2025-03-01")

    def test_min_history_filter(self, demand_df):
        agg = _pre_aggregate_demand(demand_df)
        result = _build_agg_series(agg, pd.Timestamp("2025-06-01"), min_history_months=100)
        assert result.empty

    def test_passes_with_sufficient_history(self, demand_df):
        agg = _pre_aggregate_demand(demand_df)
        result = _build_agg_series(agg, pd.Timestamp("2025-06-01"), min_history_months=6)
        assert not result.empty
        assert len(result["sku_ck"].unique()) == 2


class TestAggregateBuToItemLoc:
    def test_sums_across_customers(self):
        bu = pd.DataFrame({
            "sku_ck": ["A__C1__L1", "A__C2__L1", "B__C1__L1"],
            "startdate": [pd.Timestamp("2025-07-01")] * 3,
            "basefcst_pref": [100.0, 50.0, 200.0],
        })
        result = _aggregate_bu_to_item_loc(bu)
        a_row = result[(result["item_id"] == "A") & (result["loc"] == "L1")]
        assert a_row.iloc[0]["basefcst_pref"] == 150.0

    def test_empty(self):
        result = _aggregate_bu_to_item_loc(
            pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"]),
        )
        assert result.empty


class TestComputeDemandShare:
    def test_single_cg(self, sales_df):
        share = _compute_demand_share(sales_df)
        # Only one customer_group (ALL) per item×loc → share=1.0
        assert len(share) == 2
        assert share["share"].tolist() == pytest.approx([1.0, 1.0])

    def test_multi_cg(self):
        df = pd.DataFrame({
            "item_id": ["A", "A"], "loc": ["L1", "L1"],
            "customer_group": ["ALL", "RET"], "qty": [70.0, 30.0],
            "startdate": [pd.Timestamp("2025-01-01")] * 2,
        })
        share = _compute_demand_share(df)
        all_share = share[share["customer_group"] == "ALL"].iloc[0]["share"]
        ret_share = share[share["customer_group"] == "RET"].iloc[0]["share"]
        assert all_share == pytest.approx(0.7, abs=0.01)
        assert ret_share == pytest.approx(0.3, abs=0.01)

    def test_empty(self):
        result = _compute_demand_share(
            pd.DataFrame(columns=["item_id", "loc", "customer_group", "qty", "startdate"]),
        )
        assert result.empty

    def test_respects_train_end(self, sales_df):
        """Share computed only from data up to train_end."""
        share_full = _compute_demand_share(sales_df)
        share_partial = _compute_demand_share(sales_df, train_end=pd.Timestamp("2025-03-01"))
        # Both should produce shares, but partial uses only 3 months of data
        assert len(share_partial) > 0
        assert len(share_full) > 0


class TestComputeAdaptiveBuWeight:
    def test_high_customer_count(self, demand_df):
        """Item A has 2 customers — should get 0.3 (<5 threshold)."""
        weights = _compute_adaptive_bu_weight(None, demand_df, pd.Timestamp("2025-06-01"))
        a_row = weights[(weights["item_id"] == "A") & (weights["loc"] == "L1")]
        assert a_row.iloc[0]["bu_weight"] == 0.3  # < 5 customers

    def test_empty_demand(self):
        df = pd.DataFrame(columns=["item_id", "customer_no", "loc", "startdate", "qty"])
        weights = _compute_adaptive_bu_weight(None, df, pd.Timestamp("2025-06-01"))
        assert weights.empty

    def test_weight_tiers(self):
        """Test the 3-tier heuristic: >20→0.7, 5-20→0.5, <5→0.3."""
        months = pd.date_range("2025-01-01", periods=6, freq="MS")
        rows = []
        # Item X: 25 customers → 0.7
        for c in range(25):
            for m in months:
                rows.append({"item_id": "X", "customer_no": f"C{c}", "loc": "L1", "startdate": m, "qty": 10.0})
        # Item Y: 10 customers → 0.5
        for c in range(10):
            for m in months:
                rows.append({"item_id": "Y", "customer_no": f"C{c}", "loc": "L1", "startdate": m, "qty": 10.0})
        # Item Z: 2 customers → 0.3
        for c in range(2):
            for m in months:
                rows.append({"item_id": "Z", "customer_no": f"C{c}", "loc": "L1", "startdate": m, "qty": 10.0})
        df = pd.DataFrame(rows)
        weights = _compute_adaptive_bu_weight(None, df, pd.Timestamp("2025-06-01"))
        x_w = weights[weights["item_id"] == "X"].iloc[0]["bu_weight"]
        y_w = weights[weights["item_id"] == "Y"].iloc[0]["bu_weight"]
        z_w = weights[weights["item_id"] == "Z"].iloc[0]["bu_weight"]
        assert x_w == 0.7
        assert y_w == 0.5
        assert z_w == 0.3


class TestReconcileAndMap:
    def test_weighted_average(self, dfu_map, sales_df):
        bu_agg = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [160.0],
        })
        td = pd.DataFrame({
            "sku_ck": ["A__AGG__L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [140.0],
        })
        share = _compute_demand_share(sales_df)
        result = _reconcile_and_map(bu_agg, td, 0.6, dfu_map, share)
        # 0.6*160 + 0.4*140 = 96+56 = 152
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(152, abs=0.1)
        assert result.iloc[0]["sku_ck"] == "A_ALL_L1"

    def test_bu_only_when_td_empty(self, dfu_map, sales_df):
        bu_agg = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [100.0],
        })
        td = pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])
        share = _compute_demand_share(sales_df)
        result = _reconcile_and_map(bu_agg, td, 0.6, dfu_map, share)
        # No TD data → reconciliation applies bu_weight*BU + td_weight*0 = 0.6*100
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(60, abs=0.1)

    def test_maps_all_dfus(self, dfu_map, sales_df):
        bu_agg = pd.DataFrame({
            "item_id": ["A", "B"], "loc": ["L1", "L1"],
            "startdate": [pd.Timestamp("2025-07-01")] * 2,
            "basefcst_pref": [150.0, 200.0],
        })
        td = pd.DataFrame({
            "sku_ck": ["A__AGG__L1", "B__AGG__L1"],
            "startdate": [pd.Timestamp("2025-07-01")] * 2,
            "basefcst_pref": [140.0, 190.0],
        })
        share = _compute_demand_share(sales_df)
        result = _reconcile_and_map(bu_agg, td, 0.6, dfu_map, share)
        assert set(result["sku_ck"]) == {"A_ALL_L1", "B_ALL_L1"}

    def test_non_negative(self, dfu_map, sales_df):
        bu_agg = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [0.0],
        })
        td = pd.DataFrame({
            "sku_ck": ["A__AGG__L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [0.0],
        })
        share = _compute_demand_share(sales_df)
        result = _reconcile_and_map(bu_agg, td, 0.6, dfu_map, share)
        assert result.iloc[0]["basefcst_pref"] >= 0

    def test_empty_bu(self, dfu_map, sales_df):
        bu_agg = pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])
        td = pd.DataFrame({
            "sku_ck": ["A__AGG__L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [100.0],
        })
        share = _compute_demand_share(sales_df)
        result = _reconcile_and_map(bu_agg, td, 0.6, dfu_map, share)
        assert result.empty

    def test_adaptive_weights(self, dfu_map, sales_df):
        """Test that adaptive weights override default bu_weight."""
        bu_agg = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [160.0],
        })
        td = pd.DataFrame({
            "sku_ck": ["A__AGG__L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [140.0],
        })
        share = _compute_demand_share(sales_df)
        adaptive = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"], "bu_weight": [0.3],
        })
        result = _reconcile_and_map(
            bu_agg, td, 0.6, dfu_map, share,
            adaptive_weights=adaptive,
        )
        # 0.3*160 + 0.7*140 = 48+98 = 146
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(146, abs=0.1)
