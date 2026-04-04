"""Tests for bolt hierarchical backtest helper functions."""

import pytest
import numpy as np
import pandas as pd

from scripts.run_backtest_bolt_hierarchical import (
    _build_customer_series,
    _build_agg_series,
    _reconcile,
    _map_to_dfu_grain,
)


@pytest.fixture
def demand_df():
    """Sample customer demand data."""
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    for m in months:
        rows.append({"item_id": "A", "customer_no": "C1", "loc": "L1", "startdate": m, "qty": 100})
        rows.append({"item_id": "A", "customer_no": "C2", "loc": "L1", "startdate": m, "qty": 50})
        rows.append({"item_id": "B", "customer_no": "C1", "loc": "L1", "startdate": m, "qty": 200})
    return pd.DataFrame(rows)


@pytest.fixture
def dfu_map():
    """DFU grain mapping."""
    return pd.DataFrame([
        {"sku_ck": "A_ALL_L1", "item_id": "A", "customer_group": "ALL", "loc": "L1", "execution_lag": 0},
        {"sku_ck": "B_ALL_L1", "item_id": "B", "customer_group": "ALL", "loc": "L1", "execution_lag": 0},
    ])


@pytest.fixture
def sales_df():
    """Sales actuals for DFU mapping."""
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    for m in months:
        rows.append({"sku_ck": "A_ALL_L1", "item_id": "A", "customer_group": "ALL", "loc": "L1", "startdate": m, "qty": 140})
        rows.append({"sku_ck": "B_ALL_L1", "item_id": "B", "customer_group": "ALL", "loc": "L1", "startdate": m, "qty": 180})
    return pd.DataFrame(rows)


class TestBuildCustomerSeries:
    def test_builds_customer_level_keys(self, demand_df):
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"))
        assert "sku_ck" in result.columns
        # Should have keys like A__C1__L1, A__C2__L1, B__C1__L1
        keys = result["sku_ck"].unique()
        assert len(keys) == 3
        assert "A__C1__L1" in keys
        assert "A__C2__L1" in keys
        assert "B__C1__L1" in keys

    def test_filters_short_series(self, demand_df):
        # With min_nonzero_months=10 (higher than data range), should filter everything
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"), min_nonzero_months=10)
        assert result.empty

    def test_respects_train_end(self, demand_df):
        result = _build_customer_series(demand_df, pd.Timestamp("2025-03-01"))
        # Only months through March 2025
        assert result["startdate"].max() <= pd.Timestamp("2025-03-01")

    def test_caps_customers(self, demand_df):
        # Cap to 1 customer per item-loc
        result = _build_customer_series(demand_df, pd.Timestamp("2025-06-01"), max_customers_per_item_loc=1)
        # Item A should only keep top-1 customer (C1 with 100 > C2 with 50)
        item_a_keys = [k for k in result["sku_ck"].unique() if k.startswith("A__")]
        assert len(item_a_keys) == 1
        assert "A__C1__L1" in item_a_keys


class TestBuildAggSeries:
    def test_aggregates_to_item_loc(self, demand_df):
        result = _build_agg_series(demand_df, pd.Timestamp("2025-06-01"))
        keys = result["sku_ck"].unique()
        assert len(keys) == 2
        assert "A__AGG__L1" in keys
        assert "B__AGG__L1" in keys

    def test_sums_demand(self, demand_df):
        result = _build_agg_series(demand_df, pd.Timestamp("2025-06-01"))
        # Item A has C1(100) + C2(50) = 150 per month
        a_jan = result[(result["sku_ck"] == "A__AGG__L1") & (result["startdate"] == pd.Timestamp("2025-01-01"))]
        assert a_jan.iloc[0]["qty"] == 150


class TestReconcile:
    def test_weighted_average(self):
        bu = pd.DataFrame({
            "item_id": ["A", "A"], "loc": ["L1", "L1"],
            "startdate": [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")],
            "basefcst_pref": [160, 170],
        })
        td = pd.DataFrame({
            "item_id": ["A", "A"], "loc": ["L1", "L1"],
            "startdate": [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")],
            "basefcst_pref": [140, 150],
        })
        result = _reconcile(bu, td, bu_weight=0.6)
        # 0.6 * 160 + 0.4 * 140 = 96 + 56 = 152
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(152, abs=0.1)
        # 0.6 * 170 + 0.4 * 150 = 102 + 60 = 162
        assert result.iloc[1]["basefcst_pref"] == pytest.approx(162, abs=0.1)

    def test_bu_only_when_td_missing(self):
        bu = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [100],
        })
        td = pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])
        result = _reconcile(bu, td, bu_weight=0.6)
        # BU=100, TD=0 → 0.6*100 + 0.4*0 = 60
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(60, abs=0.1)

    def test_non_negative(self):
        bu = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [0],
        })
        td = pd.DataFrame({
            "item_id": ["A"], "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [0],
        })
        result = _reconcile(bu, td)
        assert result.iloc[0]["basefcst_pref"] >= 0


class TestMapToDfuGrain:
    def test_maps_to_sku_ck(self, dfu_map, sales_df):
        reconciled = pd.DataFrame({
            "item_id": ["A", "B"],
            "loc": ["L1", "L1"],
            "startdate": [pd.Timestamp("2025-07-01")] * 2,
            "basefcst_pref": [150, 200],
            "bu_fcst": [160, 210],
            "td_fcst": [140, 190],
        })
        result = _map_to_dfu_grain(reconciled, dfu_map, sales_df)
        assert "sku_ck" in result.columns
        assert set(result["sku_ck"]) == {"A_ALL_L1", "B_ALL_L1"}

    def test_preserves_forecast_values(self, dfu_map, sales_df):
        reconciled = pd.DataFrame({
            "item_id": ["A"],
            "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-07-01")],
            "basefcst_pref": [150],
            "bu_fcst": [160],
            "td_fcst": [140],
        })
        result = _map_to_dfu_grain(reconciled, dfu_map, sales_df)
        assert result.iloc[0]["basefcst_pref"] == 150

    def test_empty_reconciled(self, dfu_map, sales_df):
        reconciled = pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref", "bu_fcst", "td_fcst"])
        result = _map_to_dfu_grain(reconciled, dfu_map, sales_df)
        assert result.empty
