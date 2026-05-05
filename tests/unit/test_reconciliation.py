"""Tests for hierarchical forecast reconciliation."""

import pytest
import numpy as np
import pandas as pd

from scripts.algorithm_testing.reconciliation import reconcile_two_level


@pytest.fixture
def bu_preds():
    return pd.DataFrame({
        "item_id": ["A", "A"],
        "loc": ["L1", "L1"],
        "startdate": [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")],
        "basefcst_pref": [160.0, 170.0],
    })


@pytest.fixture
def td_preds():
    return pd.DataFrame({
        "item_id": ["A", "A"],
        "loc": ["L1", "L1"],
        "startdate": [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")],
        "basefcst_pref": [140.0, 150.0],
    })


@pytest.fixture
def actuals():
    return pd.DataFrame({
        "item_id": ["A"] * 6,
        "loc": ["L1"] * 6,
        "startdate": pd.date_range("2025-01-01", periods=6, freq="MS"),
        "qty": [100, 120, 110, 130, 140, 150],
    })


class TestWeightedAverage:
    def test_basic_blend(self, bu_preds, td_preds):
        result = reconcile_two_level(bu_preds, td_preds, method="weighted_average", bu_weight=0.6)
        # 0.6 * 160 + 0.4 * 140 = 152
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(152, abs=0.1)

    def test_non_negative(self, bu_preds, td_preds):
        result = reconcile_two_level(bu_preds, td_preds, method="weighted_average")
        assert (result["basefcst_pref"] >= 0).all()

    def test_has_bu_td_columns(self, bu_preds, td_preds):
        result = reconcile_two_level(bu_preds, td_preds, method="weighted_average")
        assert "bu_fcst" in result.columns
        assert "td_fcst" in result.columns

    def test_empty_td(self, bu_preds):
        td_empty = pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])
        result = reconcile_two_level(bu_preds, td_empty, method="weighted_average", bu_weight=0.6)
        # BU=160, TD=0 → 0.6*160 = 96
        assert result.iloc[0]["basefcst_pref"] == pytest.approx(96, abs=0.1)


class TestMinTrace:
    def test_mint_produces_output(self, bu_preds, td_preds, actuals):
        result = reconcile_two_level(bu_preds, td_preds, actuals, method="mint_shrink")
        assert not result.empty
        assert "basefcst_pref" in result.columns

    def test_mint_non_negative(self, bu_preds, td_preds, actuals):
        result = reconcile_two_level(bu_preds, td_preds, actuals, method="mint_shrink")
        assert (result["basefcst_pref"] >= 0).all()

    def test_mint_blends_between_bu_and_td(self, bu_preds, td_preds, actuals):
        result = reconcile_two_level(bu_preds, td_preds, actuals, method="mint_shrink")
        # Result should be between BU and TD (or close)
        for _, row in result.iterrows():
            lo = min(row["bu_fcst"], row["td_fcst"])
            hi = max(row["bu_fcst"], row["td_fcst"])
            # Allow 20% tolerance for shrinkage overshoot
            assert row["basefcst_pref"] >= lo * 0.8
            assert row["basefcst_pref"] <= hi * 1.2

    def test_mint_fallback_without_actuals(self, bu_preds, td_preds):
        # Without actuals, should fall back to weighted_average
        result = reconcile_two_level(bu_preds, td_preds, None, method="mint_shrink", bu_weight=0.5)
        assert not result.empty
