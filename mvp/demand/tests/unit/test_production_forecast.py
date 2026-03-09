"""Unit tests for F1.1 production forecast generation pure functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to import functions under test
# ---------------------------------------------------------------------------

from scripts.generate_production_forecasts import (
    build_inference_grid,
    build_sales_index,
    build_attrs_index,
    build_cat_encoders,
    generate_forecast_recursive,
    generate_forecasts_batch,
)
from common.constants import LAG_RANGE, ROLLING_WINDOWS, CAT_FEATURES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sales(item_no="ITEM001", loc="LOC1", n_months=24, start="2024-01-01") -> pd.DataFrame:
    """Synthetic sales history with n_months rows."""
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    return pd.DataFrame({
        "item_no": item_no,
        "loc": loc,
        "startdate": dates,
        "qty": np.random.default_rng(42).uniform(80, 120, n_months),
    })


def _make_dfu_attrs(item_no="ITEM001", loc="LOC1") -> pd.DataFrame:
    return pd.DataFrame([{
        "item_no": item_no,
        "loc": loc,
        "dmdgroup": "GROUP1",
        "ml_cluster": 2,
        "execution_lag": 1,
        "total_lt": 30,
        "brand": "BrandA",
        "region": "NORTH",
        "abc_vol": "A",
    }])


# ---------------------------------------------------------------------------
# build_inference_grid
# ---------------------------------------------------------------------------

def test_build_grid_returns_dataframe():
    """build_inference_grid returns a DataFrame with horizon rows."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=6)
    assert grid is not None
    assert len(grid) == 6


def test_build_grid_lag_features_present():
    """Grid contains all lag and rolling features."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    for lag in LAG_RANGE:
        assert f"qty_lag_{lag}" in grid.columns
    for w in ROLLING_WINDOWS:
        assert f"rolling_mean_{w}m" in grid.columns
        assert f"rolling_std_{w}m" in grid.columns


def test_build_grid_lag_source_metadata():
    """First row has lag_source='actual', subsequent rows have 'predicted'."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=4)
    assert grid.iloc[0]["_lag_source"] == "actual"
    for i in range(1, 4):
        assert grid.iloc[i]["_lag_source"] == "predicted"


def test_build_grid_insufficient_history_returns_none():
    """Returns None when fewer than max_lag months of history available."""
    sales = _make_sales(n_months=3)  # very short history
    attrs = _make_dfu_attrs()
    result = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=6)
    assert result is None


def test_build_grid_unknown_dfu_attrs():
    """Missing DFU attributes use __unknown__ for categorical cols."""
    sales = _make_sales(n_months=24)
    # Empty attrs — no matching row
    attrs = pd.DataFrame(columns=["item_no", "loc", "dmdgroup", "ml_cluster",
                                   "execution_lag", "total_lt", "brand", "region", "abc_vol"])
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    assert grid is not None
    assert grid.iloc[0]["ml_cluster"] == "__unknown__"


def test_build_grid_horizon_months_tracked():
    """_horizon metadata correctly increments from 1 to horizon."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=5)
    assert list(grid["_horizon"].values) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# generate_forecast_recursive
# ---------------------------------------------------------------------------

def _make_mock_model(pred_val=100.0):
    """Create a mock model that always predicts pred_val."""
    m = MagicMock()
    m.predict = MagicMock(side_effect=lambda X: np.full(len(X), pred_val))
    return m


def _make_grid(n=3) -> pd.DataFrame:
    """Minimal grid with required columns."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    return build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=n)


def test_generate_forecast_returns_rows():
    """generate_forecast_recursive returns one dict per horizon month."""
    grid = _make_grid(3)
    model = _make_mock_model(100.0)
    feature_cols = [c for c in grid.columns if not c.startswith("_")]

    rows = generate_forecast_recursive(
        model=model,
        feature_cols=feature_cols,
        grid=grid,
        horizon=3,
        item_no="ITEM001",
        loc="LOC1",
        plan_version="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cluster_id=2,
    )
    assert len(rows) == 3


def test_generate_forecast_keys():
    """Each row has expected keys."""
    grid = _make_grid(2)
    model = _make_mock_model(150.0)
    feature_cols = [c for c in grid.columns if not c.startswith("_")]

    rows = generate_forecast_recursive(
        model=model,
        feature_cols=feature_cols,
        grid=grid,
        horizon=2,
        item_no="ITEM001",
        loc="LOC1",
        plan_version="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cluster_id=2,
    )
    for row in rows:
        assert "item_no" in row
        assert "loc" in row
        assert "forecast_month" in row
        assert "forecast_qty" in row
        assert "lag_source" in row
        assert "plan_version" in row
        assert "model_id" in row
        assert "horizon_months" in row


def test_generate_forecast_qty_nonneg():
    """Forecast qty is clamped to non-negative values."""
    grid = _make_grid(3)
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.full(len(X), -50.0))  # negative preds
    feature_cols = [c for c in grid.columns if not c.startswith("_")]

    rows = generate_forecast_recursive(
        model=model,
        feature_cols=feature_cols,
        grid=grid,
        horizon=3,
        item_no="ITEM001",
        loc="LOC1",
        plan_version="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cluster_id=2,
    )
    for row in rows:
        assert row["forecast_qty"] >= 0.0


# ---------------------------------------------------------------------------
# build_sales_index
# ---------------------------------------------------------------------------

def test_build_sales_index_structure():
    """Returns dict keyed by (item_no, loc) with (dates, qty) tuples."""
    sales = _make_sales(n_months=6)
    idx = build_sales_index(sales)
    assert ("ITEM001", "LOC1") in idx
    dates, qty = idx[("ITEM001", "LOC1")]
    assert len(dates) == 6
    assert len(qty) == 6


def test_build_sales_index_sorted():
    """Qty values are sorted chronologically."""
    sales = _make_sales(n_months=6)
    sales = sales.sample(frac=1, random_state=0)  # shuffle
    idx = build_sales_index(sales)
    _, qty = idx[("ITEM001", "LOC1")]
    assert len(qty) == 6  # all rows present


def test_build_sales_index_multiple_dfus():
    """Handles multiple DFUs correctly."""
    s1 = _make_sales("ITEM001", "LOC1", n_months=6)
    s2 = _make_sales("ITEM002", "LOC2", n_months=4)
    idx = build_sales_index(pd.concat([s1, s2], ignore_index=True))
    assert ("ITEM001", "LOC1") in idx
    assert ("ITEM002", "LOC2") in idx
    assert len(idx[("ITEM002", "LOC2")][1]) == 4


# ---------------------------------------------------------------------------
# build_attrs_index
# ---------------------------------------------------------------------------

def test_build_attrs_index_structure():
    """Returns dict keyed by (item_no, loc) with attr dicts."""
    attrs = _make_dfu_attrs()
    idx = build_attrs_index(attrs)
    assert ("ITEM001", "LOC1") in idx
    assert idx[("ITEM001", "LOC1")]["brand"] == "BrandA"


def test_build_attrs_index_missing_key():
    """Missing DFU returns empty dict via .get()."""
    idx = build_attrs_index(_make_dfu_attrs())
    assert idx.get(("MISSING", "LOC99"), {}) == {}


# ---------------------------------------------------------------------------
# build_cat_encoders
# ---------------------------------------------------------------------------

def test_build_cat_encoders_returns_all_features():
    """Returns encoders for all CAT_FEATURES present in attrs."""
    attrs = _make_dfu_attrs()
    enc = build_cat_encoders(attrs)
    for col in CAT_FEATURES:
        if col in attrs.columns:
            assert col in enc


def test_build_cat_encoders_integer_codes():
    """Encoder values are non-negative integers."""
    attrs = _make_dfu_attrs()
    enc = build_cat_encoders(attrs)
    for col, mapping in enc.items():
        for val, code in mapping.items():
            assert isinstance(code, int)
            assert code >= 0


# ---------------------------------------------------------------------------
# build_inference_grid fast path (index-based)
# ---------------------------------------------------------------------------

def test_build_grid_fast_path():
    """Fast path using sales_index and attrs_index returns same shape as slow path."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    sales_index = build_sales_index(sales)
    attrs_index = build_attrs_index(attrs)

    grid = build_inference_grid(
        "ITEM001", "LOC1", 2,
        horizon=6,
        sales_index=sales_index,
        attrs_index=attrs_index,
    )
    assert grid is not None
    assert len(grid) == 6


def test_build_grid_fast_path_missing_dfu_returns_none():
    """Fast path returns None when DFU not in sales_index."""
    sales = _make_sales(n_months=24)
    sales_index = build_sales_index(sales)
    attrs_index = build_attrs_index(_make_dfu_attrs())

    result = build_inference_grid(
        "MISSING", "LOC99", 0,
        horizon=6,
        sales_index=sales_index,
        attrs_index=attrs_index,
    )
    assert result is None


# ---------------------------------------------------------------------------
# generate_forecasts_batch
# ---------------------------------------------------------------------------

def _make_artifact(pred_val: float = 100.0) -> dict:
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.full(len(X), pred_val))
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    feature_cols = [c for c in grid.columns if not c.startswith("_")]
    return {"model": model, "feature_cols": feature_cols}


def test_generate_forecasts_batch_returns_rows():
    """Returns horizon rows per DFU."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    artifact = _make_artifact(100.0)
    enc = build_cat_encoders(attrs)

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            ({"item_no": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid),
        ],
        horizon=3,
        plan_version="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )
    assert len(rows) == 3


def test_generate_forecasts_batch_multi_dfu():
    """Batch processes multiple DFUs in one call."""
    sales1 = _make_sales("ITEM001", "LOC1", n_months=24)
    sales2 = _make_sales("ITEM002", "LOC2", n_months=24)
    attrs1 = _make_dfu_attrs("ITEM001", "LOC1")
    attrs2 = _make_dfu_attrs("ITEM002", "LOC2")
    grid1 = build_inference_grid("ITEM001", "LOC1", 2, sales1, attrs1, horizon=3)
    grid2 = build_inference_grid("ITEM002", "LOC2", 2, sales2, attrs2, horizon=3)
    artifact = _make_artifact(50.0)
    enc = build_cat_encoders(pd.concat([attrs1, attrs2], ignore_index=True))

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            ({"item_no": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid1),
            ({"item_no": "ITEM002", "loc": "LOC2", "cluster_id": 2}, grid2),
        ],
        horizon=3,
        plan_version="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )
    assert len(rows) == 6  # 2 DFUs × 3 months


def test_generate_forecasts_batch_nonneg_qty():
    """Batch clamps negative predictions to 0."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.full(len(X), -999.0))
    artifact = {"model": model, "feature_cols": [c for c in grid.columns if not c.startswith("_")]}
    enc = build_cat_encoders(attrs)

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[({"item_no": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid)],
        horizon=2,
        plan_version="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )
    for row in rows:
        assert row["forecast_qty"] >= 0.0


# ---------------------------------------------------------------------------
# 18-month planning horizon
# ---------------------------------------------------------------------------

def test_build_grid_18_month_horizon():
    """Grid supports 18-month planning horizon."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=18)
    assert grid is not None
    assert len(grid) == 18
    assert list(grid["_horizon"].values) == list(range(1, 19))


def test_generate_forecasts_batch_18_months():
    """Batch inference produces 18 rows per DFU for planning horizon."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=18)
    artifact = _make_artifact(120.0)
    enc = build_cat_encoders(attrs)

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[({"item_no": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid)],
        horizon=18,
        plan_version="2026-02",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )
    assert len(rows) == 18
    assert rows[0]["lag_source"] == "actual"
    assert rows[0]["is_recursive"] is False
    assert rows[1]["lag_source"] == "predicted"
    assert rows[1]["is_recursive"] is True
