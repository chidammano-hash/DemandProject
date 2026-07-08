"""Unit tests for F1.1 production forecast generation pure functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.core.constants import CAT_FEATURES, LAG_RANGE, ROLLING_WINDOWS
from common.ml.seasonal_naive import SeasonalNaiveModel, make_dfu_key

# ---------------------------------------------------------------------------
# Helpers to import functions under test
# ---------------------------------------------------------------------------
from scripts.forecasting.generate_production_forecasts import (
    _is_model_fallback_substitution,
    _missing_required_tree_model_ids,
    _required_tree_model_ids,
    build_attrs_index,
    build_cat_encoders,
    build_inference_grid,
    build_month_routing,
    build_sales_index,
    collapse_to_dfu,
    filter_rows_to_champion_months,
    generate_forecast_recursive,
    generate_forecasts_batch,
    generate_forecasts_statistical,
    get_champion_assignments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sales(item_id="ITEM001", loc="LOC1", n_months=24, start="2024-01-01") -> pd.DataFrame:
    """Synthetic sales history with n_months rows."""
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    return pd.DataFrame({
        "item_id": item_id,
        "loc": loc,
        "startdate": dates,
        "qty": np.random.default_rng(42).uniform(80, 120, n_months),
    })


def _make_dfu_attrs(item_id="ITEM001", loc="LOC1") -> pd.DataFrame:
    return pd.DataFrame([{
        "item_id": item_id,
        "loc": loc,
        "customer_group": "GROUP1",
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
    """Returns None when fewer than min_months of history available."""
    sales = _make_sales(n_months=2)  # below cold_start_min_months default of 3
    attrs = _make_dfu_attrs()
    result = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=6)
    assert result is None


def test_build_grid_cold_start_short_history_ok():
    """DFUs with min_months history (cold-start range) return a valid grid."""
    sales = _make_sales(n_months=5)  # 5 months: above min_months=3, below old max_lag=12
    attrs = _make_dfu_attrs()
    result = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=6, min_months=3)
    assert result is not None
    assert len(result) == 6


def test_build_grid_unknown_dfu_attrs():
    """Missing DFU attributes use __unknown__ for categorical cols."""
    sales = _make_sales(n_months=24)
    # Empty attrs — no matching row
    attrs = pd.DataFrame(columns=["item_id", "loc", "customer_group", "ml_cluster",
                                   "execution_lag", "total_lt", "brand", "region", "abc_vol"])
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    assert grid is not None
    # ml_cluster was removed from CAT_FEATURES (see docs/specs/01-foundation/08-known-gaps.md §1).
    # Validate the contract on the columns that ARE categorical features.
    from common.core.constants import CAT_FEATURES
    for col in CAT_FEATURES:
        assert grid.iloc[0][col] == "__unknown__", f"{col} should default to __unknown__"


def test_build_grid_horizon_months_tracked():
    """_horizon metadata correctly increments from 1 to horizon."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=5)
    assert list(grid["_horizon"].values) == [1, 2, 3, 4, 5]


def test_build_grid_calendar_features_present():
    """Grid contains is_quarter_end, is_year_end, days_in_month."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    assert "is_quarter_end" in grid.columns
    assert "is_year_end" in grid.columns
    assert "days_in_month" in grid.columns


def test_build_grid_derived_features_present():
    """Grid contains mom_growth, demand_accel, volatility_ratio."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    assert "mom_growth" in grid.columns
    assert "demand_accel" in grid.columns
    assert "volatility_ratio" in grid.columns


def test_build_grid_is_quarter_end_values():
    """is_quarter_end is 1 for months 3,6,9,12 and 0 otherwise."""
    sales = _make_sales(n_months=24, start="2024-01-01")
    attrs = _make_dfu_attrs()
    # Start from 2024-01-01 with 24 months, last month is 2025-12-01
    # So T+1 = 2026-01-01, T+2 = 2026-02-01, T+3 = 2026-03-01
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    months = [grid.iloc[i]["_forecast_month"].month for i in range(3)]
    for i, m in enumerate(months):
        expected = 1 if m in (3, 6, 9, 12) else 0
        assert grid.iloc[i]["is_quarter_end"] == expected, f"Month {m}: expected {expected}"


def test_build_grid_mom_growth_bounded():
    """mom_growth is clipped to [-2, 2]."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    for i in range(3):
        assert -2.0 <= grid.iloc[i]["mom_growth"] <= 2.0


def test_build_grid_days_in_month_values():
    """days_in_month is a positive float (28-31)."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=6)
    for i in range(6):
        dim = grid.iloc[i]["days_in_month"]
        assert 28 <= dim <= 31


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
        item_id="ITEM001",
        loc="LOC1",
        forecast_month_generated="2026-03",
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
        item_id="ITEM001",
        loc="LOC1",
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cluster_id=2,
    )
    for row in rows:
        assert "item_id" in row
        assert "loc" in row
        assert "forecast_month" in row
        assert "forecast_qty" in row
        assert "lag_source" in row
        # Was renamed from plan_version → forecast_month_generated.
        assert "forecast_month_generated" in row
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
        item_id="ITEM001",
        loc="LOC1",
        forecast_month_generated="2026-03",
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
    """Returns dict keyed by (item_id, loc) with (dates, qty) tuples."""
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
    """Returns dict keyed by (item_id, loc) with attr dicts."""
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
            ({"item_id": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid),
        ],
        horizon=3,
        forecast_month_generated="2026-03",
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
            ({"item_id": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid1),
            ({"item_id": "ITEM002", "loc": "LOC2", "cluster_id": 2}, grid2),
        ],
        horizon=3,
        forecast_month_generated="2026-03",
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
        dfu_list=[({"item_id": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid)],
        horizon=2,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )
    for row in rows:
        assert row["forecast_qty"] >= 0.0


def test_generate_forecasts_batch_sets_seasonal_naive_context():
    """Seasonal-naive artifacts need DFU/month context outside the feature matrix."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    model = SeasonalNaiveModel(
        {
            (make_dfu_key("ITEM001", "GROUP1", "LOC1"), grid.iloc[0]["_forecast_month"].month): 17.0,
            (make_dfu_key("ITEM001", "GROUP1", "LOC1"), grid.iloc[1]["_forecast_month"].month): 23.0,
        },
        {make_dfu_key("ITEM001", "GROUP1", "LOC1"): 11.0},
    )
    artifact = {"model": model, "feature_cols": [c for c in grid.columns if not c.startswith("_")]}
    enc = build_cat_encoders(attrs)

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            ({
                "item_id": "ITEM001",
                "customer_group": "GROUP1",
                "loc": "LOC1",
                "cluster_id": 2,
            }, grid),
        ],
        horizon=2,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )

    assert [row["forecast_qty"] for row in rows] == [17.0, 23.0]


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
        dfu_list=[({"item_id": "ITEM001", "loc": "LOC1", "cluster_id": 2}, grid)],
        horizon=18,
        forecast_month_generated="2026-02",
        run_id="test-run-id",
        model_id="lgbm_cluster",
        cat_encoders=enc,
    )
    assert len(rows) == 18
    assert rows[0]["lag_source"] == "actual"
    assert rows[0]["is_recursive"] is False
    assert rows[1]["lag_source"] == "predicted"
    assert rows[1]["is_recursive"] is True


# ---------------------------------------------------------------------------
# Planning date compliance
# ---------------------------------------------------------------------------

def test_load_recent_sales_uses_planning_date():
    """load_recent_sales must use get_planning_date(), not Timestamp.now()."""
    import inspect

    from scripts.forecasting.generate_production_forecasts import load_recent_sales
    source = inspect.getsource(load_recent_sales)
    assert "get_planning_date" in source, "load_recent_sales must use get_planning_date()"
    assert "Timestamp.now()" not in source, "load_recent_sales must NOT use Timestamp.now()"


def test_main_plan_version_uses_planning_date():
    """plan_version generation must use get_planning_date(), not datetime.now()."""
    import inspect

    from scripts.forecasting.generate_production_forecasts import main
    source = inspect.getsource(main)
    assert "get_planning_date" in source, "plan_version must use get_planning_date()"


# ---------------------------------------------------------------------------
# load_config — consolidated config migration
# ---------------------------------------------------------------------------

def test_load_config_reads_pipeline_config():
    """load_config reads from forecast_pipeline_config.yaml and maps fields correctly."""
    from scripts.forecasting.generate_production_forecasts import load_config
    config = load_config()
    assert config["inference"]["horizon_months"] == 24
    assert config["model_selection"]["fallback_model_id"] == "lgbm_cluster"
    assert config["plan_version"]["keep_last_n_versions"] == 3
    assert config["_pipeline"]["lookback_months"] == 36
    assert config["_pipeline"]["min_history_months"] == 12
    assert config["_pipeline"]["cold_start_model_id"] == "rolling_mean"
    assert config["_pipeline"]["cold_start_min_months"] == 3


def test_load_config_ci_section():
    """load_config preserves confidence_interval settings from pipeline config."""
    from scripts.forecasting.generate_production_forecasts import load_config
    config = load_config()
    ci = config["confidence_interval"]
    assert ci["enabled"] is True
    for model_id in (
        "lgbm_cluster",
        "catboost_cluster",
        "xgboost_cluster",
        "lgbm_cust_enriched",
        "catboost_cust_enriched",
        "xgboost_cust_enriched",
    ):
        assert model_id in ci["source_model_ids"]
    assert ci["z_lower"] == 1.282


# ---------------------------------------------------------------------------
# Feature alignment: inference grid matches backtest feature matrix
# ---------------------------------------------------------------------------

def test_inference_grid_feature_parity_with_backtest():
    """build_inference_grid must produce the same features as build_feature_matrix."""
    from common.core.constants import CALENDAR_FEATURES, DERIVED_FEATURES
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    grid_cols = set(grid.columns)
    # All calendar features from constants must be present
    for feat in CALENDAR_FEATURES:
        assert feat in grid_cols, f"Missing calendar feature: {feat}"
    # All derived features from constants must be present
    for feat in DERIVED_FEATURES:
        assert feat in grid_cols, f"Missing derived feature: {feat}"


# ---------------------------------------------------------------------------
# Statistical / non-tree model inference
# ---------------------------------------------------------------------------


def _make_sales_index(item_id="ITEM001", loc="LOC1", n_months=24):
    """Build a sales_index dict for generate_forecasts_statistical tests."""
    dates = pd.date_range(start="2024-01-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(42)
    qty = rng.uniform(80, 120, n_months)
    return {(item_id, loc): (list(dates.values), list(qty))}


def _make_champion_df(item_id="ITEM001", loc="LOC1", cluster_id=2):
    """Build a minimal champion DataFrame for statistical tests."""
    return pd.DataFrame([{
        "item_id": item_id,
        "loc": loc,
        "cluster_id": cluster_id,
        "source_model_id": "rolling_mean",
    }])


def test_statistical_rolling_mean_returns_rows():
    """rolling_mean generates horizon rows per DFU."""
    sales_idx = _make_sales_index(n_months=24)
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="rolling_mean",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=6,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 6
    for row in rows:
        assert row["model_id"] == "rolling_mean"
        assert row["forecast_qty"] >= 0.0


def test_statistical_seasonal_naive_returns_rows():
    """seasonal_naive generates horizon rows per DFU."""
    sales_idx = _make_sales_index(n_months=24)
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="seasonal_naive",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=3,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 3
    for row in rows:
        assert row["model_id"] == "seasonal_naive"
        assert row["forecast_qty"] >= 0.0


def test_statistical_mstl_returns_rows():
    """mstl generates horizon rows with damped trend."""
    sales_idx = _make_sales_index(n_months=24)
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="mstl",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=3,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 3
    for row in rows:
        assert row["model_id"] == "mstl"
        assert row["forecast_qty"] >= 0.0


def test_statistical_rolling_median_flat_trailing_median():
    """rolling_median ships a FLAT forecast = median of the last 6 months, so a
    single spike month does NOT drag the forecast (the F-11 fix routes a
    rolling_median champion here instead of shipping the lgbm fallback)."""
    dates = pd.date_range(start="2024-01-01", periods=12, freq="MS")
    # A spike (200) in the trailing window must not move the median.
    qty = [10.0, 12.0, 11.0, 13.0, 9.0, 200.0, 14.0, 8.0, 15.0, 7.0, 16.0, 6.0]
    sales_idx = {("ITEM001", "LOC1"): (list(dates.values), qty)}
    champion_df = pd.DataFrame([{
        "item_id": "ITEM001", "loc": "LOC1", "cluster_id": 2,
        "source_model_id": "rolling_median",
    }])
    rows = generate_forecasts_statistical(
        model_id="rolling_median",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=4,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 4
    expected = round(float(np.median(qty[-6:])), 2)  # median of last 6 months, flat
    for row in rows:
        assert row["model_id"] == "rolling_median"
        assert row["forecast_qty"] == pytest.approx(expected)


def test_statistical_foundation_fallback():
    """Foundation model (chronos etc.) uses rolling mean fallback."""
    sales_idx = _make_sales_index(n_months=24)
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="chronos_bolt",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=3,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 3
    for row in rows:
        assert row["model_id"] == "chronos_bolt"
        assert row["forecast_qty"] >= 0.0


def test_statistical_skips_short_history():
    """DFUs with fewer than 3 months of history are skipped."""
    dates = pd.date_range(start="2024-01-01", periods=2, freq="MS")
    sales_idx = {("ITEM001", "LOC1"): (list(dates.values), [100.0, 110.0])}
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="rolling_mean",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=3,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 0


def test_statistical_skips_missing_dfu():
    """DFUs not found in sales_index are skipped."""
    sales_idx = {}  # empty — no sales data
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="rolling_mean",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=3,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 0


def test_statistical_output_keys():
    """Output rows have all expected keys for DB write."""
    sales_idx = _make_sales_index(n_months=24)
    champion_df = _make_champion_df()
    rows = generate_forecasts_statistical(
        model_id="rolling_mean",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=2,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    expected_keys = {
        "forecast_month_generated", "item_id", "loc", "forecast_month",
        "forecast_qty", "forecast_qty_lower", "forecast_qty_upper",
        "model_id", "cluster_id", "horizon_months", "is_recursive",
        "lag_source", "run_id", "generated_at",
    }
    assert len(rows) == 2
    assert set(rows[0].keys()) == expected_keys
    # First horizon step: lag_source=actual, not recursive
    assert rows[0]["lag_source"] == "actual"
    assert rows[0]["is_recursive"] is False
    # Second horizon step: lag_source=predicted, recursive
    assert rows[1]["lag_source"] == "predicted"
    assert rows[1]["is_recursive"] is True


def test_statistical_multi_dfu():
    """Statistical inference handles multiple DFUs correctly."""
    dates = pd.date_range(start="2024-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)
    sales_idx = {
        ("ITEM001", "LOC1"): (list(dates.values), list(rng.uniform(80, 120, 24))),
        ("ITEM002", "LOC2"): (list(dates.values), list(rng.uniform(50, 80, 24))),
    }
    champion_df = pd.DataFrame([
        {"item_id": "ITEM001", "loc": "LOC1", "cluster_id": 0, "source_model_id": "rolling_mean"},
        {"item_id": "ITEM002", "loc": "LOC2", "cluster_id": 1, "source_model_id": "rolling_mean"},
    ])
    rows = generate_forecasts_statistical(
        model_id="rolling_mean",
        sales_index=sales_idx,
        attrs_index={},
        champion_df=champion_df,
        horizon=3,
        forecast_month_generated="2026-03-01",
        run_id="test-run-id",
    )
    assert len(rows) == 6  # 2 DFUs x 3 months


# ---------------------------------------------------------------------------
# Gap-10 observability: champion-source → production-fallback substitution
#
# When a DFU's champion source_model_id is a STATISTICAL baseline (no .pkl),
# the tree-batch path silently substitutes the production fallback (lgbm_cluster)
# in _resolve_artifact. _is_model_fallback_substitution() detects exactly that
# case so the divergence is counted/logged. These tests pin both the detection
# predicate AND the unchanged routing return value.
# ---------------------------------------------------------------------------


def test_substitution_detected_when_source_model_absent():
    """Statistical champion source absent from loaded_models, tree fallback present
    → substitution detected (Gap-10 divergence)."""
    loaded_models = {"lgbm_cluster": {0: {"model": MagicMock()}}}
    assert _is_model_fallback_substitution(
        model_id="seasonal_naive",
        fallback_model_id="lgbm_cluster",
        loaded_models=loaded_models,
    ) is True


def test_no_substitution_when_requested_model_loaded():
    """Requested producer IS loaded → no substitution (genuine champion artifact)."""
    loaded_models = {
        "seasonal_naive": {0: {"model": MagicMock()}},
        "lgbm_cluster": {0: {"model": MagicMock()}},
    }
    assert _is_model_fallback_substitution(
        model_id="seasonal_naive",
        fallback_model_id="lgbm_cluster",
        loaded_models=loaded_models,
    ) is False


def test_no_substitution_when_requested_is_the_fallback():
    """Requested model_id IS the fallback itself → not a divergence even if absent."""
    assert _is_model_fallback_substitution(
        model_id="lgbm_cluster",
        fallback_model_id="lgbm_cluster",
        loaded_models={},
    ) is False


def test_no_substitution_when_fallback_also_absent():
    """Neither requested nor fallback loaded → DFU is skipped, not substituted."""
    loaded_models = {"catboost_cluster": {0: {"model": MagicMock()}}}
    assert _is_model_fallback_substitution(
        model_id="seasonal_naive",
        fallback_model_id="lgbm_cluster",
        loaded_models=loaded_models,
    ) is False


def test_required_tree_models_from_champion_routing_exclude_non_tree_models():
    """Only tree champion sources require .pkl artifacts; statistical models route separately."""
    champion_df = pd.DataFrame([
        {"source_model_id": "catboost_cluster"},
        {"source_model_id": "rolling_mean"},
        {"source_model_id": "xgboost_cust_enriched"},
    ])

    required = _required_tree_model_ids(
        requested_model_id=None,
        champion_month_df=champion_df,
        fallback_model_id="lgbm_cluster",
        non_tree_models={"rolling_mean", "seasonal_naive"},
    )

    assert required == {"catboost_cluster", "xgboost_cust_enriched"}


def test_required_tree_models_include_fallback_for_null_source_model():
    """NULL champion source falls back by config, so the fallback tree artifact is required."""
    champion_df = pd.DataFrame([
        {"source_model_id": None},
        {"source_model_id": "rolling_mean"},
    ])

    required = _required_tree_model_ids(
        requested_model_id=None,
        champion_month_df=champion_df,
        fallback_model_id="lgbm_cluster",
        non_tree_models={"rolling_mean", "seasonal_naive"},
    )

    assert required == {"lgbm_cluster"}


def test_required_tree_models_for_explicit_tree_model_override():
    champion_df = pd.DataFrame([{"source_model_id": "rolling_mean"}])

    required = _required_tree_model_ids(
        requested_model_id="catboost_cluster",
        champion_month_df=champion_df,
        fallback_model_id="lgbm_cluster",
        non_tree_models={"rolling_mean", "seasonal_naive"},
    )

    assert required == {"catboost_cluster"}


def test_required_tree_models_for_explicit_non_tree_model_override():
    champion_df = pd.DataFrame([{"source_model_id": "catboost_cluster"}])

    required = _required_tree_model_ids(
        requested_model_id="rolling_mean",
        champion_month_df=champion_df,
        fallback_model_id="lgbm_cluster",
        non_tree_models={"rolling_mean", "seasonal_naive"},
    )

    assert required == set()


def test_missing_required_tree_models_fail_before_fallback_substitution():
    """A winning tree model with no artifacts must fail instead of shipping fallback rows."""
    missing = _missing_required_tree_model_ids(
        {"catboost_cluster", "lgbm_cluster"},
        loaded_models={"lgbm_cluster": {0: {"model": MagicMock()}}},
    )

    assert missing == ["catboost_cluster"]


# ---------------------------------------------------------------------------
# F-11: non-tree champion sources must be GENERATED by their own model, not
# substituted with the lgbm_cluster fallback. load_non_tree_model_ids() drives
# the partition in main() that routes them to generate_forecasts_statistical.
# ---------------------------------------------------------------------------


def test_load_non_tree_model_ids_routes_statistical_baselines():
    """Statistical baselines (no .pkl) are classed non-tree → statistical generator."""
    from scripts.forecasting.generate_production_forecasts import load_non_tree_model_ids
    non_tree = load_non_tree_model_ids()
    for mid in ("seasonal_naive", "rolling_mean", "rolling_median", "mstl"):
        assert mid in non_tree, f"{mid} must route to the statistical generator (F-11)"


def test_load_non_tree_model_ids_excludes_tree_models():
    """Tree models keep the .pkl batch path — they must NOT be classed non-tree."""
    from scripts.forecasting.generate_production_forecasts import load_non_tree_model_ids
    non_tree = load_non_tree_model_ids()
    for mid in ("lgbm_cluster", "catboost_cluster", "xgboost_cluster"):
        assert mid not in non_tree


def test_cold_start_model_is_statistically_routed():
    """The cold-start model must be a non-tree source so cold-start DFUs are
    generated by their baseline, never the lgbm fallback."""
    from scripts.forecasting.generate_production_forecasts import (
        load_config,
        load_non_tree_model_ids,
    )
    cold = load_config()["_pipeline"]["cold_start_model_id"]
    assert cold in load_non_tree_model_ids()


# ---------------------------------------------------------------------------
# Data-integrity guard: dim_sku.ml_cluster wipe → tree forecast collapse.
# _cluster_assignments_wiped() must fail loud only for the catastrophic
# all-NULL-cluster + multi-cluster-tree signature.
# ---------------------------------------------------------------------------


def _champ_df(cluster_ids):
    return pd.DataFrame({
        "item_id": [f"I{i}" for i in range(len(cluster_ids))],
        "loc": ["L"] * len(cluster_ids),
        "source_model_id": ["lgbm_cluster"] * len(cluster_ids),
        "cluster_id": cluster_ids,
    })


def test_cluster_wipe_detected_all_null_with_multicluster_tree():
    """All cluster_id NULL + a multi-cluster tree loaded → wipe detected (abort)."""
    from scripts.forecasting.generate_production_forecasts import _cluster_assignments_wiped
    loaded = {"lgbm_cluster": {"high_volume_periodic": {}, "low_volume_periodic": {}}}
    assert _cluster_assignments_wiped(_champ_df([None, None]), loaded, {"seasonal_naive"}) is True


def test_cluster_wipe_not_flagged_when_clusters_present():
    """Populated cluster_id → not a wipe, even with multi-cluster trees loaded."""
    from scripts.forecasting.generate_production_forecasts import _cluster_assignments_wiped
    loaded = {"lgbm_cluster": {"high_volume_periodic": {}, "low_volume_periodic": {}}}
    assert _cluster_assignments_wiped(_champ_df(["high_volume_periodic", None]), loaded, set()) is False


def test_cluster_wipe_not_flagged_for_single_global_model():
    """A single (global) tree model → min() fallback is correct; NULL is expected,
    so clustering.enabled=false must NOT trip the guard."""
    from scripts.forecasting.generate_production_forecasts import _cluster_assignments_wiped
    loaded = {"lgbm_cluster": {"global": {}}}
    assert _cluster_assignments_wiped(_champ_df([None, None]), loaded, set()) is False


# ---------------------------------------------------------------------------
# Per-month champion routing (issue promote-per-month-collapse — generate side)
# ---------------------------------------------------------------------------


def _per_month_champ_df():
    """Champion frame for one DFU whose model varies by month."""
    return pd.DataFrame([
        {"item_id": "10031", "loc": "1401-BULK", "startdate": pd.Timestamp("2026-01-01"),
         "source_model_id": "seasonal_naive", "cluster_id": 2, "customer_group": "ALL"},
        {"item_id": "10031", "loc": "1401-BULK", "startdate": pd.Timestamp("2026-02-01"),
         "source_model_id": "rolling_mean", "cluster_id": 2, "customer_group": "ALL"},
    ])


def test_build_month_routing_keeps_per_month_model():
    """build_month_routing carries (item_id, loc) → {month: model_id} per month."""
    routing = build_month_routing(_per_month_champ_df())
    dfu = ("10031", "1401-BULK")
    assert routing[dfu][pd.Timestamp("2026-01-01")] == "seasonal_naive"
    assert routing[dfu][pd.Timestamp("2026-02-01")] == "rolling_mean"


def test_build_month_routing_skips_null_source():
    """NULL source_model_id rows are skipped (caller applies config fallback)."""
    df = pd.DataFrame([
        {"item_id": "A", "loc": "L", "startdate": pd.Timestamp("2026-01-01"),
         "source_model_id": None, "cluster_id": 1, "customer_group": "ALL"},
    ])
    routing = build_month_routing(df)
    assert pd.Timestamp("2026-01-01") not in routing.get(("A", "L"), {})


def test_collapse_to_dfu_one_row_per_dfu():
    """collapse_to_dfu reduces the per-month frame to one row per (item_id, loc)."""
    collapsed = collapse_to_dfu(_per_month_champ_df())
    assert len(collapsed) == 1
    # Earliest month's champion is retained deterministically (rows arrive ASC).
    assert collapsed.iloc[0]["source_model_id"] == "seasonal_naive"


def test_filter_rows_to_champion_months_keeps_winning_months_only():
    """Each model's rows survive only for the months it wins."""
    routing = build_month_routing(_per_month_champ_df())
    rows = [
        # seasonal_naive generated full horizon for this DFU
        {"item_id": "10031", "loc": "1401-BULK", "forecast_month": pd.Timestamp("2026-01-01"),
         "model_id": "seasonal_naive"},
        {"item_id": "10031", "loc": "1401-BULK", "forecast_month": pd.Timestamp("2026-02-01"),
         "model_id": "seasonal_naive"},
        # rolling_mean generated full horizon for this DFU
        {"item_id": "10031", "loc": "1401-BULK", "forecast_month": pd.Timestamp("2026-01-01"),
         "model_id": "rolling_mean"},
        {"item_id": "10031", "loc": "1401-BULK", "forecast_month": pd.Timestamp("2026-02-01"),
         "model_id": "rolling_mean"},
    ]
    kept = filter_rows_to_champion_months(rows, routing)
    by_month = {(r["forecast_month"], r["model_id"]) for r in kept}
    assert (pd.Timestamp("2026-01-01"), "seasonal_naive") in by_month
    assert (pd.Timestamp("2026-02-01"), "rolling_mean") in by_month
    # The losing rows are dropped — one model per (DFU, month).
    assert (pd.Timestamp("2026-01-01"), "rolling_mean") not in by_month
    assert (pd.Timestamp("2026-02-01"), "seasonal_naive") not in by_month
    assert len(kept) == 2


def test_filter_rows_no_winner_month_kept_from_lowest_model_only():
    """A month with no recorded champion is kept from ONE deterministic model.

    Avoids duplicate (DFU, month) staging rows when the DFU is generated under
    several models but a given month has no winner recorded.
    """
    # Only January routes (rolling_mean); February has NO recorded winner.
    routing = {("10031", "1401-BULK"): {pd.Timestamp("2026-01-01"): "rolling_mean"}}
    rows = [
        {"item_id": "10031", "loc": "1401-BULK", "forecast_month": pd.Timestamp("2026-02-01"),
         "model_id": "rolling_mean"},
        {"item_id": "10031", "loc": "1401-BULK", "forecast_month": pd.Timestamp("2026-02-01"),
         "model_id": "seasonal_naive"},
    ]
    kept = filter_rows_to_champion_months(rows, routing)
    # Lowest enqueued model for this DFU is rolling_mean (< seasonal_naive).
    assert len(kept) == 1
    assert kept[0]["model_id"] == "rolling_mean"


def test_filter_rows_passthrough_when_no_routing():
    """A DFU with no per-month routing (cold-start / override) is untouched."""
    rows = [
        {"item_id": "A", "loc": "L", "forecast_month": pd.Timestamp("2026-01-01"),
         "model_id": "rolling_mean"},
    ]
    assert filter_rows_to_champion_months(rows, {}) == rows


def test_get_champion_assignments_returns_per_month_rows():
    """get_champion_assignments keeps the startdate dimension (per-month rows)."""
    per_month = _per_month_champ_df()

    captured = {}

    def _fake_read_sql_chunked(conn, sql, params=None):
        captured["sql"] = sql
        return per_month.copy()

    with patch(
        "scripts.forecasting.generate_production_forecasts._get_promoted_champion_experiment_id",
        return_value=None,
    ), patch(
        "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
        side_effect=_fake_read_sql_chunked,
    ):
        df = get_champion_assignments(MagicMock())

    # Query must group per (item_id, loc, startdate), not per (item_id, loc).
    assert "DISTINCT ON (f.item_id, f.loc, f.startdate)" in captured["sql"]
    # Deterministic tie-break is source_model_id ASC (shared with promote side).
    assert "f.source_model_id ASC" in captured["sql"]
    assert "startdate" in df.columns
    # Two months for the one DFU survive.
    assert len(df) == 2
    assert set(df["source_model_id"]) == {"seasonal_naive", "rolling_mean"}


def test_get_champion_assignments_prefers_promoted_winners_csv(tmp_path):
    """Production generation uses the promoted experiment winners file.

    This keeps generate aligned with the promote endpoint: both treat
    experiment_<id>_winners.csv as the champion routing source of truth instead
    of stale fact_external_forecast_monthly rows.
    """
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    (champion_dir / "experiment_53_winners.csv").write_text(
        "item_id,customer_group,loc,model_id,startdate\n"
        "10031,ALL,1401-BULK,seasonal_naive,2026-01-01\n"
        "10031,ALL,1401-BULK,rolling_mean,2026-02-01\n"
        # Duplicate same DFU-month resolves lexically by model_id, matching promote.
        "10031,ALL,1401-BULK,xgboost_cluster,2026-02-01\n"
    )
    attrs = pd.DataFrame([
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "ml_cluster": 7,
            "execution_lag": 1,
            "total_lt": 14,
            "brand": "BrandA",
            "region": "NORTH",
            "abc_vol": "A",
        }
    ])

    with patch(
        "scripts.forecasting.generate_production_forecasts._get_promoted_champion_experiment_id",
        return_value=53,
    ), patch(
        "scripts.forecasting.generate_production_forecasts.CHAMPION_WINNERS_DIR",
        champion_dir,
    ), patch(
        "scripts.forecasting.generate_production_forecasts.load_dfu_attrs",
        return_value=attrs,
    ), patch(
        "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
    ) as legacy_read:
        df = get_champion_assignments(MagicMock())

    legacy_read.assert_not_called()
    assert len(df) == 2
    assert list(df["source_model_id"]) == ["seasonal_naive", "rolling_mean"]
    assert set(df["cluster_id"]) == {7}
    assert set(df["customer_group"]) == {"ALL"}
