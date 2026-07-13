"""Unit tests for F1.1 production forecast generation pure functions."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.core.constants import CAT_FEATURES, LAG_RANGE, ROLLING_WINDOWS, TS_PROFILE_FEATURES
from common.ml.feature_engineering import build_feature_matrix, compute_ts_profile_from_values

# ---------------------------------------------------------------------------
# Helpers to import functions under test
# ---------------------------------------------------------------------------
from scripts.forecasting.generate_production_forecasts import (
    _align_routes_to_population,
    _missing_required_tree_model_ids,
    _parse_ensemble_mix,
    _population_min_history,
    _resolve_champion_route,
    _to_cluster_id,
    aggregate_customer_group_forecasts,
    attach_aggregate_confidence_intervals,
    build_attrs_index,
    build_ensemble_routing,
    build_inference_grid,
    build_item_location_cluster_map,
    build_month_routing,
    build_sales_index,
    collapse_to_dfu,
    filter_rows_to_champion_months,
    generate_forecast_recursive,
    generate_forecasts_batch,
    get_champion_assignments,
    load_forecast_population,
    load_recent_sales,
    validate_customer_group_forecast_coverage,
    validate_customer_group_history,
    validate_route_history_requirements,
    write_forecast_staging,
)


def build_cat_encoders(dfu_attrs: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Test fixture encoder; production must use its persisted artifact map."""
    return {
        column: {
            value: index
            for index, value in enumerate(
                sorted(dfu_attrs[column].fillna("__unknown__").astype(str).unique())
            )
        }
        for column in CAT_FEATURES
        if column in dfu_attrs.columns
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sales(
    item_id="ITEM001",
    loc="LOC1",
    n_months=24,
    start="2024-01-01",
    customer_group="GROUP1",
) -> pd.DataFrame:
    """Synthetic sales history with n_months rows."""
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    return pd.DataFrame(
        {
            "item_id": item_id,
            "customer_group": customer_group,
            "loc": loc,
            "startdate": dates,
            "qty": np.random.default_rng(42).uniform(80, 120, n_months),
            "first_sale_month": pd.Timestamp(start),
        }
    )


def _make_dfu_attrs(item_id="ITEM001", loc="LOC1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": item_id,
                "loc": loc,
                "customer_group": "GROUP1",
                "ml_cluster": 2,
                "execution_lag": 1,
                "total_lt": 30,
                "brand": "BrandA",
                "region": "NORTH",
                "abc_vol": "A",
            }
        ]
    )


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


def test_build_grid_ts_profiles_match_training_feature_formulas():
    sales = _make_sales(n_months=24)
    sales["qty"] = np.tile([0.0, 10.0, 20.0, 5.0, 0.0, 15.0], 4)
    attrs = _make_dfu_attrs()

    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    expected = compute_ts_profile_from_values(sales["qty"].to_numpy())

    for feature_name in TS_PROFILE_FEATURES:
        assert grid.iloc[0][feature_name] == pytest.approx(expected[feature_name])


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
    attrs = pd.DataFrame(
        columns=[
            "item_id",
            "loc",
            "customer_group",
            "ml_cluster",
            "execution_lag",
            "total_lt",
            "brand",
            "region",
            "abc_vol",
        ]
    )
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    assert grid is not None
    # ml_cluster was removed from CAT_FEATURES to prevent cluster-label leakage.
    # Validate the contract on the columns that ARE categorical features.
    from common.core.constants import CAT_FEATURES

    for col in CAT_FEATURES:
        assert grid.iloc[0][col] == "__unknown__", f"{col} should default to __unknown__"


def test_build_grid_uses_training_zero_imputation_for_missing_total_lt():
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    attrs.loc[0, "total_lt"] = np.nan

    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)

    assert grid is not None
    assert set(grid["total_lt"]) == {0.0}


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
        customer_group="GROUP1",
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
        customer_group="GROUP1",
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
        customer_group="GROUP1",
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
    """Returns calendarized history plus an active-history length."""
    sales = _make_sales(n_months=6)
    idx = build_sales_index(sales)
    key = ("ITEM001", "GROUP1", "LOC1")
    assert key in idx
    dates, qty, active_length = idx[key]
    assert len(dates) == 6
    assert len(qty) == 6
    assert active_length == 6


def test_build_sales_index_sorted():
    """Qty values are sorted chronologically."""
    sales = _make_sales(n_months=6)
    sales = sales.sample(frac=1, random_state=0)  # shuffle
    idx = build_sales_index(sales)
    _, qty, _ = idx[("ITEM001", "GROUP1", "LOC1")]
    assert len(qty) == 6  # all rows present


def test_build_sales_index_multiple_dfus():
    """Uses one shared calendar across multiple DFUs."""
    s1 = _make_sales("ITEM001", "LOC1", n_months=6)
    s2 = _make_sales("ITEM002", "LOC2", n_months=4)
    idx = build_sales_index(pd.concat([s1, s2], ignore_index=True))
    assert ("ITEM001", "GROUP1", "LOC1") in idx
    assert ("ITEM002", "GROUP1", "LOC2") in idx
    item_two = idx[("ITEM002", "GROUP1", "LOC2")]
    assert len(item_two[1]) == 6
    assert item_two[1][-2:] == [0.0, 0.0]


def test_build_sales_index_preserves_pre_window_introduction_age():
    calendar = pd.date_range("2023-07-01", periods=36, freq="MS")
    sales = pd.DataFrame(
        {
            "item_id": "ITEM001",
            "customer_group": "GROUP1",
            "loc": "LOC1",
            "startdate": calendar,
            "qty": [0.0] * 31 + [10.0] * 5,
            "first_sale_month": pd.Timestamp("2020-01-01"),
        }
    )
    sales.attrs["history_start"] = calendar[0]
    sales.attrs["history_end"] = calendar[-1]

    history = build_sales_index(sales)[("ITEM001", "GROUP1", "LOC1")]

    assert history[2] == 36
    assert history[1][:31] == [0.0] * 31


# ---------------------------------------------------------------------------
# build_attrs_index
# ---------------------------------------------------------------------------


def test_build_attrs_index_structure():
    """Returns dict keyed by full training DFU grain."""
    attrs = _make_dfu_attrs()
    idx = build_attrs_index(attrs)
    key = ("ITEM001", "GROUP1", "LOC1")
    assert key in idx
    assert idx[key]["brand"] == "BrandA"


def test_build_attrs_index_missing_key():
    """Missing DFU returns empty dict via .get()."""
    idx = build_attrs_index(_make_dfu_attrs())
    assert idx.get(("MISSING", "LOC99"), {}) == {}


def test_cluster_id_normalizes_missing_scalars():
    assert _to_cluster_id(pd.NA) is None
    assert _to_cluster_id(np.nan) is None
    assert _to_cluster_id(0) == 0


def test_item_location_cluster_map_requires_one_shared_non_null_cluster():
    attrs = pd.DataFrame(
        [
            {"item_id": "A", "loc": "L1", "ml_cluster": 0},
            {"item_id": "A", "loc": "L1", "ml_cluster": 0},
            {"item_id": "B", "loc": "L2", "ml_cluster": "high"},
            {"item_id": "B", "loc": "L2", "ml_cluster": "low"},
            {"item_id": "C", "loc": "L3", "ml_cluster": pd.NA},
        ]
    )

    cluster_map = build_item_location_cluster_map(attrs)

    assert cluster_map[("A", "L1")] == "0"
    assert cluster_map[("B", "L2")] == "unknown"
    assert cluster_map[("C", "L3")] == "unknown"


def test_inference_profile_uses_full_calendar_complete_selected_customer_group():
    """Serving profile/lag inputs match training for sparse multi-customer history."""
    calendar = pd.date_range("2022-01-01", periods=40, freq="MS")
    group_one = pd.DataFrame(
        {
            "item_id": "ITEM001",
            "customer_group": "GROUP1",
            "loc": "LOC1",
            "startdate": calendar.delete(32),
            "qty": np.arange(1.0, 40.0),
        }
    )
    group_two = pd.DataFrame(
        {
            "item_id": "ITEM001",
            "customer_group": "GROUP2",
            "loc": "LOC1",
            "startdate": calendar.delete(32),
            "qty": np.full(len(calendar) - 1, 1_000.0),
        }
    )
    serving_sales = pd.concat([group_one, group_two], ignore_index=True)
    serving_sales["first_sale_month"] = calendar[0]
    serving_sales.attrs["history_start"] = calendar[-36]
    serving_sales.attrs["history_end"] = calendar[-1]

    attrs_one = _make_dfu_attrs().assign(sku_ck="SKU1")
    attrs_two = _make_dfu_attrs().assign(sku_ck="SKU2", customer_group="GROUP2")
    attrs = pd.concat([attrs_one, attrs_two], ignore_index=True)
    sales_index = build_sales_index(serving_sales)
    inference = build_inference_grid(
        "ITEM001",
        "LOC1",
        2,
        horizon=1,
        sales_index=sales_index,
        attrs_index=build_attrs_index(attrs),
        customer_group="GROUP1",
    )
    assert inference is not None

    training_sales = serving_sales.copy()
    training_sales["sku_ck"] = training_sales["customer_group"].map(
        {"GROUP1": "SKU1", "GROUP2": "SKU2"}
    )
    training_grid = build_feature_matrix(
        training_sales,
        attrs,
        pd.DataFrame(),
        sorted(training_sales["startdate"].unique()),
    )
    training_profile = training_grid[training_grid["sku_ck"] == "SKU1"].iloc[0]
    inference_row = inference.iloc[0]

    for feature_name in TS_PROFILE_FEATURES:
        assert inference_row[feature_name] == pytest.approx(training_profile[feature_name])
    completed_group_one = group_one.set_index("startdate")["qty"].reindex(calendar, fill_value=0.0)
    assert inference_row["qty_lag_1"] == completed_group_one.iloc[-1]
    assert inference_row["qty_lag_8"] == 0.0
    assert inference_row["mean_demand"] < 100.0


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
    for _col, mapping in enc.items():
        for _val, code in mapping.items():
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
        "ITEM001",
        "LOC1",
        2,
        horizon=6,
        sales_index=sales_index,
        attrs_index=attrs_index,
        customer_group="GROUP1",
    )
    assert grid is not None
    assert len(grid) == 6


def test_build_grid_fast_path_missing_dfu_returns_none():
    """Fast path returns None when DFU not in sales_index."""
    sales = _make_sales(n_months=24)
    sales_index = build_sales_index(sales)
    attrs_index = build_attrs_index(_make_dfu_attrs())

    result = build_inference_grid(
        "MISSING",
        "LOC99",
        0,
        horizon=6,
        sales_index=sales_index,
        attrs_index=attrs_index,
        customer_group="GROUP1",
    )
    assert result is None


# ---------------------------------------------------------------------------
# generate_forecasts_batch
# ---------------------------------------------------------------------------


def _make_artifact(pred_val: float = 100.0) -> dict:
    model = MagicMock()
    model.booster_ = None
    model.predict = MagicMock(side_effect=lambda X: np.full(len(X), pred_val))
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    feature_cols = [c for c in grid.columns if not c.startswith("_")]
    return {
        "model": model,
        "feature_cols": feature_cols,
        "categorical_encoders": build_cat_encoders(attrs),
    }


def test_generate_forecasts_batch_returns_rows():
    """Returns horizon rows per DFU."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    artifact = _make_artifact(100.0)
    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            ),
        ],
        horizon=3,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
    )
    assert len(rows) == 3


def test_generate_forecasts_batch_preserves_feature_names_for_prediction():
    """LightGBM receives the named columns used during model fitting."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    artifact = _make_artifact(100.0)

    generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=2,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
    )

    expected_columns = artifact["feature_cols"]
    for call in artifact["model"].predict.call_args_list:
        prediction_input = call.args[0]
        assert isinstance(prediction_input, pd.DataFrame)
        assert prediction_input.columns.tolist() == expected_columns


def test_generate_forecasts_batch_rejects_unavailable_artifact_feature():
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    artifact = _make_artifact(100.0)
    artifact["feature_cols"] = [*artifact["feature_cols"], "retired_missing_feature"]

    with pytest.raises(RuntimeError, match="retired_missing_feature"):
        generate_forecasts_batch(
            artifact=artifact,
            dfu_list=[
                (
                    {
                        "item_id": "ITEM001",
                        "customer_group": "GROUP1",
                        "loc": "LOC1",
                        "cluster_id": 2,
                    },
                    grid,
                )
            ],
            horizon=2,
            forecast_month_generated="2026-03",
            run_id="missing-feature-regression",
            model_id="lgbm_cluster",
        )


def test_generate_forecasts_batch_advances_horizon_calendar_features():
    """Recursive batch inference must not freeze calendar inputs at horizon one."""
    sales = _make_sales(n_months=24, start="2024-01-01")
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    model = MagicMock()
    model.booster_ = None
    model.predict = MagicMock(side_effect=lambda frame: frame["month"].to_numpy(dtype=float))
    artifact = {
        "model": model,
        "feature_cols": [column for column in grid.columns if not column.startswith("_")],
        "categorical_encoders": build_cat_encoders(attrs),
    }

    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=3,
        forecast_month_generated="2026-01",
        run_id="calendar-regression",
        model_id="lgbm_cluster",
    )

    assert [row["forecast_qty"] for row in rows] == [1.0, 2.0, 3.0]


def test_generate_forecasts_batch_updates_recursive_and_croston_features():
    sales = _make_sales(n_months=24, start="2024-01-01")
    sales["qty"] = np.arange(1.0, 25.0)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=3)
    feature_cols = [
        "month",
        "qty_lag_1",
        "qty_lag_2",
        "qty_lag_3",
        "qty_lag_12",
        "rolling_mean_3m",
        "rolling_std_3m",
        "rolling_mean_12m",
        "lag_ratio_yoy",
        "lag_ratio_mom",
        "n_zero_last_6m",
        "croston_demand_size",
        "croston_demand_interval",
        "croston_probability",
    ]
    model = MagicMock()
    model.booster_ = None
    model.predict.side_effect = [np.array([10.0]), np.array([20.0]), np.array([30.0])]

    generate_forecasts_batch(
        artifact={"model": model, "feature_cols": feature_cols},
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=3,
        forecast_month_generated="2026-01",
        run_id="recursive-regression",
        model_id="lgbm_cluster",
    )

    first_step = model.predict.call_args_list[0].args[0].iloc[0]
    assert first_step["croston_demand_size"] == pytest.approx(np.mean(range(13, 25)))
    assert first_step["croston_demand_interval"] == 1.0
    assert first_step["croston_probability"] == 1.0
    second_step = model.predict.call_args_list[1].args[0].iloc[0]
    assert second_step["month"] == 2
    assert second_step["qty_lag_1"] == 10.0
    assert second_step["qty_lag_2"] == 24.0
    assert second_step["qty_lag_12"] == 14.0
    assert second_step["rolling_mean_3m"] == pytest.approx((10.0 + 24.0 + 23.0) / 3)
    assert second_step["rolling_mean_12m"] == pytest.approx(np.mean([10.0, *range(14, 25)]))
    assert second_step["lag_ratio_yoy"] == pytest.approx(10.0 / 15.0)
    assert second_step["lag_ratio_mom"] == pytest.approx(10.0 / 25.0)
    assert second_step["n_zero_last_6m"] == 0
    assert second_step["croston_demand_size"] == pytest.approx(np.mean([10.0, *range(14, 25)]))
    assert second_step["croston_demand_interval"] == 1.0
    assert second_step["croston_probability"] == 1.0


def test_generate_forecasts_batch_short_history_ignores_padding():
    """Cold-start rolling and Croston inputs exclude unavailable lag padding."""
    sales = _make_sales(n_months=3, start="2025-10-01")
    sales["qty"] = [0.0, 10.0, 20.0]
    attrs = _make_dfu_attrs()
    grid = build_inference_grid(
        "ITEM001",
        "LOC1",
        2,
        sales,
        attrs,
        horizon=1,
        min_months=3,
    )
    feature_cols = [
        "rolling_mean_12m",
        "rolling_std_12m",
        "n_zero_last_6m",
        "croston_demand_size",
        "croston_demand_interval",
        "croston_probability",
    ]
    model = MagicMock()
    model.booster_ = None
    model.predict.return_value = np.array([1.0])

    generate_forecasts_batch(
        artifact={"model": model, "feature_cols": feature_cols},
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=1,
        forecast_month_generated="2026-01",
        run_id="short-history-regression",
        model_id="lgbm_cluster",
    )

    prediction_input = model.predict.call_args.args[0].iloc[0]
    assert prediction_input["rolling_mean_12m"] == pytest.approx(10.0)
    assert prediction_input["rolling_std_12m"] == pytest.approx(10.0)
    assert prediction_input["n_zero_last_6m"] == 1
    assert prediction_input["croston_demand_size"] == pytest.approx(15.0)
    assert prediction_input["croston_demand_interval"] == pytest.approx(1.5)
    assert prediction_input["croston_probability"] == pytest.approx(2.0 / 3.0)


def test_generate_forecasts_batch_rejects_grid_shorter_than_horizon():
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)

    with pytest.raises(ValueError, match="shorter than requested horizon"):
        generate_forecasts_batch(
            artifact=_make_artifact(100.0),
            dfu_list=[
                (
                    {
                        "item_id": "ITEM001",
                        "customer_group": "GROUP1",
                        "loc": "LOC1",
                        "cluster_id": 2,
                    },
                    grid,
                )
            ],
            horizon=3,
            forecast_month_generated="2026-03",
            run_id="short-grid",
            model_id="lgbm_cluster",
        )


def test_generate_forecasts_batch_rejects_non_finite_predictions(caplog):
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=1)
    artifact = _make_artifact(float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        generate_forecasts_batch(
            artifact=artifact,
            dfu_list=[
                (
                    {
                        "item_id": "ITEM001",
                        "customer_group": "GROUP1",
                        "loc": "LOC1",
                        "cluster_id": 2,
                    },
                    grid,
                )
            ],
            horizon=1,
            forecast_month_generated="2026-03",
            run_id="nan-prediction",
            model_id="lgbm_cluster",
        )

    assert "Prediction failed for cluster group" in caplog.text


def test_generate_forecasts_batch_uses_native_booster_for_encoded_categories():
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    artifact = _make_artifact(100.0)
    artifact["model"].booster_ = MagicMock()
    artifact["model"].booster_.predict.return_value = np.array([100.0])

    generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=2,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
    )

    assert artifact["model"].booster_.predict.call_count == 2
    artifact["model"].predict.assert_not_called()


def test_generate_forecasts_batch_multi_dfu():
    """Batch processes multiple DFUs in one call."""
    sales1 = _make_sales("ITEM001", "LOC1", n_months=24)
    sales2 = _make_sales("ITEM002", "LOC2", n_months=24)
    attrs1 = _make_dfu_attrs("ITEM001", "LOC1")
    attrs2 = _make_dfu_attrs("ITEM002", "LOC2")
    grid1 = build_inference_grid("ITEM001", "LOC1", 2, sales1, attrs1, horizon=3)
    grid2 = build_inference_grid("ITEM002", "LOC2", 2, sales2, attrs2, horizon=3)
    artifact = _make_artifact(50.0)
    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid1,
            ),
            (
                {"item_id": "ITEM002", "customer_group": "GROUP1", "loc": "LOC2", "cluster_id": 2},
                grid2,
            ),
        ],
        horizon=3,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
    )
    assert len(rows) == 6  # 2 DFUs x 3 months


def test_generate_forecasts_batch_nonneg_qty():
    """Batch clamps negative predictions to 0."""
    sales = _make_sales(n_months=24)
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=2)
    model = MagicMock()
    model.booster_ = None
    model.predict = MagicMock(side_effect=lambda X: np.full(len(X), -999.0))
    artifact = {
        "model": model,
        "feature_cols": [c for c in grid.columns if not c.startswith("_")],
        "categorical_encoders": build_cat_encoders(attrs),
    }
    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=2,
        forecast_month_generated="2026-03",
        run_id="test-run-id",
        model_id="lgbm_cluster",
    )
    for row in rows:
        assert row["forecast_qty"] >= 0.0


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
    rows = generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {"item_id": "ITEM001", "customer_group": "GROUP1", "loc": "LOC1", "cluster_id": 2},
                grid,
            )
        ],
        horizon=18,
        forecast_month_generated="2026-02",
        run_id="test-run-id",
        model_id="lgbm_cluster",
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

    source = inspect.getsource(load_recent_sales)
    assert "get_planning_date" in source, "load_recent_sales must use get_planning_date()"
    assert "Timestamp.now()" not in source, "load_recent_sales must NOT use Timestamp.now()"


def test_load_recent_sales_matches_training_source_and_closed_month():
    source_cursor = MagicMock()
    completed_at = pd.Timestamp("2026-07-01", tz="UTC").to_pydatetime()
    source_cursor.fetchone.side_effect = [
        ("fact_sales_monthly_original",),
        (100, completed_at, 100, completed_at, "sku_lvl2_hist_clean.csv"),
        (pd.Timestamp("2026-06-01").date(),),
    ]
    source_context = MagicMock()
    source_context.__enter__.return_value = source_cursor
    conn = MagicMock()
    conn.cursor.return_value = source_context
    captured: dict[str, object] = {}

    def _read_sql(_conn, sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return pd.DataFrame(
            [
                {
                    "sku_ck": "SKU1",
                    "item_id": "ITEM001",
                    "customer_group": "GROUP1",
                    "loc": "LOC1",
                    "startdate": pd.Timestamp("2026-06-01"),
                    "qty": 10.0,
                    "first_sale_month": pd.Timestamp("2020-01-01"),
                }
            ]
        )

    targets = pd.DataFrame(
        [
            {
                "sku_ck": "SKU1",
                "item_id": "ITEM001",
                "customer_group": "GROUP1",
                "loc": "LOC1",
            }
        ]
    )

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.get_planning_date",
            return_value=pd.Timestamp("2026-07-12").date(),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
            side_effect=_read_sql,
        ),
        patch("psycopg.sql.Identifier.as_string", return_value='"fact_sales_monthly_original"'),
    ):
        result = load_recent_sales(conn, targets, lookback_months=36)

    assert 'JOIN "fact_sales_monthly_original" sales' in str(captured["sql"])
    params = captured["params"]
    assert json.loads(params[0]) == targets.to_dict("records")
    assert params[1:] == [
        pd.Timestamp("2026-06-01").date(),
        pd.Timestamp("2023-07-01").date(),
        pd.Timestamp("2026-06-01").date(),
    ]
    assert result.attrs["history_start"] == pd.Timestamp("2023-07-01")
    assert result.attrs["history_end"] == pd.Timestamp("2026-06-01")


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
    assert config["_pipeline"]["cold_start_model_id"] == "lgbm_cluster"
    assert config["_pipeline"]["cold_start_min_months"] == 3


def test_load_config_ci_section():
    """load_config preserves confidence_interval settings from pipeline config."""
    from scripts.forecasting.generate_production_forecasts import load_config

    config = load_config()
    ci = config["confidence_interval"]
    assert ci["enabled"] is True
    for model_id in ("lgbm_cluster",):
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
# Immutable generation manifests / staging writes
# ---------------------------------------------------------------------------


def _staging_row(
    *,
    model_id: str = "lgbm_cluster",
    run_id: str = "00000000-0000-0000-0000-000000000123",
):
    return {
        "forecast_month_generated": pd.Timestamp("2026-07-01").date(),
        "item_id": "ITEM001",
        "loc": "1401-BULK",
        "forecast_month": pd.Timestamp("2026-07-01").date(),
        "forecast_qty": 42.0,
        "forecast_qty_lower": 35.0,
        "forecast_qty_upper": 50.0,
        "model_id": model_id,
        "cluster_id": "stable",
        "horizon_months": 1,
        "is_recursive": False,
        "lag_source": "actual",
        "run_id": run_id,
        "generated_at": pd.Timestamp("2026-07-10T12:00:00Z").to_pydatetime(),
    }


def _generation_evidence() -> dict:
    return {
        "champion_experiment_id": 33,
        "cluster_experiment_id": 7,
        "source_sales_batch_id": 101,
        "routing_artifact_checksum": "a" * 64,
        "champion_results_checksum": "e" * 64,
        "governed_champion_lineage": {
            "experiment_id": 33,
            "models": [
                "lgbm_cluster",
                "nhits",
                "nbeats",
                "mstl",
                "chronos2_enriched",
            ],
            "backtest_run_ids": {
                "lgbm_cluster": 101,
                "nhits": 102,
                "nbeats": 103,
                "mstl": 104,
                "chronos2_enriched": 105,
            },
            "source_sales_batch_id": 101,
            "data_checksum": "d" * 64,
            "cluster_experiment_id": 7,
            "cluster_assignment_count": 13_968,
            "cluster_assignment_checksum": "f" * 64,
        },
    }


def _snapshot_generation_evidence() -> dict:
    return {
        "governed_champion_lineage": _generation_evidence()[
            "governed_champion_lineage"
        ],
        "source_backtest_run_id": 104,
    }


def test_staging_write_rejects_empty_run_without_preserving_stale_candidate():
    conn = MagicMock()

    with pytest.raises(ValueError, match="no forecast rows"):
        write_forecast_staging(
            [],
            conn,
            "champion",
            generation_purpose="release_candidate",
            generation_evidence=_generation_evidence(),
        )

    conn.cursor.assert_not_called()


def test_staging_write_is_run_scoped_and_records_ready_manifest():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1
    cur.fetchone.side_effect = [
        (
            "release_candidate",
            "champion",
            pd.Timestamp("2026-07-01").date(),
            2,
            "generating",
            0,
        ),
        (False,),
        ("b" * 64, 2, 1, 2),
    ]
    rows = [
        _staging_row(model_id="lgbm_cluster"),
        {
            **_staging_row(model_id="nbeats"),
            "forecast_month": pd.Timestamp("2026-08-01").date(),
            "horizon_months": 2,
            "is_recursive": True,
            "lag_source": "predicted",
        },
    ]

    written = write_forecast_staging(
        rows,
        conn,
        "champion",
        generation_purpose="release_candidate",
        generation_evidence=_generation_evidence(),
    )

    assert written == 2
    statements = [call.args[0] for call in cur.execute.call_args_list]
    assert any("INSERT INTO forecast_generation_run" in sql for sql in statements)
    assert any(
        "SET champion_experiment_id" in sql and "metadata = %s" in sql
        for sql in statements
    )
    assert any(
        "UPDATE forecast_generation_run" in sql and "run_status = 'ready'" in sql
        for sql in statements
    )
    ready_update = next(
        call
        for call in cur.execute.call_args_list
        if "UPDATE forecast_generation_run" in call.args[0]
        and "run_status = 'ready'" in call.args[0]
    )
    assert ready_update.args[1][0] is False
    lineage_update = next(
        call
        for call in cur.execute.call_args_list
        if "SET champion_experiment_id" in call.args[0]
    )
    assert (
        lineage_update.args[1][5].obj["governed_champion_lineage"]
        == _generation_evidence()["governed_champion_lineage"]
    )
    assert not any("DELETE FROM fact_production_forecast_staging" in sql for sql in statements)
    insert_sql = cur.executemany.call_args.args[0]
    assert "generation_purpose" in insert_sql
    assert "candidate_model_id" in insert_sql
    assert "ON CONFLICT (run_id, generation_purpose, candidate_model_id" in " ".join(
        insert_sql.split()
    )
    inserted_rows = cur.executemany.call_args.args[1]
    assert {row["model_id"] for row in inserted_rows} == {"lgbm_cluster", "nbeats"}
    assert {row["candidate_model_id"] for row in inserted_rows} == {"champion"}
    assert {row["generation_purpose"] for row in inserted_rows} == {"release_candidate"}
    conn.commit.assert_called_once()


def test_snapshot_contender_write_is_explicitly_non_release_purpose():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1
    cur.fetchone.side_effect = [
        (
            "snapshot_contender",
            "mstl",
            pd.Timestamp("2026-07-01").date(),
            1,
            "generating",
            0,
        ),
        (False,),
        ("c" * 64, 1, 1, 1),
    ]

    write_forecast_staging(
        [_staging_row(model_id="mstl")],
        conn,
        "mstl",
        generation_purpose="snapshot_contender",
        generation_evidence=_snapshot_generation_evidence(),
    )

    manifest_params = cur.execute.call_args_list[0].args[1]
    assert manifest_params[1] == "snapshot_contender"
    assert manifest_params[2] == "mstl"


def test_missing_required_tree_models_fail_before_fallback_substitution():
    """A winning tree model with no artifacts must fail instead of shipping fallback rows."""
    missing = _missing_required_tree_model_ids(
        {"lgbm_cluster"},
        loaded_models={},
    )

    assert missing == ["lgbm_cluster"]


# ---------------------------------------------------------------------------
# F-11: non-tree champion sources must use their canonical adapter, not the
# LightGBM artifact path. load_non_tree_model_ids() drives that partition.
# ---------------------------------------------------------------------------


def test_load_non_tree_model_ids_routes_canonical_adapters():
    """The four canonical models without .pkl artifacts use direct adapters."""
    from scripts.forecasting.generate_production_forecasts import load_non_tree_model_ids

    non_tree = load_non_tree_model_ids()
    for mid in ("mstl", "nbeats", "nhits", "chronos2_enriched"):
        assert mid in non_tree, f"{mid} must route to its canonical adapter (F-11)"


def test_load_non_tree_model_ids_excludes_tree_models():
    """Tree models keep the .pkl batch path — they must NOT be classed non-tree."""
    from scripts.forecasting.generate_production_forecasts import load_non_tree_model_ids

    non_tree = load_non_tree_model_ids()
    for mid in ("lgbm_cluster",):
        assert mid not in non_tree


def test_cold_start_model_uses_lgbm_fallback():
    """Cold-start routing uses the retained LightGBM fallback."""
    from scripts.forecasting.generate_production_forecasts import (
        load_config,
        load_non_tree_model_ids,
    )

    cold = load_config()["_pipeline"]["cold_start_model_id"]
    assert cold == "lgbm_cluster"
    assert cold not in load_non_tree_model_ids()


# ---------------------------------------------------------------------------
# Data-integrity guard: empty promoted cluster assignments → tree forecast collapse.
# _cluster_assignments_wiped() must fail loud only for the catastrophic
# all-NULL-cluster + multi-cluster-tree signature.
# ---------------------------------------------------------------------------


def _champ_df(cluster_ids):
    return pd.DataFrame(
        {
            "item_id": [f"I{i}" for i in range(len(cluster_ids))],
            "loc": ["L"] * len(cluster_ids),
            "source_model_id": ["lgbm_cluster"] * len(cluster_ids),
            "cluster_id": cluster_ids,
        }
    )


def test_cluster_wipe_detected_all_null_with_multicluster_tree():
    """All cluster_id NULL + a multi-cluster tree loaded → wipe detected (abort)."""
    from scripts.forecasting.generate_production_forecasts import _cluster_assignments_wiped

    loaded = {"lgbm_cluster": {"high_volume_periodic": {}, "low_volume_periodic": {}}}
    assert _cluster_assignments_wiped(_champ_df([None, None]), loaded, {"mstl"}) is True


def test_cluster_wipe_not_flagged_when_clusters_present():
    """Populated cluster_id → not a wipe, even with multi-cluster trees loaded."""
    from scripts.forecasting.generate_production_forecasts import _cluster_assignments_wiped

    loaded = {"lgbm_cluster": {"high_volume_periodic": {}, "low_volume_periodic": {}}}
    assert (
        _cluster_assignments_wiped(_champ_df(["high_volume_periodic", None]), loaded, set())
        is False
    )


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
    return pd.DataFrame(
        [
            {
                "item_id": "10031",
                "loc": "1401-BULK",
                "startdate": pd.Timestamp("2026-01-01"),
                "source_model_id": "mstl",
                "cluster_id": 2,
                "customer_group": "ALL",
            },
            {
                "item_id": "10031",
                "loc": "1401-BULK",
                "startdate": pd.Timestamp("2026-02-01"),
                "source_model_id": "nbeats",
                "cluster_id": 2,
                "customer_group": "ALL",
            },
        ]
    )


def test_build_month_routing_keeps_per_month_model():
    """build_month_routing carries the full DFU grain per month."""
    routing = build_month_routing(_per_month_champ_df())
    dfu = ("10031", "ALL", "1401-BULK")
    assert routing[dfu][pd.Timestamp("2026-01-01")] == "mstl"
    assert routing[dfu][pd.Timestamp("2026-02-01")] == "nbeats"


def test_build_month_routing_skips_null_source():
    """NULL source_model_id rows are skipped (caller applies config fallback)."""
    df = pd.DataFrame(
        [
            {
                "item_id": "A",
                "loc": "L",
                "startdate": pd.Timestamp("2026-01-01"),
                "source_model_id": None,
                "cluster_id": 1,
                "customer_group": "ALL",
            },
        ]
    )
    routing = build_month_routing(df)
    assert pd.Timestamp("2026-01-01") not in routing.get(("A", "ALL", "L"), {})


def test_collapse_to_dfu_one_row_per_dfu():
    """collapse_to_dfu reduces the per-month frame to one row per full DFU."""
    group_one = _per_month_champ_df()
    group_two = group_one.assign(customer_group="SECOND")
    collapsed = collapse_to_dfu(pd.concat([group_one, group_two], ignore_index=True))
    assert len(collapsed) == 2
    # Earliest month's champion is retained deterministically (rows arrive ASC).
    assert set(collapsed["source_model_id"]) == {"mstl"}


def test_filter_rows_to_champion_months_keeps_winning_months_only():
    """Each model's rows survive only for the months it wins."""
    routing = build_month_routing(_per_month_champ_df())
    rows = [
        # MSTL generated the full horizon for this DFU.
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-01-01"),
            "model_id": "mstl",
        },
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "mstl",
        },
        # N-BEATS generated the full horizon for this DFU.
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-01-01"),
            "model_id": "nbeats",
        },
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "nbeats",
        },
    ]
    kept = filter_rows_to_champion_months(rows, routing)
    by_month = {(r["forecast_month"], r["model_id"]) for r in kept}
    assert (pd.Timestamp("2026-01-01"), "mstl") in by_month
    assert (pd.Timestamp("2026-02-01"), "nbeats") in by_month
    # The losing rows are dropped — one model per (DFU, month).
    assert (pd.Timestamp("2026-01-01"), "nbeats") not in by_month
    assert (pd.Timestamp("2026-02-01"), "mstl") not in by_month
    assert len(kept) == 2


def test_filter_rows_routes_customer_groups_independently():
    month = pd.Timestamp("2026-01-01")
    champion = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "G1",
                "loc": "L",
                "startdate": month,
                "source_model_id": "mstl",
            },
            {
                "item_id": "A",
                "customer_group": "G2",
                "loc": "L",
                "startdate": month,
                "source_model_id": "nbeats",
            },
        ]
    )
    rows = [
        {
            "item_id": "A",
            "customer_group": customer_group,
            "loc": "L",
            "forecast_month": month,
            "model_id": model_id,
        }
        for customer_group in ("G1", "G2")
        for model_id in ("mstl", "nbeats")
    ]

    kept = filter_rows_to_champion_months(rows, build_month_routing(champion))

    assert {(row["customer_group"], row["model_id"]) for row in kept} == {
        ("G1", "mstl"),
        ("G2", "nbeats"),
    }


def test_filter_rows_carries_latest_known_model_into_future_month():
    """A future month uses the sole latest-known model for the DFU.

    Avoids duplicate (DFU, month) staging rows when the DFU is generated under
    several models but a given month has no winner recorded.
    """
    # Only January routes (N-BEATS); February has NO recorded winner.
    routing = {("10031", "ALL", "1401-BULK"): {pd.Timestamp("2026-01-01"): "nbeats"}}
    rows = [
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "nbeats",
        },
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "mstl",
        },
    ]
    kept = filter_rows_to_champion_months(rows, routing)
    assert len(kept) == 1
    assert kept[0]["model_id"] == "nbeats"


def test_future_month_uses_latest_as_of_champion_not_lexical_model():
    routing = {
        ("10031", "ALL", "1401-BULK"): {
            pd.Timestamp("2026-01-01"): "nbeats",
            pd.Timestamp("2026-06-01"): "mstl",
        }
    }
    rows = [
        {
            "item_id": "10031",
            "customer_group": "ALL",
            "loc": "1401-BULK",
            "forecast_month": pd.Timestamp("2026-07-01"),
            "model_id": model_id,
        }
        for model_id in ("nbeats", "mstl")
    ]

    kept = filter_rows_to_champion_months(rows, routing)

    assert [row["model_id"] for row in kept] == ["mstl"]


def test_filter_rows_passthrough_when_no_routing():
    """A DFU with no per-month routing (cold-start / override) is untouched."""
    rows = [
        {
            "item_id": "A",
            "customer_group": "ALL",
            "loc": "L",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "nbeats",
        },
    ]
    assert filter_rows_to_champion_months(rows, {}) == rows


def test_filter_rows_blends_synthetic_ensemble_members():
    champion = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "ALL",
                "loc": "L",
                "startdate": pd.Timestamp("2026-01-01"),
                "source_model_id": "ensemble",
                "source_mix": [
                    {"model": "lgbm_cluster", "weight": 0.25},
                    {"model": "mstl", "weight": 0.75},
                ],
            }
        ]
    )
    rows = [
        {
            "item_id": "A",
            "customer_group": "ALL",
            "loc": "L",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "lgbm_cluster",
            "forecast_qty": 100.0,
            "forecast_qty_lower": 80.0,
            "forecast_qty_upper": 120.0,
        },
        {
            "item_id": "A",
            "customer_group": "ALL",
            "loc": "L",
            "forecast_month": pd.Timestamp("2026-02-01"),
            "model_id": "mstl",
            "forecast_qty": 200.0,
            "forecast_qty_lower": 180.0,
            "forecast_qty_upper": 220.0,
        },
    ]

    kept = filter_rows_to_champion_months(
        rows, build_month_routing(champion), build_ensemble_routing(champion)
    )

    assert len(kept) == 1
    assert kept[0]["model_id"] == "ensemble"
    assert kept[0]["forecast_qty"] == 175.0
    assert kept[0]["forecast_qty_lower"] == 155.0
    assert kept[0]["forecast_qty_upper"] == 195.0


@pytest.mark.parametrize(
    "mix",
    [
        [
            {"model": "lgbm_cluster", "weight": 0.6},
            {"model": "mstl", "weight": 0.3},
        ],
        [
            {"model": "lgbm_cluster"},
            {"model": "mstl", "weight": 1.0},
        ],
        [
            {"model": "lgbm_cluster", "weight": -0.1},
            {"model": "mstl", "weight": 1.1},
        ],
        [
            {"model": "lgbm_cluster", "weight": float("nan")},
            {"model": "mstl", "weight": 1.0},
        ],
        [
            {"model": "mstl", "weight": 0.5},
            {"model": "mstl", "weight": 0.5},
        ],
    ],
)
def test_ensemble_mix_rejects_malformed_or_non_unit_weights(mix):
    with pytest.raises(ValueError, match="ensemble source_mix"):
        _parse_ensemble_mix(mix)


def test_ensemble_mix_normalizes_small_rounding_drift():
    mix = _parse_ensemble_mix(
        [
            {"model": "lgbm_cluster", "weight": 0.50005},
            {"model": "mstl", "weight": 0.49995},
        ]
    )

    assert mix is not None
    assert sum(entry["weight"] for entry in mix) == pytest.approx(1.0)
    assert [entry["model"] for entry in mix] == ["lgbm_cluster", "mstl"]


def test_single_member_ensemble_routes_as_its_underlying_model():
    champion = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "ALL",
                "loc": "L",
                "startdate": pd.Timestamp("2026-02-01"),
                "source_model_id": "ensemble",
                "source_mix": '[{"model":"lgbm_cluster","weight":1.0}]',
            }
        ]
    )

    assert build_month_routing(champion) == {
        ("A", "ALL", "L"): {pd.Timestamp("2026-02-01"): "lgbm_cluster"}
    }
    assert build_ensemble_routing(champion) == {}


def test_aggregate_customer_groups_sums_after_independent_routing():
    generated_at = pd.Timestamp("2026-07-01", tz="UTC")
    common = {
        "forecast_month_generated": pd.Timestamp("2026-07-01").date(),
        "item_id": "A",
        "loc": "L",
        "forecast_month": pd.Timestamp("2026-08-01").date(),
        "horizon_months": 1,
        "is_recursive": False,
        "lag_source": "actual",
        "run_id": "run-1",
        "generated_at": generated_at,
    }
    rows = [
        {
            **common,
            "customer_group": "G1",
            "forecast_qty": 10.0,
            "forecast_qty_lower": 8.0,
            "forecast_qty_upper": 12.0,
            "model_id": "mstl",
            "cluster_id": 1,
        },
        {
            **common,
            "customer_group": "G2",
            "forecast_qty": 20.0,
            "forecast_qty_lower": 15.0,
            "forecast_qty_upper": 25.0,
            "model_id": "lgbm_cluster",
            "cluster_id": 2,
        },
    ]

    aggregated = aggregate_customer_group_forecasts(rows)

    assert len(aggregated) == 1
    result = aggregated[0]
    assert "customer_group" not in result
    assert result["forecast_qty"] == 30.0
    assert result["forecast_qty_lower"] is None
    assert result["forecast_qty_upper"] is None
    assert result["model_id"] == "ensemble"
    assert result["cluster_id"] is None

    with_intervals = attach_aggregate_confidence_intervals(
        aggregated,
        sigma_lookup={("A", "L"): 10.0},
        ci_cfg={"z_lower": 1.0, "z_upper": 1.0, "horizon_scaling": "none"},
    )
    assert with_intervals[0]["forecast_qty_lower"] == 20.0
    assert with_intervals[0]["forecast_qty_upper"] == 40.0


def test_aggregate_customer_groups_preserves_unique_lineage_and_requires_unique_rows():
    generated_at = pd.Timestamp("2026-07-01", tz="UTC")
    common = {
        "forecast_month_generated": pd.Timestamp("2026-07-01").date(),
        "item_id": "A",
        "loc": "L",
        "forecast_month": pd.Timestamp("2026-08-01").date(),
        "forecast_qty": 5.0,
        "forecast_qty_lower": None,
        "forecast_qty_upper": None,
        "model_id": "mstl",
        "horizon_months": 1,
        "is_recursive": False,
        "lag_source": "actual",
        "run_id": "run-1",
        "generated_at": generated_at,
    }
    rows = [
        {**common, "customer_group": "G1", "cluster_id": 1},
        {**common, "customer_group": "G2", "cluster_id": 1},
    ]
    result = aggregate_customer_group_forecasts(rows)[0]
    assert result["model_id"] == "mstl"
    assert result["cluster_id"] == 1

    ensemble_rows = [{**row, "model_id": "ensemble"} for row in rows]
    assert (
        aggregate_customer_group_forecasts(
            ensemble_rows,
        )[0]["model_id"]
        == "ensemble"
    )

    rows[1]["cluster_id"] = None
    assert (
        aggregate_customer_group_forecasts(
            rows,
        )[0]["cluster_id"]
        is None
    )

    with pytest.raises(ValueError, match="multiple forecasts survived"):
        aggregate_customer_group_forecasts(
            [rows[0], dict(rows[0])],
        )


def test_aggregate_customer_groups_merges_mixed_recursive_metadata():
    generated_at = pd.Timestamp("2026-07-01", tz="UTC")
    common = {
        "forecast_month_generated": pd.Timestamp("2026-07-01").date(),
        "item_id": "A",
        "loc": "L",
        "forecast_month": pd.Timestamp("2026-09-01").date(),
        "forecast_qty_lower": None,
        "forecast_qty_upper": None,
        "cluster_id": 1,
        "horizon_months": 2,
        "run_id": "run-1",
        "generated_at": generated_at,
    }
    rows = [
        {
            **common,
            "customer_group": "G1",
            "forecast_qty": 10.0,
            "model_id": "lgbm_cluster",
            "is_recursive": True,
            "lag_source": "predicted",
        },
        {
            **common,
            "customer_group": "G2",
            "forecast_qty": 20.0,
            "model_id": "mstl",
            "is_recursive": False,
            "lag_source": "actual",
        },
    ]

    aggregated = aggregate_customer_group_forecasts(rows)

    assert len(aggregated) == 1
    assert aggregated[0]["forecast_qty"] == 30.0
    assert aggregated[0]["is_recursive"] is True
    assert aggregated[0]["lag_source"] == "mixed"


def test_customer_group_coverage_rejects_partial_item_location_output():
    champion = pd.DataFrame(
        [
            {"item_id": "A", "customer_group": "G1", "loc": "L"},
            {"item_id": "A", "customer_group": "G2", "loc": "L"},
        ]
    )
    rows = [
        {
            "item_id": "A",
            "customer_group": "G1",
            "loc": "L",
            "forecast_month": pd.Timestamp("2026-08-01").date(),
        },
        {
            "item_id": "A",
            "customer_group": "G1",
            "loc": "L",
            "forecast_month": pd.Timestamp("2026-09-01").date(),
        },
    ]

    with pytest.raises(RuntimeError, match="incomplete for 1 customer-group"):
        validate_customer_group_forecast_coverage(rows, champion, horizon=2)


def test_customer_group_history_rejects_short_group_before_model_execution():
    champion = pd.DataFrame([{"item_id": "A", "customer_group": "G1", "loc": "L"}])
    dates = list(pd.date_range("2026-04-01", periods=2, freq="MS").values)
    sales_index = {("A", "G1", "L"): (dates, [10.0, 20.0], 2)}

    with pytest.raises(RuntimeError, match="minimum is 3 months"):
        validate_customer_group_history(
            sales_index,
            champion,
            minimum_months=3,
        )


def test_get_champion_assignments_returns_per_month_rows():
    """get_champion_assignments keeps the startdate dimension (per-month rows)."""
    per_month = _per_month_champ_df()

    captured = {}

    def _fake_read_sql_chunked(conn, sql, params=None):
        captured["sql"] = sql
        return per_month.copy()

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts._get_promoted_champion_experiment_id",
            return_value=None,
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
            side_effect=_fake_read_sql_chunked,
        ),
    ):
        df = get_champion_assignments(MagicMock())

    # Query must preserve the training DFU grain and month.
    assert "DISTINCT ON (f.item_id, f.customer_group, f.loc, f.startdate)" in captured["sql"]
    # Deterministic tie-break is source_model_id ASC (shared with promote side).
    assert "f.source_model_id ASC" in captured["sql"]
    assert "startdate" in df.columns
    # Two months for the one DFU survive.
    assert len(df) == 2
    assert set(df["source_model_id"]) == {"mstl", "nbeats"}


def test_routes_are_aligned_to_full_forecastable_population_with_fallback():
    population = pd.DataFrame(
        [
            {
                "sku_ck": "A_G_L",
                "item_id": "A",
                "customer_group": "G",
                "loc": "L",
                "cluster_id": "stable",
            },
            {
                "sku_ck": "B_G_L",
                "item_id": "B",
                "customer_group": "G",
                "loc": "L",
                "cluster_id": "lumpy",
            },
        ]
    )
    routes = pd.DataFrame(
        [
            {
                "sku_ck": "stale-a",
                "item_id": "A",
                "customer_group": "G",
                "loc": "L",
                "cluster_id": "old",
                "startdate": pd.Timestamp("2026-06-01"),
                "source_model_id": "nhits",
                "source_mix": None,
            },
            {
                "sku_ck": "C_G_L",
                "item_id": "C",
                "customer_group": "G",
                "loc": "L",
                "cluster_id": "stale",
                "startdate": pd.Timestamp("2026-06-01"),
                "source_model_id": "mstl",
                "source_mix": None,
            },
        ]
    )

    aligned = _align_routes_to_population(
        routes,
        population,
        fallback_model_id="lgbm_cluster",
        planning_month=pd.Timestamp("2026-07-01"),
    )

    assert set(aligned["item_id"]) == {"A", "B"}
    routed_a = aligned[aligned["item_id"] == "A"].iloc[0]
    fallback_b = aligned[aligned["item_id"] == "B"].iloc[0]
    assert routed_a["source_model_id"] == "nhits"
    assert routed_a["sku_ck"] == "A_G_L"
    assert routed_a["cluster_id"] == "stable"
    assert fallback_b["source_model_id"] == "lgbm_cluster"
    assert fallback_b["startdate"] == pd.Timestamp("2026-07-01")


def test_route_alignment_discards_future_only_winner_and_adds_fallback():
    population = pd.DataFrame(
        [
            {
                "sku_ck": "A_G_L",
                "item_id": "A",
                "customer_group": "G",
                "loc": "L",
                "cluster_id": "stable",
            }
        ]
    )
    routes = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "G",
                "loc": "L",
                "startdate": pd.Timestamp("2026-08-01"),
                "source_model_id": "mstl",
                "source_mix": None,
            }
        ]
    )

    aligned = _align_routes_to_population(
        routes,
        population,
        fallback_model_id="lgbm_cluster",
        planning_month=pd.Timestamp("2026-07-01"),
    )

    assert len(aligned) == 1
    assert aligned.iloc[0]["source_model_id"] == "lgbm_cluster"
    assert aligned.iloc[0]["startdate"] == pd.Timestamp("2026-07-01")


def test_forecast_population_query_matches_release_eligibility_window(monkeypatch):
    captured = {}

    def fake_read(_conn, sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return pd.DataFrame(
            columns=[
                "sku_ck",
                "item_id",
                "customer_group",
                "loc",
                "cluster_id",
            ]
        )

    monkeypatch.setattr(
        "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
        fake_read,
    )

    load_forecast_population(
        MagicMock(),
        planning_month=pd.Timestamp("2026-07-01").date(),
        min_history_months=3,
        active_window_months=12,
        sales_table="fact_sales_monthly_original",
    )

    normalized_sql = " ".join(captured["sql"].split())
    assert "group_history AS" in normalized_sql
    assert "COUNT(DISTINCT sales.startdate) AS history_months" in normalized_sql
    assert "active_customer_groups AS" in normalized_sql
    assert "GROUP BY sales.item_id, sales.customer_group, sales.loc" in normalized_sql
    assert "BOOL_AND(" in normalized_sql
    assert "active.history_months >= %s" in normalized_sql
    assert "JOIN active_customer_groups active" in normalized_sql
    assert "JOIN eligible_item_locations eligible USING (item_id, loc)" in normalized_sql
    assert captured["params"] == [
        pd.Timestamp("2026-07-01").date(),
        pd.Timestamp("2026-07-01").date(),
        12,
        3,
    ]


def test_latest_as_of_route_never_uses_future_only_winner():
    dfu = ("A", "G", "L")
    routes = {
        dfu: {
            pd.Timestamp("2026-08-01"): "mstl",
        }
    }

    assert (
        _resolve_champion_route(
            dfu,
            pd.Timestamp("2026-07-01"),
            routes,
            {},
        )
        is None
    )


def test_mstl_snapshot_population_uses_its_configured_history_floor():
    config = {
        "algorithms": {
            "mstl": {"params": {"min_history": 25}},
            "nhits": {"params": {"min_history": 12}},
            "nbeats": {"params": {"min_history": 12}},
            "chronos2_enriched": {"params": {"min_history": 3}},
        }
    }

    assert _population_min_history("mstl", config, cold_start_min_months=3) == 25
    assert _population_min_history("nhits", config, cold_start_min_months=3) == 12
    assert _population_min_history("nbeats", config, cold_start_min_months=3) == 12
    assert _population_min_history("chronos2_enriched", config, cold_start_min_months=3) == 3
    assert _population_min_history("lgbm_cluster", config, cold_start_min_months=3) == 3


def test_stale_mstl_route_fails_before_inference_when_history_is_too_short():
    champion = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "G",
                "loc": "L",
                "source_model_id": "mstl",
            }
        ]
    )
    routes = {
        ("A", "G", "L"): {pd.Timestamp("2026-06-01"): "mstl"},
    }
    sales_index = {
        ("A", "G", "L"): ([], [], 15),
    }

    with pytest.raises(RuntimeError, match=r"MSTL.*25 months"):
        validate_route_history_requirements(
            sales_index,
            champion,
            production_months=[pd.Timestamp("2026-07-01")],
            month_routing=routes,
            ensemble_routing={},
            min_history_months=12,
            mstl_min_history=25,
        )


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
        "10031,ALL,1401-BULK,mstl,2026-01-01\n"
        "10031,ALL,1401-BULK,nbeats,2026-02-01\n"
        # Duplicate same DFU-month resolves lexically by model_id, matching promote.
        "10031,ALL,1401-BULK,nhits,2026-02-01\n"
        # A second customer group must survive to independent inference.
        "10031,SECOND,1401-BULK,nhits,2026-02-01\n"
    )
    attrs = pd.DataFrame(
        [
            {
                "sku_ck": "SKU-ALL",
                "item_id": "10031",
                "customer_group": "ALL",
                "loc": "1401-BULK",
                "ml_cluster": 7,
                "execution_lag": 1,
                "total_lt": 14,
                "brand": "BrandA",
                "region": "NORTH",
                "abc_vol": "A",
            },
            {
                "sku_ck": "SKU-SECOND",
                "item_id": "10031",
                "customer_group": "SECOND",
                "loc": "1401-BULK",
                "ml_cluster": 8,
                "execution_lag": 1,
                "total_lt": 14,
                "brand": "BrandA",
                "region": "NORTH",
                "abc_vol": "A",
            },
        ]
    )

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts._get_promoted_champion_experiment_id",
            return_value=53,
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.CHAMPION_WINNERS_DIR",
            champion_dir,
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.load_dfu_attrs",
            return_value=attrs,
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.read_sql_chunked",
        ) as legacy_read,
    ):
        df = get_champion_assignments(MagicMock())

    legacy_read.assert_not_called()
    assert len(df) == 3
    assert list(df["source_model_id"]) == ["mstl", "nbeats", "nhits"]
    assert set(df["cluster_id"]) == {7, 8}
    assert set(df["customer_group"]) == {"ALL", "SECOND"}


def test_promoted_winners_reject_blank_customer_group(tmp_path):
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    (champion_dir / "experiment_54_winners.csv").write_text(
        "item_id,customer_group,loc,model_id,startdate\n10031,,1401-BULK,mstl,2026-01-01\n"
    )

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts._get_promoted_champion_experiment_id",
            return_value=54,
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.CHAMPION_WINNERS_DIR",
            champion_dir,
        ),
        pytest.raises(ValueError, match="blank customer_group"),
    ):
        get_champion_assignments(MagicMock())
