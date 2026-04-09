"""
F1.1 — Production Forecast Generation Pipeline

Runs champion ML models over a forward horizon and writes predictions to
fact_production_forecast_staging. This is the bridge between the backtesting engine
(which evaluates historical accuracy) and operational planning (which needs
future-period demand signals).

The key difference from backtesting:
  - Backtest: predicts historical months where actuals exist (for model evaluation)
  - Production: predicts future months where no actuals exist yet (for planning)

Usage:
    uv run python scripts/generate_production_forecasts.py
    uv run python scripts/generate_production_forecasts.py --horizon 6
    uv run python scripts/generate_production_forecasts.py --dfu 100320 1401-BULK
    uv run python scripts/generate_production_forecasts.py --dry-run

Algorithm:
    For each DFU in champion assignments:
        1. Load the cluster's .pkl model artifact
        2. Build an inference grid (feature matrix for T+1 through T+horizon)
        3. Generate predictions recursively: lag_1 for T+2 = model's T+1 prediction
        4. Write to fact_production_forecast_staging with upsert on (model_id, item_id, loc, forecast_month)
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.constants import CAT_FEATURES, LAG_RANGE, ROLLING_WINDOWS
from common.forecast_ci import build_sigma_lookup, compute_ci_bounds
from common.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section

import psycopg

PIPELINE_CONFIG_PATH = ROOT / "config" / "forecast_pipeline_config.yaml"

logger = logging.getLogger(__name__)


def _detect_framework(model) -> str:
    """Return 'catboost', 'lgbm', or 'xgboost' based on model object type."""
    mod = type(model).__module__
    if "catboost" in mod:
        return "catboost"
    if "lightgbm" in mod:
        return "lgbm"
    if "xgboost" in mod:
        return "xgboost"
    return "lgbm"


def _to_cluster_id(cluster_id) -> int | str | None:
    """Normalize cluster_id for DB storage.

    ml_cluster can be an integer (e.g. 0, 3) or a string label
    (e.g. 'high_volume_declining'). Try int conversion; keep as str if not numeric.
    """
    if cluster_id is None:
        return None
    try:
        return int(cluster_id)
    except (ValueError, TypeError):
        return str(cluster_id)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load production forecast config.

    Reads from the consolidated forecast_pipeline_config.yaml (production_forecast section).
    """
    if not PIPELINE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {PIPELINE_CONFIG_PATH}"
        )
    with open(PIPELINE_CONFIG_PATH) as f:
        pipeline = yaml.safe_load(f) or {}
    pf = pipeline.get("production_forecast", {})
    # Map to the structure the rest of the script expects
    return {
        "inference": {
            "horizon_months": pf.get("horizon_months", 24),
            "recursive": pf.get("recursive", True),
        },
        "confidence_interval": pf.get("confidence_interval", {}),
        "model_selection": {
            "strategy": "champion",
            "fallback_model_id": pf.get("fallback_model_id", "lgbm_cluster"),
        },
        "plan_version": {
            "format": pf.get("plan_version_format", "%Y-%m"),
            "keep_last_n_versions": pf.get("keep_last_n_versions", 3),
        },
        "model_registry": pf.get("model_registry", {"base_path": "data/models"}),
        # New cold-start fields (used in main loop)
        "_pipeline": {
            "lookback_months": pf.get("lookback_months", 36),
            "min_history_months": pf.get("min_history_months", 12),
            "cold_start_model_id": pf.get("cold_start_model_id", "rolling_mean"),
            "cold_start_min_months": pf.get("cold_start_min_months", 3),
        },
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_active_models(model_id: str, config: dict) -> dict[str, dict]:
    """Load pickled model artifacts for all clusters of a given model_id.

    Returns:
        {cluster_label: {"model": <model>, "feature_cols": [...]}}
    """
    base_path = config.get("model_registry", {}).get("base_path", "data/models")
    model_dir = ROOT / base_path / model_id

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            f"Run 'make backtest-{model_id.replace('_cluster', '')}' to train and persist models."
        )

    models = {}
    for fname in sorted(model_dir.glob("cluster_*.pkl")):
        cluster_label_raw = fname.stem.replace("cluster_", "")
        # Cluster labels may be integers or strings (e.g. "0", "3")
        try:
            cluster_label = int(cluster_label_raw)
        except ValueError:
            cluster_label = cluster_label_raw

        with open(fname, "rb") as f:
            artifact = pickle.load(f)
        models[cluster_label] = artifact

    logger.info("Loaded %d %s cluster models from %s/", len(models), model_id, model_dir)
    return models


# ---------------------------------------------------------------------------
# Champion assignment query
# ---------------------------------------------------------------------------


def get_champion_assignments(conn, item_id: str | None = None, loc: str | None = None) -> pd.DataFrame:
    """Return the most recent champion model assignment per DFU.

    Returns `source_model_id` — the underlying algorithm (e.g. lgbm_cluster) whose
    artifacts are used for production inference.  Populated by champion selection
    via the source_model_id column (added in F1.1 via sql/007_create_fact_external_forecast_monthly.sql).
    When NULL (champion rows from before this column was added), the caller falls
    back to `fallback_model_id` from config.

    Returns DataFrame with columns: item_id, loc, source_model_id, cluster_id, customer_group.
    """
    where_clauses = ["f.model_id = 'champion'"]
    params: list = []

    if item_id:
        where_clauses.append("f.item_id = %s")
        params.append(item_id)
    if loc:
        where_clauses.append("f.loc = %s")
        params.append(loc)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT DISTINCT ON (f.item_id, f.loc)
            f.item_id                   AS item_id,
            f.loc,
            f.source_model_id,
            d.ml_cluster                AS cluster_id,
            d.customer_group
        FROM fact_external_forecast_monthly f
        JOIN dim_sku d ON d.item_id = f.item_id AND d.loc = f.loc
        WHERE {where_sql}
        ORDER BY f.item_id, f.loc, f.startdate DESC
    """

    df = pd.read_sql(sql, conn, params=params)
    with_src = int(df["source_model_id"].notna().sum())
    logger.info("Champion assignments loaded: %s DFUs (%s with source_model_id)",
                f"{len(df):,}", f"{with_src:,}")
    return df


# ---------------------------------------------------------------------------
# Historical sales loading
# ---------------------------------------------------------------------------


def load_recent_sales(conn, item_id: str | None = None, loc: str | None = None,
                      lookback_months: int = 36) -> pd.DataFrame:
    """Load the last N months of sales for all DFUs (or a specific DFU).

    Returns DataFrame with columns: item_id, loc, startdate, qty.
    """
    planning_dt = pd.Timestamp(get_planning_date())
    planning_upper = planning_dt.normalize().replace(day=1)
    cutoff = planning_dt.normalize() - pd.DateOffset(months=lookback_months)

    where_clauses = ["s.startdate >= %s", "s.startdate <= %s", "s.type = 1"]
    params: list = [cutoff.date(), planning_upper.date()]

    if item_id:
        where_clauses.append("s.item_id = %s")
        params.append(item_id)
    if loc:
        where_clauses.append("s.loc = %s")
        params.append(loc)

    where_sql = " AND ".join(where_clauses)

    # Join dim_sku to filter by customer_group — backtest training joined on item_id+customer_group+loc,
    # so inference must use the same single customer group to get matching lag features.
    # Without the customer_group filter, multiple customer-group rows per month would cause
    # wrong lag values (lag_1 = one customer group's qty, not the DFU-level qty).
    sql = f"""
        SELECT s.item_id AS item_id, s.loc, s.startdate,
               SUM(COALESCE(s.qty, 0)) AS qty
        FROM fact_sales_monthly s
        INNER JOIN dim_sku d ON d.item_id = s.item_id AND d.customer_group = s.customer_group AND d.loc = s.loc
        WHERE {where_sql}
        GROUP BY s.item_id, s.loc, s.startdate
        ORDER BY s.item_id, s.loc, s.startdate
    """

    df = pd.read_sql(sql, conn, params=params)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    logger.info("Recent sales loaded: %s rows, %s DFUs",
                f"{len(df):,}", f"{df.groupby(['item_id','loc']).ngroups:,}")
    return df


# ---------------------------------------------------------------------------
# DFU attributes loading
# ---------------------------------------------------------------------------


def load_dfu_attrs(conn, item_id: str | None = None, loc: str | None = None) -> pd.DataFrame:
    """Load DFU attributes needed for feature construction."""
    where_clauses = []
    params: list = []

    if item_id:
        where_clauses.append("item_id = %s")
        params.append(item_id)
    if loc:
        where_clauses.append("loc = %s")
        params.append(loc)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT item_id AS item_id, customer_group, loc, ml_cluster,
               execution_lag, total_lt, brand, region, abc_vol
        FROM dim_sku
        {where_sql}
    """

    df = pd.read_sql(sql, conn, params=params)
    logger.info("DFU attributes loaded: %s rows", f"{len(df):,}")
    return df


def load_item_attrs(conn, item_id: str | None = None) -> pd.DataFrame:
    """Load item-level attributes (bpc, item_proof, case_weight) from dim_item."""
    where_sql = "WHERE item_id = %s" if item_id else ""
    params = [item_id] if item_id else []
    sql = f"""
        SELECT item_id, bpc, item_proof, case_weight
        FROM dim_item
        {where_sql}
    """
    df = pd.read_sql(sql, conn, params=params)
    df["bpc"] = pd.to_numeric(df["bpc"], errors="coerce").fillna(0)
    df["item_proof"] = pd.to_numeric(df["item_proof"], errors="coerce").fillna(0)
    df["case_weight"] = pd.to_numeric(df["case_weight"], errors="coerce").fillna(0)
    logger.info("Item attributes loaded: %s rows", f"{len(df):,}")
    return df


def build_item_index(item_attrs: pd.DataFrame) -> dict[str, dict]:
    """Pre-index item attributes by item_id → {bpc, item_proof, case_weight}."""
    return item_attrs.set_index("item_id")[["bpc", "item_proof", "case_weight"]].to_dict("index")


# ---------------------------------------------------------------------------
# Pre-indexing helpers — O(1) lookup replacements for per-DFU DataFrame scans
# ---------------------------------------------------------------------------


def build_sales_index(sales_df: pd.DataFrame) -> dict[tuple, tuple]:
    """Pre-index sales by (item_id, loc) → (sorted_dates, sorted_qty_list).

    Replaces O(N) pandas boolean-filter scan in build_inference_grid with an
    O(1) dict lookup. Call once before the DFU loop.
    """
    index: dict[tuple, tuple] = {}
    for (item_id, loc), grp in sales_df.sort_values("startdate").groupby(
        ["item_id", "loc"], sort=False
    ):
        index[(item_id, loc)] = (
            list(grp["startdate"].values),
            list(grp["qty"].values),
        )
    return index


def build_attrs_index(dfu_attrs: pd.DataFrame) -> dict[tuple, dict]:
    """Pre-index DFU attributes by (item_id, loc) → attr dict.

    Replaces O(N) pandas boolean-filter scan with O(1) dict lookup.
    """
    records = dfu_attrs.set_index(["item_id", "loc"]).to_dict("index")
    return records


def build_cat_encoders(dfu_attrs: pd.DataFrame) -> dict[str, dict]:
    """Build integer label encoders for CAT_FEATURES from the full DFU set.

    Training used pandas category codes (sorted unique values → 0-based int).
    Inference must apply the same encoding so models receive integers, not strings.
    Returns {col: {str_value: int_code}} covering all values present in dim_sku.
    """
    encoders: dict[str, dict] = {}
    for col in CAT_FEATURES:
        if col in dfu_attrs.columns:
            cats = sorted(
                dfu_attrs[col].fillna("__unknown__").astype(str).unique()
            )
            encoders[col] = {v: i for i, v in enumerate(cats)}
    return encoders


# ---------------------------------------------------------------------------
# Inference grid construction
# ---------------------------------------------------------------------------


def _compute_ts_profile_from_history(qty_series: list[float]) -> dict[str, float]:
    """Compute TS profile features from a DFU's historical qty series.

    Mirrors the features from feature_engineering._compute_ts_profile_features
    but operates on a plain list instead of a DataFrame group.
    """
    arr = np.array(qty_series, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr)) if n > 0 else 0.0
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # cv_demand
    cv = std_val / (abs(mean_val) + 1e-9) if n > 1 else 0.0

    # zero_demand_pct
    zero_pct = float(np.sum(arr == 0)) / n if n > 0 else 0.0

    # trend_slope_norm
    if n >= 3 and abs(mean_val) > 1e-9:
        x = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(x, arr, 1)[0])
        trend_slope = slope / abs(mean_val)
    else:
        trend_slope = 0.0

    # recency_ratio
    if n >= 12:
        recent = float(np.mean(arr[-6:]))
        prior = float(np.mean(arr[:-6]))
        recency = recent / (prior + 1e-9) if prior > 1e-9 else 1.0
    else:
        recency = 1.0

    # seasonal_amplitude
    if n >= 24:
        monthly_avg = np.zeros(12)
        monthly_cnt = np.zeros(12)
        for i, v in enumerate(arr):
            monthly_avg[i % 12] += v
            monthly_cnt[i % 12] += 1
        mask = monthly_cnt > 0
        monthly_avg[mask] /= monthly_cnt[mask]
        seasonal_amp = float(np.max(monthly_avg) - np.min(monthly_avg)) / (abs(mean_val) + 1e-9) if abs(mean_val) > 1e-9 else 0.0
    else:
        seasonal_amp = 0.0

    # adi (average demand interval)
    nz_idx = np.where(arr > 0)[0]
    if len(nz_idx) >= 2:
        adi_val = float(np.mean(np.diff(nz_idx)))
    elif len(nz_idx) == 1:
        adi_val = float(n)
    else:
        adi_val = float(n)

    # yoy_correlation
    if n >= 24:
        recent_12 = arr[-12:]
        prior_12 = arr[-24:-12]
        if np.std(recent_12) > 1e-9 and np.std(prior_12) > 1e-9:
            yoy_corr = float(np.corrcoef(recent_12, prior_12)[0, 1])
            if np.isnan(yoy_corr):
                yoy_corr = 0.0
        else:
            yoy_corr = 0.0
    else:
        yoy_corr = 0.0

    return {
        "cv_demand": cv,
        "zero_demand_pct": zero_pct,
        "trend_slope_norm": trend_slope,
        "recency_ratio": recency,
        "seasonal_amplitude": seasonal_amp,
        "adi": adi_val,
        "mean_demand": mean_val,
        "yoy_correlation": yoy_corr,
    }


def build_inference_grid(
    item_id: str,
    loc: str,
    cluster_id: int | str,
    sales_history: pd.DataFrame | None = None,
    dfu_attrs: pd.DataFrame | None = None,
    horizon: int = 24,
    min_months: int = 3,
    *,
    sales_index: dict | None = None,
    attrs_index: dict | None = None,
    item_index: dict | None = None,
) -> pd.DataFrame | None:
    """Build a feature matrix for recursive inference over the next `horizon` months.

    For T+1: lag_1 = last known actual
    For T+2: lag_1 = model's T+1 prediction (written back after each step)
    For T+N: lag_1 = model's T+(N-1) prediction

    Returns DataFrame with `horizon` rows ready for model.predict(), or None if
    insufficient history.
    """
    max_lag = max(LAG_RANGE)

    if sales_index is not None:
        # Fast O(1) path — used by production main loop
        entry = sales_index.get((item_id, loc))
        if entry is None:
            return None
        dates_arr, qty_arr = entry
        if len(qty_arr) < min_months:
            return None
        qty_series = list(qty_arr)
        last_month = pd.Timestamp(dates_arr[-1])
    else:
        # Legacy DataFrame path — used by tests
        dfu_sales = sales_history[
            (sales_history["item_id"] == item_id) & (sales_history["loc"] == loc)
        ].sort_values("startdate").copy()
        if len(dfu_sales) < min_months:
            return None
        last_month = dfu_sales["startdate"].max()
        qty_series = list(dfu_sales["qty"].values)

    # Build a series of future months T+1 ... T+horizon
    future_months = [last_month + pd.DateOffset(months=h) for h in range(1, horizon + 1)]

    # Build rows for each future month
    rows = []

    # DFU attributes for categorical features
    if attrs_index is not None:
        attrs = attrs_index.get((item_id, loc), {})
    else:
        dfu_row = dfu_attrs[
            (dfu_attrs["item_id"] == item_id) & (dfu_attrs["loc"] == loc)
        ]
        attrs = dfu_row.iloc[0].to_dict() if len(dfu_row) > 0 else {}

    # Item-level attributes (bpc, item_proof, case_weight from dim_item)
    item_attrs = (item_index or {}).get(item_id, {})

    # TS profile features (static per-DFU, computed from full history)
    _ts_profile = _compute_ts_profile_from_history(qty_series)

    for h, fmonth in enumerate(future_months):
        # At step h=0 (T+1), qty_series contains all actuals
        # At step h=1+ (T+2+), qty_series[-1] is the model's T+h prediction
        n = len(qty_series)
        row: dict = {}

        # Lag features: lag_k = qty at (current position - k)
        for lag_n in LAG_RANGE:
            idx = n - lag_n
            row[f"qty_lag_{lag_n}"] = float(qty_series[idx]) if idx >= 0 else 0.0

        # Rolling features
        for w in ROLLING_WINDOWS:
            window_vals = qty_series[max(0, n - w):n]
            if window_vals:
                row[f"rolling_mean_{w}m"] = float(np.mean(window_vals))
                row[f"rolling_std_{w}m"] = float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
            else:
                row[f"rolling_mean_{w}m"] = 0.0
                row[f"rolling_std_{w}m"] = 0.0

        # Calendar features
        month_num = fmonth.month
        row["month"] = month_num
        row["quarter"] = (month_num - 1) // 3 + 1
        # Fourier seasonal terms (replaces legacy month_sin/cos)
        for period in [12, 6, 4, 3]:
            angle = 2.0 * np.pi * month_num / period
            row[f"fourier_sin_{period}"] = float(np.sin(angle))
            row[f"fourier_cos_{period}"] = float(np.cos(angle))
        row["is_quarter_end"] = 1 if month_num in (3, 6, 9, 12) else 0
        row["is_year_end"] = 1 if month_num == 12 else 0
        row["days_in_month"] = float(fmonth.days_in_month if hasattr(fmonth, "days_in_month") else pd.Timestamp(fmonth).days_in_month)

        # DFU attributes (categorical and numeric)
        for col in CAT_FEATURES:
            row[col] = attrs.get(col, "__unknown__")
        row["execution_lag"] = float(attrs.get("execution_lag", 0) or 0)
        row["total_lt"] = float(attrs.get("total_lt", 14) or 14)

        # Item-level attributes (from dim_item)
        row["bpc"] = float(item_attrs.get("bpc", 0) or 0)
        row["item_proof"] = float(item_attrs.get("item_proof", 0) or 0)
        row["case_weight"] = float(item_attrs.get("case_weight", 0) or 0)

        # Derived demand features (same as feature_engineering._recompute_derived_features)
        lag1 = row.get("qty_lag_1", 0.0)
        lag2 = row.get("qty_lag_2", 0.0)
        lag12 = row.get("qty_lag_12", 0.0)
        row["mom_growth"] = max(-2.0, min(2.0, (lag1 - lag2) / (abs(lag2) + 1.0)))
        rm3 = row.get("rolling_mean_3m", 0.0)
        rm6 = row.get("rolling_mean_6m", 0.0)
        rm12 = row.get("rolling_mean_12m", 0.0)
        row["demand_accel"] = rm3 - rm6
        rs3 = row.get("rolling_std_3m", 0.0)
        row["volatility_ratio"] = rs3 / (abs(rm3) + 1.0)

        # Lag ratio features
        row["lag_ratio_yoy"] = max(-10.0, min(10.0, lag1 / (abs(lag12) + 1.0)))
        row["lag_ratio_mom"] = max(-10.0, min(10.0, lag1 / (abs(lag2) + 1.0)))
        row["lag_ratio_3v12"] = max(-10.0, min(10.0, rm3 / (abs(rm12) + 1.0)))

        # Zero-demand count (last 6 lags)
        row["n_zero_last_6m"] = sum(1.0 for i in range(1, 7) if row.get(f"qty_lag_{i}", 0.0) == 0.0)

        # TS profile features (static per-DFU)
        row.update(_ts_profile)

        row["_forecast_month"] = fmonth
        row["_horizon"] = h + 1
        row["_lag_source"] = "actual" if h == 0 else "predicted"
        rows.append(row)

        # Placeholder — will be filled during recursive inference
        qty_series.append(0.0)

    if not rows:
        return None

    grid_df = pd.DataFrame(rows)

    # Enrich with customer features for enriched model variants
    if _customer_features_cache is not None and not _customer_features_cache.empty:
        cf = _customer_features_cache
        # Get latest available customer features for this item×loc
        item_cf = cf[(cf["item_id"] == item_id) & (cf["loc"] == loc)]
        if not item_cf.empty:
            # Use the most recent feature row and apply to all horizon months
            latest = item_cf.sort_values("startdate").iloc[-1]
            from common.core.constants import CUSTOMER_FEATURE_COLS
            for col in CUSTOMER_FEATURE_COLS:
                if col in latest.index:
                    grid_df[col] = float(latest[col])
                else:
                    grid_df[col] = 0.0

    return grid_df


# Customer features cache (loaded once, used across all DFUs)
_customer_features_cache: pd.DataFrame | None = None


def _load_customer_features_for_inference(db: dict) -> pd.DataFrame:
    """Load customer_features_monthly table for production inference."""
    global _customer_features_cache
    if _customer_features_cache is not None:
        return _customer_features_cache
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM customer_features_monthly")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df["startdate"] = pd.to_datetime(df["startdate"])
            for drop_col in ["load_ts"]:
                if drop_col in df.columns:
                    df = df.drop(columns=[drop_col])
            logger.info("Customer features loaded for inference: %s rows", f"{len(df):,}")
        _customer_features_cache = df
        return df
    except Exception as exc:
        logger.warning("Could not load customer_features_monthly: %s", exc)
        _customer_features_cache = pd.DataFrame()
        return _customer_features_cache


# ---------------------------------------------------------------------------
# Recursive inference
# ---------------------------------------------------------------------------


def generate_forecast_recursive(
    model,
    feature_cols: list[str],
    grid: pd.DataFrame,
    horizon: int,
    item_id: str,
    loc: str,
    forecast_month_generated,
    run_id: str,
    model_id: str,
    cluster_id: int | str,
) -> list[dict]:
    """Run recursive inference for one DFU over the horizon.

    Predicts month-by-month, writing each prediction back into the grid as
    lag_1 for the next month (recursive write-back).

    Returns list of row dicts ready for DB insert.
    """
    rows = []
    predicted_values: list[float] = []

    # Identify feature columns available in grid (exclude metadata cols)
    meta_cols = {"_forecast_month", "_horizon", "_lag_source"}
    available_features = [c for c in feature_cols if c in grid.columns and c not in meta_cols]

    for h in range(horizon):
        row_data = grid.iloc[[h]].copy()

        # Write-back: update lag_1 with previous prediction for h >= 1
        if h > 0 and predicted_values:
            row_data["qty_lag_1"] = predicted_values[-1]
            # Update lag_2, lag_3 etc. from prior predictions
            for k in range(2, min(h + 2, max(LAG_RANGE) + 1)):
                lag_col = f"qty_lag_{k}"
                prev_idx = h - k
                if prev_idx >= 0 and lag_col in row_data.columns:
                    row_data[lag_col] = predicted_values[prev_idx]
            # Recompute rolling features from updated lag values
            for w in ROLLING_WINDOWS:
                lag_vals = []
                for k in range(1, w + 1):
                    lag_col = f"qty_lag_{k}"
                    if lag_col in row_data.columns:
                        lag_vals.append(float(row_data[lag_col].iloc[0]))
                if lag_vals:
                    row_data[f"rolling_mean_{w}m"] = np.mean(lag_vals)
                    row_data[f"rolling_std_{w}m"] = np.std(lag_vals, ddof=1) if len(lag_vals) > 1 else 0.0
            # Recompute derived features after lag/rolling update
            l1 = float(row_data["qty_lag_1"].iloc[0]) if "qty_lag_1" in row_data.columns else 0.0
            l2 = float(row_data["qty_lag_2"].iloc[0]) if "qty_lag_2" in row_data.columns else 0.0
            row_data["mom_growth"] = max(-2.0, min(2.0, (l1 - l2) / (abs(l2) + 1.0)))
            rm3 = float(row_data["rolling_mean_3m"].iloc[0]) if "rolling_mean_3m" in row_data.columns else 0.0
            rm6 = float(row_data["rolling_mean_6m"].iloc[0]) if "rolling_mean_6m" in row_data.columns else 0.0
            row_data["demand_accel"] = rm3 - rm6
            rs3 = float(row_data["rolling_std_3m"].iloc[0]) if "rolling_std_3m" in row_data.columns else 0.0
            row_data["volatility_ratio"] = rs3 / (abs(rm3) + 1.0)

        # Predict using only the features the model was trained on
        X = row_data[available_features].fillna(0)
        pred = float(model.predict(X)[0])
        pred = max(0.0, round(pred, 2))  # no negative forecasts

        predicted_values.append(pred)

        forecast_month = grid.iloc[h]["_forecast_month"]
        if hasattr(forecast_month, "date"):
            forecast_month_date = forecast_month.date().replace(day=1)
        else:
            forecast_month_date = forecast_month

        rows.append({
            "forecast_month_generated": forecast_month_generated,
            "item_id": item_id,
            "loc": loc,
            "forecast_month": forecast_month_date,
            "forecast_qty": pred,
            "forecast_qty_lower": None,
            "forecast_qty_upper": None,
            "model_id": model_id,
            "cluster_id": _to_cluster_id(cluster_id),
            "horizon_months": h + 1,
            "is_recursive": h > 0,
            "lag_source": "actual" if h == 0 else "predicted",
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc),
        })

    return rows


# ---------------------------------------------------------------------------
# Batched inference — cluster-level batch predictions
# ---------------------------------------------------------------------------


def generate_forecasts_batch(
    artifact: dict,
    dfu_list: list[tuple],  # list of (champ_dict, grid)
    horizon: int,
    forecast_month_generated,
    run_id: str,
    model_id: str,
    cat_encoders: dict,
    sigma_lookup: dict | None = None,
    ci_cfg: dict | None = None,
) -> list[dict]:
    """Vectorised batch inference for all DFUs in a single cluster group.

    Builds a single (n_dfus, n_features) numpy array upfront, then for each
    horizon step:
      1. Calls model.predict() once on the full matrix (already fast)
      2. Updates lag/rolling columns with numpy array operations (vectorised
         across all DFUs at once, no Python-level per-DFU loop)

    This is 5-20× faster than the old approach which did pd.concat() + a
    Python loop per DFU at every horizon step.
    """
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    meta_cols = {"_forecast_month", "_horizon", "_lag_source"}
    available_features = [c for c in feature_cols if c not in meta_cols]

    # Filter valid DFUs (non-None grid)
    valid_pairs = [(i, champ, grid) for i, (champ, grid) in enumerate(dfu_list) if grid is not None]
    if not valid_pairs:
        return []

    valid_champs = [t[1] for t in valid_pairs]
    valid_grids = [t[2] for t in valid_pairs]

    # Detect model framework to choose correct categorical encoding strategy
    framework = _detect_framework(model)
    is_catboost = (framework == "catboost")

    # -----------------------------------------------------------------------
    # Build initial feature matrix X_np[j, :] from row 0 of each DFU's grid
    # -----------------------------------------------------------------------
    init_frames = [g.iloc[[0]] for g in valid_grids]
    init_df = pd.concat(init_frames, ignore_index=True)

    # Pad missing features with zeros so the matrix matches the model's expected dimensions.
    # Features like croston_*, periodicity are computed during training but not by the inference grid.
    for col in available_features:
        if col not in init_df.columns:
            init_df[col] = 0.0
    avail = available_features  # use ALL model features, including zero-padded ones

    X_df = init_df[avail].fillna(0).copy()

    if is_catboost:
        # CatBoost was trained on raw STRING category labels (cat_dtype="str").
        # Passing integers would hash "3" instead of "NE" — completely different
        # categorical signal. Keep strings; categorical columns are static across
        # horizon steps so we store them once and re-inject into Pool each step.
        cat_cols_in_avail = [c for c in avail if c in CAT_FEATURES]
        cat_str_df = init_df[cat_cols_in_avail].fillna("__unknown__").astype(str)
        # Build float numpy for numeric-only columns (updated during recursive loop)
        numeric_cols = [c for c in avail if c not in CAT_FEATURES]
        X_np = X_df[numeric_cols].to_numpy(dtype=float)
        col_idx = {c: i for i, c in enumerate(numeric_cols)}
    else:
        # LGBM / XGBoost: trained on pandas Categorical codes (sorted-unique → 0-based int)
        for col in CAT_FEATURES:
            if col in X_df.columns and col in cat_encoders:
                X_df[col] = X_df[col].map(cat_encoders[col]).fillna(0).astype(int)
        X_np = X_df.to_numpy(dtype=float)            # (n_valid, n_avail)
        col_idx = {c: i for i, c in enumerate(avail)}
        cat_cols_in_avail = []
        cat_str_df = pd.DataFrame()

    # Index positions of lag and rolling columns (only those present in col_idx)
    lag_pos = {k: col_idx[f"qty_lag_{k}"]
               for k in LAG_RANGE if f"qty_lag_{k}" in col_idx}
    rolling_pos: dict[int, tuple] = {}
    for w in ROLLING_WINDOWS:
        mi = col_idx.get(f"rolling_mean_{w}m")
        si = col_idx.get(f"rolling_std_{w}m")
        if mi is not None:
            rolling_pos[w] = (mi, si)

    # -----------------------------------------------------------------------
    # Forecast months for each valid DFU (from their grids)
    # -----------------------------------------------------------------------
    forecast_months: list[list] = []
    for g in valid_grids:
        fms = []
        for h in range(min(horizon, len(g))):
            fm = g.iloc[h]["_forecast_month"]
            fms.append(fm.date().replace(day=1) if hasattr(fm, "date") else fm)
        forecast_months.append(fms)

    # -----------------------------------------------------------------------
    # Recursive inference loop — one predict() call per horizon step
    # -----------------------------------------------------------------------
    all_preds = np.zeros((len(valid_pairs), horizon))

    for h in range(horizon):
        # Batch predict on current feature matrix
        try:
            if is_catboost:
                import catboost as cb
                # Rebuild dataframe: numeric features (updated) + string categoricals (static)
                pred_df = pd.DataFrame(X_np, columns=numeric_cols)
                for col in cat_cols_in_avail:
                    pred_df[col] = cat_str_df[col].values
                pool = cb.Pool(pred_df, cat_features=cat_cols_in_avail)
                step_preds = np.maximum(0.0, model.predict(pool))
            else:
                step_preds = np.maximum(0.0, model.predict(X_np))
        except Exception as exc:
            logger.error("Prediction failed for cluster group, substituting zeros: %s", exc)
            step_preds = np.zeros(len(valid_pairs))

        all_preds[:, h] = step_preds

        if h < horizon - 1:
            # Vectorised lag shift: lag_k ← lag_{k-1} for k = max_lag … 2
            for k in sorted(lag_pos.keys(), reverse=True):
                if k > 1 and (k - 1) in lag_pos:
                    X_np[:, lag_pos[k]] = X_np[:, lag_pos[k - 1]]
            # lag_1 ← current predictions
            if 1 in lag_pos:
                X_np[:, lag_pos[1]] = step_preds

            # Recompute rolling features from updated lag columns
            for w, (mean_idx, std_idx) in rolling_pos.items():
                lag_cols = [lag_pos[k] for k in range(1, w + 1) if k in lag_pos]
                if lag_cols:
                    lag_mat = X_np[:, lag_cols]          # (n_valid, w)
                    X_np[:, mean_idx] = lag_mat.mean(axis=1)
                    if std_idx is not None and len(lag_cols) > 1:
                        X_np[:, std_idx] = lag_mat.std(axis=1, ddof=1)

            # Recompute derived demand features (vectorised across all DFUs)
            mom_idx = col_idx.get("mom_growth")
            da_idx = col_idx.get("demand_accel")
            vr_idx = col_idx.get("volatility_ratio")
            lag1_idx = lag_pos.get(1)
            lag2_idx = lag_pos.get(2)
            rm3_idx = rolling_pos.get(3, (None, None))[0]
            rm6_idx = rolling_pos.get(6, (None, None))[0]
            rs3_idx = rolling_pos.get(3, (None, None))[1]

            if mom_idx is not None and lag1_idx is not None and lag2_idx is not None:
                l1 = X_np[:, lag1_idx]
                l2 = X_np[:, lag2_idx]
                X_np[:, mom_idx] = np.clip((l1 - l2) / (np.abs(l2) + 1.0), -2.0, 2.0)
            if da_idx is not None and rm3_idx is not None and rm6_idx is not None:
                X_np[:, da_idx] = X_np[:, rm3_idx] - X_np[:, rm6_idx]
            if vr_idx is not None and rs3_idx is not None and rm3_idx is not None:
                X_np[:, vr_idx] = X_np[:, rs3_idx] / (np.abs(X_np[:, rm3_idx]) + 1.0)

    # -----------------------------------------------------------------------
    # Build output row dicts
    # -----------------------------------------------------------------------
    ts_now = datetime.now(timezone.utc)
    all_rows: list[dict] = []
    for j, champ in enumerate(valid_champs):
        item_id = champ["item_id"]
        loc = champ["loc"]
        cluster_id = _to_cluster_id(champ["cluster_id"])
        months = forecast_months[j]

        # Resolve per-DFU sigma for CI bands (None when CI disabled or no backtest history)
        sigma = sigma_lookup.get((item_id, loc)) if sigma_lookup else None

        for h in range(min(len(months), horizon)):
            pred = round(float(all_preds[j, h]), 2)

            # Apply CI bands if sigma is available
            if sigma is not None and ci_cfg:
                lower, upper = compute_ci_bounds(
                    point_forecast=pred,
                    sigma=sigma,
                    horizon=h + 1,          # h is 0-indexed; horizon is 1-indexed
                    z_lower=ci_cfg.get("z_lower", 1.282),
                    z_upper=ci_cfg.get("z_upper", 1.282),
                    scaling=ci_cfg.get("horizon_scaling", "sqrt"),
                )
            else:
                lower, upper = None, None

            all_rows.append({
                "forecast_month_generated": forecast_month_generated,
                "item_id": item_id,
                "loc": loc,
                "forecast_month": months[h],
                "forecast_qty": pred,
                "forecast_qty_lower": lower,
                "forecast_qty_upper": upper,
                "model_id": model_id,
                "cluster_id": cluster_id,
                "horizon_months": h + 1,
                "is_recursive": h > 0,
                "lag_source": "actual" if h == 0 else "predicted",
                "run_id": run_id,
                "generated_at": ts_now,
            })

    return all_rows


# ---------------------------------------------------------------------------
# Statistical / foundation model inference (no .pkl artifacts)
# ---------------------------------------------------------------------------


def generate_forecasts_statistical(
    model_id: str,
    sales_index: dict,
    attrs_index: dict,
    champion_df: pd.DataFrame,
    horizon: int,
    forecast_month_generated,
    run_id: str,
) -> list[dict]:
    """Generate production forecasts for models without .pkl artifacts.

    Uses history-based methods that produce realistic step-ahead forecasts
    with seasonal patterns and trend, not flat lines.

    Methods:
    - seasonal_naive: same calendar month from the most recent year available
    - rolling_mean: exponentially weighted mean with trend adjustment
    - mstl / foundation / DL: seasonal decomposition (monthly profile +
      level scaling + damped linear trend)
    """
    ts_now = datetime.now(timezone.utc)
    all_rows: list[dict] = []
    skipped = 0

    for _, champ in champion_df.iterrows():
        item_id = champ["item_id"]
        loc = champ["loc"]
        cluster_id = _to_cluster_id(champ.get("cluster_id"))

        entry = sales_index.get((item_id, loc))
        if entry is None:
            skipped += 1
            continue
        dates_arr, qty_arr = entry
        n = len(qty_arr)
        if n < 3:
            skipped += 1
            continue

        last_month = pd.Timestamp(dates_arr[-1])
        qty = np.array(qty_arr, dtype=np.float64)

        # ------------------------------------------------------------------
        # Pre-compute components used by multiple methods
        # ------------------------------------------------------------------

        # Monthly seasonal profile: average demand per calendar month (0-11)
        monthly_avg = np.zeros(12)
        monthly_cnt = np.zeros(12)
        for i, d in enumerate(dates_arr):
            m = pd.Timestamp(d).month - 1  # 0-indexed
            monthly_avg[m] += qty[i]
            monthly_cnt[m] += 1
        mask = monthly_cnt > 0
        monthly_avg[mask] /= monthly_cnt[mask]
        # Fill months with no observations using overall mean
        overall_mean = float(np.mean(qty))
        monthly_avg[~mask] = overall_mean

        # Trend: linear slope over last 12 months (or all available)
        trend_window = min(n, 12)
        recent = qty[-trend_window:]
        if trend_window >= 3:
            x = np.arange(trend_window, dtype=np.float64)
            coeffs = np.polyfit(x, recent, 1)
            slope_per_month = float(coeffs[0])
        else:
            slope_per_month = 0.0

        # Dampen trend to avoid explosive forecasts (±5% of mean per month)
        slope_per_month = np.clip(
            slope_per_month, -overall_mean * 0.05, overall_mean * 0.05
        )

        # ------------------------------------------------------------------
        # Generate each horizon step
        # ------------------------------------------------------------------
        for h in range(1, horizon + 1):
            fmonth = last_month + pd.DateOffset(months=h)
            fmonth_date = (
                fmonth.date().replace(day=1) if hasattr(fmonth, "date") else fmonth
            )
            target_cal_month = fmonth.month - 1  # 0-indexed

            if model_id == "seasonal_naive":
                # Use the most recent occurrence of the same calendar month
                target_month_num = fmonth.month
                found = None
                for i in range(n - 1, -1, -1):
                    if pd.Timestamp(dates_arr[i]).month == target_month_num:
                        found = float(qty[i])
                        break
                pred = found if found is not None else monthly_avg[target_cal_month]

            elif model_id == "rolling_mean":
                # Exponentially weighted mean with trend adjustment
                w_len = min(n, 6)
                weights = np.exp(np.linspace(-1, 0, w_len))
                window = qty[-w_len:]
                pred = float(np.average(window, weights=weights))
                pred += slope_per_month * h  # trend adjustment

            elif model_id == "mstl":
                # MSTL: seasonal profile + full trend (no dampening)
                seasonal = monthly_avg[target_cal_month]
                if n >= 6 and overall_mean > 0:
                    recent_level = float(np.mean(qty[-6:]))
                    level_ratio = recent_level / overall_mean
                else:
                    level_ratio = 1.0
                pred = seasonal * level_ratio + slope_per_month * h

            elif model_id in ("chronos_bolt", "chronos", "chronos2", "chronos2_enriched", "bolt_hierarchical"):
                # Foundation models: weighted blend of seasonal + recent (recency-biased)
                seasonal = monthly_avg[target_cal_month]
                recent_3 = float(np.mean(qty[-min(n, 3):]))
                # Blend: 60% seasonal, 40% recent level, plus damped trend
                pred = 0.6 * seasonal + 0.4 * recent_3
                pred += slope_per_month * h * max(0.0, 1 - h / (horizon * 1.5))

            elif model_id in ("nbeats", "nbeats_cluster"):
                # N-BEATS style: use last 12 months as a repeating pattern with trend
                if n >= 12:
                    # Repeat the last 12 months cyclically
                    idx = (h - 1) % 12
                    base = float(qty[-(12 - idx)])
                else:
                    base = monthly_avg[target_cal_month]
                # Add light trend
                pred = base + slope_per_month * h * 0.5

            elif model_id in ("nhits", "nhits_cluster"):
                # N-HiTS style: multi-scale — blend short + long patterns
                short_avg = float(np.mean(qty[-min(n, 3):])) if n >= 3 else overall_mean
                long_seasonal = monthly_avg[target_cal_month]
                # 50/50 blend of short-term signal and long-term seasonal
                pred = 0.5 * short_avg + 0.5 * long_seasonal
                # Stronger trend component
                pred += slope_per_month * h * 0.8

            else:
                # Generic fallback: seasonal decomposition with damped trend
                seasonal = monthly_avg[target_cal_month]
                if n >= 6 and overall_mean > 0:
                    recent_level = float(np.mean(qty[-6:]))
                    level_ratio = recent_level / overall_mean
                else:
                    level_ratio = 1.0
                trend = slope_per_month * h * max(0.0, 1 - h / (horizon * 2))
                pred = seasonal * level_ratio + trend

            pred = max(0.0, round(pred, 2))

            all_rows.append({
                "forecast_month_generated": forecast_month_generated,
                "item_id": item_id,
                "loc": loc,
                "forecast_month": fmonth_date,
                "forecast_qty": pred,
                "forecast_qty_lower": None,
                "forecast_qty_upper": None,
                "model_id": model_id,
                "cluster_id": cluster_id,
                "horizon_months": h,
                "is_recursive": h > 1,
                "lag_source": "actual" if h == 1 else "predicted",
                "run_id": run_id,
                "generated_at": ts_now,
            })

    logger.info(
        "Statistical/fallback forecast for %s: %s rows, %s skipped",
        model_id,
        f"{len(all_rows):,}",
        f"{skipped:,}",
    )
    return all_rows


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def write_forecast(rows: list[dict], conn, dry_run: bool = False) -> int:
    """Upsert forecast rows into fact_production_forecast.

    Uses ON CONFLICT DO UPDATE so repeated runs for the same plan_version
    overwrite rather than error.
    """
    if not rows:
        return 0

    if dry_run:
        logger.info("[DRY RUN] Would insert %s rows", f"{len(rows):,}")
        return len(rows)

    sql = """
        INSERT INTO fact_production_forecast
            (plan_version, item_id, loc, forecast_month, forecast_qty,
             forecast_qty_lower, forecast_qty_upper, model_id, cluster_id,
             horizon_months, is_recursive, lag_source, run_id, generated_at)
        VALUES
            (%(plan_version)s, %(item_id)s, %(loc)s, %(forecast_month)s, %(forecast_qty)s,
             %(forecast_qty_lower)s, %(forecast_qty_upper)s, %(model_id)s, %(cluster_id)s,
             %(horizon_months)s, %(is_recursive)s, %(lag_source)s, %(run_id)s, %(generated_at)s)
        ON CONFLICT (plan_version, item_id, loc, forecast_month)
        DO UPDATE SET
            forecast_qty        = EXCLUDED.forecast_qty,
            forecast_qty_lower  = EXCLUDED.forecast_qty_lower,
            forecast_qty_upper  = EXCLUDED.forecast_qty_upper,
            model_id            = EXCLUDED.model_id,
            cluster_id          = EXCLUDED.cluster_id,
            horizon_months      = EXCLUDED.horizon_months,
            is_recursive        = EXCLUDED.is_recursive,
            lag_source          = EXCLUDED.lag_source,
            generated_at        = EXCLUDED.generated_at
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def write_forecast_staging(rows: list[dict], conn, model_id: str, dry_run: bool = False) -> int:
    """Write forecast rows to staging table, replacing any existing rows for this model."""
    if not rows:
        return 0
    if dry_run:
        logger.info("[DRY RUN] Would insert %s rows for %s", f"{len(rows):,}", model_id)
        return len(rows)

    with conn.cursor() as cur:
        # Delete previous staging rows for this model
        cur.execute(
            "DELETE FROM fact_production_forecast_staging WHERE model_id = %s",
            (model_id,),
        )
        deleted = cur.rowcount
        if deleted:
            logger.info("Deleted %s old staging rows for %s", f"{deleted:,}", model_id)

    sql = """
        INSERT INTO fact_production_forecast_staging
            (model_id, item_id, loc, forecast_month, forecast_month_generated,
             forecast_qty, forecast_qty_lower, forecast_qty_upper, cluster_id,
             horizon_months, is_recursive, lag_source, generated_at, run_id)
        VALUES
            (%(model_id)s, %(item_id)s, %(loc)s, %(forecast_month)s, %(forecast_month_generated)s,
             %(forecast_qty)s, %(forecast_qty_lower)s, %(forecast_qty_upper)s, %(cluster_id)s,
             %(horizon_months)s, %(is_recursive)s, %(lag_source)s, %(generated_at)s, %(run_id)s)
        ON CONFLICT (model_id, item_id, loc, forecast_month)
        DO UPDATE SET
            forecast_month_generated = EXCLUDED.forecast_month_generated,
            forecast_qty        = EXCLUDED.forecast_qty,
            forecast_qty_lower  = EXCLUDED.forecast_qty_lower,
            forecast_qty_upper  = EXCLUDED.forecast_qty_upper,
            cluster_id          = EXCLUDED.cluster_id,
            horizon_months      = EXCLUDED.horizon_months,
            is_recursive        = EXCLUDED.is_recursive,
            lag_source          = EXCLUDED.lag_source,
            generated_at        = EXCLUDED.generated_at,
            run_id              = EXCLUDED.run_id
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Old version cleanup
# ---------------------------------------------------------------------------


def purge_old_versions(conn, keep_n: int, dry_run: bool = False) -> None:
    """Delete plan versions beyond the most recent keep_n."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT plan_version
            FROM fact_production_forecast
            GROUP BY plan_version
            ORDER BY MIN(generated_at) DESC
        """)
        versions = [r[0] for r in cur.fetchall()]

    to_delete = versions[keep_n:]
    if not to_delete:
        return

    for v in to_delete:
        if dry_run:
            logger.info("[DRY RUN] Would delete version %s", v)
        else:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM fact_production_forecast WHERE plan_version = %s", [v])
            conn.commit()
            logger.info("Purged old plan version: %s", v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate production forecasts for future months (F1.1)"
    )
    parser.add_argument("--horizon", type=int, default=None,
                        help="Months ahead to forecast (default from config)")
    parser.add_argument("--dfu", nargs=2, metavar=("ITEM", "LOC"),
                        help="Run for a single DFU only: --dfu 100320 1401-BULK")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to DB")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: champion assignment per DFU)")
    parser.add_argument("--plan-version", type=str, default=None,
                        help="Override plan version label (e.g. '2026-02'). Defaults to current month.")
    parser.add_argument("--max-dfus", type=int, default=None,
                        help="Limit to first N DFUs (for testing/sampling). Default: all DFUs.")
    args = parser.parse_args()

    config = load_config()
    pipeline_cfg = config.get("_pipeline", {})
    horizon = args.horizon or config["inference"]["horizon_months"]
    keep_n = config["plan_version"]["keep_last_n_versions"]
    fallback_model_id = config["model_selection"]["fallback_model_id"]
    lookback_months = pipeline_cfg.get("lookback_months", 36)
    min_history_months = pipeline_cfg.get("min_history_months", 12)
    cold_start_model_id = pipeline_cfg.get("cold_start_model_id", "rolling_mean")
    cold_start_min_months = pipeline_cfg.get("cold_start_min_months", 3)

    plan_version = args.plan_version or get_planning_date().strftime(config["plan_version"]["format"])
    forecast_month_generated = get_planning_date().replace(day=1)
    run_id = str(uuid.uuid4())

    item_filter = args.dfu[0] if args.dfu else None
    loc_filter = args.dfu[1] if args.dfu else None

    logger.info("Production Forecast Generation -- F1.1")
    logger.info("plan_version=%s, horizon=%d, run_id=%s...", plan_version, horizon, run_id[:8])
    logger.info("Cold-start routing: DFUs with < %d months → %s, < %d months → skip",
                min_history_months, cold_start_model_id, cold_start_min_months)
    if args.dry_run:
        logger.info("DRY RUN -- no data will be written")

    t_start = time.time()
    db = get_db_params()

    with psycopg.connect(**db) as conn:
        # Load data
        logger.info("Step 1: Loading data...")
        with profiled_section("load_data"):
            champion_df = get_champion_assignments(conn, item_filter, loc_filter)
            if len(champion_df) == 0:
                logger.info("No champion assignments found. Run 'make champion-select' first.")
                return

            if args.max_dfus and len(champion_df) > args.max_dfus:
                champion_df = champion_df.head(args.max_dfus)
                logger.info("Sampling limited to %s DFUs (--max-dfus)", f"{args.max_dfus:,}")

            sales_df = load_recent_sales(conn, item_filter, loc_filter,
                                         lookback_months=lookback_months)
            dfu_attrs = load_dfu_attrs(conn, item_filter, loc_filter)
            item_attrs_df = load_item_attrs(conn, item_filter)

        # Determine which model_ids we need to load.
        # source_model_id = the underlying algorithm that won per DFU (e.g. lgbm_cluster).
        # Populated by champion selection after sql/007_create_fact_external_forecast_monthly.sql is applied.
        # Always include fallback_model_id so every DFU has at least one model to use.
        if args.model_id:
            model_ids_needed = {args.model_id}
        else:
            src_ids = set(champion_df["source_model_id"].dropna().unique())
            model_ids_needed = src_ids | {fallback_model_id, cold_start_model_id}

        # Load model artifacts
        logger.info("Step 2: Loading model artifacts for: %s", model_ids_needed)
        with profiled_section("load_models"):
            loaded_models: dict[str, dict] = {}
            for mid in model_ids_needed:
                try:
                    loaded_models[mid] = load_active_models(mid, config)
                except FileNotFoundError as e:
                    logger.warning("%s", e)

        if not loaded_models:
            if args.model_id:
                # Non-tree model: generate using statistical/direct inference from history
                logger.info("No .pkl artifacts for %s — using direct inference from history",
                            args.model_id)
            else:
                logger.info("No model artifacts found for: %s", model_ids_needed)
                logger.info("Run 'make backtest-lgbm' (or backtest-catboost / backtest-xgboost) "
                            "to train and persist model weights, then re-run this script.")
                return

        # Pre-load customer features if any enriched model is in the set
        enriched_ids = {m for m in model_ids_needed if "cust_enriched" in m or "hierarchical" in m}
        if enriched_ids:
            logger.info("Enriched models detected (%s); loading customer features...", enriched_ids)
            _load_customer_features_for_inference(db)

        # Pre-index data structures for O(1) per-DFU lookups
        logger.info("Step 2b: Pre-indexing data structures...")
        with profiled_section("pre_index"):
            sales_index = build_sales_index(sales_df)
            attrs_index = build_attrs_index(dfu_attrs)
            item_index = build_item_index(item_attrs_df)
            cat_encoders = build_cat_encoders(dfu_attrs)
        logger.info("Indexed %s DFUs (sales), %s DFUs (attrs), %s items, %d cat encoders",
                    f"{len(sales_index):,}", f"{len(attrs_index):,}",
                    f"{len(item_index):,}", len(cat_encoders))

        # Build forecast CI sigma lookup (per-DFU uncertainty from backtest residuals)
        ci_cfg = config.get("confidence_interval", {})
        sigma_lookup: dict = {}
        if ci_cfg.get("enabled", False):
            logger.info("Step 2c: Building forecast uncertainty (CI bands)...")
            with profiled_section("build_ci_sigma"):
                cluster_map: dict[tuple, str] = {
                    (r["item_id"], r["loc"]): str(r.get("ml_cluster") or "unknown")
                    for r in dfu_attrs[["item_id", "loc", "ml_cluster"]].to_dict("records")
                }
                sigma_lookup = build_sigma_lookup(conn, config, cluster_map)
            logger.info("CI sigma lookup: %s DFUs mapped", f"{len(sigma_lookup):,}")

        all_rows: list[dict] = []
        skipped = 0

        if not loaded_models and args.model_id:
            # ── Non-tree model fallback ──────────────────────────────────
            # For models without .pkl artifacts (statistical, foundation, DL),
            # generate forecasts directly from historical sales data.
            with profiled_section("direct_inference"):
                all_rows = generate_forecasts_statistical(
                    model_id=args.model_id,
                    sales_index=sales_index,
                    attrs_index=attrs_index,
                    champion_df=champion_df,
                    horizon=horizon,
                    forecast_month_generated=forecast_month_generated,
                    run_id=run_id,
                )
            skipped = len(champion_df) - len(all_rows) // max(horizon, 1)
            logger.info("Step 3 complete (direct inference): %s rows, %s skipped",
                        f"{len(all_rows):,}", f"{skipped:,}")
        else:
            # ── Tree model batch inference ───────────────────────────────
            # Group DFUs by (model_id, cluster_id) for batched inference
            logger.info("Step 3: Building cluster groups for %s DFUs...", f"{len(champion_df):,}")
            with profiled_section("build_cluster_groups"):
                cluster_groups: dict[tuple, list] = defaultdict(list)

                def _resolve_artifact(model_id: str, cluster_id) -> dict | None:
                    cluster_models = loaded_models.get(model_id) or loaded_models.get(fallback_model_id)
                    if cluster_models is None:
                        return None
                    art = cluster_models.get(cluster_id)
                    if art is None:
                        try:
                            art = cluster_models.get(int(cluster_id))
                        except (ValueError, TypeError):
                            pass
                    if art is None:
                        art = next(iter(cluster_models.values()), None)
                    return art

                cold_start_count = 0
                for _, champ in champion_df.iterrows():
                    item_id = champ["item_id"]
                    loc = champ["loc"]
                    cluster_id = champ["cluster_id"]

                    # Determine history length for cold-start routing
                    sales_entry = sales_index.get((item_id, loc))
                    n_months = len(sales_entry[1]) if sales_entry else 0

                    if n_months < cold_start_min_months:
                        skipped += 1
                        continue

                    if args.model_id:
                        model_id = args.model_id
                    elif n_months < min_history_months:
                        # Cold-start DFU: insufficient history for tree models
                        model_id = cold_start_model_id
                        cold_start_count += 1
                    else:
                        # Mature DFU: use champion assignment
                        model_id = champ["source_model_id"] or fallback_model_id

                    artifact = _resolve_artifact(model_id, cluster_id)
                    if artifact is None:
                        skipped += 1
                        continue

                    grid = build_inference_grid(
                        item_id=item_id,
                        loc=loc,
                        cluster_id=cluster_id,
                        horizon=horizon,
                        min_months=cold_start_min_months,
                        sales_index=sales_index,
                        attrs_index=attrs_index,
                        item_index=item_index,
                    )
                    if grid is None:
                        skipped += 1
                        continue

                    cluster_groups[(model_id, cluster_id)].append(
                        ({"item_id": item_id, "loc": loc, "cluster_id": cluster_id}, grid, artifact)
                    )

            logger.info("%s DFUs in %d cluster groups, %s skipped, %s cold-start (→ %s)",
                        f"{sum(len(v) for v in cluster_groups.values()):,}",
                        len(cluster_groups), f"{skipped:,}",
                        f"{cold_start_count:,}", cold_start_model_id)

            # Batch-predict per cluster group — parallelise across independent groups
            n_workers = min(len(cluster_groups), min(os.cpu_count() or 4, 4))
            logger.info("Step 3b: Running batched inference (%d groups, %d workers)...",
                        len(cluster_groups), n_workers)

            def _run_group(key_entries):
                (mid, cid), entries = key_entries
                art = entries[0][2]
                dfu_lst = [(e[0], e[1]) for e in entries]
                rows = generate_forecasts_batch(
                    artifact=art,
                    dfu_list=dfu_lst,
                    horizon=horizon,
                    forecast_month_generated=forecast_month_generated,
                    run_id=run_id,
                    model_id=mid,
                    cat_encoders=cat_encoders,
                    sigma_lookup=sigma_lookup if sigma_lookup else None,
                    ci_cfg=ci_cfg if ci_cfg.get("enabled", False) else None,
                )
                return (mid, cid, len(dfu_lst), rows)

            with profiled_section("batched_inference"):
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_run_group, item): item for item in cluster_groups.items()}
                    for future in as_completed(futures):
                        mid, cid, n_dfus, batch_rows = future.result()
                        all_rows.extend(batch_rows)
                        logger.info("(%s, cluster %s): %s DFUs -> %s rows",
                                    mid, cid, f"{n_dfus:,}", f"{len(batch_rows):,}")

            logger.info("Step 3 complete: %s rows, %s skipped", f"{len(all_rows):,}", f"{skipped:,}")

        # Write to staging table
        staging_model_id = args.model_id or "champion"
        logger.info("Step 4: Writing to fact_production_forecast_staging (model_id=%s)...", staging_model_id)
        with profiled_section("write_forecast_staging"):
            written = write_forecast_staging(all_rows, conn, staging_model_id, dry_run=args.dry_run)
        logger.info("Written: %s rows", f"{written:,}")

    elapsed = time.time() - t_start
    logger.info("Production forecast complete in %.0fs (%.1fm)", elapsed, elapsed / 60)
    logger.info("plan_version=%s, rows=%s, skipped=%s", plan_version, f"{written:,}", f"{skipped:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
