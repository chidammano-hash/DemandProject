"""Shared feature engineering for tree-based backtest models (LGBM, CatBoost, XGBoost).

Builds the full (dfu_ck × month) feature matrix with lag, rolling, calendar,
and attribute features. Used by all tree-based backtest scripts.
"""

import time

import numpy as np
import pandas as pd

from common.constants import (
    CAT_FEATURES,
    LAG_RANGE,
    METADATA_COLS,
    NUMERIC_DFU_FEATURES,
    NUMERIC_ITEM_FEATURES,
    ROLLING_WINDOWS,
)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def build_feature_matrix(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    all_months: list[pd.Timestamp],
    cat_dtype: str = "category",
) -> pd.DataFrame:
    """Build FULL feature matrix: one row per (dfu_ck, month).

    Features are strictly causal — only data from months < target month used.
    Built ONCE for all timeframes; per-timeframe masking done externally.

    Args:
        cat_dtype: Dtype for categorical features.
            "category" for LightGBM/XGBoost, "str" for CatBoost.
    """
    t0 = time.time()
    dfu_keys = dfu_attrs[["dfu_ck", "dmdunit", "dmdgroup", "loc"]].drop_duplicates()
    n_dfus = len(dfu_keys)
    n_months = len(all_months)
    print(f"  [{_ts()}] Building grid: {n_dfus:,} DFUs × {n_months} months = {n_dfus * n_months:,} rows")

    # Build complete grid via MultiIndex (faster than cross-join merge)
    idx = pd.MultiIndex.from_product(
        [dfu_keys["dfu_ck"].values, all_months],
        names=["dfu_ck", "startdate"],
    )
    grid = pd.DataFrame(index=idx).reset_index()
    grid = grid.merge(dfu_keys, on="dfu_ck", how="left")
    print(f"  [{_ts()}] Grid built: {len(grid):,} rows ({time.time() - t0:.1f}s)")

    # Join sales (full — no cutoff; masking done per timeframe)
    t1 = time.time()
    grid = grid.merge(
        sales_df[["dfu_ck", "startdate", "qty"]],
        on=["dfu_ck", "startdate"],
        how="left",
    )
    grid["qty"] = grid["qty"].fillna(0)
    print(f"  [{_ts()}] Sales joined ({time.time() - t1:.1f}s)")

    # Sort for lag/rolling operations
    t1 = time.time()
    grid = grid.sort_values(["dfu_ck", "startdate"]).reset_index(drop=True)
    print(f"  [{_ts()}] Sorted ({time.time() - t1:.1f}s)")

    # Lag features (vectorized groupby shift)
    t1 = time.time()
    g = grid.groupby("dfu_ck", sort=False)["qty"]
    for lag_n in LAG_RANGE:
        grid[f"qty_lag_{lag_n}"] = g.shift(lag_n)
    print(f"  [{_ts()}] Lag features 1-12 done ({time.time() - t1:.1f}s)")

    # Rolling stats (vectorized — no lambda)
    t1 = time.time()
    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        rolling = shifted.groupby(grid["dfu_ck"], sort=False).rolling(w, min_periods=1)
        grid[f"rolling_mean_{w}m"] = rolling.mean().reset_index(level=0, drop=True)
        grid[f"rolling_std_{w}m"] = rolling.std().fillna(0).reset_index(level=0, drop=True)
    print(f"  [{_ts()}] Rolling stats done ({time.time() - t1:.1f}s)")

    # Calendar features
    grid["month"] = grid["startdate"].dt.month
    grid["quarter"] = grid["startdate"].dt.quarter
    grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
    grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)

    # DFU attributes
    t1 = time.time()
    dfu_feat_cols = ["dfu_ck"] + CAT_FEATURES + NUMERIC_DFU_FEATURES
    dfu_feat_cols = [c for c in dfu_feat_cols if c in dfu_attrs.columns]
    grid = grid.merge(dfu_attrs[dfu_feat_cols], on="dfu_ck", how="left")

    # Item attributes
    if len(item_attrs) > 0:
        grid = grid.merge(item_attrs, on="dmdunit", how="left")
    print(f"  [{_ts()}] Attributes joined ({time.time() - t1:.1f}s)")

    # Fill missing numerics
    for col in NUMERIC_DFU_FEATURES + NUMERIC_ITEM_FEATURES:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)

    # Set categorical dtypes
    for col in CAT_FEATURES:
        if col in grid.columns:
            if cat_dtype == "str":
                grid[col] = grid[col].fillna("__unknown__").astype(str)
            else:
                grid[col] = grid[col].fillna("__unknown__").astype("category")

    print(f"  [{_ts()}] Feature matrix complete: {grid.shape} ({time.time() - t0:.1f}s total)")
    return grid


def get_feature_columns(grid: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except metadata/target)."""
    return [c for c in grid.columns if c not in METADATA_COLS]


def mask_future_sales(grid: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Zero out qty and recompute lag/rolling features for rows after cutoff.

    Instead of rebuilding the whole grid, we mask the qty column and
    recompute only the affected features. This is much faster than
    rebuilding from scratch.
    """
    df = grid.copy()

    # Mask future sales
    future_mask = df["startdate"] > cutoff
    df.loc[future_mask, "qty"] = 0

    # Recompute lags and rolling on the masked data
    g = df.groupby("dfu_ck", sort=False)["qty"]
    for lag_n in LAG_RANGE:
        df[f"qty_lag_{lag_n}"] = g.shift(lag_n)

    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        rolling = shifted.groupby(df["dfu_ck"], sort=False).rolling(w, min_periods=1)
        df[f"rolling_mean_{w}m"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}m"] = rolling.std().fillna(0).reset_index(level=0, drop=True)

    return df
