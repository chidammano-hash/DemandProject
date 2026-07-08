"""Shared feature engineering for tree-based backtest models (LGBM, CatBoost, XGBoost).

Builds the full (sku_ck × month) feature matrix with lag, rolling, calendar,
and attribute features. Used by all tree-based backtest scripts.

Feature groups (in build order):
  1. Lag + rolling (qty_lag_*, rolling_mean_*, rolling_std_*)
  2. Calendar (month, quarter, is_quarter_end, is_year_end, days_in_month)
  3. Fourier seasonal terms (sin/cos for periods 12, 6, 4, 3 — includes period-12 which replaces legacy month_sin/cos)
  4. Derived demand (mom_growth, volatility_ratio, lag ratios, etc.)
  5. Croston decomposition (intermittent demand: size, interval, probability)
  6. TS profile (per-DFU static: cv_demand, adi, seasonal_amplitude, etc.)
  7. External forecast signal (optional enrichment via enrich_with_external_forecast)
"""

import logging
import time

import numpy as np
import pandas as pd

from common.core.constants import (
    CAT_FEATURES,
    CUSTOMER_FEATURE_COLS,
    ENHANCED_FEATURES,
    EXTERNAL_FORECAST_FEATURES,
    FORECAST_QTY_COL,
    LAG_RANGE,
    METADATA_COLS,
    NUMERIC_ITEM_FEATURES,
    NUMERIC_SKU_FEATURES,
    ROLLING_WINDOWS,
    TS_PROFILE_FEATURES,
)

logger = logging.getLogger(__name__)


def _recompute_derived_features(df: pd.DataFrame) -> None:
    """Recompute derived demand features in-place after lag/rolling features are updated.

    Called after any qty mutation (masking, recursive write-back). Centralises
    the logic to prevent three-way duplication across build_feature_matrix,
    mask_future_sales, and update_grid_with_predictions.

    Also recomputes Croston decomposition when those columns already exist in
    the DataFrame (i.e., they were previously computed during build_feature_matrix).
    """
    df["mom_growth"] = (df["qty_lag_1"] - df["qty_lag_2"]) / (df["qty_lag_2"].abs() + 1.0)
    df["mom_growth"] = df["mom_growth"].clip(-2.0, 2.0)
    df["demand_accel"] = df["rolling_mean_3m"] - df["rolling_mean_6m"]
    df["volatility_ratio"] = df["rolling_std_3m"] / (df["rolling_mean_3m"].abs() + 1.0)

    # Lag ratio features — capture relative change more directly than raw lags
    df["lag_ratio_yoy"] = df["qty_lag_1"] / (df["qty_lag_12"].abs() + 1.0)
    df["lag_ratio_yoy"] = df["lag_ratio_yoy"].clip(-10.0, 10.0)
    df["lag_ratio_mom"] = df["qty_lag_1"] / (df["qty_lag_2"].abs() + 1.0)
    df["lag_ratio_mom"] = df["lag_ratio_mom"].clip(-10.0, 10.0)
    df["lag_ratio_3v12"] = df["rolling_mean_3m"] / (df["rolling_mean_12m"].abs() + 1.0)
    df["lag_ratio_3v12"] = df["lag_ratio_3v12"].clip(-10.0, 10.0)

    # Zero-demand count in recent 6 months (intermittency signal)
    lag_cols_6m = [f"qty_lag_{i}" for i in range(1, 7)]
    existing_lag_cols = [c for c in lag_cols_6m if c in df.columns]
    if existing_lag_cols:
        df["n_zero_last_6m"] = (df[existing_lag_cols] == 0).sum(axis=1).astype(np.float32)

    # Recompute Croston features if they were previously computed
    if "croston_demand_size" in df.columns:
        _compute_croston_features(df)

    # Cross-DFU cluster aggregates intentionally stay removed to avoid ml_cluster leakage.


def _compute_rolling_numpy(qty_2d: np.ndarray, windows: list[int]) -> dict[str, np.ndarray]:
    """Compute rolling mean and std using numpy cumsum on a (n_skus, n_months) matrix.

    ~10x faster than pandas groupby rolling for uniform-sized groups.
    Uses shifted-by-1 values (causal: only past data) with min_periods=1.
    """
    n_skus, n_months = qty_2d.shape
    # Shift by 1 (causal): shifted[:, t] = qty[:, t-1]
    shifted = np.empty_like(qty_2d)
    shifted[:, 0] = np.nan
    shifted[:, 1:] = qty_2d[:, :-1]

    # Replace NaN with 0 for cumsum, track valid counts
    valid = ~np.isnan(shifted)
    filled = np.where(valid, shifted, 0.0)
    filled_sq = filled * filled

    cumsum = np.cumsum(filled, axis=1)
    cumsum_sq = np.cumsum(filled_sq, axis=1)
    cumcount = np.cumsum(valid.astype(np.float32), axis=1)

    result: dict[str, np.ndarray] = {}
    for w in windows:
        # Rolling sum/count via cumsum difference
        rsum = cumsum.copy()
        rsum_sq = cumsum_sq.copy()
        rcount = cumcount.copy()
        if w < n_months:
            rsum[:, w:] = cumsum[:, w:] - cumsum[:, :-w]
            rsum_sq[:, w:] = cumsum_sq[:, w:] - cumsum_sq[:, :-w]
            rcount[:, w:] = cumcount[:, w:] - cumcount[:, :-w]

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_vals = np.where(rcount > 0, rsum / rcount, np.nan)
            # Sample variance (ddof=1) to match pandas .std()
            variance = np.where(
                rcount > 1,
                (rsum_sq - rsum * rsum / rcount) / (rcount - 1),
                0.0,
            )
            variance = np.maximum(variance, 0.0)  # numerical stability
            std_vals = np.sqrt(variance)

        result[f"rolling_mean_{w}m"] = mean_vals.ravel()
        result[f"rolling_std_{w}m"] = np.nan_to_num(std_vals).ravel()

    return result


def _compute_lags_and_rolling(df: pd.DataFrame, group_col: str = "sku_ck") -> None:
    """Compute lag and rolling features in-place on a sorted DataFrame.

    Uses numpy 2D reshape for uniform grids (cross-product of skus × months)
    which is ~10x faster than pandas groupby rolling. Falls back to pandas
    for non-uniform group sizes.
    """
    g = df.groupby(group_col, sort=False)["qty"]

    # Lag features (vectorized groupby shift — already fast at 0.5s)
    for lag_n in LAG_RANGE:
        df[f"qty_lag_{lag_n}"] = g.shift(lag_n)

    # Check if all groups have uniform size (cross-product grid)
    # Use size() not count() — count() skips NaN (masked future qty),
    # which would report non-uniform groups on masked grids.
    group_sizes = df.groupby(group_col, sort=False).size()
    unique_sizes = group_sizes.unique()

    if len(unique_sizes) == 1 and unique_sizes[0] > 1:
        # Fast path: reshape to 2D matrix, use numpy cumsum for rolling
        n_months = int(unique_sizes[0])
        n_skus = len(group_sizes)
        qty_2d = df["qty"].values.reshape(n_skus, n_months).astype(np.float64)
        rolling_cols = _compute_rolling_numpy(qty_2d, ROLLING_WINDOWS)
        for col_name, values in rolling_cols.items():
            df[col_name] = values.astype(np.float32)
    else:
        # Fallback: pandas groupby rolling for non-uniform groups
        shifted = g.shift(1)
        grouped_shifted = shifted.groupby(df[group_col], sort=False)
        for w in ROLLING_WINDOWS:
            rolling = grouped_shifted.rolling(w, min_periods=1)
            df[f"rolling_mean_{w}m"] = rolling.mean().droplevel(0)
            df[f"rolling_std_{w}m"] = rolling.std().fillna(0).droplevel(0)


def _compute_ts_profile_features(
    grid: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute per-DFU time-series profile features.

    These are static features (one value per DFU) that summarize the demand
    pattern: volatility, intermittency, trend, seasonality.

    Args:
        grid: Feature grid with ``sku_ck``, ``startdate``, and ``qty`` columns.
        cutoff: If provided, only use data where ``startdate <= cutoff`` to
            avoid leaking future information into profile features.

    Returns:
        DataFrame with columns [sku_ck] + TS_PROFILE_FEATURES.
    """
    if cutoff is not None:
        source = grid.loc[grid["startdate"] <= cutoff]
    else:
        source = grid

    grouped = source.groupby("sku_ck", sort=False)["qty"]

    profiles: dict[str, list] = {col: [] for col in TS_PROFILE_FEATURES}
    sku_cks: list = []

    for sku_ck, qty_series in grouped:
        vals = qty_series.values.astype(np.float64)
        n = len(vals)
        mean_d = float(np.mean(vals))
        std_d = float(np.std(vals)) if n > 1 else 0.0

        sku_cks.append(sku_ck)
        profiles["mean_demand"].append(mean_d)
        profiles["cv_demand"].append(std_d / mean_d if mean_d > 0 else 0.0)
        profiles["zero_demand_pct"].append(float(np.sum(vals == 0)) / n if n > 0 else 0.0)

        # Trend: scale-invariant slope
        if n > 1:
            x = np.arange(n, dtype=np.float64)
            slope = float(np.polyfit(x, vals, 1)[0])
            profiles["trend_slope_norm"].append(slope / mean_d if mean_d > 0 else 0.0)
        else:
            profiles["trend_slope_norm"].append(0.0)

        # Recency ratio: mean of last 6m vs prior mean
        if n >= 12:
            last_6 = vals[-6:]
            prior = vals[:-6]
            prior_mean = float(prior.mean())
            profiles["recency_ratio"].append(
                float(last_6.mean()) / prior_mean if prior_mean > 0 else 1.0
            )
        else:
            profiles["recency_ratio"].append(1.0)

        # Seasonal amplitude: requires >= 12 months
        if n >= 12:
            # Use months from grid (sorted) — group by month-of-year
            month_means = np.array([
                float(np.mean(vals[i::12])) for i in range(min(12, n))
            ])
            if mean_d > 0 and len(month_means) > 1:
                profiles["seasonal_amplitude"].append(
                    (float(month_means.max()) - float(month_means.min())) / mean_d
                )
            else:
                profiles["seasonal_amplitude"].append(0.0)
        else:
            profiles["seasonal_amplitude"].append(0.0)

        # ADI (average demand interval)
        nonzero_idx = np.where(vals > 0)[0]
        if len(nonzero_idx) >= 2:
            profiles["adi"].append(float(np.mean(np.diff(nonzero_idx))))
        else:
            profiles["adi"].append(float(n))

        # Year-over-year correlation
        if n >= 24:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = float(np.corrcoef(vals[:-12], vals[12:])[0, 1])
            profiles["yoy_correlation"].append(corr if np.isfinite(corr) else 0.0)
        else:
            profiles["yoy_correlation"].append(0.0)

        # Periodicity: dominant cycle length via autocorrelation
        # Finds the lag (2-12 months) with the highest autocorrelation,
        # indicating the strongest repeating pattern. Value is the lag in
        # months (e.g., 12 = annual, 6 = semi-annual, 3 = quarterly).
        # Returns 0 if no significant periodicity or insufficient data.
        if n >= 6:
            centered = vals - mean_d
            var = float(np.dot(centered, centered))
            if var > 0:
                max_corr = 0.0
                best_lag = 0
                for lag in range(2, min(13, n)):
                    ac = float(np.dot(centered[:-lag], centered[lag:])) / var
                    if ac > max_corr:
                        max_corr = ac
                        best_lag = lag
                # Only report if autocorrelation is meaningful (> 0.2)
                profiles["periodicity"].append(float(best_lag) if max_corr > 0.2 else 0.0)
            else:
                profiles["periodicity"].append(0.0)
        else:
            profiles["periodicity"].append(0.0)

    result = pd.DataFrame({"sku_ck": sku_cks, **profiles})
    return result


def _compute_fourier_features(df: pd.DataFrame) -> None:
    """Compute Fourier seasonal terms in-place from the ``month`` column.

    Adds sin/cos pairs for periods 12, 6, 4, 3 months.  The period-12
    pair (fourier_sin_12, fourier_cos_12) replaces the legacy month_sin
    and month_cos features.  Additional periods capture sub-annual
    seasonality (quarterly promotions, biannual patterns).

    Strictly causal — derived from calendar date only.
    """
    month_vals = df["month"].values.astype(np.float64)
    for period in [12, 6, 4, 3]:
        angle = 2.0 * np.pi * month_vals / period
        df[f"fourier_sin_{period}"] = np.sin(angle).astype(np.float32)
        df[f"fourier_cos_{period}"] = np.cos(angle).astype(np.float32)


def _compute_croston_features(df: pd.DataFrame) -> None:
    """Compute Croston decomposition features in-place for intermittent demand.

    Per-DFU rolling features over the last 12 months (causal):
    - ``croston_demand_size``: Rolling mean of non-zero demand values
    - ``croston_demand_interval``: Rolling mean of intervals between non-zero demands
    - ``croston_probability``: 1 / interval (probability of demand in any month)

    Uses vectorized cumsum tricks per DFU group.  Requires ``qty`` and lag
    columns to already be present.
    """
    group_col = "sku_ck"
    window = 12  # look-back window (months)

    # Ensure sorted by (sku_ck, startdate)
    # (caller guarantees this for build_feature_matrix; safe for recompute path too)

    # Work on a per-group basis using numpy 2D reshape when possible
    g_sizes = df.groupby(group_col, sort=False).size()
    unique_sizes = g_sizes.unique()

    if len(unique_sizes) == 1 and unique_sizes[0] > 1:
        # Fast path: uniform grid → 2D reshape
        n_months = int(unique_sizes[0])
        n_skus = len(g_sizes)
        qty_2d = df["qty"].values.reshape(n_skus, n_months).astype(np.float64)

        # Shifted by 1 (causal): shifted[:, t] = qty[:, t-1]
        shifted = np.empty_like(qty_2d)
        shifted[:, 0] = np.nan
        shifted[:, 1:] = qty_2d[:, :-1]

        has_demand = (~np.isnan(shifted)) & (shifted > 0)
        demand_filled = np.where(has_demand, shifted, 0.0)

        # Cumulative sums for rolling non-zero mean
        cum_demand = np.cumsum(demand_filled, axis=1)
        cum_count = np.cumsum(has_demand.astype(np.float64), axis=1)

        # Rolling sum/count over last `window` months
        r_demand = cum_demand.copy()
        r_count = cum_count.copy()
        if window < n_months:
            r_demand[:, window:] = cum_demand[:, window:] - cum_demand[:, :-window]
            r_count[:, window:] = cum_count[:, window:] - cum_count[:, :-window]

        with np.errstate(invalid="ignore", divide="ignore"):
            demand_size = np.where(r_count > 0, r_demand / r_count, 0.0)

        # Compute rolling interval: within each window, the average gap between
        # non-zero demand occurrences.  Approximation: window / count gives the
        # average interval (number of months per non-zero occurrence).
        # For months with 0 or 1 non-zero occurrences, use the full window as interval.
        valid_mask = ~np.isnan(shifted)
        cum_valid = np.cumsum(valid_mask.astype(np.float64), axis=1)
        r_valid = cum_valid.copy()
        if window < n_months:
            r_valid[:, window:] = cum_valid[:, window:] - cum_valid[:, :-window]

        with np.errstate(invalid="ignore", divide="ignore"):
            # interval = valid_months / non_zero_count  (≈ months per demand event)
            demand_interval = np.where(
                r_count > 0,
                r_valid / r_count,
                np.where(r_valid > 0, r_valid, float(window)),
            )
            demand_interval = np.maximum(demand_interval, 1.0)

            probability = 1.0 / demand_interval

        df["croston_demand_size"] = np.nan_to_num(demand_size).ravel().astype(np.float32)
        df["croston_demand_interval"] = np.nan_to_num(demand_interval).ravel().astype(np.float32)
        df["croston_probability"] = np.nan_to_num(probability).ravel().astype(np.float32)
    else:
        # Fallback: pandas groupby for non-uniform groups
        df["croston_demand_size"] = np.float32(0)
        df["croston_demand_interval"] = np.float32(1)
        df["croston_probability"] = np.float32(0)

        for sku_ck, idx in df.groupby(group_col, sort=False).groups.items():
            qty_vals = df.loc[idx, "qty"].values.astype(np.float64)
            n = len(qty_vals)
            # Shifted by 1 (causal)
            shifted_vals = np.empty(n, dtype=np.float64)
            shifted_vals[0] = np.nan
            shifted_vals[1:] = qty_vals[:-1]

            sizes = np.zeros(n, dtype=np.float64)
            intervals = np.ones(n, dtype=np.float64)
            probs = np.zeros(n, dtype=np.float64)

            for t in range(1, n):
                start = max(0, t - window)
                win = shifted_vals[start:t]
                valid = win[~np.isnan(win)]
                nonzero = valid[valid > 0]
                n_valid = len(valid)
                n_nz = len(nonzero)

                if n_nz > 0:
                    sizes[t] = float(nonzero.mean())
                    intervals[t] = max(float(n_valid / n_nz), 1.0)
                    probs[t] = 1.0 / intervals[t]
                else:
                    intervals[t] = max(float(n_valid), 1.0) if n_valid > 0 else float(window)

            df.loc[idx, "croston_demand_size"] = sizes.astype(np.float32)
            df.loc[idx, "croston_demand_interval"] = intervals.astype(np.float32)
            df.loc[idx, "croston_probability"] = probs.astype(np.float32)


def enrich_with_external_forecast(
    grid: pd.DataFrame,
    ext_forecast_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Enrich feature grid with external forecast signal features.

    This is an OPTIONAL enrichment step, called externally after
    build_feature_matrix when external forecast data is available.

    Args:
        grid: Feature matrix from build_feature_matrix.
        ext_forecast_df: DataFrame with columns ``sku_ck``, ``startdate``,
            ``basefcst_pref`` (the external forecast quantity).
            If None, fills features with 0.

    Returns:
        grid with ``ext_fcst_ratio`` and ``ext_fcst_lag1_ratio`` added.
    """
    if ext_forecast_df is None or len(ext_forecast_df) == 0:
        for col in EXTERNAL_FORECAST_FEATURES:
            grid[col] = np.float32(0)
        logger.info("External forecast data not provided; ext_fcst features filled with 0")
        return grid

    # Merge external forecast onto grid
    ext = ext_forecast_df[["sku_ck", "startdate", FORECAST_QTY_COL]].copy()
    ext = ext.rename(columns={FORECAST_QTY_COL: "_ext_fcst"})
    ext = ext.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")

    grid = grid.merge(ext, on=["sku_ck", "startdate"], how="left")
    grid["_ext_fcst"] = grid["_ext_fcst"].fillna(0).astype(np.float64)

    # ext_fcst_ratio: external_forecast / rolling_mean_12m
    rm12 = grid["rolling_mean_12m"].fillna(0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid["ext_fcst_ratio"] = (grid["_ext_fcst"] / (rm12.abs() + 1.0)).clip(-10.0, 10.0)

    # ext_fcst_lag1_ratio: lag-1 of external forecast / qty_lag_1
    # Compute lag-1 of external forecast per DFU (causal)
    grid = grid.sort_values(["sku_ck", "startdate"]).reset_index(drop=True)
    grid["_ext_fcst_lag1"] = grid.groupby("sku_ck", sort=False)["_ext_fcst"].shift(1).fillna(0)

    qty_lag1 = grid["qty_lag_1"].fillna(0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid["ext_fcst_lag1_ratio"] = (grid["_ext_fcst_lag1"] / (qty_lag1.abs() + 1.0)).clip(-10.0, 10.0)

    # Clean up temp columns and downcast
    grid.drop(columns=["_ext_fcst", "_ext_fcst_lag1"], inplace=True)
    for col in EXTERNAL_FORECAST_FEATURES:
        grid[col] = grid[col].fillna(0).astype(np.float32)

    logger.info("External forecast features enriched: %s", EXTERNAL_FORECAST_FEATURES)
    return grid


def build_feature_matrix(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    all_months: list[pd.Timestamp],
    cat_dtype: str = "category",
    customer_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build FULL feature matrix: one row per (sku_ck, month).

    Features are strictly causal — only data from months < target month used.
    Built ONCE for all timeframes; per-timeframe masking done externally.

    Args:
        cat_dtype: Dtype for categorical features.
            "category" for LightGBM/XGBoost, "str" for CatBoost.
        customer_features: Optional customer-derived features at item×loc×month
            grain from customer_features_monthly table. Left-joined and NaN-filled.
    """
    t0 = time.time()
    dfu_keys = dfu_attrs[["sku_ck", "item_id", "customer_group", "loc"]].drop_duplicates()
    n_dfus = len(dfu_keys)
    n_months = len(all_months)
    logger.info("Building grid: %s DFUs × %s months = %s rows", f"{n_dfus:,}", n_months, f"{n_dfus * n_months:,}")

    # Build complete grid via MultiIndex (faster than cross-join merge)
    idx = pd.MultiIndex.from_product(
        [dfu_keys["sku_ck"].values, all_months],
        names=["sku_ck", "startdate"],
    )
    grid = pd.DataFrame(index=idx).reset_index()
    grid = grid.merge(dfu_keys, on="sku_ck", how="left")
    logger.info("Grid built: %s rows (%.1fs)", f"{len(grid):,}", time.time() - t0)

    # Join sales (full — no cutoff; masking done per timeframe)
    t1 = time.time()
    grid = grid.merge(
        sales_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="left",
    )
    grid["qty"] = grid["qty"].fillna(0)
    logger.info("Sales joined (%.1fs)", time.time() - t1)

    # Sort for lag/rolling operations
    t1 = time.time()
    grid = grid.sort_values(["sku_ck", "startdate"]).reset_index(drop=True)
    logger.info("Sorted (%.1fs)", time.time() - t1)

    # Lag + rolling features (factored into shared helper)
    t1 = time.time()
    _compute_lags_and_rolling(grid)
    logger.info("Lag + rolling features done (%.1fs)", time.time() - t1)

    # Calendar features
    grid["month"] = grid["startdate"].dt.month
    grid["quarter"] = grid["startdate"].dt.quarter
    # NOTE: month_sin/month_cos removed — mathematically identical to
    # fourier_sin_12/fourier_cos_12 which are computed by _compute_fourier_features().

    # Additional calendar features (leakage-free — derived from forecast date)
    grid["is_quarter_end"] = grid["startdate"].dt.month.isin([3, 6, 9, 12]).astype(int)
    grid["is_year_end"] = (grid["startdate"].dt.month == 12).astype(int)
    grid["days_in_month"] = grid["startdate"].dt.days_in_month.astype(float)

    # Fourier seasonal terms (sub-annual seasonality: periods 12, 6, 4, 3)
    t1 = time.time()
    _compute_fourier_features(grid)
    logger.debug("Fourier features done (%.1fs)", time.time() - t1)

    # Derived demand features (strictly causal — computed from lag/rolling features already built)
    _recompute_derived_features(grid)

    # Croston decomposition (intermittent demand features)
    t1 = time.time()
    _compute_croston_features(grid)
    logger.debug("Croston features done (%.1fs)", time.time() - t1)

    # Per-DFU time-series profile features (computed from full history)
    t1 = time.time()
    ts_profiles = _compute_ts_profile_features(grid)
    grid = grid.merge(ts_profiles, on="sku_ck", how="left")
    for col in TS_PROFILE_FEATURES:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0).astype(np.float32)
    logger.info("TS profile features done (%.1fs)", time.time() - t1)

    # DFU attributes
    t1 = time.time()
    dfu_feat_cols = ["sku_ck", "ml_cluster"] + CAT_FEATURES + NUMERIC_SKU_FEATURES
    dfu_feat_cols = [c for c in dfu_feat_cols if c in dfu_attrs.columns]
    grid = grid.merge(dfu_attrs[dfu_feat_cols], on="sku_ck", how="left")

    # Item attributes
    if len(item_attrs) > 0:
        grid = grid.merge(item_attrs, on="item_id", how="left")
    logger.info("Attributes joined (%.1fs)", time.time() - t1)

    # Customer-derived features (optional — enriched models only)
    if customer_features is not None and len(customer_features) > 0:
        t1 = time.time()
        # customer_features is at (item_id, loc, startdate) grain
        grid = grid.merge(customer_features, on=["item_id", "loc", "startdate"], how="left")
        for col in CUSTOMER_FEATURE_COLS:
            if col in grid.columns:
                grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(0).astype(np.float32)
        logger.info("Customer features joined: %d cols (%.1fs)", len(CUSTOMER_FEATURE_COLS), time.time() - t1)

    # Fill missing numerics (including enhanced features)
    for col in NUMERIC_SKU_FEATURES + NUMERIC_ITEM_FEATURES + ENHANCED_FEATURES:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(0)

    # Set categorical dtypes
    for col in CAT_FEATURES:
        if col in grid.columns:
            if cat_dtype == "str":
                grid[col] = grid[col].fillna("__unknown__").astype(str)
            else:
                grid[col] = grid[col].fillna("__unknown__").astype("category")

    # Downcast float64 → float32 to halve memory (~7GB savings on 9.8M rows)
    float_cols = grid.select_dtypes(include=["float64"]).columns
    grid[float_cols] = grid[float_cols].astype(np.float32)

    logger.info("Feature matrix complete: %s (%.1fs total)", grid.shape, time.time() - t0)
    return grid


def get_feature_columns(grid: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except metadata/target)."""
    return [c for c in grid.columns if c not in METADATA_COLS]


def mask_future_sales(grid: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Set future qty to NaN and recompute lag/rolling features for rows after cutoff.

    Instead of rebuilding the whole grid, we mask the qty column and
    recompute only the affected features. This is much faster than
    rebuilding from scratch.

    Using NaN (not zero) prevents artificial zeros from dragging down rolling
    means in direct-mode prediction. The rolling/lag functions skip NaN values
    when computing statistics (via ``min_periods=1``), so rolling means only
    reflect real historical data.

    After recomputation, feature columns (lags, rolling, derived) are filled
    with 0 so that downstream models can consume them without NaN issues. The
    ``qty`` column itself remains NaN for future months — models never see it
    directly as a feature (it is excluded by ``get_feature_columns``).
    """
    # Early return: if no rows are after cutoff, skip copy + recomputation
    future_mask = grid["startdate"] > cutoff
    if not future_mask.any():
        return grid.copy()

    df = grid.copy()

    # Mask future sales with NaN — not zero — so that rolling statistics
    # (which skip NaN via min_periods=1) are computed only from real history.
    # Zero would make the model believe demand was 0 in future months,
    # dragging down rolling means and corrupting Croston intermittency signals.
    df.loc[future_mask, "qty"] = np.nan

    # Recompute lags, rolling, and derived features on the masked data
    _compute_lags_and_rolling(df)
    _recompute_derived_features(df)

    # Recompute TS profile features using only pre-cutoff data to prevent leakage
    existing_ts_cols = [c for c in TS_PROFILE_FEATURES if c in df.columns]
    if existing_ts_cols:
        df.drop(columns=existing_ts_cols, inplace=True)
        ts_profiles = _compute_ts_profile_features(df, cutoff=cutoff)
        df = df.merge(ts_profiles, on="sku_ck", how="left")
        for col in TS_PROFILE_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(np.float32)

    # Fill NaN in feature columns so models can consume them. The qty column
    # stays NaN for future months (excluded from features by get_feature_columns).
    feature_cols = get_feature_columns(df)
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    return df


def update_grid_with_predictions(
    grid: pd.DataFrame,
    month: pd.Timestamp,
    predictions: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Write predicted qty for one month into the grid and recompute all lag/rolling features.

    Used by recursive multi-step inference (Feature 43). After calling this for month T,
    qty_lag_1 for month T+1 reflects the model's own prediction rather than 0.

    Args:
        grid: Full masked feature grid (all DFUs × all months).
        month: The month whose predictions are being written back.
        predictions: DataFrame with columns ``sku_ck`` and ``basefcst_pref``.
        inplace: If True, mutate ``grid`` directly instead of copying. Use when
            the caller owns the DataFrame and wants to avoid a full-grid copy
            (e.g. inside the recursive prediction loop).
    """
    df = grid if inplace else grid.copy()
    pred_map = predictions.drop_duplicates(subset="sku_ck", keep="first").set_index("sku_ck")[FORECAST_QTY_COL]
    mask = df["startdate"] == month
    df.loc[mask, "qty"] = df.loc[mask, "sku_ck"].map(pred_map).fillna(0)

    # Recompute lags, rolling, and derived features after prediction write-back
    _compute_lags_and_rolling(df)
    _recompute_derived_features(df)

    return df


def update_grid_incremental(
    grid: pd.DataFrame,
    month: pd.Timestamp,
    predictions: pd.DataFrame,
    sorted_months: list[pd.Timestamp],
    smooth_factor: float = 0.0,
) -> None:
    """Write predictions for one month and incrementally update only affected features.

    ~10x faster than update_grid_with_predictions for the recursive loop because
    it avoids recomputing lags/rolling for the entire 9.8M-row grid. Instead it:
    1. Writes predictions into the qty column at ``month``
    2. Updates lag columns only for the next 12 months (affected rows)
    3. Updates rolling stats only for the next max_window months

    When ``smooth_factor`` > 0, the lag-1 value written for the next month is
    blended between the raw prediction and the previous lag-1 value at the
    predicted month.  This dampens recursive error propagation by anchoring
    lag features partially to the prior value rather than trusting the raw
    prediction entirely::

        smoothed_lag1 = pred * (1 - smooth_factor) + old_lag1 * smooth_factor

    A typical value of 0.15 means 85% prediction / 15% prior lag.  Set to 0
    (default) to disable smoothing for full backward compatibility.

    Mutates ``grid`` in-place. Only valid when grid is a uniform cross-product
    sorted by (sku_ck, startdate).
    """
    pred_map = predictions.drop_duplicates(subset="sku_ck", keep="first").set_index("sku_ck")[FORECAST_QTY_COL]
    month_mask = grid["startdate"] == month
    grid.loc[month_mask, "qty"] = grid.loc[month_mask, "sku_ck"].map(pred_map).fillna(0).astype(grid["qty"].dtype)

    # Find position of this month in the sorted month list
    try:
        month_pos = sorted_months.index(month)
    except ValueError:
        # Fallback: full recompute
        _compute_lags_and_rolling(grid)
        _recompute_derived_features(grid)
        return

    n_months = len(sorted_months)
    max_lag = max(LAG_RANGE)  # 12
    max_window = max(ROLLING_WINDOWS)  # 12

    # Determine group structure
    g_sizes = grid.groupby("sku_ck", sort=False).size()
    if g_sizes.nunique() != 1:
        # Non-uniform: fallback to full recompute
        _compute_lags_and_rolling(grid)
        _recompute_derived_features(grid)
        return

    n_skus = len(g_sizes)
    qty_2d = grid["qty"].values.reshape(n_skus, n_months)

    # Update lag columns for affected months (month_pos+1 .. month_pos+max_lag)
    for lag_n in LAG_RANGE:
        target_pos = month_pos + lag_n
        if target_pos >= n_months:
            break
        target_month = sorted_months[target_pos]
        target_mask = grid["startdate"] == target_month
        new_lag = qty_2d[:, month_pos].astype(np.float32)

        # Smooth lag-1 to reduce recursive error propagation.
        # Blend the prediction-derived lag with the previous lag-1 value so
        # that downstream recursive steps don't compound raw prediction errors.
        if lag_n == 1 and smooth_factor > 0:
            col = "qty_lag_1"
            old_lag = grid.loc[target_mask, col].values
            # Only smooth where the old lag is finite (non-NaN); fresh rows keep
            # the raw prediction value.
            finite_mask = np.isfinite(old_lag)
            if finite_mask.any():
                smoothed = new_lag * (1 - smooth_factor) + old_lag * smooth_factor
                new_lag = np.where(finite_mask, smoothed, new_lag)

        grid.loc[target_mask, f"qty_lag_{lag_n}"] = new_lag

    # Update rolling stats for affected months
    # After changing qty at month_pos, shifted[month_pos+1] changes.
    # rolling_mean_wm at month M uses shifted[M-w+1..M].
    # So M is affected if month_pos+1 is in [M-w+1, M], i.e., M in [month_pos+1, month_pos+w].
    affected_start = month_pos + 1
    affected_end = min(month_pos + max_window + 1, n_months)

    if affected_start < n_months:
        # Build shifted array for rolling recompute
        shifted_2d = np.empty_like(qty_2d, dtype=np.float64)
        shifted_2d[:, 0] = np.nan
        shifted_2d[:, 1:] = qty_2d[:, :-1]

        for w in ROLLING_WINDOWS:
            w_end = min(month_pos + w + 1, n_months)
            for pos in range(affected_start, w_end):
                # Window: shifted[pos-w+1 .. pos] (inclusive)
                win_start = max(pos - w + 1, 0)
                window = shifted_2d[:, win_start:pos + 1]
                valid = ~np.isnan(window)
                counts = valid.sum(axis=1)
                sums = np.where(valid, window, 0.0).sum(axis=1)

                with np.errstate(invalid="ignore", divide="ignore"):
                    means = np.where(counts > 0, sums / counts, np.nan)
                    if counts.max() > 1:
                        sq_sums = np.where(valid, window * window, 0.0).sum(axis=1)
                        var = np.where(
                            counts > 1,
                            (sq_sums - sums * sums / counts) / (counts - 1),
                            0.0,
                        )
                        var = np.maximum(var, 0.0)
                        stds = np.sqrt(var)
                    else:
                        stds = np.zeros(n_skus)

                target_month = sorted_months[pos]
                target_mask = grid["startdate"] == target_month
                grid.loc[target_mask, f"rolling_mean_{w}m"] = means.astype(np.float32)
                grid.loc[target_mask, f"rolling_std_{w}m"] = np.nan_to_num(stds).astype(np.float32)

    # Recompute derived features for affected month range
    affected_months = set(sorted_months[affected_start:affected_end])
    affected_mask = grid["startdate"].isin(affected_months)
    if affected_mask.any():
        sub = grid.loc[affected_mask]
        grid.loc[affected_mask, "mom_growth"] = (
            (sub["qty_lag_1"] - sub["qty_lag_2"]) / (sub["qty_lag_2"].abs() + 1.0)
        ).clip(-2.0, 2.0)
        grid.loc[affected_mask, "demand_accel"] = sub["rolling_mean_3m"] - sub["rolling_mean_6m"]
        grid.loc[affected_mask, "volatility_ratio"] = sub["rolling_std_3m"] / (sub["rolling_mean_3m"].abs() + 1.0)

        # Lag ratio features — must be recomputed along with other derived features
        # to avoid stale values during recursive inference.
        grid.loc[affected_mask, "lag_ratio_yoy"] = (
            sub["qty_lag_1"] / (sub["qty_lag_12"].abs() + 1.0)
        ).clip(-10.0, 10.0) if "qty_lag_12" in grid.columns else 0.0
        grid.loc[affected_mask, "lag_ratio_mom"] = (
            sub["qty_lag_1"] / (sub["qty_lag_2"].abs() + 1.0)
        ).clip(-10.0, 10.0)
        grid.loc[affected_mask, "lag_ratio_3v12"] = (
            sub["rolling_mean_3m"] / (sub["rolling_mean_12m"].abs() + 1.0)
        ).clip(-10.0, 10.0)

        # Zero-demand count in recent 6 months
        lag_cols_6m = [f"qty_lag_{i}" for i in range(1, 7)]
        existing_lag_cols = [c for c in lag_cols_6m if c in grid.columns]
        if existing_lag_cols and "n_zero_last_6m" in grid.columns:
            grid.loc[affected_mask, "n_zero_last_6m"] = (
                (sub[existing_lag_cols] == 0).sum(axis=1).astype(np.float32)
            )

    # Recompute Croston and cross-DFU cluster features (full grid, not incremental —
    # these depend on rolling windows across all DFUs so partial update is unsafe)
    if "croston_demand_size" in grid.columns:
        _compute_croston_features(grid)
    # Cross-DFU cluster aggregates intentionally stay removed to avoid ml_cluster leakage.
