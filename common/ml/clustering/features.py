"""
Clustering feature engineering — pure library functions.

Extracts time series features from monthly demand data for clustering analysis.

Feature engineering covers six dimensions:
  - Volume         : mean, CV, IQR (robust spread)
  - Trend          : scale-invariant normalized slope, R², CAGR
  - Seasonality    : amplitude ratio, OLS seasonal R² (STL-lite), YoY correlation
  - Periodicity    : FFT dominant-component strength
  - Intermittency  : zero-demand pct, Croston ADI
  - Lifecycle      : months available, recency ratio (last-6m / full history)
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

try:
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — scipy optional in lightweight envs
    _HAS_SCIPY = False


def _seasonal_r2(y: np.ndarray, months: np.ndarray) -> float:
    """Fit 12-period seasonal dummy OLS and return R² explained by seasonal component.

    Uses 11 monthly indicator dummies (month 12 is the reference level) to avoid
    multicollinearity.  Requires len(y) >= 24; returns 0.0 otherwise.
    """
    if len(y) < 24:
        return 0.0
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    # Build design matrix: intercept + 11 monthly dummies (month 1..11 vs 12)
    n = len(y)
    months_int = months.astype(np.intp)
    D = np.zeros((n, 11), dtype=np.float64)
    mask = months_int < 12
    D[np.where(mask)[0], months_int[mask] - 1] = 1.0
    X_seas = np.column_stack([np.ones(n), D])
    coef, _, _, _ = lstsq(X_seas, y, rcond=None)
    y_seas = X_seas @ coef
    ss_seas = np.sum((y_seas - np.mean(y)) ** 2)
    return float(min(1.0, ss_seas / ss_tot))


def _periodicity_strength(y: np.ndarray) -> float:
    """Return the fraction of total FFT power held by the dominant non-DC component.

    High value => strong single periodic component (e.g. annual cycle).
    Returns 0.0 when len(y) < 12 or signal is flat.
    """
    if len(y) < 12:
        return 0.0
    y_centered = y - np.mean(y)
    if np.std(y_centered) == 0.0:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(y_centered)).astype(np.float64)
    fft_vals[0] = 0.0  # zero out DC component
    total_power = float(np.sum(fft_vals ** 2))
    if total_power == 0.0:
        return 0.0
    dominant_power = float(np.max(fft_vals) ** 2)
    return float(min(1.0, dominant_power / total_power))


def _adi(demand_values: np.ndarray) -> float:
    """Compute Croston's Average Demand Interval (months between nonzero periods).

    - 2+ nonzero periods: mean of gaps between them
    - 1 nonzero period : total length (worst-case sparse)
    - 0 nonzero periods: total length (completely intermittent)
    """
    nonzero_idx = np.where(demand_values > 0)[0]
    if len(nonzero_idx) >= 2:
        return float(np.mean(np.diff(nonzero_idx)))
    return float(len(demand_values))


def compute_time_series_features(df: pd.DataFrame) -> pd.Series:
    """Compute time series features from monthly demand data.

    Parameters
    ----------
    df : DataFrame with columns ``startdate`` (datetime) and ``qty`` (float/int).

    Returns
    -------
    pd.Series of named scalar features.
    """
    features: dict[str, Any] = {}

    if len(df) == 0:
        return pd.Series(features)

    df_sorted = df.sort_values("startdate").reset_index(drop=True)
    demand_values = np.asarray(df_sorted["qty"].fillna(0), dtype=np.float64)
    n = len(demand_values)
    mean_demand = float(np.mean(demand_values))

    # ── Lifecycle ────────────────────────────────────────────────────────────
    features["months_available"] = n

    # ── Volume ───────────────────────────────────────────────────────────────
    features["mean_demand"] = mean_demand
    features["median_demand"] = float(np.median(demand_values))
    std_demand = float(np.std(demand_values)) if n > 1 else 0.0
    features["std_demand"] = std_demand
    features["cv_demand"] = std_demand / mean_demand if mean_demand > 0.0 else 0.0
    features["iqr_demand"] = float(
        np.percentile(demand_values, 75) - np.percentile(demand_values, 25)
    )
    features["min_demand"] = float(np.min(demand_values))
    features["max_demand"] = float(np.max(demand_values))
    features["total_demand"] = float(np.sum(demand_values))
    features["demand_mad"] = float(np.mean(np.abs(demand_values - mean_demand)))
    features["demand_p90"] = float(np.percentile(demand_values, 90))
    if n > 2 and _HAS_SCIPY:
        features["demand_skewness"] = float(scipy_stats.skew(demand_values))
        features["demand_kurtosis"] = float(scipy_stats.kurtosis(demand_values))
    else:
        features["demand_skewness"] = 0.0
        features["demand_kurtosis"] = 0.0

    # ── Trend ────────────────────────────────────────────────────────────────
    if n > 1:
        x = np.arange(n, dtype=np.float64)
        y = demand_values
        slope = float(np.polyfit(x, y, 1)[0])
        features["trend_slope"] = slope
        # Scale-invariant slope
        features["trend_slope_norm"] = slope / mean_demand if mean_demand > 0.0 else 0.0

        # R² of linear fit, signed by direction of slope
        if n > 2:
            y_pred = slope * x + (np.mean(y) - slope * np.mean(x))
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
            features["trend_r2"] = float(r2 * np.sign(slope)) if slope != 0.0 else 0.0
        else:
            features["trend_r2"] = 0.0

        features["trend_pct_change"] = (
            (demand_values[-1] - demand_values[0]) / demand_values[0] * 100.0
            if demand_values[0] > 0.0 else 0.0
        )
        features["trend_direction"] = (
            1 if slope > 0.01 else (-1 if slope < -0.01 else 0)
        )
    else:
        features["trend_slope"] = 0.0
        features["trend_slope_norm"] = 0.0
        features["trend_r2"] = 0.0
        features["trend_pct_change"] = 0.0
        features["trend_direction"] = 0

    # ── Seasonality ──────────────────────────────────────────────────────────
    if n >= 12:
        month_nums = df_sorted["startdate"].dt.month
        monthly_means = df_sorted.groupby(month_nums)["qty"].mean()

        if len(monthly_means) > 1:
            seasonal_std = float(monthly_means.std())
            seasonal_mean = float(monthly_means.mean())
            # Legacy feature — CV of monthly averages (kept for compatibility)
            features["seasonality_strength"] = (
                seasonal_std / seasonal_mean if seasonal_mean > 0.0 else 0.0
            )
            features["peak_month"] = int(monthly_means.idxmax())
            features["trough_month"] = int(monthly_means.idxmin())
            features["peak_trough_ratio"] = (
                float(monthly_means.max()) / float(monthly_means.min())
                if float(monthly_means.min()) > 0 else 0.0
            )
            features["seasonal_index_std"] = seasonal_std

            # Amplitude ratio: (max_monthly_mean - min_monthly_mean) / overall_mean
            features["seasonal_amplitude"] = (
                (float(monthly_means.max()) - float(monthly_means.min())) / mean_demand
                if mean_demand > 0.0 else 0.0
            )
        else:
            features["seasonality_strength"] = 0.0
            features["peak_month"] = 1
            features["trough_month"] = 1
            features["peak_trough_ratio"] = 0.0
            features["seasonal_index_std"] = 0.0
            features["seasonal_amplitude"] = 0.0

        # OLS seasonal R² (STL-lite) — requires >= 24 months
        months_arr = month_nums.to_numpy(dtype=np.float64)
        features["seasonal_r2"] = _seasonal_r2(demand_values, months_arr)

        # Year-over-year correlation
        if n >= 24:
            df_sorted_yoy = df_sorted.copy()
            df_sorted_yoy["year"] = df_sorted_yoy["startdate"].dt.year
            df_sorted_yoy["month"] = df_sorted_yoy["startdate"].dt.month
            pivot = df_sorted_yoy.pivot_table(
                values="qty", index="month", columns="year", aggfunc="mean"
            )
            if pivot.shape[1] >= 2:
                corr_matrix = pivot.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                yoy_corr = corr_matrix.where(mask).stack()
                features["yoy_correlation"] = float(yoy_corr.mean()) if len(yoy_corr) > 0 else 0.0
            else:
                features["yoy_correlation"] = 0.0
        else:
            features["yoy_correlation"] = 0.0

        # Legacy alias kept for backward compatibility
        features["year_over_year_correlation"] = features["yoy_correlation"]

        # Autocorrelation at lag 12 — captures annual repetition strength
        if n >= 24:
            acf12 = float(np.corrcoef(demand_values[:-12], demand_values[12:])[0, 1])
            # Guard against NaN when one slice is constant
            features["acf_lag12"] = acf12 if not np.isnan(acf12) else 0.0
        else:
            features["acf_lag12"] = 0.0
    else:
        features["seasonality_strength"] = 0.0
        features["peak_month"] = 1
        features["trough_month"] = 1
        features["peak_trough_ratio"] = 0.0
        features["seasonal_index_std"] = 0.0
        features["seasonal_amplitude"] = 0.0
        features["seasonal_r2"] = 0.0
        features["yoy_correlation"] = 0.0
        features["year_over_year_correlation"] = 0.0
        features["acf_lag12"] = 0.0

    # ── Periodicity ──────────────────────────────────────────────────────────
    features["periodicity_strength"] = _periodicity_strength(demand_values)

    # ── Intermittency ────────────────────────────────────────────────────────
    zero_count = int(np.sum(demand_values == 0))
    features["zero_demand_pct"] = zero_count / n if n > 0 else 0.0
    features["sparsity_score"] = features["zero_demand_pct"]  # alias for compat
    features["adi"] = _adi(demand_values)
    features["demand_stability"] = 1.0 / (1.0 + features["cv_demand"])

    # Outlier count (> 2 std from mean)
    if std_demand > 0.0:
        features["outlier_count"] = int(
            np.sum(np.abs(demand_values - mean_demand) > 2.0 * std_demand)
        )
    else:
        features["outlier_count"] = 0

    # ── Growth / lifecycle ───────────────────────────────────────────────────
    if n >= 12:
        half = n // 2
        first_half_mean = float(demand_values[:half].mean()) if half > 0 else 0.0
        second_half_mean = float(demand_values[half:].mean())

        # CAGR expressed as % (kept as ``growth_rate`` for compat; also as ``cagr``)
        if first_half_mean > 0.0:
            periods = n / 12.0
            cagr = ((second_half_mean / first_half_mean) ** (1.0 / periods) - 1.0) * 100.0
        else:
            cagr = 0.0
        features["cagr"] = cagr
        features["growth_rate"] = cagr  # backward-compat alias

        # Recency ratio: mean of last 6m vs full history mean
        last_6 = demand_values[-6:]
        prior = demand_values[:-6] if n > 6 else demand_values
        recency = float(last_6.mean()) / float(prior.mean()) if float(prior.mean()) > 0.0 else 1.0
        features["recency_ratio"] = recency
        features["recent_vs_historical"] = recency  # backward-compat alias

        # Acceleration (second-derivative approximation across thirds)
        if n >= 3:
            t1 = demand_values[: n // 3]
            t2 = demand_values[n // 3 : 2 * n // 3]
            t3 = demand_values[2 * n // 3 :]
            m1, m2, m3 = float(t1.mean()), float(t2.mean()), float(t3.mean())
            g1 = (m2 - m1) / m1 if m1 > 0.0 else 0.0
            g2 = (m3 - m2) / m2 if m2 > 0.0 else 0.0
            features["acceleration"] = g2 - g1
        else:
            features["acceleration"] = 0.0
    else:
        features["cagr"] = 0.0
        features["growth_rate"] = 0.0
        features["recency_ratio"] = 1.0
        features["recent_vs_historical"] = 1.0
        features["acceleration"] = 0.0

    return pd.Series(features)


def _compute_features_for_group(args_tuple: tuple) -> dict:
    """Compute features for a single DFU group.  Designed for multiprocessing.Pool.map().

    Parameters
    ----------
    args_tuple : (sku_ck, group_df_values)
        ``group_df_values`` is a dict with ``startdate`` and ``qty`` arrays to avoid
        pickling full DataFrames.

    Returns
    -------
    dict of feature name -> value, including ``sku_ck``.
    """
    sku_ck, gdata = args_tuple
    # Reconstruct a minimal DataFrame for compute_time_series_features
    df = pd.DataFrame({"startdate": gdata["startdate"], "qty": gdata["qty"]})
    features = compute_time_series_features(df)
    result = features.to_dict()
    result["sku_ck"] = sku_ck
    return result
