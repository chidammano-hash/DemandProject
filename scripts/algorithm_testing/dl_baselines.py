"""Deep learning baselines: DLinear and NLinear.

One-layer linear models that serve as sanity-check baselines.
If these beat complex models, the features are not adding value.
Implemented from scratch (no external dependencies beyond numpy/pandas).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _dlinear_forecast(
    history: np.ndarray,
    prediction_length: int,
    kernel_size: int = 25,
) -> np.ndarray:
    """DLinear: decompose into trend (moving average) + seasonal, predict each.

    Uses simple linear regression on each component.
    """
    n = len(history)
    if n < kernel_size:
        kernel_size = max(3, n // 2)

    # Trend via moving average
    pad = kernel_size // 2
    padded = np.pad(history, (pad, pad), mode="edge")
    trend = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")[:n]

    # Seasonal = residual
    seasonal = history - trend

    # Simple linear extrapolation for trend
    x = np.arange(n, dtype=np.float64)
    x_pred = np.arange(n, n + prediction_length, dtype=np.float64)

    # Fit trend line
    if np.std(trend) > 1e-8:
        coeffs = np.polyfit(x, trend, deg=1)
        trend_pred = np.polyval(coeffs, x_pred)
    else:
        trend_pred = np.full(prediction_length, trend[-1])

    # Repeat last seasonal cycle
    if n >= 12:
        seasonal_cycle = seasonal[-12:]
        seasonal_pred = np.tile(seasonal_cycle, (prediction_length // 12) + 1)[:prediction_length]
    else:
        seasonal_pred = np.full(prediction_length, seasonal[-1])

    return trend_pred + seasonal_pred


def _nlinear_forecast(
    history: np.ndarray,
    prediction_length: int,
) -> np.ndarray:
    """NLinear: normalize by last value, apply linear, add back.

    Handles distribution shift by subtracting the last observation.
    """
    last_val = history[-1]
    normalized = history - last_val

    n = len(normalized)
    x = np.arange(n, dtype=np.float64)
    x_pred = np.arange(n, n + prediction_length, dtype=np.float64)

    # Simple linear fit on normalized series
    if np.std(normalized) > 1e-8:
        coeffs = np.polyfit(x, normalized, deg=1)
        pred_normalized = np.polyval(coeffs, x_pred)
    else:
        pred_normalized = np.zeros(prediction_length)

    return pred_normalized + last_val


def predict_dlinear(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """DLinear baseline: trend-seasonal decomposition + linear prediction.

    Args:
        sales_df: Training data with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    if sales_df.empty:
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    prediction_length = len(predict_months)
    all_results: list[dict[str, Any]] = []

    for sku_ck, group in sales_df.groupby("sku_ck", sort=False):
        values = group.sort_values("startdate")["qty"].values.astype(np.float64)
        if len(values) < 6:
            continue

        forecast = _dlinear_forecast(values, prediction_length)
        forecast = np.maximum(forecast, 0.0)

        for j, month in enumerate(predict_months):
            all_results.append({
                "sku_ck": sku_ck,
                "startdate": month,
                "basefcst_pref": float(forecast[j]),
                "algorithm_id": "dlinear",
            })

    result = pd.DataFrame(all_results)
    logger.info("DLinear: %d predictions for %d DFUs",
                len(result), result["sku_ck"].nunique() if not result.empty else 0)
    return result


def predict_nlinear(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """NLinear baseline: last-value normalization + linear prediction.

    Args:
        sales_df: Training data with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    if sales_df.empty:
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    prediction_length = len(predict_months)
    all_results: list[dict[str, Any]] = []

    for sku_ck, group in sales_df.groupby("sku_ck", sort=False):
        values = group.sort_values("startdate")["qty"].values.astype(np.float64)
        if len(values) < 3:
            continue

        forecast = _nlinear_forecast(values, prediction_length)
        forecast = np.maximum(forecast, 0.0)

        for j, month in enumerate(predict_months):
            all_results.append({
                "sku_ck": sku_ck,
                "startdate": month,
                "basefcst_pref": float(forecast[j]),
                "algorithm_id": "nlinear",
            })

    result = pd.DataFrame(all_results)
    logger.info("NLinear: %d predictions for %d DFUs",
                len(result), result["sku_ck"].nunique() if not result.empty else 0)
    return result
