"""Statistical model upgrades for Advanced Expert Panel.

Models: AutoCES, DynamicOptimizedTheta, IMAPA, TSB, ADIDA, MSTL.
All use the Nixtla statsforecast library. Gracefully skip if not installed.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_start_method
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STATSFORECAST_AVAILABLE = False
try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoCES as _AutoCES,
        DynamicOptimizedTheta as _DynTheta,
        IMAPA as _IMAPA,
        TSB as _TSB,
        ADIDA as _ADIDA,
        MSTL as _MSTL,
        SimpleExponentialSmoothing as _SES,
    )
    _STATSFORECAST_AVAILABLE = True
except ImportError:
    logger.info("statsforecast not installed; statistical upgrades will be skipped")

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

# Minimum non-NaN observations required to attempt any statsforecast model.
_MIN_VALID_OBS = 3


def _sanitize_series(series: "pd.Series") -> "pd.Series":
    """Impute NaN values in a time series using linear interpolation.

    statsforecast models raise ``Exception("no model able to be fitted")``
    when the input contains any NaN values. This helper replaces interior NaN
    values with linearly interpolated values and fills remaining boundary NaN
    values with backward/forward fill so that the resulting series is always
    NaN-free (provided at least ``_MIN_VALID_OBS`` non-NaN observations exist).

    Returns the original series unchanged if no NaN values are present.
    """
    if not series.isna().any():
        return series
    return series.interpolate(method="linear").bfill().ffill()


def _series_is_fittable(series: "pd.Series") -> bool:
    """Return True if the series has enough valid observations to attempt fitting.

    Guards against:
    - All-NaN series (no valid data at all)
    - Fewer than ``_MIN_VALID_OBS`` non-NaN observations
    """
    n_valid = series.notna().sum()
    return int(n_valid) >= _MIN_VALID_OBS


def _ses_fallback(
    series: "pd.Series",
    predict_months: "list[pd.Timestamp]",
    alpha: float = 0.3,
) -> "pd.Series":
    """SES fallback: used when primary model fails after sanitization.

    Uses SimpleExponentialSmoothing from statsforecast (already imported).
    Returns a naive last-value forecast if SES itself also fails.
    """
    try:
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": series.index,
            "y": series.values,
        })
        sf = StatsForecast(models=[_SES(alpha=alpha)], freq="MS", n_jobs=1)
        forecast = sf.forecast(df=df, h=len(predict_months))
        value_cols = [c for c in forecast.columns if c not in ("unique_id", "ds")]
        if value_cols:
            values = forecast[value_cols[0]].values
            return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SES fallback also failed: %s", exc)

    # Last resort: replicate last observed value
    last_val = float(series.dropna().iloc[-1]) if not series.dropna().empty else 0.0
    return pd.Series(np.maximum(last_val, 0.0), index=predict_months)


def _get_forecast_values(forecast_df: pd.DataFrame) -> np.ndarray:
    """Extract forecast values from statsforecast output.

    statsforecast column names can vary (e.g., AutoCES -> 'CES'),
    so we grab the first non-index column.
    """
    value_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds")]
    if not value_cols:
        return np.array([])
    return forecast_df[value_cols[0]].values


# ---------------------------------------------------------------------------
# Individual model wrappers
# ---------------------------------------------------------------------------

def _predict_autoces(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """AutoCES — Complex Exponential Smoothing with automatic selection.

    Guards applied before fitting:
    - Minimum history length check (``min_history`` param, default 12).
    - Minimum valid (non-NaN) observation check.
    - NaN imputation via linear interpolation so statsforecast never sees NaN.

    On fitting failure, falls back to SES rather than silently returning NaN.
    """
    if not _STATSFORECAST_AVAILABLE:
        return pd.Series(np.nan, index=predict_months)

    season_length = params.get("season_length", 12)
    min_history = params.get("min_history", 12)

    if len(train_series) < min_history:
        return pd.Series(np.nan, index=predict_months)

    if not _series_is_fittable(train_series):
        logger.debug(
            "AutoCES skipped: series has fewer than %d valid observations", _MIN_VALID_OBS
        )
        return pd.Series(np.nan, index=predict_months)

    clean_series = _sanitize_series(train_series)

    try:
        model = _AutoCES(season_length=season_length)
        sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": clean_series.index,
            "y": clean_series.values,
        })
        forecast = sf.forecast(df=df, h=len(predict_months))
        values = _get_forecast_values(forecast)
        return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:
        # "no model able to be fitted" is an expected failure for degenerate series
        # (e.g. constant after imputation, or season_length > n_obs). Fall back to SES.
        logger.debug("AutoCES fitting failed (%s); falling back to SES", exc)
        return _ses_fallback(clean_series, predict_months)


def _predict_dynamic_theta(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """DynamicOptimizedTheta — adaptive decomposition parameters.

    Guards applied before fitting:
    - Minimum history length check (``min_history`` param, default 12).
    - Minimum valid (non-NaN) observation check.
    - NaN imputation via linear interpolation.

    On fitting failure, falls back to SES.
    """
    if not _STATSFORECAST_AVAILABLE:
        return pd.Series(np.nan, index=predict_months)

    season_length = params.get("season_length", 12)
    min_history = params.get("min_history", 12)

    if len(train_series) < min_history:
        return pd.Series(np.nan, index=predict_months)

    if not _series_is_fittable(train_series):
        logger.debug(
            "DynamicOptimizedTheta skipped: series has fewer than %d valid observations",
            _MIN_VALID_OBS,
        )
        return pd.Series(np.nan, index=predict_months)

    clean_series = _sanitize_series(train_series)

    try:
        model = _DynTheta(season_length=season_length)
        sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": clean_series.index,
            "y": clean_series.values,
        })
        forecast = sf.forecast(df=df, h=len(predict_months))
        values = _get_forecast_values(forecast)
        return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:
        logger.debug("DynamicOptimizedTheta fitting failed (%s); falling back to SES", exc)
        return _ses_fallback(clean_series, predict_months)


def _predict_imapa(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """IMAPA — Intermittent Multiple Aggregation Prediction Algorithm.

    Guards applied before fitting:
    - Minimum non-zero observation check (``min_nonzero`` param, default 2).
    - Minimum valid (non-NaN) observation check.
    - NaN imputation via linear interpolation.

    On fitting failure, falls back to SES.
    """
    if not _STATSFORECAST_AVAILABLE:
        return pd.Series(np.nan, index=predict_months)

    min_nonzero = params.get("min_nonzero", 2)

    if not _series_is_fittable(train_series):
        logger.debug(
            "IMAPA skipped: series has fewer than %d valid observations", _MIN_VALID_OBS
        )
        return pd.Series(np.nan, index=predict_months)

    clean_series = _sanitize_series(train_series)

    if (clean_series > 0).sum() < min_nonzero:
        return pd.Series(np.nan, index=predict_months)

    try:
        model = _IMAPA()
        sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": clean_series.index,
            "y": clean_series.values,
        })
        forecast = sf.forecast(df=df, h=len(predict_months))
        values = _get_forecast_values(forecast)
        return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:
        logger.debug("IMAPA fitting failed (%s); falling back to SES", exc)
        return _ses_fallback(clean_series, predict_months)


def _predict_tsb(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """TSB — Teunter-Syntetos-Babai method (models obsolescence).

    Guards applied before fitting:
    - Minimum non-zero observation check (``min_nonzero`` param, default 2).
    - Minimum valid (non-NaN) observation check.
    - NaN imputation via linear interpolation.

    On fitting failure, falls back to SES.
    """
    if not _STATSFORECAST_AVAILABLE:
        return pd.Series(np.nan, index=predict_months)

    alpha_d = params.get("alpha_d", 0.1)
    alpha_p = params.get("alpha_p", 0.1)
    min_nonzero = params.get("min_nonzero", 2)

    if not _series_is_fittable(train_series):
        logger.debug(
            "TSB skipped: series has fewer than %d valid observations", _MIN_VALID_OBS
        )
        return pd.Series(np.nan, index=predict_months)

    clean_series = _sanitize_series(train_series)

    if (clean_series > 0).sum() < min_nonzero:
        return pd.Series(np.nan, index=predict_months)

    try:
        model = _TSB(alpha_d=alpha_d, alpha_p=alpha_p)
        sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": clean_series.index,
            "y": clean_series.values,
        })
        forecast = sf.forecast(df=df, h=len(predict_months))
        values = _get_forecast_values(forecast)
        return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:
        logger.debug("TSB fitting failed (%s); falling back to SES", exc)
        return _ses_fallback(clean_series, predict_months)


def _predict_adida(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """ADIDA — Aggregate-Disaggregate Intermittent Demand Approach.

    Guards applied before fitting:
    - Minimum non-zero observation check (``min_nonzero`` param, default 2).
    - Minimum valid (non-NaN) observation check.
    - NaN imputation via linear interpolation.

    On fitting failure, falls back to SES.
    """
    if not _STATSFORECAST_AVAILABLE:
        return pd.Series(np.nan, index=predict_months)

    min_nonzero = params.get("min_nonzero", 2)

    if not _series_is_fittable(train_series):
        logger.debug(
            "ADIDA skipped: series has fewer than %d valid observations", _MIN_VALID_OBS
        )
        return pd.Series(np.nan, index=predict_months)

    clean_series = _sanitize_series(train_series)

    if (clean_series > 0).sum() < min_nonzero:
        return pd.Series(np.nan, index=predict_months)

    try:
        model = _ADIDA()
        sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": clean_series.index,
            "y": clean_series.values,
        })
        forecast = sf.forecast(df=df, h=len(predict_months))
        values = _get_forecast_values(forecast)
        return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:
        logger.debug("ADIDA fitting failed (%s); falling back to SES", exc)
        return _ses_fallback(clean_series, predict_months)


def _predict_mstl(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """MSTL — Multiple Seasonal-Trend decomposition using LOESS."""
    if not _STATSFORECAST_AVAILABLE:
        return pd.Series(np.nan, index=predict_months)

    season_length = params.get("season_length", 12)
    min_history = params.get("min_history", 25)

    if len(train_series) < min_history:
        return pd.Series(np.nan, index=predict_months)

    try:
        model = _MSTL(season_length=season_length)
        sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": train_series.index,
            "y": train_series.values,
        })
        forecast = sf.forecast(df=df, h=len(predict_months))
        values = _get_forecast_values(forecast)
        return pd.Series(np.maximum(values, 0.0), index=predict_months)
    except Exception as exc:
        logger.warning("MSTL failure: %s", exc)
        return pd.Series(np.nan, index=predict_months)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_MODEL_DISPATCH: dict[str, Any] = {
    "autoces": _predict_autoces,
    "dynamic_theta": _predict_dynamic_theta,
    "imapa": _predict_imapa,
    "tsb": _predict_tsb,
    "adida": _predict_adida,
    "mstl": _predict_mstl,
}


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _fit_dfu_upgrades(args: tuple) -> list[dict]:
    """Worker for parallel statistical upgrade model fitting."""
    sku_ck, train_series, predict_months, enabled_models = args
    results: list[dict] = []

    for model_id, params in enabled_models.items():
        fn = _MODEL_DISPATCH.get(model_id)
        if fn is None:
            continue
        try:
            forecast = fn(train_series, predict_months, params)
        except Exception as exc:
            logger.warning("Model %s failed for DFU %s: %s", model_id, sku_ck, exc)
            continue

        if forecast.isna().all():
            continue

        for ts, val in forecast.items():
            if not np.isnan(val):
                results.append({
                    "sku_ck": sku_ck,
                    "startdate": ts,
                    "basefcst_pref": float(val),
                    "algorithm_id": model_id,
                })

    return results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _fill_monthly_series(
    group_df: pd.DataFrame, all_months: list[pd.Timestamp]
) -> pd.Series:
    """Create a complete monthly series, filling missing months with 0."""
    series = group_df.set_index("startdate")["qty"].reindex(all_months, fill_value=0.0).astype(float)
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    return series


def run_statistical_upgrades(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
    n_workers: int = 8,
) -> pd.DataFrame:
    """Run all enabled statistical upgrade models across all DFUs.

    Args:
        sales_df: Training sales with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.
        enabled_models: {model_id: params_dict} for enabled models.
        n_workers: Number of parallel workers.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    if not _STATSFORECAST_AVAILABLE:
        logger.warning("statsforecast not installed; returning empty DataFrame")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    if not enabled_models:
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    sku_cks = sales_df["sku_ck"].unique()
    n_total = len(sku_cks)
    logger.info(
        "Running %d statistical upgrade models across %d DFUs with %d workers",
        len(enabled_models), n_total, n_workers,
    )

    # Build a contiguous monthly range so _fill_monthly_series can set freq="MS"
    # without gaps.  Using unique() alone fails when any calendar month is absent
    # from all DFUs in the subset (sparse groups), because pandas then rejects the
    # non-uniform index as incompatible with freq="MS".
    _min = pd.Timestamp(sales_df["startdate"].min()).replace(day=1)
    _max = pd.Timestamp(sales_df["startdate"].max()).replace(day=1)
    all_months = pd.date_range(start=_min, end=_max, freq="MS").tolist()
    grouped = sales_df.groupby("sku_ck", sort=False)

    task_args: list[tuple] = []
    for sku_ck in sku_cks:
        group_df = grouped.get_group(sku_ck)
        train_series = _fill_monthly_series(group_df, all_months)
        task_args.append((sku_ck, train_series, predict_months, enabled_models))

    all_results: list[dict] = []
    n_done = 0

    # Parallel processing via ProcessPoolExecutor.
    # Safe when start method is 'spawn' (set by run_adv_expert_panel.py on macOS to
    # prevent fork()-after-torch-import SIGSEGV). statsforecast itself does not use
    # torch, so spawned workers re-import cleanly with no GPU state.
    # Falls back to sequential if spawn is not set or n_workers <= 1.
    use_parallel = n_workers > 1 and get_start_method(allow_none=True) == "spawn"
    if not use_parallel:
        logger.info(
            "Statistical upgrades running sequentially "
            "(start_method=%s, n_workers=%d)",
            get_start_method(allow_none=True), n_workers,
        )

    if use_parallel:
        with ProcessPoolExecutor(max_workers=min(n_workers, n_total)) as executor:
            futures = {
                executor.submit(_fit_dfu_upgrades, args): args[0]
                for args in task_args
            }
            for fut in as_completed(futures):
                sku_ck = futures[fut]
                try:
                    all_results.extend(fut.result())
                except (ValueError, RuntimeError) as exc:
                    logger.warning("DFU %s failed: %s", sku_ck, exc)
                n_done += 1
                if n_done % 500 == 0:
                    logger.info(
                        "Statistical upgrades: %d/%d DFUs complete", n_done, n_total
                    )
    else:
        for args in task_args:
            sku_ck = args[0]
            try:
                all_results.extend(_fit_dfu_upgrades(args))
            except (ValueError, RuntimeError) as exc:
                logger.warning("DFU %s failed: %s", sku_ck, exc)
            n_done += 1
            if n_done % 500 == 0:
                logger.info(
                    "Statistical upgrades: %d/%d DFUs complete", n_done, n_total
                )

    if n_done % 500 != 0:
        logger.info("Statistical upgrades: %d/%d DFUs complete", n_done, n_total)

    if not all_results:
        logger.warning("All statistical upgrade predictions failed")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    result_df = pd.DataFrame(all_results)
    logger.info(
        "Statistical upgrades: %d predictions across %d DFUs",
        len(result_df), result_df["sku_ck"].nunique(),
    )
    return result_df
