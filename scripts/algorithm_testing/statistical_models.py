"""Statistical forecasting models for the Expert Panel test.

Models: Holt-Winters, Simple ES, Croston SBA, Auto-ARIMA, Theta.
Each model works on a single DFU's univariate time series.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL

logger = logging.getLogger(__name__)

# DFU count below which sequential execution is faster than process-pool overhead
_SEQUENTIAL_THRESHOLD = 200

# Module-level globals populated once per worker process via _worker_init.
# Avoids re-pickling enabled_models and predict_months for every DFU task.
_WORKER_MODELS: dict[str, Any] = {}
_WORKER_MONTHS: list[pd.Timestamp] = []


def _worker_init(models: dict[str, Any], months: list[pd.Timestamp]) -> None:
    global _WORKER_MODELS, _WORKER_MONTHS
    _WORKER_MODELS = models
    _WORKER_MONTHS = months


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fill_monthly_series(
    group_df: pd.DataFrame, all_months: list[pd.Timestamp]
) -> pd.Series:
    """Create a complete monthly time series, filling missing months with 0."""
    series = group_df.set_index("startdate")["qty"].reindex(all_months, fill_value=0.0)
    # Rebuild a contiguous month-start index to guarantee freq="MS" even when
    # upstream preprocessing (e.g., leading-zero trim) creates gaps.
    contiguous = pd.date_range(
        start=series.index.min(), end=series.index.max(), freq="MS"
    )
    series = series.reindex(contiguous, fill_value=0.0)
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    return series


# ---------------------------------------------------------------------------
# Individual model implementations
# ---------------------------------------------------------------------------


def _predict_holt_winters(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """Holt-Winters exponential smoothing with additive trend and seasonality."""
    import warnings

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    seasonal_periods: int = params.get("seasonal_periods", 12)
    min_history: int = params.get("min_history", 2 * seasonal_periods + 1)

    if len(train_series) < min_history:
        logger.debug(
            "Holt-Winters requires %d observations, got %d",
            min_history,
            len(train_series),
        )
        return pd.Series(np.nan, index=predict_months)

    try:
        # Suppress ConvergenceWarning: the optimizer produces valid (if sub-optimal)
        # smoothing parameters even when it hasn't fully converged. Using the result
        # is safe; forcing optimized=False risks NoneType parameters on some series.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = ExponentialSmoothing(
                train_series,
                trend=params.get("trend", "add"),
                seasonal=params.get("seasonal", "add"),
                seasonal_periods=seasonal_periods,
                damped_trend=params.get("damped_trend", True),
                initialization_method="estimated",
            )
            fitted = model.fit()
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("Holt-Winters failure: %s", exc)
        return pd.Series(np.nan, index=predict_months)

    forecast = fitted.forecast(steps=len(predict_months))
    forecast.index = predict_months
    return np.maximum(forecast, 0)


def _predict_simple_es(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """Simple Exponential Smoothing — robust, works with as few as 3 points."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    min_history: int = params.get("min_history", 3)

    if len(train_series) < min_history:
        logger.debug(
            "Simple ES requires %d observations, got %d",
            min_history,
            len(train_series),
        )
        return pd.Series(np.nan, index=predict_months)

    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    smoothing_level = params.get("smoothing_level")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = SimpleExpSmoothing(train_series, initialization_method="estimated")
            fitted = model.fit(smoothing_level=smoothing_level, optimized=True)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("Simple ES failure: %s", exc)
        return pd.Series(np.nan, index=predict_months)

    forecast = fitted.forecast(steps=len(predict_months))
    forecast.index = predict_months
    return np.maximum(forecast, 0)


def _predict_croston_sba(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """Croston's method with Syntetos-Boylan Approximation (SBA) correction.

    Designed for intermittent / lumpy demand patterns.
    """
    alpha: float = params.get("alpha", 0.1)
    min_nonzero: int = params.get("min_nonzero", 2)

    values = train_series.values.astype(float)

    # Extract non-zero demand positions and values
    nonzero_mask = values > 0
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(nonzero_indices) < min_nonzero:
        logger.debug(
            "Croston SBA requires %d non-zero observations, got %d",
            min_nonzero,
            len(nonzero_indices),
        )
        return pd.Series(np.nan, index=predict_months)

    nonzero_demands = values[nonzero_indices]

    # Compute intervals between consecutive non-zero demands
    intervals = np.diff(nonzero_indices).astype(float)

    # Initialize smoothed demand size and interval
    z_t: float = float(nonzero_demands[0])
    p_t: float = float(intervals[0]) if len(intervals) > 0 else 1.0

    # Smooth demand sizes (starting from 2nd non-zero demand)
    for demand_i in nonzero_demands[1:]:
        z_t = alpha * float(demand_i) + (1 - alpha) * z_t

    # Smooth inter-demand intervals
    for interval_i in intervals[1:]:
        p_t = alpha * float(interval_i) + (1 - alpha) * p_t

    # SBA correction factor
    if p_t == 0:
        return pd.Series(np.nan, index=predict_months)

    sba_forecast: float = (z_t / p_t) * (1 - alpha / 2)
    sba_forecast = max(sba_forecast, 0.0)

    return pd.Series(sba_forecast, index=predict_months)


def _predict_auto_arima(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """Auto-ARIMA via pmdarima — gracefully skips if not installed."""
    try:
        import pmdarima as pm
    except ImportError:
        logger.info("pmdarima not installed; skipping Auto-ARIMA")
        return pd.Series(np.nan, index=predict_months)

    min_history: int = params.get("min_history", 24)

    if len(train_series) < min_history:
        logger.debug(
            "Auto-ARIMA requires %d observations, got %d",
            min_history,
            len(train_series),
        )
        return pd.Series(np.nan, index=predict_months)

    try:
        model = pm.auto_arima(
            train_series,
            max_p=params.get("max_p", 3),
            max_q=params.get("max_q", 3),
            seasonal=params.get("seasonal", True),
            m=params.get("m", 12),
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        forecast = model.predict(n_periods=len(predict_months))
        result = pd.Series(forecast, index=predict_months)
        return np.maximum(result, 0)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("Auto-ARIMA failure: %s", exc)
        return pd.Series(np.nan, index=predict_months)


def _predict_theta(
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """Theta method — simple decomposition-based forecasting."""
    from statsmodels.tsa.forecasting.theta import ThetaModel

    period: int = params.get("period", 12)
    min_history: int = params.get("min_history", 12)

    if len(train_series) < min_history:
        logger.debug(
            "Theta requires %d observations, got %d",
            min_history,
            len(train_series),
        )
        return pd.Series(np.nan, index=predict_months)

    try:
        model = ThetaModel(train_series, period=period)
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(predict_months))
        forecast.index = predict_months
        return np.maximum(forecast, 0)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        logger.warning("Theta model failure: %s", exc)
        return pd.Series(np.nan, index=predict_months)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_MODEL_DISPATCH: dict[str, Any] = {
    "holt_winters": _predict_holt_winters,
    "simple_es": _predict_simple_es,
    "croston_sba": _predict_croston_sba,
    "auto_arima": _predict_auto_arima,
    "theta": _predict_theta,
}


# ---------------------------------------------------------------------------
# Public predict entry point (single DFU)
# ---------------------------------------------------------------------------


def predict_statistical(
    model_id: str,
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.Series:
    """Predict demand for predict_months using a single DFU's training history.

    Args:
        model_id: One of 'holt_winters', 'simple_es', 'croston_sba',
                  'auto_arima', 'theta'.
        train_series: Series indexed by pd.Timestamp (startdate), values are qty.
                      Must be sorted by index, no gaps (fill zeros for missing
                      months).
        predict_months: List of future month timestamps to predict.
        params: Model-specific parameters from config.yaml.

    Returns:
        Series indexed by predict_months with predicted qty (non-negative).
        Returns Series of NaN if model fails.
    """
    fn = _MODEL_DISPATCH.get(model_id)
    if fn is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Valid models: {sorted(_MODEL_DISPATCH.keys())}"
        )
    return fn(train_series, predict_months, params)


# ---------------------------------------------------------------------------
# Core DFU worker (DFU-first: one loop per DFU, all algorithms applied inside)
# ---------------------------------------------------------------------------


def _run_all_models_for_dfu(
    sku_ck: str,
    train_series: pd.Series,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, Any],
) -> list[dict]:
    """Run all enabled models for a single DFU.

    Structured DFU-first: we pay the cost of preparing each DFU's series once
    and then apply every algorithm to it in a single pass, rather than looping
    DFUs separately for each algorithm.

    Returns:
        List of dicts with keys: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    results: list[dict] = []
    for model_id, params in enabled_models.items():
        try:
            forecast = predict_statistical(model_id, train_series, predict_months, params)
        except ValueError as exc:
            logger.warning("Model %s failed for DFU %s: %s", model_id, sku_ck, exc)
            continue
        if forecast.isna().all():
            continue
        for ts, val in forecast.items():
            if not np.isnan(val):
                results.append(
                    {
                        "sku_ck": sku_ck,
                        "startdate": ts,
                        FORECAST_QTY_COL: float(val),
                        "algorithm_id": model_id,
                    }
                )
    return results


def _fit_dfu_worker(item: tuple) -> list[dict]:
    """Parallel worker: compact args (no repeated pickling of models/months).

    Receives only (sku_ck, numpy_values, datetime_index); enabled_models and
    predict_months are pre-loaded into worker-process globals by _worker_init.
    """
    sku_ck, values, index = item
    train_series = pd.Series(values, index=pd.DatetimeIndex(index, freq="MS"))
    return _run_all_models_for_dfu(sku_ck, train_series, _WORKER_MONTHS, _WORKER_MODELS)


# ---------------------------------------------------------------------------
# Batch runner for all DFUs
# ---------------------------------------------------------------------------


def run_statistical_models(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
    n_workers: int = 8,
) -> pd.DataFrame:
    """Run all enabled statistical models across all DFUs.

    Args:
        sales_df: Sales data with columns [sku_ck, startdate, qty].
                  Must be the TRAINING portion only (already filtered to cutoff).
        predict_months: Months to predict.
        enabled_models: {model_id: params_dict} for enabled models.
        n_workers: Number of parallel workers.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
        One row per (DFU, predict_month, algorithm).
        DFU-algorithm combinations that failed are excluded.
    """
    required_cols = {"sku_ck", "startdate", "qty"}
    missing = required_cols - set(sales_df.columns)
    if missing:
        raise ValueError(f"sales_df missing required columns: {missing}")

    if not enabled_models:
        logger.warning("No enabled statistical models; returning empty DataFrame")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]
        )

    sku_cks = sales_df["sku_ck"].unique()
    n_total = len(sku_cks)

    # Determine the full monthly range for filling gaps
    all_months = sorted(sales_df["startdate"].unique())
    grouped = sales_df.groupby("sku_ck", sort=False)

    # Build compact task items: each task is (sku_ck, numpy_values, datetime_index).
    # enabled_models and predict_months are shared via worker initializer so they
    # are serialized once per worker process rather than once per DFU.
    task_items: list[tuple] = []
    for sku_ck in sku_cks:
        train_series = _fill_monthly_series(grouped.get_group(sku_ck), all_months)
        task_items.append((sku_ck, train_series.values, train_series.index))

    all_results: list[dict] = []
    failure_counts: dict[str, int] = dict.fromkeys(enabled_models, 0)
    n_done = 0

    use_sequential = n_total <= _SEQUENTIAL_THRESHOLD or n_workers <= 1
    logger.info(
        "Running %d statistical models across %d DFUs (%s, workers=%d)",
        len(enabled_models),
        n_total,
        "sequential" if use_sequential else f"{n_workers} workers",
        n_workers,
    )

    def _record(dfu_results: list[dict]) -> None:
        all_results.extend(dfu_results)
        models_with_results = {r["algorithm_id"] for r in dfu_results}
        for model_id in enabled_models:
            if model_id not in models_with_results:
                failure_counts[model_id] += 1

    if use_sequential:
        # Sequential fast-path: avoids process-pool startup overhead entirely.
        # Ideal for small DFU sets (e.g. single-location runs).
        for item in task_items:
            sku_ck, values, index = item
            train_series = pd.Series(values, index=pd.DatetimeIndex(index, freq="MS"))
            try:
                _record(_run_all_models_for_dfu(sku_ck, train_series, predict_months, enabled_models))
            except (ValueError, RuntimeError) as exc:
                logger.warning("DFU %s failed: %s", sku_ck, exc)
                for model_id in enabled_models:
                    failure_counts[model_id] += 1
            n_done += 1
            if n_done % 100 == 0:
                logger.info("Statistical models: %d/%d DFUs complete", n_done, n_total)
    else:
        # Parallel path: enabled_models and predict_months initialized once per
        # worker process; each task only carries compact numpy arrays.
        with ProcessPoolExecutor(
            max_workers=min(n_workers, n_total),
            initializer=_worker_init,
            initargs=(enabled_models, predict_months),
        ) as executor:
            futures = {executor.submit(_fit_dfu_worker, item): item[0] for item in task_items}
            for future in as_completed(futures):
                sku_ck = futures[future]
                try:
                    _record(future.result())
                except (ValueError, RuntimeError) as exc:
                    logger.warning("DFU %s worker failed: %s", sku_ck, exc)
                    for model_id in enabled_models:
                        failure_counts[model_id] += 1
                n_done += 1
                if n_done % 500 == 0:
                    logger.info("Statistical models: %d/%d DFUs complete", n_done, n_total)

    if n_done % (100 if use_sequential else 500) != 0:
        logger.info("Statistical models: %d/%d DFUs complete", n_done, n_total)

    # Log per-model failure counts
    for model_id, count in sorted(failure_counts.items()):
        if count > 0:
            logger.info(
                "Model '%s': %d/%d DFUs failed or had insufficient history",
                model_id,
                count,
                n_total,
            )

    if not all_results:
        logger.warning("All statistical model predictions failed; returning empty DataFrame")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]
        )

    result_df = pd.DataFrame(all_results)
    logger.info(
        "Statistical models produced %d prediction rows across %d DFUs",
        len(result_df),
        result_df["sku_ck"].nunique(),
    )
    return result_df
