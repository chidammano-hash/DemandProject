"""Foundation models for Advanced Expert Panel.

Models: Chronos-2, TimesFM, Moirai, TimeGPT, Lag-Llama.
Each supports zero-shot forecasting — no training required.
Gracefully skips models whose libraries are not installed.
"""

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _check_chronos() -> bool:
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


def _check_timesfm() -> bool:
    try:
        import timesfm  # noqa: F401
        return True
    except ImportError:
        return False


def _check_nixtla() -> bool:
    try:
        import nixtla  # noqa: F401
        return True
    except ImportError:
        return False


def _check_moirai() -> bool:
    try:
        import uni2ts  # noqa: F401
        return True
    except ImportError:
        return False


def _check_lag_llama() -> bool:
    try:
        import lag_llama  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to a concrete device string (mps / cuda / cpu)."""
    if device_setting != "auto":
        return device_setting
    gpu_env = os.environ.get("DEMAND_GPU", "auto").lower()
    if gpu_env == "off":
        return "cpu"
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Chronos
# ---------------------------------------------------------------------------

# Module-level pipeline cache — avoids reloading the model per timeframe.
_chronos_pipeline_cache: dict[str, Any] = {}


def _get_chronos_pipeline(model_name: str, device: str) -> Any:
    """Return a cached Chronos pipeline, loading only on first call.

    Automatically selects ChronosBoltPipeline for bolt models and
    ChronosPipeline for T5 models.
    """
    cache_key = f"{model_name}:{device}"
    if cache_key not in _chronos_pipeline_cache:
        import torch

        is_bolt = "bolt" in model_name
        if is_bolt:
            from chronos import ChronosBoltPipeline as PipelineCls
        else:
            from chronos import ChronosPipeline as PipelineCls

        logger.info("Chronos: loading %s on %s...", model_name, device)
        _chronos_pipeline_cache[cache_key] = PipelineCls.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
    else:
        logger.info("Chronos: reusing cached pipeline (%s on %s)", model_name, device)
    return _chronos_pipeline_cache[cache_key]


def _run_chronos(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Amazon Chronos foundation model (zero-shot)."""
    if not _check_chronos():
        logger.info("chronos-forecasting not installed; skipping Chronos")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    import torch

    model_size = params.get("model_size", "small")
    model_name = f"amazon/chronos-t5-{model_size}"
    device_setting = params.get("device", "auto")
    prediction_length = len(predict_months)
    num_samples = params.get("num_samples", 20)

    device = _resolve_device(device_setting)
    if device == "mps":
        logger.info("Chronos: using Apple MPS GPU")

    pipeline = _get_chronos_pipeline(model_name, device)

    sku_cks = sales_df["sku_ck"].unique()
    batch_dfs: list[pd.DataFrame] = []

    # Pre-sort once, then split — avoids per-SKU sort in the inner loop
    sorted_sales = sales_df.sort_values(["sku_ck", "startdate"])
    grouped = sorted_sales.groupby("sku_ck", sort=False)

    batch_size = params.get("batch_size", 32)
    sku_list = list(sku_cks)
    month_arr = np.array(predict_months)
    n_months = len(predict_months)

    for batch_start in range(0, len(sku_list), batch_size):
        batch_skus = sku_list[batch_start:batch_start + batch_size]
        contexts = []
        valid_skus = []

        for sku_ck in batch_skus:
            values = grouped.get_group(sku_ck)["qty"].values.astype(np.float32)
            if len(values) < 3:
                continue
            contexts.append(torch.tensor(values))
            valid_skus.append(sku_ck)

        if not contexts:
            continue

        try:
            forecasts = pipeline.predict(
                contexts,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
            # forecasts shape: (n_series, num_samples, prediction_length)
            median_forecasts = np.median(forecasts.numpy(), axis=1)

            # Validate NaN/Inf predictions
            bad_mask = ~np.isfinite(median_forecasts)
            if bad_mask.any():
                n_bad = int(bad_mask.sum())
                logger.warning(
                    "Chronos batch %d: %d NaN/Inf values replaced with 0.0",
                    batch_start, n_bad,
                )
                median_forecasts = np.nan_to_num(
                    median_forecasts, nan=0.0, posinf=0.0, neginf=0.0,
                )

            # Vectorized result construction — no per-row dict append
            n_valid = len(valid_skus)
            median_forecasts = np.maximum(median_forecasts[:, :n_months], 0.0)
            sku_rep = np.repeat(valid_skus, n_months)
            month_rep = np.tile(month_arr, n_valid)
            fcst_flat = median_forecasts.ravel()

            batch_dfs.append(pd.DataFrame({
                "sku_ck": sku_rep,
                "startdate": month_rep,
                "basefcst_pref": fcst_flat,
                "algorithm_id": "chronos",
            }))
        except (RuntimeError, ValueError) as exc:
            logger.warning("Chronos batch failed: %s", exc)

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            logger.info(
                "Chronos: %d/%d DFUs processed",
                min(batch_start + batch_size, len(sku_list)),
                len(sku_list),
            )

    result = pd.concat(batch_dfs, ignore_index=True) if batch_dfs else pd.DataFrame(
        columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    )
    logger.info("Chronos: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# Chronos Bolt (v2 — faster, native encoder architecture)
# ---------------------------------------------------------------------------

def _run_chronos_bolt(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Amazon Chronos Bolt foundation model (zero-shot).

    Bolt uses a native encoder architecture (not T5) and is up to 250x faster
    than Chronos-T5 Large with comparable accuracy. It returns quantile
    forecasts directly — no sampling required.
    """
    if not _check_chronos():
        logger.info("chronos-forecasting not installed; skipping Chronos Bolt")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    import torch

    model_size = params.get("model_size", "base")
    model_name = f"amazon/chronos-bolt-{model_size}"
    device_setting = params.get("device", "auto")
    prediction_length = len(predict_months)
    # Bolt supports num_samples but defaults to quantile output (faster)
    num_samples = params.get("num_samples", 12)

    device = _resolve_device(device_setting)
    if device == "mps":
        logger.info("Chronos Bolt: using Apple MPS GPU")

    pipeline = _get_chronos_pipeline(model_name, device)

    sku_cks = sales_df["sku_ck"].unique()
    batch_dfs: list[pd.DataFrame] = []

    sorted_sales = sales_df.sort_values(["sku_ck", "startdate"])
    grouped = sorted_sales.groupby("sku_ck", sort=False)

    batch_size = params.get("batch_size", 1024)
    sku_list = list(sku_cks)
    month_arr = np.array(predict_months)
    n_months = len(predict_months)

    for batch_start in range(0, len(sku_list), batch_size):
        batch_skus = sku_list[batch_start:batch_start + batch_size]
        contexts = []
        valid_skus = []

        for sku_ck in batch_skus:
            values = grouped.get_group(sku_ck)["qty"].values.astype(np.float32)
            if len(values) < 3:
                continue
            contexts.append(torch.tensor(values))
            valid_skus.append(sku_ck)

        if not contexts:
            continue

        try:
            forecasts = pipeline.predict(
                contexts,
                prediction_length=prediction_length,
            )
            # Bolt returns (n_series, n_quantiles, prediction_length)
            # Take the middle quantile as the point forecast
            mid_q = forecasts.shape[1] // 2
            median_forecasts = forecasts[:, mid_q, :].numpy()

            bad_mask = ~np.isfinite(median_forecasts)
            if bad_mask.any():
                n_bad = int(bad_mask.sum())
                logger.warning(
                    "Chronos Bolt batch %d: %d NaN/Inf values replaced with 0.0",
                    batch_start, n_bad,
                )
                median_forecasts = np.nan_to_num(
                    median_forecasts, nan=0.0, posinf=0.0, neginf=0.0,
                )

            n_valid = len(valid_skus)
            median_forecasts = np.maximum(median_forecasts[:, :n_months], 0.0)
            sku_rep = np.repeat(valid_skus, n_months)
            month_rep = np.tile(month_arr, n_valid)
            fcst_flat = median_forecasts.ravel()

            batch_dfs.append(pd.DataFrame({
                "sku_ck": sku_rep,
                "startdate": month_rep,
                "basefcst_pref": fcst_flat,
                "algorithm_id": "chronos_bolt",
            }))
        except (RuntimeError, ValueError) as exc:
            logger.warning("Chronos Bolt batch failed: %s", exc)

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            logger.info(
                "Chronos Bolt: %d/%d DFUs processed",
                min(batch_start + batch_size, len(sku_list)),
                len(sku_list),
            )

    result = pd.concat(batch_dfs, ignore_index=True) if batch_dfs else pd.DataFrame(
        columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    )
    logger.info("Chronos Bolt: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# Chronos 2 (latest generation — quantile output, built-in batching)
# ---------------------------------------------------------------------------

def _check_chronos2() -> bool:
    try:
        from chronos import Chronos2Pipeline  # noqa: F401
        return True
    except ImportError:
        return False


def _run_chronos2(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Amazon Chronos 2 foundation model (zero-shot).

    Chronos 2 handles batching internally via its batch_size parameter,
    returns a list of tensors (one per series) with 21 quantile forecasts.
    """
    if not _check_chronos2():
        logger.info("chronos-forecasting >= 2.0 not installed; skipping Chronos 2")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    import torch
    from chronos import Chronos2Pipeline

    device_setting = params.get("device", "auto")
    prediction_length = len(predict_months)
    batch_size = params.get("batch_size", 1024)

    device = _resolve_device(device_setting)
    if device == "mps":
        logger.info("Chronos 2: using Apple MPS GPU")

    model_name = "amazon/chronos-2"
    cache_key = f"{model_name}:{device}"
    if cache_key not in _chronos_pipeline_cache:
        logger.info("Chronos 2: loading %s on %s...", model_name, device)
        _chronos_pipeline_cache[cache_key] = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
    else:
        logger.info("Chronos 2: reusing cached pipeline")
    pipeline = _chronos_pipeline_cache[cache_key]

    sku_cks = sales_df["sku_ck"].unique()
    sorted_sales = sales_df.sort_values(["sku_ck", "startdate"])
    grouped = sorted_sales.groupby("sku_ck", sort=False)

    # Build contexts — filter DFUs with < 3 months of history
    contexts: list[torch.Tensor] = []
    valid_skus: list[str] = []
    for sku_ck in sku_cks:
        values = grouped.get_group(sku_ck)["qty"].values.astype(np.float32)
        if len(values) < 3:
            continue
        contexts.append(torch.tensor(values))
        valid_skus.append(sku_ck)

    if not contexts:
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    logger.info("Chronos 2: predicting %d DFUs (batch_size=%d)...",
                len(valid_skus), batch_size)

    month_arr = np.array(predict_months)
    n_months = len(predict_months)
    batch_dfs: list[pd.DataFrame] = []
    chunk_size = batch_size
    n_chunks = (len(contexts) + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, len(contexts))

        try:
            chunk_results = pipeline.predict(
                contexts[c_start:c_end],
                prediction_length=prediction_length,
                batch_size=chunk_size,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("Chronos 2 chunk %d/%d failed: %s", ci + 1, n_chunks, exc)
            continue

        for sku_ck, r in zip(valid_skus[c_start:c_end], chunk_results):
            n_q = r.shape[1]
            median_idx = n_q // 2
            preds = r[0, median_idx, :n_months].numpy()
            preds = np.maximum(preds, 0.0)
            if not np.all(np.isfinite(preds)):
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

            batch_dfs.append(pd.DataFrame({
                "sku_ck": sku_ck,
                "startdate": month_arr[:len(preds)],
                "basefcst_pref": preds,
                "algorithm_id": "chronos2",
            }))

        logger.info("Chronos 2: %d/%d DFUs processed",
                     min(c_end, len(valid_skus)), len(valid_skus))

    result = pd.concat(batch_dfs, ignore_index=True) if batch_dfs else pd.DataFrame(
        columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    )
    logger.info("Chronos 2: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# Chronos 2 Enriched (with covariates from feature matrix)
# ---------------------------------------------------------------------------

# Past-only covariates: known for history, NOT known for future
_C2E_PAST_ONLY_COVARIATES = [
    "qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_6", "qty_lag_12",
    "qty_rolling_mean_3", "qty_rolling_mean_6", "qty_rolling_mean_12",
    "mom_growth", "demand_accel", "volatility_ratio",
    "croston_demand_size", "croston_demand_interval", "croston_probability",
    "cluster_mean_lag1", "cluster_total_lag1", "cluster_demand_trend",
]

# Known-future covariates: calendar/seasonal features computable for any date
_C2E_FUTURE_COVARIATES = [
    "month", "quarter", "is_quarter_end", "is_year_end", "days_in_month",
    "fourier_sin_12", "fourier_cos_12", "fourier_sin_6", "fourier_cos_6",
    "fourier_sin_4", "fourier_cos_4", "fourier_sin_3", "fourier_cos_3",
]

# Categorical past covariates (Chronos 2 supports these as numpy str arrays)
_C2E_CAT_COVARIATES = ["ml_cluster", "brand", "region", "abc_vol"]


def _build_future_calendar(predict_months: list[pd.Timestamp]) -> dict[str, np.ndarray]:
    """Build known-future covariate arrays for the predict window."""
    months = np.array([m.month for m in predict_months], dtype=np.float32)
    quarters = np.array([m.quarter for m in predict_months], dtype=np.float32)
    is_qtr_end = np.isin(months, [3, 6, 9, 12]).astype(np.float32)
    is_yr_end = (months == 12).astype(np.float32)
    days = np.array([m.days_in_month for m in predict_months], dtype=np.float32)

    future: dict[str, np.ndarray] = {
        "month": months,
        "quarter": quarters,
        "is_quarter_end": is_qtr_end,
        "is_year_end": is_yr_end,
        "days_in_month": days,
    }

    # Fourier features for predict months
    for period in [12, 6, 4, 3]:
        angles = 2.0 * np.pi * months / period
        future[f"fourier_sin_{period}"] = np.sin(angles).astype(np.float32)
        future[f"fourier_cos_{period}"] = np.cos(angles).astype(np.float32)

    return future


def _run_chronos2_enriched(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
    feature_grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run Chronos 2 with covariates from feature matrix.

    Uses past_covariates (lag/rolling/croston/cluster features + categoricals)
    and future_covariates (calendar/fourier — known for any future date).
    """
    if not _check_chronos2():
        logger.info("chronos-forecasting >= 2.0 not installed; skipping Chronos 2 Enriched")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    import torch
    from chronos import Chronos2Pipeline

    device_setting = params.get("device", "auto")
    prediction_length = len(predict_months)
    batch_size = params.get("batch_size", 2048)

    device = _resolve_device(device_setting)
    if device == "mps":
        logger.info("Chronos 2 Enriched: using Apple MPS GPU")

    model_name = "amazon/chronos-2"
    cache_key = f"{model_name}:{device}"
    if cache_key not in _chronos_pipeline_cache:
        logger.info("Chronos 2 Enriched: loading %s on %s...", model_name, device)
        _chronos_pipeline_cache[cache_key] = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
    else:
        logger.info("Chronos 2 Enriched: reusing cached pipeline")
    pipeline = _chronos_pipeline_cache[cache_key]

    # Build future calendar covariates (same for all DFUs)
    future_cal = _build_future_calendar(predict_months)

    sku_cks = sales_df["sku_ck"].unique()
    sorted_sales = sales_df.sort_values(["sku_ck", "startdate"])
    grouped_sales = sorted_sales.groupby("sku_ck", sort=False)

    # Prepare feature grid lookup if provided
    has_grid = feature_grid is not None and not feature_grid.empty
    if has_grid:
        grid_sorted = feature_grid.sort_values(["sku_ck", "startdate"])
        grouped_grid = grid_sorted.groupby("sku_ck", sort=False)
        # Determine which covariates are actually present
        available_past = [c for c in _C2E_PAST_ONLY_COVARIATES if c in grid_sorted.columns]
        available_future = [c for c in _C2E_FUTURE_COVARIATES if c in grid_sorted.columns]
        available_cat = [c for c in _C2E_CAT_COVARIATES if c in grid_sorted.columns]
        all_past_keys = available_past + available_future + available_cat
        logger.info(
            "Chronos 2 Enriched: %d past covariates, %d future covariates, %d categorical",
            len(available_past), len(available_future), len(available_cat),
        )
    else:
        available_past = []
        available_future = []
        available_cat = []
        all_past_keys = []
        logger.info("Chronos 2 Enriched: no feature grid — running with calendar covariates only")

    # Build input dicts per DFU — vectorized approach
    # Pre-compute per-DFU row ranges from sorted sales to avoid get_group() calls
    t_build = time.time()

    # Get row boundaries per DFU from sorted sales
    sales_skus = sorted_sales["sku_ck"].values
    sales_qty = sorted_sales["qty"].values.astype(np.float32)
    sku_boundaries: dict[str, tuple[int, int]] = {}
    prev_sku = None
    start_idx = 0
    for i, sku in enumerate(sales_skus):
        if sku != prev_sku:
            if prev_sku is not None:
                sku_boundaries[prev_sku] = (start_idx, i)
            start_idx = i
            prev_sku = sku
    if prev_sku is not None:
        sku_boundaries[prev_sku] = (start_idx, len(sales_skus))

    # Pre-extract grid covariate arrays as contiguous numpy blocks
    grid_boundaries: dict[str, tuple[int, int]] = {}
    grid_numeric_arrays: dict[str, np.ndarray] = {}
    grid_cat_arrays: dict[str, np.ndarray] = {}
    if has_grid:
        grid_skus = grid_sorted["sku_ck"].values
        prev_sku = None
        start_idx = 0
        for i, sku in enumerate(grid_skus):
            if sku != prev_sku:
                if prev_sku is not None:
                    grid_boundaries[prev_sku] = (start_idx, i)
                start_idx = i
                prev_sku = sku
        if prev_sku is not None:
            grid_boundaries[prev_sku] = (start_idx, len(grid_skus))

        # Extract all numeric covariate columns as contiguous arrays
        all_numeric_cols = available_past + available_future
        for col in all_numeric_cols:
            arr = grid_sorted[col].values.astype(np.float32)
            np.nan_to_num(arr, copy=False, nan=0.0)
            grid_numeric_arrays[col] = arr
        for col in available_cat:
            grid_cat_arrays[col] = grid_sorted[col].fillna("__unknown__").astype(str).values

    # Future covariate keys that will be present in past_covariates
    future_keys_available = [k for k in _C2E_FUTURE_COVARIATES if k in future_cal]

    inputs: list[dict[str, Any]] = []
    valid_skus: list[str] = []

    for sku_ck in sku_cks:
        if sku_ck not in sku_boundaries:
            continue
        s, e = sku_boundaries[sku_ck]
        hist_len = e - s
        if hist_len < 3:
            continue

        entry: dict[str, Any] = {"target": torch.from_numpy(sales_qty[s:e].copy())}

        # Build past_covariates via array slicing (no get_group / per-col DataFrame access)
        past_cov: dict[str, np.ndarray] = {}

        if has_grid and sku_ck in grid_boundaries:
            gs, ge = grid_boundaries[sku_ck]
            # Align: take last hist_len rows from grid
            g_len = ge - gs
            g_start = ge - hist_len if g_len >= hist_len else gs
            g_slice_len = min(hist_len, g_len)

            if g_slice_len == hist_len:
                for col, arr in grid_numeric_arrays.items():
                    past_cov[col] = arr[g_start:ge]
                for col, arr in grid_cat_arrays.items():
                    past_cov[col] = arr[g_start:ge]
        else:
            # Fallback: calendar covariates from sales dates
            sale_dates = sorted_sales["startdate"].values[s:e]
            m_arr = pd.to_datetime(sale_dates).month.values.astype(np.float32)
            for period in [12, 6, 4, 3]:
                angles = 2.0 * np.pi * m_arr / period
                past_cov[f"fourier_sin_{period}"] = np.sin(angles).astype(np.float32)
                past_cov[f"fourier_cos_{period}"] = np.cos(angles).astype(np.float32)
            past_cov["month"] = m_arr
            past_cov["quarter"] = np.ceil(m_arr / 3).astype(np.float32)

        if past_cov:
            entry["past_covariates"] = past_cov

        # Future covariates (subset of past keys)
        future_cov = {k: future_cal[k] for k in future_keys_available if k in past_cov}
        if future_cov:
            entry["future_covariates"] = future_cov

        inputs.append(entry)
        valid_skus.append(sku_ck)

    logger.info("Input dicts built: %d DFUs (%.1fs)", len(valid_skus), time.time() - t_build)

    if not inputs:
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    logger.info("Chronos 2 Enriched: predicting %d DFUs (batch_size=%d)...",
                len(valid_skus), batch_size)

    # Feed in chunks — Chronos 2's internal collation chokes on 200K+ dicts at once
    month_arr = np.array(predict_months)
    n_months = len(predict_months)
    batch_dfs: list[pd.DataFrame] = []
    chunk_size = batch_size  # process one GPU batch worth at a time
    n_chunks = (len(inputs) + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, len(inputs))
        chunk_inputs = inputs[c_start:c_end]
        chunk_skus = valid_skus[c_start:c_end]

        try:
            chunk_results = pipeline.predict(
                chunk_inputs,
                prediction_length=prediction_length,
                batch_size=chunk_size,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("Chronos 2 Enriched chunk %d/%d failed: %s", ci + 1, n_chunks, exc)
            continue

        # Extract median forecasts — vectorized per chunk
        for sku_ck, r in zip(chunk_skus, chunk_results):
            n_q = r.shape[1]
            median_idx = n_q // 2
            preds = r[0, median_idx, :n_months].numpy()
            preds = np.maximum(preds, 0.0)
            if not np.all(np.isfinite(preds)):
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

            batch_dfs.append(pd.DataFrame({
                "sku_ck": sku_ck,
                "startdate": month_arr[:len(preds)],
                "basefcst_pref": preds,
                "algorithm_id": "chronos2_enriched",
            }))

        logger.info(
            "Chronos 2 Enriched: %d/%d DFUs processed",
            min(c_end, len(valid_skus)), len(valid_skus),
        )

    result = pd.concat(batch_dfs, ignore_index=True) if batch_dfs else pd.DataFrame(
        columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    )
    logger.info("Chronos 2 Enriched: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# TimesFM
# ---------------------------------------------------------------------------

def _run_timesfm(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Google TimesFM foundation model (zero-shot)."""
    if not _check_timesfm():
        logger.info("timesfm not installed; skipping TimesFM")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    import timesfm

    prediction_length = len(predict_months)
    context_length = params.get("context_length", 512)

    logger.info("TimesFM: loading model (context=%d)...", context_length)
    tfm = timesfm.TimesFm(
        context_len=context_length,
        horizon_len=prediction_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    tfm.load_from_checkpoint()

    grouped = sales_df.groupby("sku_ck", sort=False)
    all_results: list[dict] = []

    for sku_ck, group in grouped:
        values = group.sort_values("startdate")["qty"].values.astype(np.float32)
        if len(values) < 6:
            continue

        try:
            forecast, _ = tfm.forecast([values], freq=[0])
            preds = forecast[0][:prediction_length]

            for j, month in enumerate(predict_months):
                if j < len(preds):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(preds[j]), 0.0),
                        "algorithm_id": "timesfm",
                    })
        except (RuntimeError, ValueError) as exc:
            logger.warning("TimesFM failed for %s: %s", sku_ck, exc)

    result = pd.DataFrame(all_results)
    logger.info("TimesFM: %d predictions for %d DFUs", len(result),
                result["sku_ck"].nunique() if not result.empty else 0)
    return result


# ---------------------------------------------------------------------------
# TimeGPT
# ---------------------------------------------------------------------------

def _run_timegpt(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Nixtla TimeGPT (API-based, zero-shot)."""
    if not _check_nixtla():
        logger.info("nixtla not installed; skipping TimeGPT")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    api_key = os.environ.get("NIXTLA_API_KEY")
    if not api_key:
        logger.info("NIXTLA_API_KEY not set; skipping TimeGPT")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    from nixtla import NixtlaClient

    prediction_length = len(predict_months)
    logger.info("TimeGPT: calling API (h=%d)...", prediction_length)

    client = NixtlaClient(api_key=api_key)

    df = sales_df[["sku_ck", "startdate", "qty"]].copy()
    df = df.rename(columns={"sku_ck": "unique_id", "startdate": "ds", "qty": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    try:
        forecast_df = client.forecast(
            df=df,
            h=prediction_length,
            freq="MS",
            time_col="ds",
            target_col="y",
            id_col="unique_id",
        )

        result = forecast_df.rename(columns={
            "unique_id": "sku_ck",
            "ds": "startdate",
            "TimeGPT": "basefcst_pref",
        })
        result["basefcst_pref"] = np.maximum(result["basefcst_pref"].values, 0.0)
        result["algorithm_id"] = "timegpt"
        result = result[["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]]

        # Filter to requested predict months
        result["startdate"] = pd.to_datetime(result["startdate"])
        predict_ts = pd.to_datetime(predict_months)
        result = result[result["startdate"].isin(predict_ts)]

        logger.info("TimeGPT: %d predictions for %d DFUs", len(result),
                     result["sku_ck"].nunique() if not result.empty else 0)
        return result

    except (RuntimeError, ValueError) as exc:
        logger.warning("TimeGPT API call failed: %s", exc)
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )


# ---------------------------------------------------------------------------
# Moirai
# ---------------------------------------------------------------------------

def _run_moirai(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Salesforce Moirai foundation model (zero-shot).

    Requires: pip install "uni2ts[torch]"
    Weights downloaded from HuggingFace on first run and cached locally.
    No API key required.
    """
    if not _check_moirai():
        logger.info("uni2ts not installed; skipping Moirai. Install: pip install 'uni2ts[torch]'")
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    model_size = params.get("model_size", "small")
    device = _resolve_device(params.get("device", "auto"))
    prediction_length = len(predict_months)
    context_length = params.get("context_length", 200)
    num_samples = params.get("num_samples", 20)
    batch_size = params.get("batch_size", 32)
    hf_model_id = f"Salesforce/moirai-1.1-R-{model_size}"

    logger.info("Moirai: loading %s on %s...", hf_model_id, device)
    try:
        module = MoiraiModule.from_pretrained(hf_model_id)
        model = MoiraiForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model = model.to(device)
        model.eval()
    except (RuntimeError, OSError) as exc:
        logger.warning("Moirai: failed to load model: %s", exc)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    sku_list = list(sales_df["sku_ck"].unique())
    grouped = sales_df.groupby("sku_ck", sort=False)
    all_results: list[dict] = []

    for batch_start in range(0, len(sku_list), batch_size):
        batch_skus = sku_list[batch_start: batch_start + batch_size]
        contexts: list[torch.Tensor] = []
        valid_skus: list[str] = []

        for sku_ck in batch_skus:
            group = grouped.get_group(sku_ck).sort_values("startdate")
            values = group["qty"].values.astype(np.float32)
            if len(values) < 3:
                continue
            ctx = values[-context_length:]
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
            valid_skus.append(sku_ck)

        if not contexts:
            continue

        # Pad to uniform length within the batch
        max_len = max(t.shape[0] for t in contexts)
        padded = torch.zeros(len(contexts), max_len, 1)
        observed = torch.zeros(len(contexts), max_len, 1, dtype=torch.bool)
        is_pad = torch.ones(len(contexts), max_len, dtype=torch.bool)

        for i, ctx_tensor in enumerate(contexts):
            L = ctx_tensor.shape[0]
            padded[i, -L:, 0] = ctx_tensor
            observed[i, -L:, 0] = True
            is_pad[i, -L:] = False

        padded = padded.to(device)
        observed = observed.to(device)
        is_pad = is_pad.to(device)

        try:
            with torch.no_grad():
                samples = model(
                    past_target=padded,
                    past_observed_target=observed,
                    past_is_pad=is_pad,
                )
            # samples: (batch, num_samples, prediction_length)
            medians = samples.median(dim=1).values.cpu().numpy()

            for i, sku_ck in enumerate(valid_skus):
                for j, month in enumerate(predict_months):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(medians[i, j]), 0.0),
                        "algorithm_id": "moirai",
                    })
        except (RuntimeError, ValueError) as exc:
            logger.warning("Moirai batch %d failed: %s", batch_start, exc)

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            logger.info(
                "Moirai: %d/%d DFUs processed",
                min(batch_start + batch_size, len(sku_list)), len(sku_list),
            )

    result = pd.DataFrame(all_results)
    logger.info(
        "Moirai: %d predictions for %d DFUs",
        len(result), result["sku_ck"].nunique() if not result.empty else 0,
    )
    return result


# ---------------------------------------------------------------------------
# Lag-Llama
# ---------------------------------------------------------------------------

def _run_lag_llama(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run Lag-Llama foundation model (zero-shot).

    Requires: pip install "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama"
              pip install huggingface_hub gluonts
    Checkpoint downloaded from HuggingFace on first run and cached locally.
    No API key required.
    """
    if not _check_lag_llama():
        logger.info(
            "lag_llama not installed; skipping Lag-Llama. "
            "Install: pip install 'lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama'"
        )
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    import torch
    from gluonts.dataset.common import ListDataset
    from huggingface_hub import hf_hub_download
    from lag_llama.gluon.estimator import LagLlamaEstimator

    device = _resolve_device(params.get("device", "auto"))
    prediction_length = len(predict_months)
    context_length = params.get("context_length", 32)
    num_samples = params.get("num_samples", 20)
    batch_size = params.get("batch_size", 64)

    logger.info("Lag-Llama: downloading checkpoint from HuggingFace...")
    try:
        ckpt_path = hf_hub_download(
            "time-series-foundation-models/Lag-Llama", filename="lag-llama.ckpt"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_kwargs = ckpt["hyper_parameters"]["model_kwargs"]
    except (OSError, KeyError) as exc:
        logger.warning("Lag-Llama: failed to load checkpoint: %s", exc)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    logger.info("Lag-Llama: building estimator (device=%s)...", device)
    try:
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=context_length,
            **model_kwargs,
            num_parallel_samples=num_samples,
            batch_size=batch_size,
            nonnegative_pred_samples=True,
            device=torch.device(device),
        )
        lightning_module = estimator.load_from_checkpoint(checkpoint_path=ckpt_path)
        predictor = estimator.create_lightning_predictor(
            lightning_module, batch_size=batch_size
        )
    except (RuntimeError, TypeError) as exc:
        logger.warning("Lag-Llama: estimator build failed: %s", exc)
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    # Build GluonTS ListDataset — one entry per DFU, in sku_ck order
    sku_order: list[str] = []
    entries: list[dict] = []
    for sku_ck, group in sales_df.groupby("sku_ck", sort=False):
        group = group.sort_values("startdate")
        if len(group) < 3:
            continue
        sku_order.append(str(sku_ck))
        entries.append({
            "start": pd.Period(group["startdate"].iloc[0], freq="M"),
            "target": group["qty"].values.astype(np.float32),
        })

    if not entries:
        return pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"])

    test_data = ListDataset(entries, freq="M")

    all_results: list[dict] = []
    try:
        for sku_ck, forecast in zip(sku_order, predictor.predict(test_data)):
            # forecast.samples: (num_samples, prediction_length)
            median_preds = np.median(forecast.samples, axis=0)
            for j, month in enumerate(predict_months):
                if j < len(median_preds):
                    all_results.append({
                        "sku_ck": sku_ck,
                        "startdate": month,
                        "basefcst_pref": max(float(median_preds[j]), 0.0),
                        "algorithm_id": "lag_llama",
                    })
    except (RuntimeError, StopIteration) as exc:
        logger.warning("Lag-Llama: prediction failed: %s", exc)

    result = pd.DataFrame(all_results)
    logger.info(
        "Lag-Llama: %d predictions for %d DFUs",
        len(result), result["sku_ck"].nunique() if not result.empty else 0,
    )
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FOUNDATION_DISPATCH: dict[str, Any] = {
    "chronos": _run_chronos,
    "chronos_bolt": _run_chronos_bolt,
    "chronos2": _run_chronos2,
    "chronos2_enriched": _run_chronos2_enriched,
    "timesfm": _run_timesfm,
    "timegpt": _run_timegpt,
    "moirai": _run_moirai,
    "lag_llama": _run_lag_llama,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_foundation_models(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
    feature_grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all enabled foundation models.

    Foundation models are zero-shot — they produce forecasts without any
    training on the target dataset. They are pretrained on large-scale
    time series corpora.

    Args:
        sales_df: Historical sales with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.
        enabled_models: {model_id: params_dict} for enabled foundation models.
        feature_grid: Optional feature matrix for enriched models (e.g. chronos2_enriched).

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    empty = pd.DataFrame(
        columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    )

    if not enabled_models:
        return empty

    if sales_df.empty:
        return empty

    all_results: list[pd.DataFrame] = []

    for model_id, params in enabled_models.items():
        fn = _FOUNDATION_DISPATCH.get(model_id)
        if fn is None:
            logger.info("Foundation model '%s' not yet implemented; skipping", model_id)
            continue

        t0 = time.time()
        logger.info("Running foundation model: %s...", model_id)

        try:
            # Pass feature_grid for enriched models that accept it
            if model_id == "chronos2_enriched":
                result = fn(sales_df, predict_months, params, feature_grid=feature_grid)
            else:
                result = fn(sales_df, predict_months, params)
            if not result.empty:
                all_results.append(result)
                logger.info(
                    "%s complete: %d predictions in %.1fs",
                    model_id, len(result), time.time() - t0,
                )
            else:
                logger.info("%s: no predictions produced", model_id)
        except (RuntimeError, ValueError, ImportError) as exc:
            logger.warning("Foundation model %s failed: %s", model_id, exc)

    if not all_results:
        return empty

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        "Foundation models complete: %d predictions from %d models",
        len(combined), combined["algorithm_id"].nunique(),
    )
    return combined
