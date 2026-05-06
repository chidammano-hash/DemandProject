"""Deep learning models for Advanced Expert Panel.

Models: N-BEATS, N-HiTS, TFT, DeepAR, TiDE, TCN, PatchTST, iTransformer.
Uses Nixtla NeuralForecast library. Gracefully skips if not installed.
"""

import logging
import os
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 30  # seconds between "still training" logs


def _fit_with_heartbeat(
    nf: Any, train_df: pd.DataFrame, model_id: str, val_size: int = 0
) -> None:
    """Run nf.fit() with periodic heartbeat logs every 30s.

    Args:
        val_size: Validation window size. Must be > 0 when early stopping is
            enabled (early_stop_patience_steps > 0). Default 0 = no validation.
    """
    done = threading.Event()
    t0 = time.time()

    def _heartbeat() -> None:
        while not done.wait(_HEARTBEAT_INTERVAL):
            logger.info("%s: still training... %.0fs elapsed", model_id, time.time() - t0)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        nf.fit(df=train_df, val_size=val_size)
    finally:
        done.set()


def _check_neuralforecast() -> bool:
    """Check if neuralforecast is available (lazy, avoids importing torch at module load)."""
    try:
        import neuralforecast  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_device() -> str:
    """Detect available compute device. Prioritizes Apple MPS on macOS.

    Uses DEMAND_GPU env var: 'on' | 'off' | 'auto' (default 'auto').
    On Apple Silicon, auto-detects MPS (Metal Performance Shaders).
    """
    gpu_setting = os.environ.get("DEMAND_GPU", "auto").lower()
    if gpu_setting == "off":
        return "cpu"

    try:
        import torch
        # Apple Silicon MPS — prioritize on macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS GPU detected — using Metal Performance Shaders")
            return "mps"
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    return "cpu"


def _build_model(model_id: str, params: dict[str, Any], accelerator: str = "cpu") -> Any:
    """Instantiate a NeuralForecast model from config params.

    Lazy-imports neuralforecast models to avoid loading torch at module level.
    """
    from neuralforecast.models import (
        NBEATS, NHITS, TFT, DeepAR, TiDE, TCN, PatchTST, iTransformer,
    )
    h = params.get("h", 6)
    input_size = params.get("input_size", 24)
    max_steps = params.get("max_steps", 500)
    batch_size = params.get("batch_size", 32)
    learning_rate = params.get("learning_rate", 0.001)

    # early_stop_patience_steps=-1 disables early stopping; NeuralForecast 3.x
    # raises "Set val_size>0 if early stopping is enabled" otherwise.
    early_stop = params.get("early_stop_patience_steps", -1)

    # A8: Robust scaler (median/IQR) is configurable per model. Use "robust" for
    # erratic segments — standard z-score scaling is distorted by demand spikes.
    scaler_type = params.get("scaler_type", "standard")

    common = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "random_seed": 42,
        "scaler_type": scaler_type,
        "early_stop_patience_steps": early_stop,
        # Pad short series with zeros so DFUs with < input_size months of
        # history can still be included rather than crashing training.
        "start_padding_enabled": True,
        # accelerator passed as direct kwarg → captured by **trainer_kwargs in
        # BaseModel → forwarded to Lightning Trainer (supports 'cpu', 'mps', 'gpu')
        "accelerator": accelerator,
    }

    if model_id == "nbeats":
        variant = params.get("variant", "generic")
        if variant == "interpretable":
            return NBEATS(**common, stack_types=["trend", "seasonality"])
        return NBEATS(**common)

    if model_id == "nhits":
        return NHITS(**common)

    if model_id == "tft":
        hidden_size = params.get("hidden_size", 64)
        return TFT(**common, hidden_size=hidden_size)

    if model_id == "deepar":
        return DeepAR(**common)

    if model_id == "tide":
        return TiDE(**common)

    if model_id == "tcn":
        return TCN(**common)

    if model_id == "patchtst":
        patch_len = params.get("patch_len", 12)
        return PatchTST(**common, patch_len=patch_len)

    if model_id == "itransformer":
        return iTransformer(**common, n_series=1)

    raise ValueError(f"Unknown DL model_id: {model_id}")


def _prepare_nf_dataframe(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Prepare data in NeuralForecast format: unique_id, ds, y.

    Returns:
        (train_df, predict_months) -- train_df has columns [unique_id, ds, y].
    """
    df = sales_df[["sku_ck", "startdate", "qty"]].copy()
    df = df.rename(columns={"sku_ck": "unique_id", "startdate": "ds", "qty": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    # psycopg3 returns NUMERIC columns as Python Decimal → object dtype in pandas.
    # NeuralForecast requires y to be float64; cast explicitly here.
    df["y"] = df["y"].astype(float)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df, predict_months


def run_dl_models(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
) -> pd.DataFrame:
    """Run all enabled deep learning models across all DFUs.

    Deep learning models train a *global* model across all DFUs simultaneously
    (unlike statistical models which are per-DFU). This is a key advantage --
    they learn cross-series patterns.

    Args:
        sales_df: Training sales with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.
        enabled_models: {model_id: params_dict} for enabled DL models.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
    """
    empty = pd.DataFrame(
        columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]
    )

    if not _check_neuralforecast():
        logger.warning("neuralforecast not installed; returning empty DataFrame")
        return empty

    from neuralforecast import NeuralForecast

    if not enabled_models:
        return empty

    if sales_df.empty:
        return empty

    # Filter DFUs with sufficient history
    min_global_history = min(
        m.get("min_history", 12) for m in enabled_models.values()
    )
    dfu_counts = sales_df.groupby("sku_ck")["startdate"].nunique()
    valid_dfus = dfu_counts[dfu_counts >= min_global_history].index
    filtered_sales = sales_df[sales_df["sku_ck"].isin(valid_dfus)].copy()

    if filtered_sales.empty:
        logger.warning("No DFUs with sufficient history for DL models")
        return empty

    logger.info(
        "DL models: %d DFUs with >= %d months history (from %d total)",
        len(valid_dfus), min_global_history, sales_df["sku_ck"].nunique(),
    )

    train_df, _ = _prepare_nf_dataframe(filtered_sales, predict_months)
    h = len(predict_months)
    device = _detect_device()
    # Map device string to PyTorch Lightning accelerator
    accelerator = device if device in ("cpu", "mps") else "gpu"
    logger.info(
        "DL compute device: %s (accelerator=%s) | train rows=%d series=%d h=%d",
        device, accelerator, len(train_df), train_df["unique_id"].nunique(), h,
    )

    all_results: list[pd.DataFrame] = []

    for model_id, params in enabled_models.items():
        t0 = time.time()
        logger.info("Training DL model: %s (h=%d, max_steps=%d, accelerator=%s)...",
                     model_id, h, params.get("max_steps", 500), accelerator)

        try:
            model = _build_model(model_id, params, accelerator=accelerator)
            nf = NeuralForecast(models=[model], freq="MS")
            early_stop = params.get("early_stop_patience_steps", -1)
            val_size = 3 if early_stop > 0 else 0
            _fit_with_heartbeat(nf, train_df, model_id, val_size=val_size)
            forecast_df = nf.predict()

            # NeuralForecast returns columns like 'NBEATS', 'NHITS', etc.
            # Find the forecast column (first non-index column)
            forecast_cols = [
                c for c in forecast_df.columns
                if c not in ("unique_id", "ds")
            ]
            if not forecast_cols:
                logger.warning("%s: no forecast column in output", model_id)
                continue

            forecast_col = forecast_cols[0]
            result = forecast_df.reset_index()
            result = result.rename(columns={
                "unique_id": "sku_ck",
                "ds": "startdate",
                forecast_col: FORECAST_QTY_COL,
            })
            raw_values = result[FORECAST_QTY_COL].values
            nan_count = int(np.isnan(raw_values).sum()) + int(np.isinf(raw_values).sum())
            if nan_count > 0:
                logger.warning(
                    "%s: %d/%d NaN/Inf predictions replaced with 0.0",
                    model_id, nan_count, len(raw_values),
                )
                raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)
            result[FORECAST_QTY_COL] = np.maximum(raw_values, 0.0)
            result["algorithm_id"] = model_id
            result = result[["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]]

            # Filter to only requested predict_months
            result["startdate"] = pd.to_datetime(result["startdate"])
            predict_ts = pd.to_datetime(predict_months)
            result = result[result["startdate"].isin(predict_ts)]

            all_results.append(result)
            elapsed = time.time() - t0
            logger.info(
                "%s: %d predictions for %d DFUs in %.1fs",
                model_id, len(result), result["sku_ck"].nunique(), elapsed,
            )

        except (ValueError, RuntimeError, ImportError) as exc:
            logger.warning("DL model %s failed: %s", model_id, exc)
            continue

    if not all_results:
        logger.warning("All DL model predictions failed")
        return empty

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        "DL models complete: %d predictions from %d models",
        len(combined), combined["algorithm_id"].nunique(),
    )
    return combined
