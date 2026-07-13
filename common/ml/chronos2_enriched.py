"""Chronos 2 Enriched forecast adapter with optional covariates."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.forecast_window import build_forecast_output_window

logger = logging.getLogger(__name__)


def _resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to a concrete device string (mps / cuda / cpu)."""
    if device_setting != "auto":
        return device_setting
    gpu_env = os.environ.get(
        "DEMAND_CHRONOS_GPU",
        os.environ.get("DEMAND_TORCH_GPU", os.environ.get("DEMAND_GPU", "auto")),
    ).lower()
    if gpu_env not in {"auto", "on", "off"}:
        raise ValueError("DEMAND_CHRONOS_GPU must be one of: auto, on, off")
    if gpu_env == "off":
        return "cpu"
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError as exc:
        if gpu_env == "on":
            raise RuntimeError("GPU acceleration was required but PyTorch is unavailable") from exc
    if gpu_env == "on":
        raise RuntimeError(
            "GPU acceleration was required but neither Apple MPS nor CUDA is available"
        )
    return "cpu"


# Module-level pipeline cache avoids reloading the model per timeframe.
_chronos_pipeline_cache: dict[str, Any] = {}


def _validate_forecast_values(values: np.ndarray, *, sku_ck: str) -> np.ndarray:
    """Reject invalid Chronos output before applying the nonnegative floor."""
    forecasts = np.asarray(values, dtype=float)
    if not np.isfinite(forecasts).all():
        raise RuntimeError(
            f"Chronos 2 Enriched returned non-finite predictions for DFU {sku_ck}"
        )
    return np.maximum(forecasts, 0.0)


def _check_chronos2() -> bool:
    try:
        from chronos import Chronos2Pipeline  # noqa: F401

        return True
    except ImportError:
        return False


# Past-only covariates: known for history, NOT known for future
_C2E_PAST_ONLY_COVARIATES = [
    "qty_lag_1",
    "qty_lag_2",
    "qty_lag_3",
    "qty_lag_6",
    "qty_lag_12",
    "rolling_mean_3m",
    "rolling_mean_6m",
    "rolling_mean_12m",
    "mom_growth",
    "demand_accel",
    "volatility_ratio",
    "croston_demand_size",
    "croston_demand_interval",
    "croston_probability",
]

# Known-future covariates: calendar/seasonal features computable for any date
_C2E_FUTURE_COVARIATES = [
    "month",
    "quarter",
    "is_quarter_end",
    "is_year_end",
    "days_in_month",
    "fourier_sin_12",
    "fourier_cos_12",
    "fourier_sin_6",
    "fourier_cos_6",
    "fourier_sin_4",
    "fourier_cos_4",
    "fourier_sin_3",
    "fourier_cos_3",
]

# Categorical past covariates (Chronos 2 supports these as numpy str arrays).
# ml_cluster stays metadata only; it is not passed as a model covariate.
_C2E_CAT_COVARIATES = ["brand", "region", "abc_vol"]


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


_OUTPUT_COLUMNS = ["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]


def _empty_forecasts() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


def _normalize_sales_history(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per DFU-month, filling missing months with zero demand."""
    required = {"sku_ck", "startdate", "qty"}
    missing = required - set(sales_df.columns)
    if missing:
        raise ValueError(f"Chronos sales input is missing columns: {sorted(missing)}")
    if sales_df["sku_ck"].isna().any():
        raise ValueError("Chronos sales input contains a null sku_ck")

    sales = sales_df[["sku_ck", "startdate", "qty"]].copy()
    sales["startdate"] = (
        pd.to_datetime(sales["startdate"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if sales["startdate"].isna().any():
        raise ValueError("Chronos sales input contains an invalid startdate")
    sales["qty"] = pd.to_numeric(sales["qty"], errors="coerce")
    if not np.isfinite(sales["qty"].to_numpy(dtype=float)).all():
        raise ValueError("Chronos sales input contains a non-finite quantity")
    sales = sales.groupby(["sku_ck", "startdate"], as_index=False, sort=False)["qty"].sum()

    configured_end = sales_df.attrs.get("history_end")
    if configured_end is None:
        history_end = pd.Timestamp(sales["startdate"].max())
    else:
        history_end = pd.Timestamp(configured_end)
        if pd.isna(history_end):
            raise ValueError("Chronos sales history_end is invalid")
        history_end = history_end.to_period("M").to_timestamp()
    if (sales["startdate"] > history_end).any():
        raise ValueError("Chronos sales input contains observations after history_end")

    complete_groups: list[pd.DataFrame] = []
    for sku_ck, group in sales.groupby("sku_ck", sort=False):
        calendar = pd.DataFrame(
            {
                "sku_ck": sku_ck,
                "startdate": pd.date_range(group["startdate"].min(), history_end, freq="MS"),
            }
        )
        complete = calendar.merge(
            group,
            on=["sku_ck", "startdate"],
            how="left",
            validate="one_to_one",
        )
        complete["qty"] = complete["qty"].fillna(0.0).astype(float)
        complete_groups.append(complete)

    return pd.concat(complete_groups, ignore_index=True)


def _prepare_feature_grid(
    feature_grid: pd.DataFrame | None,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame | None, list[str], list[str], list[str]]:
    """Normalize a feature grid and require exact coverage of history keys."""
    if feature_grid is None or feature_grid.empty:
        return None, [], [], []
    missing_keys = {"sku_ck", "startdate"} - set(feature_grid.columns)
    if missing_keys:
        raise ValueError(f"Chronos feature grid is missing columns: {sorted(missing_keys)}")

    grid = feature_grid.copy()
    if grid["sku_ck"].isna().any():
        raise ValueError("Chronos feature grid contains a null sku_ck")
    grid["startdate"] = (
        pd.to_datetime(grid["startdate"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if grid["startdate"].isna().any():
        raise ValueError("Chronos feature grid contains an invalid startdate")
    if grid.duplicated(["sku_ck", "startdate"]).any():
        raise ValueError("Chronos feature grid contains duplicate DFU-month rows")

    grid = grid.set_index(["sku_ck", "startdate"])
    history_keys = pd.MultiIndex.from_frame(history[["sku_ck", "startdate"]])
    missing_history_keys = history_keys[~history_keys.isin(grid.index)]
    if len(missing_history_keys):
        raise ValueError(
            "Chronos feature grid is missing "
            f"{len(missing_history_keys)} required DFU-month row(s)"
        )

    available_past = [column for column in _C2E_PAST_ONLY_COVARIATES if column in grid]
    available_future = [column for column in _C2E_FUTURE_COVARIATES if column in grid]
    available_cat = [column for column in _C2E_CAT_COVARIATES if column in grid]
    return grid, available_past, available_future, available_cat


def _as_numpy(result: Any) -> np.ndarray:
    """Move a model result to CPU before converting it to numpy."""
    value = result.detach() if hasattr(result, "detach") else result
    value = value.cpu() if hasattr(value, "cpu") else value
    value = value.numpy() if hasattr(value, "numpy") else value
    return np.asarray(value)


def run_chronos2_enriched(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
    feature_grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run Chronos 2E with optional historical and known-future covariates."""
    if sales_df.empty or not predict_months:
        return _empty_forecasts()

    normalized_sales = _normalize_sales_history(sales_df)
    configured_length = int(params["prediction_length"])
    if configured_length <= 0:
        raise ValueError("Chronos configured prediction_length must be positive")
    try:
        min_history = params["min_history"]
    except KeyError as exc:
        raise ValueError("Chronos min_history must be configured") from exc
    if (
        not isinstance(min_history, int)
        or isinstance(min_history, bool)
        or min_history <= 0
    ):
        raise ValueError("Chronos min_history must be a positive integer")
    window = build_forecast_output_window(
        predict_months,
        history_end=pd.Timestamp(normalized_sales["startdate"].max()),
        adapter_name="Chronos",
    )
    output_months = list(window.output_months)
    inference_months = list(window.inference_months)
    output_offset = window.output_offset

    if not _check_chronos2():
        raise RuntimeError(
            "chronos-forecasting >= 2.0 is required for Chronos 2 Enriched"
        )

    import torch
    from chronos import Chronos2Pipeline

    device_setting = params["device"]
    inference_length = len(inference_months)
    batch_size = int(params["batch_size"])
    if batch_size <= 0:
        raise ValueError("Chronos batch_size must be positive")

    device = _resolve_device(device_setting)
    if device == "mps":
        logger.info("Chronos 2 Enriched: using Apple MPS GPU")

    model_name = params.get("model_name")
    model_revision = params.get("model_revision")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Chronos model_name must be configured")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("Chronos model_revision must be pinned")
    cache_key = f"{model_name}:{model_revision}:{device}"
    if cache_key not in _chronos_pipeline_cache:
        logger.info("Chronos 2 Enriched: loading %s on %s...", model_name, device)
        model_dtype = torch.bfloat16 if device != "cpu" else torch.float32
        try:
            _chronos_pipeline_cache[cache_key] = Chronos2Pipeline.from_pretrained(
                model_name,
                revision=model_revision,
                device_map=device,
                dtype=model_dtype,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Chronos 2 Enriched model loading failed") from exc
    else:
        logger.info("Chronos 2 Enriched: reusing cached pipeline")
    pipeline = _chronos_pipeline_cache[cache_key]

    future_cal = _build_future_calendar(inference_months)
    grid, available_past, available_future, available_cat = _prepare_feature_grid(
        feature_grid,
        normalized_sales,
    )
    if grid is not None:
        logger.info(
            "Chronos 2 Enriched: %d past covariates, %d future covariates, %d categorical",
            len(available_past),
            len(available_future),
            len(available_cat),
        )
    else:
        available_past = []
        available_future = []
        available_cat = []
        logger.info("Chronos 2 Enriched: no feature grid — running with calendar covariates only")

    histories = {
        sku_ck: group.reset_index(drop=True)
        for sku_ck, group in normalized_sales.groupby("sku_ck", sort=False)
    }
    valid_skus = [
        sku_ck
        for sku_ck, history in histories.items()
        if len(history) >= min_history
    ]

    if not valid_skus:
        return _empty_forecasts()

    logger.info(
        "Chronos 2 Enriched: predicting %d DFUs (batch_size=%d)...", len(valid_skus), batch_size
    )

    # Process in chunks — build input dicts lazily per chunk to avoid massive memory spike
    month_arr = np.array(output_months)
    n_months = len(output_months)
    batch_dfs: list[pd.DataFrame] = []
    chunk_size = batch_size
    n_chunks = (len(valid_skus) + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, len(valid_skus))
        chunk_skus = valid_skus[c_start:c_end]

        # Build input dicts for THIS chunk only
        chunk_inputs: list[dict[str, Any]] = []
        for sku_ck in chunk_skus:
            history = histories[sku_ck]
            qty = history["qty"].to_numpy(dtype=np.float32)
            entry: dict[str, Any] = {"target": torch.from_numpy(qty.copy())}
            dates = history["startdate"].tolist()
            past_cov = _build_future_calendar(dates)

            if grid is not None:
                keys = pd.MultiIndex.from_frame(history[["sku_ck", "startdate"]])
                aligned = grid.reindex(keys)
                for column in available_past:
                    values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(
                        dtype=np.float32
                    )
                    if np.isinf(values).any():
                        raise ValueError(
                            "Chronos feature grid contains a non-finite covariate: "
                            f"{column}"
                        )
                    past_cov[column] = np.nan_to_num(values, nan=0.0)
                for column in available_future:
                    values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(
                        dtype=np.float32
                    )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            "Chronos feature grid contains a missing or non-finite "
                            f"known-future covariate: {column}"
                        )
                    past_cov[column] = values
                for column in available_cat:
                    past_cov[column] = (
                        aligned[column].fillna("__unknown__").astype(str).to_numpy()
                    )

            entry["past_covariates"] = past_cov
            entry["future_covariates"] = {
                key: future_cal[key]
                for key in _C2E_FUTURE_COVARIATES
                if key in past_cov
            }

            chunk_inputs.append(entry)

        try:
            chunk_results = pipeline.predict(
                chunk_inputs,
                prediction_length=inference_length,
                batch_size=chunk_size,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Chronos 2 Enriched chunk {ci + 1}/{n_chunks} failed"
            ) from exc

        if len(chunk_results) != len(chunk_skus):
            raise RuntimeError(
                "Chronos 2 Enriched returned a different number of results than inputs"
            )

        for sku_ck, r in zip(chunk_skus, chunk_results, strict=True):
            result_array = _as_numpy(r)
            if (
                result_array.ndim != 3
                or result_array.shape[0] < 1
                or result_array.shape[1] < 1
                or result_array.shape[2] < inference_length
            ):
                raise RuntimeError(
                    f"Chronos 2 Enriched returned an invalid result shape for DFU {sku_ck}"
                )
            median_idx = result_array.shape[1] // 2
            preds = result_array[
                0,
                median_idx,
                output_offset : output_offset + n_months,
            ]
            preds = _validate_forecast_values(preds, sku_ck=str(sku_ck))

            batch_dfs.append(
                pd.DataFrame(
                    {
                        "sku_ck": sku_ck,
                        "startdate": month_arr[: len(preds)],
                        FORECAST_QTY_COL: preds,
                        "algorithm_id": "chronos2_enriched",
                    }
                )
            )

        # Free chunk inputs immediately
        del chunk_inputs, chunk_results

        logger.info(
            "Chronos 2 Enriched: %d/%d DFUs processed",
            min(c_end, len(valid_skus)),
            len(valid_skus),
        )

    result = pd.concat(batch_dfs, ignore_index=True) if batch_dfs else _empty_forecasts()
    result.attrs.update(
        {
            "model_name": model_name,
            "model_revision": model_revision,
            "direct_horizon": configured_length,
            "prediction_horizon": len(output_months),
            "min_history": min_history,
        }
    )
    logger.info(
        "Chronos 2 Enriched: %d predictions for %d DFUs",
        len(result),
        result["sku_ck"].nunique() if not result.empty else 0,
    )
    return result
