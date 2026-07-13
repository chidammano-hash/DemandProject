"""NeuralForecast adapter for the canonical N-HiTS and N-BEATS models."""

import hashlib
import logging
import os
import platform
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.forecast_window import ForecastOutputWindow, build_forecast_output_window
from common.ml.monthly_history import complete_monthly_history

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 30  # seconds between "still training" logs
SUPPORTED_NEURAL_MODELS = frozenset({"nhits", "nbeats"})
NEURAL_TRAINING_CONTRACT_VERSION = "calendar-complete-neural-training-v1"
_OUTPUT_COLUMNS = ["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]
_REQUIRED_PARAMS = frozenset(
    {
        "h",
        "input_size",
        "max_steps",
        "batch_size",
        "learning_rate",
        "scaler_type",
        "early_stop_patience_steps",
        "min_history",
        "random_seed",
        "start_padding_enabled",
        "val_size",
        "deterministic",
    }
)


@dataclass(frozen=True)
class NeuralCohortIdentity:
    """Deterministic identity of an exact eligible DFU roster."""

    checksum: str
    dfu_count: int


@dataclass(frozen=True)
class NeuralTrainingLineage:
    """Identity of the exact normalized frame and runtime used for a final fit."""

    training_dfu_count: int
    training_row_count: int
    training_cohort_checksum: str
    training_data_checksum: str
    training_contract_version: str
    runtime_contract: dict[str, str]


@dataclass(frozen=True)
class FittedNeuralModel:
    """One fitted global model plus the inference-critical training contract."""

    neural_forecast: Any
    model_id: str
    fitted_horizon: int
    min_history: int
    training_dfu_count: int
    training_row_count: int = 0
    training_cohort_checksum: str = ""
    training_data_checksum: str = ""
    training_contract_version: str = ""
    runtime_contract: Mapping[str, str] = field(default_factory=dict)


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def current_neural_runtime_contract() -> dict[str, str]:
    """Return serialization-critical library versions without importing model weights."""
    return {
        "neuralforecast": _installed_version("neuralforecast"),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "python": platform.python_version(),
    }


def _normalize_runtime_contract(runtime_contract: Mapping[str, str]) -> dict[str, str]:
    required = {"neuralforecast", "numpy", "pandas", "python"}
    if set(runtime_contract) != required:
        raise ValueError(
            "Neural runtime contract must contain exactly " f"{sorted(required)}"
        )
    normalized: dict[str, str] = {}
    for key in sorted(runtime_contract):
        value = runtime_contract[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Neural runtime contract {key!r} must be a non-empty string")
        normalized[key] = value.strip()
    return normalized


def _update_length_prefixed(digest: Any, value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def compute_neural_cohort_identity(
    sku_cks: Iterable[object],
    *,
    presorted: bool = False,
) -> NeuralCohortIdentity:
    """Hash an exact DFU roster with unambiguous, Unicode-safe serialization."""
    def _normalized_values() -> Iterable[str]:
        for value in sku_cks:
            if value is None:
                raise ValueError("Neural training cohort contains a null sku_ck")
            yield str(value)

    normalized = _normalized_values()
    values: Iterable[str] = normalized if presorted else sorted(normalized)
    digest = hashlib.sha256()
    _update_length_prefixed(digest, NEURAL_TRAINING_CONTRACT_VERSION)
    previous: str | None = None
    count = 0
    for sku_ck in values:
        if not sku_ck.strip():
            raise ValueError("Neural training cohort contains a blank sku_ck")
        if previous is not None and sku_ck <= previous:
            reason = "duplicate" if sku_ck == previous else "not sorted"
            raise ValueError(f"Neural training cohort is {reason}: {sku_ck!r}")
        _update_length_prefixed(digest, sku_ck)
        previous = sku_ck
        count += 1
    if count <= 0:
        raise ValueError("Neural training cohort must not be empty")
    return NeuralCohortIdentity(checksum=digest.hexdigest(), dfu_count=count)


def _fit_with_heartbeat(nf: Any, train_df: pd.DataFrame, model_id: str, val_size: int = 0) -> None:
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

    Uses DEMAND_NEURAL_GPU, falling back to the shared GPU setting.
    On Apple Silicon, auto-detects MPS (Metal Performance Shaders).
    """
    gpu_setting = os.environ.get(
        "DEMAND_NEURAL_GPU",
        os.environ.get("DEMAND_TORCH_GPU", os.environ.get("DEMAND_GPU", "auto")),
    ).lower()
    if gpu_setting not in {"auto", "on", "off"}:
        raise ValueError("DEMAND_NEURAL_GPU must be one of: auto, on, off")
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
    except ImportError as exc:
        if gpu_setting == "on":
            raise RuntimeError("GPU acceleration was required but PyTorch is unavailable") from exc
    if gpu_setting == "on":
        raise RuntimeError(
            "GPU acceleration was required but neither Apple MPS nor CUDA is available"
        )
    return "cpu"


def _validate_model_params(model_id: str, params: dict[str, Any]) -> None:
    if model_id not in SUPPORTED_NEURAL_MODELS:
        raise ValueError(f"Unsupported neural model: {model_id}")

    missing = _REQUIRED_PARAMS - set(params)
    if missing:
        raise ValueError(f"{model_id} parameters are missing: {sorted(missing)}")
    unsupported = set(params) - _REQUIRED_PARAMS
    if unsupported:
        raise ValueError(f"{model_id} parameters are unsupported: {sorted(unsupported)}")

    for name in ("h", "input_size", "max_steps", "batch_size", "min_history"):
        if not isinstance(params[name], int) or isinstance(params[name], bool):
            raise ValueError(f"{model_id}.{name} must be an integer")
        if params[name] <= 0:
            raise ValueError(f"{model_id}.{name} must be positive")
    for name in ("random_seed", "early_stop_patience_steps"):
        if not isinstance(params[name], int) or isinstance(params[name], bool):
            raise ValueError(f"{model_id}.{name} must be an integer")
    if params["random_seed"] < 0:
        raise ValueError(f"{model_id}.random_seed cannot be negative")
    if params["early_stop_patience_steps"] < -1:
        raise ValueError(f"{model_id}.early_stop_patience_steps cannot be less than -1")
    if not isinstance(params["val_size"], int) or isinstance(params["val_size"], bool):
        raise ValueError(f"{model_id}.val_size must be an integer")
    if params["val_size"] < 0:
        raise ValueError(f"{model_id}.val_size cannot be negative")
    if params["early_stop_patience_steps"] > 0 and params["val_size"] <= 0:
        raise ValueError(f"{model_id}.val_size must be positive when early stopping is enabled")
    for name in ("start_padding_enabled", "deterministic"):
        if not isinstance(params[name], bool):
            raise ValueError(f"{model_id}.{name} must be a boolean")
    if (
        not isinstance(params["learning_rate"], int | float)
        or isinstance(params["learning_rate"], bool)
        or params["learning_rate"] <= 0
    ):
        raise ValueError(f"{model_id}.learning_rate must be positive")
    if not isinstance(params["scaler_type"], str) or not params["scaler_type"].strip():
        raise ValueError(f"{model_id}.scaler_type must be a non-empty string")


def _normalize_sales_history(sales_df: pd.DataFrame, min_history: int) -> pd.DataFrame:
    """Validate sales and retain series meeting the configured history floor.

    History depth is the count of distinct monthly rows. Calendar-complete inputs
    therefore count explicit zero-demand months, matching the backtest contract.
    """
    required = {"sku_ck", "startdate", "qty"}
    missing = required - set(sales_df.columns)
    if missing:
        raise ValueError(f"Neural sales input is missing columns: {sorted(missing)}")
    if sales_df.empty:
        raise ValueError("Neural sales input must not be empty")

    selected_columns = ["sku_ck", "startdate", "qty"]
    if "first_sale_month" in sales_df.columns:
        selected_columns.append("first_sale_month")
    sales = sales_df[selected_columns].copy()
    sales.attrs = dict(sales_df.attrs)
    invalid_ids = sales["sku_ck"].isna() | (sales["sku_ck"].astype(str).str.strip() == "")
    if invalid_ids.any():
        raise ValueError("Neural sales input contains a blank sku_ck")
    sales["sku_ck"] = sales["sku_ck"].astype(str)
    sales["startdate"] = (
        pd.to_datetime(sales["startdate"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    if sales["startdate"].isna().any():
        raise ValueError("Neural sales input contains an invalid startdate")
    sales["qty"] = pd.to_numeric(sales["qty"], errors="coerce")
    quantities = sales["qty"].to_numpy(dtype=float)
    if not np.isfinite(quantities).all():
        raise ValueError("Neural sales input contains a non-finite quantity")
    if sales.duplicated(["sku_ck", "startdate"]).any():
        raise ValueError("Neural sales input contains duplicate DFU-month rows")

    sales = complete_monthly_history(sales)
    history_counts = sales.groupby("sku_ck")["startdate"].nunique()
    eligible_ids = set(history_counts[history_counts >= min_history].index)
    eligible = sales[sales["sku_ck"].isin(eligible_ids)].copy()
    if eligible.empty:
        raise RuntimeError(
            f"No DFUs have the configured minimum neural history of {min_history} month(s)"
        )
    eligible["qty"] = eligible["qty"].mask(eligible["qty"] == 0, 0.0)
    return eligible.sort_values(["sku_ck", "startdate"]).reset_index(drop=True)


def _normalize_predict_months(predict_months: list[pd.Timestamp]) -> list[pd.Timestamp]:
    if not predict_months:
        raise ValueError("Neural prediction requires at least one forecast month")
    months = [pd.Timestamp(month).to_period("M").to_timestamp() for month in predict_months]
    if any(pd.isna(month) for month in months):
        raise ValueError("Neural forecast months contain an invalid date")
    if len(set(months)) != len(months):
        raise ValueError("Neural forecast months must be unique")
    expected = list(pd.date_range(months[0], periods=len(months), freq="MS"))
    if months != expected:
        raise ValueError("Neural forecast months must be sorted and contiguous")
    return months


def _build_model(model_id: str, params: dict[str, Any], accelerator: str = "cpu") -> Any:
    """Instantiate one of the two supported NeuralForecast models.

    Lazy-imports neuralforecast models to avoid loading torch at module level.
    """
    _validate_model_params(model_id, params)

    from neuralforecast.models import NBEATS, NHITS

    h = params["h"]
    input_size = params["input_size"]
    max_steps = params["max_steps"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]

    # early_stop_patience_steps=-1 disables early stopping; NeuralForecast 3.x
    # raises "Set val_size>0 if early stopping is enabled" otherwise.
    early_stop = params["early_stop_patience_steps"]

    # A8: Robust scaler (median/IQR) is configurable per model. Use "robust" for
    # erratic segments — standard z-score scaling is distorted by demand spikes.
    scaler_type = params["scaler_type"]
    common = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "random_seed": params["random_seed"],
        "scaler_type": scaler_type,
        "early_stop_patience_steps": early_stop,
        # Pad short series with zeros so DFUs with < input_size months of
        # history can still be included rather than crashing training.
        "start_padding_enabled": params["start_padding_enabled"],
        # accelerator passed as direct kwarg → captured by **trainer_kwargs in
        # BaseModel → forwarded to Lightning Trainer (supports 'cpu', 'mps', 'gpu')
        "accelerator": accelerator,
        "deterministic": params["deterministic"],
        # Managed jobs already report durable timeframe-level progress. The
        # per-batch Lightning bar emits thousands of carriage-return lines into
        # job_history and obscures useful operator messages.
        "enable_progress_bar": False,
    }

    if model_id == "nbeats":
        return NBEATS(**common)

    return NHITS(**common)


def _prepare_nf_dataframe(
    sales_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare data in NeuralForecast format: unique_id, ds, y."""
    df = sales_df[["sku_ck", "startdate", "qty"]].copy()
    df = df.rename(columns={"sku_ck": "unique_id", "startdate": "ds", "qty": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    # psycopg3 returns NUMERIC columns as Python Decimal → object dtype in pandas.
    # NeuralForecast requires y to be float64; cast explicitly here.
    df["y"] = df["y"].astype(float)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


def _training_lineage_from_frame(
    train_df: pd.DataFrame,
    *,
    runtime_contract: Mapping[str, str],
) -> NeuralTrainingLineage:
    """Build lineage from the exact sorted float64 frame passed to ``fit``."""
    if train_df.empty:
        raise ValueError("Neural training frame must not be empty")
    expected_columns = ["unique_id", "ds", "y"]
    if list(train_df.columns) != expected_columns:
        raise ValueError(f"Neural training frame columns must be {expected_columns}")
    if train_df.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Neural training frame contains duplicate DFU-month rows")

    ids = train_df["unique_id"].drop_duplicates().astype(str).tolist()
    cohort = compute_neural_cohort_identity(ids, presorted=True)
    normalized_runtime = _normalize_runtime_contract(runtime_contract)

    digest = hashlib.sha256()
    _update_length_prefixed(digest, NEURAL_TRAINING_CONTRACT_VERSION)
    _update_length_prefixed(digest, "unique_id:string|ds:datetime64[ns]|y:float64")
    digest.update(len(train_df).to_bytes(8, byteorder="big", signed=False))
    # pandas' vectorized row hash is efficient for production-scale global fits.
    # The exact pandas version is separately embedded in the runtime contract.
    row_hashes = pd.util.hash_pandas_object(
        train_df[expected_columns],
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.astype("<u8", copy=False).tobytes())
    return NeuralTrainingLineage(
        training_dfu_count=cohort.dfu_count,
        training_row_count=len(train_df),
        training_cohort_checksum=cohort.checksum,
        training_data_checksum=digest.hexdigest(),
        training_contract_version=NEURAL_TRAINING_CONTRACT_VERSION,
        runtime_contract=normalized_runtime,
    )


def derive_neural_training_lineage(
    sales_df: pd.DataFrame,
    *,
    min_history: int,
    runtime_contract: Mapping[str, str] | None = None,
) -> NeuralTrainingLineage:
    """Derive the identity of the normalized calendar-complete training frame."""
    eligible_sales = _normalize_sales_history(sales_df, min_history)
    train_df = _prepare_nf_dataframe(eligible_sales)
    return _training_lineage_from_frame(
        train_df,
        runtime_contract=(
            current_neural_runtime_contract()
            if runtime_contract is None
            else runtime_contract
        ),
    )


def fit_neural_model(
    sales_df: pd.DataFrame,
    *,
    model_id: str,
    params: dict[str, Any],
    accelerator: str | None = None,
) -> FittedNeuralModel:
    """Fit one global model at its configured, backtest-evaluated horizon."""
    _validate_model_params(model_id, params)
    if not _check_neuralforecast():
        raise RuntimeError("neuralforecast is required to fit NHITS or NBEATS")

    eligible_sales = _normalize_sales_history(sales_df, int(params["min_history"]))
    train_df = _prepare_nf_dataframe(eligible_sales)
    training_lineage = _training_lineage_from_frame(
        train_df,
        runtime_contract=current_neural_runtime_contract(),
    )
    selected_device = accelerator or _detect_device()
    selected_accelerator = selected_device if selected_device in {"cpu", "mps"} else "gpu"

    from neuralforecast import NeuralForecast

    logger.info(
        "Initializing %s global model with accelerator=%s",
        model_id,
        selected_accelerator,
    )
    model = _build_model(model_id, params, accelerator=selected_accelerator)
    neural_forecast = NeuralForecast(models=[model], freq="MS")
    runtime_horizon = int(getattr(neural_forecast, "h", params["h"]))
    if runtime_horizon != params["h"]:
        raise RuntimeError(
            f"{model_id} runtime horizon {runtime_horizon} does not match configured h={params['h']}"
        )
    logger.info(
        "Training %s global model: rows=%d DFUs=%d fitted_h=%d accelerator=%s",
        model_id,
        len(train_df),
        train_df["unique_id"].nunique(),
        params["h"],
        selected_accelerator,
    )
    _fit_with_heartbeat(
        neural_forecast,
        train_df,
        model_id,
        val_size=int(params["val_size"]),
    )
    return FittedNeuralModel(
        neural_forecast=neural_forecast,
        model_id=model_id,
        fitted_horizon=int(params["h"]),
        min_history=int(params["min_history"]),
        training_dfu_count=training_lineage.training_dfu_count,
        training_row_count=training_lineage.training_row_count,
        training_cohort_checksum=training_lineage.training_cohort_checksum,
        training_data_checksum=training_lineage.training_data_checksum,
        training_contract_version=training_lineage.training_contract_version,
        runtime_contract=training_lineage.runtime_contract,
    )


def _normalize_prediction_output(
    forecast_df: pd.DataFrame,
    *,
    fitted: FittedNeuralModel,
    expected_ids: set[str],
    window: ForecastOutputWindow,
    prediction_horizon: int,
) -> pd.DataFrame:
    if not isinstance(forecast_df, pd.DataFrame):
        raise RuntimeError(f"{fitted.model_id} returned a non-DataFrame forecast")

    output = forecast_df.copy()
    if not {"unique_id", "ds"}.issubset(output.columns):
        output = output.reset_index()
    required = {"unique_id", "ds"}
    missing = required - set(output.columns)
    if missing:
        raise RuntimeError(
            f"{fitted.model_id} output is missing required columns: {sorted(missing)}"
        )
    forecast_columns = [column for column in output.columns if column not in required]
    if len(forecast_columns) != 1:
        raise RuntimeError(f"{fitted.model_id} output must contain exactly one forecast column")

    forecast_column = forecast_columns[0]
    if str(forecast_column).lower() != fitted.model_id:
        raise RuntimeError(
            f"{fitted.model_id} output has mismatched model identity {forecast_column!r}"
        )
    if output["unique_id"].isna().any():
        raise RuntimeError(f"{fitted.model_id} returned a null series ID")
    output["unique_id"] = output["unique_id"].astype(str)
    output["ds"] = pd.to_datetime(output["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if output["ds"].isna().any():
        raise RuntimeError(f"{fitted.model_id} returned an invalid forecast month")
    if output.duplicated(["unique_id", "ds"]).any():
        raise RuntimeError(f"{fitted.model_id} returned duplicate DFU-month forecasts")

    produced_ids = set(output["unique_id"])
    if produced_ids != expected_ids:
        raise RuntimeError(
            f"{fitted.model_id} returned mismatched series IDs: "
            f"expected {len(expected_ids)}, received {len(produced_ids)}"
        )

    full_months = list(
        pd.date_range(
            window.inference_months[0],
            periods=prediction_horizon,
            freq="MS",
        )
    )
    produced_months = set(output["ds"])
    if produced_months != set(full_months):
        raise RuntimeError(
            f"{fitted.model_id} returned unexpected forecast months: "
            f"expected {full_months[0]:%Y-%m} through {full_months[-1]:%Y-%m}"
        )

    expected_keys = {(sku_ck, month) for sku_ck in expected_ids for month in full_months}
    actual_keys = set(zip(output["unique_id"], output["ds"], strict=True))
    if actual_keys != expected_keys:
        missing_count = len(expected_keys - actual_keys)
        extra_count = len(actual_keys - expected_keys)
        raise RuntimeError(
            f"{fitted.model_id} returned incomplete forecast coverage "
            f"(missing={missing_count}, extra={extra_count})"
        )

    values = pd.to_numeric(output[forecast_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{fitted.model_id} returned non-finite forecast values")
    output[FORECAST_QTY_COL] = np.maximum(values, 0.0)
    output = output[output["ds"].isin(window.output_months)].copy()
    output = output.rename(columns={"unique_id": "sku_ck", "ds": "startdate"})
    output["algorithm_id"] = fitted.model_id
    result = output[_OUTPUT_COLUMNS].sort_values(["sku_ck", "startdate"]).reset_index(drop=True)
    result.attrs.update(
        {
            "model_id": fitted.model_id,
            "fitted_horizon": fitted.fitted_horizon,
            "prediction_horizon": prediction_horizon,
            "min_history": fitted.min_history,
            "output_offset": window.output_offset,
        }
    )
    return result


def predict_neural_model(
    fitted: FittedNeuralModel,
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Predict from an existing fitted model without changing its architecture."""
    if fitted.model_id not in SUPPORTED_NEURAL_MODELS:
        raise ValueError(f"Unsupported neural model: {fitted.model_id}")
    if fitted.fitted_horizon <= 0:
        raise ValueError("Fitted neural horizon must be positive")
    if fitted.min_history <= 0:
        raise ValueError("Fitted neural minimum history must be positive")
    runtime_horizon = int(getattr(fitted.neural_forecast, "h", fitted.fitted_horizon))
    if runtime_horizon != fitted.fitted_horizon:
        raise RuntimeError(
            f"{fitted.model_id} fitted horizon metadata does not match the loaded model"
        )

    months = _normalize_predict_months(predict_months)
    eligible_sales = _normalize_sales_history(sales_df, fitted.min_history)
    window = build_forecast_output_window(
        months,
        history_end=eligible_sales["startdate"].max(),
        adapter_name=fitted.model_id,
    )
    live_df = _prepare_nf_dataframe(eligible_sales)
    expected_ids = set(live_df["unique_id"].astype(str))
    prediction_horizon = max(fitted.fitted_horizon, window.inference_horizon)
    forecast_df = fitted.neural_forecast.predict(
        df=live_df,
        h=prediction_horizon,
    )
    return _normalize_prediction_output(
        forecast_df,
        fitted=fitted,
        expected_ids=expected_ids,
        window=window,
        prediction_horizon=prediction_horizon,
    )


def run_neural_models(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
) -> pd.DataFrame:
    """Run enabled N-HiTS/N-BEATS models across all DFUs.

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
    empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)

    unsupported = set(enabled_models) - SUPPORTED_NEURAL_MODELS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported neural model(s): {names}")

    if not enabled_models:
        return empty

    if sales_df.empty:
        return empty

    months = _normalize_predict_months(predict_months)
    all_results: list[pd.DataFrame] = []
    for model_id, params in enabled_models.items():
        t0 = time.time()
        fitted = fit_neural_model(
            sales_df,
            model_id=model_id,
            params=params,
        )
        result = predict_neural_model(fitted, sales_df, months)
        all_results.append(result)
        logger.info(
            "%s: %d predictions for %d DFUs in %.1fs",
            model_id,
            len(result),
            result["sku_ck"].nunique(),
            time.time() - t0,
        )

    combined = pd.concat(all_results, ignore_index=True)
    produced_models = set(combined["algorithm_id"].astype(str))
    if produced_models != set(enabled_models):
        raise RuntimeError("Neural model output identities do not match the enabled model roster")
    combined.attrs["fitted_horizons"] = {
        result.attrs["model_id"]: result.attrs["fitted_horizon"] for result in all_results
    }
    logger.info(
        "DL models complete: %d predictions from %d models",
        len(combined),
        combined["algorithm_id"].nunique(),
    )
    return combined
