"""Production adapters for the canonical non-tree forecast models.

The backtest and production paths call the same MSTL, N-HiTS, N-BEATS, and
Chronos 2E implementations.  No heuristic forecast may be labeled as one of
those models.
"""

from __future__ import annotations

import platform
from datetime import UTC, date, datetime
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml import mstl as mstl_adapter
from common.ml.chronos2_enriched import _check_chronos2, run_chronos2_enriched
from common.ml.feature_engineering import build_feature_matrix
from common.ml.forecast_ci import compute_ci_bounds
from common.ml.monthly_history import complete_monthly_history
from common.ml.mstl import run_mstl
from common.ml.neural_forecast import FittedNeuralModel, predict_neural_model

CANONICAL_NON_TREE_MODELS = frozenset(
    {"mstl", "nhits", "nbeats", "chronos2_enriched"}
)
_DFU_IDENTITY_COLUMNS = ("sku_ck", "item_id", "customer_group", "loc")


def _package_version(distribution: str, *, model_id: str) -> str:
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{model_id} runtime dependency {distribution!r} is not installed"
        ) from exc
    if not value.strip():
        raise RuntimeError(f"{model_id} runtime dependency {distribution!r} has no version")
    return value


def direct_model_runtime_contract(model_id: str) -> dict[str, str]:
    """Return exact installed runtime identity or fail before generation starts."""
    common = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if model_id == "mstl":
        if mstl_adapter.StatsForecast is None or mstl_adapter.MSTL is None:
            raise RuntimeError("MSTL requires the statistical dependency group")
        return {
            **common,
            "statsforecast": _package_version("statsforecast", model_id=model_id),
        }
    if model_id == "chronos2_enriched":
        if not _check_chronos2():
            raise RuntimeError("Chronos 2 runtime is not installed")
        return {
            **common,
            "chronos_forecasting": _package_version(
                "chronos-forecasting",
                model_id=model_id,
            ),
            "torch": _package_version("torch", model_id=model_id),
        }
    raise ValueError(f"{model_id!r} is not a direct-inference model")


def _normalize_predict_months(
    predict_months: list[pd.Timestamp], forecast_month_generated: date
) -> list[pd.Timestamp]:
    months = [pd.Timestamp(month).to_period("M").to_timestamp() for month in predict_months]
    if not months:
        raise ValueError("Production inference requires at least one forecast month")
    if len(set(months)) != len(months):
        raise ValueError("Production forecast months must be unique")
    expected = list(
        pd.date_range(
            pd.Timestamp(forecast_month_generated).to_period("M").to_timestamp(),
            periods=len(months),
            freq="MS",
        )
    )
    if months != expected:
        raise ValueError("Production forecast months must be contiguous from the record month")
    return months


def _run_adapter(
    *,
    model_id: str,
    complete_sales: pd.DataFrame,
    target_ids: set[str],
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    fitted_neural_model: FittedNeuralModel | None,
) -> pd.DataFrame:
    target_sales = complete_sales[complete_sales["sku_ck"].isin(target_ids)].copy()
    if model_id in {"nhits", "nbeats"}:
        if fitted_neural_model is None:
            raise RuntimeError(f"Production inference requires a fitted {model_id} production artifact")
        if fitted_neural_model.model_id != model_id:
            raise RuntimeError(
                f"Neural artifact model {fitted_neural_model.model_id} does not match {model_id}"
            )
        if fitted_neural_model.fitted_horizon != int(params["h"]):
            raise RuntimeError(f"{model_id} artifact horizon does not match configured h")
        if fitted_neural_model.min_history != int(params["min_history"]):
            raise RuntimeError(f"{model_id} artifact minimum history does not match config")
        return predict_neural_model(
            fitted_neural_model,
            target_sales,
            predict_months,
        )

    if model_id == "mstl":
        return run_mstl(
            target_sales,
            predict_months,
            season_length=int(params["season_length"]),
            min_history=int(params["min_history"]),
            # Production runs inside the managed job worker. Keep MSTL local
            # instead of nesting a process pool; this is an operational choice,
            # not a forecasting hyperparameter.
            n_workers=1,
        )

    if model_id == "chronos2_enriched":
        target_attrs = dfu_attrs[dfu_attrs["sku_ck"].astype(str).isin(target_ids)].copy()
        target_items = set(target_attrs["item_id"].astype(str))
        filtered_items = (
            item_attrs[item_attrs["item_id"].astype(str).isin(target_items)].copy()
            if not item_attrs.empty
            else item_attrs
        )
        feature_grid = build_feature_matrix(
            target_sales,
            target_attrs,
            filtered_items,
            sorted(target_sales["startdate"].unique()),
            cat_dtype="str",
        )
        return run_chronos2_enriched(
            target_sales,
            predict_months,
            params,
            feature_grid=feature_grid,
        )

    raise ValueError(f"Unsupported canonical non-tree model: {model_id}")


def _normalize_dfu_inputs(
    target_dfus: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the full DFU identity and normalize database keys to strings."""
    required_target = {*_DFU_IDENTITY_COLUMNS, "cluster_id"}
    missing_target = required_target - set(target_dfus.columns)
    if missing_target:
        raise ValueError(f"Non-tree target input is missing columns: {sorted(missing_target)}")
    if target_dfus.empty:
        raise ValueError("Non-tree target input must contain at least one DFU")

    missing_attrs = set(_DFU_IDENTITY_COLUMNS) - set(dfu_attrs.columns)
    if missing_attrs:
        raise ValueError(f"DFU attribute input is missing columns: {sorted(missing_attrs)}")

    targets = target_dfus[[*_DFU_IDENTITY_COLUMNS, "cluster_id"]].copy()
    null_target = targets[list(_DFU_IDENTITY_COLUMNS)].isna().any(axis=1)
    if null_target.any():
        raise ValueError("Non-tree target input contains a null DFU identity")
    for column in _DFU_IDENTITY_COLUMNS:
        targets[column] = targets[column].astype(str)

    if targets["sku_ck"].duplicated().any():
        raise ValueError("Non-tree target input contains duplicate sku_ck values")

    target_ids = set(targets["sku_ck"])
    attrs = dfu_attrs[dfu_attrs["sku_ck"].notna()].copy()
    attrs["sku_ck"] = attrs["sku_ck"].astype(str)
    attrs = attrs[attrs["sku_ck"].isin(target_ids)].copy()
    null_attrs = attrs[list(_DFU_IDENTITY_COLUMNS)].isna().any(axis=1)
    if null_attrs.any():
        raise ValueError("Non-tree attribute input contains a null DFU identity")
    for column in ("item_id", "customer_group", "loc"):
        attrs[column] = attrs[column].astype(str)
    if attrs["sku_ck"].duplicated().any():
        raise ValueError("DFU attribute input contains duplicate sku_ck values")

    attr_identity = attrs.set_index("sku_ck")[["item_id", "customer_group", "loc"]]
    target_identity = targets.set_index("sku_ck")[["item_id", "customer_group", "loc"]]
    aligned = attr_identity.reindex(target_identity.index)
    mismatch = aligned.isna().any(axis=1) | (aligned != target_identity).any(axis=1)
    if mismatch.any():
        sample = ", ".join(target_identity.index[mismatch].astype(str)[:5])
        raise ValueError(
            "Non-tree target identity does not match DFU attributes for sku_ck(s): "
            f"{sample}"
        )
    return targets, attrs


def _normalize_cluster_id(value: Any) -> str | None:
    if value is None or bool(pd.isna(value)):
        return None
    return str(value)


def run_canonical_non_tree_forecast(
    *,
    model_id: str,
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    target_dfus: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    params: dict[str, Any],
    forecast_month_generated: date,
    run_id: str,
    sigma_lookup: dict[tuple[str, str], float],
    ci_cfg: dict[str, Any] | None,
    fitted_neural_model: FittedNeuralModel | None = None,
) -> list[dict[str, Any]]:
    """Run one real canonical adapter and return production staging rows."""
    if model_id not in CANONICAL_NON_TREE_MODELS:
        raise ValueError(f"Unsupported canonical non-tree model: {model_id}")
    months = _normalize_predict_months(predict_months, forecast_month_generated)
    targets, normalized_attrs = _normalize_dfu_inputs(target_dfus, dfu_attrs)
    target_ids = set(targets["sku_ck"])

    missing_sales = {"sku_ck", "startdate", "qty"} - set(sales_df.columns)
    if missing_sales:
        raise ValueError(f"Non-tree sales input is missing columns: {sorted(missing_sales)}")
    normalized_sales = sales_df.copy()
    normalized_sales["sku_ck"] = normalized_sales["sku_ck"].astype(str)
    observed_target_ids = set(normalized_sales["sku_ck"]) & target_ids
    missing_history = sorted(target_ids - observed_target_ids)
    if missing_history:
        sample = ", ".join(missing_history[:5])
        raise ValueError(f"Non-tree target sku_ck(s) have no sales history: {sample}")

    # Global neural models are fitted separately on the stable full cohort.
    # Serving passes only requested histories into that immutable artifact, so
    # diagnostic target filters cannot change model weights.
    adapter_sales = normalized_sales[normalized_sales["sku_ck"].isin(target_ids)].copy()
    adapter_sales.attrs = dict(sales_df.attrs)
    complete_sales = complete_monthly_history(adapter_sales)
    record_month = pd.Timestamp(forecast_month_generated).to_period("M").to_timestamp()
    expected_history_end = record_month - pd.DateOffset(months=1)
    actual_history_end = complete_sales["startdate"].max()
    if actual_history_end != expected_history_end:
        raise ValueError(
            "Production non-tree history must end in "
            f"{expected_history_end:%Y-%m}; received {actual_history_end:%Y-%m}"
        )

    normalized_items = item_attrs.copy()
    if not normalized_items.empty:
        if "item_id" not in normalized_items.columns:
            raise ValueError("Item attribute input is missing column: item_id")
        normalized_items["item_id"] = normalized_items["item_id"].astype(str)
    predictions = _run_adapter(
        model_id=model_id,
        complete_sales=complete_sales,
        target_ids=target_ids,
        predict_months=months,
        params=params,
        dfu_attrs=normalized_attrs,
        item_attrs=normalized_items,
        fitted_neural_model=fitted_neural_model,
    )
    if predictions.empty:
        raise RuntimeError(f"{model_id} produced no production forecasts")

    required_prediction = {"sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"}
    missing_prediction = required_prediction - set(predictions.columns)
    if missing_prediction:
        raise RuntimeError(
            f"{model_id} adapter output is missing required columns: "
            f"{sorted(missing_prediction)}"
        )

    predictions = predictions.copy()
    predictions["sku_ck"] = predictions["sku_ck"].astype(str)
    predictions["startdate"] = (
        pd.to_datetime(predictions["startdate"]).dt.to_period("M").dt.to_timestamp()
    )
    produced_ids = set(predictions["algorithm_id"].dropna().astype(str))
    if predictions["algorithm_id"].isna().any() or produced_ids != {model_id}:
        raise RuntimeError(
            f"{model_id} adapter returned mismatched algorithm IDs: {sorted(produced_ids)}"
        )
    predictions = predictions[
        predictions["sku_ck"].isin(target_ids) & predictions["startdate"].isin(months)
    ]
    if predictions.duplicated(["sku_ck", "startdate"]).any():
        raise RuntimeError(f"{model_id} returned duplicate DFU-month forecasts")

    expected_keys = {(sku, month) for sku in target_ids for month in months}
    actual_keys = set(zip(predictions["sku_ck"], predictions["startdate"], strict=True))
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        raise RuntimeError(
            f"{model_id} is missing {len(missing_keys)} required DFU-month forecast(s)"
        )

    predictions[FORECAST_QTY_COL] = pd.to_numeric(
        predictions[FORECAST_QTY_COL], errors="coerce"
    )
    values = predictions[FORECAST_QTY_COL].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError(f"{model_id} returned invalid forecast quantities")

    mapping = targets.set_index("sku_ck").to_dict("index")
    direct_horizon = int(
        predictions.attrs.get(
            "direct_horizon",
            predictions.attrs.get("fitted_horizon", len(months)),
        )
    )
    if direct_horizon <= 0:
        raise RuntimeError(f"{model_id} returned an invalid direct forecast horizon")
    generated_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for prediction in predictions.sort_values(["sku_ck", "startdate"]).itertuples(index=False):
        sku_ck = str(prediction.sku_ck)
        target = mapping[sku_ck]
        forecast_month = pd.Timestamp(prediction.startdate).date()
        horizon = (
            (forecast_month.year - forecast_month_generated.year) * 12
            + forecast_month.month
            - forecast_month_generated.month
            + 1
        )
        point = round(float(getattr(prediction, FORECAST_QTY_COL)), 2)
        sigma = sigma_lookup.get((target["item_id"], target["loc"]))
        if ci_cfg is not None and sigma is not None:
            lower, upper = compute_ci_bounds(
                point,
                float(sigma),
                horizon,
                float(ci_cfg["z_lower"]),
                float(ci_cfg["z_upper"]),
                str(ci_cfg["horizon_scaling"]),
            )
        else:
            lower, upper = None, None
        rows.append(
            {
                "forecast_month_generated": forecast_month_generated,
                "item_id": target["item_id"],
                "customer_group": target["customer_group"],
                "loc": target["loc"],
                "forecast_month": forecast_month,
                "forecast_qty": point,
                "forecast_qty_lower": lower,
                "forecast_qty_upper": upper,
                "model_id": model_id,
                "cluster_id": _normalize_cluster_id(target["cluster_id"]),
                "horizon_months": horizon,
                "is_recursive": horizon > direct_horizon,
                "lag_source": "predicted" if horizon > direct_horizon else "actual",
                "run_id": run_id,
                "generated_at": generated_at,
            }
        )
    return rows
