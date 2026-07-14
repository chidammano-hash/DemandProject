"""Generation-only customer-level forecasting services (Spec 35)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from common.core.utils import load_forecast_pipeline_config
from common.ml.croston import croston_forecast

_IDENTITY_COLUMNS = ["item_id", "location_id", "customer_no"]


@dataclass(frozen=True)
class CustomerForecastWindow:
    planning_month: date
    history_start: date
    history_end: date
    forecast_start: date
    forecast_end: date
    history_months: int
    horizon_months: int
    forecast_months: tuple[date, ...]


@dataclass
class PreparedCustomerHistory:
    model_input: pd.DataFrame
    identity_by_sku: dict[str, tuple[str, str, str]]
    skipped_series: list[dict[str, str]]

    @property
    def eligible_series_count(self) -> int:
        return len(self.identity_by_sku)


def _shift_month(month: date, offset: int) -> date:
    absolute = month.year * 12 + month.month - 1 + offset
    return date(absolute // 12, absolute % 12 + 1, 1)


def build_customer_forecast_window(
    planning_date: date,
    history_months: int,
    horizon_months: int,
) -> CustomerForecastWindow:
    """Resolve closed-history and future-output windows from one planning date."""
    if history_months <= 0 or horizon_months <= 0:
        raise ValueError("Customer forecast history and horizon must be positive")
    planning_month = planning_date.replace(day=1)
    history_start = _shift_month(planning_month, -history_months)
    history_end = planning_month - timedelta(days=1)
    forecast_end = _shift_month(planning_month, horizon_months) - timedelta(days=1)
    months = tuple(_shift_month(planning_month, offset) for offset in range(horizon_months))
    return CustomerForecastWindow(
        planning_month=planning_month,
        history_start=history_start,
        history_end=history_end,
        forecast_start=planning_month,
        forecast_end=forecast_end,
        history_months=history_months,
        horizon_months=horizon_months,
        forecast_months=months,
    )


def get_customer_forecast_settings() -> dict[str, Any]:
    config = load_forecast_pipeline_config()
    try:
        settings = dict(config["customer_forecast"])
        history_months = int(settings["history_months"])
        horizon_months = int(settings["horizon_months"])
        model_id = str(settings["model_id"])
        model_params = dict(settings["model_params"])
        model_alpha = float(model_params["alpha"])
        model_variant = str(model_params["variant"])
        recent_sales_lookback_months = int(settings["recent_sales_lookback_months"])
        batch_size = int(settings["batch_size"])
        cpu_workers = int(settings["cpu_workers"])
        max_batch_attempts = int(settings["max_batch_attempts"])
        progress_interval_seconds = float(settings["progress_interval_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer forecast settings are incomplete") from exc
    if history_months <= 0 or horizon_months <= 0:
        raise ValueError("Customer forecast history and horizon must be positive")
    if model_id != "croston":
        raise ValueError("Customer forecasting requires croston")
    if not 0.0 < model_alpha <= 1.0 or model_variant not in {"classic", "sba"}:
        raise ValueError("Customer forecast Croston settings are invalid")
    if recent_sales_lookback_months <= 0 or recent_sales_lookback_months > history_months:
        raise ValueError("Customer forecast recent-sales lookback is invalid")
    if batch_size <= 0 or cpu_workers <= 0:
        raise ValueError("Customer forecast batch worker settings are invalid")
    if max_batch_attempts <= 0 or progress_interval_seconds <= 0:
        raise ValueError("Customer forecast batch execution settings are invalid")
    if not isinstance(settings.get("enabled"), bool):
        raise ValueError("Customer forecast enabled setting must be boolean")
    return {
        **settings,
        "history_months": history_months,
        "horizon_months": horizon_months,
        "model_id": model_id,
        "recent_sales_lookback_months": recent_sales_lookback_months,
        "batch_size": batch_size,
        "cpu_workers": cpu_workers,
        "max_batch_attempts": max_batch_attempts,
        "progress_interval_seconds": progress_interval_seconds,
        "model_params": {
            "alpha": model_alpha,
            "variant": model_variant,
        },
    }


def customer_forecast_config_checksum(settings: dict[str, Any]) -> str:
    generation_keys = (
        "enabled",
        "model_id",
        "model_params",
        "history_months",
        "horizon_months",
        "recent_sales_lookback_months",
        "batch_size",
        "cpu_workers",
        "max_batch_attempts",
        "progress_interval_seconds",
    )
    payload = {
        "customer_forecast": {key: settings[key] for key in generation_keys},
        "croston": settings["model_params"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_customer_forecast_readiness(
    conn: Any,
    window: CustomerForecastWindow,
    *,
    recent_sales_lookback_months: int,
) -> dict[str, Any]:
    """Return source freshness and eligibility for the resolved run window."""
    sql = """
        SELECT
            MAX(last_month),
            COUNT(*),
            COUNT(*) FILTER (
                WHERE last_sales_month >= %s
            ),
            COUNT(*) FILTER (
                WHERE last_sales_month IS NULL OR last_sales_month < %s
            ),
            0::bigint, -- identity columns are NOT NULL
            0::bigint, -- loader business key is unique at series-month grain
            0::bigint, -- demand_qty has a non-negative CHECK constraint
            (
                SELECT batch_id
                FROM audit_load_batch
                WHERE domain = 'customer_demand'
                  AND status = 'completed'
                ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                LIMIT 1
            ),
            (
                SELECT source_batch_id
                FROM customer_demand_profile_refresh_state
                WHERE singleton_id = 1
            ),
            (
                SELECT COUNT(*)
                FROM audit_load_batch
                WHERE domain = 'customer_demand'
                  AND status = 'running'
            )
        FROM mv_customer_demand_series_profile
        WHERE first_month < %s
    """
    history_end_month = window.history_end.replace(day=1)
    recent_start = _shift_month(window.forecast_start, -recent_sales_lookback_months)
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                recent_start,
                recent_start,
                window.forecast_start,
            ),
        )
        row = cur.fetchone()
    (
        latest_month,
        total_series,
        eligible_series,
        zero_series,
        invalid_keys,
        duplicate_grains,
        negative_rows,
        source_customer_demand_batch_id,
        profile_customer_demand_batch_id,
        active_customer_demand_loads,
    ) = row or (
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        0,
    )
    blockers: list[str] = []
    if source_customer_demand_batch_id is None:
        blockers.append(
            "A completed customer-demand load is required before generating forecasts"
        )
    if int(active_customer_demand_loads or 0) > 0:
        blockers.append("An active customer-demand load is in progress; wait for it to complete")
    elif source_customer_demand_batch_id is not None and (
        profile_customer_demand_batch_id is None
        or int(profile_customer_demand_batch_id) != int(source_customer_demand_batch_id)
    ):
        blockers.append(
            "The customer-demand profile does not represent the latest completed load"
        )
    if latest_month != history_end_month:
        blockers.append(f"Load customer demand through {window.history_end.strftime('%B %Y')}")
    total_series_count = int(total_series or 0)
    eligible_series_count = int(eligible_series or 0)
    zero_series_count = int(zero_series or 0)
    if total_series_count == 0:
        blockers.append("Load customer demand history before generating forecasts")
    if int(invalid_keys or 0) > 0:
        blockers.append("Resolve invalid item, location, or customer keys")
    if int(duplicate_grains or 0) > 0:
        blockers.append("Aggregate duplicate customer-demand months")
    if int(negative_rows or 0) > 0:
        blockers.append("Resolve negative customer demand quantities")
    return {
        "ready": not blockers,
        "planning_month": window.planning_month.isoformat(),
        "history_start": window.history_start.isoformat(),
        "history_end": window.history_end.isoformat(),
        "forecast_start": window.forecast_start.isoformat(),
        "forecast_end": window.forecast_end.isoformat(),
        "history_months": window.history_months,
        "horizon_months": window.horizon_months,
        "source_latest_month": latest_month.isoformat() if latest_month else None,
        "source_customer_demand_batch_id": (
            int(source_customer_demand_batch_id)
            if source_customer_demand_batch_id is not None
            else None
        ),
        "profile_customer_demand_batch_id": (
            int(profile_customer_demand_batch_id)
            if profile_customer_demand_batch_id is not None
            else None
        ),
        "active_customer_demand_loads": int(active_customer_demand_loads or 0),
        "total_series": total_series_count,
        "eligible_series": eligible_series_count,
        "croston_series": eligible_series_count,
        "dormant_series": zero_series_count,
        "forecastable_series": eligible_series_count,
        "skipped_series": zero_series_count,
        "invalid_key_rows": int(invalid_keys or 0),
        "duplicate_grains": int(duplicate_grains or 0),
        "negative_rows": int(negative_rows or 0),
        "blockers": blockers,
    }


def prepare_customer_history(
    frame: pd.DataFrame,
    window: CustomerForecastWindow,
    *,
    recent_sales_lookback_months: int,
) -> PreparedCustomerHistory:
    """Validate, filter, and densify active customer histories for Croston/SBA."""
    required = {
        *_IDENTITY_COLUMNS,
        "startdate",
        "demand_qty",
        "sales_qty",
        "series_first_month",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Customer history is missing columns: {sorted(missing)}")
    if frame.empty:
        return PreparedCustomerHistory(
            model_input=pd.DataFrame(columns=["sku_ck", "startdate", "qty"]),
            identity_by_sku={},
            skipped_series=[],
        )
    if frame[_IDENTITY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("Customer history contains an invalid identity")

    history = frame.copy()
    history["startdate"] = pd.to_datetime(history["startdate"], errors="coerce")
    history["series_first_month"] = pd.to_datetime(history["series_first_month"], errors="coerce")
    history["demand_qty"] = pd.to_numeric(history["demand_qty"], errors="coerce")
    history["sales_qty"] = pd.to_numeric(history["sales_qty"], errors="coerce")
    if history["series_first_month"].isna().any():
        raise ValueError("Customer history contains an invalid first month")
    numeric = history["demand_qty"].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Customer history contains non-finite demand")
    if (numeric < 0).any():
        raise ValueError("Customer history contains negative demand")
    sales_numeric = history["sales_qty"].to_numpy(dtype=float)
    if not np.isfinite(sales_numeric).all() or (sales_numeric < 0).any():
        raise ValueError("Customer history contains invalid sales")

    calendar = pd.date_range(window.history_start, periods=window.history_months, freq="MS")
    model_frames: list[pd.DataFrame] = []
    identities: dict[str, tuple[str, str, str]] = {}
    skipped: list[dict[str, str]] = []
    grouped = history.groupby(_IDENTITY_COLUMNS, sort=True, dropna=False)
    for identity, group in grouped:
        item_id, location_id, customer_no = (str(value) for value in identity)
        bounded = group[
            group["startdate"].notna()
            & (group["startdate"] >= pd.Timestamp(window.history_start))
            & (group["startdate"] < pd.Timestamp(window.forecast_start))
        ]
        monthly = bounded.groupby("startdate", as_index=True)["demand_qty"].sum()
        monthly_sales = bounded.groupby("startdate", as_index=True)["sales_qty"].sum()
        dense = monthly.reindex(calendar, fill_value=0.0).astype(float)
        dense_sales = monthly_sales.reindex(calendar, fill_value=0.0).astype(float)
        recent_start = pd.Timestamp(
            _shift_month(window.forecast_start, -recent_sales_lookback_months)
        )
        has_recent_sales = bool(dense_sales.loc[dense_sales.index >= recent_start].gt(0).any())
        if not has_recent_sales:
            skipped.append(
                {
                    "item_id": item_id,
                    "location_id": location_id,
                    "customer_no": customer_no,
                    "reason": "no_sales_last_6_months",
                }
            )
            continue

        sku_ck = f"croston_customer_series_{len(identities) + 1}"
        identities[sku_ck] = (item_id, location_id, customer_no)
        model_frames.append(
            pd.DataFrame({"sku_ck": sku_ck, "startdate": calendar, "qty": dense.to_numpy()})
        )

    model_input = (
        pd.concat(model_frames, ignore_index=True)
        if model_frames
        else pd.DataFrame(columns=["sku_ck", "startdate", "qty"])
    )
    model_input.attrs["history_end"] = (
        pd.Timestamp(window.history_end).to_period("M").to_timestamp()
    )
    return PreparedCustomerHistory(
        model_input,
        identities,
        skipped,
    )


def build_croston_forecast_rows(
    prepared: PreparedCustomerHistory,
    window: CustomerForecastWindow,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Forecast every active customer series with configured Croston/SBA."""
    frames: list[pd.DataFrame] = []
    for sku_ck, group in prepared.model_input.groupby("sku_ck", sort=False):
        forecast = croston_forecast(
            group.sort_values("startdate")["qty"].to_numpy(dtype=float),
            horizon=window.horizon_months,
            params=params,
        )
        identity = prepared.identity_by_sku[str(sku_ck)]
        frames.append(
            pd.DataFrame(
                {
                    **dict(zip(_IDENTITY_COLUMNS, identity, strict=True)),
                    "forecast_month": pd.DatetimeIndex(window.forecast_months),
                    "forecast_qty": forecast,
                    "lower_bound": None,
                    "upper_bound": None,
                    "model_id": "croston",
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                *_IDENTITY_COLUMNS,
                "forecast_month",
                "forecast_qty",
                "lower_bound",
                "upper_bound",
                "model_id",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        [*_IDENTITY_COLUMNS, "forecast_month"], ignore_index=True
    )


def _frame_checksum(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["sku_ck", "startdate"], ignore_index=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _resolve_run_window(
    conn: Any,
    run_id: str,
    settings: dict[str, Any],
) -> tuple[CustomerForecastWindow, str]:
    """Restore the immutable request window and reject configuration drift."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_status, planning_month, history_start, history_end, "
            "forecast_start, forecast_end, history_months, horizon_months, "
            "model_id, config_checksum, source_customer_demand_batch_id, "
            "(SELECT batch_id FROM audit_load_batch "
            " WHERE domain = 'customer_demand' AND status = 'completed' "
            " ORDER BY completed_at DESC NULLS LAST, batch_id DESC LIMIT 1), "
            "(SELECT source_batch_id FROM customer_demand_profile_refresh_state "
            " WHERE singleton_id = 1), "
            "(SELECT COUNT(*) FROM audit_load_batch "
            " WHERE domain = 'customer_demand' AND status = 'running') "
            "FROM customer_forecast_run "
            "WHERE run_id = %s::uuid",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] not in {"queued", "generating", "failed", "cancelled"}:
        raise RuntimeError("Customer forecast run is missing or not runnable")
    checksum = customer_forecast_config_checksum(settings)
    if row[8] != settings["model_id"] or row[9] != checksum:
        raise RuntimeError("Customer forecast configuration changed; submit a new run")
    if int(row[13] or 0) > 0:
        raise RuntimeError("Customer demand load is active; retry after it completes")
    if (
        row[10] is None
        or row[11] is None
        or row[12] is None
        or int(row[10]) != int(row[11])
        or int(row[10]) != int(row[12])
    ):
        raise RuntimeError("Customer demand changed after run submission; submit a new run")
    window = build_customer_forecast_window(row[1], int(row[6]), int(row[7]))
    if (row[2], row[3], row[4], row[5]) != (
        window.history_start,
        window.history_end,
        window.forecast_start,
        window.forecast_end,
    ):
        raise RuntimeError("Customer forecast run window is inconsistent")
    return window, checksum


def mark_customer_forecast_run_terminal(
    conn: Any,
    run_id: str,
    status: str,
    error_summary: str | None = None,
) -> None:
    if status not in {"failed", "cancelled"}:
        raise ValueError("Customer forecast terminal status must be failed or cancelled")
    summary = (error_summary or status)[:500]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE customer_forecast_run SET run_status = %s, error_summary = %s, "
            "completed_at = NOW() WHERE run_id = %s::uuid "
            "AND run_status IN ('queued', 'generating', 'failed')",
            (status, summary, run_id),
        )
    conn.commit()
