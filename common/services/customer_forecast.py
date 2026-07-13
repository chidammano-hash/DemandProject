"""Generation-only customer-level forecasting services (Spec 35)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.core.sql_helpers import read_sql_chunked
from common.core.utils import get_algorithm_params, load_forecast_pipeline_config
from common.ml.chronos2_enriched import run_chronos2_enriched

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
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer forecast settings are incomplete") from exc
    if history_months <= 0 or horizon_months <= 0:
        raise ValueError("Customer forecast history and horizon must be positive")
    if model_id != "chronos2_enriched":
        raise ValueError("Customer forecasting requires chronos2_enriched")
    if not isinstance(settings.get("enabled"), bool):
        raise ValueError("Customer forecast enabled setting must be boolean")
    return {
        **settings,
        "history_months": history_months,
        "horizon_months": horizon_months,
        "model_id": model_id,
    }


def customer_forecast_config_checksum(settings: dict[str, Any]) -> str:
    payload = {
        "customer_forecast": settings,
        "algorithm": get_algorithm_params(str(settings["model_id"])),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_customer_forecast_readiness(
    conn: Any,
    window: CustomerForecastWindow,
) -> dict[str, Any]:
    """Return source freshness and eligibility for the resolved run window."""
    sql = """
        WITH series AS (
            SELECT item_id, location_id, customer_no,
                   MIN(startdate) AS first_month,
                   SUM(CASE WHEN startdate >= %s AND startdate < %s
                            THEN demand_qty ELSE 0 END) AS window_demand
            FROM fact_customer_demand_monthly
            WHERE startdate < %s
            GROUP BY item_id, location_id, customer_no
        ), duplicate_grains AS (
            SELECT COUNT(*) AS duplicate_count
            FROM (
                SELECT item_id, location_id, customer_no, startdate
                FROM fact_customer_demand_monthly
                WHERE startdate >= %s AND startdate < %s
                GROUP BY item_id, location_id, customer_no, startdate
                HAVING COUNT(*) > 1
            ) duplicates
        )
        SELECT
            (SELECT MAX(startdate) FROM fact_customer_demand_monthly WHERE startdate < %s),
            COUNT(*),
            COUNT(*) FILTER (WHERE first_month <= %s AND window_demand > 0),
            (SELECT COUNT(*) FROM fact_customer_demand_monthly
             WHERE startdate >= %s AND startdate < %s
               AND (item_id IS NULL OR location_id IS NULL OR customer_no IS NULL)),
            (SELECT duplicate_count FROM duplicate_grains),
            (SELECT COUNT(*) FROM fact_customer_demand_monthly
             WHERE startdate >= %s AND startdate < %s AND demand_qty < 0)
        FROM series
    """
    history_end_month = window.history_end.replace(day=1)
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                window.history_start,
                window.forecast_start,
                window.forecast_start,
                window.history_start,
                window.forecast_start,
                window.forecast_start,
                window.history_start,
                window.history_start,
                window.forecast_start,
                window.history_start,
                window.forecast_start,
            ),
        )
        row = cur.fetchone()
    (
        latest_month,
        total_series,
        eligible_series,
        invalid_keys,
        duplicate_grains,
        negative_rows,
    ) = row or (
        None,
        0,
        0,
        0,
        0,
        0,
    )
    blockers: list[str] = []
    if latest_month != history_end_month:
        blockers.append(
            f"Load customer demand through {window.history_end.strftime('%B %Y')}"
        )
    if int(total_series or 0) == 0:
        blockers.append("Load customer demand history before generating forecasts")
    elif int(eligible_series or 0) == 0:
        blockers.append("No customer series has 18 months of history and positive demand")
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
        "total_series": int(total_series or 0),
        "eligible_series": int(eligible_series or 0),
        "skipped_series": max(int(total_series or 0) - int(eligible_series or 0), 0),
        "invalid_key_rows": int(invalid_keys or 0),
        "duplicate_grains": int(duplicate_grains or 0),
        "negative_rows": int(negative_rows or 0),
        "blockers": blockers,
    }


def load_customer_history(conn: Any, window: CustomerForecastWindow) -> pd.DataFrame:
    """Load the bounded fact history plus each series' first observed month."""
    sql = """
        WITH series_start AS (
            SELECT item_id, location_id, customer_no, MIN(startdate) AS series_first_month
            FROM fact_customer_demand_monthly
            WHERE startdate < %s
            GROUP BY item_id, location_id, customer_no
        )
        SELECT starts.item_id, starts.location_id, starts.customer_no,
               history.startdate,
               COALESCE(SUM(history.demand_qty), 0)::double precision AS demand_qty,
               starts.series_first_month
        FROM series_start starts
        LEFT JOIN fact_customer_demand_monthly history
          ON history.item_id = starts.item_id
         AND history.location_id = starts.location_id
         AND history.customer_no = starts.customer_no
         AND history.startdate >= %s
         AND history.startdate < %s
        GROUP BY starts.item_id, starts.location_id, starts.customer_no,
                 history.startdate, starts.series_first_month
        ORDER BY starts.item_id, starts.location_id, starts.customer_no, history.startdate
    """
    return read_sql_chunked(
        conn,
        sql,
        params=(window.forecast_start, window.history_start, window.forecast_start),
    )


def prepare_customer_history(
    frame: pd.DataFrame,
    window: CustomerForecastWindow,
) -> PreparedCustomerHistory:
    """Validate, filter, and densify customer histories for Chronos."""
    required = {*_IDENTITY_COLUMNS, "startdate", "demand_qty", "series_first_month"}
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
    history["series_first_month"] = pd.to_datetime(
        history["series_first_month"], errors="coerce"
    )
    history["demand_qty"] = pd.to_numeric(history["demand_qty"], errors="coerce")
    if history["series_first_month"].isna().any():
        raise ValueError("Customer history contains an invalid first month")
    numeric = history["demand_qty"].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Customer history contains non-finite demand")
    if (numeric < 0).any():
        raise ValueError("Customer history contains negative demand")

    calendar = pd.date_range(window.history_start, periods=window.history_months, freq="MS")
    model_frames: list[pd.DataFrame] = []
    identities: dict[str, tuple[str, str, str]] = {}
    skipped: list[dict[str, str]] = []
    grouped = history.groupby(_IDENTITY_COLUMNS, sort=True, dropna=False)
    for identity, group in grouped:
        item_id, location_id, customer_no = (str(value) for value in identity)
        first_month = pd.Timestamp(group["series_first_month"].min()).date()
        skip_identity = {
            "item_id": item_id,
            "location_id": location_id,
            "customer_no": customer_no,
        }
        if first_month > window.history_start:
            skipped.append({**skip_identity, "reason": "insufficient_history"})
            continue
        bounded = group[
            group["startdate"].notna()
            & (group["startdate"] >= pd.Timestamp(window.history_start))
            & (group["startdate"] < pd.Timestamp(window.forecast_start))
        ]
        monthly = bounded.groupby("startdate", as_index=True)["demand_qty"].sum()
        dense = monthly.reindex(calendar, fill_value=0.0).astype(float)
        if not (dense > 0).any():
            skipped.append({**skip_identity, "reason": "no_positive_demand"})
            continue
        sku_ck = f"customer_series_{len(identities) + 1}"
        identities[sku_ck] = (item_id, location_id, customer_no)
        model_frames.append(
            pd.DataFrame(
                {"sku_ck": sku_ck, "startdate": calendar, "qty": dense.to_numpy()}
            )
        )

    model_input = (
        pd.concat(model_frames, ignore_index=True)
        if model_frames
        else pd.DataFrame(columns=["sku_ck", "startdate", "qty"])
    )
    model_input.attrs["history_end"] = pd.Timestamp(window.history_end).to_period("M").to_timestamp()
    return PreparedCustomerHistory(model_input, identities, skipped)


def build_customer_forecast_rows(
    prepared: PreparedCustomerHistory,
    predictions: pd.DataFrame,
    window: CustomerForecastWindow,
) -> pd.DataFrame:
    """Validate Chronos output and restore item/location/customer identities."""
    required = {"sku_ck", "startdate", FORECAST_QTY_COL}
    missing = required - set(predictions.columns)
    if missing:
        raise RuntimeError(f"Customer forecast output is missing columns: {sorted(missing)}")
    output = predictions.copy()
    output["startdate"] = pd.to_datetime(output["startdate"], errors="coerce")
    output[FORECAST_QTY_COL] = pd.to_numeric(output[FORECAST_QTY_COL], errors="coerce")
    if output["startdate"].isna().any() or not np.isfinite(
        output[FORECAST_QTY_COL].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Customer forecast output contains invalid values")
    output[FORECAST_QTY_COL] = output[FORECAST_QTY_COL].clip(lower=0.0)
    if output.duplicated(["sku_ck", "startdate"]).any():
        raise RuntimeError("Customer forecast output contains duplicate series-month rows")

    expected_months = pd.DatetimeIndex(window.forecast_months)
    for sku_ck in prepared.identity_by_sku:
        actual_months = pd.DatetimeIndex(
            output.loc[output["sku_ck"] == sku_ck, "startdate"].sort_values()
        )
        if not actual_months.equals(expected_months):
            raise RuntimeError(
                f"Customer series {sku_ck} did not produce a complete "
                f"{window.horizon_months}-month forecast"
            )
    if set(output["sku_ck"]) != set(prepared.identity_by_sku):
        raise RuntimeError("Customer forecast output does not match the eligible series")

    identity_rows = [
        {"sku_ck": sku_ck, **dict(zip(_IDENTITY_COLUMNS, identity, strict=True))}
        for sku_ck, identity in prepared.identity_by_sku.items()
    ]
    rows = output.merge(pd.DataFrame(identity_rows), on="sku_ck", validate="many_to_one")
    rows = rows.rename(
        columns={"startdate": "forecast_month", FORECAST_QTY_COL: "forecast_qty"}
    )
    rows["lower_bound"] = None
    rows["upper_bound"] = None
    return rows[
        [
            *_IDENTITY_COLUMNS,
            "forecast_month",
            "forecast_qty",
            "lower_bound",
            "upper_bound",
        ]
    ].sort_values([*_IDENTITY_COLUMNS, "forecast_month"], ignore_index=True)


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
            "model_id, config_checksum FROM customer_forecast_run "
            "WHERE run_id = %s::uuid",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] not in {"queued", "generating", "failed"}:
        raise RuntimeError("Customer forecast run is missing or not runnable")
    checksum = customer_forecast_config_checksum(settings)
    if row[8] != settings["model_id"] or row[9] != checksum:
        raise RuntimeError("Customer forecast configuration changed; submit a new run")
    window = build_customer_forecast_window(row[1], int(row[6]), int(row[7]))
    if (row[2], row[3], row[4], row[5]) != (
        window.history_start,
        window.history_end,
        window.forecast_start,
        window.forecast_end,
    ):
        raise RuntimeError("Customer forecast run window is inconsistent")
    return window, checksum


def generate_customer_forecast(
    conn: Any,
    run_id: str,
    *,
    predictor: Callable[..., pd.DataFrame] = run_chronos2_enriched,
) -> dict[str, Any]:
    """Generate and atomically persist one immutable customer forecast run."""
    settings = get_customer_forecast_settings()
    if not settings["enabled"]:
        raise RuntimeError("Customer forecast generation is disabled")
    window, config_checksum = _resolve_run_window(conn, run_id, settings)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE customer_forecast_run SET run_status = 'generating', "
            "started_at = COALESCE(started_at, NOW()), error_summary = NULL "
            "WHERE run_id = %s::uuid AND run_status IN ('queued', 'generating', 'failed')",
            (run_id,),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Customer forecast run is missing or not runnable")
    conn.commit()

    readiness = load_customer_forecast_readiness(conn, window)
    if not readiness["ready"]:
        raise RuntimeError(str(readiness["blockers"][0]))
    raw_history = load_customer_history(conn, window)
    prepared = prepare_customer_history(raw_history, window)
    if prepared.eligible_series_count == 0:
        raise RuntimeError("No eligible customer series are available for forecasting")

    params = get_algorithm_params(str(settings["model_id"]))
    predict_months = [pd.Timestamp(month) for month in window.forecast_months]
    predictions = predictor(prepared.model_input, predict_months, params)
    rows = build_customer_forecast_rows(prepared, predictions, window)
    source_checksum = _frame_checksum(prepared.model_input)
    skip_reason_counts = dict(
        sorted(Counter(row["reason"] for row in prepared.skipped_series).items())
    )

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM fact_customer_forecast WHERE run_id = %s::uuid", (run_id,))
        with cur.copy(
            "COPY fact_customer_forecast "
            "(run_id, item_id, location_id, customer_no, forecast_month, forecast_qty, "
            "lower_bound, upper_bound, model_id, history_end) FROM STDIN"
        ) as copy:
            for row in rows.itertuples(index=False):
                copy.write_row(
                    (
                        run_id,
                        row.item_id,
                        row.location_id,
                        row.customer_no,
                        row.forecast_month.date(),
                        float(row.forecast_qty),
                        row.lower_bound,
                        row.upper_bound,
                        settings["model_id"],
                        window.history_end,
                    )
                )
        cur.execute(
            "UPDATE customer_forecast_run SET run_status = 'completed', "
            "eligible_series = %s, row_count = %s, skipped_series = %s, "
            "skip_reason_counts = %s::jsonb, config_checksum = %s, "
            "source_checksum = %s, completed_at = NOW(), "
            "error_summary = NULL WHERE run_id = %s::uuid",
            (
                prepared.eligible_series_count,
                len(rows),
                len(prepared.skipped_series),
                json.dumps(skip_reason_counts, sort_keys=True),
                config_checksum,
                source_checksum,
                run_id,
            ),
        )
    return {
        "run_id": run_id,
        "eligible_series": prepared.eligible_series_count,
        "row_count": len(rows),
        "skipped_series": len(prepared.skipped_series),
        "skip_reason_counts": skip_reason_counts,
        "forecast_start": window.forecast_start.isoformat(),
        "forecast_end": window.forecast_end.isoformat(),
    }


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
