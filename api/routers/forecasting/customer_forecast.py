"""Generation-only customer forecasting API (Spec 35)."""

from __future__ import annotations

import csv
import io
import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any, cast

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from api.auth import require_api_key
from api.core import get_conn, get_read_only_conn
from common.core.planning_date import get_planning_date
from common.services.cache import cached_sync
from common.services.customer_forecast import (
    build_customer_forecast_window,
    customer_forecast_config_checksum,
    get_customer_forecast_settings,
    load_customer_forecast_readiness,
    mark_customer_forecast_run_terminal,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/customer-forecast", tags=["customer-forecast"])

_RUN_SELECT = """
    SELECT run_id::text, job_id, run_status, planning_month,
           history_start, history_end, forecast_start, forecast_end,
           eligible_series, row_count, skipped_series, model_id,
           created_at, started_at, completed_at, error_summary,
           skip_reason_counts, model_route_counts,
           total_series, completed_series, total_batches, completed_batches
    FROM customer_forecast_run
"""


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _serialize_run(row: tuple[Any, ...]) -> dict[str, Any]:
    total_series = int(row[18] or row[8] or 0)
    completed_series = int(row[19] or (total_series if row[2] == "completed" else 0))
    total_batches = int(row[20] or 0)
    completed_batches = int(row[21] or (total_batches if row[2] == "completed" else 0))
    progress_pct = (
        100
        if row[2] == "completed"
        else min(99, 10 + int(89 * completed_series / total_series))
        if total_series > 0
        else 5
    )
    eta_seconds: int | None = None
    if completed_series > 0 and completed_series < total_series and row[13] is not None:
        started_at = row[13]
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        elapsed = max((datetime.now(UTC) - started_at).total_seconds(), 1.0)
        eta_seconds = int((total_series - completed_series) / (completed_series / elapsed))
    return {
        "run_id": row[0],
        "job_id": row[1],
        "status": row[2],
        "planning_month": _iso(row[3]),
        "history_start": _iso(row[4]),
        "history_end": _iso(row[5]),
        "forecast_start": _iso(row[6]),
        "forecast_end": _iso(row[7]),
        "eligible_series": int(row[8] or 0),
        "row_count": int(row[9] or 0),
        "skipped_series": int(row[10] or 0),
        "model_id": row[11],
        "created_at": _iso(row[12]),
        "started_at": _iso(row[13]),
        "completed_at": _iso(row[14]),
        "error_summary": row[15],
        "skip_reason_counts": row[16] or {},
        "model_route_counts": row[17] or {},
        "total_series": total_series,
        "completed_series": completed_series,
        "total_batches": total_batches,
        "completed_batches": completed_batches,
        "progress_pct": progress_pct,
        "eta_seconds": eta_seconds,
    }


def _resolved_window() -> tuple[dict[str, Any], Any]:
    settings = get_customer_forecast_settings()
    window = build_customer_forecast_window(
        get_planning_date(),
        settings["history_months"],
        settings["horizon_months"],
    )
    return settings, window


@router.get("/readiness")
@cached_sync(ttl=300, group="customer_forecast")
def get_readiness() -> dict[str, Any]:
    try:
        settings, window = _resolved_window()
        with get_read_only_conn() as conn:
            readiness = load_customer_forecast_readiness(
                conn,
                window,
                recent_sales_lookback_months=int(settings["recent_sales_lookback_months"]),
            )
        if not settings["enabled"]:
            readiness["ready"] = False
            readiness["blockers"].insert(0, "Enable customer forecasting in configuration")
        return readiness
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Checking customer forecast readiness failed")
        raise HTTPException(
            status_code=500, detail="customer forecast readiness check failed"
        ) from exc


@router.post("/generate", dependencies=[Depends(require_api_key)])
def generate_customer_forecasts() -> JSONResponse:
    try:
        settings, window = _resolved_window()
        with get_conn() as conn:
            readiness = load_customer_forecast_readiness(
                conn,
                window,
                recent_sales_lookback_months=int(settings["recent_sales_lookback_months"]),
            )
            if not settings["enabled"]:
                readiness["ready"] = False
                readiness["blockers"].insert(0, "Enable customer forecasting in configuration")
            if not readiness["ready"]:
                raise HTTPException(status_code=409, detail=readiness["blockers"][0])
            run_id = str(uuid.uuid4())
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO customer_forecast_run "
                    "(run_id, run_status, planning_month, history_start, history_end, "
                    "forecast_start, forecast_end, history_months, horizon_months, "
                    "model_id, config_checksum) "
                    "VALUES (%s::uuid, 'queued', %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        run_id,
                        window.planning_month,
                        window.history_start,
                        window.history_end,
                        window.forecast_start,
                        window.forecast_end,
                        window.history_months,
                        window.horizon_months,
                        settings["model_id"],
                        customer_forecast_config_checksum(settings),
                    ),
                )
            conn.commit()
    except HTTPException:
        raise
    except psycopg.errors.UniqueViolation as exc:
        raise HTTPException(
            status_code=409,
            detail="A customer forecast generation is already active",
        ) from exc
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Creating customer forecast run failed")
        raise HTTPException(
            status_code=500, detail="customer forecast run creation failed"
        ) from exc

    from common.services.job_registry import JobManager

    try:
        job_id = JobManager().submit_job(
            "generate_customer_forecast",
            {"run_id": run_id},
            label=f"Customer Forecast · {window.planning_month:%B %Y}",
            triggered_by="api",
            max_retries=1,
        )
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE customer_forecast_run SET job_id = %s WHERE run_id = %s::uuid",
                (job_id, run_id),
            )
            conn.commit()
    except (RuntimeError, ValueError, psycopg.Error) as exc:
        logger.exception("Submitting customer forecast job failed")
        try:
            with get_conn() as conn:
                mark_customer_forecast_run_terminal(conn, run_id, "failed", "job submission failed")
        except psycopg.Error:
            logger.exception("Marking unsubmitted customer forecast run failed")
        raise HTTPException(
            status_code=500, detail="customer forecast job submission failed"
        ) from exc

    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "job_id": job_id, "status": "queued"},
    )


def _load_run(
    run_id: str | None = None,
    *,
    latest_completed: bool = False,
) -> tuple[Any, ...] | None:
    sql = _RUN_SELECT
    params: tuple[Any, ...] = ()
    if run_id is not None:
        sql += " WHERE run_id = %s::uuid"
        params = (run_id,)
    else:
        if latest_completed:
            sql += " WHERE run_status = 'completed'"
        sql += " ORDER BY created_at DESC LIMIT 1"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cast(tuple[Any, ...] | None, cur.fetchone())


@router.get("/runs/latest")
def get_latest_run(completed_only: bool = False) -> dict[str, Any]:
    try:
        row = _load_run(latest_completed=completed_only)
    except (ValueError, psycopg.Error) as exc:
        logger.exception("Loading latest customer forecast run failed")
        raise HTTPException(status_code=500, detail="customer forecast run lookup failed") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="No customer forecast run found")
    return _serialize_run(row)


@router.get("/runs/{run_id}")
def get_run(run_id: uuid.UUID) -> dict[str, Any]:
    try:
        row = _load_run(str(run_id))
    except psycopg.Error as exc:
        logger.exception("Loading customer forecast run failed")
        raise HTTPException(status_code=500, detail="customer forecast run lookup failed") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Customer forecast run not found")
    return _serialize_run(row)


@router.post("/runs/{run_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_run(run_id: uuid.UUID) -> dict[str, str]:
    try:
        row = _load_run(str(run_id))
    except psycopg.Error as exc:
        logger.exception("Loading customer forecast run for cancellation failed")
        raise HTTPException(
            status_code=500, detail="customer forecast cancellation failed"
        ) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Customer forecast run not found")
    run = _serialize_run(row)
    if run["status"] not in {"queued", "generating"} or not run["job_id"]:
        raise HTTPException(status_code=409, detail="Customer forecast run is not cancellable")

    from common.services.job_registry import JobManager

    try:
        cancelled = JobManager().cancel_job(str(run["job_id"]))
        if not cancelled:
            raise HTTPException(status_code=409, detail="Customer forecast job is already terminal")
        with get_conn() as conn:
            mark_customer_forecast_run_terminal(conn, str(run_id), "cancelled")
    except HTTPException:
        raise
    except (RuntimeError, psycopg.Error) as exc:
        logger.exception("Cancelling customer forecast run failed")
        raise HTTPException(
            status_code=500, detail="customer forecast cancellation failed"
        ) from exc
    return {"run_id": str(run_id), "status": "cancelled"}


@router.post("/runs/{run_id}/retry", dependencies=[Depends(require_api_key)])
def retry_run(run_id: uuid.UUID) -> JSONResponse:
    """Resume the incomplete batches of a failed or cancelled run."""
    try:
        settings = get_customer_forecast_settings()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT run_status, planning_month, config_checksum, total_batches "
                "FROM customer_forecast_run WHERE run_id = %s::uuid FOR UPDATE",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Customer forecast run not found")
            if row[0] not in {"failed", "cancelled"} or int(row[3] or 0) <= 0:
                raise HTTPException(
                    status_code=409,
                    detail="Customer forecast run has no resumable batches",
                )
            if row[2] != customer_forecast_config_checksum(settings):
                raise HTTPException(
                    status_code=409,
                    detail="Customer forecast configuration changed; start a new generation",
                )
            planning_month = row[1]
            cur.execute(
                "UPDATE customer_forecast_batch SET attempt_count = 0 "
                "WHERE run_id = %s::uuid AND batch_status = 'failed' "
                "AND attempt_count >= %s",
                (str(run_id), int(settings["max_batch_attempts"])),
            )
            cur.execute(
                "UPDATE customer_forecast_run SET run_status = 'queued', job_id = NULL, "
                "completed_at = NULL, error_summary = NULL WHERE run_id = %s::uuid",
                (str(run_id),),
            )
            conn.commit()
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Preparing customer forecast retry failed")
        raise HTTPException(status_code=500, detail="customer forecast retry failed") from exc

    from common.services.job_registry import JobManager

    try:
        job_id = JobManager().submit_job(
            "generate_customer_forecast",
            {"run_id": str(run_id)},
            label=f"Customer Forecast · {planning_month:%B %Y} · Resume",
            triggered_by="api",
            max_retries=1,
        )
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE customer_forecast_run SET job_id = %s WHERE run_id = %s::uuid",
                (job_id, str(run_id)),
            )
            conn.commit()
    except (RuntimeError, ValueError, psycopg.Error) as exc:
        logger.exception("Submitting customer forecast retry failed")
        try:
            with get_conn() as conn:
                mark_customer_forecast_run_terminal(
                    conn,
                    str(run_id),
                    "failed",
                    "retry job submission failed",
                )
        except psycopg.Error:
            logger.exception("Marking unsubmitted customer forecast retry failed")
        raise HTTPException(
            status_code=500, detail="customer forecast retry submission failed"
        ) from exc
    return JSONResponse(
        status_code=202,
        content={"run_id": str(run_id), "job_id": job_id, "status": "queued"},
    )


@router.get("/series")
def get_series(
    item_id: str = Query(min_length=1),
    location_id: str = Query(min_length=1),
    customer_no: str = Query(min_length=1),
    run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    try:
        run_row = _load_run(
            str(run_id) if run_id else None,
            latest_completed=run_id is None,
        )
        if run_row is None:
            raise HTTPException(status_code=404, detail="No customer forecast run found")
        run = _serialize_run(run_row)
        if run["status"] != "completed":
            raise HTTPException(status_code=409, detail="Customer forecast run is not completed")
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "WITH months AS ("
                "  SELECT generate_series(%s::date, %s::date, interval '1 month')::date "
                "         AS startdate"
                "), actuals AS ("
                "  SELECT startdate, SUM(demand_qty)::double precision AS demand_qty "
                "  FROM fact_customer_demand_monthly "
                "  WHERE item_id = %s AND location_id = %s AND customer_no = %s "
                "    AND startdate >= %s AND startdate <= %s "
                "  GROUP BY startdate"
                ") SELECT months.startdate, COALESCE(actuals.demand_qty, 0) "
                "FROM months LEFT JOIN actuals USING (startdate) ORDER BY months.startdate",
                (
                    run_row[4],
                    run_row[5].replace(day=1),
                    item_id,
                    location_id,
                    customer_no,
                    run_row[4],
                    run_row[5],
                ),
            )
            history = [
                {"month": month.isoformat(), "actual_qty": float(qty)}
                for month, qty in cur.fetchall()
            ]
            cur.execute(
                "SELECT forecast_month, forecast_qty, lower_bound, upper_bound, model_id "
                "FROM fact_customer_forecast "
                "WHERE run_id = %s::uuid AND item_id = %s AND location_id = %s "
                "AND customer_no = %s ORDER BY forecast_month",
                (run["run_id"], item_id, location_id, customer_no),
            )
            forecast = [
                {
                    "month": month.isoformat(),
                    "forecast_qty": float(qty),
                    "lower_bound": float(lower) if lower is not None else None,
                    "upper_bound": float(upper) if upper is not None else None,
                    "model_id": model_id,
                }
                for month, qty, lower, upper, model_id in cur.fetchall()
            ]
    except HTTPException:
        raise
    except psycopg.Error as exc:
        logger.exception("Loading customer forecast series failed")
        raise HTTPException(
            status_code=500, detail="customer forecast series lookup failed"
        ) from exc
    if not forecast:
        raise HTTPException(status_code=404, detail="Customer forecast series not found")
    return {
        "run": run,
        "item_id": item_id,
        "location_id": location_id,
        "customer_no": customer_no,
        "history": history,
        "forecast": forecast,
    }


def _csv_rows(
    run_id: str,
    item_id: str | None,
    location_id: str | None,
    customer_no: str | None,
) -> Iterator[str]:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "run_id",
            "item_id",
            "location_id",
            "customer_no",
            "forecast_month",
            "forecast_qty",
            "lower_bound",
            "upper_bound",
            "model_id",
            "history_end",
        ]
    )
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    clauses = ["run_id = %s::uuid"]
    params: list[Any] = [run_id]
    for column, value in (
        ("item_id", item_id),
        ("location_id", location_id),
        ("customer_no", customer_no),
    ):
        if value:
            clauses.append(column + " = %s")
            params.append(value)
    sql = (
        "SELECT run_id::text, item_id, location_id, customer_no, forecast_month, "
        "forecast_qty, lower_bound, upper_bound, model_id, history_end "
        "FROM fact_customer_forecast WHERE "
        + " AND ".join(clauses)
        + " ORDER BY item_id, location_id, customer_no, forecast_month"
    )
    with get_conn() as conn, conn.cursor(name="customer_forecast_export") as cur:
        cur.execute(sql, tuple(params))
        while batch := cur.fetchmany(5_000):
            for row in batch:
                writer.writerow(row)
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)


@router.get("/export")
def export_forecast(
    run_id: uuid.UUID,
    item_id: str | None = None,
    location_id: str | None = None,
    customer_no: str | None = None,
) -> StreamingResponse:
    try:
        row = _load_run(str(run_id))
    except psycopg.Error as exc:
        logger.exception("Loading customer forecast run for export failed")
        raise HTTPException(status_code=500, detail="customer forecast export failed") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Customer forecast run not found")
    if row[2] != "completed":
        raise HTTPException(status_code=409, detail="Customer forecast run is not completed")
    return StreamingResponse(
        _csv_rows(str(run_id), item_id, location_id, customer_no),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="customer_forecast_{run_id}.csv"'},
    )
