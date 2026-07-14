"""Durable batch execution for generation-only customer forecasts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import psycopg

from common.core.sql_helpers import read_sql_chunked
from common.core.utils import get_algorithm_params
from common.ml.chronos2_enriched import run_chronos2_enriched
from common.services.customer_forecast import (
    CustomerForecastWindow,
    _frame_checksum,
    _resolve_run_window,
    _shift_month,
    build_croston_forecast_rows,
    build_customer_forecast_rows,
    get_customer_forecast_settings,
    load_customer_forecast_readiness,
    prepare_customer_history,
)


@dataclass(frozen=True)
class CustomerForecastBatch:
    batch_id: int
    route_model_id: str
    route_batch_no: int
    series_count: int
    attempt_count: int


def _recent_start(window: CustomerForecastWindow, settings: dict[str, Any]) -> date:
    return _shift_month(
        window.forecast_start,
        -int(settings["recent_sales_lookback_months"]),
    )


def initialize_customer_forecast_batches(conn: Any, run_id: str) -> dict[str, Any]:
    """Create an idempotent series-to-batch manifest and resume unfinished work."""
    settings = get_customer_forecast_settings()
    if not settings["enabled"]:
        raise RuntimeError("Customer forecast generation is disabled")
    window, config_checksum = _resolve_run_window(conn, run_id, settings)
    readiness = load_customer_forecast_readiness(
        conn,
        window,
        recent_sales_lookback_months=int(settings["recent_sales_lookback_months"]),
    )
    if not readiness["ready"]:
        raise RuntimeError(str(readiness["blockers"][0]))

    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "UPDATE customer_forecast_run SET run_status = 'generating', "
            "started_at = COALESCE(started_at, NOW()), completed_at = NULL, "
            "error_summary = NULL WHERE run_id = %s::uuid "
            "AND run_status IN ('queued', 'generating', 'failed', 'cancelled')",
            (run_id,),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Customer forecast run is missing or not runnable")
        cur.execute(
            "UPDATE customer_forecast_batch SET batch_status = 'pending', "
            "started_at = NULL, error_summary = NULL "
            "WHERE run_id = %s::uuid AND batch_status = 'running'",
            (run_id,),
        )
        cur.execute(
            "SELECT COUNT(*) FROM customer_forecast_batch WHERE run_id = %s::uuid",
            (run_id,),
        )
        existing_batches = int((cur.fetchone() or (0,))[0])
        if existing_batches == 0:
            cur.execute(
                """
                CREATE TEMP TABLE temp_customer_forecast_manifest ON COMMIT DROP AS
                WITH classified AS (
                    SELECT profile.item_id, profile.location_id, profile.customer_no,
                           CASE
                               WHEN profile.first_month <= %s THEN %s
                               ELSE %s
                           END AS route_model_id
                    FROM mv_customer_demand_series_profile profile
                    WHERE profile.first_month < %s
                      AND profile.last_sales_month >= %s
                )
                SELECT item_id, location_id, customer_no, route_model_id,
                       ((ROW_NUMBER() OVER (
                           PARTITION BY route_model_id
                           ORDER BY item_id, location_id, customer_no
                       ) - 1) / %s)::integer AS route_batch_no
                FROM classified
                """,
                (
                    window.history_start,
                    str(settings["model_id"]),
                    str(settings["fallback_model_id"]),
                    window.forecast_start,
                    _recent_start(window, settings),
                    int(settings["batch_size"]),
                ),
            )
            cur.execute("ANALYZE temp_customer_forecast_manifest")
            cur.execute(
                "INSERT INTO customer_forecast_batch "
                "(run_id, route_model_id, route_batch_no, series_count) "
                "SELECT %s::uuid, route_model_id, route_batch_no, COUNT(*) "
                "FROM temp_customer_forecast_manifest "
                "GROUP BY route_model_id, route_batch_no",
                (run_id,),
            )
            cur.execute(
                "INSERT INTO customer_forecast_batch_series "
                "(run_id, batch_id, item_id, location_id, customer_no) "
                "SELECT %s::uuid, batch.batch_id, manifest.item_id, "
                "manifest.location_id, manifest.customer_no "
                "FROM temp_customer_forecast_manifest manifest "
                "JOIN customer_forecast_batch batch "
                "ON batch.run_id = %s::uuid "
                "AND batch.route_model_id = manifest.route_model_id "
                "AND batch.route_batch_no = manifest.route_batch_no",
                (run_id, run_id),
            )

        cur.execute(
            "SELECT route_model_id, SUM(series_count), COUNT(*) "
            "FROM customer_forecast_batch WHERE run_id = %s::uuid "
            "GROUP BY route_model_id ORDER BY route_model_id",
            (run_id,),
        )
        route_rows = cur.fetchall()
        route_counts = {str(row[0]): int(row[1]) for row in route_rows}
        total_series = sum(route_counts.values())
        total_batches = sum(int(row[2]) for row in route_rows)
        if total_series <= 0 or total_series != int(readiness["forecastable_series"]):
            raise RuntimeError("Customer forecast batch manifest is incomplete")
        cur.execute(
            "SELECT COALESCE(SUM(completed_series), 0), "
            "COUNT(*) FILTER (WHERE batch_status = 'completed'), "
            "COALESCE(SUM(row_count), 0) "
            "FROM customer_forecast_batch WHERE run_id = %s::uuid",
            (run_id,),
        )
        completed_series, completed_batches, row_count = cur.fetchone() or (0, 0, 0)
        cur.execute(
            "UPDATE customer_forecast_run SET eligible_series = %s, total_series = %s, "
            "completed_series = %s, total_batches = %s, completed_batches = %s, "
            "row_count = %s, skipped_series = %s, skip_reason_counts = %s::jsonb, "
            "model_route_counts = %s::jsonb, config_checksum = %s "
            "WHERE run_id = %s::uuid",
            (
                total_series,
                total_series,
                int(completed_series or 0),
                total_batches,
                int(completed_batches or 0),
                int(row_count or 0),
                int(readiness["skipped_series"]),
                json.dumps(
                    {"no_sales_last_6_months": int(readiness["skipped_series"])},
                    sort_keys=True,
                ),
                json.dumps(route_counts, sort_keys=True),
                config_checksum,
                run_id,
            ),
        )
    return load_customer_forecast_progress(conn, run_id)


def claim_customer_forecast_batch(
    conn: Any,
    run_id: str,
    route_model_ids: Sequence[str],
    *,
    max_attempts: int,
) -> CustomerForecastBatch | None:
    """Claim one unfinished route batch without colliding with other workers."""
    if not route_model_ids:
        raise ValueError("Customer forecast worker requires at least one route")
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            WITH next_batch AS (
                SELECT batch_id
                FROM customer_forecast_batch
                WHERE run_id = %s::uuid
                  AND route_model_id = ANY(%s)
                  AND batch_status IN ('pending', 'failed')
                  AND attempt_count < %s
                ORDER BY route_model_id, route_batch_no
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE customer_forecast_batch batch
            SET batch_status = 'running',
                attempt_count = attempt_count + 1,
                started_at = NOW(),
                completed_at = NULL,
                error_summary = NULL
            FROM next_batch
            WHERE batch.batch_id = next_batch.batch_id
            RETURNING batch.batch_id, batch.route_model_id,
                      batch.route_batch_no, batch.series_count, batch.attempt_count
            """,
            (run_id, list(route_model_ids), max_attempts),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return CustomerForecastBatch(
        batch_id=int(row[0]),
        route_model_id=str(row[1]),
        route_batch_no=int(row[2]),
        series_count=int(row[3]),
        attempt_count=int(row[4]),
    )


def load_customer_forecast_batch_history(
    conn: Any,
    batch_id: int,
    window: CustomerForecastWindow,
) -> pd.DataFrame:
    """Load only the bounded history belonging to one durable batch."""
    return read_sql_chunked(
        conn,
        """
        SELECT series.item_id, series.location_id, series.customer_no,
               history.startdate,
               COALESCE(SUM(history.demand_qty), 0)::double precision AS demand_qty,
               COALESCE(SUM(history.sales_qty), 0)::double precision AS sales_qty,
               profile.first_month AS series_first_month
        FROM customer_forecast_batch_series series
        JOIN mv_customer_demand_series_profile profile
          USING (item_id, location_id, customer_no)
        LEFT JOIN fact_customer_demand_monthly history
          ON history.item_id = series.item_id
         AND history.location_id = series.location_id
         AND history.customer_no = series.customer_no
         AND history.startdate >= %s
         AND history.startdate < %s
        WHERE series.batch_id = %s
        GROUP BY series.item_id, series.location_id, series.customer_no,
                 history.startdate, profile.first_month
        ORDER BY series.item_id, series.location_id, series.customer_no,
                 history.startdate
        """,
        params=(window.history_start, window.forecast_start, batch_id),
    )


def _build_batch_rows(
    batch: CustomerForecastBatch,
    raw_history: pd.DataFrame,
    window: CustomerForecastWindow,
    settings: dict[str, Any],
    predictor: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_customer_history(
        raw_history,
        window,
        recent_sales_lookback_months=int(settings["recent_sales_lookback_months"]),
    )
    route_counts = {
        str(settings["model_id"]): prepared.eligible_series_count,
        str(settings["fallback_model_id"]): prepared.fallback_series_count,
    }
    if route_counts.get(batch.route_model_id) != batch.series_count:
        raise RuntimeError("Customer forecast batch route changed after manifest creation")

    if batch.route_model_id == settings["model_id"]:
        predictions = predictor(
            prepared.model_input,
            [pd.Timestamp(month) for month in window.forecast_months],
            get_algorithm_params(str(settings["model_id"])),
        )
        rows = build_customer_forecast_rows(
            prepared,
            predictions,
            window,
            model_id=str(settings["model_id"]),
        )
        source_input = prepared.model_input
    elif batch.route_model_id == settings["fallback_model_id"]:
        rows = build_croston_forecast_rows(
            prepared,
            window,
            dict(settings["fallback_params"]),
        )
        source_input = prepared.fallback_model_input
    else:
        raise RuntimeError("Customer forecast batch has an unsupported route")
    expected_rows = batch.series_count * window.horizon_months
    if (
        len(rows) != expected_rows
        or rows.duplicated(["item_id", "location_id", "customer_no", "forecast_month"]).any()
    ):
        raise RuntimeError("Customer forecast batch output is incomplete or duplicated")
    return rows, source_input


def _persist_completed_batch(
    conn: Any,
    run_id: str,
    batch: CustomerForecastBatch,
    rows: pd.DataFrame,
    source_checksum: str,
    history_end: Any,
) -> None:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_customer_forecast WHERE run_id = %s::uuid AND batch_id = %s",
            (run_id, batch.batch_id),
        )
        with cur.copy(
            "COPY fact_customer_forecast "
            "(run_id, batch_id, item_id, location_id, customer_no, forecast_month, "
            "forecast_qty, lower_bound, upper_bound, model_id, history_end) FROM STDIN"
        ) as copy:
            for row in rows.itertuples(index=False):
                copy.write_row(
                    (
                        run_id,
                        batch.batch_id,
                        row.item_id,
                        row.location_id,
                        row.customer_no,
                        row.forecast_month.date(),
                        float(row.forecast_qty),
                        row.lower_bound,
                        row.upper_bound,
                        row.model_id,
                        history_end,
                    )
                )
        cur.execute(
            "UPDATE customer_forecast_batch SET batch_status = 'completed', "
            "completed_series = series_count, row_count = %s, source_checksum = %s, "
            "completed_at = NOW(), error_summary = NULL "
            "WHERE batch_id = %s AND run_id = %s::uuid AND batch_status = 'running'",
            (len(rows), source_checksum, batch.batch_id, run_id),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Customer forecast batch lost its running claim")
        cur.execute(
            "UPDATE customer_forecast_run SET completed_series = completed_series + %s, "
            "completed_batches = completed_batches + 1, row_count = row_count + %s "
            "WHERE run_id = %s::uuid AND run_status = 'generating'",
            (batch.series_count, len(rows), run_id),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Customer forecast run is no longer generating")


def _mark_batch_failed(conn: Any, batch_id: int, error_summary: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE customer_forecast_batch SET batch_status = 'failed', "
            "error_summary = %s, completed_at = NOW() "
            "WHERE batch_id = %s AND batch_status = 'running'",
            (error_summary[:500], batch_id),
        )
    conn.commit()


def run_customer_forecast_worker(
    conn: Any,
    run_id: str,
    route_model_ids: Sequence[str],
    *,
    predictor: Callable[..., pd.DataFrame] = run_chronos2_enriched,
) -> int:
    """Claim and commit batches until no eligible work remains."""
    settings = get_customer_forecast_settings()
    window, _checksum = _resolve_run_window(conn, run_id, settings)
    completed = 0
    while batch := claim_customer_forecast_batch(
        conn,
        run_id,
        route_model_ids,
        max_attempts=int(settings["max_batch_attempts"]),
    ):
        try:
            raw_history = load_customer_forecast_batch_history(conn, batch.batch_id, window)
            rows, source_input = _build_batch_rows(
                batch,
                raw_history,
                window,
                settings,
                predictor,
            )
            _persist_completed_batch(
                conn,
                run_id,
                batch,
                rows,
                _frame_checksum(source_input),
                window.history_end,
            )
            completed += 1
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, psycopg.Error) as exc:
            conn.rollback()
            _mark_batch_failed(conn, batch.batch_id, str(exc))
            raise
    return completed


def load_customer_forecast_progress(conn: Any, run_id: str) -> dict[str, Any]:
    """Return exact customer-series and batch progress plus a throughput ETA."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_status, total_series, completed_series, total_batches, "
            "completed_batches, row_count, started_at, completed_at, model_route_counts "
            "FROM customer_forecast_run WHERE run_id = %s::uuid",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("Customer forecast run is missing")
    status = str(row[0])
    total_series = int(row[1] or 0)
    completed_series = int(row[2] or 0)
    total_batches = int(row[3] or 0)
    completed_batches = int(row[4] or 0)
    started_at = row[6]
    completed_at = row[7]
    if status == "completed":
        progress_pct = 100
    elif total_series > 0:
        progress_pct = min(99, 10 + int(89 * completed_series / total_series))
    else:
        progress_pct = 5
    eta_seconds: int | None = None
    if completed_series > 0 and completed_series < total_series and started_at is not None:
        end = completed_at or datetime.now(UTC)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        elapsed = max((end - started_at).total_seconds(), 1.0)
        throughput = completed_series / elapsed
        eta_seconds = int((total_series - completed_series) / throughput)
    return {
        "run_id": run_id,
        "status": status,
        "total_series": total_series,
        "completed_series": completed_series,
        "total_batches": total_batches,
        "completed_batches": completed_batches,
        "row_count": int(row[5] or 0),
        "progress_pct": progress_pct,
        "eta_seconds": eta_seconds,
        "model_route_counts": row[8] or {},
    }


def finalize_customer_forecast_batches(conn: Any, run_id: str) -> dict[str, Any]:
    """Validate every committed batch and atomically publish the run as complete."""
    settings = get_customer_forecast_settings()
    window, config_checksum = _resolve_run_window(conn, run_id, settings)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, route_model_id, route_batch_no, batch_status, "
            "series_count, completed_series, row_count, source_checksum "
            "FROM customer_forecast_batch WHERE run_id = %s::uuid "
            "ORDER BY route_model_id, route_batch_no FOR UPDATE",
            (run_id,),
        )
        batches = cur.fetchall()
        if not batches or any(row[3] != "completed" for row in batches):
            raise RuntimeError("Customer forecast run still has incomplete batches")
        total_series = sum(int(row[4]) for row in batches)
        completed_series = sum(int(row[5]) for row in batches)
        row_count = sum(int(row[6]) for row in batches)
        expected_rows = total_series * window.horizon_months
        if completed_series != total_series or row_count != expected_rows:
            raise RuntimeError("Customer forecast batch totals are incomplete")
        cur.execute(
            "SELECT COUNT(*) FROM fact_customer_forecast WHERE run_id = %s::uuid",
            (run_id,),
        )
        persisted_rows = int((cur.fetchone() or (0,))[0])
        if persisted_rows != expected_rows:
            raise RuntimeError("Customer forecast persisted rows are incomplete")
        checksum_payload = [[str(row[1]), int(row[2]), str(row[7])] for row in batches]
        source_checksum = hashlib.sha256(
            json.dumps(checksum_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cur.execute(
            "UPDATE customer_forecast_run SET run_status = 'completed', "
            "eligible_series = %s, total_series = %s, completed_series = %s, "
            "total_batches = %s, completed_batches = %s, row_count = %s, "
            "config_checksum = %s, source_checksum = %s, completed_at = NOW(), "
            "error_summary = NULL WHERE run_id = %s::uuid AND run_status = 'generating'",
            (
                total_series,
                total_series,
                completed_series,
                len(batches),
                len(batches),
                row_count,
                config_checksum,
                source_checksum,
                run_id,
            ),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Customer forecast run is no longer generating")
    return load_customer_forecast_progress(conn, run_id)
