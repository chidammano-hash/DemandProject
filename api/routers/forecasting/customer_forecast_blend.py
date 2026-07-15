"""Customer rule-router backtest and bottom-up blended champion API."""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from api.auth import require_api_key
from api.core import get_conn, get_read_only_conn
from common.core.planning_date import get_planning_date
from common.services.cache import cached_sync, invalidate_group
from common.services.customer_forecast import (
    customer_forecast_config_checksum,
    get_customer_forecast_settings,
)
from common.services.customer_forecast_backtest import (
    customer_backtest_config_checksum,
    get_customer_backtest_settings,
)
from common.services.customer_forecast_backtest_population import (
    compute_customer_backtest_source_population,
)
from common.services.customer_forecast_blend_contract import (
    customer_blend_config_checksum,
    get_customer_blend_settings,
)
from common.services.customer_forecast_blend_readiness import (
    load_customer_blend_readiness,
    reserve_customer_blend_generation,
)
from common.services.forecast_generation import invalidate_generation_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/customer-forecast", tags=["customer-forecast"])

_SUBMISSION_RECONCILIATION_GRACE_SECONDS = 300


def _current_customer_blend_contract() -> tuple[str, str, str, str]:
    """Resolve checksums required for a blend to remain current in read APIs."""
    customer_settings = get_customer_forecast_settings()
    blend_settings = get_customer_blend_settings()
    backtest_settings = get_customer_backtest_settings()
    return (
        str(customer_settings["model_id"]),
        customer_forecast_config_checksum(customer_settings),
        customer_blend_config_checksum(blend_settings),
        customer_backtest_config_checksum(
            backtest_settings,
            blend_settings,
            customer_settings,
        ),
    )


def _shift_month(value: date, offset: int) -> date:
    absolute = value.year * 12 + value.month - 1 + offset
    return date(absolute // 12, absolute % 12 + 1, 1)


def _reconcile_customer_backtest_submissions(cur: Any) -> None:
    """Retire terminal and stale pre-submission backtest manifests."""
    cur.execute(
        """UPDATE customer_forecast_backtest_run run
           SET run_status = CASE
                   WHEN job.status = 'failed' THEN 'failed'
                   WHEN job.status = 'cancelled' THEN 'cancelled'
                   WHEN job.status = 'completed' AND run.run_status <> 'completed'
                   THEN 'failed'
                   ELSE run.run_status
               END,
               error_summary = CASE
                   WHEN job.status = 'failed' THEN 'managed job failed'
                   WHEN job.status = 'cancelled' THEN 'managed job cancelled'
                   WHEN job.status = 'completed' AND run.run_status <> 'completed'
                   THEN 'managed job completed without a completed backtest manifest'
                   ELSE run.error_summary
               END,
               completed_at = CASE
                   WHEN job.status IN ('failed', 'cancelled', 'completed')
                   THEN COALESCE(job.completed_at, NOW())
                   ELSE run.completed_at
               END
           FROM job_history job
           WHERE run.run_status IN ('queued', 'generating')
             AND job.job_type = 'generate_customer_forecast_backtest'
             AND job.status IN ('failed', 'cancelled', 'completed')
             AND job.params ->> 'run_id' = run.run_id::text"""
    )
    cur.execute(
        """UPDATE customer_forecast_backtest_run run
           SET run_status = 'failed',
               error_summary = 'job submission was not persisted',
               completed_at = NOW()
           WHERE run.run_status = 'queued'
             AND run.job_id IS NULL
             AND run.created_at < NOW() - (%s * INTERVAL '1 second')
             AND NOT EXISTS (
                 SELECT 1
                 FROM job_history job
                 WHERE job.job_type = 'generate_customer_forecast_backtest'
                   AND job.params ->> 'run_id' = run.run_id::text
             )""",
        (_SUBMISSION_RECONCILIATION_GRACE_SECONDS,),
    )


@router.post(
    "/backtest/generate",
    dependencies=[Depends(require_api_key)],
)
def generate_customer_backtest() -> JSONResponse:
    """Queue one config-bound rolling rule-router/blend accuracy backtest."""
    try:
        settings = get_customer_backtest_settings()
        customer_settings = get_customer_forecast_settings()
        blend_settings = get_customer_blend_settings()
        if not settings.enabled:
            raise HTTPException(status_code=409, detail="Customer forecast backtesting is disabled")
        planning_month = get_planning_date().replace(day=1)
        evaluation_end = _shift_month(planning_month, -1)
        evaluation_start = _shift_month(evaluation_end, 1 - settings.lookback_months)
        run_id = str(uuid.uuid4())
        with get_conn() as conn, conn.cursor() as cur:
            _reconcile_customer_backtest_submissions(cur)
            cur.execute(
                """SELECT run_id
                   FROM customer_forecast_run
                   WHERE run_status = 'completed'
                     AND model_id = %s
                     AND config_checksum = %s
                     AND planning_month = %s
                     AND source_customer_demand_batch_id IS NOT NULL
                     AND source_customer_demand_batch_id = (
                         SELECT batch_id
                         FROM audit_load_batch
                         WHERE domain = 'customer_demand'
                           AND status = 'completed'
                         ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                         LIMIT 1
                     )
                     AND source_customer_demand_batch_id = (
                         SELECT source_batch_id
                         FROM customer_demand_profile_refresh_state
                         WHERE singleton_id = 1
                     )
                     AND NOT EXISTS (
                         SELECT 1
                         FROM audit_load_batch
                         WHERE domain = 'customer_demand'
                           AND status = 'running'
                     )
                   ORDER BY completed_at DESC, created_at DESC
                   LIMIT 1""",
                (
                    customer_settings["model_id"],
                    customer_forecast_config_checksum(customer_settings),
                    planning_month,
                ),
            )
            customer = cur.fetchone()
            if customer is None:
                raise HTTPException(
                    status_code=409,
                    detail="Complete a current rule-routed customer forecast before backtesting",
                )
            customer_run_id = str(customer[0])
            cur.execute(
                """SELECT promotion.id, promotion.source_run_id,
                          promotion.production_run_id
                   FROM model_promotion_log promotion
                   JOIN forecast_generation_run generation
                     ON generation.run_id = promotion.source_run_id
                   WHERE promotion.is_active = TRUE
                     AND promotion.model_id = 'champion'
                     AND generation.forecast_month_generated = %s
                     AND NOT (generation.metadata ? 'customer_bottom_up_blend')
                   ORDER BY promotion.promoted_at DESC, promotion.id DESC
                   LIMIT 1""",
                (planning_month,),
            )
            source = cur.fetchone()
            if source is None or source[1] is None or source[2] is None:
                raise HTTPException(
                    status_code=409,
                    detail="Promote a fresh unblended champion before backtesting the blend",
                )
            population = compute_customer_backtest_source_population(
                cur,
                planning_month=planning_month,
                batch_size=settings.batch_size,
            )
            if population.series_count <= 0:
                raise HTTPException(
                    status_code=409,
                    detail="Load customer demand history before backtesting the blend",
                )
            cur.execute(
                """INSERT INTO customer_forecast_backtest_run
                       (run_id, run_status, customer_run_id, planning_month,
                        evaluation_start, evaluation_end, lookback_months,
                        min_train_months, horizon_months, batch_size,
                        source_series_count, source_series_checksum,
                        customer_model_id, blend_model_id, source_promotion_id,
                        source_production_run_id, config_checksum,
                        total_batches)
                   VALUES (%s::uuid, 'queued', %s::uuid, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s::uuid, %s, %s)""",
                (
                    run_id,
                    customer_run_id,
                    planning_month,
                    evaluation_start,
                    evaluation_end,
                    settings.lookback_months,
                    settings.min_train_months,
                    settings.horizon_months,
                    settings.batch_size,
                    population.series_count,
                    population.checksum,
                    customer_settings["model_id"],
                    blend_settings.model_id,
                    int(source[0]),
                    str(source[2]),
                    customer_backtest_config_checksum(settings, blend_settings, customer_settings),
                    population.batch_count,
                ),
            )
            conn.commit()
    except HTTPException:
        raise
    except psycopg.errors.UniqueViolation as exc:
        raise HTTPException(
            status_code=409,
            detail="A customer forecast backtest is already active",
        ) from exc
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Creating customer forecast backtest failed")
        raise HTTPException(
            status_code=500, detail="customer forecast backtest creation failed"
        ) from exc

    from common.services.job_registry import JobManager

    try:
        job_id = JobManager().submit_job(
            "generate_customer_forecast_backtest",
            {"run_id": run_id, "customer_run_id": customer_run_id},
            label=f"Customer Forecast Backtest · {planning_month:%B %Y}",
            triggered_by="api",
        )
    except (RuntimeError, ValueError, psycopg.Error) as exc:
        logger.exception("Submitting customer forecast backtest failed")
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """UPDATE customer_forecast_backtest_run
                       SET run_status = 'failed', error_summary = %s,
                           completed_at = NOW()
                       WHERE run_id = %s::uuid""",
                    ("job submission failed", run_id),
                )
                conn.commit()
        except psycopg.Error:
            logger.exception("Marking unsubmitted customer backtest failed")
        raise HTTPException(
            status_code=500, detail="customer forecast backtest job submission failed"
        ) from exc
    invalidate_group("customer_forecast")

    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "job_id": job_id, "status": "queued"},
    )


@router.get("/backtest/latest")
@cached_sync(ttl=5, group="customer_forecast")
def get_latest_customer_backtest() -> dict[str, Any]:
    """Return the latest run and three-model common-cohort accuracy comparison."""
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT run.run_id::text,
                          COALESCE(
                              run.job_id,
                              (
                                  SELECT job.job_id
                                  FROM job_history job
                                  WHERE job.job_type = 'generate_customer_forecast_backtest'
                                    AND job.params ->> 'run_id' = run.run_id::text
                                  ORDER BY job.submitted_at DESC
                                  LIMIT 1
                              )
                          ),
                          run.run_status,
                          run.customer_run_id::text, run.planning_month,
                          accuracy.common_months, accuracy.common_dfus,
                          accuracy.common_rows, accuracy.actual_qty,
                          run.component_checksum, run.completed_at,
                          accuracy.gate_passed, accuracy.gate_reason,
                          accuracy.blend_wape_degradation_pct,
                          accuracy.min_common_months,
                          accuracy.min_common_dfus,
                          accuracy.max_wape_degradation_pct,
                          run.error_summary
                   FROM customer_forecast_backtest_run run
                   LEFT JOIN customer_bottom_up_backtest_accuracy accuracy
                     ON accuracy.backtest_run_id = run.run_id
                   ORDER BY run.created_at DESC
                   LIMIT 1"""
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="No customer backtest run found")
            cur.execute(
                """SELECT model_id, observations, actual_qty, absolute_error,
                          mae, wape_pct, bias_pct, accuracy_pct
                   FROM (
                       SELECT 'champion' AS model_id, common_rows AS observations,
                              actual_qty, champion_absolute_error AS absolute_error,
                              champion_mae AS mae, champion_wape_pct AS wape_pct,
                              champion_bias_pct AS bias_pct,
                              champion_accuracy_pct AS accuracy_pct, 1 AS model_order
                       FROM customer_bottom_up_backtest_accuracy
                       WHERE backtest_run_id = %s::uuid
                       UNION ALL
                       SELECT 'customer_bottom_up', common_rows, actual_qty,
                              customer_absolute_error, customer_mae,
                              customer_wape_pct, customer_bias_pct,
                              customer_accuracy_pct, 2
                       FROM customer_bottom_up_backtest_accuracy
                       WHERE backtest_run_id = %s::uuid
                       UNION ALL
                       SELECT 'customer_bottom_up_blend', common_rows, actual_qty,
                              blend_absolute_error, blend_mae,
                              blend_wape_pct, blend_bias_pct,
                              blend_accuracy_pct, 3
                       FROM customer_bottom_up_backtest_accuracy
                       WHERE backtest_run_id = %s::uuid
                   ) metrics
                   ORDER BY model_order""",
                (row[0], row[0], row[0]),
            )
            metrics = cur.fetchall()
    except HTTPException:
        raise
    except psycopg.Error as exc:
        logger.exception("Loading latest customer forecast backtest failed")
        raise HTTPException(
            status_code=500, detail="customer forecast backtest lookup failed"
        ) from exc

    return {
        "run_id": row[0],
        "job_id": row[1],
        "status": row[2],
        "customer_run_id": row[3],
        "planning_month": row[4].isoformat() if row[4] else None,
        "common_months": int(row[5] or 0),
        "common_dfus": int(row[6] or 0),
        "common_rows": int(row[7] or 0),
        "actual_qty": float(row[8]) if row[8] is not None else None,
        "component_checksum": row[9],
        "completed_at": row[10].isoformat() if row[10] else None,
        "gate_passed": bool(row[11]) if row[11] is not None else None,
        "gate_reason": row[12],
        "blend_wape_degradation_pct": (float(row[13]) if row[13] is not None else None),
        "min_common_months": int(row[14]) if row[14] is not None else None,
        "min_common_dfus": int(row[15]) if row[15] is not None else None,
        "max_wape_degradation_pct": (float(row[16]) if row[16] is not None else None),
        "error_summary": row[17],
        "metrics": [
            {
                "model_id": metric[0],
                "observations": int(metric[1] or 0),
                "actual_qty": float(metric[2]) if metric[2] is not None else None,
                "absolute_error": float(metric[3]) if metric[3] is not None else None,
                "mae": float(metric[4]) if metric[4] is not None else None,
                "wape_pct": float(metric[5]) if metric[5] is not None else None,
                "bias_pct": float(metric[6]) if metric[6] is not None else None,
                "accuracy_pct": float(metric[7]) if metric[7] is not None else None,
            }
            for metric in metrics
        ],
    }


@router.post(
    "/blend/generate",
    dependencies=[Depends(require_api_key)],
)
def generate_customer_blend(customer_run_id: uuid.UUID | None = None) -> JSONResponse:
    """Queue a server-resolved, backtest-qualified bottom-up blend candidate."""
    run_id = uuid.uuid4()
    try:
        with get_conn() as conn:
            readiness = reserve_customer_blend_generation(
                conn,
                run_id=run_id,
                customer_run_id=customer_run_id,
            )
        if not readiness["ready"]:
            raise HTTPException(status_code=409, detail=str(readiness["blockers"][0]))
    except HTTPException:
        raise
    except psycopg.errors.UniqueViolation as exc:
        raise HTTPException(
            status_code=409,
            detail="A customer bottom-up blend is already generating",
        ) from exc
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Checking customer blend readiness failed")
        raise HTTPException(status_code=500, detail="customer blend readiness failed") from exc

    from common.services.job_registry import JobManager

    resolved_customer_run_id = str(readiness["customer_run_id"])
    try:
        job_id = JobManager().submit_job(
            "generate_customer_forecast_blend",
            {"run_id": str(run_id), "customer_run_id": resolved_customer_run_id},
            label="Customer Bottom-Up Blended Champion",
            triggered_by="api",
        )
    except (RuntimeError, ValueError, psycopg.Error) as exc:
        logger.exception("Submitting customer blend job failed")
        try:
            with get_conn() as conn, conn.cursor() as cur:
                invalidate_generation_run(cur, run_id, "job submission failed")
                conn.commit()
        except (OSError, RuntimeError, ValueError, psycopg.Error):
            logger.exception("Invalidating unsubmitted customer blend failed")
        raise HTTPException(status_code=500, detail="customer blend job submission failed") from exc
    invalidate_group("customer_forecast")
    return JSONResponse(
        status_code=202,
        content={"run_id": str(run_id), "job_id": job_id, "status": "queued"},
    )


@router.get("/blend/readiness")
def get_customer_blend_readiness(
    customer_run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Return the current server-resolved blend gate and immutable lineage."""
    try:
        with get_read_only_conn() as conn:
            readiness = load_customer_blend_readiness(
                conn,
                customer_run_id,
                require_backtest=True,
            )
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Loading customer blend readiness failed")
        raise HTTPException(
            status_code=500, detail="customer blend readiness lookup failed"
        ) from exc
    return {
        "ready": bool(readiness["ready"]),
        "blockers": list(readiness["blockers"]),
        "customer_run_id": readiness["customer_run_id"],
        "source_promotion_id": readiness["source_promotion_id"],
        "source_run_id": readiness["source_run_id"],
        "source_production_run_id": readiness["source_production_run_id"],
        "backtest_run_id": readiness["backtest_run_id"],
        "backtest_gate_passed": bool(readiness["backtest_gate_passed"]),
        "promotion_enabled": bool(readiness["promotion_enabled"]),
        "promotion_reason": readiness["promotion_reason"],
    }


@router.get("/blend/latest")
@cached_sync(ttl=5, group="customer_forecast")
def get_latest_customer_blend() -> dict[str, Any]:
    """Return the latest current-vintage bottom-up blend manifest and lineage."""
    planning_month = get_planning_date().replace(day=1)
    try:
        (
            customer_model_id,
            customer_config_checksum,
            blend_config_checksum,
            backtest_config_checksum,
        ) = _current_customer_blend_contract()
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT generation.run_id::text, generation.run_status,
                          generation.forecast_month_generated,
                          generation.horizon_months, generation.row_count,
                          generation.dfu_count, generation.completed_at,
                          generation.invalid_reason,
                          (
                              SELECT job.job_id
                              FROM job_history job
                              WHERE job.job_type = 'generate_customer_forecast_blend'
                                AND job.params ->> 'run_id' = generation.run_id::text
                              ORDER BY job.submitted_at DESC
                              LIMIT 1
                          ),
                          generation.metadata
                   FROM forecast_generation_run generation
                   WHERE generation.metadata ? 'customer_bottom_up_blend'
                     AND generation.forecast_month_generated = %s
                     AND (
                         (
                             generation.run_status IN ('generating', 'ready', 'invalid')
                             AND generation.metadata
                                     -> 'customer_bottom_up_blend'
                                     ->> 'config_checksum' = %s
                             AND generation.metadata
                                     -> 'customer_bottom_up_blend'
                                     ->> 'backtest_config_checksum' = %s
                             AND generation.metadata
                                     -> 'customer_bottom_up_blend'
                                     ->> 'customer_run_id' = (
                                 SELECT customer.run_id::text
                                 FROM customer_forecast_run customer
                                 WHERE customer.run_status = 'completed'
                                   AND customer.planning_month = %s
                                   AND customer.model_id = %s
                                   AND customer.config_checksum = %s
                                   AND customer.source_customer_demand_batch_id = (
                                       SELECT batch.batch_id
                                       FROM audit_load_batch batch
                                       WHERE batch.domain = 'customer_demand'
                                         AND batch.status = 'completed'
                                       ORDER BY batch.completed_at DESC NULLS LAST,
                                                batch.batch_id DESC
                                       LIMIT 1
                                   )
                                   AND customer.source_customer_demand_batch_id = (
                                       SELECT state.source_batch_id
                                       FROM customer_demand_profile_refresh_state state
                                       WHERE state.singleton_id = 1
                                   )
                                   AND NOT EXISTS (
                                       SELECT 1
                                       FROM audit_load_batch active_load
                                       WHERE active_load.domain = 'customer_demand'
                                         AND active_load.status = 'running'
                                   )
                                 ORDER BY customer.completed_at DESC,
                                          customer.created_at DESC
                                 LIMIT 1
                             )
                             AND EXISTS (
                                 SELECT 1
                                 FROM model_promotion_log promotion
                                 WHERE promotion.is_active = TRUE
                                   AND promotion.id::text = generation.metadata
                                           -> 'customer_bottom_up_blend'
                                           ->> 'source_promotion_id'
                                   AND promotion.production_run_id::text =
                                       generation.metadata
                                           -> 'customer_bottom_up_blend'
                                           ->> 'source_production_run_id'
                             )
                         )
                         OR (
                             generation.run_status = 'promoted'
                             AND EXISTS (
                                 SELECT 1
                                 FROM model_promotion_log promotion
                                 WHERE promotion.is_active = TRUE
                                   AND promotion.source_run_id = generation.run_id
                             )
                         )
                     )
                   ORDER BY generation.created_at DESC
                   LIMIT 1""",
                (
                    planning_month,
                    blend_config_checksum,
                    backtest_config_checksum,
                    planning_month,
                    customer_model_id,
                    customer_config_checksum,
                ),
            )
            row = cur.fetchone()
    except (KeyError, TypeError, ValueError, psycopg.Error) as exc:
        logger.exception("Loading latest customer bottom-up blend failed")
        raise HTTPException(status_code=500, detail="customer blend lookup failed") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="No customer blend run found")
    metadata = dict(row[9]) if isinstance(row[9], dict) else {}
    lineage = metadata.get("customer_bottom_up_blend")
    lineage = lineage if isinstance(lineage, dict) else {}
    return {
        "run_id": row[0],
        "status": row[1],
        "planning_month": row[2].isoformat() if row[2] else None,
        "horizon_months": int(row[3] or 0),
        "row_count": int(row[4] or 0),
        "dfu_count": int(row[5] or 0),
        "completed_at": row[6].isoformat() if row[6] else None,
        "invalid_reason": row[7],
        "job_id": row[8],
        "model_id": lineage.get("model_id"),
        "customer_run_id": lineage.get("customer_run_id"),
        "source_run_id": lineage.get("source_run_id"),
        "source_production_run_id": lineage.get("source_production_run_id"),
        "source_promotion_id": lineage.get("source_promotion_id"),
        "backtest_run_id": lineage.get("backtest_run_id"),
        "blended_row_count": int(lineage.get("blended_row_count") or 0),
        "champion_fallback_row_count": int(lineage.get("fallback_row_count") or 0),
        "customer_only_excluded_count": int(lineage.get("excluded_customer_dfu_count") or 0),
        "bottom_up_staging_run_id": lineage.get("bottom_up_staging_run_id"),
        "bottom_up_staging_status": lineage.get("bottom_up_staging_status"),
        "bottom_up_staging_row_count": int(lineage.get("bottom_up_staging_row_count") or 0),
        "bottom_up_staging_dfu_count": int(lineage.get("bottom_up_staging_dfu_count") or 0),
        "promotion_enabled": bool(
            isinstance(lineage.get("promotion"), dict) and lineage["promotion"].get("enabled")
        ),
        "backtest_gate": lineage.get("backtest_gate"),
    }


@router.get("/blend/series")
@cached_sync(ttl=120, group="customer_forecast")
def get_customer_blend_series(
    item_id: str = Query(min_length=1),
    location_id: str = Query(min_length=1),
    run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Return exact monthly raw, normalized, champion, and blended components."""
    planning_month = get_planning_date().replace(day=1)
    sql = """SELECT component.run_id::text,
                    component.customer_run_id::text,
                    generation.metadata->'customer_bottom_up_blend'->>'source_run_id',
                    component.source_production_run_id::text,
                    component.item_id, component.loc, component.forecast_month,
                    component.raw_customer_demand_qty,
                    component.normalized_customer_qty, component.champion_qty,
                    component.blended_qty, component.blended_lower,
                    component.blended_upper, component.fulfillment_ratio,
                    component.effective_customer_weight,
                    component.coverage_status, component.interval_method
             FROM customer_bottom_up_blend_component component
             JOIN forecast_generation_run generation
               ON generation.run_id = component.run_id
             WHERE component.item_id = %s AND component.loc = %s"""
    params: tuple[Any, ...] = (item_id, location_id)
    if run_id is not None:
        sql += " AND component.run_id = %s::uuid"
        params += (str(run_id),)
    else:
        try:
            (
                customer_model_id,
                customer_config_checksum,
                blend_config_checksum,
                backtest_config_checksum,
            ) = _current_customer_blend_contract()
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Loading current customer forecast contract failed")
            raise HTTPException(
                status_code=500,
                detail="customer blend series lookup failed",
            ) from exc
        sql += """ AND component.run_id = (
                        SELECT latest.run_id
                        FROM forecast_generation_run latest
                        WHERE latest.metadata ? 'customer_bottom_up_blend'
                          AND latest.forecast_month_generated = %s
                          AND (
                              (
                                  latest.run_status = 'ready'
                                  AND latest.metadata
                                          -> 'customer_bottom_up_blend'
                                          ->> 'config_checksum' = %s
                                  AND latest.metadata
                                          -> 'customer_bottom_up_blend'
                                          ->> 'backtest_config_checksum' = %s
                                  AND latest.metadata
                                          -> 'customer_bottom_up_blend'
                                          ->> 'customer_run_id' = (
                                      SELECT customer.run_id::text
                                      FROM customer_forecast_run customer
                                      WHERE customer.run_status = 'completed'
                                        AND customer.planning_month = %s
                                        AND customer.model_id = %s
                                        AND customer.config_checksum = %s
                                        AND customer.source_customer_demand_batch_id = (
                                            SELECT batch.batch_id
                                            FROM audit_load_batch batch
                                            WHERE batch.domain = 'customer_demand'
                                              AND batch.status = 'completed'
                                            ORDER BY batch.completed_at DESC NULLS LAST,
                                                     batch.batch_id DESC
                                            LIMIT 1
                                        )
                                        AND customer.source_customer_demand_batch_id = (
                                            SELECT state.source_batch_id
                                            FROM customer_demand_profile_refresh_state state
                                            WHERE state.singleton_id = 1
                                        )
                                        AND NOT EXISTS (
                                            SELECT 1
                                            FROM audit_load_batch active_load
                                            WHERE active_load.domain = 'customer_demand'
                                              AND active_load.status = 'running'
                                        )
                                      ORDER BY customer.completed_at DESC,
                                               customer.created_at DESC
                                      LIMIT 1
                                  )
                                  AND EXISTS (
                                      SELECT 1
                                      FROM model_promotion_log promotion
                                      WHERE promotion.is_active = TRUE
                                        AND promotion.id::text = latest.metadata
                                                -> 'customer_bottom_up_blend'
                                                ->> 'source_promotion_id'
                                        AND promotion.production_run_id::text =
                                            latest.metadata
                                                -> 'customer_bottom_up_blend'
                                                ->> 'source_production_run_id'
                                  )
                              )
                              OR (
                                  latest.run_status = 'promoted'
                                  AND EXISTS (
                                      SELECT 1
                                      FROM model_promotion_log promotion
                                      WHERE promotion.is_active = TRUE
                                        AND promotion.source_run_id = latest.run_id
                                  )
                              )
                          )
                        ORDER BY latest.created_at DESC
                        LIMIT 1
                    )"""
        params += (
            planning_month,
            blend_config_checksum,
            backtest_config_checksum,
            planning_month,
            customer_model_id,
            customer_config_checksum,
        )
    sql += " ORDER BY component.forecast_month"
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except psycopg.Error as exc:
        logger.exception("Loading customer bottom-up blend series failed")
        raise HTTPException(status_code=500, detail="customer blend series lookup failed") from exc
    if not rows:
        raise HTTPException(status_code=404, detail="Customer blend series not found")
    first = rows[0]
    return {
        "run_id": first[0],
        "customer_run_id": first[1],
        "source_run_id": first[2],
        "source_production_run_id": first[3],
        "item_id": first[4],
        "location_id": first[5],
        "months": [
            {
                "forecast_month": row[6].isoformat(),
                "raw_customer_demand_qty": float(row[7]) if row[7] is not None else None,
                "normalized_customer_qty": float(row[8]) if row[8] is not None else None,
                "champion_qty": float(row[9]),
                "blended_qty": float(row[10]),
                "lower_bound": float(row[11]) if row[11] is not None else None,
                "upper_bound": float(row[12]) if row[12] is not None else None,
                "fulfillment_ratio": float(row[13]) if row[13] is not None else None,
                "effective_customer_weight": float(row[14]),
                "coverage_status": row[15],
                "interval_method": row[16],
            }
            for row in rows
        ],
    }
