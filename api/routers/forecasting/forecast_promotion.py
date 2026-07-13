"""Run-scoped forecast generation and transactional promotion API."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key
from api.core import get_conn
from common.core.constants import CHAMPION_MODEL_ID
from common.core.planning_date import get_planning_date
from common.core.utils import get_forecastable_model_ids, load_forecast_pipeline_config
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_promotion import (
    PromotionConflictError,
    promote_forecast_run,
)

from ._forecast_promotion_models import (
    ForecastGenerationSubmittedResponse,
    ForecastPromotionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest-management", tags=["backtest-management"])


def _validate_forecast_model_id(model_id: str) -> None:
    """Reject model routes outside the canonical production roster."""
    valid_models = [CHAMPION_MODEL_ID, *get_forecastable_model_ids()]
    if model_id not in valid_models:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown forecast model '{model_id}'. Valid models: {valid_models}",
        )

# ---------------------------------------------------------------------------
# Promotion & Candidate Endpoints
# ---------------------------------------------------------------------------


@router.get("/promotion-status")
def get_promotion_status():
    """Get the currently active model promotion.

    Returns {"promoted": {...}} when a model is promoted, or {"promoted": null} when none active.
    The promoted dict contains: id, model_id, promotion_type, champion_experiment_id,
    plan_version, promoted_at, dfu_count, total_rows, promoted_by, notes.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, model_id, promotion_type, champion_experiment_id,
                       plan_version, promoted_at, dfu_count, total_rows,
                       promoted_by, notes, source_run_id, production_run_id,
                       candidate_checksum, production_checksum,
                       archive_checksum, archived_at
                FROM model_promotion_log
                WHERE is_active = TRUE
                ORDER BY promoted_at DESC
                LIMIT 1
            """)
            row = cur.fetchone()
    except psycopg.Error:
        logger.debug("model_promotion_log table may not exist yet")
        return {"promoted": None}
    if not row:
        return {"promoted": None}
    return {
        "promoted": {
            "id": row[0],
            "model_id": row[1],
            "promotion_type": row[2],
            "champion_experiment_id": row[3],
            "plan_version": row[4],
            "promoted_at": row[5].isoformat() if row[5] else None,
            "dfu_count": row[6],
            "total_rows": row[7],
            "promoted_by": row[8],
            "notes": row[9],
            "source_run_id": str(row[10]) if row[10] else None,
            "production_run_id": str(row[11]) if row[11] else None,
            "candidate_checksum": row[12],
            "production_checksum": row[13],
            "archive_checksum": row[14],
            "archived_at": row[15].isoformat() if row[15] else None,
        }
    }


@router.get("/candidate-summary")
def get_candidate_summary():
    """Get summary of loaded candidate forecasts per model."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT model_id,
                       COUNT(*) AS row_count,
                       COUNT(DISTINCT item_id || '|' || loc) AS dfu_count,
                       MAX(loaded_at) AS last_loaded_at,
                       AVG(accuracy_pct) FILTER (WHERE accuracy_pct IS NOT NULL) AS avg_accuracy
                FROM fact_candidate_forecast
                GROUP BY model_id
                ORDER BY model_id
            """)
            rows = cur.fetchall()
    except psycopg.Error:
        logger.debug("fact_candidate_forecast table may not exist yet")
        return {}
    result: dict[str, Any] = {}
    for r in rows:
        result[r[0]] = {
            "model_id": r[0],
            "row_count": r[1],
            "dfu_count": r[2],
            "last_loaded_at": r[3].isoformat() if r[3] else None,
            "avg_accuracy": float(r[4]) if r[4] is not None else None,
        }
    return result


@router.get("/staging-summary")
def get_staging_summary():
    """Latest immutable release-candidate run per requested model."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                WITH ranked_runs AS (
                    SELECT generation.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY requested_model_id
                               ORDER BY completed_at DESC NULLS LAST, created_at DESC, run_id
                           ) AS run_rank
                    FROM forecast_generation_run generation
                    WHERE generation_purpose = 'release_candidate'
                      AND run_status IN ('ready', 'promoted')
                      AND metadata ->> %s = %s
                )
                SELECT generation.requested_model_id,
                       generation.run_id,
                       generation.run_status,
                       generation.promotion_eligible,
                       COUNT(staging.id) AS row_count,
                       COUNT(DISTINCT (staging.item_id, staging.loc)) AS dfu_count,
                       generation.forecast_month_generated,
                       generation.completed_at,
                       MIN(staging.forecast_month) AS min_forecast_month,
                       MAX(staging.forecast_month) AS max_forecast_month,
                       generation.candidate_model_count
                FROM ranked_runs generation
                JOIN fact_production_forecast_staging staging
                  ON staging.run_id = generation.run_id
                 AND staging.generation_purpose = generation.generation_purpose
                 AND staging.candidate_model_id = generation.requested_model_id
                WHERE generation.run_rank = 1
                GROUP BY generation.requested_model_id, generation.run_id,
                         generation.run_status, generation.promotion_eligible,
                         generation.forecast_month_generated,
                         generation.completed_at,
                         generation.candidate_model_count
                ORDER BY generation.requested_model_id
            """,
                (
                    GENERATOR_CONTRACT_METADATA_KEY,
                    GENERATOR_CONTRACT_VERSION,
                ),
            )
            rows = cur.fetchall()
    except psycopg.Error:
        logger.debug("immutable forecast staging schema may not exist yet")
        return {}
    result: dict[str, Any] = {}
    for r in rows:
        result[r[0]] = {
            "model_id": r[0],
            "source_run_id": str(r[1]),
            "run_status": r[2],
            "promotion_eligible": bool(r[3]),
            "generation_purpose": "release_candidate",
            "row_count": r[4],
            "dfu_count": r[5],
            "forecast_month_generated": r[6].isoformat() if r[6] else None,
            "last_generated_at": r[7].isoformat() if r[7] else None,
            "min_forecast_month": r[8].isoformat() if r[8] else None,
            "max_forecast_month": r[9].isoformat() if r[9] else None,
            "source_model_count": r[10],
        }
    return result


@router.post(
    "/{model_id}/generate",
    status_code=201,
    response_model=ForecastGenerationSubmittedResponse,
    dependencies=[Depends(require_api_key)],
)
def submit_generate_forecast(
    model_id: str,
    horizon: int | None = None,
    confidence_intervals: bool | None = None,
):
    """Submit production forecast generation for a model, writing to staging.

    Args:
        model_id: Algorithm to generate forecasts for.
        horizon: Months ahead to forecast. Omitted → pipeline config default.
        confidence_intervals: Force CI (P10/P90) bands on/off. Omitted → config
            default (``confidence_interval.enabled`` in the pipeline config).

    The horizon and CI flags are threaded into the job params so they reach
    ``generate_production_forecasts.py`` — previously they were dropped for
    single-model generation, silently ignoring the panel's controls.
    """
    _validate_forecast_model_id(model_id)

    from common.services.job_registry import JobManager

    jm = JobManager()
    source_run_id = uuid4()
    params: dict[str, Any] = {
        "run_id": str(source_run_id),
        "generation_purpose": "release_candidate",
    }
    if model_id != CHAMPION_MODEL_ID:
        params["model_id"] = model_id
    if horizon is not None:
        params["horizon"] = horizon
    if confidence_intervals is not None:
        params["confidence_intervals"] = confidence_intervals
    job_id = jm.submit_job(
        job_type="generate_production_forecast",
        params=params,
        label=f"Generate Forecast: {model_id}",
    )
    return ForecastGenerationSubmittedResponse(
        job_id=job_id,
        model_id=model_id,
        source_run_id=source_run_id,
    )


def _load_transactional_promotion_policy() -> dict[str, Any]:
    config = load_forecast_pipeline_config()
    release = config["champion"]["release_readiness"]
    snapshot = config["forecast_snapshot"]
    production = config["production_forecast"]
    if not release["enabled"]:
        raise PromotionConflictError(
            "candidate_gate_failed",
            "Forecast release policy is disabled; promotion fails closed.",
        )
    return {
        "required_months": int(snapshot["lag_count"]),
        "min_coverage_frac": float(release["min_current_plan_coverage_frac"]),
        "min_ci_coverage_frac": float(release["min_confidence_interval_coverage_frac"]),
        "min_history_months": int(production["cold_start_min_months"]),
        "active_window_months": int(snapshot["active_window_months"]),
        "quality_lookback_months": int(release["lookback_months"]),
        "min_relative_wape_lift_vs_naive_pct": float(
            release["min_relative_wape_lift_vs_naive_pct"]
        ),
        "min_accuracy_delta_vs_external_pct_points": float(
            release["min_accuracy_delta_vs_external_pct_points"]
        ),
        "max_abs_bias_pct": float(release["max_abs_bias_pct"]),
        "min_common_cohort_coverage_frac": float(release["min_common_cohort_coverage_frac"]),
        "min_common_cohort_closed_months": int(release["min_common_cohort_closed_months"]),
        "min_common_cohort_dfus": int(release["min_common_cohort_dfus"]),
        "min_common_cohort_actual_volume": float(release["min_common_cohort_actual_volume"]),
    }


@router.post(
    "/{model_id}/promote",
    status_code=201,
    response_model=ForecastPromotionResponse,
    dependencies=[Depends(require_api_key)],
)
def promote_model(
    model_id: str,
    source_run_id: UUID,
    notes: str | None = None,
    promoted_by: str | None = None,
) -> ForecastPromotionResponse:
    """Atomically promote one explicit governed-champion release candidate."""
    _validate_forecast_model_id(model_id)
    if model_id != CHAMPION_MODEL_ID:
        raise HTTPException(
            status_code=409,
            detail=(
                "champion_release_required: Only the governed champion candidate can be "
                "promoted to production; single-model candidates are diagnostic evidence."
            ),
        )
    try:
        policy = _load_transactional_promotion_policy()
        with get_conn() as conn:
            result = promote_forecast_run(
                conn,
                model_id=model_id,
                source_run_id=source_run_id,
                planning_month=get_planning_date().replace(day=1),
                promoted_by=promoted_by or "api",
                notes=notes,
                policy=policy,
            )
    except PromotionConflictError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"{exc.code}: {exc.public_message}",
        ) from None
    except (KeyError, TypeError, ValueError):
        logger.exception("Forecast promotion policy is invalid")
        raise HTTPException(
            status_code=500,
            detail="Forecast promotion policy is invalid",
        ) from None
    except psycopg.Error:
        logger.exception("Forecast promotion transaction failed")
        raise HTTPException(
            status_code=500,
            detail="Forecast promotion failed",
        ) from None
    except (OSError, RuntimeError):
        logger.exception("Forecast promotion failed outside the database transaction")
        raise HTTPException(
            status_code=500,
            detail="Forecast promotion failed",
        ) from None

    try:
        from common.core.mv_refresh import refresh_for_tables
        from common.services.cache import get_cache

        refresh_for_tables(["fact_production_forecast", "fact_forecast_snapshot"])
        get_cache().invalidate("ds:forecast_release:*")
    except psycopg.Error:
        logger.exception("Post-promotion forecast refresh failed")

    return ForecastPromotionResponse(**result.__dict__)
