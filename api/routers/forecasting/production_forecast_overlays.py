"""Item Analysis staging and historical candidate forecast overlays."""

from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException, Query
from psycopg import sql

from api.core import get_conn
from common.core.constants import CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID
from common.services.customer_forecast_blend_contract import (
    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
)
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["production-forecast"])


@router.get("/forecast/production/staging")
async def get_staging_forecasts(
    item_id: str = Query(...),
    loc: str = Query(...),
):
    """Return the latest reviewable staging run for each display model."""
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """WITH classified_runs AS (
                       SELECT generation.*,
                              CASE
                                  WHEN generation.metadata ? %s
                                  THEN %s
                                  ELSE generation.requested_model_id
                              END AS display_model_id
                       FROM forecast_generation_run generation
                       WHERE generation_purpose IN (
                                 'release_candidate', 'shadow_candidate'
                             )
                         AND run_status IN ('ready', 'promoted')
                         AND metadata ->> %s = %s
                   ), ranked_runs AS (
                       SELECT classified.*,
                              ROW_NUMBER() OVER (
                                  PARTITION BY display_model_id
                                  ORDER BY completed_at DESC NULLS LAST,
                                           created_at DESC, run_id
                              ) AS run_rank
                       FROM classified_runs classified
                   )
                   SELECT generation.display_model_id,
                          staging.model_id AS source_model_id,
                          staging.forecast_month, staging.forecast_qty,
                          forecast_qty_lower, forecast_qty_upper,
                          staging.horizon_months, staging.cluster_id,
                          staging.lag_source, staging.generated_at,
                          generation.run_id
                   FROM ranked_runs generation
                   JOIN fact_production_forecast_staging staging
                     ON staging.run_id = generation.run_id
                    AND staging.generation_purpose = generation.generation_purpose
                    AND staging.candidate_model_id = generation.requested_model_id
                   WHERE generation.run_rank = 1
                     AND staging.item_id = %s AND staging.loc = %s
                   ORDER BY generation.display_model_id, staging.forecast_month""",
                (
                    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
                    CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID,
                    GENERATOR_CONTRACT_METADATA_KEY,
                    GENERATOR_CONTRACT_VERSION,
                    item_id,
                    loc,
                ),
            )
            rows = cur.fetchall()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            logger.warning("Forecast staging schema is not installed")
            return {"item_id": item_id, "loc": loc, "models": {}}
        except psycopg.Error as exc:
            logger.exception("Failed to read immutable staging forecasts")
            raise HTTPException(
                status_code=500,
                detail="staging forecast lookup failed",
            ) from exc

    models: dict[str, list] = {}
    for row in rows:
        model_id = row[0]
        models.setdefault(model_id, []).append(
            {
                "source_model_id": row[1],
                "forecast_month": row[2].isoformat() if row[2] else None,
                "forecast_qty": float(row[3]) if row[3] is not None else None,
                "forecast_qty_lower": float(row[4]) if row[4] is not None else None,
                "forecast_qty_upper": float(row[5]) if row[5] is not None else None,
                "horizon_months": row[6],
                "cluster_id": row[7],
                "lag_source": row[8],
                "generated_at": row[9].isoformat() if row[9] else None,
                "source_run_id": str(row[10]),
            }
        )
    return {"item_id": item_id, "loc": loc, "models": models}


@router.get("/forecast/candidate")
async def get_candidate_forecasts(
    item_id: str = Query(...),
    loc: str = Query(...),
    model_id: str | None = Query(default=None),
):
    """Return per-model historical backtest predictions for one DFU."""
    params: list[str] = [item_id, loc]
    model_filter = sql.SQL("")
    if model_id:
        model_filter = sql.SQL(" AND model_id = %s")
        params.append(model_id)
    query = sql.SQL(
        """SELECT model_id, forecast_month, forecast_qty,
                  forecast_qty_lower, forecast_qty_upper,
                  actual_qty, accuracy_pct, wape, bias,
                  horizon_months, cluster_id
           FROM fact_candidate_forecast
           WHERE item_id = %s AND loc = %s{model_filter}
           ORDER BY model_id, forecast_month"""
    ).format(model_filter=model_filter)

    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(query, params)
            rows = cur.fetchall()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            logger.warning("Candidate forecast schema is not installed")
            return {"item_id": item_id, "loc": loc, "models": {}}
        except psycopg.Error as exc:
            logger.exception("Failed to read candidate forecasts")
            raise HTTPException(
                status_code=500,
                detail="candidate forecast lookup failed",
            ) from exc

    models: dict[str, list] = {}
    for row in rows:
        model_key = row[0]
        models.setdefault(model_key, []).append(
            {
                "forecast_month": row[1].isoformat() if row[1] else None,
                "forecast_qty": float(row[2]) if row[2] is not None else None,
                "forecast_qty_lower": float(row[3]) if row[3] is not None else None,
                "forecast_qty_upper": float(row[4]) if row[4] is not None else None,
                "actual_qty": float(row[5]) if row[5] is not None else None,
                "accuracy_pct": float(row[6]) if row[6] is not None else None,
                "wape": float(row[7]) if row[7] is not None else None,
                "bias": float(row[8]) if row[8] is not None else None,
                "horizon_months": row[9],
                "cluster_id": row[10],
            }
        )
    return {"item_id": item_id, "loc": loc, "models": models}
