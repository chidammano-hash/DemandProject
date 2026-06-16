"""AI Champion forward adjuster endpoints.

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md

Read the forward-only AI Champion forecast (model_id='ai_champion') and trigger
a background generation job. The job reads the promoted champion production
forecast, applies a per-DFU LLM adjustment (Ollama default, Opus 4.7 opt-in),
and writes the result to fact_ai_champion_forecast.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.core import get_conn
from common.core.sql_helpers import to_float

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-champion", tags=["ai-champion"])

_CACHE_SHORT = "public, max-age=60"


# ────────────────────────────────────────────────────────────────────────────
# 1. GET /ai-champion/latest — latest run + adjustment summary
# ────────────────────────────────────────────────────────────────────────────
@router.get("/latest")
def ai_champion_latest():
    """Latest AI Champion run metadata plus a recommendation-code rollup."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT run_id::text, plan_version, provider, ai_model, status,
                      n_dfus, n_adjusted, est_cost_usd, started_at, completed_at
               FROM ai_champion_run
               ORDER BY started_at DESC
               LIMIT 1"""
        )
        run = cur.fetchone()
        if run is None:
            return JSONResponse(content={"run": None, "by_recommendation": []},
                                headers={"Cache-Control": _CACHE_SHORT})

        run_id = run[0]
        cur.execute(
            """SELECT recommendation_code, COUNT(DISTINCT (item_id, loc))::bigint
               FROM fact_ai_champion_forecast
               WHERE run_id = %s::uuid
               GROUP BY recommendation_code
               ORDER BY 2 DESC""",
            (run_id,),
        )
        by_rec = [{"recommendation_code": r[0], "dfus": int(r[1])} for r in cur.fetchall()]

    run_obj = {
        "run_id": run[0], "plan_version": run[1], "provider": run[2], "ai_model": run[3],
        "status": run[4], "n_dfus": run[5], "n_adjusted": run[6],
        "est_cost_usd": to_float(run[7], decimals=4),
        "started_at": run[8].isoformat() if run[8] else None,
        "completed_at": run[9].isoformat() if run[9] else None,
    }
    return JSONResponse(content={"run": run_obj, "by_recommendation": by_rec},
                        headers={"Cache-Control": _CACHE_SHORT})


# ────────────────────────────────────────────────────────────────────────────
# 2. GET /ai-champion/forecast — paginated ai_champion vs champion rows
# ────────────────────────────────────────────────────────────────────────────
@router.get("/forecast")
def ai_champion_forecast(
    item_id: str = Query(default="", description="Filter by item_id"),
    adjusted_only: bool = Query(default=False, description="Only DFUs the AI changed"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Per-DFU-month AI Champion forecast vs the champion baseline (latest run).

    Fully parameterized: optional filters are folded into the WHERE with %s
    placeholders (no SQL-string interpolation) — psycopg3 + CLAUDE.md rule.
    """
    # (item_id filter: empty string = no filter; adjusted_only flag).
    filter_params = (item_id, item_id, adjusted_only)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*)
               FROM fact_ai_champion_forecast f
               WHERE f.run_id = (SELECT run_id FROM ai_champion_run ORDER BY started_at DESC LIMIT 1)
                 AND (%s = '' OR f.item_id = %s)
                 AND (NOT %s OR f.recommendation_code NOT IN ('KEEP', 'OVERRIDE_TO_BASELINE'))""",
            filter_params,
        )
        total = cur.fetchone()[0]
        cur.execute(
            """SELECT f.item_id, f.loc, f.forecast_month, f.horizon_months,
                      f.champion_qty, f.ai_qty, f.recommendation_code, f.pct_change,
                      f.confidence, f.rationale
               FROM fact_ai_champion_forecast f
               WHERE f.run_id = (SELECT run_id FROM ai_champion_run ORDER BY started_at DESC LIMIT 1)
                 AND (%s = '' OR f.item_id = %s)
                 AND (NOT %s OR f.recommendation_code NOT IN ('KEEP', 'OVERRIDE_TO_BASELINE'))
               ORDER BY f.item_id, f.loc, f.forecast_month
               LIMIT %s OFFSET %s""",
            (*filter_params, limit, offset),
        )
        rows = cur.fetchall()

    return JSONResponse(
        content={
            "total": int(total),
            "rows": [
                {
                    "item_id": r[0], "loc": r[1],
                    "forecast_month": r[2].isoformat() if r[2] else None,
                    "horizon_months": r[3],
                    "champion_qty": to_float(r[4], decimals=2),
                    "ai_qty": to_float(r[5], decimals=2),
                    "recommendation_code": r[6],
                    "pct_change": to_float(r[7], decimals=2),
                    "confidence": to_float(r[8], decimals=3),
                    "rationale": r[9],
                }
                for r in rows
            ],
        },
        headers={"Cache-Control": _CACHE_SHORT},
    )


# ────────────────────────────────────────────────────────────────────────────
# 3. POST /ai-champion/generate — trigger a forward generation job
# ────────────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    provider: str | None = Field(default=None, description="Override provider (ollama|anthropic|...)")
    limit_dfus: int | None = Field(default=None, ge=1, description="Adjust only the first N DFUs")


@router.post("/generate", dependencies=[Depends(require_api_key)])
def ai_champion_generate(req: GenerateRequest | None = None):
    """Submit a background job to generate the AI Champion forecast.

    Returns 202 with a job_id pollable via GET /jobs/{job_id}.
    """
    from common.services.job_registry import JobManager

    params = {
        "provider": req.provider if req else None,
        "limit_dfus": req.limit_dfus if req else None,
    }
    job_id = JobManager().submit_job(
        "generate_ai_champion",
        params,
        label="Generate AI Champion Forecast",
        triggered_by="api",
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})
