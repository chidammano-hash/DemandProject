"""AI Champion forward adjuster endpoints (/ai-champion/*).

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md

Interactive, single-DFU adjuster — there is no batch pipeline. The Item Analysis
tab previews an LLM adjustment of the promoted champion forecast for one DFU
(POST /adjust), then optionally persists it (POST /save). GET /forecast reads any
previously-saved adjustment for a DFU.
"""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from api.auth import require_api_key
from api.core import get_conn
from common.ai.champion_adjust_service import (
    NoChampionForecast,
    UnknownProvider,
    adjust_dfu,
    save_adjustment,
)
from common.core.sql_helpers import to_float

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-champion", tags=["ai-champion"])

_CACHE_SHORT = "public, max-age=60"


# ────────────────────────────────────────────────────────────────────────────
# 1. GET /ai-champion/forecast — saved adjustment for one DFU
# ────────────────────────────────────────────────────────────────────────────
@router.get("/forecast")
def ai_champion_forecast(
    item_id: str = Query(description="Item id (required)"),
    loc: str = Query(default="", description="Location id; empty = all locations for the item"),
):
    """Latest saved ai_champion adjustment for a DFU (champion vs AI, per month).

    Scoped to the most recently saved plan_version for the DFU. Fully
    parameterized (no SQL-string interpolation) — psycopg3 + CLAUDE.md rule.
    """
    params = (item_id, loc, loc, item_id, loc, loc)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT item_id, loc, forecast_month, horizon_months, champion_qty,
                      ai_qty, recommendation_code, pct_change, confidence, rationale
               FROM fact_ai_champion_forecast f
               WHERE f.item_id = %s AND (%s = '' OR f.loc = %s)
                 AND f.plan_version = (
                       SELECT plan_version FROM fact_ai_champion_forecast
                       WHERE item_id = %s AND (%s = '' OR loc = %s)
                       ORDER BY generated_at DESC LIMIT 1)
               ORDER BY f.loc, f.forecast_month""",
            params,
        )
        rows = cur.fetchall()

    return JSONResponse(
        content={
            "total": len(rows),
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
# 2. POST /ai-champion/adjust — preview an LLM adjustment for one DFU (no write)
# ────────────────────────────────────────────────────────────────────────────
class AdjustRequest(BaseModel):
    item_id: str
    loc: str
    provider: str | None = Field(default=None, description="ollama|google|anthropic|openai (default: config)")


@router.post("/adjust", dependencies=[Depends(require_api_key)])
def ai_champion_adjust(req: AdjustRequest):
    """Call the configured LLM once for one DFU and return a preview (no DB write)."""
    try:
        preview = adjust_dfu(req.item_id, req.loc, provider=req.provider)
    except NoChampionForecast:
        raise HTTPException(status_code=404, detail="No champion forecast for this DFU") from None
    except UnknownProvider:
        raise HTTPException(status_code=400, detail="Unknown provider") from None
    except psycopg.Error:
        log.exception("AI Champion adjust DB error")
        raise HTTPException(status_code=500, detail="Adjust failed") from None
    return JSONResponse(content=preview.to_dict())


# ────────────────────────────────────────────────────────────────────────────
# 3. POST /ai-champion/save — persist a previewed adjustment for one DFU
# ────────────────────────────────────────────────────────────────────────────
class RecommendationPayload(BaseModel):
    recommendation_code: str
    pct_change: float | None = None
    proposed_qty: list[float] | None = None
    apply_horizon_months: int = 3
    confidence: float = 0.0
    rationale: str
    evidence_keys: list[str] = Field(default_factory=list)


class SaveRequest(BaseModel):
    item_id: str
    loc: str
    provider: str | None = None
    recommendation: RecommendationPayload


@router.post("/save", dependencies=[Depends(require_api_key)])
def ai_champion_save(req: SaveRequest):
    """Persist a previewed adjustment. Quantities are re-derived server-side."""
    try:
        result = save_adjustment(
            req.item_id, req.loc, provider=req.provider,
            recommendation=req.recommendation.model_dump(),
        )
    except NoChampionForecast:
        raise HTTPException(status_code=404, detail="No champion forecast for this DFU") from None
    except UnknownProvider:
        raise HTTPException(status_code=400, detail="Unknown provider") from None
    except ValidationError:
        raise HTTPException(status_code=400, detail="Invalid recommendation payload") from None
    except psycopg.Error:
        log.exception("AI Champion save DB error")
        raise HTTPException(status_code=500, detail="Save failed") from None
    return JSONResponse(content=result)
