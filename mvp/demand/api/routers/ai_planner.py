"""AI Planner API router — IPAIfeature1.

5 endpoints under /ai-planner/*:
  POST   /ai-planner/analyze           DFU-level analysis (synchronous, ~5-10s)
  POST   /ai-planner/portfolio-scan    Trigger portfolio scan (202, background)
  GET    /ai-planner/insights          Paginated insight list with filters
  PUT    /ai-planner/insights/{id}/status   Acknowledge / resolve an insight
  GET    /ai-planner/memos             List planning memos

Uses get_conn() directly — NOT Depends(_get_pool) — to avoid 422 issues when
FastAPI inspects mock signatures in tests.
"""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn

log = logging.getLogger(__name__)
router = APIRouter(tags=["ai-planner"])

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai_planner")

# ---------------------------------------------------------------------------
# Config loader (lazy, module-level singleton)
# ---------------------------------------------------------------------------
_AI_CONFIG: dict | None = None


def _get_config() -> dict:
    global _AI_CONFIG
    if _AI_CONFIG is None:
        try:
            with open("config/ai_planner_config.yaml") as f:
                _AI_CONFIG = yaml.safe_load(f)
        except FileNotFoundError:
            _AI_CONFIG = {
                "model": "claude-opus-4-6",
                "max_tokens": 4096,
                "portfolio_scan_limit": 100,
                "forecast_lookback_months": 6,
            }
    return _AI_CONFIG


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    item_no: str
    loc: str


class StatusUpdateRequest(BaseModel):
    status: str  # open | acknowledged | resolved


# ---------------------------------------------------------------------------
# POST /ai-planner/analyze
# ---------------------------------------------------------------------------

@router.post("/ai-planner/analyze")
async def analyze_dfu(body: AnalyzeRequest, request: Request):
    """Run AI analysis for a single DFU (synchronous, ~5-15s)."""
    require_api_key(request)
    from common.ai_planner import AIPlannerAgent
    from api.core import _get_pool

    pool = _get_pool()
    scan_run_id = str(uuid.uuid4())
    agent = AIPlannerAgent(pool, _get_config())

    try:
        insights = agent.run_dfu_analysis(body.item_no, body.loc, scan_run_id)
    except Exception as exc:
        log.exception("DFU analysis failed for %s@%s", body.item_no, body.loc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "scan_run_id": scan_run_id,
        "item_no": body.item_no,
        "loc": body.loc,
        "total_insights": len(insights),
        "insights": insights,
    }


# ---------------------------------------------------------------------------
# POST /ai-planner/portfolio-scan
# ---------------------------------------------------------------------------

@router.post("/ai-planner/portfolio-scan", status_code=202)
async def trigger_portfolio_scan(request: Request):
    """Trigger an async portfolio scan.  Returns 202 immediately."""
    require_api_key(request)
    from common.ai_planner import AIPlannerAgent
    from api.core import _get_pool

    pool = _get_pool()
    scan_run_id = str(uuid.uuid4())
    config = _get_config()

    def _run():
        agent = AIPlannerAgent(pool, config)
        try:
            agent.run_portfolio_scan(scan_run_id)
        except Exception:
            log.exception("Portfolio scan failed  scan_run_id=%s", scan_run_id)

    _executor.submit(_run)
    return {"scan_run_id": scan_run_id, "status": "accepted"}


# ---------------------------------------------------------------------------
# GET /ai-planner/insights
# ---------------------------------------------------------------------------

@router.get("/ai-planner/insights")
async def get_insights(
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None, alias="status"),
    insight_type: Optional[str] = Query(None),
    item_no: Optional[str] = Query(None),
    loc: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Return paginated AI insights with optional filters."""
    valid_severities = {"critical", "high", "medium", "low"}
    valid_statuses   = {"open", "acknowledged", "resolved"}

    conditions: list[str] = []
    params: list = []

    if severity and severity in valid_severities:
        conditions.append("severity = %s")
        params.append(severity)
    if status and status in valid_statuses:
        conditions.append("status = %s")
        params.append(status)
    if insight_type:
        conditions.append("insight_type = %s")
        params.append(insight_type)
    if item_no:
        conditions.append("item_no = %s")
        params.append(item_no)
    if loc:
        conditions.append("loc = %s")
        params.append(loc)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * page_size

    count_sql   = f"SELECT COUNT(*) FROM ai_insights {where}"
    select_sql  = (
        f"SELECT * FROM ai_insights {where} "
        "ORDER BY CASE severity "
        "  WHEN 'critical' THEN 0 WHEN 'high' THEN 1 "
        "  WHEN 'medium' THEN 2 ELSE 3 END, "
        "COALESCE(financial_impact_estimate, 0) DESC, "
        "created_at DESC "
        f"LIMIT {page_size} OFFSET {offset}"
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0]

            cur.execute(select_sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    return {"total": total, "page": page, "page_size": page_size, "rows": rows}


# ---------------------------------------------------------------------------
# PUT /ai-planner/insights/{insight_id}/status
# ---------------------------------------------------------------------------

@router.put("/ai-planner/insights/{insight_id}/status")
async def update_insight_status(insight_id: int, body: StatusUpdateRequest, request: Request):
    """Acknowledge or resolve an insight."""
    require_api_key(request)
    valid = {"open", "acknowledged", "resolved"}
    if body.status not in valid:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(valid)}")

    ts_col = ""
    if body.status == "acknowledged":
        ts_col = ", acknowledged_at = NOW()"
    elif body.status == "resolved":
        ts_col = ", resolved_at = NOW()"

    sql = (
        f"UPDATE ai_insights SET status = %s, updated_at = NOW() {ts_col} "
        "WHERE insight_id = %s RETURNING insight_id, status"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (body.status, insight_id))
            row = cur.fetchone()
            conn.commit()

    if row is None:
        raise HTTPException(status_code=404, detail="Insight not found")

    return {"insight_id": row[0], "status": row[1]}


# ---------------------------------------------------------------------------
# GET /ai-planner/memos
# ---------------------------------------------------------------------------

@router.get("/ai-planner/memos")
async def get_memos(
    scope: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    """Return planning memos (latest first)."""
    valid_scopes = {"portfolio", "dfu"}
    conditions: list[str] = []
    params: list = []

    if scope and scope in valid_scopes:
        conditions.append("scope = %s")
        params.append(scope)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = (
        f"SELECT * FROM ai_planning_memos {where} "
        f"ORDER BY created_at DESC LIMIT {limit}"
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    return {"memos": rows}
