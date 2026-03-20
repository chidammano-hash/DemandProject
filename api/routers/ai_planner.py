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
    status: str  # open | acknowledged | resolved | snoozed
    action_taken: str | None = None  # optional: what the planner intends to do


class SnoozeRequest(BaseModel):
    days: int = 1  # 1 | 3 | 7 | custom


class AutoAcceptRequest(BaseModel):
    min_severity: str = "critical"  # critical | high | medium | low
    insight_types: list[str] = []   # empty = all types
    dry_run: bool = False


# ---------------------------------------------------------------------------
# POST /ai-planner/analyze
# ---------------------------------------------------------------------------

@router.post("/ai-planner/analyze", status_code=201)
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
        raise HTTPException(status_code=500, detail="Analysis failed. Check server logs for details.") from exc

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
    brand: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
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
    if brand:
        conditions.append(
            "EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = ai.item_no AND di.brand_name = ANY(%s))"
        )
        params.append(brand.split(","))
    if category:
        conditions.append(
            "EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = ai.item_no AND di.class = ANY(%s))"
        )
        params.append(category.split(","))
    if market:
        conditions.append(
            "EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = ai.loc AND dl.market = ANY(%s))"
        )
        params.append(market.split(","))
    if channel:
        conditions.append(
            "EXISTS (SELECT 1 FROM dim_customer dc "
            "JOIN fact_sales_monthly fsm ON fsm.cust_grp = dc.customer_group "
            "WHERE fsm.dmdunit = ai.item_no AND fsm.loc = ai.loc AND dc.channel = ANY(%s))"
        )
        params.append(channel.split(","))

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

    return {"total": total, "page": page, "page_size": page_size, "insights": rows}


# ---------------------------------------------------------------------------
# PUT /ai-planner/insights/{insight_id}/status
# ---------------------------------------------------------------------------

@router.post("/ai-planner/insights/{insight_id}/status")
async def update_insight_status(insight_id: int, body: StatusUpdateRequest, request: Request):
    """Acknowledge or resolve an insight, and record the outcome for feedback tracking."""
    require_api_key(request)
    valid = {"open", "acknowledged", "resolved"}
    if body.status not in valid:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(valid)}")

    ts_col = ""
    if body.status == "acknowledged":
        ts_col = ", acknowledged_at = NOW()"
    elif body.status == "resolved":
        ts_col = ", resolved_at = NOW()"

    update_sql = (
        f"UPDATE ai_insights SET status = %s, updated_at = NOW() {ts_col} "
        "WHERE insight_id = %s "
        "RETURNING insight_id, status, insight_type, item_no, loc, abc_vol, "
        "          financial_impact_estimate, dos, total_lt_days, "
        "          champion_wape, forecast_bias_pct"
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(update_sql, (body.status, insight_id))
            row = cur.fetchone()

            if row is None:
                conn.commit()
                raise HTTPException(status_code=404, detail="Insight not found")

            # Write outcome record when planner makes a decision
            if body.status in ("acknowledged", "resolved"):
                decision = "accepted" if body.status == "acknowledged" else "resolved"
                (_, _, ins_type, item_no, loc, abc_vol,
                 fin_impact, dos, lt, wape, bias) = row
                outcome_sql = """
                    INSERT INTO ai_recommendation_outcomes
                        (insight_id, insight_type, item_no, loc, abc_vol,
                         planner_decision, financial_impact_est,
                         metric_before_dos, metric_before_wape, metric_before_bias_pct,
                         lead_time_days, action_taken, executed_at, outcome_check_due_at)
                    VALUES (%s,%s,%s,%s,%s, %s,%s, %s,%s,%s, %s,%s,NOW(),NOW() + INTERVAL '30 days')
                    ON CONFLICT DO NOTHING
                """
                try:
                    cur.execute(outcome_sql, (
                        insight_id, ins_type, item_no, loc, abc_vol,
                        decision, fin_impact,
                        dos, wape, bias,
                        lt, body.action_taken,
                    ))
                except Exception:
                    log.warning("Failed to write outcome record for insight %s", insight_id)

            conn.commit()

    return {"insight_id": row[0], "status": row[1]}


# ---------------------------------------------------------------------------
# PUT /ai-planner/insights/{insight_id}/snooze  (PL-012)
# ---------------------------------------------------------------------------

@router.post("/ai-planner/insights/{insight_id}/snooze")
async def snooze_insight(insight_id: int, body: SnoozeRequest, request: Request):
    """Snooze an insight for N days — hides it from the default open queue."""
    require_api_key(request)
    if body.days < 1 or body.days > 365:
        raise HTTPException(status_code=422, detail="days must be between 1 and 365")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ai_insights
                   SET status = 'snoozed',
                       snoozed_until = NOW() + INTERVAL '1 day' * %s,
                       updated_at = NOW()
                 WHERE insight_id = %s
                RETURNING insight_id, status, snoozed_until
                """,
                (body.days, insight_id),
            )
            row = cur.fetchone()
            if row is None:
                conn.commit()
                raise HTTPException(status_code=404, detail="Insight not found")
            conn.commit()

    return {
        "insight_id": row[0],
        "status": row[1],
        "snoozed_until": row[2].isoformat() if row[2] else None,
    }


# ---------------------------------------------------------------------------
# POST /ai-planner/auto-accept
# ---------------------------------------------------------------------------

SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
VALID_INSIGHT_TYPES = {
    "stockout_risk", "excess_inventory", "forecast_bias", "policy_gap", "champion_degradation"
}


@router.post("/ai-planner/auto-accept")
async def auto_accept_insights(body: AutoAcceptRequest, request: Request):
    """Bulk-accept open insights matching severity/type rules.

    With dry_run=true returns matching count without writing.
    Writes ai_recommendation_outcomes with planner_decision='auto_accepted'.
    """
    require_api_key(request)

    min_rank = SEVERITY_RANK.get(body.min_severity, 0)
    qualifying = [s for s, r in SEVERITY_RANK.items() if r <= min_rank]

    conditions = ["status = 'open'", "severity = ANY(%s)"]
    params: list = [qualifying]

    requested_types = [t for t in body.insight_types if t in VALID_INSIGHT_TYPES]
    if requested_types:
        conditions.append("insight_type = ANY(%s)")
        params.append(requested_types)

    where = "WHERE " + " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT insight_id, insight_type, item_no, loc, abc_vol, "
                f"financial_impact_estimate, dos, total_lt_days, champion_wape, forecast_bias_pct "
                f"FROM ai_insights {where} "
                "ORDER BY CASE severity "
                "  WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, "
                "COALESCE(financial_impact_estimate, 0) DESC",
                params,
            )
            rows = cur.fetchall()

            if body.dry_run or not rows:
                conn.commit()
                return {"accepted": len(rows), "dry_run": True, "insight_ids": [r[0] for r in rows]}

            ids = [r[0] for r in rows]
            cur.execute(
                "UPDATE ai_insights "
                "SET status = 'acknowledged', acknowledged_at = NOW(), updated_at = NOW() "
                "WHERE insight_id = ANY(%s)",
                (ids,),
            )

            for row in rows:
                (iid, itype, item_no, loc, abc_vol, fin_impact, dos, lt, wape, bias) = row
                try:
                    cur.execute(
                        """
                        INSERT INTO ai_recommendation_outcomes
                            (insight_id, insight_type, item_no, loc, abc_vol,
                             planner_decision, financial_impact_est,
                             metric_before_dos, metric_before_wape, metric_before_bias_pct,
                             lead_time_days, executed_at, outcome_check_due_at)
                        VALUES (%s,%s,%s,%s,%s,'auto_accepted',%s,%s,%s,%s,%s,
                                NOW(), NOW() + INTERVAL '30 days')
                        ON CONFLICT DO NOTHING
                        """,
                        (iid, itype, item_no, loc, abc_vol, fin_impact, dos, wape, bias, lt),
                    )
                except Exception:
                    log.warning("Failed to write outcome for insight %s", iid)

            conn.commit()

    return {"accepted": len(ids), "dry_run": False, "insight_ids": ids}


# ---------------------------------------------------------------------------
# GET /ai-planner/metrics
# ---------------------------------------------------------------------------

@router.get("/ai-planner/metrics")
async def get_ai_metrics(
    days: int = Query(7, ge=1, le=90),
):
    """Return AI call log aggregates: cost estimate, tokens, latency, error rate.

    Returns per-model and per-tool aggregates for the last N days.
    Used for observability and cost monitoring.
    """
    sql = """
        SELECT
            provider,
            model,
            COUNT(*) FILTER (WHERE tool_name IS NULL)    AS llm_turns,
            COUNT(*) FILTER (WHERE tool_name IS NOT NULL) AS tool_calls,
            SUM(total_tokens) FILTER (WHERE tool_name IS NULL) AS total_tokens,
            ROUND(AVG(latency_ms) FILTER (WHERE tool_name IS NULL))::INTEGER
                                                          AS avg_llm_latency_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (
                ORDER BY latency_ms
            ) FILTER (WHERE tool_name IS NULL)            AS p95_llm_latency_ms,
            COUNT(*) FILTER (WHERE tool_success = false)  AS tool_errors,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE tool_success = false) /
                NULLIF(COUNT(*) FILTER (WHERE tool_name IS NOT NULL), 0), 2
            )                                             AS error_rate_pct
        FROM ai_call_log
        WHERE created_at >= NOW() - INTERVAL '1 day' * %s
        GROUP BY provider, model
        ORDER BY total_tokens DESC NULLS LAST
    """

    tool_sql = """
        SELECT
            tool_name,
            COUNT(*)                                      AS total_calls,
            COUNT(*) FILTER (WHERE tool_success = false)  AS failures,
            ROUND(AVG(latency_ms))::INTEGER               AS avg_latency_ms
        FROM ai_call_log
        WHERE tool_name IS NOT NULL
          AND created_at >= NOW() - INTERVAL '1 day' * %s
        GROUP BY tool_name
        ORDER BY total_calls DESC
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (days,))
            model_cols = [d[0] for d in cur.description]
            by_model = [dict(zip(model_cols, r)) for r in cur.fetchall()]

            cur.execute(tool_sql, (days,))
            tool_cols = [d[0] for d in cur.description]
            by_tool = [dict(zip(tool_cols, r)) for r in cur.fetchall()]

    return {"days": days, "by_model": by_model, "by_tool": by_tool}


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
