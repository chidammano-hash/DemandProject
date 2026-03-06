"""AI Planning Agent — core module.

IPAIfeature1: proactive exception work-queue that scans the demand+inventory
portfolio, traces causal chains across forecast → inventory → EOQ → policy
layers, and writes structured insights to the database.

NOT a chatbot.  This is a scheduled batch agent that uses a configurable LLM
(OpenAI or Anthropic) with tool_use / function-calling to read real data,
reason about it, and emit ranked, actionable insights.

Provider is set via config:
    provider: "openai"      # uses OPENAI_API_KEY  (default)
    provider: "anthropic"   # uses ANTHROPIC_API_KEY

Usage (via scripts/generate_ai_insights.py):
    agent = AIPlannerAgent(pool, config)
    summary = agent.run_portfolio_scan(scan_run_id)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date, datetime
from typing import Any, Annotated, Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit-breaker constants
# ---------------------------------------------------------------------------
MAX_TURNS = 40
TOKEN_BUDGET = 100_000  # cumulative tokens per DFU / portfolio scan


# ---------------------------------------------------------------------------
# Pydantic validation for create_insight LLM inputs
# ---------------------------------------------------------------------------

class CreateInsightInput(BaseModel):
    insight_type: Literal[
        "stockout_risk", "excess_inventory", "forecast_bias",
        "policy_gap", "champion_degradation"
    ]
    severity: Literal["critical", "high", "medium", "low"]
    item_no: str
    loc: str
    summary: str = Field(min_length=20, max_length=300)
    recommendation: str = Field(min_length=30, max_length=600)
    reasoning: str | None = None
    financial_impact_estimate: Annotated[float | None, Field(ge=0, le=10_000_000)] = None
    dos: float | None = None
    total_lt_days: int | None = None
    champion_wape: float | None = None
    forecast_bias_pct: float | None = None
    current_policy_id: str | None = None
    eoq_effective: float | None = None
    abc_vol: str | None = None
    cluster_assignment: str | None = None

    @field_validator("summary")
    @classmethod
    def summary_must_contain_metrics(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("summary must contain at least one metric value (number)")
        return v


# ---------------------------------------------------------------------------
# Tool implementations — all direct psycopg queries, no HTTP round-trips
# ---------------------------------------------------------------------------

def get_dfu_full_context(pool, item_no: str, loc: str) -> dict:
    """Return a single-row snapshot of all planning layers for a DFU."""
    sql = """
        SELECT
            d.dmdunit AS item_no, d.loc,
            d.abc_vol, d.variability_class, d.cluster_assignment,
            d.seasonality_profile, d.is_yearly_seasonal,
            -- latest inventory
            i.current_dos,
            i.total_lt_days,
            i.avg_on_hand,
            i.avg_daily_sales,
            -- EOQ
            e.eoq_effective,
            e.annual_demand,
            e.total_annual_cost,
            e.eoq_months_supply,
            e.cycle_stock_value,
            -- policy
            p.policy_id          AS current_policy_id,
            p.policy_type,
            p.review_period_days,
            p.service_level,
            p.use_safety_stock,
            -- champion WAPE (latest available)
            (
                SELECT ROUND(100.0 * SUM(ABS(basefcst_pref - tothist_dmd)) /
                             NULLIF(ABS(SUM(tothist_dmd)), 0), 2)
                FROM fact_external_forecast_monthly
                WHERE dmdunit = d.dmdunit AND loc = d.loc
                  AND model_id = 'champion'
                  AND lag = 0
                  AND startdate >= date_trunc('month', NOW() - INTERVAL '6 months')
            ) AS champion_wape
        FROM dim_dfu d
        LEFT JOIN LATERAL (
            SELECT ROUND((eom_qty_on_hand / NULLIF(avg_daily_sls, 0))::NUMERIC, 1) AS current_dos,
                   latest_lead_time_days AS total_lt_days,
                   avg_qty_on_hand AS avg_on_hand,
                   avg_daily_sls AS avg_daily_sales
            FROM agg_inventory_monthly
            WHERE item_no = d.dmdunit AND loc = d.loc
            ORDER BY month_start DESC
            LIMIT 1
        ) i ON TRUE
        LEFT JOIN LATERAL (
            SELECT effective_eoq AS eoq_effective,
                   annual_demand,
                   total_annual_cost,
                   ROUND((effective_eoq / NULLIF(annual_demand / 12.0, 0))::NUMERIC, 2) AS eoq_months_supply,
                   eoq_cycle_stock AS cycle_stock_value
            FROM fact_eoq_targets
            WHERE item_no = d.dmdunit AND loc = d.loc
            ORDER BY computed_at DESC
            LIMIT 1
        ) e ON TRUE
        LEFT JOIN LATERAL (
            SELECT rp.policy_id, rp.policy_type,
                   rp.review_cycle_days AS review_period_days,
                   rp.service_level,
                   rp.use_safety_stock
            FROM fact_dfu_policy_assignment pa
            JOIN dim_replenishment_policy rp USING (policy_id)
            WHERE pa.item_no = d.dmdunit AND pa.loc = d.loc
            ORDER BY pa.effective_date DESC
            LIMIT 1
        ) p ON TRUE
        WHERE d.dmdunit = %s AND d.loc = %s
        LIMIT 1
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc))
            row = cur.fetchone()
            if row is None:
                return {"error": f"DFU {item_no}@{loc} not found"}
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def get_forecast_performance(pool, item_no: str, loc: str, months: int = 6) -> list[dict]:
    """Return per-month champion forecast accuracy for a DFU."""
    sql = """
        SELECT
            startdate,
            basefcst_pref AS forecast_qty,
            tothist_dmd AS actual_qty,
            ROUND(100.0 * (basefcst_pref - tothist_dmd) / NULLIF(tothist_dmd, 0), 2) AS bias_pct,
            ROUND(100.0 * ABS(basefcst_pref - tothist_dmd) / NULLIF(ABS(tothist_dmd), 0), 2) AS abs_err_pct
        FROM fact_external_forecast_monthly
        WHERE dmdunit = %s AND loc = %s
          AND model_id = 'champion'
          AND lag = 0
          AND startdate >= date_trunc('month', NOW() - INTERVAL '1 month' * %s)
        ORDER BY startdate
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc, months))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def get_portfolio_exceptions(pool, limit: int = 50) -> list[dict]:
    """Return top exception DFUs ranked by severity signals."""
    sql = """
        WITH latest_inv AS (
            SELECT DISTINCT ON (item_no, loc)
                item_no, loc,
                ROUND((eom_qty_on_hand / NULLIF(avg_daily_sls, 0))::NUMERIC, 1) AS avg_dos,
                latest_lead_time_days AS total_lt_days
            FROM agg_inventory_monthly
            ORDER BY item_no, loc, month_start DESC
        ),
        champion_wape AS (
            SELECT dmdunit, loc,
                ROUND(100.0 * SUM(ABS(basefcst_pref - tothist_dmd)) /
                      NULLIF(ABS(SUM(tothist_dmd)), 0), 2) AS wape
            FROM fact_external_forecast_monthly
            WHERE model_id = 'champion' AND lag = 0
              AND startdate >= date_trunc('month', NOW() - INTERVAL '3 months')
            GROUP BY dmdunit, loc
        )
        SELECT
            d.dmdunit AS item_no, d.loc, d.abc_vol, d.variability_class, d.cluster_assignment,
            i.avg_dos,
            i.total_lt_days,
            w.wape                                                          AS champion_wape,
            CASE
                WHEN i.avg_dos < i.total_lt_days * 1.5                     THEN true
                ELSE false
            END AS stockout_risk,
            CASE
                WHEN i.avg_dos > 180                                        THEN true
                ELSE false
            END AS excess_flag,
            CASE
                WHEN w.wape > 50                                            THEN true
                ELSE false
            END AS high_wape_flag
        FROM dim_dfu d
        JOIN latest_inv i ON d.dmdunit = i.item_no AND d.loc = i.loc
        LEFT JOIN champion_wape w ON d.dmdunit = w.dmdunit AND d.loc = w.loc
        WHERE d.abc_vol IN ('A','B')
           OR w.wape > 35
           OR i.avg_dos < i.total_lt_days * 1.5
           OR i.avg_dos > 180
        ORDER BY
            CASE WHEN i.avg_dos < i.total_lt_days * 1.5 THEN 0
                 WHEN i.avg_dos > 180 THEN 1
                 WHEN w.wape > 50 THEN 2
                 ELSE 3 END,
            d.abc_vol,
            w.wape DESC NULLS LAST
        LIMIT %s
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def compute_bias_trend(pool, item_no: str, loc: str) -> dict:
    """Compute rolling bias averages for a DFU."""
    sql = """
        WITH monthly AS (
            SELECT
                startdate,
                ROUND(100.0 * (basefcst_pref - tothist_dmd) / NULLIF(tothist_dmd, 0), 2) AS bias_pct
            FROM fact_external_forecast_monthly
            WHERE dmdunit = %s AND loc = %s
              AND model_id = 'champion' AND lag = 0
              AND tothist_dmd IS NOT NULL
            ORDER BY startdate DESC
            LIMIT 6
        )
        SELECT
            AVG(bias_pct)                                    AS bias_6m_avg,
            AVG(CASE WHEN rn <= 3 THEN bias_pct END)        AS bias_3m_avg,
            COUNT(*) FILTER (WHERE bias_pct > 20)           AS over_forecast_months,
            COUNT(*) FILTER (WHERE bias_pct < -20)          AS under_forecast_months
        FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY startdate DESC) AS rn FROM monthly) t
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc))
            row = cur.fetchone()
            if row is None:
                return {}
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def get_inventory_trend(pool, item_no: str, loc: str, months: int = 6) -> list[dict]:
    """Return monthly inventory trend for a DFU."""
    sql = """
        SELECT month_start,
               ROUND((eom_qty_on_hand / NULLIF(avg_daily_sls, 0))::NUMERIC, 1) AS avg_dos,
               avg_qty_on_hand AS avg_on_hand,
               avg_daily_sls AS avg_daily_sales,
               latest_lead_time_days AS total_lt_days
        FROM agg_inventory_monthly
        WHERE item_no = %s AND loc = %s
        ORDER BY month_start DESC
        LIMIT %s
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc, months))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def get_eoq_context(pool, item_no: str, loc: str) -> dict:
    """Return EOQ health context for a DFU."""
    sql = """
        SELECT
            eoq AS eoq_qty,
            effective_eoq AS eoq_effective,
            annual_demand,
            ordering_cost AS ordering_cost_per_unit,
            annual_holding_cost AS holding_cost_annual,
            total_annual_cost,
            ROUND((effective_eoq / NULLIF(annual_demand / 12.0, 0))::NUMERIC, 2) AS eoq_months_supply,
            eoq_cycle_stock AS cycle_stock_value,
            computed_at
        FROM fact_eoq_targets
        WHERE item_no = %s AND loc = %s
        ORDER BY computed_at DESC
        LIMIT 1
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc))
            row = cur.fetchone()
            if row is None:
                return {"error": "No EOQ data"}
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def get_similar_dfus(pool, item_no: str, loc: str, limit: int = 5) -> list[dict]:
    """Return peer DFUs in the same cluster+ABC class."""
    sql = """
        SELECT
            d.dmdunit AS item_no, d.loc, d.abc_vol,
            ROUND((eom_on_hand / NULLIF(avg_sls, 0))::NUMERIC, 1) AS avg_dos,
            ROUND(100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) /
                  NULLIF(ABS(SUM(f.tothist_dmd)), 0), 2) AS peer_wape
        FROM dim_dfu d
        JOIN dim_dfu ref ON ref.cluster_assignment = d.cluster_assignment
                         AND ref.abc_vol = d.abc_vol
                         AND ref.dmdunit = %s AND ref.loc = %s
        LEFT JOIN LATERAL (
            SELECT DISTINCT ON (item_no, loc)
                eom_qty_on_hand AS eom_on_hand,
                avg_daily_sls AS avg_sls
            FROM agg_inventory_monthly
            WHERE item_no = d.dmdunit AND loc = d.loc
            ORDER BY item_no, loc, month_start DESC
        ) i ON TRUE
        LEFT JOIN fact_external_forecast_monthly f
            ON f.dmdunit = d.dmdunit AND f.loc = d.loc
           AND f.model_id = 'champion' AND f.lag = 0
           AND f.startdate >= date_trunc('month', NOW() - INTERVAL '3 months')
        WHERE d.dmdunit <> %s OR d.loc <> %s
        GROUP BY d.dmdunit, d.loc, d.abc_vol, i.eom_on_hand, i.avg_sls
        LIMIT %s
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc, item_no, loc, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def check_stockout_history(pool, item_no: str, loc: str) -> dict:
    """Return stockout/excess history for a DFU over last 6 months."""
    sql = """
        SELECT
            COUNT(*) FILTER (WHERE is_stockout)  AS stockout_months,
            COUNT(*) FILTER (WHERE is_excess)    AS excess_months,
            ROUND(AVG(dos)::NUMERIC, 1)          AS avg_dos,
            COUNT(*)                             AS total_months
        FROM mv_inventory_forecast_monthly
        WHERE item_no = %s AND loc = %s
          AND month_start >= date_trunc('month', NOW() - INTERVAL '6 months')
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (item_no, loc))
            row = cur.fetchone()
            if row is None:
                return {}
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def get_portfolio_health_summary(pool) -> dict:
    """Return aggregated health metrics across the full portfolio."""
    sql = """
        SELECT
            COUNT(DISTINCT d.dmdunit || '|' || d.loc)       AS total_dfus,
            ROUND(AVG(i.avg_dos), 1)                        AS avg_dos,
            COUNT(*) FILTER (WHERE i.avg_dos < i.total_lt_days * 1.5) AS stockout_risk_count,
            COUNT(*) FILTER (WHERE i.avg_dos > 180)         AS excess_count,
            ROUND(
                100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) /
                NULLIF(ABS(SUM(f.tothist_dmd)), 0), 2
            )                                               AS portfolio_champion_wape,
            (SELECT COUNT(*) FROM ai_insights
             WHERE status = 'open')                         AS open_insights
        FROM dim_dfu d
        LEFT JOIN LATERAL (
            SELECT ROUND((eom_qty_on_hand / NULLIF(avg_daily_sls, 0))::NUMERIC, 1) AS avg_dos,
                   latest_lead_time_days AS total_lt_days
            FROM agg_inventory_monthly
            WHERE item_no = d.dmdunit AND loc = d.loc
            ORDER BY month_start DESC LIMIT 1
        ) i ON TRUE
        LEFT JOIN fact_external_forecast_monthly f
            ON f.dmdunit = d.dmdunit AND f.loc = d.loc
           AND f.model_id = 'champion' AND f.lag = 0
           AND f.startdate >= date_trunc('month', NOW() - INTERVAL '3 months')
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if row is None:
                return {}
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def log_ai_call(
    pool,
    scan_run_id: str,
    provider: str,
    model: str,
    turn_number: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: int | None = None,
    tool_name: str | None = None,
    tool_success: bool | None = None,
    error_type: str | None = None,
    dfu_key: str | None = None,
) -> None:
    """Write a row to ai_call_log (best-effort, silently ignores failures)."""
    sql = """
        INSERT INTO ai_call_log (
            scan_run_id, dfu_key, provider, model, turn_number,
            prompt_tokens, completion_tokens, total_tokens, latency_ms,
            tool_name, tool_success, error_type
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    scan_run_id, dfu_key, provider, model, turn_number,
                    prompt_tokens, completion_tokens, total_tokens, latency_ms,
                    tool_name, tool_success, error_type,
                ))
                conn.commit()
    except Exception as exc:
        log.debug("ai_call_log insert failed (non-fatal): %s", exc)


def create_insight(
    pool,
    insight_type: str,
    severity: str,
    item_no: str,
    loc: str,
    summary: str,
    recommendation: str,
    reasoning: str | None = None,
    financial_impact_estimate: float | None = None,
    dos: float | None = None,
    total_lt_days: int | None = None,
    champion_wape: float | None = None,
    forecast_bias_pct: float | None = None,
    current_policy_id: str | None = None,
    eoq_effective: float | None = None,
    abc_vol: str | None = None,
    cluster_assignment: str | None = None,
    model_version: str | None = None,
    scan_run_id: str | None = None,
) -> int:
    """Validate with Pydantic then insert a new insight row; return new insight_id."""
    try:
        validated = CreateInsightInput(
            insight_type=insight_type,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            item_no=item_no,
            loc=loc,
            summary=summary,
            recommendation=recommendation,
            reasoning=reasoning,
            financial_impact_estimate=financial_impact_estimate,
            dos=dos,
            total_lt_days=total_lt_days,
            champion_wape=champion_wape,
            forecast_bias_pct=forecast_bias_pct,
            current_policy_id=current_policy_id,
            eoq_effective=eoq_effective,
            abc_vol=abc_vol,
            cluster_assignment=cluster_assignment,
        )
    except Exception as exc:
        log.warning("create_insight validation failed: %s — skipping INSERT", exc)
        return -1

    sql = """
        INSERT INTO ai_insights (
            insight_type, severity,
            item_no, loc, abc_vol, cluster_assignment,
            summary, recommendation, reasoning,
            financial_impact_estimate,
            dos, total_lt_days, champion_wape, forecast_bias_pct,
            current_policy_id, eoq_effective,
            model_version, scan_run_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING insight_id
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                validated.insight_type, validated.severity,
                validated.item_no, validated.loc, validated.abc_vol, validated.cluster_assignment,
                validated.summary, validated.recommendation, validated.reasoning,
                validated.financial_impact_estimate,
                validated.dos, validated.total_lt_days, validated.champion_wape,
                validated.forecast_bias_pct,
                validated.current_policy_id, validated.eoq_effective,
                model_version, scan_run_id,
            ))
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else -1


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic format — converted to OpenAI format on demand)
# ---------------------------------------------------------------------------

def _tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


_TOOL_DEFINITIONS = [
    {
        "name": "get_dfu_full_context",
        "description": (
            "Return a complete cross-layer snapshot for a single DFU: "
            "DFU attributes (ABC, cluster, seasonality), latest inventory "
            "(DOS, lead time, on-hand), EOQ targets, and current replenishment "
            "policy. Call this first when analysing any DFU."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string", "description": "Item number"},
                "loc":     {"type": "string", "description": "Location code"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "get_forecast_performance",
        "description": (
            "Return per-month champion forecast accuracy for a DFU over a "
            "trailing window.  Includes forecast_qty, actual_qty, bias_pct, "
            "abs_err_pct per month.  Use to spot systematic over/under-forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string"},
                "loc":     {"type": "string"},
                "months":  {"type": "integer", "description": "Look-back months (default 6)"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "get_portfolio_exceptions",
        "description": (
            "Return the top N DFUs with active exception signals across the "
            "portfolio: stockout risk, excess inventory, high WAPE.  Use as the "
            "entry point for a portfolio scan to decide which DFUs to analyse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max DFUs to return (default 50)"},
            },
        },
    },
    {
        "name": "compute_bias_trend",
        "description": (
            "Compute rolling bias averages (3-month and 6-month) and count of "
            "over/under-forecast months for a DFU.  Use to confirm whether "
            "a forecast bias is persistent or a one-off."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string"},
                "loc":     {"type": "string"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "get_inventory_trend",
        "description": (
            "Return monthly inventory trend for a DFU: DOS, on-hand, daily "
            "sales, lead time.  Use to understand whether inventory is declining "
            "toward stockout or accumulating excess."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string"},
                "loc":     {"type": "string"},
                "months":  {"type": "integer"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "get_eoq_context",
        "description": (
            "Return EOQ health context for a DFU: computed EOQ, effective EOQ "
            "(after MOQ/cap), annual demand, annual total cost, months of supply. "
            "Use to check whether current order quantity aligns with optimal EOQ."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string"},
                "loc":     {"type": "string"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "get_similar_dfus",
        "description": (
            "Return peer DFUs in the same cluster and ABC class.  Use to "
            "benchmark a DFU's DOS and WAPE against similar items."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string"},
                "loc":     {"type": "string"},
                "limit":   {"type": "integer"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "check_stockout_history",
        "description": (
            "Return stockout/excess event counts and average DOS over the last "
            "6 months from the inventory-forecast bridge view.  Use to quantify "
            "historical service-level failures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_no": {"type": "string"},
                "loc":     {"type": "string"},
            },
            "required": ["item_no", "loc"],
        },
    },
    {
        "name": "get_portfolio_health_summary",
        "description": (
            "Return aggregated health metrics for the entire portfolio: total "
            "DFUs, average DOS, stockout/excess counts, portfolio champion WAPE, "
            "and open insight count.  Use to ground portfolio-level memos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "create_insight",
        "description": (
            "Write a structured planning insight to the database.  Call this "
            "once per exception you identify.  Be specific in recommendation: "
            "name the exact action (change policy type, reduce EOQ by X%, "
            "trigger safety stock recalculation).  insight_type must be one of: "
            "stockout_risk | excess_inventory | forecast_bias | policy_gap | "
            "champion_degradation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "insight_type":              {"type": "string"},
                "severity":                  {"type": "string", "enum": ["critical","high","medium","low"]},
                "item_no":                   {"type": "string"},
                "loc":                       {"type": "string"},
                "summary":                   {"type": "string"},
                "recommendation":            {"type": "string"},
                "reasoning":                 {"type": "string"},
                "financial_impact_estimate": {"type": "number"},
                "dos":                       {"type": "number"},
                "total_lt_days":             {"type": "integer"},
                "champion_wape":             {"type": "number"},
                "forecast_bias_pct":         {"type": "number"},
                "current_policy_id":         {"type": "string"},
                "eoq_effective":             {"type": "number"},
                "abc_vol":                   {"type": "string"},
                "cluster_assignment":        {"type": "string"},
            },
            "required": ["insight_type","severity","item_no","loc","summary","recommendation"],
        },
    },
]

_SYSTEM_PROMPT = """You are a Demand and Inventory Planning AI agent for a supply chain platform.

## Platform context

The platform has five interconnected layers:
1. **Demand forecasting** — champion model selected per DFU per month from LGBM/CatBoost/XGBoost ensemble.
   Champion WAPE > 35% is high; > 50% is critical. Forecast bias > ±20% for 3+ months = persistent bias.
2. **Inventory analytics** — days of supply (DOS), stockout flag (DOS < lead_time × 1.5), excess flag (DOS > 180).
3. **EOQ targets** — Wilson EOQ with MOQ/cap. Effective EOQ = optimal replenishment quantity.
4. **Replenishment policies** — auto-assigned by ABC class + variability:
   - A-class low-variability → continuous_rop
   - A-class high-variability → safety_stock_buffer
   - lumpy demand → periodic_order_up_to
   - C-class / unknown → manual_review
5. **Job automation** — scheduled pipeline for re-computation.

## Your task

You are generating a proactive exception work-queue for planners.  This is NOT a chatbot.

For each DFU you analyse:
1. Call `get_dfu_full_context` first to get the full snapshot.
2. Call additional tools as needed to trace the causal chain: forecast accuracy → inventory consequence → policy effectiveness → financial impact.
3. For every exception you identify, call `create_insight` with a specific, actionable recommendation.

## Severity rules

- **critical**: DOS < lead_time (imminent stockout) OR WAPE > 50% on A-class
- **high**: DOS < lead_time × 1.5 OR WAPE > 35% OR persistent bias on A/B-class
- **medium**: DOS < lead_time × 2 OR excess > 180 DOS OR policy gap on A-class
- **low**: WAPE > 35% on C-class OR mild excess on B/C-class

## Output requirements

- Always use real data from tools before writing insights.
- Be specific: "reduce EOQ by 40% from 250 to 150 units" is better than "reduce order quantity".
- Trace the causal chain in the reasoning field: explain HOW the signals connect.
- One `create_insight` call per distinct exception type per DFU.
- summary MUST include at least one number (DOS, WAPE %, units, days, $).
- recommendation MUST name the exact action (change policy type, apply X× multiplier, reorder N units).

## Few-shot examples

### Example 1 — Stockout risk driven by over-forecast (A-class)

Tool results: current_dos=6.4, total_lt_days=14, champion_wape=63.2, bias_6m_avg=+88.5, avg_daily_sales=33

Correct create_insight call:
  insight_type    = "stockout_risk"
  severity        = "critical"
  summary         = "DOS 6.4d below lead time 14d — stockout in ~8 days. Persistent +88.5% over-forecast for 6 months exhausted safety stock."
  recommendation  = "1) Emergency reorder 462 units (14d × 33u/day) now. 2) Apply forecast multiplier 0.52× for next 3 months. 3) Switch policy continuous_rop → safety_stock_buffer."
  financial_impact_estimate = 7920.0

### Example 2 — Excess inventory from stale cluster label (B-class)

Tool results: current_dos=247, total_lt_days=21, cluster_assignment="high_volume_steady", variability_class="high", eoq_effective=300

Correct create_insight call:
  insight_type    = "excess_inventory"
  severity        = "high"
  summary         = "DOS 247d is 6.5× peer average of 38d for high_volume_steady cluster. Variability class HIGH contradicts cluster label — EOQ sized 300u for wrong demand profile."
  recommendation  = "1) Re-run clustering scenario to reclassify this DFU. 2) Switch policy continuous_rop → periodic_order_up_to for high-variability items. 3) Suspend planned orders until DOS falls below 120d."
  financial_impact_estimate = 10130.0
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class AIPlannerAgent:
    """LLM tool-use agent that scans portfolio exceptions and writes insights.

    Supports two providers (configured via ai_planner_config.yaml):
        provider: "openai"      → uses openai.OpenAI() + OPENAI_API_KEY  (default)
        provider: "anthropic"   → uses anthropic.Anthropic() + ANTHROPIC_API_KEY
    """

    def __init__(self, pool, config: dict):
        self.pool = pool
        self.config = config
        self.provider = config.get("provider", "openai").lower()
        self.model = config.get("model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.2)

        if self.provider == "anthropic":
            import anthropic as _anthropic
            self.client = _anthropic.Anthropic()
        else:
            from openai import OpenAI
            self.client = OpenAI()

    # ------------------------------------------------------------------
    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Route a tool call to the correct Python handler."""
        fns = {
            "get_dfu_full_context":         lambda: get_dfu_full_context(self.pool, **tool_input),
            "get_forecast_performance":     lambda: get_forecast_performance(self.pool, **tool_input),
            "get_portfolio_exceptions":     lambda: get_portfolio_exceptions(self.pool, **tool_input),
            "compute_bias_trend":           lambda: compute_bias_trend(self.pool, **tool_input),
            "get_inventory_trend":          lambda: get_inventory_trend(self.pool, **tool_input),
            "get_eoq_context":              lambda: get_eoq_context(self.pool, **tool_input),
            "get_similar_dfus":             lambda: get_similar_dfus(self.pool, **tool_input),
            "check_stockout_history":       lambda: check_stockout_history(self.pool, **tool_input),
            "get_portfolio_health_summary": lambda: get_portfolio_health_summary(self.pool),
            "create_insight":               lambda: create_insight(
                self.pool,
                scan_run_id=tool_input.get("scan_run_id"),
                model_version=self.model,
                **{k: v for k, v in tool_input.items() if k not in ("scan_run_id", "model_version")},
            ),
        }
        fn = fns.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn()
        except Exception as exc:
            log.warning("Tool %s failed: %s", tool_name, exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    def _run_agentic_loop(self, user_message: str, scan_run_id: str) -> tuple[str, list[int]]:
        """Dispatch to the provider-specific agentic loop."""
        if self.provider == "anthropic":
            return self._run_anthropic_loop(user_message, scan_run_id)
        return self._run_openai_loop(user_message, scan_run_id)

    # ------------------------------------------------------------------
    def _run_openai_loop(self, user_message: str, scan_run_id: str) -> tuple[str, list[int]]:
        """OpenAI function-calling agentic loop (bounded by MAX_TURNS + TOKEN_BUDGET)."""
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        insight_ids: list[int] = []
        oai_tools = _tools_to_openai(_TOOL_DEFINITIONS)
        turn = 0
        total_tokens = 0

        while turn < MAX_TURNS and total_tokens < TOKEN_BUDGET:
            turn += 1
            t0 = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=oai_tools,
                messages=messages,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            usage = response.usage
            if usage:
                total_tokens += usage.total_tokens or 0
                log_ai_call(
                    self.pool, scan_run_id, "openai", self.model, turn,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    latency_ms=latency_ms,
                )

            choice = response.choices[0]
            msg = choice.message
            # Append assistant turn (serialised to plain dict for continuity)
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ] or None,
            })

            if choice.finish_reason == "stop":
                return msg.content or "", insight_ids

            if choice.finish_reason == "tool_calls":
                for tc in msg.tool_calls or []:
                    raw_input = json.loads(tc.function.arguments)
                    if tc.function.name == "create_insight":
                        raw_input.setdefault("scan_run_id", scan_run_id)
                        raw_input.setdefault("model_version", self.model)

                    t1 = time.monotonic()
                    result = self._dispatch_tool(tc.function.name, raw_input)
                    tool_ms = int((time.monotonic() - t1) * 1000)

                    if tc.function.name == "create_insight" and isinstance(result, int) and result > 0:
                        insight_ids.append(result)

                    log_ai_call(
                        self.pool, scan_run_id, "openai", self.model, turn,
                        latency_ms=tool_ms,
                        tool_name=tc.function.name,
                        tool_success=not isinstance(result, dict) or "error" not in result,
                        error_type=result.get("error") if isinstance(result, dict) else None,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })
                continue

            log.warning("Unexpected finish_reason: %s", choice.finish_reason)
            break

        if turn >= MAX_TURNS:
            log.warning("OpenAI loop hit MAX_TURNS=%d  scan_run_id=%s", MAX_TURNS, scan_run_id)
        if total_tokens >= TOKEN_BUDGET:
            log.warning("OpenAI loop hit TOKEN_BUDGET=%d  scan_run_id=%s", TOKEN_BUDGET, scan_run_id)

        return "", insight_ids

    # ------------------------------------------------------------------
    def _run_anthropic_loop(self, user_message: str, scan_run_id: str) -> tuple[str, list[int]]:
        """Anthropic tool_use agentic loop (bounded by MAX_TURNS + TOKEN_BUDGET)."""
        messages: list[dict] = [{"role": "user", "content": user_message}]
        insight_ids: list[int] = []
        turn = 0
        total_tokens = 0

        def _wrap(tool_input: dict) -> dict:
            out = dict(tool_input)
            out.setdefault("scan_run_id", scan_run_id)
            out.setdefault("model_version", self.model)
            return out

        while turn < MAX_TURNS and total_tokens < TOKEN_BUDGET:
            turn += 1
            t0 = time.monotonic()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=_SYSTEM_PROMPT,
                tools=_TOOL_DEFINITIONS,
                messages=messages,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            if hasattr(response, "usage") and response.usage:
                used = getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)
                total_tokens += used
                log_ai_call(
                    self.pool, scan_run_id, "anthropic", self.model, turn,
                    prompt_tokens=getattr(response.usage, "input_tokens", None),
                    completion_tokens=getattr(response.usage, "output_tokens", None),
                    total_tokens=used,
                    latency_ms=latency_ms,
                )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = " ".join(
                    b.text for b in response.content if hasattr(b, "text")
                )
                return final_text, insight_ids

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    raw_input = json.loads(json.dumps(block.input))
                    if block.name == "create_insight":
                        raw_input = _wrap(raw_input)

                    t1 = time.monotonic()
                    result = self._dispatch_tool(block.name, raw_input)
                    tool_ms = int((time.monotonic() - t1) * 1000)

                    if block.name == "create_insight" and isinstance(result, int) and result > 0:
                        insight_ids.append(result)

                    log_ai_call(
                        self.pool, scan_run_id, "anthropic", self.model, turn,
                        latency_ms=tool_ms,
                        tool_name=block.name,
                        tool_success=not isinstance(result, dict) or "error" not in result,
                        error_type=result.get("error") if isinstance(result, dict) else None,
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

                messages.append({"role": "user", "content": tool_results})
                continue

            log.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        if turn >= MAX_TURNS:
            log.warning("Anthropic loop hit MAX_TURNS=%d  scan_run_id=%s", MAX_TURNS, scan_run_id)
        if total_tokens >= TOKEN_BUDGET:
            log.warning("Anthropic loop hit TOKEN_BUDGET=%d  scan_run_id=%s", TOKEN_BUDGET, scan_run_id)

        return "", insight_ids

    # ------------------------------------------------------------------
    def run_dfu_analysis(self, item_no: str, loc: str, scan_run_id: str | None = None) -> list[dict]:
        """Analyse a single DFU and return list of insight dicts created."""
        scan_run_id = scan_run_id or str(uuid.uuid4())
        task = (
            f"Analyse DFU {item_no} @ {loc}. "
            "Start with get_dfu_full_context, then drill into forecast performance, "
            "inventory trend, and EOQ context as needed. "
            "Call create_insight for every exception you identify. "
            "End with a brief summary of your findings."
        )
        log.info("DFU analysis: %s @ %s  scan_run_id=%s", item_no, loc, scan_run_id)
        _, insight_ids = self._run_agentic_loop(task, scan_run_id)
        return self._fetch_insights_by_ids(insight_ids)

    # ------------------------------------------------------------------
    def run_portfolio_scan(self, scan_run_id: str | None = None) -> dict:
        """Scan top N portfolio exceptions; return summary dict."""
        scan_run_id = scan_run_id or str(uuid.uuid4())
        limit = self.config.get("portfolio_scan_limit", 100)
        task = (
            f"Run a portfolio exception scan. "
            f"Start with get_portfolio_exceptions(limit={limit}) to identify "
            "the DFUs with the most critical signals. "
            "Then drill into the top exceptions — use get_dfu_full_context, "
            "get_forecast_performance, compute_bias_trend, check_stockout_history, "
            "and get_eoq_context as needed. "
            "Call create_insight for every material exception you find. "
            "Focus on A and B class DFUs first. "
            "End with a one-paragraph summary of key findings."
        )
        log.info("Portfolio scan started  scan_run_id=%s", scan_run_id)
        final_text, insight_ids = self._run_agentic_loop(task, scan_run_id)

        # Write a portfolio memo
        content_json: dict = {
            "insight_ids": insight_ids,
            "total_insights": len(insight_ids),
            "scan_run_id": scan_run_id,
        }
        self._write_memo(
            period=date.today().replace(day=1),
            scope="portfolio",
            narrative_text=final_text or "Portfolio scan completed.",
            content_json=content_json,
        )
        return {
            "scan_run_id": scan_run_id,
            "total_insights": len(insight_ids),
            "insight_ids": insight_ids,
            "summary": final_text,
        }

    # ------------------------------------------------------------------
    def generate_portfolio_memo(self, period: date, scan_run_id: str | None = None) -> dict:
        """Generate a narrative weekly planning memo for the portfolio."""
        scan_run_id = scan_run_id or str(uuid.uuid4())
        task = (
            f"Generate a weekly planning memo for {period.strftime('%B %Y')}. "
            "Call get_portfolio_health_summary to ground your analysis. "
            "Then review the top open insights. "
            "Write a 2-3 paragraph narrative covering: "
            "(1) portfolio health status, "
            "(2) top 3 exceptions requiring planner action this week, "
            "(3) recommended priorities. "
            "Be specific about item numbers and financial impact where possible."
        )
        log.info("Portfolio memo  period=%s  scan_run_id=%s", period, scan_run_id)
        narrative, _ = self._run_agentic_loop(task, scan_run_id)
        content_json: dict = {"scan_run_id": scan_run_id, "period": str(period)}
        memo_id = self._write_memo(
            period=period,
            scope="portfolio",
            narrative_text=narrative or "No memo generated.",
            content_json=content_json,
        )
        return {"memo_id": memo_id, "period": str(period), "narrative": narrative}

    # ------------------------------------------------------------------
    def _write_memo(
        self,
        period: date,
        scope: str,
        narrative_text: str,
        content_json: dict,
        item_no: str | None = None,
        loc: str | None = None,
    ) -> int:
        sql = """
            INSERT INTO ai_planning_memos
                (period, scope, item_no, loc, narrative_text, content_json, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING memo_id
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    period, scope, item_no, loc,
                    narrative_text, json.dumps(content_json, default=str),
                    self.model,
                ))
                row = cur.fetchone()
                conn.commit()
                return row[0] if row else -1

    # ------------------------------------------------------------------
    def _fetch_insights_by_ids(self, insight_ids: list[int]) -> list[dict]:
        if not insight_ids:
            return []
        placeholders = ",".join(["%s"] * len(insight_ids))
        sql = f"SELECT * FROM ai_insights WHERE insight_id IN ({placeholders})"
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, insight_ids)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
