"""Forecast Value Added (FVA) & ROI tracking endpoints (Spec 08-07)."""
from __future__ import annotations

import psycopg
from fastapi import APIRouter, Query

from api.core import get_conn

router = APIRouter(prefix="/fva", tags=["fva"])


STAGE_DEFS = [
    ("seasonal_naive", "Naive Seasonal", "Same-month-last-year baseline for measuring planning lift.", "actual"),
    ("external", "External", "Current ERP or external forecast before model selection.", "actual"),
    ("champion", "Champion", "Best measured statistical or ML model for the DFU-month.", "actual"),
    ("ai_adjusted", "AI Adjusted", "Reserved for AI-assisted forecast interventions once they are measured.", "planned"),
    ("planner_adjusted", "Planner Adjusted", "Reserved for human overrides once measured outcomes are available.", "planned"),
]


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None else None


def _build_stage(stage_id: str, label: str, description: str, default_state: str, model: dict | None) -> dict:
    if default_state == "planned":
        state, accuracy, n_rows = "planned", None, 0
    elif model is None:
        state, accuracy, n_rows = "missing", None, 0
    else:
        state, accuracy, n_rows = "actual", model["accuracy_pct"], model["n_rows"]
    return {
        "stage_id": stage_id,
        "label": label,
        "description": description,
        "state": state,
        "accuracy_pct": accuracy,
        "delta_vs_prev": None,
        "n_rows": n_rows,
    }


@router.get("/waterfall")
async def fva_waterfall(
    months: int = Query(12, ge=1, le=36),
):
    """FVA ladder data: naive seasonal -> external -> champion -> future adjustment stages."""
    # Shared DFU-month filter: execution-lag rows within the requested horizon.
    # See docs/specs/01-foundation/06-execution-lag.md for the canonical pattern.
    dfu_filter = (
        "FROM fact_external_forecast_monthly f "
        "JOIN dim_sku d ON d.item_id = f.item_id AND d.loc = f.loc "
        "WHERE f.startdate >= current_date - (%s * interval '1 month') "
        "AND f.lag = COALESCE(d.execution_lag, 0) "
        "AND f.tothist_dmd IS NOT NULL"
    )
    with get_conn() as conn, conn.cursor() as cur:
        # Accuracy per model_id at each DFU's execution lag (planning-relevant horizon).
        cur.execute(
            f"""SELECT f.model_id,
                       100.0 - (100.0 * sum(abs(f.basefcst_pref - f.tothist_dmd)) / NULLIF(abs(sum(f.tothist_dmd)), 0)) AS accuracy_pct,
                       count(*) AS n_rows
                {dfu_filter}
                GROUP BY f.model_id
                ORDER BY accuracy_pct DESC NULLS LAST""",
            (months,),
        )
        rows = cur.fetchall()

        # Seasonal naive: same-month-last-year sales as the forecast, evaluated
        # over the DFU-month universe at execution lag. Computed on the fly — no
        # seasonal_naive rows are loaded into fact_external_forecast_monthly.
        cur.execute(
            f"""WITH dfu_months AS (
                   SELECT DISTINCT f.item_id, f.customer_group, f.loc, f.startdate, f.tothist_dmd
                   {dfu_filter}
               )
               SELECT
                   100.0 - 100.0 * sum(abs(coalesce(s.qty, 0) - m.tothist_dmd))
                       / NULLIF(abs(sum(m.tothist_dmd)), 0) AS accuracy_pct,
                   count(*) AS n_rows
               FROM dfu_months m
               LEFT JOIN fact_sales_monthly s
                 ON s.item_id = m.item_id
                AND s.customer_group = m.customer_group
                AND s.loc = m.loc
                AND s.type = 1
                AND s.startdate = m.startdate - interval '12 months'""",
            (months,),
        )
        naive_row = cur.fetchone()

    models = {
        r[0]: {"model_id": r[0], "accuracy_pct": _round_or_none(r[1]), "n_rows": r[2]}
        for r in rows
    }
    if naive_row and naive_row[0] is not None and naive_row[1]:
        models["seasonal_naive"] = {
            "model_id": "seasonal_naive",
            "accuracy_pct": _round_or_none(naive_row[0]),
            "n_rows": naive_row[1],
        }

    # Promote `ai_adjusted` from the latest succeeded AI FVA backtest run, if any.
    # PRD: docs/specs/PRD/PRD-ai-planner-fva-backtest.md §6 "FVA waterfall integration"
    # Accuracy = 100 - WAPE  (uniform with the rest of this endpoint).
    ai_run_id: str | None = None
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """SELECT r.run_id::text, m.ai_wape_pct, COALESCE(SUM(o.n_dfus), 0)
                   FROM ai_fva_backtest_run r
                   JOIN mv_ai_fva_overall  m ON m.run_id = r.run_id
                   JOIN mv_ai_fva_overall  o ON o.run_id = r.run_id
                   WHERE r.status = 'succeeded' AND m.ai_wape_pct IS NOT NULL
                   GROUP BY r.run_id, m.ai_wape_pct, r.completed_at
                   ORDER BY r.completed_at DESC NULLS LAST
                   LIMIT 1"""
            )
            ai_row = cur.fetchone()
        except psycopg.Error:
            # Tables may not exist yet on a fresh DB. Silently fall back.
            conn.rollback()
            ai_row = None
        if ai_row and ai_row[1] is not None:
            ai_run_id = ai_row[0]
            models["ai_adjusted"] = {
                "model_id": "ai_adjusted",
                "accuracy_pct": _round_or_none(100.0 - float(ai_row[1])),
                "n_rows": int(ai_row[2] or 0),
            }

    stages = [
        _build_stage(stage_id, label, description, default_state, models.get(stage_id))
        for stage_id, label, description, default_state in STAGE_DEFS
    ]
    # Promote AI Adjusted from its "planned" default to actual when a backtest run exists.
    # _build_stage() short-circuits on default_state == "planned" and returns
    # accuracy_pct=None / n_rows=0, so we must overwrite those fields here too —
    # otherwise the downstream delta_vs_prev subtraction crashes on None.
    ai_model = models.get("ai_adjusted")
    if ai_model is not None:
        for s in stages:
            if s["stage_id"] == "ai_adjusted":
                s["state"] = "actual"
                s["accuracy_pct"] = ai_model["accuracy_pct"]
                s["n_rows"] = ai_model["n_rows"]
                s["ai_fva_run_id"] = ai_run_id
                break
    for idx in range(1, len(stages)):
        prev = stages[idx - 1]
        curr = stages[idx]
        if prev["state"] == "actual" and curr["state"] == "actual":
            curr["delta_vs_prev"] = _round_or_none(curr["accuracy_pct"] - prev["accuracy_pct"], 1)

    ceiling_model = models.get("ceiling")
    benchmark = {
        "stage_id": "ceiling",
        "label": "Ceiling Benchmark",
        "description": "Reference best-case benchmark rather than a production handoff stage.",
        "state": "actual" if ceiling_model else "missing",
        "accuracy_pct": ceiling_model["accuracy_pct"] if ceiling_model else None,
        "delta_vs_prev": None,
        "n_rows": ceiling_model["n_rows"] if ceiling_model else 0,
    }

    return {
        "months": months,
        "waterfall": {
            "stages": stages,
            "benchmark": benchmark,
            "external": models.get("external"),
            "champion": models.get("champion"),
            "ceiling": models.get("ceiling"),
            "models": list(models.values()),
        },
    }


@router.get("/interventions")
async def list_interventions(
    intervention_type: str = Query("", description="Filter by type"),
    status: str = Query("", description="Filter by status"),
    user_id: str = Query("", description="Filter by user"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List tracked interventions with before/after metrics."""
    where = []
    params: list = []
    if intervention_type:
        where.append("intervention_type = %s")
        params.append(intervention_type)
    if status:
        where.append("status = %s")
        params.append(status)
    if user_id:
        where.append("user_id = %s::uuid")
        params.append(user_id)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM fact_intervention_metrics {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT intervention_id, user_id, intervention_type, resource_type, resource_id,
                       metric_before, metric_after, financial_impact_estimate,
                       actual_financial_impact, measurement_window_start,
                       measurement_window_end, status, created_at
                FROM fact_intervention_metrics
                {where_sql}
                ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    return {
        "total": total,
        "interventions": [
            {
                "intervention_id": r[0],
                "user_id": str(r[1]) if r[1] else None,
                "intervention_type": r[2],
                "resource_type": r[3], "resource_id": r[4],
                "metric_before": r[5], "metric_after": r[6],
                "financial_impact_estimate": float(r[7]) if r[7] else None,
                "actual_financial_impact": float(r[8]) if r[8] else None,
                "measurement_window_start": r[9].isoformat() if r[9] else None,
                "measurement_window_end": r[10].isoformat() if r[10] else None,
                "status": r[11],
                "created_at": r[12].isoformat() if r[12] else None,
            }
            for r in rows
        ],
    }


@router.get("/roi-summary")
async def roi_summary(
    months: int = Query(12, ge=1, le=36),
):
    """Aggregate ROI metrics: total interventions, measured outcomes, net financial impact."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT
                 count(*) AS total_interventions,
                 count(*) FILTER (WHERE status = 'measured') AS measured,
                 count(*) FILTER (WHERE status = 'pending') AS pending,
                 coalesce(sum(financial_impact_estimate), 0) AS total_estimated_impact,
                 coalesce(sum(actual_financial_impact) FILTER (WHERE status = 'measured'), 0) AS total_actual_impact
               FROM fact_intervention_metrics
               WHERE created_at >= current_date - (%s * interval '1 month')""",
            (months,),
        )
        row = cur.fetchone()

    return {
        "months": months,
        "total_interventions": row[0],
        "measured": row[1],
        "pending": row[2],
        "total_estimated_impact": float(row[3]) if row[3] else 0,
        "total_actual_impact": float(row[4]) if row[4] else 0,
    }
