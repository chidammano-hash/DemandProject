"""Forecast Value Added (FVA) & ROI tracking endpoints (Spec 08-07)."""
from __future__ import annotations

import psycopg
from fastapi import APIRouter, Query

from api.core import get_conn
from common.core.planning_date import get_planning_date

router = APIRouter(prefix="/fva", tags=["fva"])


# Each tuple: (stage_id, label, description, default_state, missing_state).
# ``default_state`` is "planned" for stages that are always reserved (no source
# data yet). ``missing_state`` is the state a normally-measured ("actual") stage
# degrades to when its source yields no rows:
#   - external degrades to "missing" (genuinely no forecast → honest blank).
#   - champion is measured from the promoted champion-selection experiment's
#     backtest accuracy (``champion_experiment_month``, a stored 100 - WAPE over
#     per-DFU champion-selected forecasts at execution lag, windowed by months) —
#     NOT from fact_external_forecast_monthly (which only ever holds
#     model_id='external') nor fact_production_forecast (forward-only, no
#     measurable actual overlap). It degrades to "planned" (not "missing") when
#     no promoted experiment exists, so it presents consistently with the
#     AI/Planner reserved stages on a fresh DB.
STAGE_DEFS = [
    ("seasonal_naive", "Naive Seasonal", "Same-month-last-year baseline for measuring planning lift.", "actual", "missing"),
    ("external", "External", "Current ERP or external forecast before model selection.", "actual", "missing"),
    ("champion", "Champion", "Best measured statistical or ML model once champion-vs-actual outcomes are available.", "actual", "planned"),
    ("ai_adjusted", "AI Adjusted", "Reserved for AI-assisted forecast interventions once they are measured.", "planned", "planned"),
    ("planner_adjusted", "Planner Adjusted", "Reserved for human overrides once measured outcomes are available.", "planned", "planned"),
]


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None else None


def _build_stage(
    stage_id: str, label: str, description: str, default_state: str, missing_state: str, model: dict | None
) -> dict:
    if default_state == "planned":
        state, accuracy, n_rows = "planned", None, 0
    elif model is None:
        state, accuracy, n_rows = missing_state, None, 0
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
    """FVA ladder data: naive seasonal -> external -> champion -> future adjustment stages.

    ``months`` windows every measured stage to the trailing horizon. Accuracy is
    always measured at each DFU's execution lag (the horizon it is operationally
    forecast at).
    """
    # Shared DFU-month filter: execution-lag rows within the requested horizon.
    # See docs/specs/01-foundation/06-execution-lag.md for the canonical pattern.
    # F9.1: anchor the window to the planning date, NOT the DB wall-clock
    # ``current_date`` — the demo forecast horizon trails the system clock, so a
    # ``current_date``-anchored 3-month window matches zero rows and blanks the
    # ladder. The planning date is bound as a parameter (``%s::date``), in line
    # with sibling forecasting routers (production_forecast.py, consensus_plan.py).
    planning_dt = get_planning_date()
    dfu_filter = (
        "FROM fact_external_forecast_monthly f "
        "JOIN dim_sku d ON d.item_id = f.item_id AND d.loc = f.loc "
        "WHERE f.startdate >= %s::date - (%s * interval '1 month') "
        "AND f.lag = COALESCE(d.execution_lag, 0) "
        "AND f.tothist_dmd IS NOT NULL"
    )
    # Each query embedding ``dfu_filter`` must supply (planning_dt, months) in order.
    dfu_params = (planning_dt, months)
    with get_conn() as conn, conn.cursor() as cur:
        # Accuracy per model_id at each DFU's execution lag (planning-relevant horizon).
        cur.execute(
            f"""SELECT f.model_id,
                       100.0 - (100.0 * sum(abs(f.basefcst_pref - f.tothist_dmd)) / NULLIF(abs(sum(f.tothist_dmd)), 0)) AS accuracy_pct,
                       count(*) AS n_rows
                {dfu_filter}
                GROUP BY f.model_id
                ORDER BY accuracy_pct DESC NULLS LAST""",
            dfu_params,
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
            dfu_params,
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

    # `ai_adjusted` and `planner_adjusted` remain reserved ("planned") stages in
    # the ladder — there is no measured AI accuracy to source. The forward-only
    # AI Champion adjuster (docs/specs/02-forecasting/27-ai-champion-forecast.md)
    # writes a forward forecast (model_id='ai_champion') with no historical
    # actual overlap, so it cannot feed this accuracy waterfall.

    # Promote `champion` and `ceiling` from the promoted champion-selection
    # experiment, windowed by `months` over the per-month rollup
    # (champion_experiment_month), volume-weighted by n_champions so champion and
    # ceiling track the selector like external/naive. champion_accuracy /
    # ceiling_accuracy are stored as 100 - WAPE over the per-DFU champion-selected
    # (and oracle-best) backtest forecasts at execution lag — uniform with the
    # rest of this endpoint. See scripts/ml/run_champion_experiment.py and
    # docs/specs/02-forecasting/07-champion-selection.md. Returns a
    # (champion_accuracy, ceiling_accuracy, n) triple.
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """SELECT
                      SUM(m.champion_accuracy * m.n_champions) / NULLIF(SUM(m.n_champions), 0),
                      SUM(m.ceiling_accuracy  * m.n_champions) / NULLIF(SUM(m.n_champions), 0),
                      SUM(m.n_champions)
                   FROM champion_experiment_month m
                   JOIN champion_experiment e ON e.experiment_id = m.experiment_id
                   WHERE e.is_promoted = TRUE
                     AND m.month_start >= %s::date - (%s * interval '1 month')
                     AND m.month_start <  %s::date""",
                (planning_dt, months, planning_dt),
            )
            champ_row = cur.fetchone()
        except psycopg.Error:
            # Tables may not exist yet on a fresh DB. Silently fall back.
            conn.rollback()
            champ_row = None
    if champ_row is not None and len(champ_row) >= 3:
        champ_n_rows = int(champ_row[2] or 0)
        if champ_row[0] is not None:
            models["champion"] = {
                "model_id": "champion",
                "accuracy_pct": _round_or_none(float(champ_row[0])),
                "n_rows": champ_n_rows,
            }
        if champ_row[1] is not None:
            models["ceiling"] = {
                "model_id": "ceiling",
                "accuracy_pct": _round_or_none(float(champ_row[1])),
                "n_rows": champ_n_rows,
            }

    stages = [
        _build_stage(stage_id, label, description, default_state, missing_state, models.get(stage_id))
        for stage_id, label, description, default_state, missing_state in STAGE_DEFS
    ]
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
               WHERE created_at >= %s::date - (%s * interval '1 month')""",
            (get_planning_date(), months),
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
