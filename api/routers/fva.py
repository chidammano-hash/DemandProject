"""Forecast Value Added (FVA) & ROI tracking endpoints (Spec 08-07)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.core import get_conn
from common.auth import CurrentUser, get_current_user

router = APIRouter(prefix="/fva", tags=["fva"])


@router.get("/waterfall")
async def fva_waterfall(
    months: int = Query(12, ge=1, le=36),
):
    """FVA waterfall data: naive -> statistical -> champion -> planner-adjusted accuracy."""
    with get_conn() as conn, conn.cursor() as cur:
        # Get accuracy by model_id for the FVA waterfall
        cur.execute(
            """SELECT model_id,
                      100.0 - (100.0 * sum(abs(basefcst_pref - tothist_dmd)) / NULLIF(abs(sum(tothist_dmd)), 0)) AS accuracy_pct,
                      count(*) AS n_rows
               FROM fact_external_forecast_monthly
               WHERE startdate >= now() - interval '%s months'
                 AND lag = 0
                 AND tothist_dmd IS NOT NULL
               GROUP BY model_id
               ORDER BY accuracy_pct DESC NULLS LAST""",
            (months,),
        )
        rows = cur.fetchall()

    models = {}
    for r in rows:
        models[r[0]] = {"model_id": r[0], "accuracy_pct": round(float(r[1]), 2) if r[1] else None, "n_rows": r[2]}

    return {
        "months": months,
        "waterfall": {
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
               WHERE created_at >= now() - interval '%s months'""",
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
