"""Data Quality & Pipeline Observability endpoints (Spec 08-01)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.core import get_conn
from common.auth import CurrentUser, get_current_user, require_role

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


@router.get("/dashboard")
async def dq_dashboard():
    """Domain health scores and recent check trends."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT domain,
                      count(*) FILTER (WHERE status = 'pass') AS passed,
                      count(*) FILTER (WHERE status = 'fail') AS failed,
                      count(*) FILTER (WHERE status = 'warn') AS warnings,
                      count(*) AS total
               FROM fact_dq_check_results
               WHERE run_ts >= now() - interval '24 hours'
               GROUP BY domain
               ORDER BY domain"""
        )
        rows = cur.fetchall()

    domains = []
    for r in rows:
        total = r[4] or 1
        score = round(100.0 * (r[1] or 0) / total, 1)
        domains.append({
            "domain": r[0],
            "score": score,
            "passed": r[1] or 0,
            "failed": r[2] or 0,
            "warnings": r[3] or 0,
            "total": r[4] or 0,
        })

    return {"domains": domains}


@router.get("/checks")
async def dq_checks():
    """List all configured checks with last-run status."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT c.check_id, c.check_name, c.check_type, c.domain,
                      c.table_name, c.severity, c.enabled,
                      r.status, r.metric_value, r.run_ts
               FROM dim_dq_check_catalog c
               LEFT JOIN LATERAL (
                 SELECT status, metric_value, run_ts
                 FROM fact_dq_check_results
                 WHERE check_name = c.check_name
                 ORDER BY run_ts DESC LIMIT 1
               ) r ON TRUE
               ORDER BY c.domain, c.check_name"""
        )
        rows = cur.fetchall()

    return {
        "checks": [
            {
                "check_id": r[0], "check_name": r[1], "check_type": r[2],
                "domain": r[3], "table_name": r[4], "severity": r[5],
                "enabled": r[6], "last_status": r[7], "last_value": float(r[8]) if r[8] is not None else None,
                "last_run": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ]
    }


@router.get("/history")
async def dq_history(
    domain: str = Query("", description="Filter by domain"),
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, ge=1, le=1000),
):
    """Check result history."""
    where = ["run_ts >= now() - interval '%s days'" % days]
    params: list = []
    if domain:
        where.append("domain = %s")
        params.append(domain)

    where_sql = "WHERE " + " AND ".join(where)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT check_id, check_name, domain, table_name, severity,
                       status, metric_value, details, run_ts
                FROM fact_dq_check_results
                {where_sql}
                ORDER BY run_ts DESC LIMIT %s""",
            [*params, limit],
        )
        rows = cur.fetchall()

    return {
        "entries": [
            {
                "check_id": r[0], "check_name": r[1], "domain": r[2],
                "table_name": r[3], "severity": r[4], "status": r[5],
                "metric_value": float(r[6]) if r[6] is not None else None,
                "details": r[7], "run_ts": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
    }


@router.post("/run")
async def dq_run(
    domain: str = Query("", description="Run checks for specific domain only"),
    admin: CurrentUser = Depends(require_role("manager")),
):
    """Trigger an ad-hoc data quality check run (manager+ only)."""
    from common.dq_engine import DQEngine
    engine = DQEngine()
    results = engine.run_all_checks(domain=domain or None)
    return {"results": results, "total": len(results)}


@router.get("/freshness")
async def dq_freshness():
    """Per-table last-load timestamps."""
    tables = [
        "dim_item", "dim_location", "dim_customer", "dim_dfu",
        "fact_sales_monthly", "fact_external_forecast_monthly",
    ]
    results = []
    with get_conn() as conn, conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT max(load_ts) FROM {table}")
                row = cur.fetchone()
                last_load = row[0] if row else None
                results.append({
                    "table": table,
                    "last_load": last_load.isoformat() if last_load else None,
                })
            except Exception:
                results.append({"table": table, "last_load": None})

    return {"tables": results}
