"""Data Quality & Pipeline Observability endpoints (Spec 08-01)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.core import get_conn

logger = logging.getLogger(__name__)

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
):
    """Trigger an ad-hoc data quality check run."""
    from common.dq_engine import DQEngine
    engine = DQEngine()
    results = engine.run_all_checks(domain=domain or None)
    return {"triggered": len(results), "message": "ok", "results": results, "total": len(results)}


@router.get("/fix/preview")
async def dq_fix_preview(
    fix_type: str = Query("", description="Filter by fix type (range, completeness, orphans)"),
):
    """Preview all available auto-fixes as an indexed list (dry-run only).

    Each item has an `id` that can be used to selectively apply fixes via POST /fix/apply.
    """
    from scripts.ops.fix_dq_issues import preview_all_fixes, FIX_REGISTRY

    if fix_type and fix_type not in FIX_REGISTRY:
        return {"error": f"Unknown fix type: {fix_type}. Valid: {list(FIX_REGISTRY.keys())}"}

    items = preview_all_fixes(fix_type=fix_type or None)
    return {"items": items, "total": len(items)}


class FixApplyRequest(BaseModel):
    fix_ids: list[int]


@router.post("/fix/apply")
async def dq_fix_apply(body: FixApplyRequest):
    """Apply selected fixes by their preview IDs.

    Pass `fix_ids` array from the preview response. Only those fixes are applied;
    all others are skipped.
    """
    from scripts.ops.fix_dq_issues import apply_selected_fixes

    if not body.fix_ids:
        return {"error": "No fix IDs provided", "applied": [], "total_applied": 0}

    result = apply_selected_fixes(body.fix_ids)
    return result


@router.get("/corrections")
async def dq_corrections(
    item_id: str = Query("", description="Filter by item_id"),
    loc: str = Query("", description="Filter by location"),
    table_name: str = Query("", description="Filter by table (e.g. fact_sales_monthly)"),
    column_name: str = Query("", description="Filter by column (e.g. qty)"),
    fix_type: str = Query("", description="Filter by fix type (outliers, range, etc.)"),
    limit: int = Query(500, ge=1, le=5000),
):
    """Fetch DQ correction audit trail for a given item/location.

    Returns original vs corrected values so the UI can overlay before/after
    on demand and inventory charts.
    """
    where_parts: list[str] = []
    params: list = []

    if item_id:
        where_parts.append("item_id = %s")
        params.append(item_id)
    if loc:
        where_parts.append("loc = %s")
        params.append(loc)
    if table_name:
        where_parts.append("table_name = %s")
        params.append(table_name)
    if column_name:
        where_parts.append("column_name = %s")
        params.append(column_name)
    if fix_type:
        where_parts.append("fix_type = %s")
        params.append(fix_type)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT correction_id, domain, table_name, item_id, loc, "
            f"  period, column_name, old_value, new_value, "
            f"  fix_type, fix_strategy, threshold, lower_bound, upper_bound, "
            f"  applied_at "
            f"FROM fact_dq_corrections {where_sql} "
            f"ORDER BY period, column_name "
            f"LIMIT %s",
            [*params, limit],
        )
        rows = cur.fetchall()

    corrections = [
        {
            "correction_id": r[0],
            "domain": r[1],
            "table_name": r[2],
            "item_id": r[3],
            "loc": r[4],
            "period": r[5].isoformat() if r[5] else None,
            "column_name": r[6],
            "old_value": float(r[7]) if r[7] is not None else None,
            "new_value": float(r[8]) if r[8] is not None else None,
            "fix_type": r[9],
            "fix_strategy": r[10],
            "threshold": float(r[11]) if r[11] is not None else None,
            "lower_bound": float(r[12]) if r[12] is not None else None,
            "upper_bound": float(r[13]) if r[13] is not None else None,
            "applied_at": r[14].isoformat() if r[14] else None,
        }
        for r in rows
    ]

    return {"corrections": corrections, "total": len(corrections)}


@router.get("/corrections/summary")
async def dq_corrections_summary(
    domain: str = Query("", description="Filter by domain (sales, inventory)"),
    fix_type: str = Query("", description="Filter by fix type"),
    sort_by: str = Query("correction_count", description="Sort field"),
    sort_dir: str = Query("desc", description="Sort direction"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Aggregated DQ correction summary grouped by item_id + loc.

    Returns one row per corrected SKU with counts, affected columns,
    fix types, and date range.
    """
    where_parts: list[str] = []
    params: list = []
    if domain:
        where_parts.append("domain = %s")
        params.append(domain)
    if fix_type:
        where_parts.append("fix_type = %s")
        params.append(fix_type)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    allowed_sort = {"correction_count", "item_id", "loc", "latest_at", "domain"}
    sort_col = sort_by if sort_by in allowed_sort else "correction_count"
    direction = "ASC" if sort_dir.lower() == "asc" else "DESC"

    with get_conn() as conn, conn.cursor() as cur:
        # Total count for pagination
        cur.execute(
            f"SELECT count(*) FROM ("
            f"  SELECT item_id, loc FROM fact_dq_corrections {where_sql} "
            f"  GROUP BY item_id, loc"
            f") sub",
            params,
        )
        total = cur.fetchone()[0]

        # Aggregated summary
        cur.execute(
            f"SELECT item_id, loc, "
            f"  count(*) AS correction_count, "
            f"  array_agg(DISTINCT domain) AS domains, "
            f"  array_agg(DISTINCT table_name) AS tables, "
            f"  array_agg(DISTINCT column_name) AS columns, "
            f"  array_agg(DISTINCT fix_type) AS fix_types, "
            f"  array_agg(DISTINCT fix_strategy) FILTER (WHERE fix_strategy IS NOT NULL) AS strategies, "
            f"  min(period) AS earliest_period, "
            f"  max(period) AS latest_period, "
            f"  max(applied_at) AS latest_at "
            f"FROM fact_dq_corrections {where_sql} "
            f"GROUP BY item_id, loc "
            f"ORDER BY {sort_col} {direction} "
            f"LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    skus = [
        {
            "item_id": r[0],
            "loc": r[1],
            "correction_count": r[2],
            "domains": r[3] or [],
            "tables": r[4] or [],
            "columns": r[5] or [],
            "fix_types": r[6] or [],
            "strategies": r[7] or [],
            "earliest_period": r[8].isoformat() if r[8] else None,
            "latest_period": r[9].isoformat() if r[9] else None,
            "latest_at": r[10].isoformat() if r[10] else None,
        }
        for r in rows
    ]

    return {"skus": skus, "total": total}


@router.post("/fix")
async def dq_fix(
    fix_type: str = Query("", description="Specific fix type (range, completeness, orphans)"),
    apply: bool = Query(False, description="Set true to apply fixes; false for dry-run preview"),
):
    """Run statistical auto-fix for DQ issues.

    Dry-run (default) previews fixes without writing. Set apply=true to execute.
    """
    from scripts.ops.fix_dq_issues import run_all_fixes, FIX_REGISTRY

    if fix_type and fix_type not in FIX_REGISTRY:
        return {"error": f"Unknown fix type: {fix_type}. Valid: {list(FIX_REGISTRY.keys())}"}

    result = run_all_fixes(fix_type=fix_type or None, dry_run=not apply)
    return result
