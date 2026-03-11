"""Inventory Planning — IPfeature6: Inventory Health Score endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.core import get_conn

router = APIRouter(tags=["inv-planning"])


@router.get("/inv-planning/health/summary")
def get_health_summary(
    abc_vol:            Optional[str] = None,
    cluster_assignment: Optional[str] = None,
    region:             Optional[str] = None,
    variability_class:  Optional[str] = None,
    brand:              Optional[str] = None,
    category:           Optional[str] = None,
    market:             Optional[str] = None,
):
    """Aggregate health score summary with tier breakdown."""
    where_clauses: list[str] = []
    params: list = []

    if abc_vol:
        where_clauses.append("abc_vol = %s")
        params.append(abc_vol)
    if cluster_assignment:
        where_clauses.append("cluster_assignment = %s")
        params.append(cluster_assignment)
    if region:
        where_clauses.append("region = %s")
        params.append(region)
    if variability_class:
        where_clauses.append("variability_class = %s")
        params.append(variability_class)
    if brand:
        params.append(brand.split(","))
        where_clauses.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        where_clauses.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        where_clauses.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = t.loc AND dl.state_id = ANY(%s))")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    summary_sql = f"""
        SELECT
            COUNT(*)                                                    AS total_dfus,
            AVG(health_score)                                           AS avg_health_score,
            SUM(CASE WHEN health_tier = 'healthy'  THEN 1 ELSE 0 END)  AS healthy_count,
            SUM(CASE WHEN health_tier = 'monitor'  THEN 1 ELSE 0 END)  AS monitor_count,
            SUM(CASE WHEN health_tier = 'at_risk'  THEN 1 ELSE 0 END)  AS at_risk_count,
            SUM(CASE WHEN health_tier = 'critical' THEN 1 ELSE 0 END)  AS critical_count,
            AVG(score_ss_coverage)                                      AS avg_score_ss,
            AVG(score_dos_target)                                       AS avg_score_dos,
            AVG(score_stockout_risk)                                    AS avg_score_stockout,
            AVG(score_forecast_accuracy)                                AS avg_score_forecast
        FROM mv_inventory_health_score t
        {where_sql}
    """

    histogram_sql = f"""
        SELECT
            CASE
                WHEN health_score < 20  THEN '0-19'
                WHEN health_score < 40  THEN '20-39'
                WHEN health_score < 60  THEN '40-59'
                WHEN health_score < 80  THEN '60-79'
                ELSE '80-100'
            END AS bucket,
            COUNT(*) AS count
        FROM mv_inventory_health_score t
        {where_sql}
        GROUP BY 1
        ORDER BY 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, params)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            summary = dict(zip(cols, row)) if row else {}

            cur.execute(histogram_sql, params)
            hist_rows = cur.fetchall()

    total_dfus = int(summary.get("total_dfus") or 0)

    def _pct(n):
        v = summary.get(n)
        return int(v) if v is not None else 0

    by_tier = {
        "healthy":  _pct("healthy_count"),
        "monitor":  _pct("monitor_count"),
        "at_risk":  _pct("at_risk_count"),
        "critical": _pct("critical_count"),
    }

    def _fval(n):
        v = summary.get(n)
        return round(float(v), 2) if v is not None else None

    score_histogram = [{"bucket": r[0], "count": int(r[1])} for r in hist_rows]

    return {
        "total_dfus":      total_dfus,
        "by_tier":         by_tier,
        "avg_health_score": _fval("avg_health_score"),
        "component_avgs": {
            "ss_coverage":       _fval("avg_score_ss"),
            "dos_target":        _fval("avg_score_dos"),
            "stockout_risk":     _fval("avg_score_stockout"),
            "forecast_accuracy": _fval("avg_score_forecast"),
        },
        "score_histogram": score_histogram,
    }


@router.get("/inv-planning/health/detail")
def get_health_detail(
    item:               Optional[str] = None,
    location:           Optional[str] = None,
    health_tier:        Optional[str] = None,
    abc_vol:            Optional[str] = None,
    cluster_assignment: Optional[str] = None,
    variability_class:  Optional[str] = None,
    brand:              Optional[str] = None,
    category:           Optional[str] = None,
    market:             Optional[str] = None,
    limit:  int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by:  str = Query("health_score"),
    sort_dir: str = Query("asc"),
):
    """Paginated health score detail rows with filtering and sorting."""
    allowed_sort = {
        "health_score", "health_tier", "item_no", "loc", "abc_vol",
        "variability_class", "current_dos", "recent_wape",
        "score_ss_coverage", "score_dos_target", "score_stockout_risk",
        "score_forecast_accuracy",
    }
    if sort_by not in allowed_sort:
        sort_by = "health_score"
    sort_dir = "DESC" if sort_dir.upper() == "DESC" else "ASC"

    where_clauses: list[str] = []
    params: list = []

    if item:
        where_clauses.append("item_no ILIKE %s")
        params.append(f"%{item}%")
    if location:
        where_clauses.append("loc ILIKE %s")
        params.append(f"%{location}%")
    if health_tier:
        where_clauses.append("health_tier = %s")
        params.append(health_tier)
    if abc_vol:
        where_clauses.append("abc_vol = %s")
        params.append(abc_vol)
    if cluster_assignment:
        where_clauses.append("cluster_assignment = %s")
        params.append(cluster_assignment)
    if variability_class:
        where_clauses.append("variability_class = %s")
        params.append(variability_class)
    if brand:
        params.append(brand.split(","))
        where_clauses.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        where_clauses.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        where_clauses.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = t.loc AND dl.state_id = ANY(%s))")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM mv_inventory_health_score t {where_sql}"
    rows_sql = f"""
        SELECT
            item_no, loc, abc_vol, variability_class, cluster_assignment,
            health_score, health_tier,
            score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy,
            ss_coverage, current_dos, target_dos_min, target_dos_max, is_below_ss,
            recent_wape, stockout_count_3m
        FROM mv_inventory_health_score t
        {where_sql}
        ORDER BY {sort_by} {sort_dir}
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0] or 0

            cur.execute(rows_sql, params + [limit, offset])
            raw_rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    def _coerce(k, v):
        if v is None:
            return None
        if k in ("health_score", "score_ss_coverage", "score_dos_target",
                  "score_stockout_risk", "score_forecast_accuracy", "stockout_count_3m"):
            return int(v)
        if k in ("ss_coverage", "current_dos", "target_dos_min", "target_dos_max", "recent_wape"):
            return round(float(v), 4)
        return v

    rows = [
        {k: _coerce(k, v) for k, v in zip(cols, r)}
        for r in raw_rows
    ]

    return {"total": int(total), "rows": rows}


@router.get("/inv-planning/health/heatmap")
def get_health_heatmap(
    group_x: str = Query("abc_vol", description="Column for X-axis grouping"),
    group_y: str = Query("variability_class", description="Column for Y-axis grouping"),
):
    """Average health score heatmap grouped by two dimensions."""
    allowed_groups = {"abc_vol", "variability_class", "cluster_assignment", "region", "health_tier"}
    if group_x not in allowed_groups:
        group_x = "abc_vol"
    if group_y not in allowed_groups:
        group_y = "variability_class"

    sql = f"""
        SELECT
            COALESCE({group_x}::TEXT, 'Unknown') AS x_label,
            COALESCE({group_y}::TEXT, 'Unknown') AS y_label,
            AVG(health_score)                     AS avg_health_score,
            COUNT(*)                              AS count,
            SUM(CASE WHEN health_tier = 'critical' THEN 1 ELSE 0 END) AS critical_count
        FROM mv_inventory_health_score
        GROUP BY 1, 2
        ORDER BY 1, 2
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            raw_rows = cur.fetchall()

    cells = [
        {
            "x":                r[0],
            "y":                r[1],
            "avg_health_score": round(float(r[2]), 2) if r[2] is not None else None,
            "count":            int(r[3]),
            "critical_count":   int(r[4]),
        }
        for r in raw_rows
    ]

    x_labels = sorted(set(c["x"] for c in cells))
    y_labels = sorted(set(c["y"] for c in cells))

    return {"x_labels": x_labels, "y_labels": y_labels, "cells": cells}
