"""Inventory Planning — Forward-Looking Replenishment Plan endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])

_DETAIL_SORT_COLS = {
    "item_id", "loc", "abc_vol", "policy_type",
    "forecast_qty", "ss_combined", "historical_ss", "ss_delta", "ss_delta_pct",
    "eoq", "cycle_stock", "reorder_point", "order_qty", "is_below_ss", "plan_month",
}


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/summary
# ---------------------------------------------------------------------------

@router.get("/inv-planning/replenishment/summary")
def get_replenishment_summary(
    response: FastAPIResponse,
    plan_version: Optional[str] = Query(None, max_length=40),
    policy_type: Optional[str] = Query(None, max_length=80),
    abc_vol: Optional[str] = Query(None, max_length=10),
) -> dict:
    """Portfolio-level replenishment plan summary with by-policy-type breakdown.

    If plan_version is not provided, uses the latest available version.
    Cache: 120s.
    """
    set_cache(response, max_age=120)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve plan_version to latest if not provided
            if not plan_version:
                cur.execute(
                    "SELECT DISTINCT plan_version FROM fact_replenishment_plan "
                    "ORDER BY plan_version DESC LIMIT 1"
                )
                ver_row = cur.fetchone()
                resolved_version: str = ver_row[0] if ver_row else ""
            else:
                resolved_version = plan_version

            if not resolved_version:
                return {
                    "plan_version": None,
                    "total_dfus": 0,
                    "below_ss_count": 0,
                    "below_ss_pct": 0.0,
                    "avg_ss": None,
                    "avg_eoq": None,
                    "avg_ss_delta_pct": None,
                    "by_policy_type": [],
                }

            # Build summary WHERE clause
            summary_where_parts: list[str] = [
                "plan_version = %s",
                "plan_month = (SELECT MIN(plan_month) FROM fact_replenishment_plan WHERE plan_version = %s)",
            ]
            summary_params: list = [resolved_version, resolved_version]

            if policy_type:
                summary_where_parts.append("policy_type = %s")
                summary_params.append(policy_type)
            if abc_vol:
                summary_where_parts.append("abc_vol = %s")
                summary_params.append(abc_vol.strip().upper())

            summary_where_sql = "WHERE " + " AND ".join(summary_where_parts)

            summary_sql = f"""
                SELECT
                    COUNT(DISTINCT (item_id, loc))                                    AS total_dfus,
                    COUNT(DISTINCT (item_id, loc)) FILTER (WHERE is_below_ss = TRUE)  AS below_ss_count,
                    AVG(ss_combined)                                                   AS avg_ss,
                    AVG(effective_eoq)                                                 AS avg_eoq,
                    AVG(ss_delta_pct)                                                  AS avg_ss_delta_pct
                FROM fact_replenishment_plan
                {summary_where_sql}
            """

            cur.execute(summary_sql, summary_params)
            summary_row = cur.fetchone()
            # Column order: 0: total_dfus, 1: below_ss_count, 2: avg_ss,
            #               3: avg_eoq, 4: avg_ss_delta_pct

            # Breakdown by policy_type (no policy_type or abc_vol filter applied here)
            policy_sql = """
                SELECT policy_type,
                       COUNT(DISTINCT (item_id, loc)) AS dfu_count,
                       AVG(ss_combined)               AS avg_ss,
                       AVG(effective_eoq)             AS avg_eoq,
                       SUM(order_qty)                 AS total_order_qty
                FROM fact_replenishment_plan
                WHERE plan_version = %s
                  AND plan_month = (SELECT MIN(plan_month) FROM fact_replenishment_plan WHERE plan_version = %s)
                GROUP BY policy_type
                ORDER BY dfu_count DESC
            """
            cur.execute(policy_sql, [resolved_version, resolved_version])
            policy_rows = cur.fetchall()
            # Column order: 0: policy_type, 1: dfu_count, 2: avg_ss,
            #               3: avg_eoq, 4: total_order_qty

    if not summary_row:
        summary_row = (0, 0, None, None, None)

    total_dfus = int(summary_row[0] or 0)
    below_ss_count = int(summary_row[1] or 0)
    below_ss_pct = round(below_ss_count / total_dfus * 100, 2) if total_dfus else 0.0

    by_policy_type = [
        {
            "policy_type":      r[0],
            "dfu_count":        int(r[1] or 0),
            "avg_ss":           _f(r[2]),
            "avg_eoq":          _f(r[3]),
            "total_order_qty":  _f(r[4]),
        }
        for r in policy_rows
    ]

    return {
        "plan_version":     resolved_version,
        "total_dfus":       total_dfus,
        "below_ss_count":   below_ss_count,
        "below_ss_pct":     below_ss_pct,
        "avg_ss":           _f(summary_row[2]),
        "avg_eoq":          _f(summary_row[3]),
        "avg_ss_delta_pct": _f(summary_row[4]),
        "by_policy_type":   by_policy_type,
    }


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/detail
# ---------------------------------------------------------------------------

@router.get("/inv-planning/replenishment/detail")
def get_replenishment_detail(
    response: FastAPIResponse,
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    policy_type: Optional[str] = Query(None, max_length=80),
    abc_vol: Optional[str] = Query(None, max_length=10),
    is_below_ss: Optional[bool] = Query(None),
    plan_version: Optional[str] = Query(None, max_length=40),
    plan_month: Optional[str] = Query(None, max_length=20),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("item_id", max_length=40),
    sort_dir: str = Query("asc", max_length=4),
) -> dict:
    """Paginated replenishment plan detail table.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    order_col = sort_by if sort_by in _DETAIL_SORT_COLS else "item_id"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list = []

    if plan_version:
        where_parts.append("plan_version = %s")
        params.append(plan_version)
    if item:
        where_parts.append("item_id ILIKE %s")
        params.append(f"%{item}%")
    if location:
        where_parts.append("loc ILIKE %s")
        params.append(f"%{location}%")
    if policy_type:
        where_parts.append("policy_type = %s")
        params.append(policy_type)
    if abc_vol:
        where_parts.append("abc_vol = %s")
        params.append(abc_vol.strip().upper())
    if is_below_ss is not None:
        where_parts.append("is_below_ss = %s")
        params.append(is_below_ss)
    if plan_month:
        where_parts.append("plan_month = %s::date")
        params.append(plan_month)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM fact_replenishment_plan {where_sql}"

    data_params = params + [limit, offset]
    data_sql = f"""
        SELECT
            item_id, loc, plan_month, abc_vol, policy_type,
            forecast_qty, ss_combined, historical_ss,
            ss_delta, ss_delta_pct,
            eoq, cycle_stock, reorder_point, order_qty, order_up_to_level,
            is_below_ss
        FROM fact_replenishment_plan
        {where_sql}
        ORDER BY "{order_col}" {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """
    # Column order: 0: item_id, 1: loc, 2: plan_month, 3: abc_vol, 4: policy_type,
    #               5: forecast_qty, 6: ss_combined, 7: historical_ss,
    #               8: ss_delta, 9: ss_delta_pct,
    #               10: eoq, 11: cycle_stock, 12: reorder_point,
    #               13: order_qty, 14: order_up_to_level, 15: is_below_ss

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = int((cur.fetchone() or (0,))[0])
            cur.execute(data_sql, data_params)
            rows = cur.fetchall()

    return {
        "total":  total,
        "limit":  limit,
        "offset": offset,
        "rows": [
            {
                "item_id":           r[0],
                "loc":               r[1],
                "plan_month":        r[2].isoformat() if r[2] else None,
                "abc_vol":           r[3],
                "policy_type":       r[4],
                "forecast_qty":      _f(r[5]),
                "ss_combined":       _f(r[6]),
                "historical_ss":     _f(r[7]),
                "ss_delta":          _f(r[8]),
                "ss_delta_pct":      _f(r[9]),
                "eoq":               _f(r[10]),
                "cycle_stock":       _f(r[11]),
                "reorder_point":     _f(r[12]),
                "order_qty":         _f(r[13]),
                "order_up_to_level": _f(r[14]),
                "is_below_ss":       bool(r[15]) if r[15] is not None else None,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/comparison
# ---------------------------------------------------------------------------

@router.get("/inv-planning/replenishment/comparison")
def get_replenishment_comparison(
    response: FastAPIResponse,
    plan_version: Optional[str] = Query(None, max_length=40),
    abc_vol: Optional[str] = Query(None, max_length=10),
    policy_type: Optional[str] = Query(None, max_length=80),
) -> dict:
    """Safety stock comparison: forecast SS vs historical SS by ABC class.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve latest plan_version if not provided
            if not plan_version:
                cur.execute(
                    "SELECT DISTINCT plan_version FROM fact_replenishment_plan "
                    "ORDER BY plan_version DESC LIMIT 1"
                )
                ver_row = cur.fetchone()
                resolved_version = ver_row[0] if ver_row else ""
            else:
                resolved_version = plan_version

            if not resolved_version:
                return {
                    "plan_version": None,
                    "by_abc": [],
                    "total_increased": 0,
                    "total_decreased": 0,
                }

            extra_parts: list[str] = []
            extra_params: list = []
            if abc_vol:
                extra_parts.append("abc_vol = %s")
                extra_params.append(abc_vol.strip().upper())
            if policy_type:
                extra_parts.append("policy_type = %s")
                extra_params.append(policy_type)

            extra_where = (" AND " + " AND ".join(extra_parts)) if extra_parts else ""

            comparison_sql = f"""
                SELECT abc_vol,
                       COUNT(DISTINCT (item_id, loc))              AS dfu_count,
                       AVG(ss_combined)                            AS avg_forecast_ss,
                       AVG(historical_ss)                          AS avg_historical_ss,
                       AVG(ss_delta)                               AS avg_ss_delta,
                       AVG(ss_delta_pct)                           AS avg_ss_delta_pct,
                       COUNT(*) FILTER (WHERE ss_delta > 0)        AS count_increased,
                       COUNT(*) FILTER (WHERE ss_delta < 0)        AS count_decreased,
                       COUNT(*) FILTER (WHERE ss_delta = 0 OR ss_delta IS NULL) AS count_unchanged
                FROM fact_replenishment_plan
                WHERE plan_version = %s
                  AND plan_month = (SELECT MIN(plan_month) FROM fact_replenishment_plan WHERE plan_version = %s)
                  AND historical_ss IS NOT NULL
                  {extra_where}
                GROUP BY abc_vol
                ORDER BY abc_vol
            """
            # Column order: 0: abc_vol, 1: dfu_count, 2: avg_forecast_ss,
            #               3: avg_historical_ss, 4: avg_ss_delta, 5: avg_ss_delta_pct,
            #               6: count_increased, 7: count_decreased, 8: count_unchanged

            comparison_params = [resolved_version, resolved_version] + extra_params
            cur.execute(comparison_sql, comparison_params)
            rows = cur.fetchall()

    by_abc = [
        {
            "abc_vol":          r[0],
            "dfu_count":        int(r[1] or 0),
            "avg_forecast_ss":  _f(r[2]),
            "avg_historical_ss": _f(r[3]),
            "avg_ss_delta":     _f(r[4]),
            "avg_ss_delta_pct": _f(r[5]),
            "count_increased":  int(r[6] or 0),
            "count_decreased":  int(r[7] or 0),
            "count_unchanged":  int(r[8] or 0),
        }
        for r in rows
    ]

    total_increased = sum(r["count_increased"] for r in by_abc)
    total_decreased = sum(r["count_decreased"] for r in by_abc)

    return {
        "plan_version":    resolved_version,
        "by_abc":          by_abc,
        "total_increased": total_increased,
        "total_decreased": total_decreased,
    }


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/dfu
# ---------------------------------------------------------------------------

@router.get("/inv-planning/replenishment/dfu")
def get_replenishment_dfu(
    response: FastAPIResponse,
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    plan_version: Optional[str] = Query(None, max_length=40),
) -> dict:
    """Replenishment plan time series for a single item+location pair.

    Returns 404 if no rows are found.
    Cache: 120s.
    """
    set_cache(response, max_age=120)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve latest plan_version if not provided
            if not plan_version:
                cur.execute(
                    "SELECT DISTINCT plan_version FROM fact_replenishment_plan "
                    "WHERE item_id = %s AND loc = %s "
                    "ORDER BY plan_version DESC LIMIT 1",
                    [item_id, loc],
                )
                ver_row = cur.fetchone()
                resolved_version = ver_row[0] if ver_row else ""
            else:
                resolved_version = plan_version

            if not resolved_version:
                raise HTTPException(
                    status_code=404,
                    detail=f"No replenishment plan found for item_id={item_id} loc={loc}",
                )

            series_sql = """
                SELECT plan_month, horizon_months,
                       forecast_qty, forecast_qty_lower, forecast_qty_upper,
                       ss_combined, historical_ss, ss_delta,
                       eoq, cycle_stock,
                       reorder_point, order_qty, order_up_to_level,
                       avg_daily_demand, is_below_ss, sigma_method
                FROM fact_replenishment_plan
                WHERE plan_version = %s AND item_id = %s AND loc = %s
                ORDER BY plan_month
            """
            # Column order:  0: plan_month, 1: horizon_months,
            #                2: forecast_qty, 3: forecast_qty_lower, 4: forecast_qty_upper,
            #                5: ss_combined, 6: historical_ss, 7: ss_delta,
            #                8: eoq, 9: cycle_stock,
            #                10: reorder_point, 11: order_qty, 12: order_up_to_level,
            #                13: avg_daily_demand, 14: is_below_ss, 15: sigma_method

            cur.execute(series_sql, [resolved_version, item_id, loc])
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No replenishment plan found for item_id={item_id} loc={loc} plan_version={resolved_version}",
        )

    series = [
        {
            "plan_month":          r[0].isoformat() if r[0] else None,
            "horizon_months":      int(r[1]) if r[1] is not None else None,
            "forecast_qty":        _f(r[2]),
            "forecast_qty_lower":  _f(r[3]),
            "forecast_qty_upper":  _f(r[4]),
            "ss_combined":         _f(r[5]),
            "historical_ss":       _f(r[6]),
            "ss_delta":            _f(r[7]),
            "eoq":                 _f(r[8]),
            "cycle_stock":         _f(r[9]),
            "reorder_point":       _f(r[10]),
            "order_qty":           _f(r[11]),
            "order_up_to_level":   _f(r[12]),
            "avg_daily_demand":    _f(r[13]),
            "is_below_ss":         bool(r[14]) if r[14] is not None else None,
            "sigma_method":        r[15],
        }
        for r in rows
    ]

    return {
        "item_id":      item_id,
        "loc":          loc,
        "plan_version": resolved_version,
        "series":       series,
    }
