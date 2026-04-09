"""Inventory Planning — Integrated Planning Targets (SS + EOQ + ROP).

Unified view of safety stock, EOQ, and reorder point targets with
cost-benefit metrics. Reads from mv_integrated_planning_targets.
"""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, add_cross_dim_filters, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])

_DETAIL_SORT_COLS = {
    "item_id", "loc", "abc_vol",
    "safety_stock_qty", "reorder_point", "effective_eoq",
    "target_avg_inventory", "monthly_total_holding_cost",
    "monthly_total_cost", "ss_gap", "stockout_risk_score",
    "excess_qty", "excess_value_usd", "excess_risk_score",
}


# ---------------------------------------------------------------------------
# GET /inv-planning/integrated-targets/summary
# ---------------------------------------------------------------------------

@router.get("/inv-planning/integrated-targets/summary")
def get_integrated_targets_summary(
    response: FastAPIResponse,
    abc_vol: str | None = Query(None, max_length=10),
    brand: str | None = Query(None, max_length=120),
    category: str | None = Query(None, max_length=120),
    market: str | None = Query(None, max_length=120),
) -> dict:
    """Portfolio-level aggregates of integrated planning targets.

    Returns total SKUs, total monthly holding cost, avg SS, avg EOQ,
    items below SS, avg target inventory, and by-ABC breakdown.
    Cache: 120s.
    """
    set_cache(response, max_age=120)

    where_parts: list[str] = []
    params: list = []

    if abc_vol:
        where_parts.append("t.abc_vol = %s")
        params.append(abc_vol.strip().upper())
    add_cross_dim_filters(
        where_parts, params,
        brand=brand, category=category, market=market,
        item_col="t.item_id", loc_col="t.loc",
    )

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    summary_sql = f"""
        SELECT
            COUNT(*)                                              AS total_skus,
            SUM(CASE WHEN is_below_ss THEN 1 ELSE 0 END)        AS below_ss_count,
            AVG(safety_stock_qty)                                 AS avg_safety_stock,
            AVG(effective_eoq)                                    AS avg_eoq,
            AVG(target_avg_inventory)                             AS avg_target_inventory,
            SUM(monthly_total_holding_cost)                       AS total_monthly_holding_cost,
            SUM(monthly_ordering_cost)                            AS total_monthly_ordering_cost,
            SUM(monthly_total_cost)                               AS total_monthly_cost,
            AVG(stockout_risk_score)                              AS avg_risk_score,
            SUM(CASE WHEN stockout_risk_score >= 60 THEN 1 ELSE 0 END) AS high_risk_count,
            SUM(CASE WHEN stockout_risk_score >= 80 THEN 1 ELSE 0 END) AS critical_risk_count,
            SUM(excess_value_usd)                                 AS total_excess_value_usd,
            SUM(excess_holding_cost_monthly)                      AS total_excess_holding_cost_monthly,
            SUM(CASE WHEN excess_qty > 0 THEN 1 ELSE 0 END)     AS excess_sku_count,
            AVG(excess_risk_score)                                AS avg_excess_risk_score
        FROM mv_integrated_planning_targets t
        {where_sql}
    """

    abc_sql = f"""
        SELECT
            COALESCE(t.abc_vol, 'Unknown') AS abc_vol,
            COUNT(*)                       AS count,
            AVG(safety_stock_qty)          AS avg_ss,
            AVG(effective_eoq)             AS avg_eoq,
            AVG(target_avg_inventory)      AS avg_target_inv,
            SUM(monthly_total_holding_cost) AS total_holding_cost,
            SUM(monthly_total_cost)        AS total_cost,
            SUM(CASE WHEN is_below_ss THEN 1 ELSE 0 END) AS below_ss_count
        FROM mv_integrated_planning_targets t
        {where_sql}
        GROUP BY t.abc_vol
        ORDER BY t.abc_vol NULLS LAST
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, params)
            summary_row = cur.fetchone()
            cur.execute(abc_sql, params)
            abc_rows = cur.fetchall()

    if not summary_row or summary_row[0] == 0:
        return {
            "total_skus": 0,
            "below_ss_count": 0,
            "below_ss_pct": 0.0,
            "avg_safety_stock": None,
            "avg_eoq": None,
            "avg_target_inventory": None,
            "total_monthly_holding_cost": None,
            "total_monthly_ordering_cost": None,
            "total_monthly_cost": None,
            "avg_risk_score": None,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "total_excess_value_usd": None,
            "total_excess_holding_cost_monthly": None,
            "excess_sku_count": 0,
            "avg_excess_risk_score": None,
            "by_abc": [],
        }

    total_skus = int(summary_row[0])
    below_ss = int(summary_row[1] or 0)
    below_ss_pct = round(below_ss / total_skus * 100, 2) if total_skus else 0.0

    by_abc = [
        {
            "abc_vol": r[0],
            "count": int(r[1]),
            "avg_safety_stock": _f(r[2]),
            "avg_eoq": _f(r[3]),
            "avg_target_inventory": _f(r[4]),
            "total_holding_cost": _f(r[5]),
            "total_cost": _f(r[6]),
            "below_ss_count": int(r[7] or 0),
        }
        for r in abc_rows
    ]

    return {
        "total_skus": total_skus,
        "below_ss_count": below_ss,
        "below_ss_pct": below_ss_pct,
        "avg_safety_stock": _f(summary_row[2]),
        "avg_eoq": _f(summary_row[3]),
        "avg_target_inventory": _f(summary_row[4]),
        "total_monthly_holding_cost": _f(summary_row[5]),
        "total_monthly_ordering_cost": _f(summary_row[6]),
        "total_monthly_cost": _f(summary_row[7]),
        "avg_risk_score": _f(summary_row[8]),
        "high_risk_count": int(summary_row[9] or 0),
        "critical_risk_count": int(summary_row[10] or 0),
        "total_excess_value_usd": _f(summary_row[11]),
        "total_excess_holding_cost_monthly": _f(summary_row[12]),
        "excess_sku_count": int(summary_row[13] or 0),
        "avg_excess_risk_score": _f(summary_row[14]),
        "by_abc": by_abc,
    }


# ---------------------------------------------------------------------------
# GET /inv-planning/integrated-targets
# ---------------------------------------------------------------------------

@router.get("/inv-planning/integrated-targets")
def get_integrated_targets(
    response: FastAPIResponse,
    item_id: str | None = Query(None, max_length=120),
    loc: str | None = Query(None, max_length=120),
    abc_vol: str | None = Query(None, max_length=10),
    below_ss_only: bool = Query(False),
    brand: str | None = Query(None, max_length=120),
    category: str | None = Query(None, max_length=120),
    market: str | None = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("stockout_risk_score", max_length=40),
    sort_dir: str = Query("desc", max_length=4),
) -> dict:
    """Return unified SS + EOQ + ROP targets with cost metrics.

    Paginated detail from mv_integrated_planning_targets.
    Cache: 120s.
    """
    set_cache(response, max_age=120)

    order_col = sort_by if sort_by in _DETAIL_SORT_COLS else "stockout_risk_score"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list = []

    if item_id:
        where_parts.append("t.item_id ILIKE %s")
        params.append(f"%{item_id}%")
    if loc:
        where_parts.append("t.loc ILIKE %s")
        params.append(f"%{loc}%")
    if abc_vol:
        where_parts.append("t.abc_vol = %s")
        params.append(abc_vol.strip().upper())
    if below_ss_only:
        where_parts.append("t.is_below_ss = TRUE")
    add_cross_dim_filters(
        where_parts, params,
        brand=brand, category=category, market=market,
        item_col="t.item_id", loc_col="t.loc",
    )

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM mv_integrated_planning_targets t {where_sql}"

    data_params = [*params, limit, offset]
    data_sql = f"""
        SELECT
            t.item_id, t.loc, t.abc_vol, t.abc_xyz_segment,
            t.safety_stock_qty, t.reorder_point,
            t.service_level_target,
            t.demand_mean_monthly, t.demand_std_monthly, t.demand_cv,
            t.eoq_qty, t.effective_eoq, t.cycle_stock, t.orders_per_year, t.unit_cost,
            t.target_avg_inventory, t.target_min_inventory, t.target_max_inventory,
            t.current_qty_on_hand, t.current_dos, t.ss_gap, t.is_below_ss,
            t.monthly_ss_holding_cost, t.monthly_cycle_holding_cost,
            t.monthly_total_holding_cost, t.monthly_ordering_cost, t.monthly_total_cost,
            t.lead_time_mean_days, t.lead_time_std_days,
            t.policy_version, t.forecast_source, t.forecast_model_id, t.computed_at,
            t.stockout_risk_score,
            t.excess_qty, t.excess_value_usd, t.excess_holding_cost_monthly,
            t.excess_months_supply, t.excess_risk_score
        FROM mv_integrated_planning_targets t
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """
    # Column indices documented below for maintainability:
    # 0: item_id, 1: loc, 2: abc_vol, 3: abc_xyz_segment,
    # 4: safety_stock_qty, 5: reorder_point, 6: service_level_target,
    # 7: demand_mean_monthly, 8: demand_std_monthly, 9: demand_cv,
    # 10: eoq_qty, 11: effective_eoq, 12: cycle_stock, 13: orders_per_year, 14: unit_cost,
    # 15: target_avg_inventory, 16: target_min_inventory, 17: target_max_inventory,
    # 18: current_qty_on_hand, 19: current_dos, 20: ss_gap, 21: is_below_ss,
    # 22: monthly_ss_holding_cost, 23: monthly_cycle_holding_cost,
    # 24: monthly_total_holding_cost, 25: monthly_ordering_cost, 26: monthly_total_cost,
    # 27: lead_time_mean_days, 28: lead_time_std_days,
    # 29: policy_version, 30: forecast_source, 31: forecast_model_id, 32: computed_at,
    # 33: stockout_risk_score,
    # 34: excess_qty, 35: excess_value_usd, 36: excess_holding_cost_monthly,
    # 37: excess_months_supply, 38: excess_risk_score

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = int((cur.fetchone() or (0,))[0])
            cur.execute(data_sql, data_params)
            rows = cur.fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "item_id":                  r[0],
                "loc":                      r[1],
                "abc_vol":                  r[2],
                "abc_xyz_segment":          r[3],
                "safety_stock_qty":         _f(r[4]),
                "reorder_point":            _f(r[5]),
                "service_level_target":     _f(r[6]),
                "demand_mean_monthly":      _f(r[7]),
                "demand_std_monthly":       _f(r[8]),
                "demand_cv":                _f(r[9]),
                "eoq_qty":                  _f(r[10]),
                "effective_eoq":            _f(r[11]),
                "cycle_stock":              _f(r[12]),
                "orders_per_year":          _f(r[13]),
                "unit_cost":                _f(r[14]),
                "target_avg_inventory":     _f(r[15]),
                "target_min_inventory":     _f(r[16]),
                "target_max_inventory":     _f(r[17]),
                "current_qty_on_hand":      _f(r[18]),
                "current_dos":              _f(r[19]),
                "ss_gap":                   _f(r[20]),
                "is_below_ss":              bool(r[21]) if r[21] is not None else None,
                "monthly_ss_holding_cost":  _f(r[22]),
                "monthly_cycle_holding_cost": _f(r[23]),
                "monthly_total_holding_cost": _f(r[24]),
                "monthly_ordering_cost":    _f(r[25]),
                "monthly_total_cost":       _f(r[26]),
                "lead_time_mean_days":      _f(r[27]),
                "lead_time_std_days":       _f(r[28]),
                "policy_version":           r[29],
                "forecast_source":          r[30],
                "forecast_model_id":        r[31],
                "computed_at":              r[32].isoformat() if r[32] else None,
                "stockout_risk_score":      int(r[33]) if r[33] is not None else None,
                "excess_qty":               _f(r[34]),
                "excess_value_usd":         _f(r[35]),
                "excess_holding_cost_monthly": _f(r[36]),
                "excess_months_supply":     _f(r[37]),
                "excess_risk_score":        int(r[38]) if r[38] is not None else None,
            }
            for r in rows
        ],
    }
