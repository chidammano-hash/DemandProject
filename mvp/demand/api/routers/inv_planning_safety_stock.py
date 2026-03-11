"""Inventory Planning — IPfeature3: Safety Stock Engine endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import _f, _s, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])


class SafetyStockOverrideBody(BaseModel):
    item_no: str
    loc: str
    ss_override_qty: float
    reason: str




@router.get("/inv-planning/safety-stock/summary")
def get_ss_summary(
    response: FastAPIResponse,
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    cluster_assignment: Optional[str] = Query(None, max_length=120),
    policy_version: str = Query("v1", max_length=20),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
) -> dict:
    """Portfolio-level safety stock summary with by-class breakdown and top gaps.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    where_parts: list[str] = ["s.policy_version = %s"]
    params: list = [policy_version]

    if item:
        where_parts.append("s.item_no ILIKE %s")
        params.append(f"%{item}%")
    if location:
        where_parts.append("s.loc ILIKE %s")
        params.append(f"%{location}%")
    if abc_vol:
        where_parts.append("s.abc_vol = %s")
        params.append(abc_vol.strip().upper())
    if cluster_assignment:
        where_parts.append("d.cluster_assignment ILIKE %s")
        params.append(f"%{cluster_assignment.strip()}%")
    if brand:
        params.append(brand.split(","))
        where_parts.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = s.item_no AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        where_parts.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = s.item_no AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        where_parts.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = s.loc AND dl.state_id = ANY(%s))")

    where_sql = "WHERE " + " AND ".join(where_parts)

    summary_sql = f"""
        SELECT
            COUNT(*)                                                AS total_dfus,
            SUM(CASE WHEN s.is_below_ss THEN 1 ELSE 0 END)        AS below_ss_count,
            AVG(s.ss_coverage)                                     AS avg_ss_coverage,
            AVG(s.target_dos_min)                                  AS avg_ss_days,
            SUM(CASE WHEN s.ss_gap < 0 THEN s.ss_gap ELSE 0 END)  AS total_ss_gap_units
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_dfu d
               ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
    """

    by_class_sql = f"""
        SELECT
            s.abc_vol,
            COUNT(*)                                             AS total,
            SUM(CASE WHEN s.is_below_ss THEN 1 ELSE 0 END)     AS below_ss_count,
            AVG(s.ss_combined)                                   AS avg_ss_combined,
            AVG(s.ss_coverage)                                   AS avg_coverage
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_dfu d
               ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
        GROUP BY s.abc_vol
        ORDER BY s.abc_vol NULLS LAST
    """

    top_gaps_sql = f"""
        SELECT
            s.item_no,
            s.loc,
            s.ss_combined,
            s.current_qty_on_hand,
            s.ss_gap,
            s.ss_coverage
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_dfu d
               ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
          AND s.ss_gap < 0
        ORDER BY s.ss_gap ASC NULLS LAST
        LIMIT 10
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, params)
            row = cur.fetchone()
            # Positional access — column order matches SELECT list:
            # 0: total_dfus, 1: below_ss_count, 2: avg_ss_coverage,
            # 3: avg_ss_days, 4: total_ss_gap_units
            summary_row = row if row else (0, 0, None, None, None)

            cur.execute(by_class_sql, params)
            class_rows = cur.fetchall()
            # Column order: 0: abc_vol, 1: total, 2: below_ss_count,
            #               3: avg_ss_combined, 4: avg_coverage

            cur.execute(top_gaps_sql, params)
            gap_rows = cur.fetchall()

    total_dfus = int(summary_row[0] or 0)
    below_ss = int(summary_row[1] or 0)
    below_ss_pct = round(below_ss / total_dfus * 100, 2) if total_dfus else 0.0

    by_class: dict[str, dict] = {}
    for r in class_rows:
        key = r[0] or "unknown"
        by_class[key] = {
            "count":          int(r[1] or 0),
            "below_ss_count": int(r[2] or 0),
            "avg_ss_combined": _f(r[3]),
            "avg_coverage":   _f(r[4]),
        }

    top_gaps = [
        {
            "item_no":       r[0],
            "loc":           r[1],
            "ss_combined":   _f(r[2]),
            "current_qty":   _f(r[3]),
            "ss_gap":        _f(r[4]),
            "ss_coverage":   _f(r[5]),
        }
        for r in gap_rows
    ]

    return {
        "total_dfus":         total_dfus,
        "below_ss_count":     below_ss,
        "below_ss_pct":       below_ss_pct,
        "total_ss_gap_units": _f(summary_row[4]),
        "avg_ss_coverage":    _f(summary_row[2]),
        "avg_ss_days":        _f(summary_row[3]),
        "by_class":           by_class,
        "top_gaps":           top_gaps,
    }


@router.get("/inv-planning/safety-stock/detail")
def get_ss_detail(
    response: FastAPIResponse,
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    is_below_ss: Optional[bool] = Query(None),
    cluster_assignment: Optional[str] = Query(None, max_length=120),
    policy_version: str = Query("v1", max_length=20),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("ss_gap", max_length=40),
    sort_dir: str = Query("asc", max_length=4),
) -> dict:
    """Paginated safety stock detail table.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    allowed_sort = {
        "ss_gap", "ss_coverage", "ss_combined", "reorder_point", "target_dos_min",
        "item_no", "loc", "abc_vol",
    }
    order_col = sort_by if sort_by in allowed_sort else "ss_gap"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = ["s.policy_version = %s"]
    params: list = [policy_version]

    if item:
        where_parts.append("s.item_no ILIKE %s")
        params.append(f"%{item}%")
    if location:
        where_parts.append("s.loc ILIKE %s")
        params.append(f"%{location}%")
    if abc_vol:
        where_parts.append("s.abc_vol = %s")
        params.append(abc_vol.strip().upper())
    if is_below_ss is not None:
        where_parts.append("s.is_below_ss = %s")
        params.append(is_below_ss)
    if cluster_assignment:
        where_parts.append("d.cluster_assignment ILIKE %s")
        params.append(f"%{cluster_assignment.strip()}%")
    if brand:
        params.append(brand.split(","))
        where_parts.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = s.item_no AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        where_parts.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = s.item_no AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        where_parts.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = s.loc AND dl.state_id = ANY(%s))")

    where_sql = "WHERE " + " AND ".join(where_parts)

    count_sql = f"""
        SELECT COUNT(*)
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_dfu d ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
    """

    data_params = params + [limit, offset]
    data_sql = f"""
        SELECT
            s.item_no, s.loc, s.abc_vol,
            s.service_level_target, s.z_score,
            s.ss_combined, s.reorder_point,
            s.current_qty_on_hand, s.current_dos,
            s.ss_gap, s.ss_coverage, s.is_below_ss,
            s.target_dos_min
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_dfu d ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0] or 0
            cur.execute(data_sql, data_params)
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "item_no":              r[0],
                "loc":                  r[1],
                "abc_vol":              r[2],
                "service_level_target": _f(r[3]),
                "z_score":              _f(r[4]),
                "ss_combined":          _f(r[5]),
                "reorder_point":        _f(r[6]),
                "current_qty_on_hand":  _f(r[7]),
                "current_dos":          _f(r[8]),
                "ss_gap":               _f(r[9]),
                "ss_coverage":          _f(r[10]),
                "is_below_ss":          bool(r[11]) if r[11] is not None else None,
                "target_dos_min":       _f(r[12]),
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/safety-stock/waterfall")
def get_ss_waterfall(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
    policy_version: str = Query("v1", max_length=20),
) -> dict:
    """Safety stock waterfall decomposition for a single item+location pair.

    Returns demand component, LT component, combined SS, ROP, and current position.
    Cache: 120s.
    """
    set_cache(response, max_age=120)

    sql = """
        SELECT
            item_no, loc,
            ss_demand_only, ss_lt_only, ss_combined,
            reorder_point,
            current_qty_on_hand,
            ss_gap,
            z_score, service_level_target,
            lead_time_mean_days, lead_time_std_days,
            demand_mean_monthly, demand_std_monthly
        FROM fact_safety_stock_targets
        WHERE item_no = %s
          AND loc = %s
          AND policy_version = %s
        LIMIT 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location, policy_version])
            row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No safety stock record found for item={item} loc={location} version={policy_version}",
        )

    return {
        "item_no":               row[0],
        "loc":                   row[1],
        "demand_component":      _f(row[2]),
        "lt_component":          _f(row[3]),
        "combined_ss":           _f(row[4]),
        "reorder_point":         _f(row[5]),
        "current_on_hand":       _f(row[6]),
        "ss_gap":                _f(row[7]),
        "z_score":               _f(row[8]),
        "service_level_target":  _f(row[9]),
        "lt_mean_days":          _f(row[10]),
        "lt_std_days":           _f(row[11]),
        "demand_mean_monthly":   _f(row[12]),
        "demand_std_monthly":    _f(row[13]),
    }


@router.post("/inv-planning/safety-stock/override", status_code=201)
def override_safety_stock(
    body: SafetyStockOverrideBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Manually override the ss_combined value for a specific item+location.

    Sets ss_method = 'manual' and recalculates is_below_ss / ss_coverage / ss_gap
    based on the new override quantity.
    Auth required.
    """
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)

    sql = """
        UPDATE fact_safety_stock_targets
        SET
            ss_combined   = %s,
            ss_method     = 'manual',
            ss_gap        = current_qty_on_hand - %s,
            is_below_ss   = current_qty_on_hand < %s,
            ss_coverage   = CASE WHEN %s > 0
                                 THEN current_qty_on_hand / %s
                                 ELSE NULL
                            END,
            modified_ts   = %s
        WHERE item_no = %s
          AND loc     = %s
        RETURNING
            item_no, loc, ss_combined, ss_method, modified_ts
    """

    qty = body.ss_override_qty
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [qty, qty, qty, qty, qty, now, body.item_no, body.loc])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No safety stock record found for item={body.item_no} loc={body.loc}",
        )

    return {
        "item_no":     row[0],
        "loc":         row[1],
        "ss_combined": float(row[2]) if row[2] is not None else None,
        "ss_method":   row[3],
        "modified_ts": row[4].isoformat() if row[4] else None,
    }


@router.get("/inv-planning/safety-stock/config")
def get_ss_config(
    response: FastAPIResponse,
) -> dict:
    """Return the current safety_stock_config.yaml as JSON.

    Cache: 600s.
    """
    import os as _os

    set_cache(response, max_age=600)

    config_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
        "config",
        "safety_stock_config.yaml",
    )

    try:
        import yaml as _yaml

        with open(config_path) as fh:
            cfg = _yaml.safe_load(fh)
        return cfg
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="safety_stock_config.yaml not found")
