"""Inventory Planning — IPfeature3: Safety Stock Engine endpoints."""
from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import _f, add_cross_dim_filters, get_conn, set_cache
from common.core.utils import load_config

router = APIRouter(tags=["inv-planning"])


class SafetyStockOverrideBody(BaseModel):
    item_id: str
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
        where_parts.append("s.item_id ILIKE %s")
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
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market,
                          item_col="s.item_id", loc_col="s.loc")

    where_sql = "WHERE " + " AND ".join(where_parts)

    # Single CTE-based query: one scan of fact_safety_stock_targets produces
    # summary aggregates, per-class breakdown, and top gaps in one round-trip.
    combined_sql = f"""
        WITH filtered AS (
            SELECT
                s.item_id, s.loc, s.abc_vol,
                s.ss_combined, s.ss_coverage, s.ss_gap,
                s.is_below_ss, s.target_dos_min, s.current_qty_on_hand,
                s.computed_at
            FROM fact_safety_stock_targets s
            LEFT JOIN dim_sku d
                   ON s.item_id = d.item_id AND s.loc = d.loc
            {where_sql}
        ),
        summary AS (
            SELECT
                COUNT(*)                                                AS total_dfus,
                SUM(CASE WHEN is_below_ss THEN 1 ELSE 0 END)          AS below_ss_count,
                -- Median is robust to overstocked outliers (single SKU at 100x coverage
                -- would otherwise drag the mean to absurd values like 4000%).
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ss_coverage) AS avg_ss_coverage,
                AVG(target_dos_min)                                    AS avg_ss_days,
                SUM(CASE WHEN ss_gap < 0 THEN ss_gap ELSE 0 END)      AS total_ss_gap_units,
                MAX(computed_at)                                        AS last_computed_at
            FROM filtered
        ),
        by_class AS (
            SELECT
                abc_vol,
                COUNT(*)                                             AS total,
                SUM(CASE WHEN is_below_ss THEN 1 ELSE 0 END)        AS below_ss_count,
                AVG(ss_combined)                                     AS avg_ss_combined,
                AVG(ss_coverage)                                     AS avg_coverage
            FROM filtered
            GROUP BY abc_vol
            ORDER BY abc_vol NULLS LAST
        ),
        top_gaps AS (
            SELECT
                item_id, loc, ss_combined, current_qty_on_hand,
                ss_gap, ss_coverage
            FROM filtered
            WHERE ss_gap < 0
            ORDER BY ss_gap ASC NULLS LAST
            LIMIT 10
        )
        -- Return all three result sets as tagged rows
        SELECT 'S' AS _tag,
               total_dfus::TEXT, below_ss_count::TEXT,
               avg_ss_coverage::TEXT, avg_ss_days::TEXT,
               total_ss_gap_units::TEXT, last_computed_at::TEXT, NULL, NULL
        FROM summary
        UNION ALL
        SELECT 'C' AS _tag,
               abc_vol, total::TEXT, below_ss_count::TEXT,
               avg_ss_combined::TEXT, avg_coverage::TEXT,
               NULL, NULL, NULL
        FROM by_class
        UNION ALL
        SELECT 'G' AS _tag,
               item_id, loc, ss_combined::TEXT,
               current_qty_on_hand::TEXT, ss_gap::TEXT,
               ss_coverage::TEXT, NULL, NULL
        FROM top_gaps
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(combined_sql, params)
            all_rows = cur.fetchall()

    # Parse tagged rows into summary, class, and gap result sets
    summary_row = (0, 0, None, None, None, None)
    class_rows: list[tuple] = []
    gap_rows: list[tuple] = []

    for row in all_rows:
        tag = row[0]
        if tag == "S":
            summary_row = (
                int(row[1] or 0),
                int(row[2] or 0),
                float(row[3]) if row[3] else None,
                float(row[4]) if row[4] else None,
                float(row[5]) if row[5] else None,
                row[6],  # last_computed_at (ISO string or None)
            )
        elif tag == "C":
            class_rows.append((
                row[1],                                    # abc_vol
                int(row[2] or 0),                          # total
                int(row[3] or 0),                          # below_ss_count
                float(row[4]) if row[4] else None,         # avg_ss_combined
                float(row[5]) if row[5] else None,         # avg_coverage
            ))
        elif tag == "G":
            gap_rows.append((
                row[1],                                    # item_id
                row[2],                                    # loc
                float(row[3]) if row[3] else None,         # ss_combined
                float(row[4]) if row[4] else None,         # current_qty_on_hand
                float(row[5]) if row[5] else None,         # ss_gap
                float(row[6]) if row[6] else None,         # ss_coverage
            ))

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
            "item_id":       r[0],
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
        "total_skus":         total_dfus,
        "below_ss_count":     below_ss,
        "below_ss_pct":       below_ss_pct,
        "total_ss_gap_units": _f(summary_row[4]),
        "avg_ss_coverage":    _f(summary_row[2]),
        "avg_ss_days":        _f(summary_row[3]),
        "computed_at":        summary_row[5],
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
        "item_id", "loc", "abc_vol",
    }
    order_col = sort_by if sort_by in allowed_sort else "ss_gap"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = ["s.policy_version = %s"]
    params: list = [policy_version]

    if item:
        where_parts.append("s.item_id ILIKE %s")
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
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market,
                          item_col="s.item_id", loc_col="s.loc")

    where_sql = "WHERE " + " AND ".join(where_parts)

    count_sql = f"""
        SELECT COUNT(*)
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_sku d ON s.item_id = d.item_id AND s.loc = d.loc
        {where_sql}
    """

    data_params = params + [limit, offset]
    data_sql = f"""
        SELECT
            s.item_id, s.loc, s.abc_vol,
            s.service_level_target, s.z_score,
            s.ss_combined, s.reorder_point,
            s.current_qty_on_hand, s.current_dos,
            s.ss_gap, s.ss_coverage, s.is_below_ss,
            s.target_dos_min
        FROM fact_safety_stock_targets s
        LEFT JOIN dim_sku d ON s.item_id = d.item_id AND s.loc = d.loc
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
                "item_id":              r[0],
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
            item_id, loc,
            ss_demand_only, ss_lt_only, ss_combined,
            reorder_point,
            current_qty_on_hand,
            ss_gap,
            z_score, service_level_target,
            lead_time_mean_days, lead_time_std_days,
            demand_mean_monthly, demand_std_monthly
        FROM fact_safety_stock_targets
        WHERE item_id = %s
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
        "item_id":               row[0],
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


@router.get("/inv-planning/safety-stock/explain")
def explain_safety_stock(
    response: FastAPIResponse,
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    policy_version: str = Query("v1", max_length=20),
) -> dict:
    """Return safety stock decomposition for a single DFU.

    Shows each component with actual values substituted into the formula,
    plus sensitivity analysis (what-if scenarios).
    Cache: 120s.
    """
    set_cache(response, max_age=120)

    sql = """
        SELECT
            item_id, loc,
            abc_vol, abc_xyz_segment,
            service_level_target, z_score,
            demand_mean_monthly, demand_std_monthly, demand_cv,
            lead_time_mean_days, lead_time_std_days,
            ss_demand_only, ss_lt_only, ss_combined,
            reorder_point,
            current_qty_on_hand, current_dos,
            is_below_ss, ss_gap,
            forecast_source, target_dos_min,
            avg_daily_demand
        FROM fact_safety_stock_targets
        WHERE item_id = %s
          AND loc = %s
          AND policy_version = %s
        LIMIT 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item_id, loc, policy_version])
            row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No safety stock record for item={item_id} loc={loc} version={policy_version}",
        )

    (
        r_item, r_loc,
        abc_vol, abc_xyz_seg,
        svc_level, z,
        d_mean, d_std, d_cv,
        lt_mean, lt_std,
        ss_demand, ss_lt, ss_comb,
        rop,
        on_hand, dos,
        is_below, gap,
        fcst_src, _target_dos,
        avg_daily,
    ) = row

    # Convert to floats for computation (NULL -> 0 where needed for math)
    z_val = float(z) if z is not None else 0.0
    d_mean_f = float(d_mean) if d_mean is not None else 0.0
    d_std_f = float(d_std) if d_std is not None else 0.0
    lt_mean_f = float(lt_mean) if lt_mean is not None else 0.0
    lt_std_f = float(lt_std) if lt_std is not None else 0.0
    ss_comb_f = float(ss_comb) if ss_comb is not None else 0.0
    avg_daily_f = float(avg_daily) if avg_daily is not None else (d_mean_f / 30.44 if d_mean_f else 0.0)

    # Demand variability component: Z * sqrt(LT_days * (sigma_d_daily)^2)
    # Convert monthly std to daily: sigma_d_daily = sigma_d_monthly / sqrt(30.44)
    d_std_daily = d_std_f / math.sqrt(30.44) if d_std_f else 0.0
    demand_comp = z_val * math.sqrt(lt_mean_f * d_std_daily ** 2) if lt_mean_f > 0 else 0.0

    # Lead time variability component: Z * D_bar_daily * sigma_LT
    lt_comp = z_val * avg_daily_f * lt_std_f if lt_std_f else 0.0

    # Combined: Z * sqrt(LT * sigma_d_daily^2 + D_bar_daily^2 * sigma_LT^2)
    combined_calc = z_val * math.sqrt(
        lt_mean_f * d_std_daily ** 2 + avg_daily_f ** 2 * lt_std_f ** 2
    ) if (lt_mean_f > 0 or lt_std_f > 0) else 0.0

    # Use stored values if available, else computed
    demand_val = float(ss_demand) if ss_demand is not None else round(demand_comp, 1)
    lt_val = float(ss_lt) if ss_lt is not None else round(lt_comp, 1)
    comb_val = ss_comb_f if ss_comb_f else round(combined_calc, 1)

    total = comb_val if comb_val else 1.0  # avoid division by zero
    demand_pct = round(demand_val / total * 100, 1) if total else 0.0
    lt_pct = round(lt_val / total * 100, 1) if total else 0.0

    # Build substituted formula strings
    d_std_daily_r = round(d_std_daily, 1)
    formula_main = "SS = Z * sqrt(LT * sigma_d^2 + D_bar^2 * sigma_LT^2)"
    formula_sub = (
        f"SS = {z_val:.3f} * sqrt({lt_mean_f:.0f} * {d_std_daily_r}^2"
        f" + {avg_daily_f:.1f}^2 * {lt_std_f:.1f}^2)"
        f" = {round(comb_val)} units"
    )

    # ── Sensitivity analysis ────────────────────────────────────────────
    def _calc_ss(z_s: float, d_std_d: float, lt_m: float, d_bar: float, lt_s: float) -> float:
        return z_s * math.sqrt(lt_m * d_std_d ** 2 + d_bar ** 2 * lt_s ** 2) if (lt_m > 0 or lt_s > 0) else 0.0

    sensitivity: list[dict] = []
    base_ss = comb_val

    # Service level scenarios
    for label, z_new in [("Service Level -> 98%", 2.054), ("Service Level -> 90%", 1.282)]:
        if abs(z_new - z_val) > 0.01:
            ss_new = round(_calc_ss(z_new, d_std_daily, lt_mean_f, avg_daily_f, lt_std_f))
            delta = ss_new - round(base_ss)
            pct = round(delta / base_ss * 100) if base_ss else 0
            sign = "+" if delta >= 0 else ""
            sensitivity.append({
                "scenario": label,
                "ss_result": ss_new,
                "delta": f"{sign}{delta} units ({sign}{pct}%)",
            })

    # Demand Std +20%
    d_std_up = d_std_daily * 1.2
    ss_d_up = round(_calc_ss(z_val, d_std_up, lt_mean_f, avg_daily_f, lt_std_f))
    delta_d = ss_d_up - round(base_ss)
    pct_d = round(delta_d / base_ss * 100) if base_ss else 0
    sensitivity.append({
        "scenario": "Demand Std +20%",
        "ss_result": ss_d_up,
        "delta": f"+{delta_d} units (+{pct_d}%)",
    })

    # Lead Time Std +50%
    lt_std_up = lt_std_f * 1.5
    ss_lt_up = round(_calc_ss(z_val, d_std_daily, lt_mean_f, avg_daily_f, lt_std_up))
    delta_lt = ss_lt_up - round(base_ss)
    pct_lt = round(delta_lt / base_ss * 100) if base_ss else 0
    sensitivity.append({
        "scenario": "Lead Time Std +50%",
        "ss_result": ss_lt_up,
        "delta": f"+{delta_lt} units (+{pct_lt}%)",
    })

    # History depth context
    on_hand_f = float(on_hand) if on_hand is not None else None
    gap_f = float(gap) if gap is not None else None

    return {
        "item_id": r_item,
        "loc": r_loc,
        "abc_vol": abc_vol,
        "abc_xyz_segment": abc_xyz_seg,
        "service_level": _f(svc_level),
        "z_score": _f(z),
        "formula": formula_main,
        "formula_substituted": formula_sub,
        "components": {
            "demand_component": {
                "label": "Demand Variability",
                "value": round(demand_val, 1),
                "pct_of_total": demand_pct,
                "formula": (
                    f"Z * sqrt(LT * sigma_d^2) = {z_val:.3f}"
                    f" * sqrt({lt_mean_f:.0f} * {d_std_daily_r}^2)"
                ),
                "inputs": {
                    "demand_mean_monthly": _f(d_mean),
                    "demand_std_monthly": _f(d_std),
                    "demand_cv": _f(d_cv),
                },
            },
            "leadtime_component": {
                "label": "Lead Time Variability",
                "value": round(lt_val, 1),
                "pct_of_total": lt_pct,
                "formula": (
                    f"Z * D_bar_daily * sigma_LT = {z_val:.3f}"
                    f" * {avg_daily_f:.1f} * {lt_std_f:.1f}"
                ),
                "inputs": {
                    "lead_time_mean_days": _f(lt_mean),
                    "lead_time_std_days": _f(lt_std),
                    "lead_time_cv": round(lt_std_f / lt_mean_f, 3) if lt_mean_f else None,
                },
            },
            "combined": {
                "label": "Combined Safety Stock",
                "value": round(comb_val, 1),
                "formula": formula_main,
            },
        },
        "sensitivity": sensitivity,
        "context": {
            "current_on_hand": round(on_hand_f) if on_hand_f is not None else None,
            "current_dos": _f(dos),
            "reorder_point": _f(rop),
            "is_below_ss": bool(is_below) if is_below is not None else None,
            "gap_qty": round(gap_f) if gap_f is not None else None,
            "forecast_source": fcst_src or "historical",
            "avg_daily_demand": round(avg_daily_f, 1),
        },
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
        WHERE item_id = %s
          AND loc     = %s
        RETURNING
            item_id, loc, ss_combined, ss_method, modified_ts
    """

    qty = body.ss_override_qty
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [qty, qty, qty, qty, qty, now, body.item_id, body.loc])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No safety stock record found for item={body.item_id} loc={body.loc}",
        )

    return {
        "item_id":     row[0],
        "loc":         row[1],
        "ss_combined": float(row[2]) if row[2] is not None else None,
        "ss_method":   row[3],
        "modified_ts": row[4].isoformat() if row[4] else None,
    }


@router.post("/inv-planning/safety-stock/what-if")
def simulate_what_if(
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    demand_change_pct: float = Query(0, ge=-50, le=100),
    lt_change_days: float = Query(0, ge=-30, le=60),
    service_level_override: float | None = Query(None, ge=0.80, le=0.999),
    policy_version: str = Query("v1", max_length=20),
    _: None = Depends(require_api_key),
) -> dict:
    """Simulate safety stock under modified parameters.

    Returns current SS alongside the simulated SS for comparison.
    Does NOT write to DB -- read-only simulation.
    """
    # Z-score lookup table from shared_constants.yaml
    z_table: dict[float, float] = {
        0.85: 1.036,
        0.90: 1.282,
        0.93: 1.476,
        0.95: 1.645,
        0.97: 1.881,
        0.98: 2.054,
        0.99: 2.326,
    }

    # Fetch current parameters from fact_safety_stock_targets
    ss_sql = """
        SELECT
            demand_mean_monthly, demand_std_monthly,
            lead_time_mean_days, lead_time_std_days,
            service_level_target, z_score,
            ss_combined, reorder_point,
            avg_daily_demand
        FROM fact_safety_stock_targets
        WHERE item_id = %s
          AND loc = %s
          AND policy_version = %s
        LIMIT 1
    """

    # Fetch unit_cost from fact_eoq_targets (fallback to 1.0)
    eoq_sql = """
        SELECT unit_cost
        FROM fact_eoq_targets
        WHERE item_id = %s AND loc = %s
        LIMIT 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ss_sql, [item_id, loc, policy_version])
            ss_row = cur.fetchone()
            cur.execute(eoq_sql, [item_id, loc])
            eoq_row = cur.fetchone()

    if not ss_row:
        raise HTTPException(
            status_code=404,
            detail=f"No safety stock record for item={item_id} loc={loc} version={policy_version}",
        )

    (
        d_mean, d_std,
        lt_mean, lt_std,
        svc_level, _z_current,
        ss_current, rop_current,
        avg_daily,
    ) = ss_row

    d_mean_f = float(d_mean) if d_mean is not None else 0.0
    d_std_f = float(d_std) if d_std is not None else 0.0
    lt_mean_f = float(lt_mean) if lt_mean is not None else 0.0
    lt_std_f = float(lt_std) if lt_std is not None else 0.0
    svc_level_f = float(svc_level) if svc_level is not None else 0.95
    ss_current_f = float(ss_current) if ss_current is not None else 0.0
    rop_current_f = float(rop_current) if rop_current is not None else 0.0
    unit_cost = float(eoq_row[0]) if eoq_row and eoq_row[0] is not None else 1.0

    # Apply overrides
    demand_mean_adj = d_mean_f * (1 + demand_change_pct / 100)
    demand_std_adj = d_std_f * (1 + demand_change_pct / 100)
    lt_mean_adj = max(1.0, lt_mean_f + lt_change_days)  # floor at 1 day
    sl = service_level_override if service_level_override is not None else svc_level_f

    # Look up z-score for the service level (nearest match)
    def _lookup_z(target_sl: float) -> float:
        best_key = min(z_table.keys(), key=lambda k: abs(k - target_sl))
        return z_table[best_key]

    z = _lookup_z(sl)

    # Recompute safety stock
    d_daily = demand_mean_adj / 30.44
    s_daily = demand_std_adj / math.sqrt(30.44) if demand_std_adj else 0.0
    ss_new = z * math.sqrt(lt_mean_adj * s_daily ** 2 + d_daily ** 2 * lt_std_f ** 2) if (lt_mean_adj > 0 or lt_std_f > 0) else 0.0
    rop_new = d_daily * lt_mean_adj + ss_new

    # Current holding cost (monthly)
    carrying_pct = 0.25
    holding_current = ss_current_f * unit_cost * carrying_pct / 12
    holding_new = ss_new * unit_cost * carrying_pct / 12

    # Deltas
    delta_ss = ss_new - ss_current_f
    delta_ss_pct = round(delta_ss / ss_current_f * 100, 1) if ss_current_f else 0.0
    delta_rop = rop_new - rop_current_f
    delta_holding = holding_new - holding_current

    return {
        "current": {
            "ss_combined": round(ss_current_f),
            "reorder_point": round(rop_current_f),
            "monthly_holding_cost": round(holding_current, 2),
        },
        "simulated": {
            "ss_combined": round(ss_new),
            "reorder_point": round(rop_new),
            "monthly_holding_cost": round(holding_new, 2),
        },
        "delta": {
            "ss_change": round(delta_ss),
            "ss_change_pct": delta_ss_pct,
            "rop_change": round(delta_rop),
            "holding_cost_change_monthly": round(delta_holding, 2),
        },
        "inputs_used": {
            "demand_mean": round(demand_mean_adj, 1),
            "demand_std": round(demand_std_adj, 1),
            "lt_mean_days": round(lt_mean_adj, 1),
            "lt_std_days": round(lt_std_f, 1),
            "service_level": sl,
            "z_score": round(z, 3),
            "unit_cost": round(unit_cost, 2),
        },
    }


@router.get("/inv-planning/safety-stock/config")
def get_ss_config(
    response: FastAPIResponse,
) -> dict:
    """Return the current safety_stock_config.yaml as JSON (with _includes merged).

    Cache: 600s.
    """
    set_cache(response, max_age=600)

    cfg = load_config("safety_stock_config")
    if not cfg:
        raise HTTPException(status_code=404, detail="safety_stock_config.yaml not found")
    return cfg
