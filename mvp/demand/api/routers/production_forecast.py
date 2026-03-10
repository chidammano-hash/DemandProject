"""F1.1 + F2.2 — Production Forecast and Quantile Demand Plan API endpoints.

Serves forward-looking ML forecasts from fact_production_forecast (F1.1)
and multi-horizon quantile demand plan from fact_demand_plan (F2.2).

Endpoints (F1.1):
    GET /forecast/production           — DFU-level forecast series
    GET /forecast/production/summary   — Portfolio-level aggregate
    GET /forecast/production/versions  — Available plan versions

Endpoints (F2.2):
    GET /forecast/demand-plan          — Quantile forecast (P10/P50/P90) per DFU
    GET /forecast/demand-plan/versions — List all plan versions
    GET /forecast/demand-plan/comparison — Compare two plan versions
    GET /forecast/demand-plan/weekly   — Weekly disaggregated forecast
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.core import get_conn
from common.planning_date import get_planning_date

router = APIRouter(tags=["production-forecast"])


# ---------------------------------------------------------------------------
# GET /forecast/production
# ---------------------------------------------------------------------------

@router.get("/forecast/production")
async def get_production_forecast(
    item_no: str,
    loc: str,
    horizon: int = 18,
    plan_version: str | None = None,
):
    """Return production forecast series for a specific DFU.

    Args:
        item_no: Item number (exact match).
        loc: Location code (exact match).
        horizon: Max months ahead to return (1–18).
        plan_version: Specific plan version (e.g. '2026-02'). Defaults to latest.

    Returns:
        Forecast rows with confidence intervals and lag source metadata.
    """
    horizon = max(1, min(horizon, 18))

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve plan_version if not provided
            if not plan_version:
                cur.execute("""
                    SELECT plan_version
                    FROM fact_production_forecast
                    WHERE item_no = %s AND loc = %s
                    ORDER BY generated_at DESC
                    LIMIT 1
                """, [item_no, loc])
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No production forecast found for {item_no}/{loc}. "
                               f"Run 'make forecast-generate' to generate forecasts."
                    )
                plan_version = row[0]

            cur.execute("""
                SELECT
                    forecast_month,
                    forecast_qty,
                    forecast_qty_lower,
                    forecast_qty_upper,
                    model_id,
                    cluster_id,
                    horizon_months,
                    is_recursive,
                    lag_source,
                    generated_at
                FROM fact_production_forecast
                WHERE item_no = %s
                  AND loc = %s
                  AND plan_version = %s
                  AND horizon_months <= %s
                ORDER BY forecast_month
            """, [item_no, loc, plan_version, horizon])
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast rows found for {item_no}/{loc} in version {plan_version}."
        )

    model_id = rows[0][4]
    generated_at = rows[-1][9]  # use last row's generated_at for display

    return {
        "item_no": item_no,
        "loc": loc,
        "plan_version": plan_version,
        "model_id": model_id,
        "generated_at": generated_at.isoformat() if generated_at else None,
        "horizon_months": horizon,
        "is_recursive": any(r[7] for r in rows),
        "forecasts": [
            {
                "forecast_month": r[0].isoformat() if r[0] else None,
                "forecast_qty": float(r[1]) if r[1] is not None else None,
                "forecast_qty_lower": float(r[2]) if r[2] is not None else None,
                "forecast_qty_upper": float(r[3]) if r[3] is not None else None,
                "model_id": r[4],
                "cluster_id": r[5],
                "horizon_months": r[6],
                "is_recursive": r[7],
                "lag_source": r[8],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /forecast/production/summary
# ---------------------------------------------------------------------------

@router.get("/forecast/production/summary")
async def get_production_forecast_summary(
    plan_version: str | None = None,
    horizon_months: int = 18,
    brand: str | None = None,
    category: str | None = None,
):
    """Return portfolio-level aggregated production forecast.

    Args:
        plan_version: Defaults to latest available.
        horizon_months: Horizon to aggregate over (1–18).
        brand: Filter by brand (optional).
        category: Filter by category (optional, joins dim_item).

    Returns:
        Total DFU count, total forecast qty, breakdown by ABC class.
    """
    horizon_months = max(1, min(horizon_months, 18))

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve latest plan_version if not specified
            if not plan_version:
                cur.execute("""
                    SELECT plan_version
                    FROM fact_production_forecast
                    GROUP BY plan_version
                    ORDER BY MIN(generated_at) DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if not row:
                    return {
                        "plan_version": None,
                        "horizon_months": horizon_months,
                        "total_dfu_count": 0,
                        "total_forecast_qty": 0.0,
                        "generated_at": None,
                        "by_abc_class": [],
                        "ci_coverage_pct": 0.0,
                        "avg_ci_width": None,
                    }
                plan_version = row[0]

            # Build optional joins/filters
            join_clause = ""
            where_extra = ""
            params: list = [plan_version, horizon_months]

            if brand or category:
                join_clause = "JOIN dim_dfu d ON d.dmdunit = f.item_no AND d.loc = f.loc"
                if brand:
                    where_extra += " AND d.brand = %s"
                    params.append(brand)
                if category:
                    where_extra += " AND d.category = %s"
                    params.append(category)

            # Summary query
            cur.execute(f"""
                SELECT
                    COUNT(DISTINCT (f.item_no, f.loc))  AS total_dfu_count,
                    SUM(f.forecast_qty)                 AS total_forecast_qty,
                    MIN(f.generated_at)                 AS generated_at
                FROM fact_production_forecast f
                {join_clause}
                WHERE f.plan_version = %s
                  AND f.horizon_months <= %s
                  {where_extra}
            """, params)
            summary = cur.fetchone()

            # ABC class breakdown
            cur.execute(f"""
                SELECT
                    COALESCE(d.abc_vol, 'Unknown')  AS abc_class,
                    COUNT(DISTINCT (f.item_no, f.loc)) AS dfu_count,
                    SUM(f.forecast_qty)             AS forecast_qty
                FROM fact_production_forecast f
                JOIN dim_dfu d ON d.dmdunit = f.item_no AND d.loc = f.loc
                WHERE f.plan_version = %s
                  AND f.horizon_months <= %s
                  {where_extra}
                GROUP BY d.abc_vol
                ORDER BY d.abc_vol
            """, params)
            abc_rows = cur.fetchall()

            # CI coverage stats
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE forecast_qty_lower IS NOT NULL) AS ci_count,
                    COUNT(*) AS total_count,
                    AVG(forecast_qty_upper - forecast_qty_lower)
                        FILTER (WHERE forecast_qty_lower IS NOT NULL) AS avg_ci_width
                FROM fact_production_forecast
                WHERE plan_version = %s
            """, [plan_version])
            ci_row = cur.fetchone()

    ci_count = int(ci_row[0]) if ci_row and ci_row[0] else 0
    total_count = int(ci_row[1]) if ci_row and ci_row[1] else 0
    avg_ci_width = float(ci_row[2]) if ci_row and ci_row[2] is not None else None
    ci_coverage_pct = round(ci_count / total_count * 100, 1) if total_count > 0 else 0.0

    return {
        "plan_version": plan_version,
        "horizon_months": horizon_months,
        "total_dfu_count": int(summary[0]) if summary and summary[0] else 0,
        "total_forecast_qty": float(summary[1]) if summary and summary[1] else 0.0,
        "generated_at": summary[2].isoformat() if summary and summary[2] else None,
        "by_abc_class": [
            {
                "abc_class": r[0],
                "dfu_count": int(r[1]) if r[1] else 0,
                "forecast_qty": float(r[2]) if r[2] else 0.0,
            }
            for r in abc_rows
        ],
        "ci_coverage_pct": ci_coverage_pct,
        "avg_ci_width": avg_ci_width,
    }


# ---------------------------------------------------------------------------
# GET /forecast/production/versions
# ---------------------------------------------------------------------------

@router.get("/forecast/production/versions")
async def get_production_forecast_versions():
    """List available plan versions with row counts and generation timestamps."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    plan_version,
                    COUNT(DISTINCT (item_no, loc))  AS dfu_count,
                    COUNT(*)                        AS total_rows,
                    MIN(generated_at)               AS generated_at
                FROM fact_production_forecast
                GROUP BY plan_version
                ORDER BY MIN(generated_at) DESC
            """)
            rows = cur.fetchall()

    return {
        "versions": [
            {
                "plan_version": r[0],
                "dfu_count": int(r[1]) if r[1] else 0,
                "total_rows": int(r[2]) if r[2] else 0,
                "generated_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]
    }


# ===========================================================================
# F2.2 — Multi-Horizon Demand Plan (Quantile Forecasts)
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /forecast/demand-plan/versions  (must come before /{param} routes)
# ---------------------------------------------------------------------------

@router.get("/forecast/demand-plan/versions")
async def get_demand_plan_versions():
    """List all demand plan versions from fact_plan_versions."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    plan_version, plan_date, plan_label,
                    model_id, horizon_months, dfu_count,
                    status, generated_at
                FROM fact_plan_versions
                ORDER BY generated_at DESC
            """)
            rows = cur.fetchall()

    return {
        "versions": [
            {
                "plan_version": r[0],
                "plan_date": r[1].isoformat() if r[1] else None,
                "plan_label": r[2],
                "model_id": r[3],
                "horizon_months": r[4],
                "dfu_count": r[5],
                "status": r[6],
                "generated_at": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan/comparison
# ---------------------------------------------------------------------------

@router.get("/forecast/demand-plan/comparison")
async def get_demand_plan_comparison(
    v1: str,
    v2: str,
    item_no: str,
    loc: str,
):
    """Compare P10/P50/P90 between two demand plan versions for a DFU."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    plan_month,
                    plan_version,
                    quantile,
                    forecast_qty
                FROM fact_demand_plan
                WHERE item_no = %s AND loc = %s
                  AND plan_version IN (%s, %s)
                  AND quantile IN (0.10, 0.50, 0.90)
                ORDER BY plan_month, plan_version, quantile
            """, [item_no, loc, v1, v2])
            rows = cur.fetchall()

    from collections import defaultdict
    pivot: dict = defaultdict(lambda: defaultdict(dict))
    for plan_month, plan_version, quantile, qty in rows:
        pivot[plan_month.isoformat()][plan_version][float(quantile)] = float(qty) if qty else None

    months = []
    for month_key in sorted(pivot.keys()):
        d1 = pivot[month_key].get(v1, {})
        d2 = pivot[month_key].get(v2, {})
        v1_p50 = d1.get(0.50)
        v2_p50 = d2.get(0.50)
        delta_p50 = round(v1_p50 - v2_p50, 2) if v1_p50 is not None and v2_p50 is not None else None
        delta_pct = round((delta_p50 / v2_p50) * 100, 2) if delta_p50 is not None and v2_p50 else None
        months.append({
            "plan_month": month_key,
            "v1_p10": d1.get(0.10),
            "v1_p50": v1_p50,
            "v1_p90": d1.get(0.90),
            "v2_p10": d2.get(0.10),
            "v2_p50": v2_p50,
            "v2_p90": d2.get(0.90),
            "delta_p50": delta_p50,
            "delta_pct": delta_pct,
        })

    return {
        "item_no": item_no,
        "loc": loc,
        "v1": v1,
        "v2": v2,
        "months": months,
    }


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan/weekly
# ---------------------------------------------------------------------------

@router.get("/forecast/demand-plan/weekly")
async def get_demand_plan_weekly(
    item_no: str,
    loc: str,
    plan_version: str | None = None,
    weeks_ahead: int = 8,
):
    """Return weekly disaggregated quantile forecast for a DFU."""
    weeks_ahead = max(1, min(weeks_ahead, 52))

    with get_conn() as conn:
        with conn.cursor() as cur:
            if not plan_version:
                cur.execute("""
                    SELECT plan_version FROM fact_demand_plan_weekly
                    WHERE item_no = %s AND loc = %s
                    ORDER BY generated_at DESC LIMIT 1
                """, [item_no, loc])
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No weekly demand plan found for {item_no}/{loc}."
                    )
                plan_version = row[0]

            planning_dt = get_planning_date()
            cur.execute("""
                SELECT
                    plan_week, iso_week, iso_year, plan_month,
                    quantile, forecast_qty, weekly_weight
                FROM fact_demand_plan_weekly
                WHERE item_no = %s AND loc = %s AND plan_version = %s
                  AND plan_week >= %s
                ORDER BY plan_week, quantile
                LIMIT %s
            """, [item_no, loc, plan_version, planning_dt, weeks_ahead * 3])  # 3 quantiles per week
            rows = cur.fetchall()

    # Group by week
    from collections import defaultdict
    by_week: dict = defaultdict(dict)
    for plan_week, iso_week, iso_year, plan_month, quantile, qty, weight in rows:
        wk = plan_week.isoformat()
        q = float(quantile)
        if wk not in by_week:
            by_week[wk] = {
                "plan_week": wk,
                "iso_week": iso_week,
                "iso_year": iso_year,
                "parent_month": plan_month.isoformat() if plan_month else None,
                "weekly_weight": float(weight) if weight else None,
            }
        label = {0.10: "p10_weekly", 0.50: "p50_weekly", 0.90: "p90_weekly"}.get(q)
        if label:
            by_week[wk][label] = float(qty) if qty is not None else None

    sorted_weeks = sorted(by_week.values(), key=lambda x: x["plan_week"])[:weeks_ahead]

    return {
        "item_no": item_no,
        "loc": loc,
        "plan_version": plan_version,
        "weeks": sorted_weeks,
    }


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan  (most general, last)
# ---------------------------------------------------------------------------

@router.get("/forecast/demand-plan")
async def get_demand_plan(
    item_no: str,
    loc: str,
    plan_version: str | None = None,
    quantile: float | None = None,
    horizon: int = 12,
):
    """Return quantile (P10/P50/P90) demand plan for a specific DFU.

    Pivots 3-quantile rows per month into a single row with p10/p50/p90 fields.
    """
    horizon = max(1, min(horizon, 18))

    with get_conn() as conn:
        with conn.cursor() as cur:
            if not plan_version:
                cur.execute("""
                    SELECT plan_version FROM fact_plan_versions
                    WHERE status = 'active'
                    ORDER BY generated_at DESC LIMIT 1
                """)
                row = cur.fetchone()
                if not row:
                    # Fall back to any version
                    cur.execute("""
                        SELECT plan_version FROM fact_demand_plan
                        WHERE item_no = %s AND loc = %s
                        ORDER BY generated_at DESC LIMIT 1
                    """, [item_no, loc])
                    row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No demand plan found for {item_no}/{loc}."
                    )
                plan_version = row[0]

            params = [item_no, loc, plan_version, horizon]
            quantile_filter = ""
            if quantile is not None:
                quantile_filter = " AND quantile = %s"
                params.append(quantile)

            cur.execute(f"""
                SELECT
                    plan_month,
                    quantile,
                    forecast_qty,
                    lower_bound,
                    upper_bound,
                    sigma_forecast,
                    sigma_demand,
                    sigma_combined,
                    horizon_months
                FROM fact_demand_plan
                WHERE item_no = %s AND loc = %s
                  AND plan_version = %s
                  AND horizon_months <= %s
                  {quantile_filter}
                ORDER BY plan_month, quantile
            """, params)
            rows = cur.fetchall()

            # Get version metadata
            cur.execute("""
                SELECT generated_at FROM fact_plan_versions
                WHERE plan_version = %s
            """, [plan_version])
            ver_row = cur.fetchone()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No demand plan rows found for {item_no}/{loc} version {plan_version}."
        )

    # Pivot by plan_month
    from collections import defaultdict
    by_month: dict = defaultdict(dict)
    for plan_month, q, forecast_qty, lower, upper, sf, sd, sc, h_months in rows:
        mkey = plan_month.isoformat()
        q = float(q)
        if mkey not in by_month:
            by_month[mkey] = {
                "plan_month": mkey,
                "horizon_months": h_months,
                "sigma_forecast": float(sf) if sf is not None else None,
                "sigma_demand": float(sd) if sd is not None else None,
                "sigma_combined": float(sc) if sc is not None else None,
            }
        label = {0.10: "p10", 0.50: "p50", 0.90: "p90"}.get(q)
        if label:
            by_month[mkey][label] = float(forecast_qty) if forecast_qty is not None else None

    sorted_months = sorted(by_month.values(), key=lambda x: x["plan_month"])

    return {
        "item_no": item_no,
        "loc": loc,
        "plan_version": plan_version,
        "generated_at": ver_row[0].isoformat() if ver_row and ver_row[0] else None,
        "horizon_months": horizon,
        "rows": sorted_months,
    }
