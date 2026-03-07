"""F1.1 — Production Forecast API endpoints.

Serves forward-looking ML forecasts from fact_production_forecast.
These are future-period predictions (months with no actuals yet),
distinct from backtest rows in fact_external_forecast_monthly.

Endpoints:
    GET /forecast/production           — DFU-level forecast series
    GET /forecast/production/summary   — Portfolio-level aggregate
    GET /forecast/production/versions  — Available plan versions
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.core import get_conn

router = APIRouter(tags=["production-forecast"])


# ---------------------------------------------------------------------------
# GET /forecast/production
# ---------------------------------------------------------------------------

@router.get("/forecast/production")
async def get_production_forecast(
    item_no: str,
    loc: str,
    horizon: int = 12,
    plan_version: str | None = None,
):
    """Return production forecast series for a specific DFU.

    Args:
        item_no: Item number (exact match).
        loc: Location code (exact match).
        horizon: Max months ahead to return (1–12).
        plan_version: Specific plan version (e.g. '2026-03'). Defaults to latest.

    Returns:
        Forecast rows with confidence intervals and lag source metadata.
    """
    horizon = max(1, min(horizon, 12))

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
    horizon_months: int = 3,
    brand: str | None = None,
    category: str | None = None,
):
    """Return portfolio-level aggregated production forecast.

    Args:
        plan_version: Defaults to latest available.
        horizon_months: Horizon to aggregate over (1–12).
        brand: Filter by brand (optional).
        category: Filter by category (optional, joins dim_item).

    Returns:
        Total DFU count, total forecast qty, breakdown by ABC class.
    """
    horizon_months = max(1, min(horizon_months, 12))

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
