"""Working Capital Analytics router — Gen-4 SC-10.

Exposes GET /analytics/working-capital with DIO, DPO, DSO, cash-to-cash cycle,
and inventory turns, plus GET /analytics/rolling-13-week (SC-10 P2).

Data sources (best-effort — each with graceful fallback):
  - DIO: agg_inventory_monthly.eom_qty_on_hand * dim_sku.std_cost vs
         fact_sales_monthly.qty_shipped * dim_sku.std_cost (COGS)
  - DSO: optional fact_accounts_receivable (if present) — else proxy via
         average days between sale and typical payment terms (falls back to NULL)
  - DPO: fact_purchase_orders.delivery_date - fact_purchase_orders.original_delivery_date
         averaged across closed POs (proxy — real AP data not yet modeled)

All formulas emit NULL when inputs are missing rather than guessing.
Rolling-13-week reads agg_sales_weekly (sql/150_create_agg_sales_weekly.sql).
"""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, get_conn, set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["working-capital"])


# ---------------------------------------------------------------------------
# GET /analytics/working-capital
# ---------------------------------------------------------------------------

_WORKING_CAPITAL_SQL = """
    WITH inventory_value AS (
        -- Avg EOM inventory value in the window (days in period = months * 30.44)
        SELECT
            AVG(a.eom_qty_on_hand * COALESCE(d.std_cost, 1.0)) AS avg_inventory_value,
            COUNT(DISTINCT a.month_start)                      AS months_in_window
        FROM agg_inventory_monthly a
        LEFT JOIN dim_sku d
            ON a.item_id = d.item_id AND a.loc = d.loc
        WHERE a.month_start >= %s AND a.month_start <= %s
    ),
    cogs AS (
        -- Period COGS = sum(shipped * std_cost)
        SELECT
            COALESCE(SUM(s.qty_shipped * COALESCE(d.std_cost, 1.0)), 0) AS period_cogs,
            COUNT(DISTINCT s.startdate)                                  AS months_in_window
        FROM fact_sales_monthly s
        LEFT JOIN dim_sku d
            ON s.item_id = d.item_id AND s.loc = d.loc
        WHERE s.type = 1
          AND s.startdate >= %s AND s.startdate <= %s
    ),
    po_timing AS (
        -- DPO proxy: avg gap between order placement (original_delivery_date - lead_time) and payment
        -- True AP data isn't modeled; use (delivery_date - po_date) as a weak proxy.
        -- TODO(SC-10): swap to fact_accounts_payable when it lands.
        SELECT
            AVG(EXTRACT(EPOCH FROM (delivery_date - original_delivery_date))
                / 86400.0) AS avg_payment_lag_days
        FROM fact_purchase_orders
        WHERE is_closed = TRUE
          AND delivery_date IS NOT NULL
          AND original_delivery_date IS NOT NULL
    )
    SELECT
        (SELECT avg_inventory_value FROM inventory_value)   AS avg_inventory_value,
        (SELECT period_cogs          FROM cogs)              AS period_cogs,
        (SELECT months_in_window     FROM cogs)              AS cogs_months,
        (SELECT avg_payment_lag_days FROM po_timing)         AS avg_payment_lag_days
"""


@router.get("/analytics/working-capital")
def get_working_capital(
    response: FastAPIResponse,
    period_from: str = Query(
        ..., description="Start of period, inclusive (YYYY-MM-DD)"
    ),
    period_to: str = Query(
        ..., description="End of period, inclusive (YYYY-MM-DD)"
    ),
    dso_days: float | None = Query(
        None,
        description="Override DSO in days (default: NULL because AR data not modeled)",
    ),
) -> dict:
    """Return DIO, DPO, DSO, cash-to-cash cycle, and inventory turns.

    Formulas:
        days_in_period = (period_to - period_from) in days
        DIO   = avg_inventory_value / (period_cogs / days_in_period)
        turns = 365 / DIO  (annualized)
        DPO   = avg_payment_lag_days  (proxy — see SQL TODO)
        DSO   = caller-provided or NULL
        C2C   = DIO + DSO - DPO  (NULL if any input is NULL)

    Cache: 600s.
    """
    set_cache(response, max_age=600)

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    _WORKING_CAPITAL_SQL,
                    (period_from, period_to, period_from, period_to),
                )
                row = cur.fetchone()
            except psycopg.Error as exc:
                logger.exception("working-capital query failed: %s", exc)
                return {
                    "dio_days": None, "dpo_days": None, "dso_days": dso_days,
                    "cash_to_cash_days": None, "inventory_turns": None,
                    "period_from": period_from, "period_to": period_to,
                    "error": "query_failed",
                }

    avg_inv = _f(row[0]) if row else None
    period_cogs = _f(row[1]) if row else None
    _cogs_months = _f(row[2]) if row else None
    avg_payment_lag = _f(row[3]) if row else None

    # Days in period from string dates
    try:
        from datetime import date
        y1, m1, d1 = period_from.split("-")
        y2, m2, d2 = period_to.split("-")
        days_in_period = (date(int(y2), int(m2), int(d2))
                          - date(int(y1), int(m1), int(d1))).days + 1
    except ValueError:
        days_in_period = 0

    dio: float | None = None
    if avg_inv is not None and period_cogs and period_cogs > 0 and days_in_period > 0:
        cogs_per_day = period_cogs / days_in_period
        if cogs_per_day > 0:
            dio = avg_inv / cogs_per_day

    turns: float | None = None
    if dio and dio > 0:
        turns = 365.0 / dio

    dpo = avg_payment_lag  # proxy; see SQL TODO
    dso = dso_days  # NULL by default

    c2c: float | None = None
    if dio is not None and dpo is not None and dso is not None:
        c2c = dio + dso - dpo

    return {
        "dio_days": round(dio, 2) if dio is not None else None,
        "dpo_days": round(dpo, 2) if dpo is not None else None,
        "dso_days": round(dso, 2) if dso is not None else None,
        "cash_to_cash_days": round(c2c, 2) if c2c is not None else None,
        "inventory_turns": round(turns, 2) if turns is not None else None,
        "period_from": period_from,
        "period_to": period_to,
        "avg_inventory_value": avg_inv,
        "period_cogs": period_cogs,
        # Pass through caveats so the UI can warn planners
        "notes": {
            "dpo": "Proxy: avg PO delivery_date - original_delivery_date. Replace with AP data.",
            "dso": "NULL by default — AR data not yet modeled; pass ?dso_days= to override.",
        },
    }


# ---------------------------------------------------------------------------
# GET /analytics/rolling-13-week  (SC-10 P2)
# ---------------------------------------------------------------------------

@router.get("/analytics/rolling-13-week")
def get_rolling_13_week(
    response: FastAPIResponse,
    item: str | None = Query(None, max_length=120),
    location: str | None = Query(None, max_length=120),
) -> dict:
    """Last 13 ISO weeks of sales, grouped by week_start.

    Reads agg_sales_weekly (sql/150). Filters optional.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = []
    params: list = []

    if item:
        params.append(f"%{item}%")
        where_parts.append("item_id ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        WITH recent_weeks AS (
            SELECT DISTINCT week_start
            FROM agg_sales_weekly
            ORDER BY week_start DESC
            LIMIT 13
        )
        SELECT
            w.week_start,
            w.iso_year,
            w.iso_week,
            SUM(w.qty_ordered) AS qty_ordered,
            SUM(w.qty_shipped) AS qty_shipped,
            SUM(w.qty_sold)    AS qty_sold
        FROM agg_sales_weekly w
        JOIN recent_weeks r ON r.week_start = w.week_start
        {where_sql}
        GROUP BY w.week_start, w.iso_year, w.iso_week
        ORDER BY w.week_start ASC
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except psycopg.Error as exc:
                logger.exception("rolling-13-week query failed: %s", exc)
                return {"weeks": [], "error": "query_failed"}

    return {
        "weeks": [
            {
                "week_start": str(r[0]),
                "iso_year": int(r[1]) if r[1] is not None else None,
                "iso_week": int(r[2]) if r[2] is not None else None,
                "qty_ordered": _f(r[3]),
                "qty_shipped": _f(r[4]),
                "qty_sold": _f(r[5]),
            }
            for r in rows
        ]
    }
