"""Purchase Orders endpoints — comprehensive PO history (open + closed)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache, qident

router = APIRouter(prefix="/purchase-orders", tags=["purchase-orders"])


@router.get("/rows")
def po_rows(
    response: FastAPIResponse,
    po_number: str = Query(default="", max_length=120),
    item: str = Query(default="", max_length=120),
    loc: str = Query(default="", max_length=120),
    supplier: str = Query(default="", max_length=120),
    status: str = Query(default="", max_length=30),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="delivery_date", max_length=60),
    sort_dir: str = Query(default="desc", max_length=4),
):
    """Paginated PO rows with filters."""
    set_cache(response, max_age=120)

    allowed_sort = {
        "po_number", "item_id", "loc", "delivery_date", "ordered_qty",
        "net_price", "closure_code", "supplier_name", "original_delivery_date",
    }
    order_col = sort_by if sort_by in allowed_sort else "delivery_date"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list[Any] = []
    if po_number.strip():
        where_parts.append("po_number ILIKE %s")
        params.append(f"%{po_number.strip()}%")
    if item.strip():
        where_parts.append("item_id ILIKE %s")
        params.append(f"%{item.strip()}%")
    if loc.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{loc.strip()}%")
    if supplier.strip():
        where_parts.append("(supplier_name ILIKE %s OR supplier_id ILIKE %s)")
        params.extend([f"%{supplier.strip()}%", f"%{supplier.strip()}%"])
    if status.strip():
        if status.strip().upper() == "OPEN":
            where_parts.append("(closure_code IS NULL OR closure_code = '')")
        elif status.strip().upper() == "CLOSED":
            where_parts.append("closure_code = 'CLOSED'")

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM fact_purchase_orders {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT po_ck, po_number, site_id, loc, source, item_id,
                       ordered_qty, net_price, gross_value, closure_code,
                       po_hdr_status, po_line_status, receipt_status,
                       supplier_id, supplier_name, carrier_name,
                       delivery_date, original_delivery_date,
                       current_ship_date, original_ship_date,
                       po_type, is_closed,
                       lead_time_planned, lead_time_actual
                FROM fact_purchase_orders {where_sql}
                ORDER BY {qident(order_col)} {order_dir}
                LIMIT %s OFFSET %s""",
            [*params, limit, offset],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    return {"total": total, "rows": rows}


@router.get("/search")
def po_search(
    response: FastAPIResponse,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=200),
):
    """Full-text search across PO fields."""
    set_cache(response, max_age=60)
    pattern = f"%{q.strip()}%"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT po_ck, po_number, item_id, loc, source,
                      supplier_name, closure_code, delivery_date,
                      ordered_qty, net_price
               FROM fact_purchase_orders
               WHERE po_number ILIKE %s OR item_id ILIKE %s
                  OR loc ILIKE %s OR supplier_name ILIKE %s
               LIMIT %s""",
            [pattern, pattern, pattern, pattern, limit],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"rows": rows}


@router.get("/by-po/{po_num}")
def po_by_number(po_num: str, response: FastAPIResponse):
    """All lines for a given PO number."""
    set_cache(response, max_age=120)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT po_ck, po_number, site_id, loc, source, item_id,
                      ordered_qty, orig_po_qty, net_price, gross_value,
                      closure_code, po_hdr_status, po_line_status, receipt_status,
                      supplier_id, supplier_name, carrier_name,
                      delivery_date, original_delivery_date,
                      current_ship_date, original_ship_date,
                      po_type, is_closed, lead_time_planned, lead_time_actual
               FROM fact_purchase_orders
               WHERE po_number = %s
               ORDER BY item_id, loc""",
            [po_num],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"rows": rows, "total": len(rows)}


@router.get("/summary")
def po_summary(response: FastAPIResponse):
    """Aggregate stats: open/closed counts, total values."""
    set_cache(response, max_age=300)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                count(*) AS total_lines,
                count(*) FILTER (WHERE is_closed) AS closed_lines,
                count(*) FILTER (WHERE NOT is_closed) AS open_lines,
                count(DISTINCT po_number) AS distinct_pos,
                count(DISTINCT supplier_id) AS distinct_suppliers,
                count(DISTINCT item_id) AS distinct_items,
                COALESCE(SUM(gross_value), 0) AS total_value,
                COALESCE(SUM(gross_value) FILTER (WHERE NOT is_closed), 0) AS open_value,
                COALESCE(SUM(gross_value) FILTER (WHERE is_closed), 0) AS closed_value
            FROM fact_purchase_orders
        """)
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


@router.get("/aging")
def po_aging(response: FastAPIResponse):
    """Open PO aging buckets: 0-30, 30-60, 60-90, 90+ days."""
    set_cache(response, max_age=300)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                CASE
                    WHEN CURRENT_DATE - delivery_date <= 30 THEN '0-30'
                    WHEN CURRENT_DATE - delivery_date <= 60 THEN '30-60'
                    WHEN CURRENT_DATE - delivery_date <= 90 THEN '60-90'
                    ELSE '90+'
                END AS age_bucket,
                count(*) AS line_count,
                COALESCE(SUM(gross_value), 0) AS total_value
            FROM fact_purchase_orders
            WHERE NOT is_closed
              AND delivery_date IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"buckets": rows}


@router.get("/otd")
def po_on_time_delivery(
    response: FastAPIResponse,
    supplier: str = Query(default="", max_length=120),
):
    """On-time delivery rate by supplier for closed POs."""
    set_cache(response, max_age=300)
    where_parts: list[str] = ["is_closed", "delivery_date IS NOT NULL", "original_delivery_date IS NOT NULL"]
    params: list[Any] = []
    if supplier.strip():
        where_parts.append("(supplier_name ILIKE %s OR supplier_id ILIKE %s)")
        params.extend([f"%{supplier.strip()}%", f"%{supplier.strip()}%"])

    where_sql = f"WHERE {' AND '.join(where_parts)}"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT
                    supplier_id,
                    MAX(supplier_name) AS supplier_name,
                    count(*) AS total_closed,
                    count(*) FILTER (WHERE delivery_date <= original_delivery_date) AS on_time,
                    ROUND(100.0 * count(*) FILTER (WHERE delivery_date <= original_delivery_date)
                          / NULLIF(count(*), 0), 1) AS otd_pct,
                    ROUND(AVG(lead_time_actual), 1) AS avg_lead_time_days
                FROM fact_purchase_orders
                {where_sql}
                GROUP BY supplier_id
                ORDER BY total_closed DESC
                LIMIT 100""",
            params,
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"suppliers": rows}
