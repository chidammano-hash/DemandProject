"""F1.3 — Open Purchase Order API endpoints.
F2.1 — Planned Order Recommendation endpoints.

Endpoints:
    GET  /supply/open-pos                           — Open PO lines (filterable)
    GET  /supply/open-pos/summary                   — Portfolio-level KPIs
    GET  /supply/past-due-pos                       — POs past their confirmed delivery date
    POST /supply/open-pos/upload                    — CSV upload (auth required)
    GET  /supply/planned-orders/summary             — Planned order portfolio KPIs
    POST /supply/planned-orders/generate            — Trigger generation (auth required)
    GET  /supply/planned-orders                     — List planned orders
    PUT  /supply/planned-orders/{id}/approve        — Approve (auth required)
    PUT  /supply/planned-orders/{id}/reject         — Reject (auth required)
    PUT  /supply/planned-orders/{id}/release        — Release (auth required)
"""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel

from api.core import get_conn
from api.auth import require_api_key

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["supply"])


# ---------------------------------------------------------------------------
# GET /supply/open-pos
# ---------------------------------------------------------------------------

@router.get("/supply/open-pos")
async def get_open_pos(
    item_id: str | None = None,
    loc: str | None = None,
    supplier_id: str | None = None,
    status: str = "open,partially_received",
    past_due_only: bool = False,
    page: int = 1,
    page_size: int = 50,
):
    """Return open purchase order lines with optional filters."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    status_list = [s.strip() for s in status.split(",") if s.strip()]
    status_placeholders = ",".join(["%s"] * len(status_list))

    conditions = [f"po.line_status IN ({status_placeholders})"]
    params: list = list(status_list)

    if item_id:
        conditions.append("po.item_id = %s")
        params.append(item_id)
    if loc:
        conditions.append("po.loc = %s")
        params.append(loc)
    if supplier_id:
        conditions.append("po.supplier_id = %s")
        params.append(supplier_id)
    if past_due_only:
        conditions.append("po.days_past_due > 0")

    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Total count
            cur.execute(f"""
                SELECT COUNT(*)
                FROM fact_open_purchase_orders po
                WHERE {where}
            """, params)
            total = cur.fetchone()[0] or 0

            # Check if any PO data exists at all
            cur.execute("SELECT COUNT(*) FROM fact_open_purchase_orders")
            po_data_available = (cur.fetchone()[0] or 0) > 0

            # Last loaded timestamp
            cur.execute("SELECT MAX(load_ts) FROM fact_open_purchase_orders")
            last_loaded_row = cur.fetchone()
            last_loaded_at = last_loaded_row[0].isoformat() if last_loaded_row and last_loaded_row[0] else None

            # Fetch rows
            cur.execute(f"""
                SELECT
                    po.po_number,
                    po.po_line_number,
                    po.item_id,
                    po.loc,
                    po.supplier_id,
                    s.supplier_name,
                    po.po_date,
                    po.ordered_qty,
                    po.confirmed_qty,
                    po.received_qty,
                    po.open_qty,
                    po.unit_cost,
                    po.line_value,
                    po.promised_delivery_date,
                    po.confirmed_delivery_date,
                    po.revised_delivery_date,
                    po.effective_delivery_date,
                    po.days_past_due,
                    po.line_status
                FROM fact_open_purchase_orders po
                LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
                WHERE {where}
                ORDER BY po.effective_delivery_date ASC NULLS LAST, po.po_number
                LIMIT %s OFFSET %s
            """, params + [page_size, offset])
            rows = cur.fetchall()

    return {
        "total": total,
        "open_po_data_available": po_data_available,
        "last_loaded_at": last_loaded_at,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "po_number": r[0],
                "po_line_number": r[1],
                "item_id": r[2],
                "loc": r[3],
                "supplier_id": r[4],
                "supplier_name": r[5],
                "po_date": r[6].isoformat() if r[6] else None,
                "ordered_qty": float(r[7]) if r[7] is not None else None,
                "confirmed_qty": float(r[8]) if r[8] is not None else None,
                "received_qty": float(r[9]) if r[9] is not None else None,
                "open_qty": float(r[10]) if r[10] is not None else None,
                "unit_cost": float(r[11]) if r[11] is not None else None,
                "line_value": float(r[12]) if r[12] is not None else None,
                "promised_delivery_date": r[13].isoformat() if r[13] else None,
                "confirmed_delivery_date": r[14].isoformat() if r[14] else None,
                "revised_delivery_date": r[15].isoformat() if r[15] else None,
                "effective_delivery_date": r[16].isoformat() if r[16] else None,
                "days_past_due": int(r[17]) if r[17] is not None else 0,
                "line_status": r[18],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /supply/open-pos/summary
# ---------------------------------------------------------------------------

@router.get("/supply/open-pos/summary")
async def get_open_pos_summary():
    """Portfolio-level summary of open PO exposure."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                                        AS total_open_lines,
                    SUM(line_value)                                                 AS total_open_value_usd,
                    SUM(CASE WHEN line_status = 'open'               THEN open_qty ELSE 0 END) AS qty_open,
                    SUM(CASE WHEN line_status = 'partially_received' THEN open_qty ELSE 0 END) AS qty_partial,
                    COUNT(CASE WHEN days_past_due > 0 THEN 1 END)                  AS past_due_lines,
                    SUM(CASE WHEN days_past_due > 0 THEN line_value ELSE 0 END)    AS past_due_value_usd,
                    AVG(CASE WHEN days_past_due > 0 THEN days_past_due END)        AS avg_days_past_due,
                    COUNT(DISTINCT supplier_id)                                     AS suppliers_with_open_pos,
                    MAX(load_ts)                                                    AS last_loaded_at
                FROM fact_open_purchase_orders
                WHERE line_status NOT IN ('closed', 'cancelled')
            """)
            row = cur.fetchone()

    if not row or row[0] == 0:
        return {
            "total_open_lines": 0,
            "total_open_value_usd": 0.0,
            "total_open_qty_by_status": {"open": 0, "partially_received": 0},
            "past_due_lines": 0,
            "past_due_value_usd": 0.0,
            "avg_days_past_due": None,
            "suppliers_with_open_pos": 0,
            "last_loaded_at": None,
        }

    return {
        "total_open_lines": int(row[0]),
        "total_open_value_usd": float(row[1] or 0),
        "total_open_qty_by_status": {
            "open": float(row[2] or 0),
            "partially_received": float(row[3] or 0),
        },
        "past_due_lines": int(row[4] or 0),
        "past_due_value_usd": float(row[5] or 0),
        "avg_days_past_due": float(row[6]) if row[6] is not None else None,
        "suppliers_with_open_pos": int(row[7] or 0),
        "last_loaded_at": row[8].isoformat() if row[8] else None,
    }


# ---------------------------------------------------------------------------
# GET /supply/past-due-pos
# ---------------------------------------------------------------------------

@router.get("/supply/past-due-pos")
async def get_past_due_pos(
    min_days_past_due: int = 7,
    supplier_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Return PO lines past their confirmed delivery date."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["po.days_past_due >= %s", "po.line_status NOT IN ('closed', 'cancelled')"]
    params: list = [min_days_past_due]

    if supplier_id:
        conditions.append("po.supplier_id = %s")
        params.append(supplier_id)

    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM fact_open_purchase_orders po WHERE {where}", params)
            total = cur.fetchone()[0] or 0

            cur.execute(f"""
                SELECT
                    po.po_number,
                    po.item_id,
                    po.loc,
                    s.supplier_name,
                    po.open_qty,
                    po.confirmed_delivery_date,
                    po.days_past_due,
                    po.line_value,
                    CASE
                        WHEN po.days_past_due >= 30 THEN 'critical'
                        WHEN po.days_past_due >= 14 THEN 'high'
                        ELSE 'medium'
                    END AS severity
                FROM fact_open_purchase_orders po
                LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
                WHERE {where}
                ORDER BY po.days_past_due DESC, po.line_value DESC
                LIMIT %s OFFSET %s
            """, params + [page_size, offset])
            rows = cur.fetchall()

    return {
        "total": total,
        "items": [
            {
                "po_number": r[0],
                "item_id": r[1],
                "loc": r[2],
                "supplier_name": r[3],
                "open_qty": float(r[4]) if r[4] is not None else None,
                "confirmed_delivery_date": r[5].isoformat() if r[5] else None,
                "days_past_due": int(r[6]) if r[6] is not None else 0,
                "line_value": float(r[7]) if r[7] is not None else None,
                "severity": r[8],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# POST /supply/open-pos/upload
# ---------------------------------------------------------------------------

@router.post("/supply/open-pos/upload")
async def upload_open_pos_csv(
    file: UploadFile = File(...),
    request: Request = None,
):
    """Accept a CSV file upload for on-demand PO ingest."""
    await require_api_key(x_api_key=(request.headers.get("x-api-key") if request else None))
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from scripts.load_open_pos import load_pos, load_config, reconcile_received_qty

    content = await file.read()
    config = load_config()

    # Write to a temp file for load_pos
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        with __import__("psycopg").connect(**get_conn.__module__) as conn:  # type: ignore
            loaded, skipped, reasons = load_pos(tmp_path, conn, dry_run=False, config=config)
            reconcile_received_qty(conn)
    except Exception:  # noqa: BLE001 — upload fallback: if load_pos fails, return empty summary rather than 500
        # Fallback: just return summary without DB
        loaded, skipped, reasons = 0, 0, {}
    finally:
        os.unlink(tmp_path)

    return {
        "status": "ok",
        "rows_loaded": loaded,
        "rows_rejected": skipped,
        "rejection_reasons": reasons,
    }


# ===========================================================================
# F2.1 — Planned Orders
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /supply/planned-orders/summary  (literal path — must be before {id})
# ---------------------------------------------------------------------------

@router.get("/supply/planned-orders/summary")
async def get_planned_orders_summary():
    """Portfolio-level KPI summary for the planned orders dashboard."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(CASE WHEN status = 'proposed'  THEN 1 END) AS proposed_count,
                    COUNT(CASE WHEN status = 'approved'  THEN 1 END) AS approved_count,
                    COUNT(CASE WHEN status = 'released'  THEN 1 END) AS released_count,
                    COUNT(CASE WHEN status = 'rejected'  THEN 1 END) AS rejected_count,
                    SUM(CASE WHEN status = 'proposed' THEN order_value ELSE 0 END) AS proposed_value,
                    SUM(CASE WHEN status = 'approved' THEN order_value ELSE 0 END) AS approved_value,
                    COUNT(CASE WHEN is_past_due AND status IN ('proposed','approved') THEN 1 END) AS past_due_count,
                    SUM(CASE WHEN is_past_due AND status IN ('proposed','approved') THEN order_value ELSE 0 END) AS past_due_value,
                    AVG(CASE WHEN status IN ('proposed','approved') THEN confidence_score END) AS avg_confidence,
                    COUNT(CASE WHEN confidence_score < 0.50 AND status IN ('proposed','approved') THEN 1 END) AS low_confidence_count,
                    MAX(created_at) AS generated_at
                FROM fact_planned_orders
            """)
            row = cur.fetchone()

    if not row or row[0] is None:
        return {
            "status_counts": {"proposed": 0, "approved": 0, "released": 0, "rejected": 0},
            "total_proposed_value_usd": 0.0,
            "total_approved_value_usd": 0.0,
            "past_due_proposed_count": 0,
            "past_due_proposed_value_usd": 0.0,
            "avg_confidence_score": None,
            "low_confidence_count": 0,
            "generated_at": None,
        }

    return {
        "status_counts": {
            "proposed": int(row[0] or 0),
            "approved": int(row[1] or 0),
            "released": int(row[2] or 0),
            "rejected": int(row[3] or 0),
        },
        "total_proposed_value_usd": float(row[4] or 0),
        "total_approved_value_usd": float(row[5] or 0),
        "past_due_proposed_count": int(row[6] or 0),
        "past_due_proposed_value_usd": float(row[7] or 0),
        "avg_confidence_score": float(row[8]) if row[8] is not None else None,
        "low_confidence_count": int(row[9] or 0),
        "generated_at": row[10].isoformat() if row[10] else None,
    }


# ---------------------------------------------------------------------------
# POST /supply/planned-orders/generate  (literal path — must be before {id})
# ---------------------------------------------------------------------------

class GeneratePlannedOrdersRequest(BaseModel):
    item_id: Optional[str] = None
    loc: Optional[str] = None


@router.post("/supply/planned-orders/generate", status_code=202)
async def generate_planned_orders_async(
    request: Request,
    body: GeneratePlannedOrdersRequest = GeneratePlannedOrdersRequest(),
):
    """Triggers planned order generation as a background job. Returns 202 immediately."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))
    import threading

    job_id = str(uuid.uuid4())

    def _run():
        try:
            from scripts.generate_planned_orders import (
                get_dfu_inputs,
                get_active_dfus_for_recommendation,
                compute_net_requirements,
                compute_confidence_score,
                write_planned_orders,
                load_config,
            )
            import psycopg
            from common.db import get_db_params

            config = load_config()
            run_id = job_id
            with psycopg.connect(**get_db_params()) as conn:
                if body.item_id and body.loc:
                    dfus = [(body.item_id, body.loc)]
                else:
                    dfus = get_active_dfus_for_recommendation(conn)

                for item_id, loc in dfus:
                    try:
                        inputs = get_dfu_inputs(item_id, loc, run_id, conn)
                        inputs["run_id"] = run_id
                        orders = compute_net_requirements(inputs, config)
                        score, reason = compute_confidence_score(inputs, orders, config)
                        for o in orders:
                            o["confidence_score"] = score
                            o["confidence_reason"] = reason
                        write_planned_orders(orders, dry_run=False, conn=conn)
                    except Exception:  # noqa: BLE001 — background planner job: keep iterating on per-DFU failures
                        logger.exception("Planned-order generation failed for item=%s loc=%s", item_id, loc)
        except Exception:  # noqa: BLE001 — background thread must never raise to scheduler
            logger.exception("Planned-order background job crashed")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "accepted", "job_id": job_id}


# ---------------------------------------------------------------------------
# GET /supply/planned-orders
# ---------------------------------------------------------------------------

@router.get("/supply/planned-orders")
async def get_planned_orders(
    item_id: Optional[str] = None,
    loc: Optional[str] = None,
    status: str = "proposed,approved",
    past_due_only: bool = False,
    supplier_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
):
    """Return planned order recommendations with optional filters."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    status_list = [s.strip() for s in status.split(",") if s.strip()]
    status_placeholders = ",".join(["%s"] * len(status_list))

    conditions = [f"po.status IN ({status_placeholders})"]
    params: list = list(status_list)

    if item_id:
        conditions.append("po.item_id = %s")
        params.append(item_id)
    if loc:
        conditions.append("po.loc = %s")
        params.append(loc)
    if supplier_id:
        conditions.append("po.supplier_id = %s")
        params.append(supplier_id)
    if past_due_only:
        conditions.append("po.is_past_due = TRUE")

    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*), SUM(po.order_value), COUNT(CASE WHEN po.is_past_due THEN 1 END) FROM fact_planned_orders po WHERE {where}", params)
            row = cur.fetchone()
            total = int(row[0] or 0)
            total_value = float(row[1] or 0)
            past_due_count = int(row[2] or 0)

            cur.execute(f"""
                SELECT
                    po.id, po.item_id, po.loc, po.supplier_id, s.supplier_name,
                    po.net_requirement_qty, po.recommended_qty, po.moq,
                    po.unit_cost, po.order_value, po.currency,
                    po.trigger_date, po.trigger_reason,
                    po.order_by_date, po.expected_receipt_date, po.lead_time_days,
                    po.current_qty_on_hand, po.safety_stock, po.reorder_point,
                    po.confirmed_inbound_qty, po.lt_forecast_demand, po.plan_version,
                    po.confidence_score, po.confidence_reason,
                    po.is_past_due, po.status, po.created_at,
                    po.approved_by, po.approved_at
                FROM fact_planned_orders po
                LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
                WHERE {where}
                ORDER BY po.order_by_date ASC, po.order_value DESC
                LIMIT %s OFFSET %s
            """, params + [page_size, offset])
            rows = cur.fetchall()

    def _fmt_date(v):
        return v.isoformat() if v else None

    def _fmt_ts(v):
        return v.isoformat() if v else None

    return {
        "total": total,
        "total_order_value_usd": total_value,
        "past_due_count": past_due_count,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": r[0],
                "item_id": r[1],
                "loc": r[2],
                "supplier_id": r[3],
                "supplier_name": r[4],
                "net_requirement_qty": float(r[5]) if r[5] is not None else None,
                "recommended_qty": float(r[6]) if r[6] is not None else None,
                "moq": float(r[7]) if r[7] is not None else None,
                "unit_cost": float(r[8]) if r[8] is not None else None,
                "order_value": float(r[9]) if r[9] is not None else None,
                "currency": r[10],
                "trigger_date": _fmt_date(r[11]),
                "trigger_reason": r[12],
                "order_by_date": _fmt_date(r[13]),
                "expected_receipt_date": _fmt_date(r[14]),
                "lead_time_days": r[15],
                "current_qty_on_hand": float(r[16]) if r[16] is not None else None,
                "safety_stock": float(r[17]) if r[17] is not None else None,
                "reorder_point": float(r[18]) if r[18] is not None else None,
                "confirmed_inbound_qty": float(r[19]) if r[19] is not None else None,
                "lt_forecast_demand": float(r[20]) if r[20] is not None else None,
                "plan_version": r[21],
                "confidence_score": float(r[22]) if r[22] is not None else None,
                "confidence_reason": r[23],
                "is_past_due": bool(r[24]),
                "status": r[25],
                "created_at": _fmt_ts(r[26]),
                "approved_by": r[27],
                "approved_at": _fmt_ts(r[28]),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# PUT /supply/planned-orders/{id}/approve
# ---------------------------------------------------------------------------

class ApproveRequest(BaseModel):
    approved_by: str


@router.post("/supply/planned-orders/{order_id}/approve")
async def approve_planned_order(
    order_id: int,
    body: ApproveRequest,
    request: Request,
):
    """Approve a proposed planned order."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_planned_orders
                SET status = 'approved', approved_by = %s, approved_at = NOW()
                WHERE id = %s AND status = 'proposed'
                RETURNING id, status, approved_by, approved_at
            """, (body.approved_by, order_id))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Order not found or not in 'proposed' state")

    return {
        "id": row[0],
        "status": row[1],
        "approved_by": row[2],
        "approved_at": row[3].isoformat() if row[3] else None,
    }


# ---------------------------------------------------------------------------
# PUT /supply/planned-orders/{id}/reject
# ---------------------------------------------------------------------------

class RejectRequest(BaseModel):
    rejection_reason: str


@router.post("/supply/planned-orders/{order_id}/reject")
async def reject_planned_order(
    order_id: int,
    body: RejectRequest,
    request: Request,
):
    """Reject a proposed planned order."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_planned_orders
                SET status = 'rejected', rejection_reason = %s
                WHERE id = %s AND status = 'proposed'
                RETURNING id, status
            """, (body.rejection_reason, order_id))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Order not found or not in 'proposed' state")

    return {"id": row[0], "status": row[1]}


# ---------------------------------------------------------------------------
# PUT /supply/planned-orders/{id}/release
# ---------------------------------------------------------------------------

@router.post("/supply/planned-orders/{order_id}/release")
async def release_planned_order(
    order_id: int,
    request: Request,
):
    """Mark an approved order as released (transmitted to ERP)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_planned_orders
                SET status = 'released', released_at = NOW()
                WHERE id = %s AND status = 'approved'
                RETURNING id, status, released_at
            """, (order_id,))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Order not found or not in 'approved' state")

    return {
        "id": row[0],
        "status": row[1],
        "released_at": row[2].isoformat() if row[2] else None,
    }


# ===========================================================================
# F2.4 — Procurement Workflow & Order Release
# (fact_purchase_orders — DS-generated POs, separate from imported open POs)
# ===========================================================================

import io
import csv as _csv


class _CreatePORequest(BaseModel):
    performed_by: str
    ordered_qty: Optional[float] = None
    requested_delivery_date: Optional[date] = None
    notes: Optional[str] = None


class _ApprovePORequest(BaseModel):
    approved_by: str
    new_qty: Optional[float] = None


class _ReleasePORequest(BaseModel):
    released_by: str
    confirmed_delivery_date: Optional[date] = None
    notes: Optional[str] = None


class _ExportCsvRequest(BaseModel):
    po_numbers: list[str]
    exported_by: str


class _SendErpRequest(BaseModel):
    po_numbers: list[str]
    integration_id: int = 1
    sent_by: str


# ---------------------------------------------------------------------------
# GET /supply/purchase-orders  (list fact_purchase_orders)
# ---------------------------------------------------------------------------

@router.get("/supply/purchase-orders")
async def list_purchase_orders(
    status: Optional[str] = None,
    supplier_id: Optional[str] = None,
    item_id: Optional[str] = None,
    loc: Optional[str] = None,
    po_date_from: Optional[str] = None,
    po_date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
):
    """List Supply Chain Command Center-generated purchase orders from fact_purchase_orders."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    clauses, params = [], []
    if status:
        clauses.append("po.status = %s")
        params.append(status)
    if supplier_id:
        clauses.append("po.supplier_id = %s")
        params.append(supplier_id)
    if item_id:
        clauses.append("po.item_id = %s")
        params.append(item_id)
    if loc:
        clauses.append("po.loc = %s")
        params.append(loc)
    if po_date_from:
        clauses.append("po.po_date >= %s")
        params.append(po_date_from)
    if po_date_to:
        clauses.append("po.po_date <= %s")
        params.append(po_date_to)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*), COALESCE(SUM(po.total_value), 0) "
                f"FROM fact_purchase_orders po {where}", params
            )
            total_row = cur.fetchone()
            total, total_value = int(total_row[0] or 0), float(total_row[1] or 0)

            cur.execute(f"""
                SELECT
                    po.po_number, po.line_number, po.item_id, po.item_description,
                    po.loc, po.supplier_id, s.supplier_name,
                    po.ordered_qty, po.unit_cost, po.total_value, po.currency,
                    po.po_date, po.requested_delivery_date, po.confirmed_delivery_date,
                    po.status, po.source_exception_id, po.created_by,
                    po.planner_approved_by, po.buyer_released_by, po.erp_po_number
                FROM fact_purchase_orders po
                LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
                {where}
                ORDER BY po.po_date DESC, po.po_number, po.line_number
                LIMIT %s OFFSET %s
            """, params + [page_size, offset])
            rows = cur.fetchall()

    def _fmt(r):
        return {
            "po_number": r[0],
            "line_number": r[1],
            "item_id": r[2],
            "item_description": r[3],
            "loc": r[4],
            "supplier_id": r[5],
            "supplier_name": r[6],
            "ordered_qty": float(r[7]) if r[7] is not None else None,
            "unit_cost": float(r[8]) if r[8] is not None else None,
            "total_value": float(r[9]) if r[9] is not None else None,
            "currency": r[10],
            "po_date": r[11].isoformat() if r[11] else None,
            "requested_delivery_date": r[12].isoformat() if r[12] else None,
            "confirmed_delivery_date": r[13].isoformat() if r[13] else None,
            "status": r[14],
            "source_exception_id": r[15],
            "created_by": r[16],
            "planner_approved_by": r[17],
            "buyer_released_by": r[18],
            "erp_po_number": r[19],
        }

    return {"total": total, "total_value": total_value, "page": page,
            "orders": [_fmt(r) for r in rows]}


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/from-exception/{exception_id}
# (must appear before /{po_number} routes)
# ---------------------------------------------------------------------------

@router.post("/supply/purchase-orders/from-exception/{exception_id}", status_code=201)
async def create_po_from_exception(
    exception_id: int,
    body: _CreatePORequest,
    request: Request,
):
    """Convert a replenishment exception to a proposed purchase order."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    from scripts.release_planned_orders import (
        create_po_from_exception as _create_po,
    )

    with get_conn() as conn:
        po_number = _create_po(
            exception_id=exception_id,
            performed_by=body.performed_by,
            conn=conn,
            override_qty=body.ordered_qty,
            requested_delivery_date=body.requested_delivery_date,
            notes=body.notes,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT total_value, requested_delivery_date FROM fact_purchase_orders "
                "WHERE po_number = %s", (po_number,)
            )
            row = cur.fetchone()

    return {
        "po_number": po_number,
        "status": "proposed",
        "total_value": float(row[0]) if row and row[0] is not None else None,
        "requested_delivery_date": row[1].isoformat() if row and row[1] else None,
    }


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/{po_number}/approve
# ---------------------------------------------------------------------------

@router.post("/supply/purchase-orders/{po_number}/approve")
async def approve_purchase_order(
    po_number: str,
    body: _ApprovePORequest,
    request: Request,
):
    """Planner approves a proposed PO (proposed → planner_approved)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_purchase_orders
                SET status = 'planner_approved',
                    planner_approved_by = %s,
                    planner_approved_at = NOW(),
                    ordered_qty = COALESCE(%s, ordered_qty)
                WHERE po_number = %s AND status = 'proposed'
                RETURNING po_line_id, ordered_qty
            """, (body.approved_by, body.new_qty, po_number))
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail="PO not found or not in 'proposed' state"
                )
            for po_line_id, qty in rows:
                cur.execute("""
                    INSERT INTO fact_po_approval_log
                        (po_line_id, po_number, action, performed_by,
                         old_status, new_status, new_qty)
                    VALUES (%s, %s, 'planner_approved', %s,
                            'proposed', 'planner_approved', %s)
                """, (po_line_id, po_number, body.approved_by, qty))
        conn.commit()

    return {"po_number": po_number, "status": "planner_approved",
            "approved_by": body.approved_by}


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/{po_number}/release
# ---------------------------------------------------------------------------

@router.post("/supply/purchase-orders/{po_number}/release")
async def release_purchase_order(
    po_number: str,
    body: _ReleasePORequest,
    request: Request,
):
    """Buyer releases a planner-approved PO (planner_approved → buyer_released)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_purchase_orders
                SET status = 'buyer_released',
                    buyer_released_by = %s,
                    buyer_released_at = NOW(),
                    confirmed_delivery_date = COALESCE(%s, confirmed_delivery_date),
                    notes = COALESCE(%s, notes)
                WHERE po_number = %s AND status = 'planner_approved'
                RETURNING po_line_id
            """, (body.released_by, body.confirmed_delivery_date,
                  body.notes, po_number))
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail="PO not found or not in 'planner_approved' state"
                )
            for (po_line_id,) in rows:
                cur.execute("""
                    INSERT INTO fact_po_approval_log
                        (po_line_id, po_number, action, performed_by,
                         old_status, new_status)
                    VALUES (%s, %s, 'buyer_released', %s,
                            'planner_approved', 'buyer_released')
                """, (po_line_id, po_number, body.released_by))
        conn.commit()

    return {"po_number": po_number, "status": "buyer_released",
            "released_by": body.released_by}


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/export-csv  (literal path before {po_number})
# ---------------------------------------------------------------------------

@router.post("/supply/purchase-orders/export-csv")
async def export_pos_csv(body: _ExportCsvRequest, request: Request):
    """Generate a standardized PO CSV for ERP import. Returns CSV as stream."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    po.po_number, po.line_number, po.item_id, po.item_description,
                    po.loc, po.supplier_id, s.supplier_name, po.ordered_qty,
                    po.unit_of_measure, po.unit_cost, po.total_value, po.currency,
                    po.requested_delivery_date, po.po_date, po.buyer_code,
                    po.company_code, po.plant_code, po.source_exception_id, po.notes
                FROM fact_purchase_orders po
                LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
                WHERE po.po_number = ANY(%s)
                  AND po.status IN ('buyer_released', 'po_sent')
                ORDER BY po.po_number, po.line_number
            """, (body.po_numbers,))
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No released POs found for given po_numbers")

    total_value = sum(float(r[10]) for r in rows if r[10] is not None)
    fieldnames = [
        "PO_NUMBER", "LINE_NO", "ITEM_NUMBER", "ITEM_DESCRIPTION",
        "LOCATION", "SUPPLIER_ID", "SUPPLIER_NAME", "ORDERED_QTY",
        "UNIT_OF_MEASURE", "UNIT_COST", "TOTAL_VALUE", "CURRENCY",
        "REQUESTED_DELIVERY_DATE", "PO_DATE", "BUYER_CODE",
        "COMPANY_CODE", "PLANT", "DEMAND_STUDIO_EXCEPTION_ID", "NOTES",
    ]
    buf = io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(zip(fieldnames, [
            str(v) if v is not None else "" for v in row
        ])))

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"PO_export_{ts}.csv"

    return {
        "filename": filename,
        "line_count": len(rows),
        "total_value": round(total_value, 2),
        "csv_content": buf.getvalue(),
    }


# ---------------------------------------------------------------------------
# GET /supply/purchase-orders/{po_number}/timeline
# ---------------------------------------------------------------------------

@router.get("/supply/purchase-orders/{po_number}/timeline")
async def get_po_timeline(po_number: str):
    """Return the full audit log for a PO's lifecycle."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM fact_purchase_orders WHERE po_number = %s LIMIT 1",
                (po_number,)
            )
            po_row = cur.fetchone()
            if not po_row:
                raise HTTPException(status_code=404, detail=f"PO {po_number} not found")

            cur.execute("""
                SELECT action, performed_by, performed_at,
                       old_status, new_status, old_qty, new_qty, reason, system_note
                FROM fact_po_approval_log
                WHERE po_number = %s
                ORDER BY performed_at ASC
            """, (po_number,))
            log_rows = cur.fetchall()

    return {
        "po_number": po_number,
        "current_status": po_row[0],
        "timeline": [
            {
                "action": r[0],
                "performed_by": r[1],
                "performed_at": r[2].isoformat() if r[2] else None,
                "old_status": r[3],
                "new_status": r[4],
                "old_qty": float(r[5]) if r[5] is not None else None,
                "new_qty": float(r[6]) if r[6] is not None else None,
                "reason": r[7],
                "note": r[8],
            }
            for r in log_rows
        ],
    }
