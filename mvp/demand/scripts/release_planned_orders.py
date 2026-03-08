"""
F2.4 — Procurement Workflow & Order Release

Convert approved exception recommendations into purchase orders and
optionally send them to an ERP system.

Usage:
    # Create PO from a replenishment exception
    uv run scripts/release_planned_orders.py \\
        --exception-id 7834 \\
        --action create_po \\
        --performed-by jane.smith@company.com

    # Approve a proposed PO
    uv run scripts/release_planned_orders.py \\
        --po-number DS-2026-04-001 \\
        --action approve \\
        --performed-by jane.smith@company.com

    # Release an approved PO for ERP send
    uv run scripts/release_planned_orders.py \\
        --po-number DS-2026-04-001 \\
        --action release \\
        --performed-by bob.chen@company.com

    # Export released POs to CSV
    uv run scripts/release_planned_orders.py \\
        --po-numbers DS-2026-04-001 DS-2026-04-002 \\
        --action export_csv \\
        --output-dir data/po_exports/

    # Send released POs to ERP via REST API
    uv run scripts/release_planned_orders.py \\
        --po-numbers DS-2026-04-001 \\
        --action send_erp \\
        --integration-id 1

Config: config/procurement_config.yaml
"""

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import psycopg
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.db import get_db_params

_cfg = yaml.safe_load(open("config/procurement_config.yaml"))
_LT_FALLBACK = _cfg["procurement"]["lead_time_fallback_days"]


# ---------------------------------------------------------------------------
# PO Number Generation
# ---------------------------------------------------------------------------

def generate_po_number(conn: psycopg.Connection) -> str:
    """
    Generate a sequential, collision-safe PO number.
    Format: DS-{YYYY}-{MM}-{ZERO_PADDED_SEQ}
    Example: DS-2026-04-001, DS-2026-04-002

    Uses a PostgreSQL sequence scoped to year+month.
    """
    now = datetime.utcnow()
    sequence_name = f"po_seq_{now.year}_{now.month:02d}"
    with conn.cursor() as cur:
        cur.execute("""
            DO $$
            DECLARE
                seq_name TEXT := %s;
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_sequences WHERE sequencename = seq_name
                ) THEN
                    EXECUTE 'CREATE SEQUENCE ' || quote_ident(seq_name) || ' START 1';
                END IF;
            END$$;
        """, (sequence_name,))
        cur.execute("SELECT nextval(%s)", (sequence_name,))
        seq_val = cur.fetchone()[0]
    return f"DS-{now.year}-{now.month:02d}-{seq_val:03d}"


# ---------------------------------------------------------------------------
# PO Creation
# ---------------------------------------------------------------------------

def create_po_from_exception(
    exception_id: int,
    performed_by: str,
    conn: psycopg.Connection,
    override_qty: float | None = None,
    requested_delivery_date: date | None = None,
    notes: str | None = None,
) -> str:
    """
    Read a replenishment exception and create a proposed PO.

    Returns:
        po_number of the created PO
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.item_no, e.loc, e.recommended_order_qty,
                   e.recommended_reorder_date,
                   i.item_description,
                   s.supplier_id, s.default_lead_time_days, s.currency
            FROM fact_replenishment_exceptions e
            LEFT JOIN dim_item i ON i.item_no = e.item_no
            LEFT JOIN dim_item_supplier ims ON ims.item_no = e.item_no
                AND ims.loc = e.loc AND ims.is_primary = TRUE
            LEFT JOIN dim_supplier s ON s.supplier_id = ims.supplier_id
            WHERE e.id = %s
        """, (exception_id,))
        row = cur.fetchone()

    if not row:
        raise ValueError(f"Exception {exception_id} not found")

    item_no, loc, rec_qty, reorder_date, item_desc, supplier_id, lead_time, currency = row

    ordered_qty = override_qty if override_qty is not None else float(rec_qty or 0)
    lt_days = lead_time or _LT_FALLBACK
    delivery_date = (
        requested_delivery_date
        or (date.today() + __import__("datetime").timedelta(days=lt_days))
    )
    currency = currency or _cfg["procurement"]["default_currency"]

    po_number = generate_po_number(conn)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fact_purchase_orders (
                po_number, line_number, item_no, item_description, loc,
                supplier_id, ordered_qty, unit_of_measure, currency,
                po_date, requested_delivery_date, status,
                source_exception_id, created_by, notes
            ) VALUES (
                %s, 1, %s, %s, %s,
                %s, %s, %s, %s,
                CURRENT_DATE, %s, 'proposed',
                %s, %s, %s
            )
            RETURNING po_line_id
        """, (
            po_number, item_no, item_desc, loc,
            supplier_id, ordered_qty,
            _cfg["procurement"]["default_unit_of_measure"], currency,
            delivery_date, exception_id, performed_by, notes,
        ))
        po_line_id = cur.fetchone()[0]

        # Audit log entry
        cur.execute("""
            INSERT INTO fact_po_approval_log
                (po_line_id, po_number, action, performed_by,
                 new_status, new_qty, system_note)
            VALUES (%s, %s, 'proposed', %s,
                    'proposed', %s, %s)
        """, (
            po_line_id, po_number, performed_by, ordered_qty,
            f"Auto-created from exception EXC-{exception_id}",
        ))

    conn.commit()
    print(f"[PO CREATED] {po_number} from exception {exception_id}")
    return po_number


# ---------------------------------------------------------------------------
# PO Approval (Planner)
# ---------------------------------------------------------------------------

def approve_po(
    po_number: str,
    approved_by: str,
    conn: psycopg.Connection,
    new_qty: float | None = None,
) -> None:
    """
    Planner approves a proposed PO (optionally adjusting quantity).
    proposed → planner_approved
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE fact_purchase_orders
            SET status = 'planner_approved',
                planner_approved_by = %s,
                planner_approved_at = NOW(),
                ordered_qty = COALESCE(%s, ordered_qty)
            WHERE po_number = %s AND status = 'proposed'
            RETURNING po_line_id, ordered_qty
        """, (approved_by, new_qty, po_number))
        rows = cur.fetchall()

        if not rows:
            raise ValueError(f"PO {po_number} not found or not in 'proposed' state")

        for po_line_id, qty in rows:
            cur.execute("""
                INSERT INTO fact_po_approval_log
                    (po_line_id, po_number, action, performed_by,
                     old_status, new_status, new_qty)
                VALUES (%s, %s, 'planner_approved', %s,
                        'proposed', 'planner_approved', %s)
            """, (po_line_id, po_number, approved_by, qty))

    conn.commit()
    print(f"[PO APPROVED] {po_number} by {approved_by}")


# ---------------------------------------------------------------------------
# PO Release (Buyer)
# ---------------------------------------------------------------------------

def release_po(
    po_number: str,
    released_by: str,
    conn: psycopg.Connection,
    confirmed_delivery_date: date | None = None,
    notes: str | None = None,
) -> None:
    """
    Buyer releases a planner-approved PO.
    planner_approved → buyer_released
    """
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
        """, (released_by, confirmed_delivery_date, notes, po_number))
        rows = cur.fetchall()

        if not rows:
            raise ValueError(
                f"PO {po_number} not found or not in 'planner_approved' state"
            )

        for (po_line_id,) in rows:
            cur.execute("""
                INSERT INTO fact_po_approval_log
                    (po_line_id, po_number, action, performed_by,
                     old_status, new_status)
                VALUES (%s, %s, 'buyer_released', %s,
                        'planner_approved', 'buyer_released')
            """, (po_line_id, po_number, released_by))

    conn.commit()
    print(f"[PO RELEASED] {po_number} by {released_by}")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "PO_NUMBER", "LINE_NO", "ITEM_NUMBER", "ITEM_DESCRIPTION",
    "LOCATION", "SUPPLIER_ID", "SUPPLIER_NAME", "ORDERED_QTY",
    "UNIT_OF_MEASURE", "UNIT_COST", "TOTAL_VALUE", "CURRENCY",
    "REQUESTED_DELIVERY_DATE", "PO_DATE", "BUYER_CODE",
    "COMPANY_CODE", "PLANT", "DEMAND_STUDIO_EXCEPTION_ID", "NOTES",
]


def export_pos_to_csv(
    po_numbers: list[str],
    output_path: Path,
    conn: psycopg.Connection,
) -> int:
    """
    Generate a standardized PO CSV for ERP import.

    Returns:
        Number of lines written to CSV
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                po.po_number, po.line_number, po.item_no, po.item_description,
                po.loc, po.supplier_id, s.supplier_name, po.ordered_qty,
                po.unit_of_measure, po.unit_cost, po.total_value, po.currency,
                po.requested_delivery_date, po.po_date, po.buyer_code,
                po.company_code, po.plant_code, po.source_exception_id, po.notes
            FROM fact_purchase_orders po
            LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
            WHERE po.po_number = ANY(%s)
              AND po.status IN ('buyer_released', 'po_sent')
            ORDER BY po.po_number, po.line_number
        """, (po_numbers,))
        rows = cur.fetchall()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(FIELDNAMES, [
                str(v) if v is not None else "" for v in row
            ])))

    return len(rows)


# ---------------------------------------------------------------------------
# ERP Field Mapping
# ---------------------------------------------------------------------------

def _map_fields(po_data: tuple, field_mapping: dict | None) -> dict:
    """Apply ERP field mapping from dim_erp_integration.field_mapping JSONB."""
    ds_fields = [
        "po_number", "item_no", "loc", "supplier_id", "ordered_qty",
        "unit_of_measure", "unit_cost", "currency",
        "requested_delivery_date", "po_date", "buyer_code",
        "company_code", "plant_code",
    ]
    ds_dict = dict(zip(ds_fields, po_data))
    mapping = field_mapping or {}
    return {
        mapping.get(k, k): str(v) if isinstance(v, date) else v
        for k, v in ds_dict.items()
        if v is not None
    }


# ---------------------------------------------------------------------------
# ERP Send (Tier B: REST API)
# ---------------------------------------------------------------------------

def send_po_to_erp(
    po_number: str,
    integration_id: int,
    conn: psycopg.Connection,
) -> dict:
    """
    Send a released PO to ERP via REST API integration.

    Returns:
        dict with keys: success (bool), erp_po_number (str|None), error (str|None)
    """
    import httpx  # import here to keep it optional

    with conn.cursor() as cur:
        cur.execute("""
            SELECT erp_type, endpoint_url, auth_method, field_mapping, auth_credential_ref
            FROM dim_erp_integration WHERE integration_id = %s AND active = TRUE
        """, (integration_id,))
        integration = cur.fetchone()

    if not integration:
        return {"success": False, "erp_po_number": None, "error": "Integration not found"}

    erp_type, endpoint_url, auth_method, field_mapping, _ = integration

    if erp_type == "csv_export":
        return {"success": False, "erp_po_number": None,
                "error": "Use export_csv action for CSV export integration"}

    with conn.cursor() as cur:
        cur.execute("""
            SELECT po_number, item_no, loc, supplier_id, ordered_qty,
                   unit_of_measure, unit_cost, currency,
                   requested_delivery_date, po_date, buyer_code,
                   company_code, plant_code
            FROM fact_purchase_orders WHERE po_number = %s
        """, (po_number,))
        po_data = cur.fetchone()

    if not po_data:
        return {"success": False, "erp_po_number": None, "error": "PO not found"}

    payload = _map_fields(po_data, field_mapping)

    try:
        timeout = _cfg["procurement"]["erp_integration"]["timeout_seconds"]
        response = httpx.post(
            endpoint_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=float(timeout),
        )
        resp_json = response.json()
        erp_po_number = resp_json.get("po_number") or resp_json.get("EBELN")
        success = response.status_code in (200, 201)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_purchase_orders
                SET status = 'po_sent',
                    erp_po_number = %s,
                    erp_sent_at = NOW(),
                    erp_response_code = %s,
                    erp_response_payload = %s::jsonb,
                    erp_integration_type = %s
                WHERE po_number = %s
            """, (erp_po_number, str(response.status_code),
                  json.dumps(resp_json), erp_type, po_number))
        conn.commit()

        return {"success": success, "erp_po_number": erp_po_number, "error": None}

    except Exception as e:
        error_msg = str(e)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_purchase_orders
                SET erp_response_payload = %s::jsonb
                WHERE po_number = %s
            """, (json.dumps({"error": error_msg}), po_number))
        conn.commit()
        return {"success": False, "erp_po_number": None, "error": error_msg}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Procurement Workflow CLI (F2.4)")
    parser.add_argument("--action", required=True,
                        choices=["create_po", "approve", "release", "export_csv", "send_erp"])
    parser.add_argument("--exception-id", type=int, help="Exception ID to create PO from")
    parser.add_argument("--po-number", help="PO number for approve/release/send actions")
    parser.add_argument("--po-numbers", nargs="+", help="PO numbers for batch actions")
    parser.add_argument("--performed-by", default="system", help="User performing the action")
    parser.add_argument("--new-qty", type=float, help="Override quantity when approving")
    parser.add_argument("--output-dir", default="data/po_exports/", help="CSV output dir")
    parser.add_argument("--integration-id", type=int, default=1, help="ERP integration ID")
    parser.add_argument("--dry-run", action="store_true", help="Preview without committing")
    args = parser.parse_args()

    with psycopg.connect(**get_db_params()) as conn:
        if args.dry_run:
            conn.autocommit = False

        if args.action == "create_po":
            if not args.exception_id:
                parser.error("--exception-id required for create_po action")
            po_number = create_po_from_exception(
                args.exception_id, args.performed_by, conn
            )
            print(f"Created PO: {po_number}")
            if args.dry_run:
                conn.rollback()

        elif args.action == "approve":
            if not args.po_number:
                parser.error("--po-number required for approve action")
            approve_po(args.po_number, args.performed_by, conn, args.new_qty)
            if args.dry_run:
                conn.rollback()

        elif args.action == "release":
            if not args.po_number:
                parser.error("--po-number required for release action")
            release_po(args.po_number, args.performed_by, conn)
            if args.dry_run:
                conn.rollback()

        elif args.action == "export_csv":
            po_numbers = args.po_numbers or ([args.po_number] if args.po_number else [])
            if not po_numbers:
                parser.error("--po-numbers required for export_csv action")
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"PO_export_{ts}.csv"
            output_path = Path(args.output_dir) / filename
            n = export_pos_to_csv(po_numbers, output_path, conn)
            print(f"Exported {n} lines to {output_path}")

        elif args.action == "send_erp":
            po_numbers = args.po_numbers or ([args.po_number] if args.po_number else [])
            if not po_numbers:
                parser.error("--po-numbers required for send_erp action")
            for pn in po_numbers:
                result = send_po_to_erp(pn, args.integration_id, conn)
                status = "OK" if result["success"] else "FAILED"
                print(f"[{status}] {pn} → ERP PO: {result['erp_po_number']} / {result['error']}")


if __name__ == "__main__":
    main()
