"""
F1.3 — load_open_pos.py

Loads open purchase order data from CSV exports into:
  - dim_supplier (from suppliers_*.csv)
  - fact_open_purchase_orders (from open_pos_*.csv)
  - fact_po_receipts (from po_receipts_*.csv)

Reconciles received_qty on fact_open_purchase_orders from fact_po_receipts
after each load.

Usage:
    uv run python scripts/load_open_pos.py [--dry-run]
    uv run python scripts/load_open_pos.py --file datafiles/open_pos_20260306.csv
    uv run python scripts/load_open_pos.py --receipts --file datafiles/po_receipts_20260306.csv
    uv run python scripts/load_open_pos.py --suppliers --file datafiles/suppliers_20260306.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import psycopg
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.db import get_db_params

CONFIG_PATH = "config/po_integration_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _parse_date(val) -> date | None:
    if pd.isna(val) or val == "" or val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return pd.to_datetime(str(val)).date()
    except Exception:
        return None


def validate_po_row(row: dict, config: dict) -> tuple[bool, str]:
    """Returns (is_valid, reason). Rejects rows that fail business rules."""
    # Must have at least one delivery date
    dates = [row.get("promised_delivery_date"), row.get("confirmed_delivery_date"),
             row.get("revised_delivery_date")]
    if not any(d for d in dates if d is not None):
        return False, "no_delivery_date"

    # Skip closed/cancelled lines
    if row.get("line_status") in ("closed", "cancelled"):
        return False, "line_closed_or_cancelled"

    # Reject POs too far past due
    max_past_due = config["validation"]["reject_pos_past_due_days"]
    eff_date = (row.get("revised_delivery_date")
                or row.get("confirmed_delivery_date")
                or row.get("promised_delivery_date"))
    if eff_date and (date.today() - eff_date).days > max_past_due:
        return False, f"past_due_exceeds_{max_past_due}_days"

    # Open qty must be positive
    confirmed_or_ordered = row.get("confirmed_qty") or row.get("ordered_qty") or 0
    received = row.get("received_qty") or 0
    open_qty = confirmed_or_ordered - received
    if open_qty <= 0:
        return False, "open_qty_zero_or_negative"

    return True, "ok"


def load_suppliers(filepath: str, conn, dry_run: bool) -> int:
    """Upsert supplier rows from CSV. Returns row count."""
    df = pd.read_csv(filepath, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    count = 0
    sql = """
        INSERT INTO dim_supplier
            (supplier_id, supplier_name, country_code, address_line1, city,
             state_province, postal_code, payment_terms, default_lead_time_days,
             reliability_score, on_time_pct, is_active, modified_ts)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (supplier_id) DO UPDATE SET
            supplier_name           = EXCLUDED.supplier_name,
            country_code            = EXCLUDED.country_code,
            payment_terms           = EXCLUDED.payment_terms,
            default_lead_time_days  = EXCLUDED.default_lead_time_days,
            reliability_score       = EXCLUDED.reliability_score,
            on_time_pct             = EXCLUDED.on_time_pct,
            is_active               = EXCLUDED.is_active,
            modified_ts             = NOW()
    """
    for _, row in df.iterrows():
        is_active = str(row.get("is_active", "true")).lower() in ("true", "1", "yes")
        params = (
            row.get("supplier_id"), row.get("supplier_name"),
            row.get("country_code") or None, row.get("address_line1") or None,
            row.get("city") or None, row.get("state_province") or None,
            row.get("postal_code") or None, row.get("payment_terms") or None,
            int(row["default_lead_time_days"]) if not pd.isna(row.get("default_lead_time_days", "")) else None,
            float(row["reliability_score"]) if not pd.isna(row.get("reliability_score", "")) else None,
            float(row["on_time_pct"]) if not pd.isna(row.get("on_time_pct", "")) else None,
            is_active,
        )
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        count += 1

    if not dry_run:
        conn.commit()
    print(f"  Suppliers: {count} rows upserted{' (dry-run)' if dry_run else ''}")
    return count


def load_pos(filepath: str, conn, dry_run: bool, config: dict) -> tuple[int, int, dict]:
    """Load PO lines from CSV. Returns (loaded, skipped, rejection_reasons)."""
    df = pd.read_csv(filepath, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    loaded = 0
    skipped = 0
    reasons: dict[str, int] = {}

    sql = """
        INSERT INTO fact_open_purchase_orders
            (po_number, po_line_number, item_no, loc, supplier_id, po_date,
             ordered_qty, confirmed_qty, received_qty, unit_cost, currency,
             promised_delivery_date, confirmed_delivery_date, revised_delivery_date,
             po_status, line_status, source_file, modified_ts)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (po_number, po_line_number) DO UPDATE SET
            confirmed_qty           = EXCLUDED.confirmed_qty,
            received_qty            = EXCLUDED.received_qty,
            unit_cost               = EXCLUDED.unit_cost,
            revised_delivery_date   = EXCLUDED.revised_delivery_date,
            po_status               = EXCLUDED.po_status,
            line_status             = EXCLUDED.line_status,
            modified_ts             = NOW()
    """

    for _, row in df.iterrows():
        r = {
            "po_number": row.get("po_number"),
            "po_line_number": int(row.get("po_line_number", 1)),
            "item_no": row.get("item_no"),
            "loc": row.get("loc"),
            "supplier_id": row.get("supplier_id") or None,
            "po_date": _parse_date(row.get("po_date")),
            "ordered_qty": float(row.get("ordered_qty") or 0),
            "confirmed_qty": float(row["confirmed_qty"]) if not pd.isna(row.get("confirmed_qty", "")) else None,
            "received_qty": float(row.get("received_qty") or 0),
            "unit_cost": float(row["unit_cost"]) if not pd.isna(row.get("unit_cost", "")) else None,
            "currency": row.get("currency") or "USD",
            "promised_delivery_date": _parse_date(row.get("promised_delivery_date")),
            "confirmed_delivery_date": _parse_date(row.get("confirmed_delivery_date")),
            "revised_delivery_date": _parse_date(row.get("revised_delivery_date")),
            "po_status": row.get("po_status") or "open",
            "line_status": row.get("line_status") or "open",
        }

        valid, reason = validate_po_row(r, config)
        if not valid:
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    r["po_number"], r["po_line_number"], r["item_no"], r["loc"],
                    r["supplier_id"], r["po_date"], r["ordered_qty"], r["confirmed_qty"],
                    r["received_qty"], r["unit_cost"], r["currency"],
                    r["promised_delivery_date"], r["confirmed_delivery_date"],
                    r["revised_delivery_date"], r["po_status"], r["line_status"],
                    os.path.basename(filepath),
                ))
        loaded += 1

    if not dry_run:
        conn.commit()
    return loaded, skipped, reasons


def load_receipts(filepath: str, conn, dry_run: bool) -> int:
    """Load goods receipt postings from CSV. Returns row count."""
    df = pd.read_csv(filepath, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    count = 0
    sql = """
        INSERT INTO fact_po_receipts
            (receipt_number, po_number, po_line_number, item_no, loc,
             received_qty, unit_cost, actual_receipt_date, receipt_status, source_file)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (receipt_number, po_number, po_line_number) DO UPDATE SET
            received_qty        = EXCLUDED.received_qty,
            actual_receipt_date = EXCLUDED.actual_receipt_date,
            receipt_status      = EXCLUDED.receipt_status
    """
    for _, row in df.iterrows():
        receipt_date = _parse_date(row.get("actual_receipt_date"))
        if not receipt_date:
            continue
        params = (
            row.get("receipt_number"), row.get("po_number"),
            int(row.get("po_line_number", 1)), row.get("item_no"), row.get("loc"),
            float(row.get("received_qty") or 0),
            float(row["unit_cost"]) if not pd.isna(row.get("unit_cost", "")) else None,
            receipt_date, row.get("receipt_status") or "posted",
            os.path.basename(filepath),
        )
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        count += 1

    if not dry_run:
        conn.commit()
    print(f"  Receipts: {count} rows upserted{' (dry-run)' if dry_run else ''}")
    return count


def reconcile_received_qty(conn) -> int:
    """
    After loading receipts, update fact_open_purchase_orders.received_qty
    from fact_po_receipts aggregates. Returns number of rows updated.
    """
    sql = """
        UPDATE fact_open_purchase_orders po
        SET received_qty = COALESCE(r.total_received, 0),
            line_status = CASE
                WHEN COALESCE(r.total_received, 0) >= COALESCE(po.confirmed_qty, po.ordered_qty)
                THEN 'closed'
                WHEN COALESCE(r.total_received, 0) > 0
                THEN 'partially_received'
                ELSE po.line_status
            END,
            modified_ts = NOW()
        FROM (
            SELECT po_number, po_line_number, SUM(received_qty) AS total_received
            FROM fact_po_receipts
            WHERE receipt_status = 'posted'
            GROUP BY po_number, po_line_number
        ) r
        WHERE po.po_number = r.po_number
          AND po.po_line_number = r.po_line_number
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        count = cur.rowcount
    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="Load open PO data from CSV exports")
    parser.add_argument("--file", help="Specific CSV file to load (overrides glob pattern)")
    parser.add_argument("--receipts", action="store_true", help="Load receipts file")
    parser.add_argument("--suppliers", action="store_true", help="Load suppliers file")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    config = load_config()
    dry_run = args.dry_run

    with psycopg.connect(**get_db_params()) as conn:
        if args.suppliers:
            files = [args.file] if args.file else glob.glob(config["ingest"]["csv"]["supplier_file_pattern"])
            for f in sorted(files):
                print(f"Loading suppliers from {f}...")
                load_suppliers(f, conn, dry_run)

        elif args.receipts:
            files = [args.file] if args.file else glob.glob(config["ingest"]["csv"]["receipt_file_pattern"])
            for f in sorted(files):
                print(f"Loading receipts from {f}...")
                load_receipts(f, conn, dry_run)
            if not dry_run and files:
                updated = reconcile_received_qty(conn)
                print(f"  Reconciled received_qty on {updated} PO lines.")

        else:
            # Load suppliers first (FK dependency)
            sup_files = glob.glob(config["ingest"]["csv"]["supplier_file_pattern"])
            for f in sorted(sup_files):
                print(f"Loading suppliers from {f}...")
                load_suppliers(f, conn, dry_run)

            # Load POs
            po_files = [args.file] if args.file else glob.glob(config["ingest"]["csv"]["po_file_pattern"])
            total_loaded = total_skipped = 0
            all_reasons: dict[str, int] = {}
            for f in sorted(po_files):
                print(f"Loading POs from {f}...")
                loaded, skipped, reasons = load_pos(f, conn, dry_run, config)
                total_loaded += loaded
                total_skipped += skipped
                for k, v in reasons.items():
                    all_reasons[k] = all_reasons.get(k, 0) + v

            print(f"\nPO load complete: {total_loaded} loaded, {total_skipped} skipped")
            if all_reasons:
                print("  Rejection reasons:", all_reasons)

            # Reconcile receipts
            if not dry_run:
                updated = reconcile_received_qty(conn)
                print(f"  Reconciled received_qty on {updated} PO lines.")

    print("Done." if not dry_run else "Dry-run complete — no rows written.")


if __name__ == "__main__":
    main()
