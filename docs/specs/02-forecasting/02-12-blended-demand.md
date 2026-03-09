# F1.3 — Open Purchase Order Integration

**Phase:** Evolution to Operations — Phase 1
**Feature Number:** F1.3
**Status:** Not Started
**Priority:** Critical (required for realistic inventory projection in F1.2 and order recommendations in F2.1)

---

## 1. Problem Statement

The `fact_inventory_snapshot` table has a `qty_on_hand_on_order` column. It is a single aggregated number — a snapshot of "how many units are currently on order" with no delivery date, no supplier, no PO reference number, and no split across multiple open orders.

**What this means in practice:**

> Item 100320 at LOC 1401-BULK: `qty_on_hand = 120`, `qty_on_order = 200`

The planner sees 200 units on order. But:
- Are they arriving tomorrow or in 8 weeks?
- Is it one PO or five separate orders from three suppliers?
- Has any portion already been partially received?
- What is the unit cost — is this $200 or $20,000 of capital?
- Has the supplier confirmed the delivery date or is it still "promised"?

Without this data, every inventory projection (F1.2) assumes zero inbound supply for the `with_open_po` scenario, making stockout warnings wildly inaccurate.

### Concrete Failure Scenario

March 6, 2026. Item 100320. `qty_on_order = 200`. The projection system (F1.2 without F1.3) shows:

```
No-Order scenario: stockout in 12 days (Mar 18) — CRITICAL ⚠
With-PO scenario: [DEGRADED — identical to no_order because no delivery dates]
```

The planner panics and places an emergency order for 300 units at spot pricing (30% markup = $4,500 in excess cost).

**Reality:** Two open POs already exist:
- PO-4521: 150 units, confirmed delivery March 14 (8 days away)
- PO-4522: 50 units, confirmed delivery March 21 (15 days away)

With this data, the true projection is:
```
With-PO scenario: no stockout — adequate through April 22
```

The $4,500 emergency order was unnecessary. This is a $4,500 waste that would show up in the portfolio as excess inventory, not a supply failure.

---

## 2. Input Data Required from External Systems

This feature is an **integration feature** — it requires data from systems outside Demand Studio. The data does not exist in any current table. There are three ingest strategies.

### 2.1 Required Fields from ERP (SAP/Oracle/NetSuite/etc.)

#### PO Header (one per purchase order)
| Field | ERP Source | Type | Example |
|---|---|---|---|
| `po_number` | SAP ME21N.EBELN / Oracle PO_HEADERS.PO_NUM | VARCHAR(50) | 'PO-4521' |
| `po_date` | Document date | DATE | 2026-02-15 |
| `supplier_id` | Vendor master key | VARCHAR(50) | 'VENDOR-0042' |
| `currency` | Document currency | CHAR(3) | 'USD' |
| `payment_terms` | NET30 / NET60 | VARCHAR(30) | 'NET30' |
| `status` | open / partially_received / closed / cancelled | VARCHAR(30) | 'open' |

#### PO Line (one per item-location within a PO)
| Field | ERP Source | Type | Example |
|---|---|---|---|
| `po_number` | FK to header | VARCHAR(50) | 'PO-4521' |
| `po_line_number` | Line sequence | INTEGER | 1 |
| `item_no` | Material / Item number | VARCHAR(50) | '100320' |
| `loc` | Plant / Warehouse | VARCHAR(50) | '1401-BULK' |
| `ordered_qty` | Original ordered quantity | NUMERIC(12,2) | 200.0 |
| `confirmed_qty` | Supplier-confirmed quantity | NUMERIC(12,2) | 150.0 |
| `received_qty` | Already received (from GR) | NUMERIC(12,2) | 0.0 |
| `open_qty` | confirmed_qty - received_qty | NUMERIC(12,2) | 150.0 |
| `unit_cost` | Per-unit cost in document currency | NUMERIC(12,4) | 12.5000 |
| `promised_delivery_date` | Original promise date | DATE | 2026-03-14 |
| `confirmed_delivery_date` | Supplier-acknowledged date | DATE | 2026-03-14 |
| `revised_delivery_date` | Latest supplier revision | DATE | NULL |
| `line_status` | open / partial / closed / cancelled | VARCHAR(30) | 'open' |

### 2.2 Required Fields from WMS / Goods Receipt

| Field | Source | Type | Example |
|---|---|---|---|
| `receipt_number` | WMS GR document | VARCHAR(50) | 'GR-20260314-001' |
| `po_number` | FK to PO | VARCHAR(50) | 'PO-4521' |
| `po_line_number` | FK to PO line | INTEGER | 1 |
| `item_no` | Item received | VARCHAR(50) | '100320' |
| `loc` | Receiving location | VARCHAR(50) | '1401-BULK' |
| `received_qty` | Quantity confirmed received | NUMERIC(12,2) | 150.0 |
| `actual_receipt_date` | Physical receipt date | DATE | 2026-03-14 |
| `receipt_status` | posted / reversed | VARCHAR(20) | 'posted' |

### 2.3 Ingest Strategies (choose one or combine)

| Strategy | Pros | Cons | Best For |
|---|---|---|---|
| **A: Scheduled CSV Export** | Simple, no ERP API needed, works with any ERP | Data is stale (1 export/day), manual coordination | Small-medium deployments |
| **B: ERP REST API Poll** | Near-real-time (hourly), structured | Requires ERP API access, auth setup | SAP S/4HANA OData, Oracle REST |
| **C: EDI 850/855** | Standard format, event-driven | Complex EDI infrastructure, partner onboarding | Large enterprise with EDI in place |

**This feature implements Strategy A (CSV export) as the baseline.** Strategy B is listed as a future extension.

---

## 3. Data Model

### 3.1 New Table: `dim_supplier`

```sql
CREATE TABLE IF NOT EXISTS dim_supplier (
    supplier_id          VARCHAR(50)     PRIMARY KEY,
    supplier_name        VARCHAR(200)    NOT NULL,
    country_code         CHAR(2),                    -- ISO 3166-1 alpha-2
    address_line1        VARCHAR(200),
    city                 VARCHAR(100),
    state_province       VARCHAR(100),
    postal_code          VARCHAR(20),
    payment_terms        VARCHAR(30),                -- 'NET30', 'NET60', etc.
    default_lead_time_days  INTEGER,                 -- historical average lead time
    reliability_score    NUMERIC(4, 3),              -- 0.000 to 1.000 (OTIF rate)
    on_time_pct          NUMERIC(5, 2),              -- % of POs delivered on-time
    is_active            BOOLEAN         NOT NULL DEFAULT TRUE,
    load_ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_supplier_name ON dim_supplier (supplier_name);
```

---

### 3.2 New Table: `dim_item_supplier`

Maps items to their approved suppliers at specific locations, with sourcing lead times.

```sql
CREATE TABLE IF NOT EXISTS dim_item_supplier (
    id                   BIGSERIAL PRIMARY KEY,
    item_no              VARCHAR(50)     NOT NULL,
    loc                  VARCHAR(50)     NOT NULL,
    supplier_id          VARCHAR(50)     NOT NULL REFERENCES dim_supplier(supplier_id),
    is_preferred         BOOLEAN         NOT NULL DEFAULT FALSE,
    lead_time_days       INTEGER,                    -- supplier-specific LT for this item-loc
    moq                  NUMERIC(12, 2),             -- minimum order quantity
    price_per_unit       NUMERIC(12, 4),
    currency             CHAR(3),
    effective_from       DATE,
    effective_to         DATE,
    load_ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_item_supplier
    ON dim_item_supplier (item_no, loc, supplier_id);

CREATE INDEX idx_item_supplier_item_loc
    ON dim_item_supplier (item_no, loc);
```

---

### 3.3 New Table: `fact_open_purchase_orders`

```sql
CREATE TABLE IF NOT EXISTS fact_open_purchase_orders (
    id                       BIGSERIAL PRIMARY KEY,
    po_number                VARCHAR(50)     NOT NULL,
    po_line_number           INTEGER         NOT NULL DEFAULT 1,
    item_no                  VARCHAR(50)     NOT NULL,
    loc                      VARCHAR(50)     NOT NULL,
    supplier_id              VARCHAR(50)     REFERENCES dim_supplier(supplier_id),
    po_date                  DATE            NOT NULL,
    ordered_qty              NUMERIC(12, 2)  NOT NULL,
    confirmed_qty            NUMERIC(12, 2),
    received_qty             NUMERIC(12, 2)  NOT NULL DEFAULT 0.0,
    open_qty                 NUMERIC(12, 2)  GENERATED ALWAYS AS
                                 (COALESCE(confirmed_qty, ordered_qty) - received_qty) STORED,
    unit_cost                NUMERIC(12, 4),
    currency                 CHAR(3)         NOT NULL DEFAULT 'USD',
    line_value               NUMERIC(14, 2)  GENERATED ALWAYS AS
                                 (COALESCE(confirmed_qty, ordered_qty) * COALESCE(unit_cost, 0)) STORED,
    promised_delivery_date   DATE,
    confirmed_delivery_date  DATE,
    revised_delivery_date    DATE,
    effective_delivery_date  DATE GENERATED ALWAYS AS
                                 (COALESCE(revised_delivery_date,
                                           confirmed_delivery_date,
                                           promised_delivery_date)) STORED,
    po_status                VARCHAR(30)     NOT NULL DEFAULT 'open',
                             -- open / partially_received / closed / cancelled
    line_status              VARCHAR(30)     NOT NULL DEFAULT 'open',
    days_past_due            INTEGER GENERATED ALWAYS AS
                                 (CASE WHEN COALESCE(revised_delivery_date,
                                                     confirmed_delivery_date,
                                                     promised_delivery_date) < CURRENT_DATE
                                       THEN (CURRENT_DATE - COALESCE(revised_delivery_date,
                                                                      confirmed_delivery_date,
                                                                      promised_delivery_date))
                                       ELSE 0 END) STORED,
    source_file              VARCHAR(200),           -- CSV filename that loaded this row
    load_ts                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_open_pos_number_line
    ON fact_open_purchase_orders (po_number, po_line_number);

CREATE INDEX idx_open_pos_item_loc
    ON fact_open_purchase_orders (item_no, loc);

CREATE INDEX idx_open_pos_delivery_date
    ON fact_open_purchase_orders (effective_delivery_date)
    WHERE line_status NOT IN ('closed', 'cancelled');

CREATE INDEX idx_open_pos_past_due
    ON fact_open_purchase_orders (item_no, loc, effective_delivery_date)
    WHERE days_past_due > 0 AND line_status = 'open';
```

**Grain:** one row per `(po_number, po_line_number)`

---

### 3.4 New Table: `fact_po_receipts`

```sql
CREATE TABLE IF NOT EXISTS fact_po_receipts (
    id                  BIGSERIAL PRIMARY KEY,
    receipt_number      VARCHAR(50)     NOT NULL,
    po_number           VARCHAR(50)     NOT NULL,
    po_line_number      INTEGER         NOT NULL DEFAULT 1,
    item_no             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    supplier_id         VARCHAR(50),
    received_qty        NUMERIC(12, 2)  NOT NULL,
    unit_cost           NUMERIC(12, 4),
    actual_receipt_date DATE            NOT NULL,
    receipt_status      VARCHAR(20)     NOT NULL DEFAULT 'posted',
                        -- 'posted' / 'reversed'
    source_file         VARCHAR(200),
    load_ts             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_po_receipts_number
    ON fact_po_receipts (receipt_number, po_number, po_line_number);

CREATE INDEX idx_po_receipts_item_loc_date
    ON fact_po_receipts (item_no, loc, actual_receipt_date);
```

---

### 3.5 New Config: `config/po_integration_config.yaml`

```yaml
ingest:
  strategy: csv_export              # 'csv_export' | 'erp_api' (future)
  csv:
    po_file_pattern: "datafiles/open_pos_*.csv"    # glob for inbound PO exports
    receipt_file_pattern: "datafiles/po_receipts_*.csv"
    supplier_file_pattern: "datafiles/suppliers_*.csv"
    archive_after_load: true
    archive_dir: "datafiles/archive/"

validation:
  reject_pos_past_due_days: 180     # ignore POs > 180 days past due (likely data error)
  require_confirmed_delivery_date: false   # if false, promised_delivery_date used as fallback
  max_open_qty_ratio: 10.0          # flag if open_qty > 10x item's avg monthly demand

data_quality:
  past_due_threshold_days: 7        # POs this many days past due are flagged as at-risk
  delivery_date_tolerance_days: 2   # receipt within ±2 days of confirmed date = "on time"

scheduler:
  job_type: load_open_pos
  cron: "0 */4 * * *"              # reload every 4 hours (CSV export frequency)
```

---

## 4. Python Scripts / Pipeline

### 4.1 `scripts/load_open_pos.py`

```python
"""
load_open_pos.py

Loads open purchase order data from CSV exports into fact_open_purchase_orders.
Reconciles received_qty from fact_po_receipts after each load.

Usage:
    uv run python scripts/load_open_pos.py [--file PATH] [--dry-run]
    uv run python scripts/load_open_pos.py --receipts --file PATH  # load receipts only

Key functions:
    main()
    load_suppliers(filepath: str, conn, dry_run: bool) -> int
    load_pos(filepath: str, conn, dry_run: bool) -> int
    load_receipts(filepath: str, conn, dry_run: bool) -> int
    reconcile_received_qty(conn) -> int
    validate_po_row(row: dict, config: dict) -> tuple[bool, str]
    upsert_po_row(row: dict, conn) -> str    # 'inserted' | 'updated' | 'skipped'
    flag_past_due_pos(conn) -> int           # returns count of past-due POs flagged
    archive_source_file(filepath: str, config: dict) -> None
"""

import argparse, yaml, glob, os
import pandas as pd
from datetime import date, timedelta
from common.db import get_db_params
import psycopg

CONFIG_PATH = "config/po_integration_config.yaml"

# Expected CSV columns for open_pos file
PO_CSV_COLUMNS = {
    "po_number": str,
    "po_line_number": int,
    "item_no": str,
    "loc": str,
    "supplier_id": str,
    "po_date": "date",
    "ordered_qty": float,
    "confirmed_qty": float,
    "received_qty": float,
    "unit_cost": float,
    "currency": str,
    "promised_delivery_date": "date",
    "confirmed_delivery_date": "date",
    "revised_delivery_date": "date",
    "po_status": str,
    "line_status": str,
}

# Expected CSV columns for receipts file
RECEIPT_CSV_COLUMNS = {
    "receipt_number": str,
    "po_number": str,
    "po_line_number": int,
    "item_no": str,
    "loc": str,
    "received_qty": float,
    "unit_cost": float,
    "actual_receipt_date": "date",
    "receipt_status": str,
}


def validate_po_row(row: dict, config: dict) -> tuple:
    """
    Returns (is_valid: bool, reason: str).
    Rejects rows that fail business rules.
    """
    # Must have at least a promised delivery date
    if not any([row.get("promised_delivery_date"),
                row.get("confirmed_delivery_date"),
                row.get("revised_delivery_date")]):
        return False, "no_delivery_date"

    # Ignore closed/cancelled lines
    if row.get("line_status") in ("closed", "cancelled"):
        return False, "line_closed_or_cancelled"

    # Reject POs too far past due (likely data errors)
    max_past_due = config["validation"]["reject_pos_past_due_days"]
    eff_date = (row.get("revised_delivery_date")
                or row.get("confirmed_delivery_date")
                or row.get("promised_delivery_date"))
    if eff_date and (date.today() - eff_date).days > max_past_due:
        return False, f"past_due_exceeds_{max_past_due}_days"

    # Open qty must be positive
    confirmed_or_ordered = row.get("confirmed_qty") or row.get("ordered_qty", 0)
    received = row.get("received_qty", 0)
    open_qty = confirmed_or_ordered - received
    if open_qty <= 0:
        return False, "open_qty_zero_or_negative"

    return True, "ok"


def reconcile_received_qty(conn) -> int:
    """
    After loading receipts, update fact_open_purchase_orders.received_qty
    to match the sum of posted receipts. Returns number of rows updated.
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
```

### 4.2 Makefile Targets

```makefile
## Open PO Integration
po-schema:
	uv run python -c "import psycopg; conn=psycopg.connect(**__import__('common.db',fromlist=['get_db_params']).get_db_params()); conn.autocommit=True; [conn.execute(open(f).read()) for f in ['sql/040_create_open_pos.sql', 'sql/041_create_po_receipts.sql', 'sql/042_create_supplier_master.sql']]; print('done')"

po-load:
	uv run python scripts/load_open_pos.py

po-load-file:
	uv run python scripts/load_open_pos.py --file $(FILE)

po-load-dry:
	uv run python scripts/load_open_pos.py --dry-run

po-receipts-load:
	uv run python scripts/load_open_pos.py --receipts --file $(FILE)

po-all: po-schema po-load
```

---

## 5. API Endpoints

### 5.1 `GET /supply/open-pos`

Returns open purchase order lines for an item-location.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `item_no` | string | optional | Filter by item |
| `loc` | string | optional | Filter by location |
| `supplier_id` | string | optional | Filter by supplier |
| `status` | string | `open,partially_received` | Comma-sep status filter |
| `past_due_only` | bool | false | Only return past-due POs |
| `page` | int | 1 | Pagination |
| `page_size` | int | 50 | Rows per page |

**Response:**
```json
{
  "total": 2,
  "open_po_data_available": true,
  "last_loaded_at": "2026-03-06T08:00:01Z",
  "items": [
    {
      "po_number": "PO-4521",
      "po_line_number": 1,
      "item_no": "100320",
      "loc": "1401-BULK",
      "supplier_id": "VENDOR-0042",
      "supplier_name": "Acme Supply Co.",
      "po_date": "2026-02-15",
      "ordered_qty": 150.0,
      "confirmed_qty": 150.0,
      "received_qty": 0.0,
      "open_qty": 150.0,
      "unit_cost": 12.50,
      "line_value": 1875.00,
      "promised_delivery_date": "2026-03-14",
      "confirmed_delivery_date": "2026-03-14",
      "revised_delivery_date": null,
      "effective_delivery_date": "2026-03-14",
      "days_past_due": 0,
      "line_status": "open"
    },
    {
      "po_number": "PO-4522",
      "po_line_number": 1,
      "item_no": "100320",
      "loc": "1401-BULK",
      "supplier_id": "VENDOR-0042",
      "supplier_name": "Acme Supply Co.",
      "po_date": "2026-02-20",
      "ordered_qty": 50.0,
      "confirmed_qty": 50.0,
      "received_qty": 0.0,
      "open_qty": 50.0,
      "unit_cost": 12.50,
      "line_value": 625.00,
      "promised_delivery_date": "2026-03-21",
      "confirmed_delivery_date": "2026-03-21",
      "revised_delivery_date": null,
      "effective_delivery_date": "2026-03-21",
      "days_past_due": 0,
      "line_status": "open"
    }
  ]
}
```

### 5.2 `GET /supply/open-pos/summary`

Portfolio-level summary of open PO exposure.

**Response:**
```json
{
  "total_open_lines": 4821,
  "total_open_value_usd": 2847320.50,
  "total_open_qty_by_status": {
    "open": 312000,
    "partially_received": 48500
  },
  "past_due_lines": 203,
  "past_due_value_usd": 184200.00,
  "avg_days_past_due": 12.4,
  "suppliers_with_open_pos": 87,
  "last_loaded_at": "2026-03-06T08:00:01Z"
}
```

### 5.3 `POST /supply/open-pos/upload`

Accepts a CSV file upload for on-demand ingest (requires API key).

**Request:** `multipart/form-data` with `file` field (CSV)

**Response:**
```json
{
  "status": "ok",
  "rows_loaded": 4821,
  "rows_rejected": 14,
  "rejection_reasons": {
    "past_due_exceeds_180_days": 8,
    "open_qty_zero_or_negative": 4,
    "no_delivery_date": 2
  }
}
```

### 5.4 `GET /supply/past-due-pos`

Returns PO lines past their confirmed delivery date.

**Query params:** `min_days_past_due` (default 7), `supplier_id`, `page`, `page_size`

**Response:**
```json
{
  "total": 203,
  "items": [
    {
      "po_number": "PO-3901",
      "item_no": "200147",
      "loc": "1401-BULK",
      "supplier_name": "Beta Components Inc.",
      "open_qty": 300.0,
      "confirmed_delivery_date": "2026-02-20",
      "days_past_due": 14,
      "line_value": 3750.00,
      "severity": "high"
    }
  ]
}
```

---

## 6. Frontend Components

### 6.1 Open Orders Column in Inventory Position Table

In the existing Inventory tab's position table, add an "Open Orders" column:

```
Item No  │ Loc       │ On Hand │ On Order │ Open Orders           │ DOS
100320   │ 1401-BULK │ 120     │ 200      │ 200u arriving Mar 14  │ 7.4d
         │           │         │          │ (PO-4521 + PO-4522)   │
```

Clicking "200u arriving Mar 14" opens a mini-popover:
```
┌──────────────────────────────────────────┐
│ Open POs for 100320 @ 1401-BULK          │
├─────────┬──────┬─────────────┬───────────┤
│ PO #    │ Qty  │ Delivery    │ Status    │
├─────────┼──────┼─────────────┼───────────┤
│ PO-4521 │ 150  │ Mar 14 2026 │ Open      │
│ PO-4522 │  50  │ Mar 21 2026 │ Open      │
└─────────┴──────┴─────────────┴───────────┘
```

### 6.2 New Panel: `OpenPOPanel` in Inv. Planning Tab

**File:** `frontend/src/tabs/inv-planning/OpenPOPanel.tsx`

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  OPEN PURCHASE ORDERS        Last loaded: Mar 6 08:00   [Upload CSV] [Refresh]│
├───────────────────┬─────────────────────┬────────────────────────────────────┤
│  $2.85M           │  4,821 lines        │  203 Past-Due Lines                 │
│  Total Open Value │  (312K units)       │  ($184K at risk)                    │
├───────────────────┴─────────────────────┴────────────────────────────────────┤
│  Filters: Item [──────────────] Loc [──────────────] Supplier [──── ▼]       │
│           Status: [All ▼]   [✓ Past-Due Only]                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  PO #    │ Item   │ Loc        │ Supplier        │ Qty   │ Delivery  │ Status │
│  PO-4521 │ 100320 │ 1401-BULK  │ Acme Supply Co. │ 150   │ Mar 14 ✓  │ Open   │
│  PO-4522 │ 100320 │ 1401-BULK  │ Acme Supply Co. │  50   │ Mar 21 ✓  │ Open   │
│  PO-3901 │ 200147 │ 1401-BULK  │ Beta Components │ 300   │ Feb 20 ⚠  │ Open   │
│  ⚠ 14 days past due                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Color coding:**
- Green check: delivery date in future, not past due
- Yellow warning: 1–7 days past due
- Red warning: 7+ days past due

---

## 7. Worked Example: CSV Load → Projection Impact

### Step 1 — ERP CSV Export File

`datafiles/open_pos_20260306.csv`:

```csv
po_number,po_line_number,item_no,loc,supplier_id,po_date,ordered_qty,confirmed_qty,received_qty,unit_cost,currency,promised_delivery_date,confirmed_delivery_date,revised_delivery_date,po_status,line_status
PO-4521,1,100320,1401-BULK,VENDOR-0042,2026-02-15,150.0,150.0,0.0,12.50,USD,2026-03-14,2026-03-14,,open,open
PO-4522,1,100320,1401-BULK,VENDOR-0042,2026-02-20,50.0,50.0,0.0,12.50,USD,2026-03-21,2026-03-21,,open,open
PO-3901,1,200147,1401-BULK,VENDOR-0099,2026-01-10,300.0,300.0,0.0,12.50,USD,2026-02-20,2026-02-20,,open,open
PO-3901,2,100330,1401-BULK,VENDOR-0099,2026-01-10,200.0,200.0,200.0,8.75,USD,2026-02-20,2026-02-20,,open,closed
PO-4600,1,100500,2200-RACK,VENDOR-0055,2026-03-01,100.0,100.0,0.0,25.00,USD,2026-03-30,2026-03-30,,open,open
```

### Step 2 — Validation

| Row | PO-Line | Outcome | Reason |
|---|---|---|---|
| PO-4521-1 | 100320 @ 1401 | LOADED | Valid, future delivery |
| PO-4522-1 | 100320 @ 1401 | LOADED | Valid, future delivery |
| PO-3901-1 | 200147 @ 1401 | LOADED with WARNING | 14 days past due (flagged, not rejected) |
| PO-3901-2 | 100330 @ 1401 | SKIPPED | line_status = 'closed' |
| PO-4600-1 | 100500 @ 2200 | LOADED | Valid |

Result: 4 rows loaded, 1 skipped, 1 flagged past-due

### Step 3 — Projection Impact for Item 100320

**Before loading POs** (F1.2 degraded mode):
```
With-PO scenario = No-Order scenario (no delivery data available)
Stockout date (no_order): Mar 18 — 12 days away
Stockout date (with_po): Mar 18 — 12 days away  ← same (wrong)
```

**After loading POs** (F1.2 correct mode):
```
fact_open_purchase_orders for item_no='100320', loc='1401-BULK':
  Mar 14: +150 units (PO-4521)
  Mar 21: +50 units (PO-4522)

Projection simulation:
  Mar 06: qty=120
  Mar 14: qty = (120 - 8×16.3) + 150 = 120 - 130.4 + 150 = 139.6
  Mar 21: qty = (139.6 - 7×16.3) + 50  = 139.6 - 114.1 + 50 = 75.5
  Apr 04: qty = 75.5 - 14×16.3 = 75.5 - 228.2 = 0 (clamped) → PO needed

Stockout date (no_order): Mar 18 — 12 days away  ← still critical
Stockout date (with_po):  Apr 4 — 29 days away   ← more time to react
```

The planner now has 29 days instead of 12 to arrange the next replenishment. The emergency order is avoided; a standard planned order for April delivery is sufficient.

---

## 8. Data Quality Rules

### 8.1 Past-Due POs (delivery date < today)
- **Action:** Load the row but set a `data_quality_flag = 'past_due'` (not implemented in MVP — logged to stderr)
- **Why not reject:** The PO may still be in transit with a revised delivery date pending supplier confirmation
- **Threshold:** POs > 180 days past due are rejected (config: `reject_pos_past_due_days`)

### 8.2 Over-Receipts (received_qty > ordered_qty)
- **Action:** Cap `open_qty` at 0 (not negative), set `line_status = 'closed'`
- **Example:** Ordered 100, received 105 — accepted in many ERP systems as tolerance receipt

### 8.3 Split Deliveries (partial receipt + remaining open)
- **Handled by:** `reconcile_received_qty()` after each receipt load
- `received_qty` is updated from `fact_po_receipts` aggregate
- `line_status` auto-transitions: open → partially_received → closed

### 8.4 Duplicate POs (same po_number+line_number in two files)
- **Handled by:** UPSERT on `(po_number, po_line_number)` unique index
- Latest file's values win for mutable fields (qty, status, revised_delivery_date)
- `load_ts` and `modified_ts` updated on each upsert

### 8.5 PO Without a Delivery Date
- **Action:** Reject the line, log as `rejection_reason = 'no_delivery_date'`
- **Why:** A PO with no delivery date provides zero value to the projection engine

---

## 9. Dependencies

| Dependency | Status | Notes |
|---|---|---|
| ERP/WMS CSV export | External | Must be arranged with IT / ERP team; not automated by Demand Studio |
| F1.2 Inventory Projection | Downstream consumer | Reads `fact_open_purchase_orders.effective_delivery_date + open_qty` |
| F2.1 Order Recommendation | Downstream consumer | Reads confirmed inbound to net off requirements |
| `fact_inventory_snapshot` | Exists | `qty_on_order` field validates against total open_qty from this table |

---

## 10. Out of Scope

- ERP API / OData integration (Strategy B) — future feature
- EDI 850/855 message parsing — future feature
- Three-way PO matching (PO vs. GR vs. invoice) — accounts payable concern
- Vendor portal for supplier delivery date updates
- Multi-currency conversion (all values stored in document currency; FX conversion future)
- Blanket POs / scheduling agreements (treated as separate line items if exported as such)

---

## 11. Test Requirements

### Backend Unit Tests (`tests/unit/test_po_integration.py`)
- `test_validate_po_row_valid` — valid row returns (True, 'ok')
- `test_validate_po_row_no_delivery_date` — returns (False, 'no_delivery_date')
- `test_validate_po_row_past_due_exceeds_threshold` — returns (False, 'past_due_exceeds_180_days')
- `test_validate_po_row_closed_line` — returns (False, 'line_closed_or_cancelled')
- `test_validate_po_row_zero_open_qty` — returns (False, 'open_qty_zero_or_negative')
- `test_reconcile_received_qty_updates_status_to_partially_received`
- `test_reconcile_received_qty_updates_status_to_closed_when_fully_received`
- `test_effective_delivery_date_priority` — revised > confirmed > promised

### Backend API Tests (`tests/api/test_open_pos.py`)
- `test_get_open_pos_success` — 200, returns list of open PO lines
- `test_get_open_pos_past_due_only_filter` — filters correctly
- `test_get_open_pos_summary_success` — 200, returns portfolio KPIs
- `test_get_past_due_pos_success` — 200, returns past-due list
- `test_post_upload_csv_success` — 200, rows_loaded > 0
- `test_post_upload_csv_requires_auth` — 401 when API_KEY set

### Frontend Tests (`src/tabs/__tests__/InvPlanningTab.test.tsx`)
- `test_open_po_panel_renders`
- `test_open_po_panel_shows_portfolio_kpis`
- `test_open_po_panel_past_due_warning_shown`
