# Feature F2.4 — Procurement Workflow & Order Release

**Phase:** Evolution to Operations — Phase 2 (Demand Planning)
**Feature Number:** F2.4
**Status:** Design — Not Implemented
**Depends On:** F2.1 (Planned Order Generation — future), F2.3 (Consensus Plan), IPfeature5 (Replenishment Policy), IPfeature7 (Exception Queue)

---

## 1. Problem Statement

### What Fails Today

The Demand Studio replenishment engine can detect exception conditions (stockout risk, excess inventory, reorder point breaches) and generate planned order recommendations in `fact_replenishment_exceptions`. However, these recommendations exist only as database rows. There is no mechanism to:

- Convert a planned order recommendation into an actual purchase order
- Route that order to a buyer for review and release
- Send the released order to an ERP system or directly to a supplier
- Track the order through the full lifecycle from proposal to goods receipt

**Concrete failure — what happens today:**

The exception engine flags Item 100320 for reorder: recommended_order_qty=316 units, recommended_reorder_date=2026-04-15. The exception appears in the Exception Queue UI. The supply chain planner reads the recommendation.

Then what? Today: the planner manually creates a PO in SAP. They copy the item number, location, quantity, and supplier from the screen. There is no confirmation the order was placed. There is no tracking. If the planner is out sick, the exception sits unacted-on for days. No audit trail connects the exception to the PO in SAP.

The system is advisory-only. Orders do not flow.

### The Execution Gap

```
TODAY:
  Exception Queue → Planner reads → Manual SAP entry → No connection back

WITH THIS FEATURE:
  Exception Queue → Approve in Demand Studio → Automated PO creation
                 → CSV/API to ERP → ERP PO number returned
                 → PO tracked in Demand Studio through delivery
```

---

## 2. Workflow States

Every order flows through a defined state machine. State transitions are audited.

```
PROPOSED
    │
    │ (planner reviews exception, decides to proceed)
    ▼
PLANNER_APPROVED
    │
    │ (buyer reviews quantity, supplier, pricing — may modify)
    ▼
BUYER_RELEASED
    │
    │ (system or buyer triggers ERP/supplier send)
    ▼
PO_SENT ─────────────────────────────► CANCELLED (at any stage before sent)
    │
    │ (supplier acknowledges — EDI 855 or manual)
    ▼
SUPPLIER_CONFIRMED
    │
    │ (first delivery arrives — ASN or receipt posted)
    ▼
PARTIALLY_RECEIVED
    │
    │ (final delivery closes the order)
    ▼
FULLY_RECEIVED
    │
    │ (invoice matched, 3-way match complete)
    ▼
CLOSED
```

### Role-Based Visibility

| Role | Can See | Can Approve | Can Release | Can Send to ERP |
|---|---|---|---|---|
| Demand Planner | All DFUs in assigned categories | Yes (PROPOSED → PLANNER_APPROVED) | No | No |
| Buyer | Items in their supplier portfolio | No | Yes (PLANNER_APPROVED → BUYER_RELEASED) | Yes |
| Supply Chain Manager | All items | Yes (all stages) | Yes | Yes |
| Read-Only Analyst | All items | No | No | No |

Role assignment is managed via `dim_user_roles` (defined in this feature, future work to connect to enterprise identity provider).

---

## 3. ERP Integration Options

Three integration tiers are designed in priority order:

### Tier A: CSV/Excel Export (Implemented First)

The simplest integration path. Demand Studio generates a standardized PO CSV file that a buyer can import into any ERP system. No API connectivity required.

**CSV format** (standard ERP import layout):

```csv
PO_NUMBER,LINE_NO,ITEM_NUMBER,ITEM_DESCRIPTION,LOCATION,SUPPLIER_ID,
SUPPLIER_NAME,ORDERED_QTY,UNIT_OF_MEASURE,UNIT_COST,TOTAL_VALUE,CURRENCY,
REQUESTED_DELIVERY_DATE,PO_DATE,BUYER_CODE,COMPANY_CODE,PLANT,
DEMAND_STUDIO_EXCEPTION_ID,NOTES
DS-2026-04-001,1,100320,Bulk Cleaning Solution 5L,1401-BULK,SUP-4821,
ABC Trading Co,316,EA,24.00,7584.00,USD,2026-04-28,2026-04-15,
JSMITH,COMP001,PLT01,EXC-7834,Auto-generated from stockout exception
DS-2026-04-001,2,204771,Industrial Degreaser 20L,2203-STD,SUP-4821,
ABC Trading Co,80,EA,67.50,5400.00,USD,2026-04-28,2026-04-15,
KLEE,COMP001,PLT01,EXC-7891,Auto-generated from reorder exception
```

### Tier B: REST API Webhook to ERP

Demand Studio posts a JSON payload to a configured ERP REST endpoint. The ERP returns a confirmation with its internal PO number. Used for SAP S/4HANA (OData), Oracle NetSuite (SuiteScript), or Microsoft Dynamics 365 (Power Automate / Dataverse API).

### Tier C: EDI 850 Purchase Order

Industry-standard EDI 850 transaction set for large suppliers and EDI-capable trading partners. Transmitted via AS2 or SFTP. Acknowledgement via EDI 997 (functional acknowledgement) and EDI 855 (PO acknowledgement). Full EDI implementation is out of scope for this feature but the data model is designed to support it.

### SAP BAPI Field Mapping (Tier B Reference)

For SAP integration via BAPI_PO_CREATE1, field mapping is:

| Demand Studio Field | SAP BAPI Field | SAP Table |
|---|---|---|
| `item_no` | `MATNR` | EKPO |
| `loc` (plant code) | `WERKS` | EKPO |
| `supplier_id` | `LIFNR` | EKKO |
| `ordered_qty` | `MENGE` | EKPO |
| `unit_of_measure` | `MEINS` | EKPO |
| `unit_cost` | `NETPR` | EKPO |
| `currency` | `WAERS` | EKKO |
| `requested_delivery_date` | `EINDT` | EKPO |
| `po_date` | `BEDAT` | EKKO |
| `buyer_code` | `EKGRP` | EKKO |
| `company_code` | `BUKRS` | EKKO |

---

## 4. Data Model

### 4.1 New Table: `fact_purchase_orders`

**Grain:** `po_number + line_number`

Each purchase order can have multiple line items (one per item/location).

```sql
CREATE TABLE fact_purchase_orders (
    po_line_id              BIGSERIAL       PRIMARY KEY,
    po_number               VARCHAR(50)     NOT NULL,   -- DS-generated: DS-{YYYY}-{MM}-{SEQ}
    line_number             INTEGER         NOT NULL,
    item_no                 VARCHAR(50)     NOT NULL,
    item_description        VARCHAR(255),
    loc                     VARCHAR(50)     NOT NULL,
    supplier_id             VARCHAR(50)     NOT NULL,
    supplier_name           VARCHAR(255),
    ordered_qty             NUMERIC(12,2)   NOT NULL,
    unit_of_measure         VARCHAR(10)     NOT NULL DEFAULT 'EA',
    unit_cost               NUMERIC(12,4),
    total_value             NUMERIC(14,2),
    currency                VARCHAR(3)      NOT NULL DEFAULT 'USD',
    po_date                 DATE            NOT NULL,
    requested_delivery_date DATE            NOT NULL,
    confirmed_delivery_date DATE,
    received_qty            NUMERIC(12,2)   DEFAULT 0,
    remaining_qty           NUMERIC(12,2),              -- Computed: ordered - received
    status                  VARCHAR(30)     NOT NULL DEFAULT 'proposed',
    source_exception_id     BIGINT,                     -- FK to fact_replenishment_exceptions
    source_planned_order_id BIGINT,                     -- FK to future fact_planned_orders table
    created_by              VARCHAR(100)    NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    planner_approved_by     VARCHAR(100),
    planner_approved_at     TIMESTAMPTZ,
    buyer_released_by       VARCHAR(100),
    buyer_released_at       TIMESTAMPTZ,
    erp_po_number           VARCHAR(100),               -- ERP's internal PO number (returned after send)
    erp_sent_at             TIMESTAMPTZ,
    erp_response_code       VARCHAR(20),                -- '200', 'SUCCESS', 'ERROR', etc.
    erp_response_payload    JSONB,                      -- Full ERP response for debugging
    erp_integration_type    VARCHAR(20),                -- 'csv', 'rest_api', 'edi_850'
    buyer_code              VARCHAR(50),
    company_code            VARCHAR(20),
    plant_code              VARCHAR(20),
    notes                   TEXT,
    CONSTRAINT uq_po_line UNIQUE (po_number, line_number)
);

CREATE INDEX idx_po_status
    ON fact_purchase_orders (status, po_date DESC);

CREATE INDEX idx_po_supplier
    ON fact_purchase_orders (supplier_id, status);

CREATE INDEX idx_po_item_loc
    ON fact_purchase_orders (item_no, loc, status);

CREATE INDEX idx_po_number
    ON fact_purchase_orders (po_number);

CREATE INDEX idx_po_exception_source
    ON fact_purchase_orders (source_exception_id)
    WHERE source_exception_id IS NOT NULL;

CREATE INDEX idx_po_erp_number
    ON fact_purchase_orders (erp_po_number)
    WHERE erp_po_number IS NOT NULL;
```

### 4.2 New Table: `fact_po_approval_log`

Immutable audit trail. Every state transition is recorded.

```sql
CREATE TABLE fact_po_approval_log (
    log_id          BIGSERIAL       PRIMARY KEY,
    po_line_id      BIGINT          NOT NULL REFERENCES fact_purchase_orders(po_line_id),
    po_number       VARCHAR(50)     NOT NULL,
    action          VARCHAR(30)     NOT NULL,   -- proposed, planner_approved, planner_rejected,
                                                -- buyer_modified, buyer_released, po_sent,
                                                -- supplier_confirmed, partially_received,
                                                -- fully_received, closed, cancelled
    performed_by    VARCHAR(100)    NOT NULL,
    performed_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    old_status      VARCHAR(30),
    new_status      VARCHAR(30),
    old_qty         NUMERIC(12,2),
    new_qty         NUMERIC(12,2),
    old_delivery_date DATE,
    new_delivery_date DATE,
    reason          TEXT,
    system_note     TEXT
);

CREATE INDEX idx_po_log_po_line
    ON fact_po_approval_log (po_line_id, performed_at DESC);

CREATE INDEX idx_po_log_action
    ON fact_po_approval_log (action, performed_at DESC);
```

### 4.3 New Table: `dim_erp_integration`

ERP connection configuration. Managed by system admin via API or config file.

```sql
CREATE TABLE dim_erp_integration (
    integration_id      SERIAL          PRIMARY KEY,
    erp_type            VARCHAR(50)     NOT NULL,   -- 'sap_s4hana', 'oracle_netsuite', 'dynamics365', 'csv_export', 'edi_850'
    integration_name    VARCHAR(100)    NOT NULL,
    endpoint_url        TEXT,
    auth_method         VARCHAR(30),                -- 'oauth2', 'basic', 'api_key', 'none'
    auth_credential_ref VARCHAR(100),               -- Reference to secrets manager key (not stored here)
    field_mapping       JSONB,                      -- Maps DS fields to ERP field names
    default_company_code VARCHAR(20),
    default_plant_code  VARCHAR(20),
    active              BOOLEAN         NOT NULL DEFAULT TRUE,
    last_sync_at        TIMESTAMPTZ,
    last_sync_status    VARCHAR(20),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    notes               TEXT
);
```

### 4.4 New Table: `dim_supplier`

Supplier master (currently missing from the system — `mv_supplier_performance` references supplier data from inventory but there is no supplier dim table).

```sql
CREATE TABLE dim_supplier (
    supplier_id         VARCHAR(50)     PRIMARY KEY,
    supplier_name       VARCHAR(255)    NOT NULL,
    erp_supplier_id     VARCHAR(100),
    contact_email       VARCHAR(255),
    contact_phone       VARCHAR(50),
    country             VARCHAR(100),
    currency            VARCHAR(3)      DEFAULT 'USD',
    lead_time_days      INTEGER,
    payment_terms       VARCHAR(50),    -- 'NET30', 'NET60', '2/10 NET30'
    min_order_value     NUMERIC(12,2),
    preferred_po_method VARCHAR(20),    -- 'email', 'edi', 'api', 'portal'
    active              BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_at         TIMESTAMPTZ
);
```

---

## 5. Python Scripts

### 5.1 `scripts/release_planned_orders.py`

```python
"""
Convert approved exception recommendations into purchase orders and
optionally send them to an ERP system.

Usage:
    # Release all planner-approved items for a supplier to buyer queue
    uv run scripts/release_planned_orders.py \
        --supplier-id SUP-4821 \
        --action approve_all_planner \
        --performed-by jane.smith@company.com

    # Generate CSV export for a set of released POs
    uv run scripts/release_planned_orders.py \
        --po-numbers DS-2026-04-001 DS-2026-04-002 \
        --action export_csv \
        --output-dir data/po_exports/

    # Send released POs to ERP via REST API
    uv run scripts/release_planned_orders.py \
        --po-numbers DS-2026-04-001 \
        --action send_erp \
        --integration-id 1

Config: config/procurement_config.yaml
"""

import yaml
import csv
import json
import httpx
import psycopg
from datetime import date, datetime
from pathlib import Path
from common.db import get_db_params


def generate_po_number(conn: psycopg.Connection) -> str:
    """
    Generate a sequential, collision-safe PO number.
    Format: DS-{YYYY}-{MM}-{ZERO_PADDED_SEQ}
    Example: DS-2026-04-001, DS-2026-04-002, DS-2026-04-003

    Uses a PostgreSQL sequence scoped to year+month.
    """
    now = datetime.utcnow()
    sequence_name = f"po_seq_{now.year}_{now.month:02d}"
    with conn.cursor() as cur:
        cur.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_sequences WHERE sequencename = %s
                ) THEN
                    EXECUTE 'CREATE SEQUENCE ' || quote_ident(%s) || ' START 1';
                END IF;
            END$$;
        """, (sequence_name, sequence_name))
        cur.execute(f"SELECT nextval(%s)", (sequence_name,))
        seq_val = cur.fetchone()[0]
    return f"DS-{now.year}-{now.month:02d}-{seq_val:03d}"


def create_po_from_exception(
    exception_id: int,
    performed_by: str,
    conn: psycopg.Connection,
) -> str:
    """
    Read a replenishment exception and create a proposed PO.

    Args:
        exception_id: Primary key of fact_replenishment_exceptions
        performed_by: User submitting the order
        conn: DB connection

    Returns:
        po_number of the created PO
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.item_no, e.loc, e.recommended_order_qty,
                   e.exception_type, e.recommended_reorder_date,
                   i.item_description, s.supplier_id, s.lead_time_days,
                   s.currency
            FROM fact_replenishment_exceptions e
            LEFT JOIN dim_item i ON i.item_no = e.item_no
            LEFT JOIN dim_supplier s ON s.supplier_id = (
                SELECT supplier_id FROM dim_supplier
                WHERE active = TRUE LIMIT 1   -- TODO: item-supplier mapping table
            )
            WHERE e.id = %s
        """, (exception_id,))
        row = cur.fetchone()

    if not row:
        raise ValueError(f"Exception {exception_id} not found")

    (item_no, loc, qty, exc_type, reorder_date,
     item_desc, supplier_id, lead_time, currency) = row

    po_number = generate_po_number(conn)
    delivery_date = date.today() + __import__("datetime").timedelta(days=lead_time or 14)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fact_purchase_orders (
                po_number, line_number, item_no, item_description, loc,
                supplier_id, ordered_qty, unit_cost, total_value, currency,
                po_date, requested_delivery_date, status,
                source_exception_id, created_by
            ) VALUES (
                %s, 1, %s, %s, %s,
                %s, %s, NULL, NULL, %s,
                CURRENT_DATE, %s, 'proposed',
                %s, %s
            )
        """, (
            po_number, item_no, item_desc, loc,
            supplier_id, qty, currency,
            delivery_date, exception_id, performed_by
        ))
    return po_number


def approve_po(
    po_number: str,
    approved_by: str,
    new_qty: float | None,
    conn: psycopg.Connection,
) -> None:
    """
    Planner approves a proposed PO (optionally adjusting quantity).
    Transitions status: proposed → planner_approved.

    Args:
        po_number: PO to approve
        approved_by: Planner's email
        new_qty: If provided, overrides the originally proposed qty
        conn: DB connection
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE fact_purchase_orders
            SET status = 'planner_approved',
                planner_approved_by = %s,
                planner_approved_at = NOW(),
                ordered_qty = COALESCE(%s, ordered_qty)
            WHERE po_number = %s AND status = 'proposed'
        """, (approved_by, new_qty, po_number))

        cur.execute("""
            INSERT INTO fact_po_approval_log
                (po_line_id, po_number, action, performed_by, old_status, new_status, new_qty)
            SELECT po_line_id, %s, 'planner_approved', %s, 'proposed', 'planner_approved', ordered_qty
            FROM fact_purchase_orders WHERE po_number = %s
        """, (po_number, approved_by, po_number))


def release_po(
    po_number: str,
    released_by: str,
    conn: psycopg.Connection,
) -> None:
    """
    Buyer releases an approved PO for sending to ERP.
    Transitions status: planner_approved → buyer_released.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE fact_purchase_orders
            SET status = 'buyer_released',
                buyer_released_by = %s,
                buyer_released_at = NOW()
            WHERE po_number = %s AND status = 'planner_approved'
        """, (released_by, po_number))


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
    sql = """
        SELECT
            po.po_number, po.line_number, po.item_no, po.item_description,
            po.loc, po.supplier_id, s.supplier_name, po.ordered_qty,
            po.unit_of_measure, po.unit_cost, po.total_value, po.currency,
            po.requested_delivery_date, po.po_date, po.buyer_code,
            po.company_code, po.plant_code, po.source_exception_id, po.notes
        FROM fact_purchase_orders po
        LEFT JOIN dim_supplier s ON s.supplier_id = po.supplier_id
        WHERE po.po_number = ANY(%s)
          AND po.status = 'buyer_released'
        ORDER BY po.po_number, po.line_number
    """
    with conn.cursor() as cur:
        cur.execute(sql, (po_numbers,))
        rows = cur.fetchall()

    fieldnames = [
        "PO_NUMBER", "LINE_NO", "ITEM_NUMBER", "ITEM_DESCRIPTION",
        "LOCATION", "SUPPLIER_ID", "SUPPLIER_NAME", "ORDERED_QTY",
        "UNIT_OF_MEASURE", "UNIT_COST", "TOTAL_VALUE", "CURRENCY",
        "REQUESTED_DELIVERY_DATE", "PO_DATE", "BUYER_CODE",
        "COMPANY_CODE", "PLANT", "DEMAND_STUDIO_EXCEPTION_ID", "NOTES"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(fieldnames, row)))

    return len(rows)


def send_po_to_erp(
    po_number: str,
    integration_id: int,
    conn: psycopg.Connection,
) -> dict:
    """
    Send a released PO to ERP via REST API integration.

    Loads integration config from dim_erp_integration, constructs the
    ERP-specific payload using field_mapping, POSTs to endpoint_url,
    stores response in erp_response_payload.

    Returns:
        dict with keys: success (bool), erp_po_number (str or None), error (str or None)
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT erp_type, endpoint_url, auth_method, field_mapping, "
            "       auth_credential_ref "
            "FROM dim_erp_integration WHERE integration_id = %s AND active = TRUE",
            (integration_id,)
        )
        integration = cur.fetchone()

    if not integration:
        return {"success": False, "erp_po_number": None, "error": "Integration not found"}

    erp_type, endpoint_url, auth_method, field_mapping, cred_ref = integration

    # Build ERP payload by mapping DS fields to ERP field names
    with conn.cursor() as cur:
        cur.execute(
            "SELECT po_number, item_no, loc, supplier_id, ordered_qty, "
            "       unit_of_measure, unit_cost, currency, "
            "       requested_delivery_date, po_date, buyer_code, "
            "       company_code, plant_code "
            "FROM fact_purchase_orders WHERE po_number = %s",
            (po_number,)
        )
        po_data = cur.fetchone()

    # Map fields using integration config
    payload = _map_fields(po_data, field_mapping)

    # POST to ERP endpoint
    try:
        response = httpx.post(
            endpoint_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        erp_po_number = response.json().get("po_number") or response.json().get("EBELN")
        success = response.status_code in (200, 201)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_purchase_orders
                SET status = 'po_sent',
                    erp_po_number = %s,
                    erp_sent_at = NOW(),
                    erp_response_code = %s,
                    erp_response_payload = %s
                WHERE po_number = %s
            """, (erp_po_number, str(response.status_code),
                  json.dumps(response.json()), po_number))

        return {"success": success, "erp_po_number": erp_po_number, "error": None}

    except Exception as e:
        return {"success": False, "erp_po_number": None, "error": str(e)}


def _map_fields(po_data: tuple, field_mapping: dict) -> dict:
    """Apply ERP field mapping from dim_erp_integration.field_mapping JSONB."""
    ds_fields = [
        "po_number", "item_no", "loc", "supplier_id", "ordered_qty",
        "unit_of_measure", "unit_cost", "currency",
        "requested_delivery_date", "po_date", "buyer_code",
        "company_code", "plant_code"
    ]
    ds_dict = dict(zip(ds_fields, po_data))
    return {
        field_mapping.get(k, k): v
        for k, v in ds_dict.items()
        if v is not None
    }
```

### 5.2 `config/procurement_config.yaml`

```yaml
procurement:
  po_number_format: "DS-{YEAR}-{MONTH:02d}-{SEQ:03d}"
  default_currency: USD
  default_unit_of_measure: EA

  approval_thresholds:
    requires_buyer_release_above_value: 1000    # POs > $1,000 require buyer release step
    auto_release_below_value: 0                 # Set > 0 to enable auto-release for small orders

  csv_export:
    output_dir: data/po_exports/
    filename_format: "PO_export_{TIMESTAMP}.csv"
    include_internal_ids: true

  erp_integration:
    default_integration_id: 1
    retry_on_failure: true
    max_retries: 3
    retry_delay_seconds: 30
    timeout_seconds: 30

  notifications:
    notify_buyer_on_planner_approval: true
    notify_planner_on_buyer_release: true
    notify_on_erp_failure: true
    notification_method: email   # 'email', 'slack', 'none'

  lead_time_fallback_days: 14    # Used when dim_supplier.lead_time_days is NULL
```

---

## 6. API Endpoints

### 6.1 `GET /supply/purchase-orders`

List purchase orders with filtering.

**Parameters:** `status`, `supplier_id`, `item_no`, `loc`, `po_date_from`, `po_date_to`, `page`, `page_size`

**Response:**

```json
{
  "total": 5,
  "total_value": 47500.00,
  "page": 1,
  "orders": [
    {
      "po_number": "DS-2026-04-001",
      "line_number": 1,
      "item_no": "100320",
      "item_description": "Bulk Cleaning Solution 5L",
      "loc": "1401-BULK",
      "supplier_id": "SUP-4821",
      "supplier_name": "ABC Trading Co",
      "ordered_qty": 316.0,
      "unit_cost": 24.00,
      "total_value": 7584.00,
      "currency": "USD",
      "po_date": "2026-04-15",
      "requested_delivery_date": "2026-04-28",
      "status": "planner_approved",
      "source_exception_id": 7834,
      "created_by": "jane.smith@company.com",
      "erp_po_number": null
    }
  ]
}
```

### 6.2 `POST /supply/planned-orders/{exception_id}/approve`

Convert an exception recommendation to a proposed PO, then immediately approve.

**Request body:**

```json
{
  "performed_by": "jane.smith@company.com",
  "ordered_qty": 316.0,
  "requested_delivery_date": "2026-04-28",
  "notes": "Expedited delivery — promotion starts May 15"
}
```

**Response (201 Created):**

```json
{
  "po_number": "DS-2026-04-001",
  "status": "planner_approved",
  "total_value": 7584.00,
  "requested_delivery_date": "2026-04-28"
}
```

### 6.3 `POST /supply/planned-orders/{exception_id}/reject`

Mark an exception as rejected (no PO will be created).

**Request body:** `{ "rejected_by": "...", "reason": "..." }`

### 6.4 `POST /supply/purchase-orders/{po_number}/release`

Buyer releases a planner-approved PO.

**Request body:**

```json
{
  "released_by": "bob.chen@company.com",
  "confirmed_delivery_date": "2026-04-28",
  "notes": "Confirmed lead time with supplier"
}
```

**Response:** `{ "po_number": "DS-2026-04-001", "status": "buyer_released" }`

### 6.5 `POST /supply/purchase-orders/export-csv`

Generate CSV export file for a set of POs.

**Request body:**

```json
{
  "po_numbers": ["DS-2026-04-001", "DS-2026-04-002"],
  "exported_by": "bob.chen@company.com"
}
```

**Response:**

```json
{
  "filename": "PO_export_2026-04-15T14-23-01.csv",
  "line_count": 5,
  "total_value": 47500.00,
  "download_url": "/supply/purchase-orders/exports/PO_export_2026-04-15T14-23-01.csv"
}
```

### 6.6 `POST /supply/erp/send-pos`

Send released POs to ERP via configured REST API integration.

**Request body:**

```json
{
  "po_numbers": ["DS-2026-04-001"],
  "integration_id": 1,
  "sent_by": "bob.chen@company.com"
}
```

**Response:**

```json
{
  "results": [
    {
      "po_number": "DS-2026-04-001",
      "success": true,
      "erp_po_number": "4500012847",
      "erp_response_code": "200",
      "new_status": "po_sent"
    }
  ],
  "total_sent": 1,
  "total_failed": 0
}
```

### 6.7 `GET /supply/purchase-orders/{po_number}/timeline`

Return the full audit log for a PO's lifecycle.

**Response:**

```json
{
  "po_number": "DS-2026-04-001",
  "current_status": "po_sent",
  "timeline": [
    { "action": "proposed",          "performed_by": "system",                    "performed_at": "2026-04-15T09:14:00Z", "note": "Auto-created from exception EXC-7834" },
    { "action": "planner_approved",  "performed_by": "jane.smith@company.com",    "performed_at": "2026-04-15T09:22:00Z", "note": null },
    { "action": "buyer_released",    "performed_by": "bob.chen@company.com",      "performed_at": "2026-04-15T11:05:00Z", "note": "Confirmed lead time with supplier" },
    { "action": "po_sent",           "performed_by": "system",                    "performed_at": "2026-04-15T11:06:03Z", "note": "ERP PO 4500012847 created" }
  ]
}
```

---

## 7. Frontend Components

### 7.1 Planned Orders Panel

Located in: `frontend/src/tabs/inv-planning/PlannedOrdersPanel.tsx`

Two-column layout:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PLANNED ORDERS                                    [Filter ▼] [Export]  │
├──────────────────────────────────┬──────────────────────────────────────┤
│  ORDER QUEUE                     │  PO DETAIL — DS-2026-04-001          │
│                                  │                                      │
│  ● DS-2026-04-001  PENDING       │  Item: 100320 – Bulk Cleaning Sol.   │
│    ABC Trading Co  $7,584        │  Loc: 1401-BULK                      │
│    5 items  [Approve] [Reject]   │  Supplier: ABC Trading Co (SUP-4821) │
│                                  │                                      │
│  ● DS-2026-04-002  APPROVED      │  Ordered Qty:    316 EA              │
│    XYZ Supply      $12,400       │  Unit Cost:      $24.00              │
│    3 items  [Release]            │  Total Value:    $7,584.00           │
│                                  │                                      │
│  ● DS-2026-04-003  RELEASED      │  PO Date:        Apr 15, 2026        │
│    ABC Trading Co  $28,000       │  Req. Delivery:  Apr 28, 2026        │
│    8 items  [Send to ERP]        │                                      │
│                                  │  TIMELINE                            │
│                                  │  ● Apr 15 09:14  Proposed (system)  │
│                                  │  ● Apr 15 09:22  Planner Approved   │
│                                  │    jane.smith@company.com            │
│                                  │  ● Apr 15 11:05  Buyer Released     │
│                                  │    bob.chen@company.com              │
│                                  │  ○ Awaiting ERP Send                 │
│                                  │                                      │
│                                  │  [Send to ERP] [Export CSV]         │
└──────────────────────────────────┴──────────────────────────────────────┘
```

### 7.2 Release Confirmation Modal

```
┌────────────────────────────────────────────────────────────┐
│  RELEASE PURCHASE ORDER DS-2026-04-001                     │
├────────────────────────────────────────────────────────────┤
│  You are about to release the following order for sending  │
│  to the ERP system:                                        │
│                                                            │
│  Supplier:    ABC Trading Co (SUP-4821)                    │
│  Total Lines: 5                                            │
│  Total Value: $47,500.00 USD                               │
│                                                            │
│  Items Summary:                                            │
│   • 100320  Bulk Cleaning Sol.   316 EA  $7,584            │
│   • 204771  Industrial Degreaser  80 EA  $5,400            │
│   • 301102  Floor Sealer          45 EA  $3,150            │
│   • 402251  Safety Gloves        200 EA  $2,400            │
│   • 501820  Dispensers            12 EA  $1,440            │
│                                                            │
│  Once released, the buyer (bob.chen@company.com) will be  │
│  notified to review and send to ERP.                       │
│                                                            │
│               [Cancel]     [Confirm Release]               │
└────────────────────────────────────────────────────────────┘
```

### 7.3 PO Status Badge Colors

| Status | Badge Color |
|---|---|
| proposed | Gray |
| planner_approved | Blue |
| buyer_released | Indigo |
| po_sent | Amber |
| supplier_confirmed | Green outline |
| partially_received | Orange |
| fully_received | Green solid |
| closed | Slate |
| cancelled | Red |

---

## 8. Worked Example: Supplier ABC Trading — 5 Orders

**Scenario:** April planning cycle. Exception engine has flagged 5 items sourced from ABC Trading (SUP-4821). Total exposure: $47,500.

**Step 1: Exceptions Generated**

```
fact_replenishment_exceptions:
  EXC-7834  100320  1401-BULK  REORDER_POINT  recommended_qty=316  priority=HIGH
  EXC-7835  204771  2203-STD   STOCKOUT_RISK  recommended_qty=80   priority=CRITICAL
  EXC-7836  301102  1401-BULK  REORDER_POINT  recommended_qty=45   priority=MEDIUM
  EXC-7837  402251  2203-STD   REORDER_POINT  recommended_qty=200  priority=LOW
  EXC-7838  501820  1401-BULK  EXCESS_INV     recommended_qty=0    priority=INFO
```

Note: EXC-7838 is excess inventory — no order. Jane rejects it. Only 4 proceed.

**Step 2: Jane Reviews and Approves (Planner)**

Jane opens Exception Queue → selects all 4 REORDER/STOCKOUT exceptions → clicks "Create Order Batch".

System calls `create_po_from_exception()` for each. A single PO number DS-2026-04-001 is created with 4 line items (lines 1–4). Jane adjusts line 2 quantity: EXC-7835 recommended 80, but Jane increases to 100 based on consensus plan showing promotion effect.

```
DS-2026-04-001, Line 1: 100320, qty=316, $7,584
DS-2026-04-001, Line 2: 204771, qty=100, $6,750  [Jane modified from 80 → 100]
DS-2026-04-001, Line 3: 301102, qty=45,  $3,150
DS-2026-04-001, Line 4: 402251, qty=200, $2,400
Total: $19,884
```

Jane clicks "Approve". Status → `planner_approved`. Bob (buyer) receives email notification.

**Step 3: Bob Reviews and Releases (Buyer)**

Bob opens Planned Orders. Sees DS-2026-04-001 status APPROVED. Reviews line items. Confirms delivery date Apr 28 with ABC Trading. Clicks "Release".

Status → `buyer_released`.

**Step 4: CSV Export Generated**

Bob clicks "Export CSV". System calls `export_pos_to_csv()`.

Sample output (`PO_export_2026-04-15T14-23-01.csv`):

```
PO_NUMBER,LINE_NO,ITEM_NUMBER,ITEM_DESCRIPTION,LOCATION,SUPPLIER_ID,...
DS-2026-04-001,1,100320,Bulk Cleaning Solution 5L,1401-BULK,SUP-4821,...
DS-2026-04-001,2,204771,Industrial Degreaser 20L,2203-STD,SUP-4821,...
DS-2026-04-001,3,301102,Floor Sealer,1401-BULK,SUP-4821,...
DS-2026-04-001,4,402251,Safety Gloves,2203-STD,SUP-4821,...
```

Bob imports this CSV into SAP. SAP creates PO 4500012847. Bob enters this ERP number back into Demand Studio via `PUT /supply/purchase-orders/DS-2026-04-001/erp-confirm` (future enhancement). Status → `po_sent`.

**Step 5: Audit Trail**

```
fact_po_approval_log for DS-2026-04-001:
  2026-04-15 09:14  proposed           system              Auto-created from exceptions
  2026-04-15 09:22  buyer_modified     jane.smith@co.com   Line 2 qty changed 80→100
  2026-04-15 09:22  planner_approved   jane.smith@co.com
  2026-04-15 11:05  buyer_released     bob.chen@co.com     Delivery confirmed Apr 28
  2026-04-15 11:06  po_sent            system              CSV exported; ERP PO pending
```

---

## 9. Dependencies

| Dependency | Type | Status |
|---|---|---|
| IPfeature7 — Exception Queue | Hard (source of planned order recommendations) | Implemented |
| F2.3 — Consensus Plan | Soft (improves order quantity accuracy) | Design |
| IPfeature5 — Replenishment Policy | Soft (policy determines reorder trigger) | Implemented |
| `dim_supplier` table | Hard (must be created in this feature) | Not implemented |
| User identity / roles | Soft (role-based access) | Not in system |
| ERP system connectivity (Tier B/C) | Soft (Tier A CSV is self-contained) | Not in system |

---

## 10. Out of Scope

- Three-way invoice matching (PO quantity vs receipt vs supplier invoice)
- Goods receipt posting (updating on-hand inventory from receipts)
- Supplier portal for suppliers to view and confirm their orders online
- EDI 850/855/856 full implementation
- Multi-currency PO management with exchange rate conversion
- Blanket POs and scheduling agreements
- Drop-ship orders (PO sent to supplier but delivered directly to customer)
- Return purchase orders (RPOs) for supplier defect returns

---

## 11. Makefile Targets

```makefile
procurement-schema:
    uv run python -c "import psycopg; from common.db import get_db_params; ..." \
        sql/041_create_purchase_orders.sql \
        sql/042_create_supplier.sql

procurement-export:
    uv run scripts/release_planned_orders.py \
        --po-numbers $(PO_NUMBERS) \
        --action export_csv

procurement-send-erp:
    uv run scripts/release_planned_orders.py \
        --po-numbers $(PO_NUMBERS) \
        --action send_erp \
        --integration-id $(INTEGRATION_ID)
```

---

## 12. Test Requirements

### Backend Unit Tests (`tests/unit/test_procurement.py`)

- `test_generate_po_number_format`: Assert format matches `DS-{YYYY}-{MM}-{NNN}`
- `test_generate_po_number_sequential`: Two calls in same month produce consecutive numbers
- `test_apply_override_multiplier`: `apply_override(450, 'PROMO', None, 1.40, 0, False)` → `(630.0, 180.0)`
- `test_create_po_from_exception_fields`: Correct item_no, qty, supplier_id mapped from exception
- `test_approve_po_state_transition`: proposed → planner_approved, sets approved_by and approved_at
- `test_release_po_state_transition`: planner_approved → buyer_released
- `test_release_po_wrong_state_raises`: Attempting release from 'proposed' raises ValueError
- `test_export_csv_column_order`: CSV headers match ERP import format exactly
- `test_export_csv_line_count`: 4 exceptions → 4 CSV lines
- `test_map_fields_sap_mapping`: DS fields mapped correctly using SAP field_mapping dict

### Backend API Tests (`tests/api/test_procurement.py`)

- `test_get_purchase_orders_list`: Returns list of POs with correct shape
- `test_approve_exception_creates_po_201`: POST approve returns 201 with po_number
- `test_approve_exception_wrong_status_404`: Non-existent exception returns 404
- `test_release_po_requires_planner_approval_first`: Releasing from 'proposed' returns 422
- `test_export_csv_returns_filename`: Response includes download_url
- `test_send_erp_success`: Mock httpx, assert status transitions to po_sent
- `test_send_erp_failure_logged`: ERP timeout → error stored in erp_response_payload
- `test_get_timeline_returns_all_events`: Timeline includes all audit log entries

### Frontend Tests (`frontend/src/tabs/__tests__/PlannedOrdersPanel.test.tsx`)

- Two-column layout renders (order queue + detail panel)
- Clicking order in queue populates detail panel
- Approve button calls approve API and refreshes list
- Release confirmation modal shows order summary
- "Export CSV" button calls export endpoint
- Status badge renders correct color class per status
- Empty state shown when no orders in queue
