-- F1.3: Open Purchase Orders
-- fact_open_purchase_orders: one row per PO line
-- mv_open_po_summary: portfolio-level aggregate view

CREATE TABLE IF NOT EXISTS fact_open_purchase_orders (
    id                       BIGSERIAL PRIMARY KEY,
    po_number                VARCHAR(50)     NOT NULL,
    po_line_number           INTEGER         NOT NULL DEFAULT 1,
    item_id                  VARCHAR(50)     NOT NULL,
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
    line_status              VARCHAR(30)     NOT NULL DEFAULT 'open',
    days_past_due            INTEGER DEFAULT 0,
    source_file              VARCHAR(200),
    load_ts                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_open_po_number_line
    ON fact_open_purchase_orders (po_number, po_line_number);

CREATE INDEX IF NOT EXISTS idx_open_po_item_loc
    ON fact_open_purchase_orders (item_id, loc);

CREATE INDEX IF NOT EXISTS idx_open_po_delivery_date
    ON fact_open_purchase_orders (effective_delivery_date)
    WHERE line_status NOT IN ('closed', 'cancelled');

CREATE INDEX IF NOT EXISTS idx_open_po_past_due
    ON fact_open_purchase_orders (item_id, loc, effective_delivery_date)
    WHERE days_past_due > 0 AND line_status = 'open';

CREATE INDEX IF NOT EXISTS idx_open_po_supplier
    ON fact_open_purchase_orders (supplier_id, line_status);
