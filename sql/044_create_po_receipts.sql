-- F1.3: PO Receipts (goods received postings)
-- fact_po_receipts: one row per goods receipt posting

CREATE TABLE IF NOT EXISTS fact_po_receipts (
    id                  BIGSERIAL PRIMARY KEY,
    receipt_number      VARCHAR(50)     NOT NULL,
    po_number           VARCHAR(50)     NOT NULL,
    po_line_number      INTEGER         NOT NULL DEFAULT 1,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    supplier_id         VARCHAR(50),
    received_qty        NUMERIC(12, 2)  NOT NULL,
    unit_cost           NUMERIC(12, 4),
    actual_receipt_date DATE            NOT NULL,
    receipt_status      VARCHAR(20)     NOT NULL DEFAULT 'posted',
    source_file         VARCHAR(200),
    load_ts             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_po_receipt_number
    ON fact_po_receipts (receipt_number, po_number, po_line_number);

CREATE INDEX IF NOT EXISTS idx_po_receipts_item_loc_date
    ON fact_po_receipts (item_id, loc, actual_receipt_date);

CREATE INDEX IF NOT EXISTS idx_po_receipts_po_number
    ON fact_po_receipts (po_number, po_line_number);
