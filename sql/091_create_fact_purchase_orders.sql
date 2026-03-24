-- Purchase Orders: comprehensive PO history (open + closed)
-- Grain: po_number × item_id × loc (one row per PO line)

CREATE TABLE IF NOT EXISTS fact_purchase_orders (
    po_ck                   TEXT            PRIMARY KEY,
    po_number               VARCHAR(50)     NOT NULL,
    site_id                 VARCHAR(50),
    loc                     VARCHAR(50)     NOT NULL,
    source                  VARCHAR(50),
    item_id                 VARCHAR(50)     NOT NULL,
    ordered_qty             NUMERIC(12, 2),
    orig_po_qty             NUMERIC(12, 2),
    orig_po_uom             VARCHAR(20),
    one_time_buy_flg        CHAR(1),
    ordered_qty_uom         VARCHAR(20),
    net_price               NUMERIC(12, 4),
    gross_value             NUMERIC(14, 2),
    closure_code            VARCHAR(30),
    po_hdr_status           VARCHAR(30),
    po_line_status          VARCHAR(30),
    receipt_status          VARCHAR(30),
    po_hdr_receipt_status   VARCHAR(30),
    supplier_id             VARCHAR(50),
    supplier_name           VARCHAR(200),
    vendor_type             VARCHAR(50),
    carrier_no              VARCHAR(50),
    carrier_name            VARCHAR(200),
    goods_supplier_no       VARCHAR(50),
    goods_supplier_name     VARCHAR(200),
    delivery_date           DATE,
    original_delivery_date  DATE,
    current_ship_date       DATE,
    original_ship_date      DATE,
    po_type                 VARCHAR(10),
    po_inco_terms           VARCHAR(10),
    freight_inco_terms      VARCHAR(10),
    -- Computed columns
    lead_time_planned       INTEGER GENERATED ALWAYS AS
        (delivery_date - original_ship_date) STORED,
    lead_time_actual        INTEGER GENERATED ALWAYS AS
        (delivery_date - original_delivery_date) STORED,
    is_closed               BOOLEAN GENERATED ALWAYS AS
        (closure_code IS NOT NULL AND closure_code != '') STORED,
    load_ts                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_po_bk
    ON fact_purchase_orders (po_number, item_id, loc);

CREATE INDEX IF NOT EXISTS idx_po_item_loc
    ON fact_purchase_orders (item_id, loc);

CREATE INDEX IF NOT EXISTS idx_po_supplier
    ON fact_purchase_orders (supplier_id, delivery_date DESC);

CREATE INDEX IF NOT EXISTS idx_po_source
    ON fact_purchase_orders (source);

CREATE INDEX IF NOT EXISTS idx_po_closure
    ON fact_purchase_orders (closure_code)
    WHERE closure_code IS NULL OR closure_code = '';

CREATE INDEX IF NOT EXISTS idx_po_delivery
    ON fact_purchase_orders (delivery_date)
    WHERE delivery_date IS NOT NULL;
