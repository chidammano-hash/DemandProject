-- F2.4: Procurement Workflow & Order Release
-- fact_released_purchase_orders: application-generated POs, distinct from
-- imported purchase-order history in fact_purchase_orders (sql/091)
-- fact_po_approval_log: immutable audit trail for PO state transitions
-- dim_erp_integration: ERP connection configuration

-- ---------------------------------------------------------------------------
-- fact_released_purchase_orders
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_released_purchase_orders (
    po_line_id              BIGSERIAL       PRIMARY KEY,
    po_number               VARCHAR(50)     NOT NULL,
    line_number             INTEGER         NOT NULL DEFAULT 1,
    item_id                 VARCHAR(50)     NOT NULL,
    item_description        VARCHAR(255),
    loc                     VARCHAR(50)     NOT NULL,
    supplier_id             VARCHAR(50)     REFERENCES dim_supplier(supplier_id),
    ordered_qty             NUMERIC(12,2)   NOT NULL,
    unit_of_measure         VARCHAR(10)     NOT NULL DEFAULT 'EA',
    unit_cost               NUMERIC(12,4),
    total_value             NUMERIC(14,2)   GENERATED ALWAYS AS
                                (COALESCE(ordered_qty * unit_cost, NULL)) STORED,
    currency                VARCHAR(3)      NOT NULL DEFAULT 'USD',
    po_date                 DATE            NOT NULL DEFAULT CURRENT_DATE,
    requested_delivery_date DATE            NOT NULL,
    confirmed_delivery_date DATE,
    received_qty            NUMERIC(12,2)   NOT NULL DEFAULT 0,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'proposed',
    source_exception_id     BIGINT,
    source_planned_order_id BIGINT,
    created_by              VARCHAR(100)    NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    planner_approved_by     VARCHAR(100),
    planner_approved_at     TIMESTAMPTZ,
    buyer_released_by       VARCHAR(100),
    buyer_released_at       TIMESTAMPTZ,
    erp_po_number           VARCHAR(100),
    erp_sent_at             TIMESTAMPTZ,
    erp_response_code       VARCHAR(20),
    erp_response_payload    JSONB,
    erp_integration_type    VARCHAR(20),
    buyer_code              VARCHAR(50),
    company_code            VARCHAR(20),
    plant_code              VARCHAR(20),
    notes                   TEXT,
    CONSTRAINT uq_po_line UNIQUE (po_number, line_number)
);

CREATE INDEX IF NOT EXISTS idx_fpo_status
    ON fact_released_purchase_orders (status, po_date DESC);

CREATE INDEX IF NOT EXISTS idx_fpo_supplier
    ON fact_released_purchase_orders (supplier_id, status);

CREATE INDEX IF NOT EXISTS idx_fpo_item_loc
    ON fact_released_purchase_orders (item_id, loc, status);

CREATE INDEX IF NOT EXISTS idx_fpo_po_number
    ON fact_released_purchase_orders (po_number);

CREATE INDEX IF NOT EXISTS idx_fpo_exception_source
    ON fact_released_purchase_orders (source_exception_id)
    WHERE source_exception_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fpo_erp_number
    ON fact_released_purchase_orders (erp_po_number)
    WHERE erp_po_number IS NOT NULL;


-- ---------------------------------------------------------------------------
-- fact_po_approval_log — immutable audit trail
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_po_approval_log (
    log_id          BIGSERIAL       PRIMARY KEY,
    po_line_id      BIGINT          REFERENCES fact_released_purchase_orders(po_line_id) ON DELETE CASCADE,
    po_number       VARCHAR(50)     NOT NULL,
    action          VARCHAR(30)     NOT NULL,
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

-- Older databases may already have the audit table from the conflicting
-- purchase-order migration, but without a foreign key. Preserve historical
-- rows while enforcing the correct released-order lineage for new writes.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'fact_po_approval_log'::regclass
          AND conname = 'fact_po_approval_log_po_line_id_fkey'
    ) THEN
        ALTER TABLE fact_po_approval_log
            ADD CONSTRAINT fact_po_approval_log_po_line_id_fkey
            FOREIGN KEY (po_line_id)
            REFERENCES fact_released_purchase_orders(po_line_id)
            ON DELETE CASCADE
            NOT VALID;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_po_log_po_line
    ON fact_po_approval_log (po_line_id, performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_po_log_po_number
    ON fact_po_approval_log (po_number, performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_po_log_action
    ON fact_po_approval_log (action, performed_at DESC);


-- ---------------------------------------------------------------------------
-- dim_erp_integration — ERP connection configuration
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_erp_integration (
    integration_id      SERIAL          PRIMARY KEY,
    erp_type            VARCHAR(50)     NOT NULL,
    integration_name    VARCHAR(100)    NOT NULL,
    endpoint_url        TEXT,
    auth_method         VARCHAR(30),
    auth_credential_ref VARCHAR(100),
    field_mapping       JSONB,
    default_company_code VARCHAR(20),
    default_plant_code  VARCHAR(20),
    active              BOOLEAN         NOT NULL DEFAULT TRUE,
    last_sync_at        TIMESTAMPTZ,
    last_sync_status    VARCHAR(20),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    notes               TEXT
);

-- Seed CSV-export integration (always available, no endpoint needed)
INSERT INTO dim_erp_integration
    (erp_type, integration_name, auth_method, active, notes)
VALUES
    ('csv_export', 'CSV Export (ERP Import)', 'none', TRUE,
     'Tier A: generate PO CSV for manual import into any ERP')
ON CONFLICT DO NOTHING;
