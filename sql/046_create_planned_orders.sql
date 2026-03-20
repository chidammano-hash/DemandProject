-- F2.1: Order Recommendation Engine
-- fact_planned_orders: one row per recommended order event (one per reorder cycle per DFU per run)

CREATE TABLE IF NOT EXISTS fact_planned_orders (
    id                      BIGSERIAL PRIMARY KEY,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    supplier_id             VARCHAR(50)     REFERENCES dim_supplier(supplier_id),
    policy_id               INTEGER,

    -- Quantities
    net_requirement_qty     NUMERIC(12, 2)  NOT NULL,
    recommended_qty         NUMERIC(12, 2)  NOT NULL,
    moq                     NUMERIC(12, 2)  NOT NULL DEFAULT 1,
    unit_cost               NUMERIC(12, 4),
    order_value             NUMERIC(14, 2)  GENERATED ALWAYS AS
                                (recommended_qty * COALESCE(unit_cost, 0)) STORED,
    currency                CHAR(3)         NOT NULL DEFAULT 'USD',

    -- Timing
    trigger_date            DATE            NOT NULL,
    trigger_reason          VARCHAR(50)     NOT NULL,
    order_by_date           DATE            NOT NULL,
    expected_receipt_date   DATE            NOT NULL,
    lead_time_days          INTEGER         NOT NULL,
    review_cycle_days       INTEGER,
    is_past_due             BOOLEAN GENERATED ALWAYS AS (order_by_date < CURRENT_DATE) STORED,

    -- Demand inputs used in calculation
    current_qty_on_hand     NUMERIC(12, 2)  NOT NULL,
    safety_stock            NUMERIC(12, 2)  NOT NULL,
    reorder_point           NUMERIC(12, 2)  NOT NULL,
    confirmed_inbound_qty   NUMERIC(12, 2)  NOT NULL DEFAULT 0,
    lt_forecast_demand      NUMERIC(12, 2)  NOT NULL,
    plan_version            VARCHAR(30),

    -- Confidence
    confidence_score        NUMERIC(4, 3),
    confidence_reason       TEXT,

    -- Workflow
    status                  VARCHAR(20)     NOT NULL DEFAULT 'proposed',
    -- proposed / approved / released / rejected / cancelled / closed
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_by             VARCHAR(100),
    approved_at             TIMESTAMPTZ,
    released_at             TIMESTAMPTZ,
    cancelled_at            TIMESTAMPTZ,
    rejection_reason        TEXT,
    run_id                  UUID            NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_planned_orders_item_loc
    ON fact_planned_orders (item_no, loc, status);

CREATE INDEX IF NOT EXISTS idx_planned_orders_order_by_date
    ON fact_planned_orders (order_by_date, status)
    WHERE status IN ('proposed', 'approved');

CREATE INDEX IF NOT EXISTS idx_planned_orders_past_due
    ON fact_planned_orders (item_no, loc, order_by_date)
    WHERE is_past_due AND status IN ('proposed', 'approved');

CREATE INDEX IF NOT EXISTS idx_planned_orders_status_created
    ON fact_planned_orders (status, created_at DESC);
