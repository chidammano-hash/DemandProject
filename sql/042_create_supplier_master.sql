-- F1.3: Supplier Master + Item-Supplier mapping
-- dim_supplier: one row per supplier
-- dim_item_supplier: approved suppliers per item-location

-- ---------------------------------------------------------------------------
-- dim_supplier
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_supplier (
    supplier_id             VARCHAR(50)     PRIMARY KEY,
    supplier_name           VARCHAR(200)    NOT NULL,
    country_code            CHAR(2),
    address_line1           VARCHAR(200),
    city                    VARCHAR(100),
    state_province          VARCHAR(100),
    postal_code             VARCHAR(20),
    payment_terms           VARCHAR(30),
    default_lead_time_days  INTEGER,
    reliability_score       NUMERIC(4, 3),
    on_time_pct             NUMERIC(5, 2),
    is_active               BOOLEAN         NOT NULL DEFAULT TRUE,
    load_ts                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_supplier_name
    ON dim_supplier (supplier_name);

-- ---------------------------------------------------------------------------
-- dim_item_supplier
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_item_supplier (
    id              BIGSERIAL PRIMARY KEY,
    item_id         VARCHAR(50)     NOT NULL,
    loc             VARCHAR(50)     NOT NULL,
    supplier_id     VARCHAR(50)     NOT NULL REFERENCES dim_supplier(supplier_id),
    is_preferred    BOOLEAN         NOT NULL DEFAULT FALSE,
    lead_time_days  INTEGER,
    moq             NUMERIC(12, 2),
    price_per_unit  NUMERIC(12, 4),
    currency        CHAR(3)         NOT NULL DEFAULT 'USD',
    effective_from  DATE,
    effective_to    DATE,
    load_ts         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_item_supplier
    ON dim_item_supplier (item_id, loc, supplier_id);

CREATE INDEX IF NOT EXISTS idx_item_supplier_item_loc
    ON dim_item_supplier (item_id, loc);

CREATE INDEX IF NOT EXISTS idx_item_supplier_preferred
    ON dim_item_supplier (item_id, loc, is_preferred)
    WHERE is_preferred = TRUE;
