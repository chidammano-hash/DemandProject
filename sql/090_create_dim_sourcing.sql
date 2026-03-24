-- Sourcing: item-location to supply source mapping
-- One row per item × location × source (supplier-plant)

CREATE TABLE IF NOT EXISTS dim_sourcing (
    sourcing_ck    TEXT            PRIMARY KEY,
    site_id        VARCHAR(50)     NOT NULL,
    item_id        VARCHAR(50)     NOT NULL,
    loc            VARCHAR(50)     NOT NULL,
    source_cd      VARCHAR(50)     NOT NULL,
    transit_mode   VARCHAR(50),
    supplier_id    VARCHAR(50),
    plant_id       VARCHAR(50),
    load_ts        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sourcing_bk
    ON dim_sourcing (item_id, loc, source_cd);

CREATE INDEX IF NOT EXISTS idx_sourcing_supplier
    ON dim_sourcing (supplier_id);

CREATE INDEX IF NOT EXISTS idx_sourcing_item_loc
    ON dim_sourcing (item_id, loc);

CREATE INDEX IF NOT EXISTS idx_sourcing_plant
    ON dim_sourcing (plant_id);
