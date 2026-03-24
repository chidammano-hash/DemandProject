-- F3.5 — Network / Multi-Echelon Planning

CREATE TABLE IF NOT EXISTS dim_echelon_network (
    id                          BIGSERIAL   PRIMARY KEY,
    parent_loc                  VARCHAR(50) NOT NULL,
    child_loc                   VARCHAR(50) NOT NULL,
    echelon_level               INTEGER     NOT NULL DEFAULT 1,  -- 1=DC, 2=regional, 3=store
    link_type                   VARCHAR(30) NOT NULL DEFAULT 'replenishment',
    replenishment_lead_time_days INTEGER,
    is_active                   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_echelon_link UNIQUE (parent_loc, child_loc)
);

CREATE INDEX IF NOT EXISTS idx_echelon_network_parent
    ON dim_echelon_network (parent_loc, echelon_level);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_echelon_ss_targets (
    id                      BIGSERIAL       PRIMARY KEY,
    item_id                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    echelon_level           INTEGER         NOT NULL,
    echelon_ss_qty          NUMERIC(12,2)   NOT NULL,
    standalone_ss_qty       NUMERIC(12,2),
    pooling_benefit_pct     NUMERIC(6,3),   -- (standalone-echelon)/standalone * 100
    service_level_target    NUMERIC(5,4),
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_echelon_ss UNIQUE (item_id, loc, echelon_level)
);

CREATE INDEX IF NOT EXISTS idx_echelon_ss_item_loc
    ON fact_echelon_ss_targets (item_id, loc);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_echelon_reorder_points (
    id                      BIGSERIAL       PRIMARY KEY,
    item_id                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    echelon_level           INTEGER         NOT NULL,
    reorder_point_qty       NUMERIC(12,2)   NOT NULL,
    echelon_ss_qty          NUMERIC(12,2),
    demand_during_lt_qty    NUMERIC(12,2),
    cascade_risk_flag       BOOLEAN         NOT NULL DEFAULT FALSE,
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_echelon_rop UNIQUE (item_id, loc, echelon_level)
);

CREATE INDEX IF NOT EXISTS idx_echelon_rop_cascade
    ON fact_echelon_reorder_points (cascade_risk_flag, echelon_level)
    WHERE cascade_risk_flag = TRUE;
