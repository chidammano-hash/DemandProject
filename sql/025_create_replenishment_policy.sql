-- IPfeature5: Replenishment Policy Management
-- Table 1: dim_replenishment_policy — policy definitions
-- Table 2: fact_dfu_policy_assignment — DFU-to-policy mapping

CREATE TABLE IF NOT EXISTS dim_replenishment_policy (
    policy_sk          BIGSERIAL PRIMARY KEY,
    policy_id          TEXT UNIQUE NOT NULL,
    policy_name        TEXT NOT NULL,
    policy_type        TEXT NOT NULL CHECK (policy_type IN
                       ('continuous_rop','periodic_review','min_max','manual')),
    segment            TEXT,             -- descriptive (e.g. 'A', 'CZ', 'lumpy')
    review_cycle_days  INTEGER,          -- for periodic_review: days between checks
    service_level      NUMERIC(6,4),
    use_eoq            BOOLEAN DEFAULT TRUE,
    use_safety_stock   BOOLEAN DEFAULT TRUE,
    active             BOOLEAN DEFAULT TRUE,
    notes              TEXT,
    created_ts         TIMESTAMPTZ DEFAULT NOW(),
    modified_ts        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_dfu_policy_assignment (
    assignment_sk      BIGSERIAL PRIMARY KEY,
    item_no            TEXT NOT NULL,
    loc                TEXT NOT NULL,
    policy_id          TEXT NOT NULL REFERENCES dim_replenishment_policy(policy_id),
    override_reason    TEXT,
    assigned_by        TEXT DEFAULT 'system',   -- 'system' | 'manual'
    effective_date     DATE NOT NULL,
    created_ts         TIMESTAMPTZ DEFAULT NOW(),
    modified_ts        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (item_no, loc)
);

CREATE INDEX IF NOT EXISTS idx_dfu_policy_policy_id ON fact_dfu_policy_assignment (policy_id);
CREATE INDEX IF NOT EXISTS idx_dfu_policy_item_loc  ON fact_dfu_policy_assignment (item_no, loc);
