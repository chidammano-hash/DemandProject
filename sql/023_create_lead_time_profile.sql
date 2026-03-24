-- IPfeature2: Lead Time Variability Profiling
-- Creates dim_item_lead_time_profile table and indexes.

CREATE TABLE IF NOT EXISTS dim_item_lead_time_profile (
    id                      SERIAL          PRIMARY KEY,
    item_id                 TEXT            NOT NULL,
    loc                     TEXT            NOT NULL,
    lt_mean_days            NUMERIC(10, 4),
    lt_std_days             NUMERIC(10, 4),
    lt_cv                   NUMERIC(10, 6),
    lt_min_days             NUMERIC(10, 4),
    lt_max_days             NUMERIC(10, 4),
    lt_p25_days             NUMERIC(10, 4),
    lt_p50_days             NUMERIC(10, 4),
    lt_p75_days             NUMERIC(10, 4),
    lt_p95_days             NUMERIC(10, 4),
    observation_count       INTEGER,        -- number of LT change-points detected
    observation_months      INTEGER,        -- calendar months of history analysed
    lt_variability_class    TEXT,           -- stable | moderate | volatile
    computed_at             TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (item_id, loc)
);

CREATE INDEX IF NOT EXISTS idx_dim_item_lt_profile_class
    ON dim_item_lead_time_profile (lt_variability_class);

CREATE INDEX IF NOT EXISTS idx_dim_item_lt_profile_item_id
    ON dim_item_lead_time_profile (item_id);
