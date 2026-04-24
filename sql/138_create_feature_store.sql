-- Gen-4 Roadmap Cross-cutting #5: Feature Store scaffold (local, Postgres-backed).
--
-- Purpose: metadata tables that describe feature views over existing fact
-- tables. Point-in-time correctness is provided at query time by
-- `common.feature_store.get_point_in_time_features`, which joins against a
-- `*_history` table matching the view's backing table.
--
-- TODO(gen-4): Swap this scaffold for Feast (or equivalent) once event-
-- sourced features are standardized. Keep the column names stable so the
-- migration is a lift-and-shift.

CREATE TABLE IF NOT EXISTS feature_store_entity (
    id              BIGSERIAL       PRIMARY KEY,
    name            VARCHAR(128)    NOT NULL UNIQUE,           -- entity name (e.g. 'dfu', 'sku')
    entity_keys     TEXT[]          NOT NULL,                  -- column names that form the entity key
    description     TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feature_store_feature_view (
    id              BIGSERIAL       PRIMARY KEY,
    name            VARCHAR(128)    NOT NULL UNIQUE,           -- feature view name
    entity_id       BIGINT          NOT NULL REFERENCES feature_store_entity(id),
    features        TEXT[]          NOT NULL,                  -- list of feature (column) names
    source_table    VARCHAR(128)    NOT NULL,                  -- backing table name
    event_ts_col    VARCHAR(64)     NOT NULL DEFAULT 'event_ts', -- column holding row timestamp for PIT joins
    history_table   VARCHAR(128),                              -- optional *_history table (defaults to source_table || '_history')
    owner           VARCHAR(64),
    description     TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_store_fv_entity ON feature_store_feature_view (entity_id);
