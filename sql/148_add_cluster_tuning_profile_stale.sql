-- Gen-4 Roadmap SC-9: Per-cluster tuning profile auto-invalidation.
--
-- cluster_tuning_profiles currently lives as a YAML config (config/cluster_tuning_profiles.yaml),
-- NOT a DB table. This migration creates a companion DB table that tracks which
-- profiles have gone stale (e.g. after cluster scenario promotion invalidated
-- the training-time cluster partitions). promote_scenario() writes stale rows
-- here; the tuning pipeline honors `stale = TRUE` and re-runs those clusters.
--
-- The table is additive: existing YAML-based resolution keeps working if this
-- table is empty. When rows are present, the tuning pipeline checks them first.

CREATE TABLE IF NOT EXISTS cluster_tuning_profile_state (
    id                SERIAL          PRIMARY KEY,
    cluster_name      TEXT            NOT NULL UNIQUE,  -- matches cluster_tuning_profiles.yaml keys
    stale             BOOLEAN         NOT NULL DEFAULT FALSE,
    stale_reason      TEXT,                             -- e.g. 'cluster_promotion:sc_20260420_...'
    stale_since       TIMESTAMPTZ,
    cleared_at        TIMESTAMPTZ,                      -- when the profile was re-tuned
    load_ts           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    modified_ts       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cluster_tuning_state_stale
    ON cluster_tuning_profile_state (stale)
    WHERE stale = TRUE;

COMMENT ON TABLE cluster_tuning_profile_state IS
    'Gen-4 SC-9: tracks staleness of per-cluster tuning profiles after scenario promotion.';
