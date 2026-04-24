-- 157_create_fact_lineage_event.sql
-- Gen-4 Stream G — Governance / AI-10: Minimal OpenLineage-compatible events.
--
-- Append-only log of data/ML lineage events. Intentionally loose schema —
-- we capture OpenLineage-like JSON bodies so a future full exporter can
-- forward these rows to a lineage backend (Marquez, DataHub) without a
-- schema migration.

CREATE TABLE IF NOT EXISTS fact_lineage_event (
    id              BIGSERIAL PRIMARY KEY,

    -- Event kind: 'START', 'COMPLETE', 'FAIL', 'ABORT' (OpenLineage vocab)
    kind            VARCHAR(20)     NOT NULL,

    -- Logical job identifier ('backtest_lgbm', 'promote_model', etc.)
    job_id          VARCHAR(120)    NOT NULL,

    -- Run identifier — groups START/COMPLETE for the same execution
    run_id          UUID,

    -- Input dataset refs as JSON array of {namespace, name, facets}
    inputs          JSONB,

    -- Output dataset refs as JSON array of {namespace, name, facets}
    outputs         JSONB,

    -- Free-form facets (owner, parent run, nominal time, etc.)
    facets          JSONB,

    -- When the event was emitted
    ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Primary query: latest lineage events for a given job
CREATE INDEX IF NOT EXISTS idx_lineage_event_job_ts
    ON fact_lineage_event (job_id, ts DESC);

-- Lookup by run_id for a full trace
CREATE INDEX IF NOT EXISTS idx_lineage_event_run
    ON fact_lineage_event (run_id);
