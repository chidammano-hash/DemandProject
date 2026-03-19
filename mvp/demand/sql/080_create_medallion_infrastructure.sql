-- Medallion Pipeline Infrastructure: Load Batch Registry
-- Tracks every load run through bronze → silver → gold layers.

CREATE TABLE IF NOT EXISTS audit_load_batch (
    batch_id            BIGSERIAL PRIMARY KEY,
    domain              TEXT NOT NULL,
    layer               TEXT NOT NULL DEFAULT 'bronze',
    source_file         TEXT,
    source_hash         TEXT,
    row_count_in        BIGINT,
    row_count_out       BIGINT,
    row_count_quarantined BIGINT DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'running',
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,
    metadata            JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_load_batch_domain
    ON audit_load_batch (domain, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_load_batch_status
    ON audit_load_batch (status);
