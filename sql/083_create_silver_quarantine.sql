-- Silver Quarantine: rejected rows that failed DQ gate checks or type casting.
-- DDL2 NOTE: uses `quarantined_at` (vs `created_at` in audit_row_lineage).
-- Naming kept as-is for backward compatibility.

CREATE TABLE IF NOT EXISTS silver_quarantine (
    quarantine_id       BIGSERIAL PRIMARY KEY,
    domain              TEXT NOT NULL,
    _bronze_id          BIGINT,
    _load_batch_id      BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    rejection_reason    TEXT NOT NULL,
    rejection_details   JSONB,
    raw_row             JSONB NOT NULL,
    quarantined_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved            BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT
);

CREATE INDEX IF NOT EXISTS idx_quarantine_domain
    ON silver_quarantine (domain, quarantined_at DESC);
CREATE INDEX IF NOT EXISTS idx_quarantine_batch
    ON silver_quarantine (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_quarantine_unresolved
    ON silver_quarantine (domain) WHERE NOT resolved;
