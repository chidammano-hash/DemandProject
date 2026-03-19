-- DQ Corrections Audit: tracks every auto-fix with before/after values.

CREATE TABLE IF NOT EXISTS audit_dq_corrections (
    correction_id       BIGSERIAL PRIMARY KEY,
    domain              TEXT NOT NULL,
    table_name          TEXT NOT NULL,
    row_key             TEXT NOT NULL,
    column_name         TEXT NOT NULL,
    old_value           TEXT,
    new_value           TEXT,
    fix_type            TEXT NOT NULL,
    fix_strategy        TEXT NOT NULL,
    applied_by          TEXT NOT NULL DEFAULT 'system',
    applied_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    load_batch_id       BIGINT REFERENCES audit_load_batch(batch_id),
    revert_batch_id     BIGINT,
    metadata            JSONB
);

CREATE INDEX IF NOT EXISTS idx_corrections_domain
    ON audit_dq_corrections (domain, applied_at DESC);
CREATE INDEX IF NOT EXISTS idx_corrections_batch
    ON audit_dq_corrections (load_batch_id);
CREATE INDEX IF NOT EXISTS idx_corrections_row_key
    ON audit_dq_corrections (domain, row_key);
