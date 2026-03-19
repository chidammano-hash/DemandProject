-- Row Lineage: traces each row from bronze -> silver -> gold.
-- DDL2 NOTE: uses `created_at` (vs `quarantined_at` in silver_quarantine).
-- Naming kept as-is for backward compatibility.

CREATE TABLE IF NOT EXISTS audit_row_lineage (
    lineage_id          BIGSERIAL PRIMARY KEY,
    domain              TEXT NOT NULL,
    load_batch_id       BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    bronze_id           BIGINT,
    silver_id           BIGINT,
    gold_id             BIGINT,
    business_key        TEXT NOT NULL,
    layer_reached       TEXT NOT NULL DEFAULT 'bronze',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lineage_domain_bk
    ON audit_row_lineage (domain, business_key);
CREATE INDEX IF NOT EXISTS idx_lineage_batch
    ON audit_row_lineage (load_batch_id);
