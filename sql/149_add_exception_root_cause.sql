-- Gen-4 Roadmap SC-8: Group exceptions by root-cause key + SLAs.
--
-- Adds three columns to exception_queue:
--   - root_cause_key: deterministic hash of primary detector factors (e.g. type + primary metric)
--     used to group related exceptions into themes (same root cause).
--   - severity_band: derived category from severity (critical|high|medium|low)
--   - sla_due_at: when the SLA for response expires, based on severity_band
--     (driven by config/exception_sla.yaml).

ALTER TABLE exception_queue
    ADD COLUMN IF NOT EXISTS root_cause_key TEXT,
    ADD COLUMN IF NOT EXISTS severity_band  TEXT,
    ADD COLUMN IF NOT EXISTS sla_due_at     TIMESTAMPTZ;

COMMENT ON COLUMN exception_queue.root_cause_key IS
    'Gen-4 SC-8: deterministic hash grouping exceptions with the same underlying cause.';
COMMENT ON COLUMN exception_queue.severity_band IS
    'Gen-4 SC-8: categorical band (critical|high|medium|low) derived from severity score.';
COMMENT ON COLUMN exception_queue.sla_due_at IS
    'Gen-4 SC-8: SLA response deadline; driven by config/exception_sla.yaml (severity -> hours).';

CREATE INDEX IF NOT EXISTS idx_eq_root_cause
    ON exception_queue (root_cause_key);
CREATE INDEX IF NOT EXISTS idx_eq_sla_due
    ON exception_queue (sla_due_at)
    WHERE status IN ('open', 'investigating');
CREATE INDEX IF NOT EXISTS idx_eq_severity_band
    ON exception_queue (severity_band, status);
