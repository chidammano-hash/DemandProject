-- Gen-4 Roadmap SC-3: S&OP decision audit log.
--
-- Logs every S&OP scenario decision (promote, approve, publish, override) with
-- actor, timestamp, and JSONB rationale so gap-closure analysis and audits can
-- trace WHO decided WHAT and WHY.

CREATE TABLE IF NOT EXISTS fact_sop_decisions (
    id                BIGSERIAL     PRIMARY KEY,
    cycle_id          INTEGER,                 -- FK to fact_sop_cycles.cycle_id (nullable: some decisions pre-cycle)
    scenario_id       TEXT,                    -- optional scenario identifier (e.g. 'sc_20260420_093021_ab12')
    decision_type     TEXT          NOT NULL,  -- promote | approve | publish | override | demote | reject | rollback
    decided_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    decided_by        TEXT          NOT NULL,  -- user id / email / service account
    rationale         JSONB,                   -- structured: {notes, gap_summary, override_reason, ...}
    prior_state       JSONB,                   -- snapshot of what changed (before)
    new_state         JSONB,                   -- snapshot of what changed (after)
    load_ts           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    CONSTRAINT fact_sop_decisions_type_chk CHECK (
        decision_type IN (
            'promote', 'approve', 'publish', 'override',
            'demote', 'reject', 'rollback', 'advance'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_sop_decisions_cycle     ON fact_sop_decisions (cycle_id);
CREATE INDEX IF NOT EXISTS idx_sop_decisions_scenario  ON fact_sop_decisions (scenario_id);
CREATE INDEX IF NOT EXISTS idx_sop_decisions_type_at   ON fact_sop_decisions (decision_type, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_sop_decisions_actor     ON fact_sop_decisions (decided_by, decided_at DESC);

COMMENT ON TABLE fact_sop_decisions IS
    'Gen-4 SC-3: immutable audit log of S&OP decisions (promote/approve/publish/override) with JSONB rationale.';
