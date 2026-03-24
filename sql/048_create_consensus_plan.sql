-- F2.3 — Consensus Forecasting & Planner Overrides
-- Tables: fact_forecast_overrides, fact_consensus_plan

CREATE TABLE IF NOT EXISTS fact_forecast_overrides (
    override_id                 BIGSERIAL       PRIMARY KEY,
    item_id                     VARCHAR(50)     NOT NULL,
    loc                         VARCHAR(50)     NOT NULL,
    override_month              DATE            NOT NULL,
    override_type               VARCHAR(20)     NOT NULL,
    override_qty                NUMERIC(12,2),
    override_multiplier         NUMERIC(6,4),
    override_additive_qty       NUMERIC(12,2)   DEFAULT 0,
    is_hard_override            BOOLEAN         NOT NULL DEFAULT FALSE,
    override_reason             TEXT            NOT NULL,
    override_note               TEXT,
    created_by                  VARCHAR(100)    NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    valid_from                  DATE            NOT NULL,
    valid_to                    DATE            NOT NULL,
    approved_by                 VARCHAR(100),
    approved_at                 TIMESTAMPTZ,
    rejected_by                 VARCHAR(100),
    rejected_at                 TIMESTAMPTZ,
    rejection_reason            TEXT,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'draft',
    requires_approval           BOOLEAN         NOT NULL DEFAULT TRUE,
    priority_rank               INTEGER         NOT NULL DEFAULT 5,
    statistical_qty_at_creation NUMERIC(12,2),
    estimated_impact_units      NUMERIC(12,2),
    estimated_impact_value      NUMERIC(14,2),
    currency                    VARCHAR(3)      DEFAULT 'USD',
    expires_auto                BOOLEAN         NOT NULL DEFAULT TRUE,
    plan_version_applied        VARCHAR(50),
    parent_override_id          BIGINT          REFERENCES fact_forecast_overrides(override_id)
);

CREATE INDEX IF NOT EXISTS idx_override_item_loc_month
    ON fact_forecast_overrides (item_id, loc, override_month);

CREATE INDEX IF NOT EXISTS idx_override_status
    ON fact_forecast_overrides (status, override_month);

CREATE INDEX IF NOT EXISTS idx_override_created_by
    ON fact_forecast_overrides (created_by, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_override_pending_approval
    ON fact_forecast_overrides (status, requires_approval)
    WHERE status = 'pending_approval';

CREATE INDEX IF NOT EXISTS idx_override_valid_dates
    ON fact_forecast_overrides (valid_from, valid_to);


CREATE TABLE IF NOT EXISTS fact_consensus_plan (
    id                  BIGSERIAL       PRIMARY KEY,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_month          DATE            NOT NULL,
    plan_version        VARCHAR(50)     NOT NULL,
    statistical_qty     NUMERIC(12,2)   NOT NULL,
    statistical_p10     NUMERIC(12,2),
    statistical_p90     NUMERIC(12,2),
    override_qty        NUMERIC(12,2)   DEFAULT 0,
    consensus_qty       NUMERIC(12,2)   NOT NULL,
    consensus_p10       NUMERIC(12,2),
    consensus_p90       NUMERIC(12,2),
    override_applied    BOOLEAN         NOT NULL DEFAULT FALSE,
    override_id         BIGINT          REFERENCES fact_forecast_overrides(override_id),
    override_type       VARCHAR(20),
    override_multiplier NUMERIC(6,4),
    is_hard_override    BOOLEAN         DEFAULT FALSE,
    overrider           VARCHAR(100),
    approver            VARCHAR(100),
    uplift_pct          NUMERIC(8,4),
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_consensus_plan UNIQUE (item_id, loc, plan_month, plan_version)
);

CREATE INDEX IF NOT EXISTS idx_consensus_plan_item_loc_month
    ON fact_consensus_plan (item_id, loc, plan_month);

CREATE INDEX IF NOT EXISTS idx_consensus_plan_version
    ON fact_consensus_plan (plan_version, plan_month);

CREATE INDEX IF NOT EXISTS idx_consensus_plan_overridden
    ON fact_consensus_plan (plan_version)
    WHERE override_applied = TRUE;
