-- Grounded Copilot persistence and release-linked inventory opportunities.

BEGIN;

CREATE TABLE IF NOT EXISTS ai_copilot_session (
    session_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    owner_role TEXT NOT NULL,
    client_request_id TEXT NOT NULL,
    page_context TEXT NOT NULL,
    context_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    UNIQUE (owner_user_id, client_request_id),
    CONSTRAINT ck_ai_copilot_session_context
        CHECK (jsonb_typeof(context_json) = 'object'),
    CONSTRAINT ck_ai_copilot_session_hash
        CHECK (char_length(context_hash) = 64),
    CONSTRAINT ck_ai_copilot_session_time
        CHECK (expires_at > created_at AND last_active_at >= created_at)
);

CREATE INDEX IF NOT EXISTS idx_ai_copilot_session_owner
    ON ai_copilot_session (owner_user_id, closed_at, last_active_at DESC);

CREATE TABLE IF NOT EXISTS ai_copilot_turn (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES ai_copilot_session (session_id),
    turn_number INTEGER NOT NULL,
    client_request_id TEXT NOT NULL,
    status TEXT NOT NULL,
    question_text TEXT,
    answer_text TEXT,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_hash TEXT,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    tool_call_count INTEGER,
    latency_ms BIGINT,
    safe_error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    content_expires_at TIMESTAMPTZ,
    content_redacted_at TIMESTAMPTZ,
    UNIQUE (session_id, turn_number),
    UNIQUE (session_id, client_request_id),
    CONSTRAINT ck_ai_copilot_turn_status
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT ck_ai_copilot_turn_hashes CHECK (
        char_length(request_hash) = 64
        AND (response_hash IS NULL OR char_length(response_hash) = 64)
    ),
    CONSTRAINT ck_ai_copilot_turn_usage CHECK (
        (prompt_tokens IS NULL OR prompt_tokens >= 0)
        AND (completion_tokens IS NULL OR completion_tokens >= 0)
        AND (tool_call_count IS NULL OR tool_call_count >= 0)
        AND (latency_ms IS NULL OR latency_ms >= 0)
    ),
    CONSTRAINT ck_ai_copilot_turn_redaction CHECK (
        content_redacted_at IS NULL
        OR (question_text IS NULL AND answer_text IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_ai_copilot_turn_session
    ON ai_copilot_turn (session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_ai_copilot_turn_retention
    ON ai_copilot_turn (content_expires_at)
    WHERE content_redacted_at IS NULL;

CREATE TABLE IF NOT EXISTS ai_copilot_evidence (
    evidence_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES ai_copilot_turn (turn_id) ON DELETE CASCADE,
    evidence_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    source_relation TEXT NOT NULL,
    source_business_key TEXT NOT NULL,
    scope_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    promotion_id INTEGER REFERENCES model_promotion_log (id) ON DELETE RESTRICT,
    production_run_id UUID,
    inventory_run_id UUID,
    data_as_of_at TIMESTAMPTZ NOT NULL,
    freshness_label TEXT NOT NULL,
    value_snapshot JSONB NOT NULL,
    citation_claim TEXT,
    content_hash TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (turn_id, evidence_number),
    UNIQUE (turn_id, evidence_id),
    CONSTRAINT ck_ai_copilot_evidence_json CHECK (
        jsonb_typeof(scope_json) = 'object'
        AND jsonb_typeof(value_snapshot) = 'object'
    ),
    CONSTRAINT ck_ai_copilot_evidence_hash
        CHECK (char_length(content_hash) = 64)
);

CREATE INDEX IF NOT EXISTS idx_ai_copilot_evidence_turn
    ON ai_copilot_evidence (turn_id, evidence_number);

CREATE TABLE IF NOT EXISTS inventory_planning_run (
    inventory_run_id UUID PRIMARY KEY,
    plan_version TEXT NOT NULL,
    source_promotion_id INTEGER NOT NULL,
    source_production_run_id UUID NOT NULL,
    inventory_snapshot_date DATE NOT NULL,
    planning_date DATE NOT NULL,
    status TEXT NOT NULL,
    target_method TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    rows_read BIGINT NOT NULL DEFAULT 0,
    rows_written BIGINT NOT NULL DEFAULT 0,
    exclusions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    safe_error_code TEXT,
    CONSTRAINT fk_inventory_planning_release
        FOREIGN KEY (source_promotion_id, source_production_run_id)
        REFERENCES model_promotion_log (id, production_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT ck_inventory_planning_run_status
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT ck_inventory_planning_run_method
        CHECK (target_method IN ('analytical_ci', 'quantile_protection_shadow')),
    CONSTRAINT ck_inventory_planning_run_hash
        CHECK (char_length(config_hash) = 64),
    CONSTRAINT ck_inventory_planning_run_counts
        CHECK (rows_read >= 0 AND rows_written >= 0),
    CONSTRAINT ck_inventory_planning_run_exclusions
        CHECK (jsonb_typeof(exclusions) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_inventory_planning_run_release
    ON inventory_planning_run (source_promotion_id, created_at DESC);

ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS inventory_run_id UUID;
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS source_promotion_id INTEGER;
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS source_production_run_id UUID;
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS inventory_snapshot_date DATE;
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS policy_source TEXT;
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS target_method TEXT;
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS opening_inventory_qty NUMERIC(15,4);
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS scheduled_receipt_qty NUMERIC(15,4);
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS recommended_receipt_qty NUMERIC(15,4);
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS projected_ending_inventory_qty NUMERIC(15,4);
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS shortage_qty NUMERIC(15,4);
ALTER TABLE fact_replenishment_plan
    ADD COLUMN IF NOT EXISTS excess_qty NUMERIC(15,4);

CREATE INDEX IF NOT EXISTS idx_replenishment_plan_inventory_run
    ON fact_replenishment_plan (inventory_run_id, item_id, loc, plan_month)
    WHERE inventory_run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS fact_inventory_opportunity (
    opportunity_id TEXT PRIMARY KEY,
    inventory_run_id UUID NOT NULL
        REFERENCES inventory_planning_run (inventory_run_id) ON DELETE RESTRICT,
    item_id TEXT NOT NULL,
    loc TEXT NOT NULL,
    plan_month DATE NOT NULL,
    opportunity_type TEXT NOT NULL,
    allocation_priority SMALLINT NOT NULL,
    current_qty NUMERIC(20,4) NOT NULL,
    remaining_qty NUMERIC(20,4) NOT NULL,
    reducible_qty NUMERIC(20,4) NOT NULL,
    current_book_value NUMERIC(20,2) NOT NULL,
    purchase_avoidance_value NUMERIC(20,2) NOT NULL,
    annual_carrying_cost_savings NUMERIC(20,2) NOT NULL,
    recoverable_cash_value NUMERIC(20,2) NOT NULL,
    enterprise_reduction_value NUMERIC(20,2) NOT NULL,
    evidence_quality_score NUMERIC(10,6),
    risk_level TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    rationale TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (inventory_run_id, item_id, loc, plan_month, opportunity_type),
    CONSTRAINT ck_inventory_opportunity_type CHECK (
        opportunity_type IN (
            'open_po_reduction',
            'rebalance_transfer',
            'excess_stock_reduction'
        )
    ),
    CONSTRAINT ck_inventory_opportunity_quantities CHECK (
        current_qty >= 0
        AND remaining_qty >= 0
        AND reducible_qty >= 0
        AND current_qty = remaining_qty + reducible_qty
    ),
    CONSTRAINT ck_inventory_opportunity_values CHECK (
        current_book_value >= 0
        AND purchase_avoidance_value >= 0
        AND annual_carrying_cost_savings >= 0
        AND recoverable_cash_value >= 0
        AND enterprise_reduction_value >= 0
    ),
    CONSTRAINT ck_inventory_opportunity_score CHECK (
        evidence_quality_score IS NULL
        OR evidence_quality_score BETWEEN 0 AND 1
    ),
    CONSTRAINT ck_inventory_opportunity_risk
        CHECK (risk_level IN ('low', 'medium', 'high'))
);

CREATE INDEX IF NOT EXISTS idx_inventory_opportunity_run_rank
    ON fact_inventory_opportunity (
        inventory_run_id,
        risk_level,
        enterprise_reduction_value DESC
    );
CREATE INDEX IF NOT EXISTS idx_inventory_opportunity_dfu
    ON fact_inventory_opportunity (item_id, loc, created_at DESC);

CREATE TABLE IF NOT EXISTS planning_decision_event (
    decision_event_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    state_version BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    prior_state TEXT,
    new_state TEXT NOT NULL,
    actor_user_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    note TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    ai_ledger_id BIGINT REFERENCES ai_decision_ledger (id) ON DELETE RESTRICT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_type, entity_id, state_version),
    CONSTRAINT ck_planning_decision_entity
        CHECK (entity_type = 'inventory_opportunity'),
    CONSTRAINT ck_planning_decision_state
        CHECK (new_state IN ('accepted', 'dismissed', 'deferred')),
    CONSTRAINT ck_planning_decision_version CHECK (state_version > 0),
    CONSTRAINT ck_planning_decision_metadata
        CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_planning_decision_entity
    ON planning_decision_event (entity_type, entity_id, state_version DESC);

CREATE OR REPLACE FUNCTION planning_decision_event_block_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'planning_decision_event is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_planning_decision_event_immutable
    ON planning_decision_event;
CREATE TRIGGER trg_planning_decision_event_immutable
    BEFORE UPDATE OR DELETE ON planning_decision_event
    FOR EACH ROW EXECUTE FUNCTION planning_decision_event_block_mutation();

COMMIT;
