"""Structural guards for grounded AI and inventory-opportunity persistence."""

from pathlib import Path

MIGRATION = Path("sql/204_create_grounded_copilot_inventory_opportunities.sql")


def _ddl() -> str:
    return MIGRATION.read_text()


def test_copilot_session_turn_and_evidence_contracts_are_owner_scoped() -> None:
    ddl = _ddl()
    compact = " ".join(ddl.split())

    for table in ("ai_copilot_session", "ai_copilot_turn", "ai_copilot_evidence"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl
    assert "owner_user_id TEXT NOT NULL" in compact
    assert "UNIQUE (owner_user_id, client_request_id)" in compact
    assert "UNIQUE (session_id, client_request_id)" in compact
    assert "content_hash TEXT NOT NULL" in compact
    assert "char_length(content_hash) = 64" in compact
    assert "value_snapshot JSONB NOT NULL" in compact
    assert "content_expires_at TIMESTAMPTZ" in compact
    assert "content_redacted_at TIMESTAMPTZ" in compact


def test_inventory_plan_gains_release_snapshot_and_method_lineage() -> None:
    ddl = _ddl()

    assert "CREATE TABLE IF NOT EXISTS inventory_planning_run" in ddl
    assert "source_promotion_id INTEGER NOT NULL" in ddl
    assert "source_production_run_id UUID NOT NULL" in ddl
    assert "config_hash TEXT NOT NULL" in ddl
    assert "exclusions JSONB NOT NULL" in ddl
    assert "ALTER TABLE fact_replenishment_plan" in ddl
    for column in (
        "inventory_run_id UUID",
        "source_promotion_id INTEGER",
        "source_production_run_id UUID",
        "inventory_snapshot_date DATE",
        "policy_source TEXT",
        "target_method TEXT",
        "opening_inventory_qty NUMERIC(15,4)",
        "scheduled_receipt_qty NUMERIC(15,4)",
        "projected_ending_inventory_qty NUMERIC(15,4)",
        "shortage_qty NUMERIC(15,4)",
        "excess_qty NUMERIC(15,4)",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in ddl


def test_opportunities_enforce_physical_and_financial_invariants() -> None:
    ddl = _ddl()
    compact = " ".join(ddl.split())

    assert "CREATE TABLE IF NOT EXISTS fact_inventory_opportunity" in ddl
    assert "current_qty = remaining_qty + reducible_qty" in compact
    for column in (
        "current_book_value NUMERIC(20,2) NOT NULL",
        "purchase_avoidance_value NUMERIC(20,2) NOT NULL",
        "annual_carrying_cost_savings NUMERIC(20,2) NOT NULL",
        "recoverable_cash_value NUMERIC(20,2) NOT NULL",
        "enterprise_reduction_value NUMERIC(20,2) NOT NULL",
    ):
        assert column in compact
    assert "opportunity_type IN (" in ddl
    assert "'rebalance_transfer'" in ddl


def test_review_events_are_immutable_versioned_and_idempotent() -> None:
    ddl = _ddl()

    assert "CREATE TABLE IF NOT EXISTS planning_decision_event" in ddl
    assert "UNIQUE (entity_type, entity_id, state_version)" in ddl
    assert "idempotency_key TEXT NOT NULL UNIQUE" in ddl
    assert "accepted" in ddl and "dismissed" in ddl and "deferred" in ddl
    assert "planning_decision_event_block_mutation" in ddl
    assert "BEFORE UPDATE OR DELETE ON planning_decision_event" in ddl
