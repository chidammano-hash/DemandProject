# 05 — AI Decision Ledger + Policy Engine

**Gen-4 Roadmap AI-10 P0 + AI-1 P0.** Append-only audit trail and
declarative autonomy guardrails for every AI write action.

**Status:** Ledger foundation landed (2026-04-23) and is callable from any
agent. The policy engine prototype (`common/ai/policy_engine.py`) was built
and unit-tested but never wired into a runtime path, and has since been
removed as unwired code. The guardrail design below is therefore **not yet
implemented** — it remains the intended Phase 0 (weeks 1–8) work, to be
rebuilt against real agent call sites rather than as a standalone module.

---

## Why

Gen-4 agents will propose and, in some cases, auto-apply changes to
forecasts, safety stock, transfers, and S&OP plans. Two invariants must
hold before any agent writes:

1. **Auditability.** Every action is logged with enough detail to replay,
   diff, or roll back, and the log must be tamper-evident.
2. **Bounded autonomy.** The action must pass declarative guardrails that
   encode what the organization considers safe for this agent to do
   without a human in the loop.

Without these two, giving agents write access is unsafe.

---

## Ledger

- DDL: [sql/137_create_ai_decision_ledger.sql](../../sql/137_create_ai_decision_ledger.sql)
- Python helper: [common/ai/decision_ledger.py](../../common/ai/decision_ledger.py)
- Tests: [tests/unit/test_decision_ledger.py](../../tests/unit/test_decision_ledger.py)

Each row is a SHA-256 hash over the canonical concatenation of the row's
fields plus the prior row's hash. The first row references a GENESIS
sentinel (64 zero hex characters). Two DB triggers enforce the contract:

1. **append-only** — `BEFORE UPDATE OR DELETE` raises, blocking mutation.
2. **chain verification** — `BEFORE INSERT` recomputes the expected
   `prev_hash` and the canonical SHA-256 digest; mismatches are rejected.

Columns:

| Column | Purpose |
|---|---|
| `ts` | server timestamp |
| `agent_id` | which agent produced the row (e.g. `demand_agent`) |
| `action_type` | canonical action verb (`promote_model`, `auto_resolve_exception`, ...) |
| `autonomy_tier` | `advisory` / `suggestive` / `auto_within_policy` / `autonomous` |
| `subject_kind`, `subject_id` | the entity acted on (`dfu`, `po`, `model_id`, ...) |
| `payload` | JSONB of inputs and outputs (model scores, quantities, etc.) |
| `policy_id` | the policy in `agent_autonomy.yaml` that authorized this |
| `prev_hash` | previous row's `row_hash` (or GENESIS) |
| `row_hash` | SHA-256 over the canonical tuple |
| `actor` | `system` or user_id when a human overrode |
| `outcome` | `applied` / `rolled_back` / `rejected` / `superseded` |

**Writing:** Use `append_decision(cursor, DecisionRecord(...))`. The helper
fetches the latest row's hash, computes the new hash, and inserts under
the active transaction. A chain break raises a DB exception; the caller
can either retry under a savepoint or surface the failure.

**Auditing:** Use `verify_chain(cursor)` to walk the full ledger. Every
mismatch (bad `prev_hash` or recomputed `row_hash`) is appended to the
error list, so a single pass surfaces every break.

---

## Policy Engine (planned — not yet implemented)

- Policies: [config/ai/agent_autonomy.yaml](../../config/ai/agent_autonomy.yaml) (config exists)
- Engine: not yet implemented. An earlier `common/ai/policy_engine.py`
  prototype was removed as unwired (no production callers); the engine will
  be rebuilt during Phase 0 against real agent write sites.

The design below describes the intended behavior. Every write action will
supply an `ActionContext` with the runtime facts
(policy id, requested tier, blast radius, magnitude, human review flag,
etc.). `evaluate(ctx)` returns a `PolicyDecision`:

- `permitted: bool` — whether the action is allowed
- `effective_tier: str` — tier to record on the ledger row
- `reasons: list[str]` — one string per guardrail that fired (empty if
  permitted)

Guardrails planned for the engine:

- `tier` ceiling (requested must not exceed max)
- `requires_human_review`
- `max_blast_radius_skus`
- `max_pct_change_per_sku`
- `max_units_per_action`, `max_dollar_per_action`
- `cooldown_hours`
- `requires_open_exceptions_resolved`

New guardrails follow the same pattern: add a branch in
`policy_engine.evaluate`, a field in `ActionContext`, and config entries.

---

## Intended Integration

Once the engine is rebuilt, every agent-originated write will go through
this envelope:

```python
ctx = ActionContext(
    policy_id="demand.promote_champion_model",
    requested_tier="suggestive",
    blast_radius_skus=len(affected_dfus),
    has_human_review=user_confirmed,
)
decision = evaluate(ctx)
if not decision.permitted:
    raise PolicyDenied(decision.reasons)

with conn, conn.cursor() as cur:
    append_decision(cur, DecisionRecord(
        agent_id="demand_agent",
        action_type="promote_model",
        autonomy_tier=decision.effective_tier,
        subject_kind="model_id",
        subject_id=model_id,
        payload={"wape_before": ..., "wape_after": ...},
        policy_id=decision.policy_id,
    ))
    # ... perform the actual mutation ...
    conn.commit()
```

---

## Follow-ups (subsequent roadmap work)

- Phase 0: wire the existing `AIPlannerAgent` monolith to the policy engine
  before the planned split into specialist agents.
- Phase 0: expose `/admin/ledger/verify` API for operators; fail CI on any
  chain break.
- Phase 4: add fairness and counterfactual enrichments to the payload so
  the ledger is the data source for the XAI layer.
