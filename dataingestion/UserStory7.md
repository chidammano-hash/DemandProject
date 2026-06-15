# User Story 7: Repoint IntegrationRunner, delete load.py

**Phase:** 2 — Single Load Engine
**Depends on:** US6
**Complexity:** M  **Risk:** HIGH

> **REFRAMED during implementation (user-approved).** Since US6 established
> that `load.py` *is* the canonical mode dispatcher that `IntegrationRunner`
> already invokes (and which delegates bulk loads to `load_dataset_postgres.py`),
> `load.py` is **NOT deleted**. US7 reduces to: verify `IntegrationRunner`
> already targets the unified engine, confirm its cascade-safety / file-sandbox
> gates are intact, and confirm onetime delegates to the bulk loader. No
> reimplementation of the JSON/audit contract.

## Story
As a **platform engineer**, I want **IntegrationRunner to call the unified engine and load.py removed**, so that **the UI and CLI share one code path with no divergence**.

## Background / Current State
`common/services/integration_runner.py` spawns `subprocess.Popen` against `scripts/etl/load.py`. After US6 the unified engine lives in `load_dataset_postgres.py`. `IntegrationRunner` also owns safety gates (cascade detection, file-path sandboxing to `INTEGRATION_DATA_ROOT`) that must be preserved.

## Acceptance Criteria
- [ ] **AC1** — `IntegrationRunner` invokes `load_dataset_postgres.py` with the mode mapped from the job's `mode` (`onetime→full`, `delta→delta`, `file→file`).
- [ ] **AC2** — Cascade-safety warnings and file-path sandboxing behave exactly as before (same refusals, same allowed paths).
- [ ] **AC3** — `scripts/etl/load.py` is deleted; no remaining importers (`grep -rn "etl.load\b\|scripts/etl/load.py"` clean).
- [ ] **AC4** — Existing `IntegrationTab` submit/monitor flow works end-to-end against the new target (job status transitions unchanged).
- [ ] **AC5** — `integration_jobs` rows and `audit_load_batch` rows are still written for UI-triggered loads.

## TDD Plan
### Write first (red)
- `tests/unit/test_integration_runner.py::test_mode_mapping_onetime_delta_file`
- `::test_cascade_safety_refuses_unconfirmed_destructive`
- `::test_file_path_sandbox_rejects_outside_root`
- `::test_invokes_unified_engine_not_load_py`
- `tests/api/test_integration.py` — submit job → status path still 202 + terminal status.
### Then implement (green) → Refactor
- Repoint runner; delete `load.py`; remove dead imports.

## Implementation Notes
- Preserve `triggered_by` tagging and the `reindex` opt-in flag passthrough.
- This is the riskiest swap — rely on US1 + US6 parity tests as the guardrail; ship alone, not bundled with perf changes.

## Definition of Done
- [ ] `load.py` gone; IntegrationRunner on the unified engine.
- [ ] Safety gates verified intact by tests.
- [ ] `make test-all`, lint, type-check green; manual UI smoke of one load per mode.
