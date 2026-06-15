# User Story 17: Converge job backends into one

**Phase:** 5 — Unified Orchestration
**Depends on:** US16
**Complexity:** L  **Risk:** HIGH

## Story
As a **platform engineer**, I want **a single job backend and a single job history**, so that **UI-triggered loads and scheduled pipeline runs share one source of truth instead of `integration_jobs` vs `job_history`**.

## Background / Current State
Two backends/tables today:
- `IntegrationRunner` → `integration_jobs` (UI single/chain loads)
- `JobManager`/APScheduler → `job_history` (everything else)

Recommendation: **JobManager** is canonical (richer features). `IntegrationRunner` job submission/tracking migrates onto it; `integration_jobs` becomes a read-compatibility view or is migrated.

## Acceptance Criteria
- [ ] **AC1** — Integration job submission goes through `JobManager.submit_job("etl_pipeline"/load type, …)`; new loads land in `job_history`.
- [ ] **AC2** — `GET /integration/jobs` and the chain endpoints continue to return the same shape to the UI (backed by the unified store — view or adapter).
- [ ] **AC3** — Existing `integration_jobs` rows remain readable during transition (no data loss); migration path documented (DDL in `sql/` next sequence, added to RUNBOOK cleanup + `db-truncate-data`).
- [ ] **AC4** — Cascade-safety and file-sandbox gates (from US7) still enforced at submission.
- [ ] **AC5** — Both `IntegrationTab` and `JobsTab` show the same jobs from the unified backend.

## TDD Plan
### Write first (red)
- `tests/api/test_integration.py::test_submit_routes_to_jobmanager`
- `::test_integration_jobs_list_shape_unchanged`
- `::test_legacy_integration_jobs_still_readable`
- `tests/unit/test_job_registry.py::test_load_job_safety_gates_enforced`
### Then implement (green) → Refactor
- Adapter so `/integration/*` reads/writes the unified store; deprecate `IntegrationRunner`'s separate table writes.

## Implementation Notes
- Prefer a backward-compatible **view** (`integration_jobs` → `job_history`) over a hard cutover, to keep the UI shape stable.
- This is the highest-risk story — ship alone, behind full test coverage; do not bundle with US18/US19.

## Definition of Done
- [ ] One backend (JobManager) authoritative; one history store.
- [ ] UI endpoints unchanged in shape; legacy rows preserved.
- [ ] Migration DDL + RUNBOOK cleanup entry in same change. `make test-all`, `make audit-routers` green.
