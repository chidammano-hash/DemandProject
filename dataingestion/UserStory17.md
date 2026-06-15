# User Story 17: Converge job backends into one

**Phase:** 5 — Unified Orchestration
**Depends on:** US16
**Complexity:** L  **Risk:** HIGH

> **SPLIT (deferred).** This story is too large/risky to ship as one unit — the
> two backends diverge in schema (UUID vs TEXT PK), status vocabulary
> (`success` vs `completed`), domain-columns vs `params/result` JSONB, execution
> model (subprocess vs in-process), plus a separate chains runner and two UIs.
> It is split into five independently shippable sub-stories, risk rising toward
> the write cutover:
> - **[US17a](UserStory17a.md)** — Shape/status adapter (read-only, zero behavior change). LOW.
> - **[US17b](UserStory17b.md)** — Unified read view; `/integration/jobs` merges both sources. MEDIUM.
> - **[US17c](UserStory17c.md)** — Submission cutover to a `load_domain` JobManager job. **HIGH — ship alone.**
> - **[US17d](UserStory17d.md)** — Chains on JobManager `submit_pipeline`. MEDIUM–HIGH.
> - **[US17e](UserStory17e.md)** — UI convergence + legacy retirement (the actual deletion/simplification). MEDIUM.
>
> Note: the epic's user-facing goal (run full+incremental from the UI, unified
> lineage) is already met by US16/18/19/20 — US17 is internal consolidation.
> A low-risk stopping point is US17a+US17b only (unified *appearance* without the
> risky write cutover). The text below is the original single-story framing.

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
