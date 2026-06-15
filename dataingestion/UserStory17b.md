# User Story 17b: Unified read view

**Phase:** 5 — Unified Orchestration (US17 split, part 2 of 5)
**Depends on:** US17a
**Complexity:** M  **Risk:** MEDIUM (read-path only)

## Story
As a **planner/operator**, I want **`GET /integration/jobs` to return both legacy `integration_job` rows and ingestion jobs from `job_history` in one normalized list**, so that **the integration UI shows every ingestion run regardless of which backend ran it — without losing history**.

## Background / Current State
US17a gives the row→shape translation. This story exposes a **single read surface** so the UI sees one merged stream. Writes still go to whichever backend owns them (cutover is US17c). Legacy `integration_job` rows must remain visible.

## Acceptance Criteria
- [ ] **AC1** — A SQL view `integration_job_unified` UNIONs: (a) all `integration_job` rows, and (b) ingestion rows from `job_history` (`job_type IN ('etl_pipeline','load_domain')`), each normalized to the integration `Job` columns using the same status map as US17a (`completed→success`).
- [ ] **AC2** — `GET /integration/jobs` (and the domain filter) reads the unified view; response shape is **byte-compatible** with today's `Job` model (existing API tests pass unchanged).
- [ ] **AC3** — Ordering is stable and sensible (most-recent first across both sources); `limit`/`domain` filters apply to the merged set.
- [ ] **AC4** — Legacy `integration_job` rows still appear (no data loss); a freshly-submitted `etl_pipeline` job appears too.
- [ ] **AC5** — DDL added in `sql/` (next sequence), registered in `docs/RUNBOOK.md` cleanup section and `db-truncate-data` is reviewed (view needs no truncate, but document it).

## TDD Plan
### Write first (red)
- `tests/api/test_integration_router.py::test_list_jobs_includes_job_history_etl` (mock pool returns merged rows → both a legacy and an etl_pipeline row surface)
- `::test_list_jobs_shape_unchanged` (fields match the `Job` model exactly)
- `::test_list_jobs_status_mapped_completed_to_success`
- `tests/unit/test_job_shape.py` reused for the per-row mapping (US17a).
### Then implement (green) → Refactor
- Add the view DDL; point `IntegrationRunner.list`/the list endpoint at it (or a query that selects from the view).

## Implementation Notes
- The view does the status/derivation mapping in SQL mirroring US17a's Python (single source of truth for the vocabulary — add a test asserting they agree).
- Keep `GET /integration/jobs/{id}` working for both id spaces (UUID legacy vs TEXT job_id) — detect/branch, or have the view expose a unified `id`.
- Read-path only: do not touch submission yet.

## Definition of Done
- [ ] One merged, normalized read surface; existing API tests green.
- [ ] View DDL + RUNBOOK cleanup entry in the same change. `make test-all`, `make audit-routers` green.
