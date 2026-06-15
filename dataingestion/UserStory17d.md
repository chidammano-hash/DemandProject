# User Story 17d: Chains on the unified backend

**Phase:** 5 — Unified Orchestration (US17 split, part 4 of 5)
**Depends on:** US17c
**Complexity:** M  **Risk:** MEDIUM–HIGH

## Story
As a **planner/operator**, I want **multi-step ingestion chains to run on the unified job backend**, so that **the scan→chain flow and per-domain loads share one job system instead of a separate `integration_chain` runner**.

## Background / Current State
`integration_chain_runner.py` (~334 lines) + the `integration_chain` table (sql/092) orchestrate sequential domain loads, separate from JobManager. JobManager already has `submit_pipeline(steps, ...)` for sequential chained jobs. After US17c the per-step unit (`load_domain`) is a JobManager job, so a chain becomes a JobManager pipeline of `load_domain` steps.

## Acceptance Criteria
- [ ] **AC1** — `POST /integration/chains` builds a JobManager pipeline (`submit_pipeline`) of `load_domain` steps in dependency order, instead of inserting `integration_chain` rows.
- [ ] **AC2** — `GET /integration/chains` and `/chains/{id}` return the same response shape to the UI, sourced from the unified store (pipeline id + per-step job rows), via an adapter mirroring US17a/b.
- [ ] **AC3** — `GET /integration/scan` (directory scan → proposed chain) is unchanged; only the submission/tracking backend changes.
- [ ] **AC4** — Step ordering, per-step status, and stop-on-failure semantics match the legacy chain runner.
- [ ] **AC5** — Existing `integration_chain` rows remain readable during transition (union/adapter); no deletion here.

## TDD Plan
### Write first (red)
- `tests/api/test_integration_chain.py::test_submit_chain_builds_pipeline` (asserts submit_pipeline called with ordered load_domain steps)
- `::test_chain_list_shape_unchanged`
- `::test_chain_detail_aggregates_step_jobs`
- `::test_scan_endpoint_unchanged`
### Then implement (green) → Refactor
- Repoint chain submission to `JobManager.submit_pipeline`; add a chain→pipeline adapter; keep the scan logic.

## Implementation Notes
- Reuse the US17a shape adapter for per-step rows; the chain "envelope" maps to JobManager's `pipeline_id`/`pipeline_step` columns (already in `job_history`).
- Decide whether to keep `integration_chain` as a thin read-compat view or fully migrate; prefer a view to avoid a hard cutover.

## Definition of Done
- [ ] Chains run as JobManager pipelines; UI shape unchanged; legacy chains still readable.
- [ ] `make test-all`, `make audit-routers` green.
