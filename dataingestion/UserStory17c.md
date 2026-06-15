# User Story 17c: Submission cutover (single-domain loads → JobManager)

**Phase:** 5 — Unified Orchestration (US17 split, part 3 of 5)
**Depends on:** US17a, US17b
**Complexity:** L  **Risk:** HIGH — **ship entirely alone**

## Story
As a **platform engineer**, I want **per-domain integration loads to be submitted as JobManager jobs instead of `integration_job` inserts**, so that **all ingestion writes converge on one backend (`job_history`) while the UI and safety gates stay intact**.

## Background / Current State
This is the real write cutover. Today `IntegrationRunner.submit()` inserts `integration_job` and spawns `python -m scripts.etl.load`. After this story, submission creates a JobManager job (a new `load_domain` job type) whose handler runs the same loader. The read surface (US17b unified view) already merges sources, so the list keeps working through the transition.

## Acceptance Criteria
- [ ] **AC1** — A new `load_domain` JobManager job type (group `etl`) runs a single-domain load (`domain`, `mode` onetime|delta|file, `slice`, `file`, `reindex`) via the unified engine (`scripts.etl.load`), recording status/rows in `job_history`.
- [ ] **AC2** — `POST /integration/jobs` submits the `load_domain` job; response shape (`job_id`, `status`) is unchanged for the UI.
- [ ] **AC3** — Cascade-safety (confirm_destructive for onetime/file) and file-path sandboxing (`INTEGRATION_DATA_ROOT`) are enforced **before** submission, exactly as today (same 409/422 behavior; existing router tests pass).
- [ ] **AC4** — Status vocabulary written as `job_history` values (`completed`), surfaced as `success` via the US17a/b mapping.
- [ ] **AC5** — Legacy `integration_job` rows remain readable (US17b view) — no migration/deletion in this story.
- [ ] **AC6** — `GET /integration/jobs/{id}` resolves a `load_domain` job id.

## TDD Plan
### Write first (red)
- `tests/unit/test_job_registry.py::test_load_domain_job_registered` (+ group concurrency)
- `tests/unit/test_load_domain_job.py::test_handler_invokes_unified_engine` (mock subprocess/loader)
- `tests/api/test_integration_router.py::test_submit_routes_to_jobmanager`
- `::test_submit_preserves_cascade_guard` / `::test_submit_preserves_file_sandbox`
- `::test_get_job_resolves_load_domain_id`
### Then implement (green) → Refactor
- Add `load_domain` job type + handler; repoint `IntegrationRunner.submit` (keep its gates); leave the legacy table writes off the new path.

## Implementation Notes
- Keep the gates in the API/runner layer (don't push them into the job handler) so rejection happens before a job row exists.
- Decide subprocess vs in-process: simplest is the handler shelling out to `scripts.etl.load` (same as today) wrapped in a JobManager callable — least behavior drift.
- **Do not bundle** with 17d/17e or any other story; this is the cutover. Rely on US17b's read view + the existing router tests as guardrails.

## Definition of Done
- [ ] New single-domain loads land in `job_history`; UI unaffected; gates intact; legacy rows still listed.
- [ ] `make test-all`, `make audit-routers` green; manual UI smoke of one load per mode.
