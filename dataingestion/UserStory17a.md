# User Story 17a: Shape/status adapter (read-only, zero behavior change)

**Phase:** 5 — Unified Orchestration (US17 split, part 1 of 5)
**Depends on:** US16
**Complexity:** S  **Risk:** LOW

## Story
As a **platform engineer**, I want **pure functions that translate a `job_history` row into the `/integration/jobs` `Job` shape**, so that **a JobManager-backed ingestion job can later be surfaced through the existing integration UI without changing its response contract**.

## Background / Current State
The two job backends store different shapes:
- `integration_job` (sql/090): `id` UUID, `domain`, `mode` (onetime|delta|file), `slice`, `file_path`, `status` (queued|running|**success**|failed|skipped), `rows_loaded`, `error_message`, `started_at`, `completed_at`, `duration_ms`, `triggered_by`.
- `job_history` (sql/020): `job_id` TEXT, `job_type`, `job_label`, `status` (queued|running|**completed**|failed|...), `params` JSONB, `result` JSONB, `progress_pct`, `logs`, `triggered_by`, timestamps.

Converging them starts with a **translation layer** — no writes, no endpoints change yet. The `etl_pipeline` job (US16) stores `mode`/`domains` in `params` and `loaded`/`results` in `result`; a `load_domain` job (US17c) will store `domain`/`mode`/`slice` in `params` and `rows_loaded` in `result`.

## Acceptance Criteria
- [ ] **AC1** — New module (e.g. `common/services/job_shape.py`) exposes `job_history_to_integration_job(row: dict) -> dict` returning the exact `Job` field set the `/integration/jobs` API model uses (`id`, `domain`, `mode`, `slice`, `status`, `rows_loaded`, `error_message`, `started_at`, `completed_at`, `triggered_by`).
- [ ] **AC2** — Status vocabulary is mapped both ways via a single source of truth: `completed → success`, `success → completed`; all other statuses pass through unchanged. Expose `to_integration_status()` / `to_job_history_status()`.
- [ ] **AC3** — `domain`/`mode`/`slice` are derived from `params`; `rows_loaded` from `result` (fall back to 0/None when absent). A non-ETL `job_history` row maps to a sensible default without raising.
- [ ] **AC4** — Functions are **pure** (dict → dict), no DB or network access; importing the module triggers no DB connection.
- [ ] **AC5** — **Zero behavior change** elsewhere: no endpoint, table, runner, or UI is modified in this story.

## TDD Plan
### Write first (red)
- `tests/unit/test_job_shape.py::test_status_round_trip` (completed↔success, passthrough for queued/running/failed/skipped)
- `::test_maps_etl_pipeline_row_to_integration_shape`
- `::test_maps_load_domain_row_to_integration_shape`
- `::test_missing_params_result_defaults` (no KeyError; rows_loaded=0)
- `::test_pure_no_db_import` (import + call with a plain dict; no get_db_params/connect)
### Then implement (green) → Refactor
- Add `common/services/job_shape.py`; keep it dependency-free.

## Implementation Notes
- Keep the field list in lockstep with `api/routers/platform/integration.py::Job`. Add a comment cross-referencing it so US17b/US17e reuse it.
- Do **not** import this into any router yet — that's US17b.

## Definition of Done
- [ ] Pure adapter + status map with full unit coverage.
- [ ] No other file's behavior changed. `make test-all` green.
