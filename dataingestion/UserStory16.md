# User Story 16: Register etl_pipeline job type (full + refresh)

**Phase:** 5 — Unified Orchestration
**Depends on:** US6
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **planner**, I want **the full/incremental pipeline runnable as a managed job**, so that **`run_pipeline.py` is no longer CLI-only and gets retries, logs, scheduling, and restart-recovery for free**.

## Background / Current State
`run_pipeline.py` (full ETL orchestration) is only invokable via Make/CLI — there is **no** job type for it in `JOB_TYPE_REGISTRY` (`common/services/job_registry.py` L103–478). `JobManager` (APScheduler) already provides persistence, retries, logs, recurring schedules, pipeline chaining, and PID restart-recovery.

## Acceptance Criteria
- [ ] **AC1** — A new `etl_pipeline` job type is registered with params `{mode: "full"|"refresh", domains: list|null, parallel: bool}` (group e.g. `"etl"`, max 1 concurrent).
- [ ] **AC2** — Submitting the job runs the unified orchestrator and streams progress via the JobManager callback (visible in `/jobs/{id}/logs`).
- [ ] **AC3** — Job status transitions (queued→running→completed/failed) persist to `job_history`; failures retain the error.
- [ ] **AC4** — The job can be scheduled recurring (cron/interval) via existing `schedule_recurring`.
- [ ] **AC5** — Concurrency: a second `etl_pipeline` submission while one runs is queued, not run in parallel (group lock).

## TDD Plan
### Write first (red)
- `tests/unit/test_job_registry.py::test_etl_pipeline_job_registered`
- `::test_etl_pipeline_params_validated` (mode/domains/parallel)
- `::test_etl_pipeline_group_concurrency_limit`
- `::test_etl_pipeline_progress_callback_emits_logs`
### Then implement (green) → Refactor
- Add registry entry + a job handler that calls the orchestrator function (not subprocess where avoidable).

## Implementation Notes
- Long full-loads may belong on pg-queue, not APScheduler, per CLAUDE.md ("don't put multi-hour jobs on APScheduler"). **Decision point:** if a full load exceeds the APScheduler comfort window, route `mode=full` to `pg_queue` (`common/services/pg_queue.py`) and keep `mode=refresh` on APScheduler. Capture the chosen split in the implementation + docs.
- Reuse the orchestrator as an importable function so the handler doesn't shell out.

## Definition of Done
- [ ] `etl_pipeline` job type available with full+refresh modes.
- [ ] `tests/unit/test_job_registry.py`, `make test-all` green.
- [ ] Long-job routing decision documented.
