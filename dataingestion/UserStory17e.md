# User Story 17e: UI convergence + legacy retirement

**Phase:** 5 — Unified Orchestration (US17 split, part 5 of 5)
**Depends on:** US17c, US17d
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **maintainer**, I want **both UIs to read the one unified job backend and the legacy `integration_job`/`integration_chain` write paths retired**, so that **the ingestion stack is genuinely one system — this is where the net simplification (≈700 fewer LOC, one fewer table pair) actually lands**.

## Background / Current State
After US17a–d, all ingestion writes go to `job_history`; `integration_runner.py` / `integration_chain_runner.py` and the legacy tables are dead weight kept only for read-compat. This story removes them and converges the frontend.

## Acceptance Criteria
- [ ] **AC1** — IntegrationTab and JobsTab read the same unified job source (shared query module + job shape); the two duplicate job-polling stacks collapse to one.
- [ ] **AC2** — `IntegrationRunner` / `integration_chain_runner` legacy code paths are removed (no `INSERT INTO integration_job` / `integration_chain` remain); the read view becomes the historical archive or is migrated.
- [ ] **AC3** — Migration: any still-needed legacy rows are migrated into `job_history` (or the union view is kept as the permanent archive). DDL in `sql/`; `db-truncate-data` + `docs/RUNBOOK.md` cleanup updated; `data/`/MV refs reviewed.
- [ ] **AC4** — No `: any` in queries; tab files < 600 lines (split if needed); `no-raw-fetch` passes; charts/theme rules unaffected.
- [ ] **AC5** — Docs updated (`docs/ARCHITECTURE.md` job/ingestion section; RUNBOOK) describing the single backend; `dataingestion/STATUS.md` marks US17 complete.

## TDD Plan
### Write first (red)
- `frontend` — IntegrationTab/JobsTab tests against the unified query module; one polling path.
- `tests/api` — assert no endpoint still depends on the legacy tables; `/integration/jobs` + `/jobs` serve from one store.
- `tests/unit` — assert `integration_runner` legacy insert path is gone (grep-style guard).
### Then implement (green) → Refactor
- Converge FE queries; delete legacy runners/tables; migrate/retire; cross-cutting self-review of the merged result.

## Implementation Notes
- This is the only story that *deletes*; do it after 17c/17d have proven the unified path in production-like runs.
- Prefer keeping the unified read view as the permanent history surface so old run data stays queryable.

## Definition of Done
- [ ] One job backend, one FE job stack; legacy runners/tables removed or archived.
- [ ] Migration DDL + RUNBOOK/cleanup in the same change; full gate green; US17 closed in STATUS.md.
