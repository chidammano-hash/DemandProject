# User Story 20: UI — unified load history & lineage

**Phase:** 6 — UI Integration
**Depends on:** US18
**Complexity:** S  **Risk:** LOW

## Story
As a **planner/operator**, I want **one place to see load history and lineage across full and incremental runs**, so that **I can audit what loaded, when, with what row counts and status**.

## Background / Current State
`DataQualityTab.tsx` surfaces `audit_load_batch` (Pipeline Lineage) via `GET /data-quality/batches`. `IntegrationTab` shows its own job history. Post-US17 the job stores are unified; the history views should reflect one coherent picture (batches + jobs).

## Acceptance Criteria
- [ ] **AC1** — Load history shows, per run: domain(s), mode (full/delta/file), status, rows in/out, started/completed, error (if any) — sourced from `audit_load_batch` + unified jobs.
- [ ] **AC2** — `customer_demand` now appears in lineage (it records batches after US5/US15).
- [ ] **AC3** — History is filterable by domain and status; failed runs show the (sanitized) error.
- [ ] **AC4** — All HTTP via `frontend/src/api/queries/` typed functions; no `: any`.
- [ ] **AC5** — Tabs stay < 600 lines.

## TDD Plan
### Write first (red)
- `frontend/src/tabs/__tests__/DataQualityTab.test.tsx` — lineage renders incl. customer_demand; filter by domain/status; failed row shows error.
- query-module test for the batches/jobs fetchers (typed shape).
### Then implement (green) → Refactor
- Extend lineage view to include mode + unified job linkage; reuse existing query fns.

## Implementation Notes
- No new endpoint expected (reuse `/data-quality/batches` + unified job list); if a join view is needed, prefer a backend MV/view over client-side merging.
- Mock API with `vi.mock`; `TestQueryWrapper`.

## Definition of Done
- [ ] One coherent history/lineage view incl. customer_demand and run mode.
- [ ] `DataQualityTab.test.tsx`, vitest suite green.
