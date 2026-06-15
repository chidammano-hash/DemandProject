# User Story 19: UI — full + incremental triggers with live status

**Phase:** 6 — UI Integration
**Depends on:** US18
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **planner/operator**, I want **to trigger a full load or incremental refresh (per-domain or all-domains) from the UI and watch it live**, so that **I don't need CLI/Make access to run ingestion**.

## Background / Current State
`frontend/src/tabs/IntegrationTab.tsx` already does single-domain submit + active-job polling (2s) + history. It lacks an "all-domains pipeline" control for `full`/`refresh`. Components live in `frontend/src/components/integration/`.

## Acceptance Criteria
- [ ] **AC1** — IntegrationTab gains a "Run Pipeline" control: choose `full` or `refresh`, all-domains or a subset, optional `parallel`.
- [ ] **AC2** — Submitting calls the US18 endpoint via a typed function in `frontend/src/api/queries/` (no raw `fetch` — `fetchJson` only).
- [ ] **AC3** — Live status + logs stream via the unified job polling (reuses existing job-status query); progress visible without refresh.
- [ ] **AC4** — Destructive `full` run shows the existing cascade-safety confirmation before submit.
- [ ] **AC5** — Query types mirror the backend Pydantic schema; no `: any`/`as any` in `src/api/queries/`.
- [ ] **AC6** — Tab file stays < 600 lines (split into `frontend/src/tabs/integration/` panels if needed).

## TDD Plan
### Write first (red)
- `frontend/src/tabs/__tests__/IntegrationTab.test.tsx` — render pipeline control; submit full → calls query fn with `{mode:"full"}`; submit refresh subset → correct payload; destructive confirm gate; live status renders from polled job.
- `frontend/src/tabs/__tests__/no-raw-fetch.test.ts` — still passes (no new raw fetch).
### Then implement (green) → Refactor
- Add query fn + UI control + status wiring; split panels to respect line limit.

## Implementation Notes
- Charts/colors via context (not relevant here, but keep theme rules if any chart added).
- Mock API with `vi.mock("../api/queries")`; wrap with `TestQueryWrapper`.

## Definition of Done
- [ ] Full + incremental runnable from UI with live status.
- [ ] `IntegrationTab.test.tsx`, `no-raw-fetch` green; vitest suite green.
- [ ] Tab < 600 lines.
