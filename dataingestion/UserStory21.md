# User Story 21: Docs + final verification

**Phase:** 7 — Verification
**Depends on:** all (US1–US20)
**Complexity:** S  **Risk:** LOW

## Story
As a **maintainer**, I want **docs updated and the full quality gate green**, so that **the unified pipeline is documented and verifiably correct before it's considered done**.

## Background / Current State
CLAUDE.md requires docs in the same commit as code; several were touched piecemeal across US1–US20. This story is the consolidation + end-to-end verification pass.

## Acceptance Criteria
- [ ] **AC1** — `docs/ARCHITECTURE.md` data-flow section reflects: one load engine, one orchestrator, one job backend, UI triggers.
- [ ] **AC2** — `docs/RUNBOOK.md` documents the unified commands, the full vs refresh job, the long-job routing decision (US16), before/after benchmarks (US2 vs US8–US11), and DB cleanup for any new tables/views (US17).
- [ ] **AC3** — `docs/specs/01-foundation/01-infrastructure.md` "Implemented Features" updated; relevant `docs/specs/<domain>/` ETL spec updated.
- [ ] **AC4** — `CLAUDE.md` updated only if a new critical rule emerged (e.g. "one load engine — don't reintroduce load.py"); MEMORY.md index pointer added if useful.
- [ ] **AC5** — Full gate green: `make test-all`, `make audit-routers`, `make lint`, `make type-check`; rule gate (`check_unenforced_rules.sh`) clean for `scripts/etl/`.
- [ ] **AC6** — E2E: `frontend/e2e/tests/navigation.spec.ts` covers any new/changed sidebar control (semantic selectors only).

## TDD Plan
Verification story — assertions are the gates themselves.
### Run
- `~/.local/bin/uv run pytest tests/ -q`
- frontend vitest (per MEMORY.md command)
- `make audit-routers`, lint, type-check, rule gate
- targeted E2E for the integration tab pipeline control.

## Implementation Notes
- Cross-cutting self-review of the merged result (CLAUDE.md orchestrator review): re-read the full diff across all stories; remove dead code, stray shims, leftover `load.py` references.

## Definition of Done
- [ ] All docs updated and consistent with the shipped code.
- [ ] Every gate green; epic Definition of Done (README) satisfied.
- [ ] No references to deleted `load.py` / divergent loaders remain (`grep` clean).
