# UX Hardening Loop — Summary

A persona-driven critique→fix loop run against the **live** app (React :5173 → host
uvicorn :8000, real Postgres). Each cycle: a **Demand Planner** agent drove Playwright
over 14 tabs and filed findings; a **Usability/Simplification** agent reviewed rendered
UI + code; a **Technical Fixer** applied changes under **strict test-first TDD**
(red→green) and re-verified against the live endpoints.

## Outcome (Cycle 1 → Cycle 9)

| Signal | Before | After |
|---|---|---|
| Console errors across 14 tabs | 22 (on 7 tabs) | **0** |
| 500s on planner core flows | Command Center, Demand History, Control Tower, AI Planner, Item Analysis | none |
| 404s | Data Quality lineage | none |
| Issues fixed & verified | — | **40** |
| Backend suite | baseline | 4155 passing (pre-existing torch/cross-dim failures excluded) |

Loop ran 9 productive cycles (1 manual + 2–9 TDD); cycles 10–11 hit the session
limit, which tripped the 2-cycle dry-stop. Cycle 9's scan was already clean.

## Highest-impact fixes
- **Action Feed** showed 0 actions despite 6,142 critical exceptions (wrong column +
  shared-transaction abort) → real critical actions; summary later corrected from a
  20-row undercount to the full population (6,214 / 4,252 critical / $12,099.96).
- **Command Center** stopped showing a false "Portfolio looks healthy!"; KPIs now read
  from live exception counts with an honest stale-data banner.
- **Error handling**: raw `{"detail":...}` JSON no longer leaks into toasts.
- **Navigation**: deep-links survive refresh; Back/Forward no longer hits dead tabs.
- **Honest data reads**: negative accuracy floored to `<0%*`; KPI arrows colored by
  good/bad direction; `"null"` cells → `–`; dirty CA dropdowns normalized.
- **Performance**: Item×State heatmap 9.4s → 0.43s via `mv_ca_item_state`.

## Artifacts
- `LEDGER.md` — per-cycle issue → FIXED/DEFERRED index with red→green evidence.
- `testinputN.md` / `usabilityN.md` — each cycle's findings.
- `cycleN/fixes-applied.md` — fixer write-ups; `cycleN/capture-digest.md` — captured UI text.
- `_harness/` — the Playwright capture script + the workflow loop script.
- Screenshots (`cycleN/screens/`) and raw `capture-dump.json` are gitignored (20MB).

## Known deferrals (not defects / out of scope)
S&OP "New Cycle" needs a `POST /sop/cycles` route; six oversized tab files want a
pure refactor; some MV-level filter cleanup; FVA Champion "No data" is a genuine
empty-data state. See `LEDGER.md` Deferred entries.
