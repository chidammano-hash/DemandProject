# Cycle 2 — Fixes Applied

Date: 2026-06-14
Branch: main (working tree, not committed)
Engineer: senior full-stack pass, strict TDD (red → green → refactor → live-verify)

Sources: `tests/Automated_tests/testinput2.md` (planner) + `tests/Automated_tests/usability2.md` (usability).
6 items fixed this cycle. Each had a test written FIRST that failed for the right reason, then a minimal fix that made it pass.

---

## F2.2 — FVA Champion rung read "No data" (state="missing") while AI/Planner read "Coming Soon" (state="planned") [P2]

**What was wrong:** `/fva/waterfall` computes every stage's accuracy from `fact_external_forecast_monthly`, which only ever contains `model_id='external'`. The champion forecast lives in `fact_production_forecast` and has **no measurable overlap with actuals in the window** (verified: champion-vs-`fact_sales_monthly` join over the 12-month pre-planning window returns 0 rows — champion item/loc pairs don't join to sales in 2026-01..03). So champion always fell to `state="missing"`, rendering as a broken-looking "No data" sandwiched between the AI/Planner reserved "Coming Soon" stages. Real champion accuracy is genuinely unmeasurable here — fabricating it would be dishonest, so option (b) (reserved treatment) is the correct fix.

**Fix (files):**
- `api/routers/forecasting/fva.py` — added a per-stage `missing_state` to `STAGE_DEFS`; champion's missing-fallback is now `"planned"` (reserved) instead of `"missing"`. `_build_stage()` takes the new arg and degrades a measured-but-empty "actual" stage to its `missing_state`. A measured champion row (if one ever appears in the external table) still surfaces as `state="actual"`.

**Verification (red → green):**
- Test: `tests/api/test_fva.py::test_fva_waterfall_champion_missing_renders_as_reserved_not_broken`
- RED: `AssertionError: assert 'missing' == 'planned'`
- GREEN: 17 passed in test_fva.py + test_fva_waterfall_ai.py.
- Guard test `test_fva_waterfall_champion_actual_when_measured` confirms a measured champion row still reads `actual` (78.3).
- Live: `curl /fva/waterfall?months=12` → stages now `[seasonal_naive=actual, external=actual, champion=planned, ai_adjusted=planned, planner_adjusted=planned]` (was champion=missing).

**Acceptance met:** Yes — the ladder presents three consistent reserved forward stages; champion no longer reads as a failure.

---

## U2.1 — Command Center "Open Exceptions" tile rendered an unseparated integer ("6141") [P1]

**What was wrong:** `CommandCenterTab.tsx` rendered `value={String(ex?.open_exceptions_total ?? 0)}` and badge ``${ex.critical_exceptions} critical`` — bare integers — while the feed footer on the same screen renders `formatInt(...)` → "6,141". Two formats for one number; the most prominent surface was the unformatted one. `formatInt` was already imported.

**Fix (files):**
- `frontend/src/tabs/CommandCenterTab.tsx` — wrapped the Open Exceptions tile value and critical badge in the existing `formatInt`. Null → "--" placeholder.

**Verification (red → green):**
- Test: `CommandCenterTab.test.tsx > comma-formats the Open Exceptions tile + critical badge matching the feed footer (U2.1)` (mocks `open_exceptions_total: 6141, critical_exceptions: 2464`).
- RED: `Unable to find an element with the text: 6,141`.
- GREEN: 17 passed. Existing "12" / "3 critical" test still green (formatInt of small ints has no separator).

**Acceptance met:** Yes — tile shows "6,141", badge "2,464 critical"; bare "6141"/"2464 critical" absent.

---

## U2.2 — Today's Plan priority badges (Urgent / High) rendered raw integers [P2]

**What was wrong:** `TodaysPlanBanner.tsx` `PriorityBadge` rendered `{value ?? count ?? 0}` with no formatting, showing "Urgent 2537 / High 1715" directly above the comma-formatted Action-Feed KPIs "Critical 2,537 / High Priority 1,715".

**Fix (files):**
- `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx` — `PriorityBadge` now formats an integer `count` via `toLocaleString()` (value strings pass through unchanged).

**Verification (red → green):**
- Test: `TodaysPlanBanner.test.tsx > comma-formats the Urgent/High priority counts matching the Action-Feed KPIs (U2.2)` (mock critical=2537, high=1715).
- RED: `Unable to find an element with the text: 2,537`.
- GREEN: 5 passed in TodaysPlanBanner.test.tsx.

**Acceptance met:** Yes — Urgent renders "2,537", High "1,715"; bare forms absent.

---

## F2.1 — Today's Plan banner "At Risk" chip was the unscoped bare-label one [P2]

**What was wrong:** The banner chip rendered bare label "At Risk" with `formatCompactCurrency(summary?.financial_at_risk)` ($12K-class, 7-day lost-margin basis), while Command Center shows "Order Value at Risk $246.8K" (a different, larger metric: proposed order value). Two ~20× different headline "at risk" dollars on the two triage landing surfaces, only one self-explaining. The cycle-1 basis sublabel landed on the Action Feed panel but not the banner ribbon.

**Fix (files):**
- `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx` — chip label changed "At Risk" → "$ at Risk"; `PriorityBadge` now accepts an optional `title` (tooltip), and the chip passes `summary.financial_at_risk_basis` ("7-day lost gross margin (open exceptions) + proposed order value") — the same basis text the Action Feed panel surfaces. The chip is now self-explaining on hover and scoped by label.

**Verification (red → green):**
- Test: `TodaysPlanBanner.test.tsx > names the at-risk basis on the banner chip so it is self-explaining (F2.1)`.
- RED: `expected null not to be null` (no `[title]` ancestor carrying the basis).
- GREEN: 5 passed.

**Acceptance met:** Yes — the banner chip names its basis (label + tooltip), matching the Action Feed panel; a planner can tell the window/basis without cross-referencing the Command Center tile.

---

## U2.3 — Customer-Analytics dropdown / Clear button / ranking header used bare bg-white / bg-gray-* (illegible in Dark theme) [P1]

**What was wrong:** The item-search autocomplete dropdown (`bg-white`), its option hover (`hover:bg-gray-100`), the Clear button (`bg-gray-100 hover:bg-gray-200`), and the Customer Ranking sticky header + sort buttons + bar track + row hover (`bg-white` / `bg-gray-50` / `bg-gray-100`) had no `dark:` variant → opaque light panels with white-on-white text risk in Dark theme.

**Fix (files):**
- `frontend/src/tabs/CustomerAnalyticsTab.tsx` — dropdown → `bg-popover text-popover-foreground`; option hover → `hover:bg-accent hover:text-accent-foreground`; Clear button → `bg-muted text-muted-foreground hover:bg-accent`.
- `frontend/src/tabs/customer-analytics/CustomerRanking.tsx` — sticky header `bg-white` → `bg-card`; row hover `hover:bg-gray-50` → `hover:bg-muted/50`; bar track `bg-gray-100` → `bg-muted`; inactive sort buttons `bg-gray-100 text-gray-600` → `bg-muted text-muted-foreground`.

**Verification (red → green):**
- Test: `CustomerAnalyticsTab.theme.test.ts` (source guard; same pattern as `CommandCenterTab.theme.test.ts`) asserts no bare `bg-white`/`bg-gray-NN` (negative-lookbehind for `dark:`).
- RED: `expected [ 'bg-white', 'bg-gray-100', ... ] to deeply equal []` (and CustomerRanking `[ 'bg-gray-50', 'bg-gray-100', ... ]`).
- GREEN: 3 passed.

**Acceptance met:** Yes — no bare `bg-white`/`bg-gray-*` surface class remains in either file; surfaces use theme tokens that adapt to Dark.

---

## U2.4 — KpiSummaryCards loading skeleton used hardcoded bg-gray-200 (invisible on dark) [P3]

**What was wrong:** `customer-analytics/KpiSummaryCards.tsx:160` shimmer placeholder was `animate-pulse bg-gray-200` with no `dark:` variant.

**Fix (files):**
- `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx` — skeleton `bg-gray-200` → `bg-muted` (theme token, visible in Light + Dark).

**Verification (red → green):**
- Test: `CustomerAnalyticsTab.theme.test.ts > KpiSummaryCards skeleton uses a theme token, not bg-gray-200`.
- RED: `expected [ 'bg-gray-200' ] to deeply equal []`.
- GREEN: 3 passed.

**Acceptance met:** Yes — skeleton uses `bg-muted`; grep for `bg-gray-200` in the file returns none.

---

## Test runs (this cycle)
- Backend: `pytest tests/api/test_fva.py tests/api/test_fva_waterfall_ai.py` → 17 passed. `pytest tests/api/ -k "fva or waterfall"` → 82 passed.
- Frontend (touched): `CommandCenterTab.test.tsx`, `TodaysPlanBanner.test.tsx`, `CustomerAnalyticsTab.theme.test.ts`, `customer-analytics/*` → 44 passed.
- Full frontend suite: 1122 passed, **2 pre-existing failures unrelated to this cycle** (`AppSidebar.test.tsx > renders all nav items`, `DemandReferencePanel.test.tsx > shows KPI cards` — neither file touched this cycle; KPI "1,200"/"14d" formatting + nav-item drift, present before my changes).
- `tsc --noEmit` clean on all touched files.

---

## Deferred

- **F2.3 / U1.6** (P3/P2 — DataQuality "166 vs 83" denominator + critical-badge-vs-status) — DEFERRED. Requires careful `DataQualityTab.tsx` rework to unify the header-tile denominator (`dashboard.domains[]` per-domain + per-referential-pair expansion) with the catalog (`/data-quality/checks`, 83 distinct). Touching the tile aggregation risks the severity-aware "Failed 0" semantics the planner now considers defensible. Too large for one safe TDD slice this cycle.
- **F2.4** (P3 — cold `/customer-analytics/affinity` ~11.6s) — DEFERRED. Perf/MV item; resolves correctly once Redis-warm (not a correctness defect). Needs a pre-aggregating MV or index, plus a `make` target + RUNBOOK cleanup entry — out of scope for a UI-trust cycle.
- **U2.5 / U1.7** (P2 — 7 tabs > 600 lines, CommandCenter 941) — DEFERRED. Mechanical subpanel splits, low correctness value, high diff churn; risk of destabilizing the many existing CommandCenter tests for no behavioral gain this cycle.
- **U2.6** (P2 — retired tab keys `?tab=aiPlanner`/`controlTower` show Command Center with stale URL) — DEFERRED. Router/IA change in `useUrlState.ts` + `App.tsx`; needs a routing test and a UX decision (URL rewrite vs breadcrumb). Reasonable next-cycle slice but not started.
- **U2.7** (P3 — Item Analysis breadcrumb shows bare "Item 15502") — DEFERRED. Straightforward but lower value than the dark-mode/format trust items prioritized this cycle.
- **U2.8** (P3 — sidebar vs page-heading naming) — DEFERRED. Cosmetic IA consistency, lowest value.
- **U1.3** (P2 — raw `fetch(` in model-tuning panels) — DEFERRED (carried). Pre-existing tsc errors in those files make a clean slice risky, as noted in cycle 1.

## Risk / notes
- F2.2 deliberately uses option (b) (reserved treatment) rather than computing champion accuracy: the champion forecast has **zero measurable overlap with actuals** in the window (verified against the live DB), so any computed number would be fabricated/empty. The fix is honest and reversible — a real measured champion row still promotes to `state="actual"`.
- All frontend U2.3/U2.4 fixes are className-only (theme tokens), no behavioral change; guarded by a source-scan test mirroring the existing `CommandCenterTab.theme.test.ts` convention.
- The 2 failing frontend tests are pre-existing and in files untouched this cycle; not introduced here.
- No commits made (per instructions). Changes live in the working tree; backend hot-reloaded (FVA live-verified), frontend HMR.
