# Cycle 3 — Fixes Applied

Branch: `restructure`. Strict TDD (red → green → refactor → live-verify) per item.
Reviewer inputs: `testinput3.md` (planner), `usability3.md` (UX). All changes left in the
working tree (not committed).

---

## F3.1 [P1] — Command Center KPIs all-zero (MV stale) with no live fallback to exceptions

**What was wrong:** `GET /control-tower/kpis` read only `mv_control_tower_kpis`. When that MV
is stale/missing it degraded to an all-zero payload, so the default landing tab showed
`Open Exceptions 0 / Critical 0` even though 6,142 open exceptions (2,465 critical) live in
`fact_replenishment_exceptions` (the same table the Action Feed reads successfully). Morning
triage from the Command Center was un-actionable.

**Fix (files):**
- `api/routers/operations/control_tower.py` — added `_exceptions_fallback()` helper that, on
  `ObjectNotInPrerequisiteState`/`UndefinedTable`, runs a live `COUNT(*) … GROUP BY severity`
  over `fact_replenishment_exceptions WHERE status='open'` (sum of `estimated_order_value`) and
  returns honest open/critical/high counts + `source:"fact_replenishment_exceptions"`. The
  fallback itself degrades to zeros if even the table is unavailable (never 500). The MV-stale
  branch now merges this into the payload while keeping the actionable `warning`.

**Red→green evidence:**
- Test `test_control_tower_kpis_falls_back_to_exceptions_table_when_mv_stale` (tests/api/test_control_tower.py).
- RED: `assert 0 == ((2465 + 1715) + 1962)` — endpoint returned `open_exceptions_total: 0`.
- GREEN: 15 passed in test_control_tower.py (incl. the new fallback test + a both-unavailable
  degrade test). Existing MV-missing/unpopulated tests still pass.

**Live-verify:** `curl /control-tower/kpis` exceptions block:
before → `{open_exceptions_total:0, critical:0, high:0, recommended_order_value:0}`
after  → `{open_exceptions_total:6142, critical_exceptions:2465, high_exceptions:1715, recommended_order_value:246723.23, source:"fact_replenishment_exceptions"}`,
reconciling exactly with the DB (`GROUP BY severity`: critical 2465 / high 1715 / low 1962).
`warning` still present so the cycle-2 amber stale banner persists.

**Acceptance met:** YES (acceptance branch (b)).

---

## U3.1 [P1] — Five+ tabs use raw `fetch(` instead of `fetchJson` (swallow/leak errors)

**What was wrong:** `ItemAnalysisTab.tsx` (4 raw fetches) swallowed analysis/forecast failures
with empty `catch {}` (blank chart, no diagnosis); `SopTab.tsx` (advance/approve POSTs) bypassed
the cycle-2 error-sanitization layer (`fetchJson` status-attach + FastAPI-detail parse).

**Fix (files):**
- `frontend/src/api/queries/evolution.ts` — added `advanceSopCycle()` / `approveSopCycle()`
  (fetchJson POSTs).
- `frontend/src/tabs/SopTab.tsx` — advance/approve now go through the new query fns; approve is a
  proper `useMutation` with an error display (was a fire-and-forget raw fetch).
- `frontend/src/tabs/ItemAnalysisTab.tsx` — replaced the 4 raw fetches with existing query fns
  (`fetchSamplePair`, `fetchDomainSuggest`, `fetchSkuAnalysis`); added a `skuError` state so a
  failed `/sku/analysis` shows a "Could not load analysis" panel instead of a silent blank chart;
  removed now-unused `SamplePairPayload`/`SuggestPayload` imports.

**Red→green evidence:**
- Test `src/tabs/__tests__/no-raw-fetch.test.ts` (source-guard: no bare `fetch(` in the two tabs).
- RED: failed listing the offending lines (ItemAnalysisTab 140/166/216/233; SopTab 116/288).
- GREEN: 2 files / 11 tests pass (guard + existing ItemAnalysisTab tests); SopTab 5 tests pass.

**Live-verify:** UI behavior — ItemAnalysis now surfaces an error panel on `/sku/analysis`
failure (no more blank chart); SOP advance/approve route through the sanitized error layer.

**Acceptance met:** YES (for the two named tabs). Model-tuning/clusters subpanels deferred (see below).

---

## U3.2 [P1] — Data Quality empty-state instructs a 404 `/dq/run` curl while a working button exists

**What was wrong:** The DQ empty-state "HOW TO POPULATE" card told planners to
`curl -X POST http://localhost:8000/dq/run` (confirmed **404**) and run
`scripts/dq_run_checks.py` (does not exist), while a wired "Run Checks Now" button
(`POST /data-quality/run`) sat directly above.

**Fix (files):**
- `frontend/src/tabs/DataQualityTab.tsx` — empty-state now offers a primary in-app action
  ("Run DQ checks now" → `runChecksMutation`) and a single correct CLI hint
  (`uv run python scripts/populate_dq_checks.py`, which exists). The `/dq/run` curl and the
  bogus `dq_run_checks.py` line are gone.

**Red→green evidence:**
- Tests (DataQualityTab.test.tsx): `U3.2: empty state does not instruct the stale /dq/run 404 path`
  + `U3.2: empty state offers an in-app run action wired to runDQChecks`.
- RED: first asserted text still contained `/dq/run`; CTA assertion failed (no in-app action button).
- GREEN: 31/31 DataQualityTab tests pass (disambiguated the empty-state CTA label so the existing
  header-button tests stay unambiguous).

**Live-verify:** UI behavior — empty-state CTA triggers `POST /data-quality/run` in-app; no
on-screen command points at a 404.

**Acceptance met:** YES.

---

## U3.3 [P1] — Customer Map "State" filter full of garbage codes (`.`, `00`, `0D`, `XX`, `null`)

**What was wrong:** The State `<select>` rendered `filterOptions.states` verbatim from
`mv_customer_filter_options` — 135 raw values including `.`, `0`, `00`, `0D`, numeric junk, `XX`,
`null`, intermixed with real codes — making the facet unscannable.

**Fix (files):**
- `frontend/src/api/queries/customer-analytics.ts` — added `normalizeStateOptions()` (whitelist
  of US state/territory + Canadian province codes; uppercased, de-duped, sorted) and applied it in
  `fetchCustomerAnalyticsFilterOptions()`. Minimum-safe client-side normalization until the MV
  (sql/173) is cleaned upstream. Whitelisting (not just `^[A-Z]{2}$`) also drops the `XX`
  placeholder.

**Red→green evidence:**
- Test `src/api/queries/__tests__/customer-analytics-states.test.ts`.
- RED: `normalizeStateOptions` did not exist (import failure). A later iteration caught that a bare
  `/^[A-Z]{2}$/` kept `XX` (`expected [...,'XX'] to deeply equal [...]`) — fixed by whitelisting.
- GREEN: 3/3 states tests + 24/24 CustomerAnalyticsTab tests pass.

**Live-verify:** `curl /customer-analytics/filter-options` → raw 135 codes; after normalization
60 clean codes kept (`AB, AK, AL, …`), junk (`.`, `0`, `00`, `0D`, numerics) dropped.

**Acceptance met:** YES (frontend-only; MV cleanup deferred to a coordinated DDL change).

---

## F3.2 / U3.6 [P2] — Accuracy heatmap renders unbounded negative accuracy (BEER −263.9%) with no clamp/legend

**What was wrong:** Accuracy = 100 − WAPE goes strongly negative on low-base/intermittent
categories (BEER, L2_6). The heatmap showed raw `-263.9%`, reading as a bug rather than a
tiny-denominator artifact, in the exact screen used for intervention ranking.

**Fix (files):**
- `frontend/src/tabs/aggregate-analysis/aggregateShared.ts` — added `formatHeatmapAccuracy()`:
  floors the *displayed* value at `<0%*` for negatives, otherwise `xx.x%`.
- `frontend/src/tabs/AggregateAnalysisTab.tsx` — the accuracy `HeatmapGrid` now uses that
  `valueFormat`, shows the legend (`<0%` → `100%`), and renders a one-line caption explaining
  "Accuracy = 100 − WAPE; cells marked <0%* have actuals near zero — review WAPE." The numeric
  value still drives the (already-saturated) color scale.

**Red→green evidence:**
- Test `src/tabs/aggregate-analysis/__tests__/formatHeatmapAccuracy.test.ts`.
- RED: `formatHeatmapAccuracy` did not exist (import failure).
- GREEN: 2/2 formatter tests + 10/10 AggregateAnalysisTab + 19/19 HeatmapGrid tests pass.

**Live-verify:** UI behavior — a `-263.9%` cell now renders `<0%*` with an explanatory caption and
legend; healthy cells unchanged (`73.9%`).

**Acceptance met:** YES (heatmap). The cluster-comparison table's negative accuracy was left
as-is this cycle (separate panel; see Deferred).

---

## Deferred

- **U3.1 (remainder)** — model-tuning / clusters subpanel raw fetches (ExperimentBuilder,
  LogViewer, EnhancedPromoteModal, EnhancedComparisonPanel, ClusterExperimentBuilder,
  FeatureLabPanel). Out of scope this cycle; the two highest-impact tabs (ItemAnalysis, SOP) are
  fixed. Follow-up sweep.
- **U3.4 [P2]** — S&OP "New Cycle" create flow. Requires a new `POST /sop/cycles` backend route +
  `createCycle()` + empty-state button. A genuine backend feature; deferred to keep this cycle's
  fixes tight and fully verified.
- **U3.5 [P2]** — Demand History unlabeled `%` column. Needs the metric's true definition
  (CV% vs YoY%) confirmed against the backend before labeling; deferred to avoid mislabeling.
- **U3.7 [P2]** — Customer Concentration treemap empty / `/customer-analytics/concentration` 404.
  Needs backend route/query investigation; deferred.
- **U3.8 [P3]** — oversized tab files (CommandCenterTab 844, etc.). Pure refactor; deferred.
- **F3.3 (CLI/script note)** — covered the stale-path half via U3.2; the script-existence
  verification of populate/fix scripts is done (`scripts/populate_dq_checks.py`,
  `scripts/fix_dq_issues.py` exist).
- **F3.4 [P3]** — FVA Champion "No data" is a genuinely-empty data state (no promoted backtest);
  no code defect.

## Risk / notes

- F3.1 fallback is read-only and only fires on the MV-stale/missing path; the populated-MV path is
  unchanged. The extra `source` key on the `exceptions` object is additive (frontend TS interface
  ignores unknown extras; existing fields unchanged).
- U3.3 is a client-side whitelist; the source MV is still dirty (the demand map's WHERE predicate
  is unchanged). A full fix requires cleaning `mv_customer_filter_options` + matching the map
  endpoint's WHERE — a coordinated DDL change deferred per the reviewer's own note.
- Pre-existing TS errors (DataQualityTab:128 header-button `mutate()`, ChampionPanel props,
  customer-analytics chart prop typings) and pre-existing ruff nits (Optional→`X | None`,
  dict-comprehension) in untouched regions were left alone — out of scope, vitest/pytest green.
- The working tree also contained unrelated uncommitted changes from prior cycles
  (CommandCenterTab, KpiSummaryCards, conftest, several test_*.py) that I did not author or modify.
