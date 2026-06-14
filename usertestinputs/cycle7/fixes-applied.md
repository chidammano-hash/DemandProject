# Cycle 7 — Fixes Applied

Branch: `restructure`. Method: strict TDD (RED → GREEN → REFACTOR → live-verify) per item.
All changes left in the working tree (no commit).

Picked 3 highest-value items: the one NEW planner P2 (F7.1) and the two NEW usability
items (U7.1 P1, U7.2 P2). U7.3 / U7.4 (P3) deferred.

---

## F7.1 [P2, planner] — DQ "Domain Health" sub-100% scores with all-green breakdowns

**What was wrong:** `GET /data-quality/dashboard` computed `score = passed / total`
where `total` counted *all* statuses including `skip`. Skipped checks (source table
absent) were never surfaced in the payload, so a domain with 10 pass / 0 fail / 0 warn /
6 skip showed **62.5%** with a breakdown that read perfectly healthy. The summary KPI bar
likewise omitted skips, leaving 18 of 166 checks unexplained.

**Fix (files):**
- `api/routers/platform/data_quality.py` — `dq_dashboard()`: added a
  `count(*) FILTER (WHERE status = 'skip')` column; score denominator is now
  `passed + failed + warnings` (skips excluded → all-passing scored checks read 100%);
  emits a new `skipped` field per domain so `passed+failed+warnings+skipped == total`.
- `frontend/src/api/queries/platform.ts` — added `skipped: number` to `DQDomainScore`.
- `frontend/src/tabs/DataQualityTab.tsx` — domain card now renders `{d.skipped} skip`
  (when > 0); summary KPI bar gains a "Skipped" tile (grid 6→7 cols) so the count
  reconciles with Total.

**RED → GREEN evidence:**
- Backend: `tests/api/test_data_quality.py::test_dq_dashboard_excludes_skipped_from_score_denominator`
  (new) + updated `test_dq_dashboard_200`.
  - RED: `assert 166.7 == 100.0` (old code divided pass by the wrong column; no `skipped`
    field present). Also `test_dq_dashboard_200` RED `assert 800.0 == 80.0`.
  - GREEN: `17 passed in 0.58s`.
- Frontend: `src/tabs/__tests__/DataQualityTab.test.tsx` — new
  "F7.1: Domain Health card surfaces the skipped count and the summary bar adds a
  Skipped tile".
  - RED: `Unable to find an element with the text: 6 skip`.
  - GREEN: `32 passed`.

**Live-verify (curl):**
`GET /data-quality/dashboard` before → `forecast {score:75.0, passed:12, total:16}` (no
`skipped`). After → `forecast {score:100.0, passed:12, ..., skipped:4, total:16}` and
`item {score:100.0, passed:10, ..., skipped:6, total:16}` — reconciles 10+0+0+6=16.

**Acceptance met:** Yes (criterion 1 + 2b + 3 all satisfied).

---

## U7.1 [P1, usability] — Customer Concentration treemap still blank

**What was wrong:** `CustomerTreemap.tsx` set `visualMap.dimension: "fill_rate"`. ECharts
treats `visualMap.dimension` as a numeric index into a node's `value` array, but a treemap
node's `value` is a scalar — so the string `"fill_rate"` resolved every node out-of-range
and painted them with the (transparent) out-of-range color. The legend rendered
independently → the observed "legend present, rectangles blank" symptom. (The cycle-6 width
fix addressed a different, already-fixed cause.)

**Fix (files):**
- `frontend/src/tabs/customer-analytics/CustomerTreemap.tsx` — removed the entire
  `visualMap` block. Added a `fillRateColor()` red→amber→green ramp + `colorizeTree()` that
  recursively attaches an explicit `itemStyle.color` (mapped from each node's `fill_rate`)
  to every node, so the treemap draws independent of any visualMap. Added a static
  `<div>` legend (linear-gradient bar) below the chart to replace the removed ECharts
  legend. New `ColoredNode` type keeps the colorizer type-honest.

**RED → GREEN evidence:**
- `src/tabs/customer-analytics/__tests__/CustomerTreemap.test.tsx` — two new tests under
  "U7.1": (a) the rendered ECharts option has no `visualMap.dimension`; (b) each leaf node
  carries an `itemStyle.color` string.
  - RED: `expected "fill_rate" to be undefined` / `expect(child?.itemStyle?.color).toBeTruthy()`
    received `undefined`.
  - GREEN: `4 passed`.

**Live-verify:** `GET /customer-analytics/treemap` returns one root `FL value=12088589.5
fill_rate=97.9` with nested children each carrying a numeric `fill_rate`. With the fix each
node now gets `itemStyle.color` ≈ green for ~98% fill → the treemap draws nested rectangles
(FL → Off Premise Chains → PUBLIX WAREHOUSE…) colored by fill rate instead of blank.
(Render change; verified via the unit test asserting the emitted option + via payload shape.)

**Acceptance met:** Yes — emitted option has no `visualMap.dimension`; node colors present.

---

## U7.2 [P2, usability] — AI Planner FVA run list dead-ends on failed / 0-rec rows

**What was wrong:** Every "Recent Runs" row was clickable; selecting a `failed` run or a
`succeeded`-but-0-recommendation run loaded three "No data yet." panels — a dead end. The
API already returns `error_message` (and the TS type already had it), but it was never
rendered.

**Fix (files):**
- `frontend/src/tabs/AiPlannerFvaTab.tsx`:
  - `RunsListPanel`: `onSelect` now passes the full `RunMetadata`; failed rows render an
    inline `error_message` sub-row (truncated cell + `title` tooltip). Row wrapped in
    `<Fragment key>` to host the extra error row.
  - Main shell: tracks the selected `RunMetadata` object. When the selected run is `failed`
    → renders a "This run failed: <error_message>" card; when `succeeded` with
    `n_recommendations === 0` → renders a single "No recommendations were generated for this
    run" card; otherwise → the existing KPI/FVA detail panels. The blank KPI/table cards are
    suppressed for failed/0-rec runs.

**RED → GREEN evidence:**
- `src/tabs/__tests__/AiPlannerFvaTab.test.tsx` — three new tests under "U7.2": (1) failed
  row shows its error_message inline; (2) selecting a failed run shows "This run failed" +
  the error in the detail column and no "FVA by Month" panel; (3) selecting a succeeded/0-rec
  run shows "No recommendations were generated for this run".
  - RED: `Unable to find an element with the text: No recommendations were generated for this run`
    (and the failed-run assertions).
  - GREEN: `34 passed` (after relaxing one assertion to `getAllByText` since the error now
    correctly appears both inline and in the detail column).

**Live-verify (curl):** `GET /ai-planner/fva-backtest/runs?limit=50` returns 7 runs incl.
1 `failed` with a real `error_message` ("2 validation errors for Recommendation …") and 2
`succeeded` runs with `n_recommendations: 0`. With the fix the failed row shows the reason
inline; selecting it shows the explicit failure card; the 0-rec runs show the single
"no recommendations" card instead of three blank "No data yet." panels.

**Acceptance met:** Yes.

---

## Deferred

- **U7.3 [P3, consistency]** — FVA nav label ("AI FVA Backtest") vs page H1
  ("AI Planner — FVA Backtest") drift. Lower value than the 3 fixed items; touches the
  sidebar NAV_ITEMS + the AppSidebar drift-guard test (already failing pre-existing on a
  stale nav count). Deferred to avoid entangling with that pre-existing test failure.
- **U7.4 [P3, accessibility]** — Customer Map KPI strip MoM deltas unlabeled. Mirrors the
  cycle-6 U6.5 demand-history pattern; lower priority than P1/P2 work this cycle.
- **F4.3 (P2, carried 5th cycle)** — health/fill-rate live fallback in `control_tower.py`.
  Honest amber banner already mitigates; not a trust hazard.
- **F4.5 / U5.4 (P2, carried)** — Store Type taxonomy needs an upstream raw→canonical
  mapping table + searchable combobox; out of scope for a single TDD cycle.
- **F6.2 (P3, carried)** — dead `/customer-analytics/concentration` 404 route; no UI calls
  it (treemap is served by `/treemap`). Cleanup only.

## Risk / notes

- All three fixes are low-risk: F7.1 backend is a pure score/denominator change + additive
  field (no breaking removals); U7.1 / U7.2 are presentation-layer.
- Backend `tests/api/test_data_quality.py` 17 passed; `test_storyboard.py` 46 combined.
- Frontend touched files: 70 passed across the 3 test files; full frontend suite
  **1060 passed / 2 failed**. The 2 failures are the documented pre-existing ones
  (`AppSidebar.test.tsx` stale nav count, `DemandReferencePanel.test.tsx` recharts mock) —
  confirmed unrelated to this cycle's changes.
- `tsc --noEmit`: my new code is clean. The one remaining `CustomerTreemap.tsx` TS error
  (`ExportButtons getData` `TreemapNode[]` vs `Record<string, unknown>[]`) is pre-existing
  (documented in LEDGER cycle 6) and untouched.
- No commits made.
