# Cycle 4 — Fixes Applied

Four items fixed under strict TDD (RED → GREEN → live/typecheck verify). All frontend; no backend/DB changes needed this cycle. Changes left in the working tree (not committed).

---

## F4.1 — Clusters tab default empty-state misleads ("No cluster assignments") while 310K SKUs are clustered via Source

**What was wrong:** The Clusters Overview defaults the source toggle to "ML Pipeline" (`useState<"ml"|"source">("ml")`). `dim_sku.ml_cluster` is 100% NULL (the ML KMeans pipeline has never been run), so `GET /domains/sku/clusters?source=ml` returns `total_assigned=0` and the panel shows only "No cluster assignments yet. Run the clustering pipeline…" — implying the platform has no clustering. In reality `source=source` returns `total_assigned=310558` (26 buckets), and that data drives every per-cluster accuracy view in Aggregate Analysis.

**Fix (files):**
- `frontend/src/tabs/clusters/ClusterOverviewPanel.tsx` — when the active source is empty, probe the OTHER source (`useQuery` on `skuClusters(otherSource)`, enabled only when the current summary is empty). The empty state now renders a two-state hint naming the populated alternative: "The ML KMeans pipeline has no assignments yet. 310,558 SKUs are currently assigned via Source (sku.txt) — switch the Source dropdown above, or run the clustering pipeline…". Falls back to the bare message only when BOTH paths are empty.
- `frontend/src/tabs/clusters/__tests__/ClusterOverviewPanel.test.tsx` — new render test.

**RED → GREEN evidence:**
- Test: `ClusterOverviewPanel empty-ML / populated-source hint (F4.1) › names the populated Source alternative when ML is empty but source has assignments`.
- RED: hint paragraph containing both `310,558` and `Source (sku.txt)` not found (old copy had neither) — `TestingLibraryElementError: Unable to find element`.
- GREEN: 2 passed. Second test (both-empty → bare message, no `310,558`) also green.

**Live verify:** `curl /domains/sku/clusters?source=ml` → `{"total_assigned":0,"clusters":[]}`; `source=source` → `total_assigned=310558, n_clusters=26`. Default ML view now surfaces the populated-Source hint instead of the misleading bare message.

**Acceptance met:** YES.

---

## U4.1 — Dark-theme low-contrast regression in Demand-History panels (default Workbench view)

**What was wrong:** The cycle-3 U3.1 fix swapped Customer-Analytics off bare grays, but the same defect lived untouched in Demand-History. `WorkbenchPanel.tsx` (default-visible) + siblings had ~25 bare `text-gray-400/500` label/empty-state sites with no `dark:` companion — gray-on-near-black, below WCAG AA on the tab's primary working surface.

**Fix (files):**
- `frontend/src/tabs/demand-history/WorkbenchPanel.tsx`, `MatrixPanel.tsx`, `DecompositionPanel.tsx`, `ComparisonPanel.tsx` — bare `text-gray-[456]00` → `text-muted-foreground`, bare `bg-gray-[12]00` → `bg-muted` (theme tokens that adapt Light/Soft/Dark). Active blue/emerald accents untouched. One redundant `group-hover:text-muted-foreground` cleaned to `group-hover:text-foreground`.
- `frontend/src/tabs/demand-history/__tests__/darkThemeLegibility.test.ts` — new source-guard test (mirrors the CA U3.1 guard) over all 4 panels.

**RED → GREEN evidence:**
- Test: `Demand-History dark-mode legibility (U4.1) › <panel> uses theme tokens, not bare grays`.
- RED: 4 failed — `expect(source.match(BARE_GRAY_TEXT)).toEqual([])` got `["text-gray-400", …]`.
- GREEN: 4 passed; `grep text-gray-[456]00|bg-gray-[12]00` (sans `dark:`) over `tabs/demand-history/*.tsx` → 0.

**Live verify:** Rendering Demand-History Workbench/Matrix/Decomposition in Dark theme now shows legible muted labels/empty-states. tsc clean on touched files.

**Acceptance met:** YES.

---

## U4.2 — "Forecast vs Actual" chart tooltip shows raw unformatted integers (e.g. 2157763)

**What was wrong:** `ForecastTrendChart`'s ECharts `tooltip` had no `valueFormatter`, so hovering a point surfaced a 7-digit unseparated string (`2157763`) — inconsistent with the compact K/M Y-axis and the "2.2M" KPI tiles above the chart.

**Fix (files):**
- `frontend/src/components/ForecastTrendChart.tsx` — extracted the ECharts option into an exported pure `buildForecastTrendOption(...)` (testable without mounting ECharts) and added `tooltip.valueFormatter: (v) => formatInt(typeof v === "number" ? v : null)`, reusing the existing `formatInt` helper. Removed the now-duplicated `months/forecasts/actuals` in the component body.
- `frontend/src/components/__tests__/ForecastTrendChart.test.tsx` — new unit test on the tooltip formatter.

**RED → GREEN evidence:**
- Test: `ForecastTrendChart › tooltip has a valueFormatter that thousands-separates large integers (U4.2)`.
- RED: `buildForecastTrendOption` not exported — `ReferenceError`/import undefined.
- GREEN: 11 passed; `fmt(2157763) === "2,157,763"`, `fmt(null) === "—"`.

**Live verify:** Hovering a "Forecast vs Actual" point now renders `2,157,763`, matching the axis and KPI tiles. tsc clean.

**Acceptance met:** YES.

---

## U4.4 — AI-Planner-FVA run list marks succeeded-with-0-recs identically to productive runs

**What was wrong:** A run with `status="succeeded"` and `n_recommendations=0` rendered as an emerald "succeeded" next to a bare `0` — indistinguishable from "ran but did nothing useful". No affordance for the legitimate "ran cleanly, nothing actionable" outcome.

**Fix (files):**
- `frontend/src/tabs/AiPlannerFvaTab.tsx` — new exported pure `runYieldNote(status, nRecs)` returning a "No recommendations — ran cleanly, nothing actionable…" note only for `succeeded && recs === 0` (else `null`). Rendered as a muted inline sub-row in the run list, mirroring the existing failed-error inline row.
- `frontend/src/tabs/__tests__/AiPlannerFvaTab.test.tsx` — new helper tests.

**RED → GREEN evidence:**
- Test: `runYieldNote — zero-yield success affordance (U4.4)`.
- RED: `TypeError: runYieldNote is not a function`.
- GREEN: 41 passed (whole file). `runYieldNote("succeeded",0)` matches `/no recommendations/i`; returns `null` for productive/failed/running.

**Live verify:** A succeeded 0-rec run now shows a distinct muted "No recommendations" sub-note under the row instead of an unexplained green `0`.

**Acceptance met:** YES.

---

## Deferred (not actioned this cycle)

- **F2.3 / U1.6 (P2)** — DataQuality "166 Total" vs "Check Catalog (83)" denominator + CRITICAL-vs-pass badge. Carried; needs a tile-vs-catalog aggregation reconciliation (each check double-counted under source + referential-integrity domain). Out of scope for a one-slice TDD fix.
- **U4.3 (P2)** — CA below-fold panels ("Channel Mix" sunburst + 7 siblings) stuck on "Loading…". Requires confirming/auditing LazyPanel IntersectionObserver gating + bounded skeletons + retry across the heavy-CA-panel group, plus an E2E timing assertion. Larger than a safe single TDD slice this cycle; the above-fold map/treemap render real data (not stuck), so severity is bounded.
- **U3.6 / U2.5 / U1.7 (P2)** — 7 tab files > 600 LoC. Mechanical splits, low correctness value. Carried.
- **U1.3 (P2)** — raw `fetch()` in 4 model-tuning panels; blocked by pre-existing tsc errors in those files. Carried.

## Risk / notes

- `AiPlannerFvaTab.tsx` grew 610 → 636 lines (already over the 600-line CLAUDE.md limit — a pre-existing carried debt, U3.6). The U4.4 addition is minimal (a pure helper + a small inline row mirroring the existing failed-error row); the file split remains a separate deferred mechanical item, not regressed by this change.
- The Demand-History gray swap also touched `group-hover:text-gray-500` (a hover state, not strictly the dark-theme issue) → `group-hover:text-foreground`. Intentional: keeps the hover affordance legible and removes the bare gray. Active blue/emerald accents were deliberately left as-is.
- `ForecastTrendChart` still uses `EChartContainer` (retired per memory for general charts, but this component pre-dates that and was not in scope to migrate). The refactor here only extracts the option builder; engine migration is out of scope.
- No backend, SQL, or config changes — no doc-sync (ARCHITECTURE/specs) required per CLAUDE.md "what does not require doc updates" (UI copy + presentation fixes, no interface change).
