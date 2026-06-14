# Usability Review — Cycle 7

Branch `restructure`. Live app http://localhost:5173 → API :8000. Method: read cycle7 `capture-digest.md` + `capture-dump.json`, viewed cycle7 screenshots, then read-only code inspection. NEW items first. Prior-cycle deferrals (Store-Type IA, CommandCenterTab >600 lines, Item-Analysis date pickers, health/fill-rate live fallback) are re-confirmed at the bottom and are NOT counted as NEW.

Note on the digest: `?tab=aiPlanner`, `?tab=controlTower`, `?tab=storyboard` all render Command Center content — that is **by design** (`TAB_REDIRECTS` in `useUrlState.ts`, cycle-5 U5.1). Not a defect.

---

## NEW

## U7.1 — Customer Concentration treemap is STILL blank in cycle 7 despite the cycle-6 width fix; the real cause is an invalid `visualMap.dimension` on a scalar-`value` treemap [P1] [usability]
- **Category:** usability
- **Evidence:** `customerAnalytics.png` (cycle7) — the "Customer Concentration" card shows only the 0%–100% fill-rate color legend; the treemap rectangle area is empty. This persists even though the cycle-6 U6.3 fix (`style={{height:360,width:"100%"}}` + `className="w-full"`) is present in the working tree (`CustomerTreemap.tsx:84-85`). The endpoint is healthy: `GET /customer-analytics/treemap` returns one root `FL value=12088589.5 fill_rate=97.9` with nested `Off Premise Chains → PUBLIX WAREHOUSE…`, and **every node carries a numeric `fill_rate`**. So width is no longer the blocker.
- **Root cause:** `CustomerTreemap.tsx:34-46` sets `visualMap: { dimension: "fill_rate", min:0, max:100, … }`. In ECharts a treemap node's `value` is a **scalar number**, and `visualMap.dimension` is an **index into the node's `value` array** — `"fill_rate"` is not a valid dimension index, so the visualMap resolves every node to out-of-range and paints them with the (transparent) out-of-range color. The legend still renders (it's independent), which is exactly the observed "legend present, rectangles blank" symptom. The working `CustomerHeatmap.tsx:111-122` visualMap has **no `dimension`** and maps on the cell's scalar value — which is why the heatmap draws and the treemap doesn't.
- **File:** `frontend/src/tabs/customer-analytics/CustomerTreemap.tsx:34-64`
- **Recommendation:** Stop using `visualMap.dimension` for the treemap. Two clean options: (a) drop the `visualMap` entirely and set each node's `itemStyle.color` server-side or in the `useMemo` by mapping `fill_rate` through the same `#ef4444→#eab308→#22c55e` ramp (keep a static legend as a sibling `<div>`); or (b) shape each node's `value` as `[demand, fill_rate]` and set `visualMap.dimension: 1`, with `series.data[].value` being the 2-tuple. Option (a) is lower-risk and removes the fragile dimension binding. Keep the existing empty-state guard.
- **Acceptance:** With the live FL-dominant payload the treemap draws nested rectangles (FL → Off Premise Chains → PUBLIX WAREHOUSE…) colored by fill rate; a unit/visual test asserts the rendered ECharts option has no `visualMap.dimension` string (or that node colors are present), and resizing keeps it filled, not blank.

## U7.2 — AI Planner FVA "Recent Runs" makes every row clickable — including `failed` runs and `succeeded`-but-0-recommendation runs — leading to dead-end detail panels with no reason surfaced, even though the API already returns `error_message` [P2] [usability]
- **Category:** usability
- **Evidence:** `aiPlannerFva.png` + digest 2723-2730 — the run list shows `failed 2025-12-01 ollama — —` and three `succeeded … 50 DFUS 0 RECS` rows. Every row is `className="cursor-pointer" onClick={onSelect}` (`AiPlannerFvaTab.tsx:262-266`). Clicking the failed row (or a 0-rec succeeded row) selects it and the right column loads `SummaryKpis`/`ByRecommendationPanel`/`ByMonthPanel`, all of which return "No data yet." / empty tables — a dead end with no explanation. The backend `RunSummary` already selects and returns `error_message` (`api/routers/forecasting/ai_fva_backtest.py:171`) and the frontend `RunMetadata` type already has `error_message: string | null` (`ai-planner-fva-backtest.ts:37`) — it is simply never rendered.
- **Impact:** A planner who clicks the red `failed` run learns nothing about why it failed; the 0-rec succeeded runs look identical in affordance to runs with 143 recs but yield blank panels. The most actionable datum (the failure reason) is fetched and discarded.
- **File:** `frontend/src/tabs/AiPlannerFvaTab.tsx:261-288` (RunList row) + the detail panels at `:301-401`
- **Recommendation:** (1) On a `failed` row, show the `error_message` inline (truncated cell + `title` tooltip, or an expandable detail) and render an explicit error state in the right column ("This run failed: …") instead of empty KPI/table panels. (2) For `succeeded` runs with `n_recommendations === 0`, render a "No recommendations were generated for this run" empty state in the detail column rather than three separate "No data yet." cards. (3) Optionally de-emphasize (not disable) failed rows so the click expectation is clear.
- **Acceptance:** Selecting the `failed 2025-12-01` run shows its `error_message`; selecting a `succeeded`/0-rec run shows one clear "no recommendations" message; a unit test asserts the failed-run branch renders the error text and the 0-rec branch renders the no-recs empty state.

## U7.3 — AI Planner FVA sidebar label ("AI FVA Backtest") does not match the page heading ("AI Planner — FVA Backtest"); the nav term for this tab is inconsistent with its own title [P3] [consistency]
- **Category:** consistency
- **Evidence:** `aiPlannerFva.png` — left nav item reads **"AI FVA Backtest"** while the page H1 reads **"AI Planner — FVA Backtest"**. The sibling "FVA & ROI" tab (heading "Forecast Value Added") is also adjacent, so a planner sees three FVA-flavored labels ("FVA & ROI", "AI FVA Backtest", "Forecast Value Added") with no consistent naming spine.
- **Impact:** Minor navigational friction — the user cannot map the nav label to the page they land on at a glance, and the two FVA tabs are easy to confuse.
- **File:** sidebar nav definition (e.g. `frontend/src/components/AppSidebar.tsx` NAV_ITEMS) + `frontend/src/tabs/AiPlannerFvaTab.tsx` H1.
- **Recommendation:** Pick one term per tab and reuse it verbatim in both the nav and the heading, e.g. nav "AI Planner FVA" ↔ heading "AI Planner — FVA Backtest", and ensure "FVA & ROI" vs "AI Planner FVA" are visually distinct in the nav grouping. Keep the existing drift-guard test pattern (NAV_ITEMS ↔ VALID_TABS) and extend it to assert label↔heading parity if cheap.
- **Acceptance:** The nav label and the page H1 for this tab share a consistent FVA term; a planner can predict which page each FVA nav item opens.

## U7.4 — Customer Map KPI strip deltas are labeled only "MoM" with no tooltip/aria and no partial-period guard; a "↑ 42.9% MoM" on Lost Sales reads alarming without context [P3] [accessibility]
- **Category:** accessibility
- **Evidence:** `customerAnalytics.png` + digest 3275-3292 — "Total Demand 23.0M cases ↑ 28.1% MoM", "Lost Sales (OOS) 461.0K cases ↑ 42.9% MoM". `DeltaBadge` renders `{arrow} {Math.abs(delta).toFixed(1)}% MoM` (`KpiSummaryCards.tsx:104-108`) with no `title`/`aria-label`. Coloring is correct (demand-up green, OOS-up red via `goodDirection`), but a screen reader announces a bare "42.9% MoM" and a sighted planner gets no hover explaining the comparison window or whether the latest month is complete.
- **Impact:** Large month-over-month swings (28–43%) on the top-of-page KPIs read as dramatic shifts with no context; assistive-tech users get an unlabeled number. This mirrors the demand-history MoM concern (cycle-6 U6.5) but on the Customer Map strip.
- **File:** `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:104-131`
- **Recommendation:** Add a `title`/`aria-label` to the delta span ("Month-over-month change vs prior full month: +28.1%"). If the latest month can be partial, footnote or suppress the delta for an incomplete period. Reuse the cycle-6 `WorkbenchPanel` MoM-aria pattern for consistency across surfaces.
- **Acceptance:** Hovering a KPI delta shows a tooltip explaining the MoM comparison; a screen reader announces a labeled value; partial-month deltas are footnoted or hidden.

---

## RE-CONFIRMED prior-cycle deferrals (NOT new; re-pinned with cycle-7 evidence)

- **U5.4 (P2) — Store Type filter is a flat ~300-option native `<select>` with no typeahead/grouping.** Still live: digest 3381-3658 lists the full uncanonicalized taxonomy (`**OBSOLETE **`, `ALL`, `AIRLINE`/`AIRLINES`/`AIRLINE/SHIP/EX`, `CHAIN GROCERY` vs `Chain Grocery Store`, … `Z NIGHT CLUB`). Needs a searchable combobox and/or a canonical `store_type_group` MV column. (Channel dropdown digest 3358-3380 shows the same casing-dupe issue: `On premise` vs `On Premise Accounts` vs `ON PREMISE CHAINS`.)
- **U5.5 (P2) — `CommandCenterTab.tsx` is 844 lines (>600 rule)** and is the single surface behind 4 redirected routes. Pure refactor; split into sub-panels. (Also still >600: `ForecastPanel` 1215, `UnifiedChartPanel` 1120, `EnhancedComparisonPanel` 999, `WorkbenchPanel` 842, several inv-planning/lgbm panels.)
- **U5.6 (P2) — Item Analysis FROM/TO are two ~36-row raw ISO `<option>` lists with no TO ≥ FROM validation** (digest 3029-3102).
- **F4.3 (P2) — Command Center "Portfolio Health 0/100" + "Fill Rate (3m) --"** still rely on the amber stale banner (digest 7-23); live health/fill-rate fallback in `control_tower.py` still deferred. Honest banner mitigates.

---

## Summary

The product remains mature and the cycle-6 fixes (KPI delta good-direction coloring, Explorer null-sentinel, demand-history disambiguation/aria) hold. The strongest NEW finding is **U7.1** — the Customer Concentration treemap is **still blank** because the cycle-6 width fix addressed only one of two causes; the live blocker is an invalid `visualMap.dimension: "fill_rate"` on a scalar-`value` treemap, which the working heatmap avoids. **U7.2** is a clean usability win: the AI Planner FVA run list dead-ends on `failed`/0-rec rows even though the API already returns `error_message` the UI never shows. **U7.3** (FVA nav/heading label drift) and **U7.4** (unlabeled MoM deltas on the Customer Map strip) are lower-severity polish. Remaining Store-Type IA, oversized-tab, Item-Analysis date-picker, and health-fallback items are re-pinned from prior cycles.

NEW actionable (P0/P1/P2): **2** — U7.1 (P1), U7.2 (P2). U7.3 and U7.4 are P3.
