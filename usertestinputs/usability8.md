# Usability Review — Cycle 8

Branch `restructure`. Live app http://localhost:5173 → API :8000. Method: read cycle8 `capture-digest.md` + `capture-dump.json`, viewed cycle8 screenshots (`invPlanning.png`, `customerAnalytics.png`, `dataQuality.png`), then read-only code inspection + live endpoint/DB probes. NEW items first; prior-cycle deferrals re-confirmed at the bottom and NOT counted as NEW.

Confirmed RESOLVED this cycle (do not re-report):
- **U7.1 — Customer Concentration treemap** now draws nested colored rectangles (`customerAnalytics.png` shows FL/Off-Premise-Chains rectangles with the fill-rate ramp). The `visualMap.dimension` fix held.
- **CA ECharts 0-width collapse** is now globally prevented: `mergeEchartsProps()` (`echarts-modular.tsx:75`) defaults `style.width:"100%"`, so the panels that still pass only `style={{height:N}}` (ChannelSunburst, CustomerHeatmap, CustomerItemAffinity, etc.) render correctly. Not a defect.
- The 8× "Loading…" in the Customer Map digest (lines 3703-3711) are below-fold `LazyPanel` tiles captured before IntersectionObserver mounted them; `/customer-analytics/channel-mix` and `/segment-trends` both return 200 in <10ms. Snapshot artifact, not a defect.

---

## NEW

## U8.1 — "Today's Plan" banner shows the SAME financial-at-risk figure as "$4K" while the Action Feed directly below shows "$3.6K"; identical metric, two different roundings on one screen [P2] [consistency]
- **Category:** consistency
- **Evidence:** `invPlanning.png` — the blue "At Risk" tile in the Today's Plan banner reads **$4K**; the "Financial Impact at Risk" KPI in the Unified Action Feed immediately below reads **$3.6K**. Both are fed by the same `/inv-planning/action-feed` `summary.financial_at_risk = 3598.89` (verified live). The banner computes `$${(summary.financial_at_risk/1000).toFixed(0)}K` → `3598.89/1000 = 3.59 → toFixed(0) = "4"` → **$4K** (`TodaysPlanBanner.tsx:99-101`), a 11% over-statement vs the feed's $3.6K.
- **Impact:** A planner reads the top-of-page "$4K at risk", then sees "$3.6K at risk" one card lower and cannot tell which is authoritative — it looks like two different exposures. Rounding $3,599 *up* to $4K also overstates the headline number.
- **File:** `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx:96-104`
- **Recommendation:** Use one decimal for sub-$10K values so the banner matches the feed: `$${(v/1000).toFixed(1)}K` → "$3.6K"; or share a single `formatCompactCurrency()` helper between the banner tile and the Action Feed KPI so they round identically at every magnitude.
- **Acceptance:** With `financial_at_risk = 3598.89`, the banner tile and the Action Feed KPI render the identical string ("$3.6K"); a unit test asserts the banner formatter matches the feed formatter for a sub-$10K value.

## U8.2 — "Today's Plan" stats row prints "0 SKUs" next to "3,152 at risk" — a self-contradictory line driven by unpopulated daily-briefing fields rendered as real zeros [P2] [usability]
- **Category:** usability
- **Evidence:** `invPlanning.png` summary line under the priority ribbon: **"0 SKUs · 3,152 at risk · 0 excess ($0K)"**. Live `/inv-planning/daily-briefing` `stats` returns `total_skus: 0`, `below_ss_count: 3152`, `excess_count: 0`, `total_excess_value: 0.0`, `avg_health_score: null`. So three of the four stats (`total_skus`, `excess_count`, `total_excess_value`) and the health score are unpopulated, but the row prints them as literal `0`/`$0K` and silently drops Health (the `avg_health_score != null` guard). The result is the nonsensical "0 SKUs … 3,152 at risk": a portfolio cannot have 0 total SKUs yet 3,152 at-risk SKUs.
- **Impact:** The first line a planner reads on the daily plan is internally inconsistent and erodes trust in the whole banner — the same "stale zeros read as real data" failure mode the Command Center guards against with its amber "data unavailable" banner (digest 7-9), but here there is no guard at all.
- **File:** `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx:136-152`
- **Recommendation:** Treat `0`/`null` briefing fields as "no data" rather than real values: render "—" (or omit the chip) when `total_skus === 0`/`excess_count === 0` while another stat is non-zero, mirroring the existing `avg_health_score != null` guard already applied on the same row. Better: have `/inv-planning/daily-briefing` populate `total_skus` from the live SKU count so the line reconciles (3,152 at risk should be a subset of total).
- **Acceptance:** With the live payload (`total_skus:0, below_ss_count:3152`), the row no longer shows "0 SKUs" alongside a non-zero at-risk count — either the SKU total is populated or the empty chips degrade to "—"; a unit test asserts a zero-`total_skus` stat does not render "0 SKUs".

## U8.3 — Data Quality scores `sku_to_item` and `sku_to_location` domains at a screaming-red 0% when their ONLY failing check is INFO severity; the health score ignores severity entirely [P2] [consistency]
- **Category:** consistency
- **Evidence:** `dataQuality.png` + digest 2851-2860 — the Domain Health grid shows **`Sku_to_item 0%` (2 fail)** and **`Sku_to_location 0%` (2 fail)** in alarm red, identical visual weight to genuinely-broken domains. But the catalog rows for those domains are `referential_integrity_sku_to_item` **severity `info`** and `referential_integrity_sku_to_location` **severity `info`** (verified in `fact_dq_check_results`). The scorer `score = round(100*passed/(passed+failed+warnings),1)` (`data_quality.py:45-46`) counts every `fail` row equally regardless of severity, so a single *informational* referential-integrity notice craters the domain to 0%. The same row in the Check Catalog (digest 2910-2912) correctly renders these as muted **INFO** badges — so the per-check view and the domain score disagree on how serious they are.
- **Impact:** A planner triaging DQ sees five 0% red domains and assumes the data is broken, when two of them carry only informational notices. Severity is the whole point of a DQ catalog; the domain score discards it, over-alarming and burying the genuinely-failing domains (`forecast_to_sku`, `inventory_to_item`, `sourcing_to_location` are `warning`-severity fails).
- **File:** `api/routers/platform/data_quality.py:36-55` (score formula) + `frontend/src/tabs/DataQualityTab.tsx:220-249` (card coloring via `scoreBadgeClass`/`scoreRingColor`).
- **Recommendation:** Weight the score by severity — e.g. exclude `info`-severity fails from the score denominator (treat like skips, surface them as an "info" count), or weight `critical` > `warning` > `info` so an info-only domain reads 100% (or a muted "info" state) rather than 0% red. Keep the raw fail count visible. Add `severity` to the dashboard GROUP BY so the API can return per-severity fail counts. Mirror the cycle-6 F6.1 pattern (severity pill muted when the underlying status is benign).
- **Acceptance:** `sku_to_item` / `sku_to_location` no longer render as 0% alarm-red when their only fail is `info` severity; the dashboard response carries per-severity fail counts; a backend test asserts a domain with one `info`-fail and no warning/critical fails scores ≥ a domain with a `critical`-fail, and a frontend test asserts the info-only domain card is not painted in the critical-red bucket.

## U8.4 — Explorer "Item" domain leads with two visually identical columns "Item Ck" and "Item Id" (both `1`, `100004`, `10001`…), wasting the most-scanned left edge of the table on a redundant surrogate key [P3] [information-architecture]
- **Category:** information-architecture
- **Evidence:** digest 3199-3201 column order `Item Ck | Item Id | Item Desc …` and rows 3251-3252 — `1  1  FLOR DE CANA…`, `100004  100004  WOLF BLASS…`, `10001  10001  MAKERS MARK…`. Across the visible page `item_ck` and `item_id` are identical for every row, so the planner's eye hits two duplicate-looking numeric columns before reaching the human-readable description. `item_ck` is the internal surrogate/composite key, not a planner-facing field.
- **Impact:** The leftmost (most-scanned) columns of the flagship raw-data browser are a redundant key pair; the meaningful "Item Desc" is pushed two columns right. Adds noise on every Explorer Item view.
- **File:** Explorer field ordering / default visible columns (`frontend/src/tabs/ExplorerTab.tsx` + its field-list subpanel; the `/domains/item` field metadata controls order).
- **Recommendation:** Either hide `item_ck` by default (keep it toggle-able in the Fields panel) or move surrogate `_ck` keys to the far right for all domains, so business keys (`item_id`) and descriptions lead. A generic "demote `*_ck` surrogate columns" rule in the column-ordering logic fixes this across every domain at once.
- **Acceptance:** The default Explorer Item view leads with `Item Id` then `Item Desc`; `Item Ck` is hidden-by-default or right-most; a test asserts surrogate `_ck` columns are not in the first two default-visible positions.

## U8.5 — Portfolio Accuracy Heatmap BEER row shows four `<0%*` cells with no inline reason, while the legend/caption explaining `<0%*` sits far below the grid — the alarming cells are seen before the explanation [P3] [usability]
- **Category:** usability
- **Evidence:** digest 711-744 — the heatmap BEER row is `<0%* <0%* <0%* <0%*` across Nov 25–Feb 26, every other row (FOOD/WINE/SPIRITS…) is a normal 60-80%. The caption "Accuracy = 100 − WAPE. Cells marked <0%* have actuals near zero on a tiny base…" (digest 744) renders *after* the whole grid + a second `<0%/100%` legend. A planner scanning the matrix sees an entire category flagged `<0%*` red before reaching the footnote that says it's a tiny-base artifact, not a real -100% accuracy.
- **Impact:** BEER reads as catastrophically broken at a glance. The cycle-3 `<0%*` flooring + caption (F3.2) was the right call, but the explanation is too far from the cells; a per-cell tooltip would close the gap. Low severity since the caption exists.
- **File:** `frontend/src/tabs/aggregate-analysis/*Heatmap*` (heatmap cell renderer + the `formatHeatmapAccuracy` consumer).
- **Recommendation:** Add a `title`/tooltip on every `<0%*` cell ("Actuals near zero on a tiny base — read WAPE, not this %"), and/or render the one-line caption directly above the heatmap rather than below it. Reuse the existing `formatHeatmapAccuracy` sentinel detection to decide which cells get the tooltip.
- **Acceptance:** Hovering a `<0%*` heatmap cell surfaces the tiny-base explanation inline; the caption is visible without scrolling past the grid; a test asserts `<0%*` cells carry a `title`.

## U8.6 — S&OP tab is permanently inert for a planner: "No active S&OP cycles. Create one via the API or CLI." with no in-app create affordance — the entire tab is a dead end [P3] [usability]
- **Category:** usability
- **Evidence:** digest 2009-2026 — the S&OP tab renders its full 6-stage explainer, then "0 active cycles", "No active S&OP cycles. **Create one via the API or CLI.**", empty Approved Plan, empty Decision Log. There is no button to create a cycle; a planner who reads the rich process description has literally no way to start one from the UI. (This is the 4×-deferred U3.4/U4.4 "New Cycle" item, re-surfacing in cycle 8 with the tab still 100% empty.)
- **Impact:** A first-time planner lands on a well-documented workflow tab that cannot be used at all without dropping to a CLI — the worst kind of dead end because the copy promises a workflow the UI doesn't deliver.
- **File:** `frontend/src/tabs/SopTab.tsx` (empty-state) + a new guarded `POST /sop/cycles` backend route (`api/routers/operations/`).
- **Recommendation:** Add a "Create S&OP Cycle" button in the empty-state that POSTs to a new guarded `POST /sop/cycles` (`dependencies=[Depends(require_api_key)]`, `get_conn()`, `%s`) seeding a Demand-Review-stage cycle for the current period; the existing `advanceSopCycle`/`approveSopCycle` fetchers (cycle-3 U3.1) then drive it forward. If backend is out of scope this cycle, at minimum the empty-state should say *which* CLI command creates a cycle, not the vague "via the API or CLI".
- **Acceptance:** The S&OP empty-state offers an in-app way to create a cycle (button → guarded POST) OR names the exact command; a planner can reach a non-empty S&OP view without leaving the app.

---

## RE-CONFIRMED prior-cycle deferrals (NOT new; re-pinned with cycle-8 evidence)

- **U5.4 (P2) — Store Type filter is a flat ~270-option native `<select>` with no typeahead/grouping/canonicalization.** Still live: digest 3391-3665 lists the full uncanonicalized taxonomy (`**OBSOLETE **`, `ALL`, `AIRLINE`/`AIRLINES`/`AIRLINE/SHIP/EX`, `CHAIN GROCERY` vs `Chain Grocery Store`, … `Z NIGHT CLUB`). The **Channel** dropdown (digest 3367-3388) shows the same casing-dupe family: `Off premise` / `Off Premise Chains` / `On premise` / `On Premise Accounts` / `ON PREMISE CHAINS`. Needs a searchable combobox and/or a canonical `store_type_group` MV column.
- **U5.6 (P2) — Item Analysis FROM/TO are two ~36-row raw ISO `<option>` lists (digest 3037-3110) with no TO ≥ FROM validation.**
- **U5.5 (P2) — `CommandCenterTab.tsx` >600 lines** and the single surface behind 4 redirected routes (commandCenter/controlTower/aiPlanner/storyboard all render identical Command Center content — digest confirms aiPlanner/controlTower == commandCenter). Pure refactor; split into sub-panels. (Also still >600: ForecastPanel, UnifiedChartPanel, EnhancedComparisonPanel, WorkbenchPanel.)
- **F4.3 (P2) — Command Center "Portfolio Health 0/100" + "Fill Rate (3m) --"** still rely on the amber stale banner (digest 7-23); live health/fill-rate fallback in `control_tower.py` still deferred. Honest banner mitigates.
- **U7.4 (P3) — Customer Map KPI strip MoM deltas** ("↑ 28.1% MoM", "↑ 42.9% MoM" — digest 3285-3300) remain unlabeled (no `title`/`aria` on the delta span, no partial-period guard). Re-pinned from cycle 7.

---

## Summary

The product remains mature; cycle-7's treemap and AI-Planner-FVA fixes hold and the CA 0-width collapse is now globally prevented. The strongest NEW findings cluster on **trust in headline numbers**: **U8.1** (the *same* at-risk dollar figure shown as "$4K" in the banner and "$3.6K" in the feed, one screen apart) and **U8.2** (the daily-plan line "0 SKUs · 3,152 at risk", a self-contradiction from unpopulated briefing fields rendered as real zeros). **U8.3** is a clean consistency fix — the DQ domain score ignores severity, so an `info`-only check craters two domains to alarm-red 0% while the Check Catalog correctly badges them INFO. **U8.4** (Explorer redundant `Item Ck`/`Item Id` lead columns), **U8.5** (heatmap `<0%*` explanation too far from the cells), and **U8.6** (S&OP tab is a CLI-only dead end) are lower-severity polish. Store-Type/Channel IA, Item-Analysis date pickers, CommandCenterTab split, health-fallback, and Customer-Map MoM aria are re-pinned from prior cycles.

NEW actionable (P0/P1/P2): **3** — U8.1 (P2), U8.2 (P2), U8.3 (P2). U8.4/U8.5/U8.6 are P3.
