# Usability Review — Cycle 6

Branch `restructure`. Live app http://localhost:5173 → API :8000. Method: read cycle6 capture-digest + screenshots, then code inspection (read-only). NEW items first. Prior-cycle deferrals (U5.4 Store-Type IA, U5.2 blank treemap, U5.5 CommandCenterTab 844 lines, U5.6 Item-Analysis date pickers, U4.3 Demand-History MoM%) are re-confirmed and re-pinned at the bottom; they are NOT counted as NEW.

Note on the digest: `?tab=aiPlanner`, `?tab=controlTower`, and `?tab=storyboard` all render Command Center content. That is **by design** — `TAB_REDIRECTS` in `useUrlState.ts` consolidates those retired keys onto `commandCenter` (the cycle-5 U5.1 fix). Not a defect; not reported.

---

## NEW

## U6.1 — Portfolio KPI cards show a delta sign that contradicts its own color: WAPE/Bias display the *negated* delta, so "−1.9pp" is painted red [P1] [consistency]
- **Category:** consistency
- **Evidence:** `aggregateAnalysis.png` — "WAPE % 26.1% … Target: <= 10% … −1.9pp vs prev 3mo" rendered RED with a down-arrow; "Bias % 6.6% … −9.8pp vs prev 3mo" also red. A planner reads "WAPE −1.9pp" as *error fell* (good) but the cell is red (bad). The two signals disagree.
- **Root cause:** `AggregateAnalysisTab.tsx:445` passes `delta: -kpi.deltas.wape_pct` (and `:457` `delta: -Math.abs(kpi.deltas.bias_pct)`) so that `trendDirection()` produces the correct *color* for a "lower-is-better" metric — but `KpiCard` then prints that same negated number verbatim (`KpiCard.tsx:124` `{trend.delta > 0 ? "+" : ""}{trend.delta.toFixed(1)}{unit}`). Color is right; the displayed magnitude's sign is now a lie about the metric's actual movement. Bias always shows a negative because of `-Math.abs(...)`.
- **File:** `frontend/src/tabs/AggregateAnalysisTab.tsx:445,457` + `frontend/src/components/KpiCard.tsx:123-127`
- **Recommendation:** Decouple display sign from color. Extend `KpiCard.trend` with an explicit `goodDirection: "up" | "down"` (reuse the cycle-2 `deltaPresentation()`/`goodDirection` pattern already used on Command-Center KPIs). Pass the **raw** delta for display (`kpi.deltas.wape_pct`, `kpi.deltas.bias_pct`) and let the card color by `(direction === goodDirection)`. For Bias, color by movement toward zero (good = |delta| shrinking) but display the true signed change.
- **Acceptance:** When WAPE drops from 28.0% to 26.1% the card shows "−1.9pp" in GREEN (improvement); when WAPE rises it shows "+Xpp" in red. Bias card shows the real signed change, green when |bias| moves toward 0. Accuracy card unchanged (already correct). Unit test in `KpiCard.test.tsx` asserts sign-of-display ≠ coupled-to-color.

## U6.2 — Explorer data table renders the literal string `null` in cells (e.g. UPC) because the API returns `'null'` (string), not JSON null [P2] [consistency]
- **Category:** consistency
- **Evidence:** `explorer.png` + digest 3192 — Item rows show `… REGIONAL  100050  TREASURY WINE ESTATES AMERICAS  N  WOLF BLASS` with UPC = `null`. Confirmed live: `GET /domains/item?limit=5` returns `"upc": "null"` (a string) for items 100004/10001, while real UPCs are zero-padded digit strings. `formatCell()` (`formatters.ts:23`) only maps JS `null`/`undefined`/`""` to `"-"`, so the string `"null"` falls through to `String("null")` → `"null"`.
- **Impact:** Every Explorer domain inherits this. A planner browsing raw item/forecast/inventory data sees `null` as if it were a value; it's indistinguishable from a genuine textual `"null"` and clutters the densest data surface in the app.
- **File:** `frontend/src/lib/formatters.ts:22-26` (consumed by `frontend/src/tabs/explorer/ExplorerTable.tsx:133-135`)
- **Recommendation:** In `formatCell`, treat the case-insensitive sentinel strings `"null"`, `"none"`, `"na"`, `"undefined"` as empty → `"-"` (mirror the load-time null-normalization rule `'' / 'null' / 'none' / 'NA' → NULL` from CLAUDE.md). Durable fix: the API/MV should emit JSON null, but the frontend guard is the safe, low-risk surface and protects every domain at once. Keep the `title` tooltip (`ExplorerTable.tsx:133`) blank for these so hover doesn't reveal `"null"` either.
- **Acceptance:** Explorer Item table shows `-` (not `null`) for empty UPC; a unit test asserts `formatCell("null") === "-"` and `formatCell("NULL") === "-"` while `formatCell("Null Object Brand")` is preserved.

## U6.3 — Customer Concentration treemap is blank despite a valid 200 payload (height set, width unset → series area collapses) [P1] [usability]
- **Category:** usability
- **Evidence:** `customerAnalytics.png` — "Customer Concentration" card shows only the 0%–100% fill-rate color legend; the treemap rectangles area is empty. The endpoint is healthy: `GET /customer-analytics/treemap` returns `{tree:[{name:"FL", value:12088589.5, children:[{name:"Off Premise Chains", …PUBLIX WAREHOUSE…}]}]}`. The visualMap legend renders (so ECharts mounted) but the treemap series has zero drawable area.
- **Root cause:** `CustomerTreemap.tsx:85` renders `<ReactECharts option={option} style={{ height: 360 }} …/>` with **no width**, inside a `<div role="img">` that also has no width. ECharts measures 0px width → treemap lays out into nothing while the bottom-anchored visualMap still paints. (This is the same class as the cycle-5 U5.2 width default, which evidently doesn't reach this panel's inner wrapper.)
- **File:** `frontend/src/tabs/customer-analytics/CustomerTreemap.tsx:84-86`
- **Recommendation:** Give the chart an explicit responsive box: `style={{ height: 360, width: "100%" }}` and ensure the wrapping `div role="img"` is `className="w-full"`. Add a `<ReactECharts>` `opts={{ renderer: "svg" }}` or an `onChartReady`/resize observer if the card animates open. Verify the same pattern on the other CA echarts panels (ChannelSunburst, CustomerHeatmap, OosImpactBubble, DemandFlowSankey) which share the wrapper.
- **Acceptance:** With the live FL-dominant payload, the treemap draws nested rectangles (FL → Off Premise Chains → PUBLIX WAREHOUSE…) at ≥1 px width; resizing the window or collapsing the sidebar keeps it filled, not blank.

## U6.4 — Demand-History rail lists the same item description multiple times with no disambiguator; planner cannot tell duplicate DFUs apart [P2] [information-architecture]
- **Category:** information-architecture
- **Evidence:** `demandHistory.png` + digest 845-855 — "TITOS HANDMADE VODKA 80" appears as 1,361,016 **and** 793,610 **and** 256,525 **and** 152,313, with no location/cluster/id to distinguish them. Same for "BACARDI RUM SUPERIOR WHITE 80" (×3) and "CASAMIGOS TEQUILA BLANCO 80" (×2). In "Item" series mode the rows are different item_ids sharing a description; the user sees four identical labels with wildly different totals.
- **Impact:** Selecting "the Titos series" is a coin-flip — the planner can't know which DFU they clicked, and overlay-compare (cmd-click, max 3) silently mixes unrelated SKUs under one name.
- **File:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx` (TreeNode label, ~line 234-260; series label = `s.label || s.key`)
- **Recommendation:** When the chosen SERIES granularity is "Item", append a stable disambiguator to the visible label — the item_id (and pack/size if available), e.g. `TITOS HANDMADE VODKA 80 · #105430`. The `key` already carries identity; surface a short suffix in the rendered label (truncate the description, keep the id visible). Optionally collapse exact-description duplicates under a parent with the id as the leaf.
- **Acceptance:** No two rail rows share an identical visible label in Item mode; each Titos/Bacardi/Casamigos row shows a distinct id/size suffix; the tooltip/title gives the full id.

## U6.5 — Demand-History rail's trailing colored "%" is an unlabeled single-month MoM with no header, tooltip, or aria; 520.9% green up-arrow is misleading [P3] [usability]
- **Category:** usability
- **Evidence:** `demandHistory.png` (`63.4%`, `6.1%`, `77.5%`, `26.1%` after each sparkline) + digest 905/945/991 — `CAPT MORGAN … 130.4%`, `PINNACLE VODKA 80 PET … 520.9%`, `REAL SANGRIA RED … 144.7%`. The number is the last-month-over-prior-month change (`WorkbenchPanel.tsx:164` `((last-prev)/prev)*100`), shown with a green/red arrow but **no column header, no legend, no tooltip**. A 520.9% green up-arrow is a single-month spike on a low base, not a trend.
- **Impact:** Reads as "this SKU is up 520%" — alarming and uninterpretable. There's also no `aria-label` on the value (`:201-209`) so screen readers announce a bare "520.9%".
- **File:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx:142-213`
- **Recommendation:** Label it. Add a one-time rail header ("last value · MoM Δ") and an `aria-label`/`title` on the delta span ("Month-over-month change: +520.9%"). Consider replacing raw single-month MoM with a more stable signal (3-month CAGR or last-vs-trailing-mean) and suppress/footnote values where the prior month is near zero (the cause of 500%+ readings).
- **Acceptance:** The rail has a visible/aria label explaining the % is month-over-month; hovering the value shows a tooltip; near-zero-base spikes are footnoted or capped; screen reader announces a labeled value.

## U6.6 — Demand-History series rows are icon `<button>`s with no `aria-pressed` and no accessible name for the multi-select state [P3] [accessibility]
- **Category:** accessibility
- **Evidence:** `WorkbenchPanel.tsx:235` — each series is a `<button onClick=…>` that toggles selection (single / cmd-click overlay up to 3) but exposes no `aria-pressed`, and selection is conveyed only by background color (`bg-blue-50`/ring). The "overlay max 3" affordance (`:559`) is text-only; there's no announced count or selected state for assistive tech.
- **File:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx:235-260`
- **Recommendation:** Add `aria-pressed={isSelected}` to the row button and an `aria-label` combining the item description + id + selected state. Announce the live overlay count (`role="status"`) when it changes; disable/aria-disable additional rows once 3 are selected rather than silently ignoring the click.
- **Acceptance:** Each row button reports its pressed state to a screen reader; reaching the 3-overlay cap is announced, not silent; keyboard users can perceive which series are active without relying on color.

---

## RE-CONFIRMED prior-cycle deferrals (NOT new; re-pinned with cycle-6 evidence)

- **U5.4 (P2) — Store Type filter is a flat ~150-option native `<select>` with no typeahead/grouping.** Still live: digest 3329-3605 lists ~150+ surviving entries (`**OBSOLETE **`, `ALL`, `AIRLINE`/`AIRLINES`/`AIRLINE/SHIP/EX`, `CHAIN GROCERY` vs `Chain Grocery Store`, … `Z NIGHT CLUB`). `CustomerAnalyticsTab.tsx` Store-Type select. Needs a searchable combobox and/or a canonical `store_type_group` MV column.
- **U5.2 (P1) — see U6.3 above** for the still-blank treemap (this cycle reframes it with the no-width root cause; the documented cycle-5 width default does not reach this panel).
- **U5.5 (P2) — `CommandCenterTab.tsx` is 844 lines (>600 rule) and is the single surface behind 3 routes** (`commandCenter`/`controlTower`/`aiPlanner`/`storyboard` all redirect here). Pure refactor; split into sub-panels.
- **U5.6 (P2) — Item Analysis FROM/TO are two ~36-row raw ISO `<option>` lists with no TO ≥ FROM validation** (digest 2977-3050).
- **F4.3 (P2) — Command Center "Portfolio Health 0/100" + "Fill Rate (3m) --"** still rely on the amber stale banner; live health/fill-rate fallback in `control_tower.py` still deferred (digest 11-23). Honest banner mitigates.

---

## Summary

The product remains in good shape; cycle-5 fixes (popstate redirect, item×state MV, slice-table flooring) hold. Strongest NEW finding is **U6.1** — Portfolio KPI cards paint a delta whose displayed sign contradicts its own color (WAPE "−1.9pp" in red), a trust-eroding inconsistency on the primary accuracy surface, fixable by decoupling display-sign from good-direction coloring in `KpiCard`. **U6.3** (blank Concentration treemap despite a valid payload — height set, width unset) is a high-visibility usability defect. **U6.2** (literal `"null"` strings in the Explorer table) is a low-risk, high-leverage one-line guard that cleans every Explorer domain. **U6.4/U6.5/U6.6** harden the Demand-History rail (duplicate-name ambiguity, unlabeled/misleading MoM%, missing aria). The remaining Store-Type IA, treemap, CommandCenterTab-size, and Item-Analysis date-picker items are re-pinned from cycle 5.

NEW actionable (P0/P1/P2): **3** — U6.1 (P1), U6.3 (P1), U6.2 (P2). U6.4 is P2 (count: 4 incl. U6.4). U6.5/U6.6 are P3.
