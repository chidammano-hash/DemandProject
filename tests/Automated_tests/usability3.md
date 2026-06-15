# Usability Review — Cycle 3

App: Supply Chain Command Center (React + Vite, branded "Vrantis"). Evidence: `tests/Automated_tests/cycle3/` (capture-digest.md, capture-dump.json, screens/*.png) + read-only code inspection of `frontend/src/tabs` and `src/components`.

Cross-checked against `LEDGER.md`, `usability1.md`, `usability2.md`. Confirmed-fixed and NOT re-reported: U1.1 banner as-of date ("Plan as of Apr 2, 2026" live on Portfolio/Item/Inv), U1.2/U1.8/F1.2 (item descriptions + at-risk basis in feeds), U2.1/U2.2 (Command Center + banner integer comma-formatting — digest shows "6,141" / "2,464 critical" / "Urgent 2,537"), U2.3 (CA dropdown/ranking/clear off bare bg-white), U2.4 (KpiSummaryCards skeleton now `bg-muted` — verified `KpiSummaryCards.tsx:160`), F2.1/F2.2 (FVA tooltip + champion degrade), U8.3/F7.2 (DataQuality "Failed 0" now severity-aware — `DataQualityTab.tsx:116-138`). The CA MoM badges already carry direction-aware color + period-anchored aria labels (`deltaPresentation`/`deltaAriaLabel`) — verified non-issue.

This cycle surfaces a **new family of dark-theme color regressions in the Customer-Analytics chart panels** that cycle-2's U2.3 fix did not reach, plus two new consistency/format items. Carried items (U2.5 oversized tabs, U2.7 item breadcrumb, U2.6 retired-tab URL, U1.3 raw fetch) are re-stated at the end and NOT re-counted as new.

---

## U3.1 — Customer-Analytics chart-panel segmented toggles use hardcoded `bg-gray-100 text-gray-600` (no dark variant) — low-contrast / illegible in Dark theme
- **Category:** accessibility
- **Severity:** P1
- **Evidence:** Cycle-2 U2.3 migrated the CA *filter dropdown / ranking header / Clear button* off bare `bg-white`/`bg-gray-*`, and `CustomerRanking.tsx:126` already uses the theme token (`bg-muted text-muted-foreground`). But the metric/grain/group-by toggle pills inside **six** CA chart panels were left on raw Tailwind grays with no `dark:` variant, so in Dark theme the inactive pills render gray-on-gray (the active pills are saturated indigo/teal, fine; the inactive set is the problem). 9 call sites: `CustomerDemandMap.tsx:233,244`, `CustomerHeatmap.tsx:149,160,169`, `ChannelSunburst.tsx:191`, `OosImpactBubble.tsx:138`, `SegmentSparklines.tsx:110`. Also the search-clear "×" in `CustomerAnalyticsTab.tsx:191` is `text-gray-400 hover:text-gray-600` with no dark variant. (App ships a Dark toggle — `screens/customerAnalytics.png` shows Light/Soft/Dark; captures are Light so the defect is code-evidenced, not visible in the PNG.)
- **File:** `frontend/src/tabs/customer-analytics/CustomerDemandMap.tsx:233,244`; `CustomerHeatmap.tsx:149,160,169`; `ChannelSunburst.tsx:191`; `OosImpactBubble.tsx:138`; `SegmentSparklines.tsx:110`; `frontend/src/tabs/CustomerAnalyticsTab.tsx:191`.
- **Recommendation:** Replace the inactive-pill `bg-gray-100 text-gray-600 hover:bg-gray-200` with the same theme tokens `CustomerRanking.tsx:126` already uses: `bg-muted text-muted-foreground hover:bg-accent`. Keep the active state as-is (or move to `bg-primary text-primary-foreground` for full theme-consistency). Swap the clear-× to `text-muted-foreground hover:text-foreground`.
- **Acceptance:** Rendering each CA chart panel in Dark theme shows legible inactive toggle pills (foreground contrast ≥ active); grep for `bg-gray-100`/`bg-gray-200`/`text-gray-600` in `tabs/customer-analytics/*.tsx` and the clear-× line returns none. A source-guard test (mirroring the U2.3 guard) asserts no bare `bg-gray-1` toggle classes remain.

## U3.2 — "Total Demand" renders in two formats on one Customer-Analytics screen: compact "23.0M cases" KPI tile vs full-digit "22,986,295 cases" map footer
- **Category:** consistency
- **Severity:** P2
- **Evidence:** `screens/customerAnalytics.png` / digest lines 3462-3463 — the KPI tile reads **"Total Demand 23.0M cases"** (`formatCompactKMB`), while the Customer Demand Map footer directly below (digest line 3558) reads **"22,986,295 cases total demand"** (`formatInt`, imported as `fmtNum`). Same metric, same screen, two precisions/formats — a planner reading the tile and the map sees "23.0M" vs "22,986,295" and must reconcile them mentally.
- **File:** KPI tile `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:63` (`format: fmtNum` = `formatCompactKMB`); map footer `frontend/src/tabs/customer-analytics/CustomerDemandMap.tsx:15` (`import { formatInt as fmtNum }`) used at `:253`.
- **Recommendation:** Pick one demand format for the page. Simplest: render the map footer with `formatCompactKMB` so it matches the KPI tile ("23.0M cases"); or keep the precise count in the footer but add the compact form in parentheses. Don't ship both bare forms unlabeled.
- **Acceptance:** The CA KPI "Total Demand" value and the map footer "total demand" value use the same formatter (both compact or both full); a render test asserts the two strings derive from one formatter helper.

## U3.3 — Customer-Analytics map metric/group-by toggles are pure-color affordances with no aria/role, and the active state is encoded by color alone
- **Category:** accessibility
- **Severity:** P2
- **Evidence:** The map "state/city/zip" group-by row and the "Customers / Demand / Sales / OOS / Fill Rate %" metric row (digest lines 3549-3556, `screens/customerAnalytics.png`) are rendered as bare `<button>` pills whose selected state is conveyed only by `bg-indigo-600 text-white` vs `bg-gray-100` — no `aria-pressed`, no `role="tab"`/`aria-selected`. A screen-reader user (or a color-blind user in the indigo/teal vs gray case) cannot tell which metric is active.
- **File:** `frontend/src/tabs/customer-analytics/CustomerDemandMap.tsx:233-247`; same pattern in `CustomerHeatmap.tsx:149-169`, `ChannelSunburst.tsx:191`, `OosImpactBubble.tsx:138`, `SegmentSparklines.tsx:110`.
- **Recommendation:** Add `aria-pressed={metric === m}` (or wrap each group in `role="tablist"` with `role="tab" aria-selected`) so the active selection is exposed non-visually. Pair with the U3.1 token fix so the active pill also has a non-color cue (weight/ring) if feasible.
- **Acceptance:** Each CA toggle button exposes its selected state via `aria-pressed`/`aria-selected`; a test asserts the active metric button has the pressed attribute set and inactive ones do not.

## U3.4 — Customer-Analytics KPI deltas (e.g. "→ 0.0% MoM" on Demand Concentration / Order-to-Demand) render a flat arrow with no anchor for *why* it's flat vs *no data*
- **Category:** usability
- **Severity:** P3
- **Evidence:** `screens/customerAnalytics.png` / digest lines 3475-3479 — "Demand Concentration 17.7% → 0.0% MoM" and "Order-to-Demand Ratio 0.98 → 0.0% MoM" both show the neutral "→ 0.0%" badge. The `deltaPresentation` flat branch (`KpiSummaryCards.tsx:42`) treats `|delta| < 0.05` as flat with a `→` arrow, but a true 0.0 (genuinely unchanged) and a missing/unavailable delta both render identically as "→ 0.0% MoM". A planner can't tell "concentration is stable" from "we have no prior month to compare".
- **File:** `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:42-44,116-124`.
- **Recommendation:** When the backend delta is null/undefined (no prior period), render "— no prior month" instead of synthesizing "→ 0.0% MoM". Keep the flat "→ 0.0%" only for a real near-zero change. The aria label already distinguishes "No material change" — extend the visible text similarly.
- **Acceptance:** A KPI with a null delta shows a "no prior period" affordance (not "→ 0.0% MoM"); a render test covers null-delta vs zero-delta producing distinct text.

## U3.5 — Item Analysis breadcrumb still labels the DFU as a bare numeric code ("Item 185690"), no product name
- **Category:** usability
- **Severity:** P3
- **Evidence:** `screens/itemAnalysis.png` / digest lines 3161-3163 — the page breadcrumb reads **"Item 185690 · 1401-BULK"** using only `item_id`, while the same product is human-readable everywhere else (Demand History digest line 882 "TITOS HANDMADE VODKA 80", Command Center feed rows carry the description). This is the cycle-2 U2.7 gap, re-confirmed live this cycle with fresh evidence (item changed to 185690 but the bare-code label persists). The fix U1.8 applied to the action feeds was never carried to this breadcrumb.
- **File:** `frontend/src/tabs/ItemAnalysisTab.tsx:429` (`{ label: \`Item ${skuItem}\`, ... }`).
- **Recommendation:** When the selected DFU's `item_desc` is loaded (it is fetched for the attributes/KPI section), render "185690 — Tito's Handmade Vodka 80" in the breadcrumb, falling back to the bare code while loading.
- **Acceptance:** With a selected item whose description is loaded, the Item Analysis breadcrumb shows the description alongside the id; verified by a render test. (Re-stated U2.7 — counts as carried, not new.)

## U3.6 — CommandCenterTab is still 941 lines (largest tab, unchanged from cycle 2), over the 600-line rule
- **Category:** simplification
- **Severity:** P2
- **Evidence:** `wc -l tabs/*.tsx`: `CommandCenterTab.tsx` **941**, then `DataQualityTab.tsx` 707, `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 672, `AggregateAnalysisTab.tsx` 668, `SettingsTab.tsx` 649, `AiPlannerFvaTab.tsx` 610. CLAUDE.md: "Tab files MUST be < 600 lines." Unchanged since cycle 2 (U2.5) — still 7 tabs over, CommandCenter unchanged at 941.
- **File:** `frontend/src/tabs/CommandCenterTab.tsx` (and the 6 others).
- **Recommendation:** Extract from CommandCenterTab the KPI-tile row, the filter toolbar, and the exception-row renderer into `tabs/command-center/<Subpanel>.tsx`. Prioritize CommandCenter (largest).
- **Acceptance:** Every `frontend/src/tabs/*.tsx` is < 600 lines; existing tests green. (Re-stated U2.5/U1.7 — carried, not new.)

---

### Re-stated (still-open) prior items — not re-counted as new
- **U2.6** (P2, information-architecture) — `?tab=aiPlanner` / `?tab=controlTower` still render byte-identical Command Center content (digest lines 1497, 2243) while the URL keeps the retired tab key; no breadcrumb/redirect signals the move. Open.
- **U1.3** (P2, consistency) — raw `fetch(` still bypasses `fetchJson` in `tabs/model-tuning/{EnhancedPromoteModal(×4), LogViewer, ExperimentBuilder, EnhancedComparisonPanel}.tsx` (7 call sites confirmed this cycle). Open.
- **U2.8** (P3, consistency) — sidebar "Portfolio" vs page "Portfolio Analysis"; CA route id "Customer Map" vs h1 "Customer Analytics" (sidebar now reads "Customer Analytics", so half-reconciled). Open, low value.
