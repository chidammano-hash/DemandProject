# Usability Review — Cycle 1

App: Supply Chain Command Center (React + Vite). Evidence: `tests/Automated_tests/cycle1/` (capture-digest.md, screens/*.png) + code inspection of `frontend/src/tabs` and `src/components`.

No prior `LEDGER.md` or `usability0.md` present — all items below are NEW.

Note on non-issues verified this cycle (NOT reported): `?tab=aiPlanner`, `?tab=controlTower`, `?tab=storyboard` rendering "Command Center" is intentional (`useUrlState.ts` `TAB_REDIRECTS` — retired tabs consolidated). Customer-Analytics "Loading..." tiles are sanctioned below-fold `LazyPanel`/IntersectionObserver fallbacks. Data-Quality summary counts reconcile (116 pass + 0 fail + 26 warn + 6 info + 18 skip = 166).

---

## U1.1 — "Today's Plan" banner stamped with browser system date, not the planning/data as-of date
- **Category:** consistency
- **Severity:** P1
- **Evidence:** `screens/invPlanning.png` — header reads "Today's Plan · Sunday, Jun 14" while every action row below is dated "Apr 2, 2026" (the planning date / data as-of). A planner reads "Today's Plan" and assumes same-day data, but the figures are anchored to Apr 2, 2026.
- **File:** `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx:76` (`new Date().toLocaleDateString(...)`)
- **Recommendation:** Replace the `new Date()` stamp with the planning/data as-of date that the action feed and KPIs are computed against (already available from the backend planning date — surface it via the daily-briefing/action-feed response or a shared planning-date query). Render it as e.g. "Plan as of Apr 2, 2026" so the date label matches the data shown. Never derive a user-facing data anchor from `new Date()`.
- **Acceptance:** The date shown in the Today's Plan banner equals the as-of date of the action rows it summarizes (Apr 2, 2026 in the captured state), verified by a component test that mocks the action-feed/briefing as-of date and asserts the banner renders that date (not the wall-clock date).

## U1.2 — Inline hex chart colors in CommandCenterTab (theme rule violation; breaks dark mode + theming)
- **Category:** consistency
- **Severity:** P1
- **Evidence:** `tabs/CommandCenterTab.tsx:651` `stroke="#3b82f6"` and `:660` `stroke="#10b981"` on the Portfolio Trend lines. CLAUDE.md forbids inline hex in `tabs/`; charts must read color from `useChartColors()`/`useThemeContext()`. Hardcoded hex won't adapt to Soft/Dark themes.
- **File:** `frontend/src/tabs/CommandCenterTab.tsx:651,660`
- **Recommendation:** Pull line colors from `useChartColors()` (series palette) instead of literals, matching the pattern used by other Recharts panels in the app.
- **Acceptance:** `grep -n '#[0-9a-fA-F]\{6\}' frontend/src/tabs/CommandCenterTab.tsx` returns no chart-color literals; the Portfolio Trend lines render with theme palette colors in Light/Soft/Dark (visual or snapshot check).

## U1.3 — Raw `fetch()` in Model Tuning panels bypasses the fetchJson error-sanitization layer
- **Category:** consistency
- **Severity:** P2
- **Evidence:** 5 raw `fetch(` calls outside the queries layer: `tabs/model-tuning/LogViewer.tsx:67`, `EnhancedComparisonPanel.tsx:98`, `EnhancedPromoteModal.tsx:79,91,106,121`, `ExperimentBuilder.tsx:89`. CLAUDE.md: all HTTP must go through `src/api/queries/<module>.ts` via `fetchJson`. The existing guard `tabs/__tests__/no-raw-fetch.test.ts` does not watch these files, so a 404/500 here surfaces raw/unsanitized errors (the exact failure mode the guard was built to prevent).
- **File:** `frontend/src/tabs/model-tuning/{LogViewer,EnhancedComparisonPanel,EnhancedPromoteModal,ExperimentBuilder}.tsx`
- **Recommendation:** Move these calls into a `src/api/queries/model-tuning.ts` (or existing tuning query module) using `fetchJson`, mirroring backend schemas as typed interfaces (no `: any`). Then add the four files to the `WATCHED` list in `no-raw-fetch.test.ts`.
- **Acceptance:** No bare `fetch(` remains in `tabs/model-tuning/`; the four files appear in `no-raw-fetch.test.ts` `WATCHED` and the guard passes.

## U1.4 — Generic "Loading..." fallback for below-fold Customer-Analytics panels gives no sense of what is coming
- **Category:** usability
- **Severity:** P3
- **Evidence:** `capture-digest.md` Customer Map section shows 8 identical "Loading..." blocks; `tabs/CustomerAnalyticsTab.tsx:52-61` `PanelFallback` renders bare text "Loading...". With 8 lazy panels stacked, the planner sees a wall of identical placeholders with no titles or shape.
- **File:** `frontend/src/tabs/CustomerAnalyticsTab.tsx:52`
- **Recommendation:** Give `PanelFallback` an optional `label` (panel name) and/or a lightweight skeleton (header bar + chart-shaped shimmer) so each placeholder communicates which visual (Heatmap, Sankey, Channel Mix, …) is loading. Pass the panel name at each `LazyPanel` call site.
- **Acceptance:** Each below-fold Customer-Analytics placeholder shows the panel's title (or a labeled skeleton) instead of bare "Loading..."; verified by a render test asserting the fallback contains the panel name.

## U1.5 — "At Risk" total disagrees between Today's Plan banner ($12K) and Action Feed KPI ($12.1K)
- **Category:** consistency
- **Severity:** P3
- **Evidence:** `screens/invPlanning.png` / digest: Today's Plan banner "At Risk $12K" vs. Action Feed KPI "Financial Impact at Risk $12.1K" — same underlying figure rounded two different ways on the same screen.
- **File:** `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx` (`formatCompactCurrency` via `todaysPlanFormat`) vs. the Action Feed KPI formatter in `tabs/inv-planning/ActionFeedPanel.tsx`
- **Recommendation:** Use a single shared compact-currency formatter (one rounding rule) for both the banner and the Action Feed KPI so the same value renders identically.
- **Acceptance:** For an identical `financial_at_risk` value, the banner "At Risk" and the Action Feed "Financial Impact at Risk" render the same string; covered by a unit test on the shared formatter.

## U1.6 — Data Quality catalog: "CRITICAL" severity badges read as failures next to a "0 Failed" headline
- **Category:** information-architecture
- **Severity:** P2
- **Evidence:** `screens/dataQuality.png` + digest lines ~2943-3017: headline tiles say "0 Failed", yet the Check Catalog lists many rows with a red "CRITICAL" badge and "0.00 (violations)" / passing last run. The "CRITICAL" badge is the check's *severity classification* (impact if it ever fails), but visually it reads as a current critical failure, contradicting "0 Failed".
- **File:** `frontend/src/tabs/DataQualityTab.tsx` (catalog severity column rendering)
- **Recommendation:** Distinguish *result status* from *configured severity* in the catalog — e.g. show a green "Pass" status pill prominently and render configured severity as a muted/neutral chip (or relabel the column header "Severity if failed"). Ensure a passing critical-severity check does not present in alarm-red.
- **Acceptance:** In the catalog, a passing check with critical severity renders with a pass-colored status indicator and a non-alarm severity chip; no red "CRITICAL" styling appears on a row whose last run passed. Verified by a render test on a passing critical-severity row.

## U1.7 — CommandCenterTab (896 LoC) and DataQualityTab (707) exceed the 600-line tab limit
- **Category:** simplification
- **Severity:** P2
- **Evidence:** `wc -l`: `CommandCenterTab.tsx` 896, `DataQualityTab.tsx` 707, `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 672, `AggregateAnalysisTab.tsx` 667, `SettingsTab.tsx` 649, `AiPlannerFvaTab.tsx` 610 — all above the CLAUDE.md "Tab files MUST be < 600 lines" rule. CommandCenter is the worst (nearly 50% over).
- **File:** `frontend/src/tabs/CommandCenterTab.tsx` (and the 6 others listed)
- **Recommendation:** Split each over-limit tab into `tabs/<tab-name>/<Subpanel>.tsx` per the established pattern. Prioritize CommandCenterTab: extract the KPI tiles, the exceptions list, and the Portfolio Trend chart into subpanels.
- **Acceptance:** Every file in `frontend/src/tabs/*.tsx` is < 600 lines (`for f in tabs/*.tsx; do [ $(wc -l <"$f") -lt 600 ]; done`), with existing tests still green.

## U1.8 — Exception/action labels are pure code pairs ("627099 @ 1401-BULK") with no product name
- **Category:** usability
- **Severity:** P2
- **Evidence:** Command Center, Control Tower, AI Planner, and Inventory Planning all list exceptions as "Stockout — 627099 @ 1401-BULK" using only the numeric item_id and a location code. The same items ARE human-readable elsewhere (Demand History shows "TITOS HANDMADE VODKA 80 #84587"). A planner triaging the morning queue can't recognize what 627099 is without clicking "View Item".
- **File:** `frontend/src/tabs/CommandCenterTab.tsx` (exception row rendering) and `frontend/src/tabs/inv-planning/ActionFeedPanel.tsx`
- **Recommendation:** Include the item description (already joined to `dim_item` for other tabs) alongside the id in exception/action rows — e.g. "Stockout — Tito's Handmade Vodka 80 (627099 @ 1401-BULK)". If the description isn't in the current payload, add it to the action-feed/exceptions response.
- **Acceptance:** Each exception/action row shows a human-readable item description in addition to the id+location code; verified by a render test asserting the description text appears for a mocked row.
