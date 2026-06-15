# Usability Review — Cycle 2

App: Supply Chain Command Center (React + Vite). Evidence: `tests/Automated_tests/cycle2/` (capture-digest.md, capture-dump.json, screens/*.png) + read-only code inspection of `frontend/src/tabs` and `src/components`.

Cross-checked against `LEDGER.md` and `usability1.md`. Cycle-1 FIXED items (U1.1 banner as-of date, U1.2 inline hex, U1.8 item descriptions in feeds, F1.2 at-risk basis) are confirmed live in this capture and are NOT re-reported. Cycle-1 DEFERRED items (U1.3 raw fetch in model-tuning, U1.6 DQ critical-badge, U1.7 oversized tabs, U1.5 currency rounding, F1.1/F1.3/F1.4) remain open and are re-stated below only where this cycle adds new evidence.

Verified non-issues this cycle (NOT reported): Customer-Analytics MoM badges already carry direction-aware color + period-anchored aria labels (U2.3/U2.4/U9.3 done — `KpiSummaryCards.tsx`); the 8 trailing "Loading..." blocks on Customer Map are sanctioned below-fold `LazyPanel` fallbacks; `?tab=clusters` rendering standalone is by design (removed from sidebar, reached via Jobs scenario flow — `App.tsx:37`); FVA "Coming Soon / Reserved" stages are documented known states.

---

## U2.1 — Command Center "Open Exceptions" KPI renders an unseparated integer ("6141"), while the same value is comma-formatted elsewhere on the page
- **Category:** consistency
- **Severity:** P1
- **Evidence:** `screens/commandCenter.png` / digest line 13 — the prominent Open Exceptions tile reads **"6141"** and its badge reads **"2464 critical"**, both without thousands separators. The feed footer on the *same screen* (digest line 41: "Showing top 50 of 6,141 exceptions") renders **"6,141"** via `formatInt`. Two formats for one number, with the most prominent surface being the unformatted one.
- **File:** `frontend/src/tabs/CommandCenterTab.tsx:442` (`value={String(ex?.open_exceptions_total ?? 0)}`) and `:445` (badge `${ex.critical_exceptions} critical`). `formatInt` is already imported at `:45` and used at `:618-620`.
- **Recommendation:** Wrap both in the existing `formatInt`: `value={ex?.open_exceptions_total != null ? formatInt(ex.open_exceptions_total) : "--"}` and badge `${formatInt(ex.critical_exceptions)} critical`. Zero new imports.
- **Acceptance:** For `open_exceptions_total = 6141`, the tile renders "6,141" and the badge "2,464 critical"; a render test asserts the tile text matches the feed-footer formatting.

## U2.2 — "Today's Plan" priority badges (Urgent / High) render raw integers, mismatching the comma-formatted Action-Feed KPIs directly below
- **Category:** consistency
- **Severity:** P2
- **Evidence:** `screens/invPlanning.png` / digest lines 1143-1149 — banner shows **"Urgent 2537"**, **"High 1715"** (no separators), while the Action-Feed KPI block immediately below (digest lines 1200-1206) shows **"Critical 2,537"**, **"High Priority 1,715"** (comma-formatted). Same counts, two formats, stacked vertically on one screen.
- **File:** `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx:45` — `PriorityBadge` renders `{value ?? count ?? 0}` with no formatting; callers at `:101-102` pass `summary?.critical` / `summary?.high`.
- **Recommendation:** Format the count inside `PriorityBadge` with `toLocaleString()` (the formatter already used for `stats.*` at `:142-149` in the same file) or the shared `formatInt`.
- **Acceptance:** For `critical = 2537`, the Urgent badge renders "2,537", matching the Action-Feed "Critical" KPI; covered by a component test.

## U2.3 — Customer-Analytics filter dropdown and ranking header use `bg-white` / `bg-gray-*` with no dark-mode variant — illegible in Dark theme
- **Category:** accessibility
- **Severity:** P1
- **Evidence:** The item-search autocomplete dropdown is `className="... bg-white border rounded shadow-lg ..."` with no `dark:` background, so in Dark theme it renders an opaque white panel over a dark page (white-on-white text risk). The Customer Ranking sticky header has the same `bg-white` gap. The "Clear" button (`bg-gray-100 ... hover:bg-gray-200`, no `dark:`) is also off-theme. (App ships a Dark theme toggle — `screens/clusters.png` shows the Light/Soft/Dark control.)
- **File:** `frontend/src/tabs/CustomerAnalyticsTab.tsx:198` (`bg-white` dropdown), `:203` (`hover:bg-gray-100`), `:294` (`bg-gray-100 ... hover:bg-gray-200`); `frontend/src/tabs/customer-analytics/CustomerRanking.tsx:33` (`bg-white` sticky header).
- **Recommendation:** Replace `bg-white` with the theme token `bg-popover`/`bg-card` and add `dark:` variants (e.g. `bg-popover text-popover-foreground`, `hover:bg-accent`) matching the pattern already used across `demand-history/` and `clusters/` panels.
- **Acceptance:** Rendering the Customer-Analytics filter dropdown and ranking header in Dark theme shows a dark surface with legible foreground text; no bare `bg-white` remains in `CustomerAnalyticsTab.tsx` or `customer-analytics/CustomerRanking.tsx` (grep returns none).

## U2.4 — KpiSummaryCards loading skeleton uses hardcoded `bg-gray-200`, invisible against a dark surface
- **Category:** accessibility
- **Severity:** P3
- **Evidence:** `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:160` — the shimmer placeholder is `animate-pulse bg-gray-200 rounded ...` with no `dark:` variant, so the loading state is barely visible in Dark theme (low contrast on a dark card).
- **File:** `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:160`
- **Recommendation:** Use a theme-aware skeleton token (e.g. `bg-muted` / the shared `Skeleton` component) instead of `bg-gray-200`.
- **Acceptance:** The KPI skeleton uses a theme token; visible in Light and Dark. Grep for `bg-gray-200` in the file returns none.

## U2.5 — CommandCenterTab is now 941 lines (worst tab, grew from 896), well over the 600-line rule
- **Category:** simplification
- **Severity:** P2
- **Evidence:** `wc -l`: `CommandCenterTab.tsx` **941** (was 896 in cycle 1; the gap widened), plus `DataQualityTab.tsx` 707, `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 672, `AggregateAnalysisTab.tsx` 668, `SettingsTab.tsx` 649, `AiPlannerFvaTab.tsx` 610. CLAUDE.md: "Tab files MUST be < 600 lines." This re-states cycle-1 U1.7 with the note that CommandCenter is actively drifting further over, not converging.
- **File:** `frontend/src/tabs/CommandCenterTab.tsx` (and the 6 others).
- **Recommendation:** Extract from CommandCenterTab the KPI-tile row (`KpiSummaryCard` block, ~`:418-486`), the filter toolbar (`:489-560`), and the exception-row renderer (`:820-910`) into `tabs/command-center/<Subpanel>.tsx`. Prioritize CommandCenter; it is the largest and growing.
- **Acceptance:** Every `frontend/src/tabs/*.tsx` is < 600 lines (`for f in tabs/*.tsx; do [ $(wc -l <"$f") -lt 600 ]; done`), existing tests green.

## U2.6 — Three tabs (Command Center, Control Tower, AI Planner) render byte-identical content, giving the planner three sidebar entries that all show the same page
- **Category:** information-architecture
- **Severity:** P2
- **Evidence:** digest sections for `?tab=controlTower` (line 1497) and `?tab=aiPlanner` (line 2243) reproduce the Command Center header verbatim ("Unified morning triage: portfolio health KPIs ..."), the same 4 KPI tiles (58/100, 6141, 98.2%, $246.8K), and the same exception list. These are `TAB_REDIRECTS` consolidations, but `controlTower`/`aiPlanner` are still **valid URL tabs** (`useUrlState.ts` `VALID_TABS`) that a deep-link, bookmark, or stale link lands on — and they silently show "Command Center" with no breadcrumb explaining the redirect. A planner who bookmarked "AI Planner" sees "Command Center" with no signal it moved.
- **File:** `frontend/src/hooks/useUrlState.ts` (`VALID_TABS` includes `controlTower`, `aiPlanner`, `storyboard`, `exceptions`; redirect map consolidates them) and `frontend/src/App.tsx` tab router.
- **Recommendation:** When a retired tab key resolves to Command Center via redirect, either (a) rewrite the URL to `?tab=commandCenter` on landing (so the address bar reflects the real tab), or (b) show a one-line breadcrumb/toast "AI Planner is now part of Command Center." Pick one; today the URL stays `?tab=aiPlanner` while the page says Command Center, which is confusing.
- **Acceptance:** Navigating to `?tab=aiPlanner` either redirects the URL to `?tab=commandCenter` or surfaces a visible "moved" affordance; a routing test asserts the chosen behavior.

## U2.7 — Item Analysis breadcrumb labels the DFU as a bare code ("Item 15502") with no product name
- **Category:** usability
- **Severity:** P3
- **Evidence:** `screens/itemAnalysis.png` / digest lines 3162-3163 — the page header / breadcrumb reads **"Item 15502 · 1401-BULK"** using only the numeric item_id, while the same product is human-readable on Demand History (digest line 882: "TITOS HANDMADE VODKA 80 #84587"). Same gap U1.8 fixed for the action feeds, but not carried to the Item Analysis breadcrumb.
- **File:** `frontend/src/tabs/ItemAnalysisTab.tsx:429` (`{ label: \`Item ${skuItem}\` }`).
- **Recommendation:** When the selected DFU's `item_desc` is available (already fetched for the chart/attributes), render "627099 — Tito's Handmade Vodka 80" in the breadcrumb label, falling back to the bare code while loading.
- **Acceptance:** With a selected item whose description is loaded, the Item Analysis breadcrumb shows the description alongside the id; verified by a render test.

## U2.8 — Sidebar label "Portfolio" vs page heading "Portfolio Analysis", and "Customer Analytics" nav vs internal "Customer Map" naming, are inconsistent across the IA
- **Category:** consistency
- **Severity:** P3
- **Evidence:** Sidebar (`AppSidebar.tsx:44`) labels the aggregate tab **"Portfolio"** but the page heading (digest line 634) is **"Portfolio Analysis"** — minor. More notably, the nav (`AppSidebar.tsx:49`) says **"Customer Analytics"**, the page heading (digest line 3458) says **"Customer Analytics"**, yet the capture/route id and internal naming is **"Customer Map"** (`?tab=customerAnalytics` digest header) — three names for one surface. Inconsistent naming makes the command-palette and breadcrumbs harder to scan.
- **File:** `frontend/src/components/AppSidebar.tsx:44,49` and the respective tab heading strings in `AggregateAnalysisTab.tsx` / `CustomerAnalyticsTab.tsx`.
- **Recommendation:** Pick one canonical name per tab and use it in the sidebar label, page `<h1>`, breadcrumb, and command palette. Either "Portfolio" everywhere or "Portfolio Analysis" everywhere; same for "Customer Analytics".
- **Acceptance:** For each tab, the sidebar label and the page heading use the same canonical string; a test asserts `navItem.label` is a prefix of / equal to the rendered page heading for these two tabs.

---

### Re-stated (still-open) cycle-1 items — not re-counted as new
- **U1.3** (P2, consistency) — raw `fetch(` in `tabs/model-tuning/{LogViewer,EnhancedPromoteModal,ExperimentBuilder,EnhancedComparisonPanel}.tsx` still bypasses `fetchJson`. Confirmed still present this cycle (7 call sites).
- **U1.7** (P2, simplification) — 7 tabs > 600 lines; see U2.5 (CommandCenter actively grew).
- **U1.6 / F1.3** (P2/P3) — Data Quality "0 Failed" headline beside red CRITICAL-severity catalog rows (digest lines 3082-3156); severity-vs-status still reads as alarm.
