# Usability Review — Cycle 4

App: Supply Chain Command Center (React + Vite, branded "Vrantis"). Evidence: `tests/Automated_tests/cycle4/` (capture-digest.md, capture-dump.json, screens/*.png) + read-only code inspection of `frontend/src/tabs` and `src/components`.

Cross-checked against `LEDGER.md`, `usability1.md`, `usability2.md`, `usability3.md`. **Confirmed-fixed and NOT re-reported:**
- **U2.6 retired-tab URL** — now resolved. `useUrlState.ts:12-34` has `TAB_REDIRECTS` (`aiPlanner`/`controlTower`/`storyboard` → `commandCenter`) + `resolveTab()` shared by `getInitialTab` and `usePopstateSync`. The cycle-4 digest still *shows* Command Center content under `?tab=aiPlanner`/`?tab=controlTower` (lines 1497, 2243) because that is exactly the redirect target; the URL is rewritten to `tab=commandCenter` by the `updateUrlState` effect. Working as intended — not re-reported.
- **U3.1/U3.2/U3.3/U3.4/U3.5** (CA dark-theme toggles, total-demand format, aria-pressed, null-delta "— no prior period", Item Analysis breadcrumb) — all live this cycle: digest line 3476/3479 shows "— no prior period"; line 3162 shows "Item 134418 — GUENOC MERLOT CALIF(SC)"; CA toggles use theme tokens. Verified non-issues.

This cycle the highest-value NEW finding is that the **dark-theme bare-gray regression (cycle-3 U3.1) extends well beyond Customer-Analytics** into the Demand-History Workbench (the default-visible panel, 22 sites), plus a missing chart-tooltip number format on the Aggregate "Forecast vs Actual" chart. Two carried items (DataQuality 166-vs-83, oversized tabs) are re-stated and NOT counted as new.

---

## U4.1 — Dark-theme low-contrast regression in Demand-History Workbench (default panel): 22 `text-gray-400/500` label sites with no `dark:` variant

- **Category:** accessibility
- **Severity:** P1
- **Evidence:** Cycle-3 U3.1 fixed the *Customer-Analytics* panels off bare grays, but the same defect lives untouched in the **Demand History** tab — whose default view is the Workbench (digest line 868 "Workbench" is the first/active sub-tab). `WorkbenchPanel.tsx` has **22** bare `text-gray-400`/`text-gray-500` sites with no `dark:` companion, used for the series-count header ("50 OF 50 SERIES · LAST · MOM %", digest line 881), the MoM/value columns, the period chips, the search icon, and the "Loading forecast…" / "No series selected" empty states (digest lines 1133-1135). In Dark theme these render gray-on-near-black, well below WCAG AA for the primary working surface of the tab. The pattern repeats in sibling panels: `MatrixPanel.tsx` (11 sites), `DecompositionPanel.tsx` (5), `ComparisonPanel.tsx` (1). Captures are Light, so the defect is code-evidenced (same basis the cycle-3 CA finding used).
- **File:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx` (22 sites incl. :197, :257, :264, :269, :393, :427, :445, :459, :496, :532, :542, :576, :693, :697, :709); `MatrixPanel.tsx` (11); `DecompositionPanel.tsx` (5); `ComparisonPanel.tsx` (1).
- **Recommendation:** Replace the bare `text-gray-400`/`text-gray-500` label/empty-state classes with the theme token `text-muted-foreground` (the same swap the CA fix used), and any `bg-gray-100/200` chips with `bg-muted`. Keep the blue/emerald *active* accents as-is. Add a source-guard test (mirroring the CA U3.1 guard) asserting no bare `text-gray-[456]00` / `bg-gray-[12]00` survive in `tabs/demand-history/*.tsx`.
- **Acceptance:** Rendering Demand-History Workbench/Matrix/Decomposition in Dark theme shows legible muted labels and empty states (contrast ≥ AA); grep for `text-gray-[456]00`/`bg-gray-[12]00` (sans `dark:`) in `tabs/demand-history/*.tsx` returns none; source-guard test green.

## U4.2 — "Forecast vs Actual" chart tooltip shows raw unformatted integers (e.g. `2157763`) — no thousands separators, inconsistent with the K/M axis and every other number on the screen

- **Category:** consistency
- **Severity:** P2
- **Evidence:** Portfolio Analysis "Forecast vs Actual" chart (digest lines 703-718) renders monthly forecast/actual as bare digits — `2025-05 2157763 1944956`, `2025-12 2692933 2993230`. The chart's Y-axis *is* compact-formatted (`ForecastTrendChart.tsx:90-94` → "M"/"K"), and the KPI tiles above read "Forecast Vol 2.2M / Actual Vol 2.1M" (digest 697-701), but the ECharts `tooltip` block (`ForecastTrendChart.tsx:65-70`) has **no `valueFormatter`** — so hovering a point surfaces the raw `2157763`. A planner reading "2.2M" in the tile then hovering the chart sees a 7-digit unseparated string and must mentally reconcile them.
- **File:** `frontend/src/components/ForecastTrendChart.tsx:64-70` (tooltip option, missing `valueFormatter`).
- **Recommendation:** Add `valueFormatter: (v) => formatInt(v)` (or the same compact M/K helper the axis uses) to the `tooltip` option so hover values render `2,157,763` (or `2.16M`) consistently with the axis and the KPI tiles. Reuse the existing formatter rather than inlining a new one.
- **Acceptance:** Hovering a "Forecast vs Actual" point shows a thousands-separated or compact value matching the axis; a render/unit test on `ForecastTrendChart` asserts the tooltip `valueFormatter` is set and produces a separated string for a 7-digit input.

## U4.3 — Customer-Analytics "Channel Mix" sunburst (and 7 sibling panels) stuck on "Loading…" at capture time — below-fold panels never settle / silent slow path

- **Category:** performance
- **Severity:** P2
- **Evidence:** `screens/customerAnalytics.png` / digest lines 3582-3591 — the "Channel Mix" section header and **eight** stacked "Loading…" placeholders are the last visible content; the sunburst, heatmap, and order-pattern panels below the Demand Map never resolved within the capture window. This is the cold-affinity / heavy-CA-panel slowness flagged generically before (F1.4/F2.4 "cold ~11s") but here it manifests as a *persistent* "Loading…" wall with no timeout, no error, and no "this panel is heavy / expand to load" affordance — a planner scrolling down sees an indefinitely-spinning page.
- **File:** `frontend/src/tabs/customer-analytics/ChannelSunburst.tsx:202` (and the LazyPanel-wrapped siblings); CA tab below-fold panel group.
- **Recommendation:** (1) Confirm these panels are `LazyPanel`/IntersectionObserver-gated so they only fetch when scrolled into view (per the project's `LazyPanel` pattern) — a wall of 8 simultaneous "Loading…" suggests they fire on mount. (2) Give each heavy panel a bounded skeleton with a "still loading — large query" sublabel after ~3s, and an error/retry on failure, rather than an unbounded "Loading…". (3) Verify the affinity/sunburst endpoints hit the Redis-warm MV path (mv_customer_activity_monthly) and not a cold 11s scan.
- **Acceptance:** Scrolling the CA tab, below-fold panels fetch lazily (network shows no request until in-view); a panel exceeding a threshold shows a "large query" hint; a failed panel shows retry, not a perpetual "Loading…". An E2E/integration check asserts the panels are gated and resolve or error within a bounded time.

## U4.4 — AI-Planner-FVA run list marks `succeeded` runs that produced `0` recommendations identically to productive runs — no "no actions found" affordance

- **Category:** usability
- **Severity:** P3
- **Evidence:** `screens/aiPlannerFva.png` / digest lines 2886, 2888 — two rows read `succeeded 2026-04-01 ollama 50 0` (emerald "succeeded", 50 DFUs sampled, **0** recommendations). The status cell (`AiPlannerFvaTab.tsx:303-316`) colors "succeeded" emerald regardless of yield, and the RECS cell just shows `0`. A planner can't distinguish "the AI ran cleanly and found nothing actionable" (a legitimate, informative outcome) from "the run technically succeeded but did nothing useful" — both look like green successes next to a bare `0`.
- **File:** `frontend/src/tabs/AiPlannerFvaTab.tsx:303-321`.
- **Recommendation:** When `status === "succeeded" && n_recommendations === 0`, render a muted "no recommendations" sub-note (mirroring the existing `humanizeRunError` inline row at :325-335) or a neutral badge, so a zero-yield success reads as informational rather than an unexplained green `0`. The detail pane already computes this (`AiPlannerFvaTab.tsx:535-536`) — surface the same signal in the list.
- **Acceptance:** A succeeded run with 0 recommendations shows a distinct "no recommendations" affordance in the run list; a render test covers succeeded-with-recs vs succeeded-zero-recs producing distinct output.

---

### Re-stated (still-open) prior items — NOT re-counted as new

- **F2.3 / U1.6** (P2, consistency) — DataQuality headline "Total Checks **166**" (`DataQualityTab.tsx:131`, `Σ domain.total`) vs "Check Catalog (**83**)" (`CheckCatalogPanel.tsx:76`, `checkList.length`) on the same screen (digest lines 2905, 3040). The per-domain rollup counts each physical check under both its source domain and its referential-integrity domain (e.g. `Sales` + `Sales_to_sku`), doubling 83 → 166. A planner sees two different "how many checks" numbers. Open — needs a tile-vs-catalog denominator reconciliation. (Carried.)
- **U3.6 / U2.5 / U1.7** (P2, simplification) — 7 tab files still > 600 lines: `CommandCenterTab.tsx` **941** (largest, unchanged), `DataQualityTab.tsx` 707, `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 672, `AggregateAnalysisTab.tsx` 668, `SettingsTab.tsx` 649, `AiPlannerFvaTab.tsx` 610. (`ItemAnalysisTab.tsx` now 566, still under the rule.) CLAUDE.md: tabs MUST be < 600 lines. Mechanical splits, low correctness value. Open. (Carried.)
- **U1.3** (P2, consistency) — raw `fetch(` still bypasses `fetchJson` in 4 model-tuning panels: `EnhancedPromoteModal.tsx`, `ExperimentBuilder.tsx`, `LogViewer.tsx`, `EnhancedComparisonPanel.tsx` (confirmed this cycle). Pre-existing tsc errors in those files make a clean slice risky. Open. (Carried.)
