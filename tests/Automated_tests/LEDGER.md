# Improvement Ledger

## Cycle 1

- **U1.1** (P1) → FIXED — TodaysPlanBanner now stamps the planning/data as-of date (`formatAsOfDate(briefing.date)`), not browser `new Date()`.
- **U1.2** (P1) → FIXED — CommandCenterTab Portfolio Trend lines read `trendColors` from `useChartColors()`; no inline hex.
- **U1.8** (P2) → FIXED — action-feed source SELECTs LEFT JOIN dim_item; rows + detail now carry `item_desc`; ActionFeedPanel renders the product name. (Command Center storyboard rows: follow-up.)
- **F1.2** (P2) → FIXED — `summary.financial_at_risk_basis` added ("7-day lost gross margin (open exceptions) + proposed order value"); ActionFeedPanel surfaces it as the tile sublabel.
- **F1.1** (P2) → DEFERRED — Champion FVA rung needs a separate champion-accuracy query path (champion data is in fact_production_forecast, not the external table the waterfall reads); too large for one safe TDD slice.
- **F1.3 / U1.6** (P3/P2) → DEFERRED — Data Quality tile/catalog denominator + CRITICAL-badge-vs-status reconciliation; needs careful DataQualityTab work.
- **F1.4** (P3) → DEFERRED — cold 11s affinity query; perf/MV item, resolves Redis-warm.
- **U1.3** (P2) → DEFERRED — raw fetch() in model-tuning panels; pre-existing tsc errors in those files make a clean slice risky.
- **U1.5** (P3) → DEFERRED — banner vs feed currency rounding; two intentionally different formatters, low value.
- **U1.7** (P2) → DEFERRED — 7 tab files > 600 lines; mechanical splits, low correctness value.

## Cycle 2

- **F2.2** (P2) → FIXED — FVA champion rung degrades to reserved "planned" (consistent with AI/Planner) instead of broken "missing"; champion has zero measurable overlap with actuals, so a measured row still promotes to "actual". `fva.py` per-stage `missing_state`.
- **U2.1** (P1) → FIXED — CommandCenterTab Open Exceptions tile + critical badge now `formatInt` (6,141 / 2,464 critical), matching the feed footer.
- **U2.2** (P2) → FIXED — TodaysPlanBanner `PriorityBadge` comma-formats integer counts (Urgent 2,537 / High 1,715), matching the Action-Feed KPIs below.
- **F2.1** (P2) → FIXED — banner "$ at Risk" chip now carries the `financial_at_risk_basis` tooltip ("7-day lost gross margin …"), self-explaining vs the Command Center "Order Value at Risk" tile.
- **U2.3** (P1) → FIXED — Customer-Analytics dropdown/Clear/ranking surfaces moved off bare bg-white/bg-gray-* to theme tokens (bg-popover/bg-card/bg-muted/bg-accent); legible in Dark. Source-guard test.
- **U2.4** (P3) → FIXED — KpiSummaryCards skeleton bg-gray-200 → bg-muted (theme-aware).
- **F2.3 / U1.6** (P3/P2) → DEFERRED — DataQuality 166-vs-83 denominator + critical-badge; needs careful tile-aggregation rework.
- **F2.4** (P3) → DEFERRED — cold affinity ~11.6s; perf/MV item, resolves Redis-warm.
- **U2.5 / U1.7** (P2) → DEFERRED — 7 tabs > 600 lines; mechanical splits, high churn / low correctness value.
- **U2.6** (P2) → DEFERRED — retired tab keys show Command Center with stale URL; router/IA change + UX decision needed.
- **U2.7** (P3) → DEFERRED — Item Analysis breadcrumb bare "Item 15502"; lower value.
- **U2.8** (P3) → DEFERRED — sidebar vs page-heading naming; cosmetic.
- **U1.3** (P2) → DEFERRED (carried) — raw fetch() in model-tuning panels; pre-existing tsc errors make a clean slice risky.

## Cycle 3

- **U3.1** (P1) → FIXED — CA chart-panel toggle pills + tab clear-× moved off bare `bg-gray-*`/`text-gray-*` to theme tokens via shared `togglePillClass()`; legible in Dark. Source-guard test over 5 panels + tab.
- **U3.3** (P2) → FIXED — every CA toggle button now exposes `aria-pressed`; active pill also gains `font-medium` (non-color cue). Render test on CustomerHeatmap.
- **U3.2** (P2) → FIXED — Customer Demand Map footer total-demand now uses `formatCompactKMB`, matching the "Total Demand" KPI tile ("23.0M"). Render test.
- **U3.4** (P3) → FIXED (root-caused) — concentration_top10 / order_demand_ratio had a fabricated backend `delta: 0.0` (no MoM computed); now `delta: None` + UI renders "— no prior period" instead of "→ 0.0% MoM". Backend + frontend tests; live-verified null.
- **U3.5 / U2.7** (P3) → FIXED — `/sku/analysis` returns `dim_item.item_desc`; Item Analysis breadcrumb renders "Item <id> — <desc>" via new pure `itemBreadcrumbLabel`. Backend + frontend helper tests; live-verified item_desc.
- **U3.6 / U2.5 / U1.7** (P2) → DEFERRED (carried) — 7 tabs > 600 lines; mechanical splits, low correctness value.
- **U2.6** (P2) → DEFERRED (carried) — retired tab keys / stale URL; router + product decision.
- **U1.3** (P2) → DEFERRED (carried) — raw fetch() in model-tuning panels; pre-existing tsc errors.
- **U2.8** (P3) → DEFERRED (carried) — sidebar vs page-heading naming; cosmetic.

## Cycle 4

- **F4.1** (P2) → FIXED — Clusters Overview empty-ML state now probes the other source and names the populated alternative ("310,558 SKUs assigned via Source (sku.txt)…") instead of the misleading bare "No cluster assignments yet". `ClusterOverviewPanel.tsx` + render test. Live: ml=0 / source=310558.
- **U4.1** (P1) → FIXED — Demand-History Workbench/Matrix/Decomposition/Comparison panels swapped ~25 bare `text-gray-[456]00`/`bg-gray-[12]00` → `text-muted-foreground`/`bg-muted` (theme tokens, WCAG-legible in Dark). Source-guard test over all 4 panels.
- **U4.2** (P2) → FIXED — `ForecastTrendChart` tooltip now thousands-separates hover values (`formatInt`, "2,157,763"), matching the compact K/M axis + KPI tiles. Extracted pure `buildForecastTrendOption` + unit test.
- **U4.4** (P3) → FIXED — AiPlannerFvaTab run list shows a muted "No recommendations" sub-note for succeeded-with-0-recs runs (new pure `runYieldNote`), distinguishing clean-empty from unproductive. Helper tests.
- **F2.3 / U1.6** (P2) → DEFERRED (carried) — DataQuality 166-vs-83 denominator + CRITICAL-badge; needs tile-vs-catalog aggregation rework.
- **U4.3** (P2) → DEFERRED — CA below-fold panels stuck on "Loading…"; needs LazyPanel gating audit + bounded skeleton/retry + E2E timing; larger than one safe slice.
- **U3.6 / U2.5 / U1.7** (P2) → DEFERRED (carried) — 7 tabs > 600 LoC; mechanical splits.
- **U1.3** (P2) → DEFERRED (carried) — raw fetch() in model-tuning panels; pre-existing tsc errors.

## Cycle 5

- **U5.1** (P1) → FIXED — shared themed severity/status badge: new `severityBadgeClass()` (`lib/severityBadge.ts`) + semantic `badgeVariants` (critical/high/warning/info/success), each with a `dark:` tint pair. Migrated SopTab + PortfolioHealthPanel (13 hand-rolled `bg-*-100` lines) off Light-only chips. severityBadge unit test + SopTab dark-variant test + PortfolioHealthPanel source-guard.
- **U5.2** (P2) → FIXED — "AI FVA Backtest" nav icon changed BarChart3 → `Beaker` so it no longer collides with "FVA & ROI" in collapsed icon-only mode. AppSidebar distinct-icon test (also corrected stale NAV_ITEMS count 16→18).
- **U5.4** (P3) → FIXED — S&OP zero-cycle detail pane now says "Start a cycle to see its stages…" (not the contradictory unactionable "Select a cycle"). SopTab empty-state test.
- **F5.3 / F2.3 / U1.6** (P2, carried 4 cycles) → FIXED — DataQuality "Total Checks" tile relabeled "Check Runs … across N definitions", so the per-domain run total (166) and the distinct catalog denominator (83) self-explain. Display-layer only; live-verified 83 defs / 166 runs. DataQualityTab test.
- **F5.1 / U5.3** (P2) → DEFERRED — CA below-fold bounded skeleton + slow-query hint + error/retry; needs shared PanelLoading wired to each lazy panel's useQuery across 4+ panels.
- **F5.2 / F2.4 / F1.4** (P2) → DEFERRED (carried) — cold affinity ~11s; perf/MV task, resolves Redis-warm.
- **F5.6** (P3) → DEFERRED — Customer Demand Map auto-fit + populated-only State dropdown; Leaflet fitBounds + dropdown source, cosmetic.
- **F5.4** (P3) → NOT A BUG — BEER `<0%*` accuracy is genuine source over-forecast; UI honest.
- **F5.5 / F1.1** (P3) → DEFERRED (carried) — FVA AI/Planner ladder rungs need measured-vs-actual query path.
- **U3.6 / U2.5 / U1.7** (P2) → DEFERRED (carried) — 7 tabs > 600 LoC; mechanical splits.
- **U1.3** (P2) → DEFERRED (carried) — raw fetch() in model-tuning panels; pre-existing tsc errors.
