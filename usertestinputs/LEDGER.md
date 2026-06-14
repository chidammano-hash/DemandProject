# Improvement Ledger — Supply Chain Command Center UX Hardening Loop

Append-only index of issues found and resolved across cycles. Reviewers read this to avoid re-reporting resolved items. Branch: `restructure`. Method: live Playwright scan of 14 planner tabs → critique → applied+tested fix.

## Cycle 1 (manual)
- F1.1 (P0) → FIXED: Action Feed showed 0 actions despite 6,142 open critical exceptions. Wrong column (`created_at`→`exception_date`) + shared transaction abort; isolated each source via SAVEPOINT. Verified: feed now returns real critical actions.
- F1.2 (P0) → FIXED: Command Center 500 on missing `mv_control_tower_kpis`. Added `UndefinedTable` to graceful-catch + applied migration sql/035. 500→200.
- F1.3 (P1) → FIXED: fill-rate / inventory / inv-backtest trend 500s → graceful empty+warning degradation. 3× 500→200.
- F1.4 (P1) → FIXED: Item Analysis side-panel 500s from `/inventory/kpis` on unpopulated `agg_inventory_monthly` MV → SAVEPOINT degrade MV-derived KPIs to null, keep base totals. 500→200.
- F1.7 (P2) → FIXED: ClustersTab setState-during-render warning → moved `onDomainChange("sku")` into useEffect.
- F1.5 (P1) → DEFERRED: negative-accuracy heatmap presentation.
- F1.6 (P2) → DEFERRED: Data Quality "Run Checks" manual-only.
- F1.8 (P2) → DEFERRED: FVA Champion needs a backtest run (data state).
- F1.9 (P3) → DEFERRED: S&OP "New Cycle" only via API/CLI.
- Notes: full API suite 1239 passed. Pre-existing unrelated failures: tests/api/test_fva.py (3, IndexError fva.py:130), tests/unit/test_backtest_chronos.py (no torch). Tests added: tests/api/test_action_feed.py + graceful-degradation tests.

## Cycle 2
- F2.1 (P0) → FIXED: Command Center showed green "Portfolio looks healthy!" while KPIs were stale/zeroed. CommandCenterTab now derives `kpisStale` from the endpoint `warning` and shows an amber "data unavailable" banner + stale empty-state instead of the false all-clear.
- F2.2 / U2.5 (P1) → FIXED: DataQuality lineage 404s — frontend called `/data-quality/lineage/*`; repointed `platform.ts` fetchers to the mounted `/data-quality/batches|corrections` paths. 404→200.
- U2.1 (P1) → FIXED: raw `{"detail":...}` leaked into toasts — `fetchJson` now parses FastAPI detail + attaches `status` so `formatApiError` maps 404→friendly copy.
- U2.2 + U2.9 (P1/P3) → FIXED: `demandHistory` added to `VALID_TABS` (deep-link/refresh now survives); added NAV_ITEMS↔VALID_TABS drift-guard test.
- U2.3 + U2.4 (P2/P3) → FIXED: KPI delta now colors by per-metric `goodDirection` (rising OOS/concentration = red) and renders near-zero deltas flat/neutral. New pure `deltaPresentation()` + tests.
- U2.11 (P3) → FIXED: raw `below_ss` enum no longer leaks into Action Feed — `_EXCEPTION_TYPE_LABELS` map renders "Below Safety Stock" in title + detail.
- F2.4 (P2) → FIXED: `/fva/waterfall` ai_adjusted IndexError — added `len(ai_row) >= 3` guard + a 3-col promotion test. 3 failing FVA tests now green (8 pass).
- F2.3 / U2.7 (P1/P2) → DEFERRED: negative-accuracy heatmap framing (design decision).
- F2.6 (P3) → DEFERRED: Item Analysis low-volume default item (couples to F2.3).
- F2.5 (P2) → DEFERRED: DQ "Run Checks Now" button wire-up + stale `/dq/run` copy.
- U2.6 (P2) → DEFERRED: empty Customer Concentration treemap / concentration 404.
- U2.10 (P1) → DEFERRED: dirty Customer Analytics filter options — needs coordinated MV + WHERE-clause normalization.
- U2.8 (P3) → DEFERRED: six oversized tab files (pure refactor).
- Notes: backend API suite 1244 passed; frontend 1008 passed across my 7 touched/new test files. Pre-existing unrelated failures untouched: AppSidebar.test.tsx (stale nav count), DemandReferencePanel.test.tsx (recharts mock), test_backtest_chronos.py (no torch).

## Cycle 3
- F3.1 (P1) → FIXED: Command Center KPIs all-zero when `mv_control_tower_kpis` stale. Added `_exceptions_fallback()` live `COUNT(*) GROUP BY severity` over `fact_replenishment_exceptions` in control_tower.py; curl now shows 6142 open / 2465 critical / $246K (was 0). Keeps the stale `warning`.
- U3.1 (P1) → FIXED (2 of N tabs): ItemAnalysisTab (4 raw fetches → fetchSamplePair/fetchDomainSuggest/fetchSkuAnalysis + error panel, no more silent blank chart) and SopTab (advance/approve → new `advanceSopCycle`/`approveSopCycle` fetchJson POSTs). Source-guard test added. Model-tuning/clusters subpanels deferred.
- U3.2 (P1) → FIXED: DQ empty-state's `/dq/run` 404 curl + bogus `dq_run_checks.py` replaced with an in-app "Run DQ checks now" CTA (→ runDQChecks `POST /data-quality/run`) + correct `populate_dq_checks.py` hint.
- U3.3 (P1) → FIXED (frontend): `normalizeStateOptions()` whitelist (US/CA 2-letter codes) applied in `fetchCustomerAnalyticsFilterOptions`; State dropdown 135 raw → 60 clean codes, junk (`.`,`00`,`0D`,`XX`,numeric,null) dropped. MV cleanup still deferred.
- F3.2 / U3.6 (P2) → FIXED (heatmap): `formatHeatmapAccuracy()` floors negative accuracy to `<0%*` + legend + caption ("Accuracy=100−WAPE; <0%* = tiny base, see WAPE"). Cluster-comparison table negatives left for a follow-up.
- U3.4 (P2) → DEFERRED: S&OP New Cycle create (needs new `POST /sop/cycles` backend route).
- U3.5 (P2) → DEFERRED: Demand History unlabeled `%` column (needs metric definition confirmed).
- U3.7 (P2) → DEFERRED: Customer Concentration treemap empty / concentration 404 (needs backend route fix).
- U3.8 (P3) → DEFERRED: oversized tab files (pure refactor).
- F3.4 (P3) → DEFERRED: FVA Champion "No data" (genuinely-empty data state, no defect).
- Notes: control_tower backend 17 passed; 105 frontend tests passed across 9 touched files (incl. 4 new tests). Pre-existing TS errors (DataQualityTab:128 header button, ChampionPanel props, CA chart prop typings) and ruff nits (Optional→`X|None`) in untouched regions left alone. Working tree also held unrelated prior-cycle uncommitted changes (CommandCenterTab, KpiSummaryCards, conftest, several test_*.py) — not authored this cycle.

## Cycle 4
- F4.2 / U4.1 (P1) → FIXED: DQ Check Catalog empty + "Last Run: Never" while 166 checks ran. `/data-quality/checks` rewritten to drive existence off `fact_dq_check_results` (DISTINCT ON latest per check_name) with `dim_dq_check_catalog` demoted to LEFT JOIN enrichment. curl: `[]` → 83 checks with real `last_run`. New test `test_dq_checks_derives_from_results_when_catalog_empty`.
- F4.1 (P1) → FIXED: Command Center feed read empty `exception_queue` (tile "6142" above "Exception data unavailable"). `storyboard.list_exceptions` now falls back to `fact_replenishment_exceptions` when queue count=0 via `_replenishment_fallback()` (text→numeric severity, headline, same envelope, `source` tag). curl `/storyboard/exceptions` total 0 → 6142. New test `test_list_exceptions_falls_back_to_replenishment_when_queue_empty`.
- F4.5 / U4.2 (P1) → FIXED: Customer Analytics Channel/Store Type dirty dropdowns. Added `normalizeLabelOptions()` (trim, drop nullish, case-insensitive de-dupe keeping first casing, sort) applied to channels+store_types in `fetchCustomerAnalyticsFilterOptions`. channels 33→21, store_types 293→275. New test file `customer-analytics-labels.test.ts` (5 cases).
- F4.3 (P2) → DEFERRED: Portfolio Health 0/100 + Fill Rate "--" — needs live health/fill-rate fallback in control_tower.py (honest amber banner already mitigates).
- F4.4 (P2) → DEFERRED: cluster-comparison table raw negative accuracy — needs `formatHeatmapAccuracy()`-style flooring on the table.
- U4.3 (P2) → DEFERRED: Demand History `%` badge unbounded single-month MoM (needs metric decision).
- U4.4 (P2) → DEFERRED (3rd time): S&OP New Cycle — needs new guarded `POST /sop/cycles` + UI.
- U4.5 (P2) → DEFERRED: 8 raw-fetch subpanels — multi-file migration + guard-test expansion.
- U4.6 (P3) → DEFERRED: sidebar shortcut-digit styling.
- F4.6 (P3) → DEFERRED: Run-Checks button live-trigger polish + FVA Champion "No data" (no defect).
- Notes: backend test_data_quality 16 + test_storyboard 29 + test_action_feed + test_customer_analytics all green (74 passed in the combined CA/DQ/storyboard run); frontend labels+states 8 passed. Touched query file TS-clean; CA chart-component TS errors and `storyboard.py`/`data_quality.py` ruff `B905`/`I001` nits are pre-existing and in untouched code. No commits.

## Cycle 5
- U5.1 (P1) → FIXED: popstate skipped TAB_REDIRECTS so Back/Forward reached dead controlTower/aiPlanner tab branches. Extracted `resolveTab()` (useUrlState.ts) used by both getInitialTab + popstate; deleted the dead App.tsx branches + lazy imports. RED resolveTab undefined → GREEN 23 passed.
- F5.1 (P2) → FIXED: Item×State heatmap cold load ~9.4s. New `mv_ca_item_state` MV (sql/187) + heatmap routes through it via `_build_where_item_state`; never touches dim_item. Wired into refresh-ca-mvs + refresh-mvs-tiered. RED (stashed fix) asserted mv absent → GREEN; live cold curl 9.4s → 0.43s (filtered 0.30s), real data.
- F4.4 (P2) → FIXED: cluster slice table printed raw negative accuracy (-12.89%). Extracted `formatSliceCell()` (SliceTablePanel.tsx) routing accuracy_pct through `formatHeatmapAccuracy` (floors to `<0%*`), WAPE/bias unchanged. RED export missing → GREEN 5 passed.
- U5.2 (P1, blank-treemap + doc-drift halves) → FIXED: 7 CA echarts panels had height but no width → 0-width collapse. `mergeEchartsProps()` (echarts-modular.tsx) defaults `style.width:"100%"` for all panels; corrected MEMORY.md false "echarts deleted" line to sanction echarts-modular for the 8 CA panels. RED export missing → GREEN 4 passed.
- F4.3 (P2) → DEFERRED: health/fill-rate live fallback in control_tower.py (amber banner mitigates).
- F4.5 / U5.4 (P2) → DEFERRED: Store Type taxonomy needs upstream raw→canonical mapping + searchable combobox.
- U5.3 (P2) → DEFERRED: CA chart inline hex colors not theme-aware (multi-file theming change).
- U5.5 (P2) → DEFERRED: CommandCenterTab 844-line split (pure refactor).
- U5.6 (P2) → DEFERRED: Item Analysis FROM/TO raw ISO dropdowns + range validation.
- U5.7 (P3) → DEFERRED: cross-filter badge palette literals.
- Notes: frontend 58 passed across 5 touched test areas; backend test_customer_analytics 30 + test_storyboard 29 green. New ruff in touched code: none (RUF003 at geo.py:104,324 pre-existing in untouched comments). tsc clean on touched files. No commits.

## Cycle 6
- U6.1 (P1) → FIXED: Portfolio KPI delta sign contradicted its color (WAPE "−1.9pp" painted red). Added `goodDirection` to `KpiCard.trend` (colors by good/bad movement, decoupled from displayed sign); `AggregateAnalysisTab` now passes RAW deltas + goodDirection (WAPE down, Accuracy up, Bias toward-zero). RED color-key test failed → GREEN 40 passed; live `/dashboard/kpis` wape delta +1.85 now shown "+1.85pp" red (true).
- U6.3 (P1) → FIXED: Customer Concentration treemap blank (height-only style → 0-width collapse). `CustomerTreemap` chart now `width:"100%"` + wrapper `w-full`. RED new CustomerTreemap.test (no w-full / undefined width) → GREEN 2 passed; live `/treemap` payload valid.
- U6.2 (P2) → FIXED: Explorer rendered literal string "null" (API emits `'null'`). `formatCell` + new `isEmptyCell()` treat case-insensitive null/none/na/undefined sentinels as "-"; ExplorerTable title uses it. RED `formatCell("none")` → GREEN 36 passed; live `/domains/item` 24/50 string-null upcs now "-".
- U6.4 (P2) → FIXED: Demand-History rail listed duplicate item descriptions with no disambiguator. Exported `formatSeriesTitle`; rail row appends muted `#<item_id>` suffix + full-id title. RED import-not-exported → GREEN 3 passed.
- U6.5 (P3) → FIXED: unlabeled single-month MoM% (520% spike misleads). MoM span gets aria-label/title "Month-over-month change: …", low-base (>200%) `*` footnote, one-time rail header. RED aria-label absent → GREEN 8 passed.
- U6.6 (P3) → FIXED: series toggle buttons had no aria-pressed. Added `aria-pressed={isSelected}`. RED attribute absent → GREEN. (Overlay-count role=status announcement deferred.)
- F6.1 (P3) → FIXED: DQ Check Catalog over-weighted configured severity (28 passing critical rows read as broken). Severity pill muted when `last_status==="pass"`; "Last Value (violations)" header + per-cell defect-count tooltip. RED (fix stashed) passing pill still red / no violation title → GREEN 3 passed; live `/data-quality/checks` 28 critical+passing rows.
- F4.3 (P2) → DEFERRED (4th): health/fill-rate live fallback in control_tower.py (amber banner mitigates).
- F4.5 / U5.4 (P2) → DEFERRED: Store Type taxonomy needs upstream raw→canonical mapping + searchable combobox.
- U5.5 (P2) → DEFERRED: CommandCenterTab >600-line split (pure refactor).
- U5.6 (P2) → DEFERRED: Item Analysis FROM/TO raw ISO dropdowns + range validation.
- F6.2 (P3) → DEFERRED: dead `/customer-analytics/concentration` 404 route (no UI calls it; cleanup only).
- Notes: all fixes frontend-only, live via HMR. Targeted 9 files 157 passed; full frontend 1054 passed. 2 pre-existing unrelated failures (AppSidebar stale nav count, DemandReferencePanel recharts mock) confirmed failing on clean tree. Touched files tsc-clean (pre-existing CustomerTreemap:71 + AggregateAnalysisTab:626 TS errors left). No commits. Working tree holds prior-cycle uncommitted changes not authored this cycle.

## Cycle 7
- F7.1 (P2) → FIXED: DQ Domain Health cards read 62.5%/75% with "0 fail / 0 warn" because skipped checks were in the score denominator but never surfaced. Backend `dq_dashboard()` now FILTERs a `skipped` count, excludes skips from the score denominator (all-passing scored checks → 100%), and emits `skipped` so passed+failed+warn+skipped==total; frontend `DataQualityTab` shows `{d.skipped} skip` on each card + a "Skipped" summary tile; `DQDomainScore` gains `skipped`. RED backend `assert 166.7==100.0` / frontend "6 skip" not found → GREEN 17 backend + 32 frontend passed; live `/data-quality/dashboard` forecast 75%→100% (skipped:4), item 62.5%→100% (skipped:6).
- U7.1 (P1) → FIXED: Customer Concentration treemap still blank — root cause was `visualMap.dimension:"fill_rate"` (a string is not a valid scalar-value dimension index → every node out-of-range/transparent). Removed the visualMap; `colorizeTree()` attaches an explicit `itemStyle.color` (red→amber→green fill-rate ramp) to every node + a static gradient legend `<div>`. RED emitted-option `visualMap.dimension` still "fill_rate" / no node color → GREEN 4 passed; live `/treemap` nodes all carry fill_rate so rectangles now draw colored.
- U7.2 (P2) → FIXED: AI Planner FVA run list dead-ended on `failed` / `succeeded`-0-rec rows (3 blank "No data yet." panels) despite the API returning `error_message`. `RunsListPanel` passes the full RunMetadata + shows `error_message` inline on failed rows; main shell renders a "This run failed: <msg>" card for failed runs and a single "No recommendations were generated" card for 0-rec runs, suppressing the blank KPI/FVA panels. RED "No recommendations were generated…" / failed-detail text not found → GREEN 34 passed; live `/runs` has 1 failed (real pydantic error) + 2 zero-rec runs.
- U7.3 (P3) → DEFERRED: FVA nav label ("AI FVA Backtest") vs H1 ("AI Planner — FVA Backtest") drift; entangles with the pre-existing AppSidebar stale-nav-count test failure.
- U7.4 (P3) → DEFERRED: Customer Map KPI strip MoM deltas unlabeled (mirrors cycle-6 U6.5).
- F4.3 (P2) → DEFERRED (5th): health/fill-rate live fallback in control_tower.py (amber banner mitigates).
- F4.5 / U5.4 (P2) → DEFERRED: Store Type taxonomy needs upstream raw→canonical mapping + searchable combobox.
- U5.5 (P2) → DEFERRED: CommandCenterTab >600-line split (pure refactor).
- U5.6 (P2) → DEFERRED: Item Analysis FROM/TO raw ISO dropdowns + range validation.
- F6.2 (P3) → DEFERRED: dead `/customer-analytics/concentration` 404 route (cleanup only).
- Notes: backend test_data_quality 17 + test_storyboard 46 combined green. Frontend full suite 1060 passed / 2 failed (the documented pre-existing AppSidebar stale-nav-count + DemandReferencePanel recharts-mock failures, unrelated to this cycle). Touched files tsc-clean apart from the pre-existing CustomerTreemap ExportButtons `getData` TS error (documented cycle 6). No commits.

## Cycle 8
- U8.1 (P2) → FIXED: "Today's Plan" At-Risk tile showed "$4K" vs the Action Feed's "$3.6K" (same $3598.89 metric). New `todaysPlanFormat.ts::formatCompactCurrency()` uses one decimal sub-$10K so the banner reads "$3.6K"; wired into TodaysPlanBanner. RED format import missing / RED component found "$4K" → GREEN 8 passed.
- U8.2 (P2) → FIXED: "Today's Plan" row printed "0 SKUs · 3,152 at risk · 0 excess ($0K)" from unpopulated briefing fields. `shouldRenderStat()` degrades 0/null total_skus to "—" and drops the empty excess chip (mirrors the avg_health_score guard). RED component found `<span>0 SKUs</span>` → GREEN.
- U8.3 (P2) → FIXED: DQ domain score ignored severity — an INFO-only fail cratered sku_to_item/sku_to_location to 0% alarm-red. Dashboard SQL now FILTERs `info_fails` and excludes them from the score denominator (like skips); `info_fails` returned; raw fail count kept; frontend `DQDomainScore.info_fails` + a muted "{n} info" card chip. RED backend `assert 0.0==100.0` / RED frontend "2 info" not found → GREEN 18 backend + 33 frontend passed; live sku_to_item/sku_to_location 0%→100% (info_fails:2), genuine fails unchanged.
- U8.4 (P3) → DEFERRED: Explorer redundant Item Ck/Item Id lead columns (needs generic `*_ck` demotion in field-ordering metadata).
- U8.5 (P3) → DEFERRED: Heatmap `<0%*` cells need inline tooltip (caption already mitigates; polish).
- U8.6 (P3) → DEFERRED (5th, ex-U3.4/U4.4): S&OP CLI-only dead end — needs new guarded `POST /sop/cycles` + UI.
- F4.3 (P2) → DEFERRED (6th): Control Tower health/fill-rate live fallback (amber banner mitigates).
- F4.5/U5.4, U5.5, U5.6 (P2) → DEFERRED (carried): Store Type taxonomy / CommandCenterTab split / Item Analysis date pickers.
- F6.2 (P3) → DEFERRED: dead `/customer-analytics/concentration` 404 route (cleanup only).
- Notes: backend test_data_quality 18 + test_action_feed combined 21 passed; frontend 72 passed across 4 touched files (incl. 3 new tests). Planner reported 0 new defects. Pre-existing untouched: DataQualityTab:130 header-button TS error, router ruff nits (123/174/364), test-file RUF059 (`_make_pool` convention). No commits.

## Cycle 9
- U9.1 (P1) → FIXED: Unified Action Feed summary counted only the 20-row display page (`actions[:limit]` before the aggregation), so "20 critical · $3.6K" understated the true population ~70×. Added a full-population UNION-ALL aggregate query (own SAVEPOINT, same urgency thresholds) feeding the summary; degrades to page-level counts on failure; added `displayed` + frontend "Showing top N of {total}" caption. RED backend `assert 2==4180` / RED frontend "showing top 20 of 6,214" absent → GREEN 5 backend + 2 frontend passed; live summary 20/20/$3.6K → 6214/4252/$12,099.96 (displayed:20).
- F9.1 (P2) → FIXED: FVA waterfall blanked at 3-month window because it anchored to wall-clock `current_date` (months ahead of the demo horizon). Replaced both `current_date` anchors (waterfall `dfu_filter` + roi-summary) with a bound `%s::date` `get_planning_date()`. RED `AttributeError: fva has no get_planning_date` → GREEN 9 passed; live `/fva/waterfall?months=3` seasonal_naive/external "missing"/0 rows → 65.82%/72.57% @ 10,889 rows; 12m unchanged.
- U9.3 (P2) → FIXED (accessibility half): Customer-Map MoM delta badges had no aria-label/period anchor. New `deltaAriaLabel()` → `aria-label`/`title` "Up/Down N% month-over-month vs prior month" on `DeltaBadge`. RED getByLabelText absent → GREEN 1 + 6 (KpiDelta) passed. Partial-month base correction (kpis SQL) deferred.
- U9.2 (P3) → DEFERRED: Today's Plan Top Actions duplicates first 3 feed rows (IA polish).
- U9.4 (P3) → DEFERRED: April-dated exceptions under June planning date need "as of <date>" vintage line.
- U9.5 (P3) → DEFERRED: DQ "0.00" passing-value cell ambiguity (muted-zero styling).
- F4.3 (P2) → DEFERRED (7th): Control Tower health/fill-rate live fallback (amber banner mitigates).
- F4.5/U5.4, U5.5, U5.6 (P2) → DEFERRED (carried): Store Type taxonomy / oversized tabs / Item Analysis date pickers.
- F6.2 (P3) → DEFERRED: dead `/customer-analytics/concentration` 404 route (cleanup only).
- Notes: backend test_action_feed (5) + test_fva (9) + test_storyboard + test_control_tower combined green (58 in the broader run); frontend 21 passed across inv-planning + customer-analytics dirs. Touched backend files lint clean for new code (pre-existing RUF005/Optional nits at untouched inv_planning_insights.py:15,403,704-706,763 left alone). Working tree holds prior-cycle uncommitted changes. No commits.
