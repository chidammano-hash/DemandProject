# Demand Planner UX Findings — Cycle 4
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-4 live capture digest + dump (14 tabs, **0 console errors** on every tab) + curl / DB / code confirmation. Branch: restructure._

## Summary
The product continues to improve. **All 14 tabs loaded with ZERO console errors** (capture-dump: every entry `ok=true`, `consoleErrors=[]`). Three cycle-3 items are verified RESOLVED:

- **F3.1 (exceptions fallback) RESOLVED** — `/control-tower/kpis` now returns `exceptions.open_exceptions_total=6142`, `critical_exceptions=2465`, `high=1715`, `recommended_order_value=$246,723`, `"source":"fact_replenishment_exceptions"`. Command Center KPI tiles now show **Open Exceptions 6142 / 2465 critical** and **Critical Items 2465** (were 0 last cycle).
- **F3.2 (heatmap negatives) RESOLVED** — the BEER row now renders **`<0%*`** in all four months with the caption "Cells marked <0%* have actuals near zero on a tiny base (WAPE > 100%) — review WAPE rather than reading the negative literally." Code: `formatHeatmapAccuracy()` in `aggregateShared.ts:109` returns `"<0%*"` for `value < 0`.
- **F3.3 (DQ checks never run) — DATA HALF RESOLVED** — the DQ battery ran: dashboard shows **166 total checks, 64% overall health, 116 passed / 26 failed / 6 warnings**, 20 domain cards, and a 32-row "Recent Issues" panel (latest "11m ago"). `fact_dq_check_results` = 166 rows / 83 distinct checks in the last 24h.

**Two NEW inconsistencies surfaced this cycle**, both caused by a screen reading a *different, empty* table than the one that actually holds the data — i.e. the fallbacks added in cycle 3 fixed the KPI tiles but exposed a deeper split between the "tile" data source and the "list/catalog" data source on the same two screens:

1. **F4.1 (P1)** — Command Center's exception **feed** is wired to the empty `exception_queue` table, structurally disconnected from the 6,142 real replenishment exceptions. The same screen now shows "**6142** Open Exceptions" in a KPI tile and "**Exception data unavailable**" in the feed directly below it — self-contradictory, and the feed will stay empty even after `make refresh-mvs-tiered`.
2. **F4.2 (P1)** — Data Quality shows "**Last Run: Never**" and "**Check Catalog (0) — No checks configured yet**" while the same page shows 166 checks ran 11m ago. `/data-quality/checks` reads the empty `dim_dq_check_catalog`; the results live in `fact_dq_check_results`.

The rest are carried presentation/data-state items from prior cycles. No 500s, no 404s the app actually calls, no broken charts.

---

## NEW Findings (prioritized)

### F4.1 — Command Center exception feed reads the EMPTY `exception_queue` table; same screen shows "6142 Open Exceptions" in a tile and "Exception data unavailable" in the feed below  [SEV: P1]  (NEW — exposed by the F3.1 KPI-tile fix)
- **Workflow blocked:** Morning portfolio triage on the default landing tab (also the Control Tower and AI Planner routes). The planner cannot triage individual exceptions from the home screen — they must jump to the Inventory Planning tab.
- **Evidence:** Tab `commandCenter` (`screens/commandCenter.png`, dump entry 0, **0 console errors**). KPI tiles read **Open Exceptions 6142 / 2465 critical**, **Critical Items 2465** (real, from the F3.1 fallback) — but the feed area directly below shows the amber empty-state "**Exception data unavailable** — Analytics views are stale, so this feed cannot be trusted to be empty." Curl proof of the split:
  - `/control-tower/kpis` → `exceptions.open_exceptions_total=6142` … `"source":"fact_replenishment_exceptions"` (drives the tiles).
  - `/storyboard/exceptions?limit=5` → `{"total":0,"rows":[]}` and `/storyboard/exceptions/summary` → all-zero (drives the feed).
  - `/inv-planning/action-feed?limit=3` → returns 3 real critical actions (627099/664631/… @ 1401-BULK) — the Inventory Planning tab renders all 20.
  - DB: `SELECT count(*) FROM exception_queue` → **0**; `fact_replenishment_exceptions WHERE status='open'` → **6142**.
- **Root cause:** `CommandCenterTab.tsx` builds its feed (`unified`, line 256) from `fetchSbExceptions` → `/storyboard/exceptions` (`api/routers/intelligence/storyboard.py`, `list_exceptions`, `FROM exception_queue` at line 130) + AI insights. `exception_queue` is a forecast-storyboard table (forecast_bias / stockout_risk / accuracy_drop types) that is empty/never populated. The 6,142 actionable replenishment exceptions live in `fact_replenishment_exceptions` and are only reachable via `/inv-planning/action-feed`. Because `unified.length===0` AND `kpisStale` is true, `CommandCenterTab.tsx:505` renders "Exception data unavailable" — which is the right copy for a stale-MV situation but the **wrong** conclusion here: the data exists, it's just in a table this feed doesn't read. The amber banner masks a permanently-empty feed; refreshing MVs will not fill it.
- **Acceptance criterion:** On the Command Center, the exception feed shows the same actionable items the Inventory Planning Action Feed shows. Concretely: when `exception_queue` is empty but `fact_replenishment_exceptions` has open rows, the Command Center feed renders rows sourced from `/inv-planning/action-feed` (or `control_tower._exceptions_fallback` extended to return rows, not just counts), so a planner can triage critical stockouts from the home screen. An httpx/RTL test asserts that with `exception_queue` empty and ≥1 open replenishment exception, the feed is non-empty and the "Exception data unavailable" empty-state is NOT shown. At minimum (cheaper), the KPI tile and the feed must not contradict — don't show "6142 Open Exceptions" above an "Exception data unavailable / cannot be trusted to be empty" panel.
- **Planner impact:** The home screen looks both alarming (6142!) and broken (feed unavailable) at the same time. I can't act from it, and the two halves disagree, which erodes trust in the one screen designed to be my morning starting point. Newly visible this cycle precisely because the KPI tiles started showing real numbers.

### F4.2 — Data Quality shows "Last Run: Never" and "Check Catalog (0) — No checks configured yet" while 166 checks demonstrably ran 11m ago  [SEV: P1]  (NEW — exposed by the F3.3 data-half fix)
- **Workflow blocked:** Trusting upstream data quality / data freshness from inside the app before relying on forecasts and inventory numbers.
- **Evidence:** Tab `dataQuality` (`screens/dataQuality.png`, dump entry 9, **0 console errors**). Header tiles: **64% Overall Health, 166 Total Checks, 116 Passed, 26 Failed, 6 Warnings, Last Run: Never**. The "Check Catalog (N)" panel reads "**Check Catalog (0)**" / "**No checks configured yet.**" Yet "Recent Issues (32)" lists real failures stamped "**11m ago**" / "23m ago". Curl proof of the split:
  - `/data-quality/dashboard` → 20 domains, populated (`{"domain":"inventory","passed":20,"failed":8,...}`), but the payload has **no `last_run` field** (top key is only `domains`).
  - `/data-quality/checks` → `{"checks":[]}` (empty).
  - `/data-quality/history?days=7` → real entries with `run_ts:"2026-06-14T06:..."`.
  - DB: `dim_dq_check_catalog` = **0** rows; `fact_dq_check_results` (last 24h) = **166** rows / **83** distinct check_names.
- **Root cause:** `api/routers/platform/data_quality.py` `dq_checks()` (line 49) selects `FROM dim_dq_check_catalog c LEFT JOIN LATERAL (… fact_dq_check_results …)` — i.e. the catalog **drives** the query, and the catalog table is empty, so it returns `[]`. The DQ run populated only the *results* table, not the *catalog* dimension. In `DataQualityTab.tsx`, the "Last Run" tile (line 196) derives from `lastRun = reduce(checkList, max(c.last_run))` over that empty `checkList` (line 111), so it falls back to "Never"; the Check Catalog panel (line 256) is driven by the same empty list. Both render empty/Never despite 166 results existing one table over.
- **Acceptance criterion:** When `fact_dq_check_results` has rows in the last 24h, the "Last Run" tile shows the real timestamp (e.g. "11m ago"), and the Check Catalog lists the 83 distinct checks with their last status/value/run. Fix either by populating `dim_dq_check_catalog` during the DQ run, or by deriving the catalog/last-run from `DISTINCT check_name` over `fact_dq_check_results` (the same source `/dashboard` and `/history` already use). Tests: a backend test asserts `/data-quality/checks` returns ≥1 row when results exist; an RTL test asserts "Last Run" renders a relative time (not "Never") given a non-empty checks payload.
- **Planner impact:** The DQ page is now self-contradictory — it tells me 26 checks are failing and 64% healthy (so checks clearly ran) but simultaneously claims they've "Never" run and "No checks configured." I can't trust the freshness signal, and the Check Catalog (the place I'd go to see exactly which check covers which table) is blank.

---

## Carried items (re-verified this cycle — NOT new)

### F4.3 — Command Center Portfolio Health 0/100 and Fill Rate "--" (health/fill-rate MVs unpopulated; no live fallback)  [SEV: P2]  (carried; data-half of F3.1 / F2.1)
- **Evidence:** `/control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0, below_ss_count:0}`, `fill_rate.portfolio_fill_rate_3m:null`, `warning:"mv_control_tower_kpis not yet refreshed"`. Tiles show Portfolio Health **0/100**, Fill Rate (3m) **--**, Portfolio Trend "No trend data available." DB: `mv_control_tower_kpis`, `mv_fill_rate_monthly`, `agg_inventory_monthly`, `mv_inventory_health_score` all `ispopulated=f`; base data present (`fact_inventory_snapshot`=4.91M rows).
- **Root cause:** `control_tower.py get_control_tower_kpis` reads only `mv_control_tower_kpis`; on `ObjectNotInPrerequisiteState` it degrades the health + fill-rate halves to zero/null. The cycle-3 fallback covers `exceptions` only — there is no equivalent live fallback for health (a `COUNT … GROUP BY health_tier` over inventory) or fill rate.
- **Acceptance criterion:** After `make refresh-mvs-tiered` the health/fill-rate tiles show non-zero numbers; OR a live fallback computes below-SS coverage / fill rate from base tables when the MV is stale (mirroring the exceptions fallback). Downgraded to P2 because the honest amber banner removes the trust hazard and the actionable exception counts now render.

### F4.4 — Cluster Accuracy Comparison table still renders raw negative accuracy (`-128.04%`, `-61.08%`, `-12.89%`)  [SEV: P2]  (carried; deferred half of F3.2)
- **Evidence:** `aggregateAnalysis` "MODEL COMPARISON — 13 CLUSTER ASSIGNMENT BUCKETS": L2_4 ★-12.89% (WAPE 112.89%), L2_5 ★-61.08%, L2_6 ★-128.04% (WAPE 228.04%), L2_6S ★-18.19%, L2_99 ★-6.17%, biases up to ⚠-69.2%. The **heatmap** half was floored to `<0%*` (F3.2 fixed), but this table was not.
- **Root cause:** The cluster-comparison table component does not apply `formatHeatmapAccuracy()`-style flooring/annotation — it prints raw `accuracy_pct`. Same low-base/intermittent artifact of `100 − 100·Σ|F−A|/|ΣA|`.
- **Acceptance criterion:** Low-base rows floor accuracy at 0% with a "low base — see WAPE" marker (consistent with the heatmap), or the table caption explains negative = forecast ≫ actual on a small base. Snapshot test covers a row with `accuracy_pct < 0`.

### F4.5 — Customer Analytics Channel & Store Type filter dropdowns still show dirty/duplicated raw values  [SEV: P2]  (carried; U2.10 / U3.x — State was cleaned, Channel/Store Type were not)
- **Evidence:** `customerAnalytics` dump: Channel dropdown lists case-variant duplicates ("Off Premise Chains" / "OFF PREMISE CHAINS" / "Off Premise Chains" ×3, "On Premise Accounts" ×3, "Off Premise Independents" ×3), plus `null`, "Undefined", "Unassigned Accounts" ×2. Store Type has dozens of raw case-variant duplicates ("BAR"/"Bar"/"BARS", "CASINO"/"Casino", "CHAIN GROCERY"/"Chain Grocery Store"/"CHAIN GROCERY STORE", …). The **State** dropdown is now clean (US/CA 2-letter codes — U3.3 fix verified).
- **Root cause:** `fetchCustomerAnalyticsFilterOptions` normalizes State (`normalizeStateOptions()` whitelist) but Channel/Store Type pass through raw distinct values from the MV without case-folding/dedup/null-drop.
- **Acceptance criterion:** Channel and Store Type options are de-duplicated case-insensitively, `null`/"Undefined" dropped, and a single canonical label shown per group — same treatment State received.

### F4.6 — Data Quality "Run Checks Now" button wiring + FVA Champion "No data"  [SEV: P3]  (carried; button-half of F3.3 and F3.4 — both honest empty/decorative states, no defect)
- **Run Checks Now:** the DQ battery now has run (166 results), so the empty-state CTA is no longer the only path; verifying the button issues `POST /data-quality/run` in-app remains the open polish item. (Note: this is largely moot now that data exists, but the button should still trigger a re-run without leaving the app.)
- **FVA Champion:** `fva` tab still shows Naive Seasonal 65.6% → External 70.8% (+5.2 pts, 76,995 rows) then **Champion "No data"**, AI/Planner "Coming Soon". `/fva/waterfall` → `champion.state="missing"`. `fact_candidate_forecast`=0 (no promoted backtest). Genuinely-empty data state, honestly labeled. No code defect — populates after a backtest is promoted.

---

## Tabs working well (no action)
- **Inventory Planning** — Action Feed: 20 critical actions, $3.6K at risk, individual SKUs (627099, 664631, … @ 1401-BULK) reconciling with the 6,142-row exceptions table. Today's Plan, exceptions, projection all populated. The strongest tab.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias, forecast-vs-actual series (2025-05…2026-02), lag curve (Lag 0→4, 63→75%), 13-bucket cluster comparison all render. Heatmap negatives now floored (F3.2). Modulo F4.4 cluster-table presentation.
- **Demand History** — Workbench lists 50/50 series with volume + sparkline + MoM% (large values like 520.9% are legitimate intermittent swings).
- **Item Analysis** — defaults to a representative DFU (177797 @ 1401-BULK, 28.85% acc / 71.15% WAPE / FCST 92.5 / ACTUAL 86.1), full chart + SHAP + Forecast KPIs + DQ Corrections panels.
- **Customer Map** — 23.0M cases, 98% fill, 461K lost sales, 32.5K active customers, rich filters (State now clean; Channel/Store Type still dirty per F4.5).
- **Explorer** — fast raw data across 9 domains.
- **AI Planner FVA Backtest** — run history populated (succeeded/failed, provider/DFU/rec counts).
- **Data Quality (data)** — 166 checks, 64% health, 20 domains, 32 recent issues — genuinely informative now (modulo the Last Run / Catalog split in F4.2).
- **Clusters / S&OP** — genuinely empty with honest empty states + run-pipeline / create-cycle CTAs. Not defects.

## Harness note (carried, not a product defect)
`controlTower` and `aiPlanner` routes still render Command Center content in the dump (entries 4 and 7 byte-identical to entry 0) — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries. Flagged so they aren't mistaken for additional broken tabs.

---

## Resolved since Cycle 3 (verified this cycle)
- **F3.1 (exceptions fallback) — RESOLVED (exceptions half).** `/control-tower/kpis` exceptions block now live from `fact_replenishment_exceptions` (6142 / 2465 critical / $246K). Health + fill-rate halves still zero → carried as F4.3.
- **F3.2 (heatmap negatives) — RESOLVED.** `formatHeatmapAccuracy()` floors to `<0%*` + caption. Cluster *table* still raw → carried as F4.4.
- **F3.3 (DQ never run) — RESOLVED (data half).** 166 checks ran; dashboard/issues populated. Catalog/Last-Run still broken → NEW F4.2; button wiring → carried in F4.6.
- **U3.3 (dirty State dropdown) — RESOLVED.** State now shows clean 2-letter codes. Channel/Store Type still dirty → carried as F4.5.
