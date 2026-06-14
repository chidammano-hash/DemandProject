# Demand Planner UX Findings — Cycle 1
_Persona: senior demand planner. Date: 2026-06-14. Method: live Playwright scan of 14 tabs + curl/API-log/DB confirmation of every error._

## Summary
The product is roughly half-usable from a planner's chair today. The analytics core is genuinely good — Portfolio/Accuracy, Item Analysis, Customer Map, and Explorer all render real numbers and charts. But my two most important daily landing pages are broken: the **Command Center** (morning triage) throws a 500 and shows "--" health, and the **Inventory Planning Action Feed** ("your morning starting point") shows **0 actions while 6,142 open critical exceptions exist in the database**. Both are caused by code/schema bugs and stale materialized views, not by missing data. Until those two are fixed I cannot start my day in this tool — I'd have to fall back to the raw Exceptions list.

Two distinct root-cause classes:
1. **Real code/schema bugs** (action-feed references a non-existent `created_at` column; a missing `mv_control_tower_kpis` migration).
2. **Operational state** — MVs never refreshed and several pipelines (clustering, champion backtest, DQ checks) never run, which is mostly handled gracefully but leaves key screens blank.

## Findings (prioritized)

### F1.1 — Inventory Planning Action Feed shows 0 actions despite 6,142 open critical exceptions  [SEV: P0]
- **Workflow blocked:** Morning inventory triage. The Action Feed is explicitly billed as "your morning starting point" — it is the first thing I open. It is empty and tells me "No pending actions" while the portfolio is on fire.
- **Evidence:** Tab `invPlanning` (`screens/invPlanning.png`). Header banner reads "3,152 at risk" yet the feed cards read "Total Actions 0 / Critical 0 / No pending actions". `curl /inv-planning/action-feed` → `{"actions":[],"summary":{"total":0,...}}` (HTTP 200, silently empty). Meanwhile `curl /inv-planning/exceptions` → `{"total":6142,...}` all `severity=critical, status=open` (below_ss, stockout). API log on every action-feed call:
  `action-inbox: exceptions table query failed: column "created_at" does not exist`
  followed by `action-inbox: planned_orders table query failed: current transaction is aborted` and `integrated_targets MV query failed: current transaction is aborted`.
- **Root cause (best guess):** `api/routers/inventory/inv_planning_insights.py` `get_action_feed` (Source 1 query, ~line 137-152) SELECTs `created_at` from `fact_replenishment_exceptions`. That column does not exist (table has `exception_date`, `load_ts`, `modified_ts`, `acknowledged_ts`). The `psycopg.Error` is swallowed by a `try/except psycopg.Error` that only logs a warning, AND it aborts the shared transaction so Sources 2 and 3 also silently fail. Net result: a 200 with zero actions. Triggered from `InvPlanningTab` action-feed panel via `queries` action-feed fetcher.
- **Acceptance criterion:** `/inv-planning/action-feed` returns the open critical exceptions (top-N by urgency/financial impact); the Action Feed panel shows non-zero Total/Critical counts that reconcile with the 6,142-row exceptions list; no "column does not exist" / "transaction is aborted" warnings in the API log. Each source query should run on its own statement so one failure can't zero out the others.
- **Planner impact:** This is the single most damaging defect. A planner trusting this screen would conclude there's nothing to do while thousands of SKUs are below safety stock or stocked out — directly causing missed replenishments and lost sales.

### F1.2 — Command Center health KPIs are blank and throw 500 (missing materialized view)  [SEV: P0]
- **Workflow blocked:** Morning portfolio triage / executive glance. Command Center is the default landing tab (sidebar item #1).
- **Evidence:** Tab `commandCenter` (`screens/commandCenter.png`) shows a red "Internal Server Error" toast, Portfolio Health "--", Fill Rate (3m) "--", "Portfolio looks healthy!" (false-positive) and "No trend data available." 3 console 500s. `curl /control-tower/kpis` → HTTP 500. API log: `psycopg.errors.UndefinedTable: relation "mv_control_tower_kpis" does not exist` at `api/routers/control_tower.py:70`. DB confirms `to_regclass('mv_control_tower_kpis')` is NULL — the MV was never created.
- **Root cause (best guess):** `GET /control-tower/kpis` (control_tower.py) queries `mv_control_tower_kpis`, whose DDL migration is missing/never applied. Consumed by `CommandCenterTab.tsx` via `fetchControlTowerKpis` (`queries/control-tower.ts → /control-tower/kpis`).
- **Acceptance criterion:** Command Center Portfolio Health, Fill Rate, and Critical Items render real numbers (not "--"), with no 500 in console. The `mv_control_tower_kpis` MV exists and is populated (apply the missing migration + `make refresh-mvs-tiered`).
- **Planner impact:** My default screen lies to me — it says "Portfolio looks healthy!" while it actually failed to load any health data. Worse than an honest error, because it looks reassuring.

### F1.3 — Control Tower trend / Fill-Rate / Inventory trends empty or 500 (unpopulated MVs)  [SEV: P1]
- **Workflow blocked:** Portfolio trend review (6M direction-of-travel) and fill-rate/inventory monitoring.
- **Evidence:** Command Center "Portfolio Trend (6M)" reads "No trend data available." `curl /control-tower/trend` → 200 but `{"trend":[],"warning":"Upstream materialized view not yet refreshed. Run make refresh-mvs-tiered"}`. `curl /fill-rate/trend` → 500, `curl /inventory/trend` → 500, `curl /inventory-backtest/trend` → 500. DB: 6 of 29 MVs are unpopulated (`mv_fill_rate_monthly`, `agg_inventory_monthly`, `agg_sales_weekly`, `mv_inventory_forecast_monthly`, `mv_inventory_health_score`, `mv_fairness_audit`).
- **Root cause (best guess):** MVs exist but were never refreshed (`REFRESH MATERIALIZED VIEW` never run). `/control-tower/trend` degrades gracefully (empty + warning); `/fill-rate/trend` and `/inventory/trend` do NOT — they 500 on the unpopulated `mv_fill_rate_monthly` / `agg_inventory_monthly` instead of returning an empty-with-warning payload like the trend endpoint does.
- **Acceptance criterion:** After `make refresh-mvs-tiered`, all trend endpoints return data. Independently, the fill-rate and inventory trend endpoints should degrade to the same `{data:[], warning:...}` pattern as `/control-tower/trend` instead of 500-ing when an MV is unpopulated.
- **Planner impact:** I can't see whether accuracy/fill-rate are trending up or down — trend context is core to demand review and S&OP prep.

### F1.4 — Item Analysis SHAP / DQ sub-panels 500 (6 console errors)  [SEV: P1]
- **Workflow blocked:** Single-DFU deep dive (root-causing a forecast miss before an item review).
- **Evidence:** Tab `itemAnalysis` (`screens/itemAnalysis.png`) — the main chart and Forecast KPIs (Accuracy 72.59%, WAPE 27.41%, Bias 0.07) render correctly, but a red "Internal Server Error (x2)" toast is shown and the digest logged 6 console 500s. The 500s are on the secondary panels (SHAP, DQ Corrections), not the core forecast view.
- **Root cause (best guess):** SHAP and DQ-corrections endpoints for the selected DFU likely depend on artifacts/MVs that aren't populated (SHAP values not computed; DQ checks never run — see F1.6). Triggered from `ItemAnalysisTab` SHAP/DQ sub-panels.
- **Acceptance criterion:** SHAP and DQ panels either render data or show a clean "not computed yet" empty state — no 500 toast and no console errors on a normally-loading item.
- **Planner impact:** The forecast view works, so this is friction rather than a block, but the error toast erodes trust in an otherwise-strong screen, and SHAP is exactly what I use to explain a bad forecast to the commercial team.

### F1.5 — Aggregate Accuracy heatmap shows wildly negative accuracy (-186%, -263%) for BEER and several clusters  [SEV: P1]
- **Workflow blocked:** Forecast accuracy & FVA review; deciding which segments to intervene on.
- **Evidence:** Tab `aggregateAnalysis` (no console errors — this is a data/metric finding). Accuracy Heatmap: BEER row shows -186.4% / -263.9% / -92.4% / -78.3%. Cluster comparison shows L2_4 -12.89%, L2_5 -61.08%, L2_6 -128.04% with biases up to -69.2%. These are mathematically valid for the `100 - 100*Σ|F-A|/|ΣA|` formula when forecast hugely overshoots a tiny actual, but presented as raw "Accuracy %" they're confusing and visually alarming.
- **Root cause (best guess):** Not an error — the accuracy formula goes deeply negative for low-volume / intermittent segments (BEER, sparse clusters). The UI displays unbounded negative accuracy without flooring at 0 or switching these rows to a WAPE/bias presentation.
- **Acceptance criterion:** Low-volume/intermittent rows either floor accuracy at 0% with a "low base — see WAPE" annotation, or the heatmap legend explains that negative values mean forecast >> actual on a small base. A planner shouldn't have to mentally translate "-263% accuracy."
- **Planner impact:** I'd waste a review cycle panicking over BEER before realizing it's a small-base artifact. Misleading severity ranking.

### F1.6 — Data Quality dashboard fully empty; "Run Checks" path is manual-only  [SEV: P2]
- **Workflow blocked:** Trusting upstream data before I act on a forecast/exception.
- **Evidence:** Tab `dataQuality` (`screens/` digest) — Overall Health 0%, 0 checks, "No data quality checks have been run yet", empty lineage. The real endpoints work but are empty: `curl /data-quality/dashboard` → `{"domains":[]}` (200), `/data-quality/checks` → `{"checks":[]}` (200). (The 3 "404" console errors in the digest were `/dq/*` probes from my own investigation, not the app — the tab correctly uses `/data-quality/*`.)
- **Root cause (best guess):** Genuinely empty — DQ battery never run. The screen does the right thing and shows a "HOW TO POPULATE" guide. Friction is that it requires a curl/CLI step (`POST /data-quality/run`) rather than the visible "Run Checks Now" button doing it.
- **Acceptance criterion:** The "Run Checks Now" button triggers `POST /data-quality/run` from the UI and the dashboard repopulates without leaving the app. Until checks exist, keep the current graceful empty state (it's good).
- **Planner impact:** Low daily impact, but I have no in-app signal of data trustworthiness, which matters when an exception looks wrong.

### F1.7 — Clusters tab empty + React "setState during render" warning  [SEV: P2]
- **Workflow blocked:** Segment-based analysis (clusters drive per-cluster accuracy and model selection).
- **Evidence:** Tab `clusters` — "No cluster assignments yet. Run the clustering pipeline." DB confirms `dim_sku.ml_cluster` has 0 non-null rows. Console warning: `Cannot update a component (App) while rendering a different component (ClustersTab) ... setState() call inside ClustersTab`.
- **Root cause (best guess):** Empty state is genuine (clustering pipeline never run). Separately, `ClustersTab.tsx` calls a setState/URL-state setter during render (likely `setScenarioJobParam`/`setPollingScenarioId` in render body) — a real React bug independent of the empty data.
- **Acceptance criterion:** No "setState in render" warning on mount; the setState is moved into a `useEffect`. Empty state remains acceptable until the pipeline is run.
- **Planner impact:** Low today, but the Accuracy tab already segments by cluster (L2_* buckets) — those exist from backtests, so the disconnect (clusters show in Accuracy but Clusters tab says "none") is confusing.

### F1.8 — FVA Champion stage shows "No data"; AI/Planner stages "Coming Soon"  [SEV: P2]
- **Workflow blocked:** Forecast Value Added review (proving planning lift to leadership).
- **Evidence:** Tab `fva` (`screens/fva.png`). Ladder renders Naive Seasonal 65.6% → External 70.8% (+5.2pts, 76,995 rows) correctly, but Champion = "No data", AI Adjusted / Planner Adjusted = "Coming Soon". Total Interventions 0, Estimated/Actual Impact $0. `curl /fva/waterfall` → 200 with valid stages. DB: `fact_candidate_forecast` = 0 rows, `fact_production_forecast` = 182,688 rows.
- **Root cause (best guess):** Champion stage depends on candidate-backtest data (`fact_candidate_forecast`), which is empty — the champion backtest/promotion pipeline was never run. AI/Planner stages are intentionally reserved. Not a crash.
- **Acceptance criterion:** After a champion backtest run, the Champion stage shows a measured accuracy and delta vs External. The "No data" copy is acceptable as an interim state.
- **Planner impact:** I can show External-vs-Naive lift today, but not the full ladder I'd present in an S&OP/exec review.

### F1.9 — S&OP tab has zero cycles and can only be created "via API or CLI"  [SEV: P3]
- **Workflow blocked:** Monthly S&OP cycle management.
- **Evidence:** Tab `sop` (`screens/sop.png`) — "0 active cycles", "No active S&OP cycles. Create one via the API or CLI." Empty approved plan and decision log. No errors.
- **Root cause (best guess):** Genuinely empty (no cycle seeded). The friction is there's no in-app "Create Cycle" action — it points me to the CLI.
- **Acceptance criterion:** An in-UI "New S&OP Cycle" button creates a cycle; empty state otherwise stays as-is.
- **Planner impact:** Low — S&OP is monthly, not daily. But "go use the CLI" is a poor handoff for a business user.

### F1.10 — Capture harness note: Control Tower / AI Planner (and Demand History in this run) rendered as Command Center  [SEV: P3 / not a product defect]
- **Workflow blocked:** None (investigation hygiene note).
- **Evidence:** Digest shows `demandHistory`, `controlTower`, `aiPlanner` all rendering identical Command Center text + 3 CC 500s. But `controlTower` and `aiPlanner` are NOT sidebar items (only reachable via `?tab=`), so the harness fell back to the default Command Center view — those CC 500s are F1.2, double-counted. Demand History IS a real, working tab: `curl /demand-history/matrix` → 200 with data; it appears to have been a navigation-timing artifact in this run.
- **Root cause (best guess):** Harness navigation defaults to Command Center when a `?tab=` value isn't a clickable sidebar entry.
- **Acceptance criterion:** Re-run the harness driving Control Tower / AI Planner via direct URL state to capture their true content before filing defects against them.
- **Planner impact:** None — flagged so these aren't mistaken for additional broken tabs.

## Tabs that work well (no action needed)
- **Portfolio / Aggregate Accuracy** — KPIs, forecast-vs-actual table, lag curve all render (modulo F1.5 presentation).
- **Item Analysis** — core forecast chart + KPIs are strong (modulo F1.4 side panels).
- **Customer Map** — full KPI strip (23.0M cases demand, 98% fill rate, 461K lost sales), geo map with 32,469 customers, rich filters. The standout screen.
- **Explorer** — fast, real raw data across all domains.
- **AI Planner FVA Backtest** — run history table populated (succeeded/failed runs with provider/DFU/rec counts).
