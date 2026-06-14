# Demand Planner UX Findings — Cycle 3
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-3 live capture digest + dump (14 tabs, 0 console errors) + curl/DB/code confirmation. Branch: restructure._

## Summary
The product is in good shape this cycle. **All 14 tabs loaded with ZERO console errors** (capture-dump). Three cycle-2 items are verified RESOLVED:
- **F2.2 (lineage 404s) RESOLVED** — `platform.ts` repointed to `/data-quality/batches` (no `/lineage/` segment); the Pipeline Lineage panel now renders Batches #195–#214 with zero console errors. (The old `/data-quality/lineage/batches` path still 404s, but the app no longer calls it.)
- **F2.4 (FVA tests) RESOLVED** — `pytest tests/api/test_fva.py -q` → **8 passed**.
- **F2.6 (Item Analysis default) RESOLVED** — the tab now opens on a representative item (186639 @ 1401-BULK: **87.86% accuracy, 12.14% WAPE, FCST 5.1K / ACTUAL 5.2K**) instead of a near-zero-volume artifact item.

The cycle-2 P0 (F2.1) is **half-fixed**: the Command Center UI now correctly shows an honest amber **"Portfolio health data unavailable"** banner + **"Exception data unavailable"** instead of the dangerous false-positive "Portfolio looks healthy!". That is a real improvement and removes the trust hazard. **But the underlying workflow is still blocked** — the 7 KPI/health MVs remain unpopulated (`ispopulated=f`), so my default landing tab still shows Portfolio Health 0/100, Open Exceptions 0, Fill Rate --, Critical Items 0, and "No trend data available", while 6,142 open exceptions (2,465 critical) sit in `fact_replenishment_exceptions` and render perfectly in the Inventory Planning Action Feed. Morning portfolio triage on the Command Center is still not possible. This is the single remaining high-value item; the rest are carried presentation/UX issues.

No genuinely-new product defects surfaced this cycle.

---

## Findings (prioritized)

### F3.1 — Command Center morning triage still blocked: KPIs show all-zeros (MVs unpopulated) with no live fallback to `fact_replenishment_exceptions`  [SEV: P1]  (data-half of F2.1; UI false-positive already fixed)
- **Workflow blocked:** Morning portfolio triage on the default landing tab (sidebar #1, also rendered by the Control Tower and AI Planner routes).
- **Evidence:** Tab `commandCenter` (`screens/commandCenter.png`, capture-dump entry 0, **0 console errors**). Tiles: Portfolio Health **0/100**, Open Exceptions **0**, Fill Rate (3m) **--**, Critical Items **0**, Portfolio Trend "No trend data available". Honest banners now present ("Portfolio health data unavailable… Refresh the analytics views"; "Exception data unavailable… this feed cannot be trusted to be empty"). Live confirmation: `curl /control-tower/kpis` → 200 with every field 0/null and `"warning":"mv_control_tower_kpis not yet refreshed…"`. `curl /inv-planning/exceptions` → `{"total":6142}` all open. DB: `SELECT count(*) FILTER (WHERE status='open') FROM fact_replenishment_exceptions` → **6142**; `pg_matviews` shows `mv_control_tower_kpis`, `mv_fill_rate_monthly`, `agg_inventory_monthly`, `mv_inventory_health_score`, `mv_fairness_audit`, `agg_sales_weekly`, `mv_inventory_forecast_monthly` all `ispopulated=f`.
- **Root cause:** Two layers. (1) Operational/data: the 7 MVs were never refreshed; `make refresh-mvs-tiered` would populate them from base data (4.9M `fact_inventory_snapshot` rows). (2) Product: `api/routers/operations/control_tower.py` `get_control_tower_kpis` (lines ~70–110) reads ONLY `mv_control_tower_kpis`; on `ObjectNotInPrerequisiteState` it degrades to an all-zero `empty_payload` + `warning`. There is no fallback to the same `fact_replenishment_exceptions` table the Action Feed (`/inv-planning/...`) reads successfully, so the open/critical exception counts are zero even though the live data exists and is one query away.
- **Acceptance criterion:** Either (a) after `make refresh-mvs-tiered`, Command Center Portfolio Health / Fill Rate / Critical Items show non-zero numbers reconciling with `/inv-planning/exceptions` (critical = 2,465); OR — more robustly and testably — (b) when `mv_control_tower_kpis` is stale, `/control-tower/kpis` populates `exceptions.open_exceptions_total` / `critical_exceptions` / `high_exceptions` from a live `COUNT(*)` over `fact_replenishment_exceptions WHERE status='open'` (grouped by severity), so the planner's open-exception count is never silently zero while exceptions exist. An httpx test asserts that with the MV unrefreshed, `open_exceptions_total > 0` and equals the table count.
- **Planner impact:** My default screen is still un-actionable for the morning triage it exists for. The cycle-2 banner means I'm no longer *lied to* (big improvement — downgraded from P0 to P1), but I still cannot triage from the Command Center and must jump to the Inventory Planning tab to see the 20 critical actions. A planner who relies on the home screen sees a wall of zeros.

### F3.2 — Negative-accuracy heatmap & cluster comparison still render raw "-263.9%", "-128.04%" with no flooring or low-base annotation  [SEV: P2]  (carried from F2.3 / F1.5, still unaddressed)
- **Workflow blocked:** Forecast accuracy & FVA review; ranking which segments/clusters to intervene on.
- **Evidence:** Tab `aggregateAnalysis` (capture-dump entry 1, 0 console errors). Accuracy Heatmap BEER row: **-186.4% / -263.9% / -92.4% / -78.3%** (all other categories healthy 61–80%). Cluster comparison: L2_4 **-12.89%** (WAPE 112.89%), L2_5 **-61.08%**, L2_6 **-128.04%** (WAPE 228.04%), L2_99 **-6.17%**, with biases to ⚠-69.2%. Portfolio Accuracy KPI is a healthy 73.9%, confirming these are low-base/intermittent artifacts of `100 − 100·Σ|F−A|/|ΣA|`, not data errors.
- **Root cause:** Presentation. The heatmap cell renderer and the cluster-comparison table in `frontend/src/tabs/aggregate-analysis/` (`aggregateShared.ts` formats accuracy; grep finds no `floor`/`Math.max(0,…)`/`low base` handling) show unbounded negative accuracy with no flooring or "low base — see WAPE" marker.
- **Acceptance criterion:** Low-base/intermittent rows either floor displayed accuracy at 0% with a "low base — see WAPE" badge, or the heatmap legend explains that negative = forecast ≫ actual on a small base. A unit/snapshot test covers a row with `accuracy_pct < 0` rendering the floored/annotated form.
- **Planner impact:** I waste a review cycle alarmed over BEER / L2_6 before realizing they're small-base artifacts, in exactly the screen I use for intervention ranking. Carried from two prior cycles; design decision still pending.

### F3.3 — Data Quality "Run Checks Now" button still decorative; in-card guide cites stale `/dq/run` path  [SEV: P2]  (carried from F2.5 / F1.6)
- **Workflow blocked:** Trusting upstream data quality from inside the app.
- **Evidence:** Tab `dataQuality` (capture-dump entry 9, 0 console errors). Overall Health **0%**, 0 checks, "No data quality checks have been run yet", a visible **"Run Checks Now"** button, and a "HOW TO POPULATE" card instructing `curl -X POST http://localhost:8000/dq/run` and `uv run python scripts/dq_run_checks.py`. `curl /data-quality/dashboard` → `{"domains":[]}` (200, genuinely empty). The live endpoint is `POST /data-quality/run`, not `/dq/run` — the card copy is stale, and the button does not trigger the run from the UI.
- **Root cause:** Genuinely-empty data state (DQ battery never run) + "Run Checks Now" not wired to `POST /data-quality/run` + stale `/dq/run` path in the empty-state instructions.
- **Acceptance criterion:** "Run Checks Now" issues `POST /data-quality/run` and the dashboard repopulates without leaving the app; the empty-state instructions cite `/data-quality/run`. Graceful empty state persists until checks exist.
- **Planner impact:** Low daily impact, but I have no in-app signal of data trustworthiness and the one button that should fix it does nothing — and the instructions point at a 404 path.

### F3.4 — FVA "Champion" stage shows "No data" because no backtest has been promoted (genuinely-empty, not broken)  [SEV: P3]  (carried from F1.8)
- **Workflow blocked:** Forecast Value Ladder completeness for leadership-facing FVA review.
- **Evidence:** Tab `fva` (capture-dump entry 6, 0 console errors). Ladder renders Naive Seasonal **65.6%** → External **70.8% (+5.2 pts, 76,995 rows)** correctly, then **Champion: "No data"**, AI Adjusted "Coming Soon", Planner Adjusted "Coming Soon". `curl /fva/waterfall` → 200 with `champion.state="missing", accuracy_pct=null`. DB: `fact_candidate_forecast` count = **0**; no `fact_backtest_forecast` table — i.e. no champion has been promoted via `POST /backtest-management/{model_id}/promote`.
- **Root cause:** Genuinely-empty data state — the champion stage has no measured backtest to draw from. The endpoint and UI handle it correctly (honest "No data" / "missing"). Not a code defect.
- **Acceptance criterion:** After a backtest run is promoted, the Champion stage shows a real accuracy and delta vs External. No code change required unless we want an explicit "run a backtest to populate Champion" CTA in the missing state (nice-to-have).
- **Planner impact:** Minor — the ladder is incomplete for an executive readout, but the two populated stages are correct and the empty stage is honest.

---

## Resolved since Cycle 2 (verified this cycle)
- **F2.2 (P1) RESOLVED** — DataQuality lineage repointed in `frontend/src/api/queries/platform.ts` to `/data-quality/batches`/`/corrections` (lines 123/127/138); lineage panel renders Batches #195–#214; **0 console errors** on the tab (was the only tab with errors last cycle).
- **F2.4 (P2) RESOLVED** — `tests/api/test_fva.py` → **8 passed** (ai_adjusted `len()`-guard landed).
- **F2.6 (P3) RESOLVED** — Item Analysis defaults to representative item 186639 (87.86% accuracy) instead of a single-digit-volume artifact item.
- **F2.1 (P0) → DOWNGRADED to P1 (F3.1)** — the dangerous false-positive "Portfolio looks healthy!" is gone; the Command Center now shows honest "data unavailable" banners. The remaining gap is purely the un-actionable all-zero KPI state (MVs unpopulated + no live fallback).

## Tabs working well (no action)
- **Inventory Planning** — Action Feed shows 20 critical actions ($3.6K at risk) reconciling with the 6,142-row exceptions table; Today's Plan, exceptions, projection all populated. Strong.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias, forecast-vs-actual series, lag curve, 13-bucket cluster comparison all render (modulo F3.2 presentation).
- **Demand History** — Workbench lists 50 series with volume + sparkline + MoM% (large MoM% values like 520.9% are legitimate intermittent-series swings, correctly labeled). Matrix endpoint returns real data.
- **Customer Map** — 23.0M cases, 98% fill, 461K lost sales, 32,469 customers on the geo map, rich state/channel/store-type filters.
- **Explorer** — fast raw data across 9 domains.
- **AI Planner FVA Backtest** — run history populated (succeeded/failed runs, provider/DFU/rec counts).
- **Clusters / S&OP** — genuinely empty with honest empty states + run-pipeline / create-cycle CTAs (0 cluster_assignment rows; 0 S&OP cycles). Not defects.

## Note on the capture harness (carried, not a product defect)
`controlTower` and `aiPlanner` routes still render Command Center content in the dump (entries 4 and 7 are byte-identical to entry 0). Per cycle-2 F2.7 this is the harness's `?tab=` navigation falling back to the default tab for non-clickable sidebar entries — not a product bug. Flagged so these aren't mistaken for additional broken tabs.
