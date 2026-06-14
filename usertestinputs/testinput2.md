# Demand Planner UX Findings — Cycle 2
_Persona: senior demand planner. Date: 2026-06-14. Method: live capture digest + curl/DB/code confirmation of every error. Branch: restructure._

## Summary
Cycle 1's two worst defects are **half-fixed**: the Inventory Planning Action Feed (F1.1) now correctly shows 20 critical actions ($3.6K at risk) reconciling with the 6,142-row exceptions table — fully resolved. The fill-rate / inventory / inv-backtest trend endpoints (F1.3) no longer 500 — resolved. Item Analysis (F1.4) renders with no error toasts this cycle — resolved.

But my **default landing page, the Command Center, still lies to me** (F2.1). The 500 is gone, but it was replaced by a 200 full of zeros: it shows "Portfolio Health 0/100", "Open Exceptions 0", and a green "Portfolio looks healthy!" checkmark — while 6,142 open critical exceptions and 3,152 at-risk SKUs exist (and the Inventory tab shows them fine). The root cause is `mv_control_tower_kpis` (and 6 sibling MVs) were never refreshed; the endpoint degrades to an all-zero payload with a `warning` the UI silently ignores. This is the single most dangerous remaining defect — a reassuring false-positive on the first screen a planner opens.

One genuinely **new code bug** surfaced: the Data Quality tab's Pipeline Lineage panel calls `/data-quality/lineage/batches` but the router exposes `/data-quality/batches` (no `/lineage` segment) — a hard 404 path-contract mismatch that will never load even after the ETL runs. Cycle 1 wrongly dismissed these 404s as "my own probe traffic"; this fresh automated capture shows they are the app's own calls.

Two presentation issues from cycle 1 remain deferred (negative-accuracy heatmap F1.5; DQ run button F1.6).

---

## Findings (prioritized)

### F2.1 — Command Center shows "Portfolio looks healthy!" + 0/100 health while 6,142 critical exceptions exist  [SEV: P0]  (regressed-presentation of F1.2)
- **Workflow blocked:** Morning portfolio triage. Command Center is the default landing tab (sidebar #1). It is the first thing I see each morning.
- **Evidence:** Tab `commandCenter` (`screens/commandCenter.png`). Tiles: Portfolio Health **0/100** (red), Open Exceptions **0**, Fill Rate (3m) **--**, Critical Items **0**, then a green check and **"Portfolio looks healthy! No open exceptions matching your filters."** No error toast this cycle. Meanwhile `curl /inv-planning/exceptions` → `{"total":6142}` all open (2,465 critical / 1,715 high / 1,962 low), and the Inventory Planning tab Action Feed correctly shows 20 critical actions. `curl /control-tower/kpis` → HTTP **200** but every field 0/null with `"warning":"mv_control_tower_kpis not yet refreshed..."`. DB: `mv_control_tower_kpis` exists but `ispopulated=f`; 7 MVs unpopulated (`mv_control_tower_kpis`, `mv_fill_rate_monthly`, `agg_inventory_monthly`, `agg_sales_weekly`, `mv_inventory_forecast_monthly`, `mv_inventory_health_score`, `mv_fairness_audit`).
- **Root cause:** Two layers. (1) Operational: the MVs were never refreshed (base data exists — `fact_inventory_snapshot` has 4.9M rows — so a tiered refresh would populate them; a plain `REFRESH MATERIALIZED VIEW mv_control_tower_kpis` fails because dependents like `mv_fill_rate_monthly` are also unpopulated, so `make refresh-mvs-tiered` is required). (2) Product: `api/routers/operations/control_tower.py` `get_control_tower_kpis` (lines 72-78) catches `ObjectNotInPrerequisiteState` and returns an all-zero `empty_payload` + `warning`. The Command Center UI (`frontend/src/tabs/CommandCenterTab.tsx`, lines 283-285, 390, 485) renders `ex?.open_exceptions_total ?? 0` and the "Portfolio looks healthy!" empty state **without checking `kpisQ.data?.warning`**, so a "MV not refreshed" state is indistinguishable from a genuinely healthy portfolio. The exception list itself is also fed only by `/ai-insights` and `/storyboard/exceptions` (both empty / 404 — the `ai_insights` table is empty and there is no storyboard table), never by the live `fact_replenishment_exceptions` the Action Feed uses successfully.
- **Acceptance criterion:** After `make refresh-mvs-tiered`, Command Center Portfolio Health, Fill Rate, and Critical Items show real numbers reconciling with `/inv-planning/exceptions` (non-zero critical count). Independently and more importantly: when `/control-tower/kpis` returns a `warning` (MV unrefreshed), the UI must NOT show "Portfolio looks healthy!" — it must show a degraded/"data unavailable — refresh required" banner, OR the KPI tiles should fall back to a live count from `fact_replenishment_exceptions` (the same source the Action Feed reads) so the open-exception count is never silently zero while exceptions exist.
- **Planner impact:** Catastrophic-trust defect. My default screen tells me everything is fine while thousands of SKUs are stocked out or below safety stock. A planner who trusts it skips the entire morning triage. Worse than cycle 1's 500 because there is now no error indicator at all.

### F2.2 — Data Quality Pipeline Lineage panel 404s: frontend calls `/data-quality/lineage/batches`, router serves `/data-quality/batches`  [SEV: P1]  (NEW — mis-attributed in cycle 1)
- **Workflow blocked:** Verifying upstream data lineage / pipeline batch health before trusting a forecast or exception.
- **Evidence:** Tab `dataQuality` is the ONLY tab with console errors this cycle — **3× "Failed to load resource: 404"** (capture-dump.json entry 9). Probing the frontend's actual paths: `/data-quality/dashboard`, `/checks`, `/history`, `/corrections`, `/corrections/summary`, `/fix/preview` all 200; only the lineage family 404s. `curl /data-quality/lineage/batches` → **404**, but `curl /data-quality/batches` → **200**. The 3 errors map to the 3 lineage fan-out calls fired on mount (batches list, row, corrections).
- **Root cause:** Path-contract mismatch. `api/routers/platform/medallion.py` mounts `APIRouter(prefix="/data-quality")` with routes `@router.get("/batches")` and `/batches/{batch_id}` (real paths `/data-quality/batches`). But `frontend/src/api/queries/platform.ts` calls `/data-quality/lineage/batches` (line 123), `/data-quality/lineage/batches/{id}` (127), `/data-quality/lineage/row/...` (130), `/data-quality/lineage/corrections` (138) — an extra `/lineage/` segment the backend never exposes. Either the router prefix should be `/data-quality/lineage` or the frontend should drop `/lineage`.
- **Acceptance criterion:** The Pipeline Lineage panel's batch/row/corrections calls return 200 (path aligned on both sides); no 404s in the DataQuality console. An `httpx` test asserts `GET /data-quality/lineage/batches` (the path the frontend uses) returns 200, OR the frontend query module is repointed and an existing test covers `/data-quality/batches`.
- **Planner impact:** The lineage panel can never load even after `make load-all` — the "no pipeline batches yet" empty state is permanent and misleading. Cycle 1 dismissed these 404s as investigation noise; they are a real, persistent app bug.

### F2.3 — Negative-accuracy heatmap still shows raw "-186%", "-263%", "-128%" with no flooring or annotation  [SEV: P1]  (carried from F1.5, still unaddressed)
- **Workflow blocked:** Forecast accuracy & FVA review; ranking which segments to intervene on.
- **Evidence:** Tab `aggregateAnalysis` (digest, no console errors). Accuracy Heatmap BEER row: **-186.4% / -263.9% / -92.4% / -78.3%**. Cluster comparison: L2_4 -12.89%, L2_5 -61.08%, **L2_6 -128.04%** (WAPE 228%), L2_99 -6.17%, with biases to -69.2%. The portfolio Accuracy KPI (73.9%) is healthy, so these are low-base/intermittent artifacts of the `100 - 100·Σ|F-A|/|ΣA|` formula, not errors. No flooring or legend annotation was added since cycle 1.
- **Root cause:** Presentation, not data. The heatmap renders unbounded negative accuracy without flooring at 0% or annotating low-base rows. `frontend/src/tabs/aggregate-analysis/` (heatmap cell renderer) shows raw `accuracy_pct`.
- **Acceptance criterion:** Low-base/intermittent rows either floor accuracy at 0% with a "low base — see WAPE" marker, or the legend explains that negative = forecast >> actual on a small base. A planner should not have to mentally translate "-263% accuracy."
- **Planner impact:** I waste a review cycle alarmed over BEER / L2_6 before realizing it's a small-base artifact. Misleads severity ranking in exactly the screen I use for intervention decisions.

### F2.4 — `/fva/waterfall` ai_adjusted promotion: 3 FVA tests fail with IndexError (test-mock vs SQL column-count mismatch)  [SEV: P2]  (carried, code-quality not workflow)
- **Workflow blocked:** None for the live planner — `curl /fva/waterfall` → 200 with valid stages; the FVA tab renders Naive 65.6% → External 70.8% correctly. This is a CI/quality defect.
- **Evidence:** `~/.local/bin/uv run pytest tests/api/test_fva.py -q` → **3 failed, 4 passed**, all `IndexError: tuple index out of range` at `api/routers/forecasting/fva.py:130` (`int(ai_row[2] or 0)`). The SQL (lines 110-119) selects 3 columns (`run_id`, `ai_wape_pct`, `COALESCE(SUM(o.n_dfus),0)`), so production is safe; the test mock supplies only a 2-tuple, so `ai_row[2]` overruns.
- **Root cause:** Test fixture in `tests/api/test_fva.py` mocks the ai_fva query with a 2-element tuple; the endpoint indexes `[2]`. Either the mock must return a 3-tuple, or the endpoint should `len()`-guard the optional ai_adjusted promotion.
- **Acceptance criterion:** `pytest tests/api/test_fva.py -q` passes; the mock returns a 3-column row matching the SELECT (and/or `fva.py` guards `len(ai_row) > 2` before indexing). No assertion weakening.
- **Planner impact:** None directly, but a red FVA test suite erodes confidence in the FVA numbers I'd present to leadership, and masks future regressions in the ai_adjusted promotion path.

### F2.5 — Data Quality "Run Checks Now" button is decorative; population still requires curl/CLI  [SEV: P2]  (carried from F1.6)
- **Workflow blocked:** Trusting upstream data quality in-app.
- **Evidence:** Tab `dataQuality` — Overall Health 0%, 0 checks, "No data quality checks have been run yet", and a "HOW TO POPULATE" guide that tells me to `curl -X POST http://localhost:8000/dq/run` or run `uv run python scripts/dq_run_checks.py`. The visible "Run Checks Now" button does not trigger the run from the UI. `curl /data-quality/dashboard` → `{"domains":[]}` (200) — genuinely empty, not broken. (Note: the guide's `/dq/run` path is also stale — the live endpoint is `POST /data-quality/run`.)
- **Root cause:** Genuinely empty (DQ battery never run) + the "Run Checks Now" button is not wired to `POST /data-quality/run`. The in-card instructions reference the wrong `/dq/run` prefix.
- **Acceptance criterion:** "Run Checks Now" triggers `POST /data-quality/run` from the UI and the dashboard repopulates without leaving the app; the empty-state instructions cite the correct `/data-quality/run` path. Graceful empty state stays until checks exist.
- **Planner impact:** Low daily impact, but I have no in-app signal of data trustworthiness, and the one button that should fix it does nothing.

### F2.6 — Item Analysis selected DFU (10205 @ 1401-BULK) shows 2.72% accuracy / 97.28% WAPE — likely a sparse/near-zero default item  [SEV: P3]  (NEW, presentation/UX)
- **Workflow blocked:** Single-DFU deep dive — the default item the tab opens on looks broken.
- **Evidence:** Tab `itemAnalysis` (`screens/itemAnalysis.png`, no console errors — chart and KPIs render). Forecast KPIs for the default item 10205: **Accuracy 2.72%, WAPE 97.28%, Bias -0.17, FCST 8.9, ACTUAL 10.8** over 12mo. Tiny volumes (single-digit units) → same low-base artifact as F2.3, presented as a headline "2.72% ACCURACY."
- **Root cause:** The tab defaults to a low-volume item; the per-item accuracy uses the same unbounded formula. Not a code error — `/item-analysis` returns 200. UX: a near-zero-volume default item makes the strongest analytics screen look broken on open.
- **Acceptance criterion:** Either default Item Analysis to a representative high-volume DFU, or annotate sub-threshold-volume items with a "low base — accuracy unreliable" badge consistent with the F2.3 heatmap fix.
- **Planner impact:** Minor — erodes first-impression trust in an otherwise-strong screen. A planner who lands here first might think the forecast engine is failing.

### F2.7 — Capture harness renders Control Tower / AI Planner / Demand History as Command Center  [SEV: P3 / not a product defect]  (carried from F1.10)
- **Workflow blocked:** None — investigation hygiene.
- **Evidence:** Digest shows `controlTower`, `aiPlanner`, and `demandHistory` all rendering identical Command Center text. But `App.tsx` (lines 249-310) DOES render distinct panels for all three, and `curl /demand-history/matrix` → 200 with real data. Control Tower / AI Planner are not sidebar-clickable entries, so the harness's `?tab=` navigation falls back to the default Command Center; Demand History is a real working tab caught mid-navigation.
- **Root cause:** Harness navigation defaults to Command Center when a `?tab=` value isn't a clickable sidebar item. Not a product bug.
- **Acceptance criterion:** Re-run the harness driving Control Tower / AI Planner via direct `setActiveTab` (or add them to the nav) so their true content is captured before filing defects against them.
- **Planner impact:** None — flagged so these aren't mistaken for additional broken tabs.

---

## Resolved since Cycle 1 (verified this cycle)
- **F1.1 (P0) RESOLVED** — Inventory Planning Action Feed shows 20 critical actions ($3.6K at risk) reconciling with the 6,142 exceptions. No "created_at"/transaction-aborted warnings.
- **F1.3 (P1) RESOLVED** — `/fill-rate/trend`, `/inventory/trend`, `/inventory-backtest/trend` all return 200 (graceful degradation; no more 500s).
- **F1.4 (P1) RESOLVED** — Item Analysis renders with no "Internal Server Error" toast and zero console errors this cycle.
- **F1.2 (P0) PARTIAL** — the `mv_control_tower_kpis` migration was applied and the 500 is gone, but the screen now silently shows a false-positive "healthy" state instead of an honest one → re-filed as **F2.1**.

## Tabs working well (no action)
- **Portfolio / Aggregate Accuracy** — KPIs (73.9% acc, 26.1% WAPE, 6.6% bias), forecast-vs-actual, lag curve, cluster comparison all render (modulo F2.3 presentation).
- **Inventory Planning** — Action Feed, exceptions, Today's Plan all populated and reconciling. Strong.
- **Customer Map** — 23.0M cases, 98% fill, 461K lost sales, 32,469 customers on the geo map, rich filters. Standout screen.
- **Explorer** — fast raw data across all 9 domains.
- **AI Planner FVA Backtest** — run history populated (succeeded/failed runs with provider/DFU/rec counts).
