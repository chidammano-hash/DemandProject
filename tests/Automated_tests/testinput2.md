# Acceptance Test — Cycle 2 Findings (Senior Demand Planner)

Date: 2026-06-14
Reviewer: automated planner-acceptance pass (cycle 2)
Environment: UI :5173 → API :8000 → Postgres :5440, Redis :6379

## Verdict

The product remains in **broadly good shape and improved since cycle 1.** All 14 captured tabs loaded with `ok:true`, **zero console errors, and zero 4xx/5xx at the page level** (capture-dump.json). Every workflow endpoint I spot-curled returned HTTP 200.

Cycle-1 fixes verified live:
- **F1.2 resolved on the Action Feed surface** — the Inventory Planning "Financial Impact at Risk" tile now carries the explicit sublabel *"7-day lost gross margin (open exceptions) + proposed order value"* and `summary.financial_at_risk_basis` is returned by `/inv-planning/action-feed`.
- **U2.3 improvement** — Command Center now shows a distinct, well-scoped **"Order Value at Risk $246.8K"** tile fed by `recommended_order_value` (proposed replenishment order value), instead of restating the critical-count badge. Verified: `/control-tower/kpis` → `exceptions.recommended_order_value = 246761`.
- **U1.1** — Today's Plan / TodaysPlanBanner stamps the planning as-of date.

**No P0 or P1 defects this cycle.** Findings below are previously-known deferred items that still reproduce (F1.1, F1.3, F1.4) plus one genuinely-new small consistency item (F2.1). I did **not** invent issues; the captured surfaces are largely correct with honest empty/error states (S&OP 0 cycles, FVA interventions 0, Clusters no assignments, AI-FVA failed run shows "AI returned no quantity … run skipped").

---

## F2.1 — Two different "at risk" dollar figures on the two daily-triage surfaces; the Today's Plan banner is the unscoped one [SEV P2] (NEW)

**Workflow blocked:** Morning portfolio triage. A planner opens Command Center ("Order Value at Risk **$246.8K**") and Inventory Planning Today's Plan ("At Risk **$12K**") in the same session. Two headline "at risk" dollars that differ ~20× and use the bare word "At Risk" on the banner force the planner to reconcile them or distrust both.

**Evidence:**
- Command Center (screens/commandCenter.png): tile **"Order Value at Risk $246.8K"**, caption "Proposed replenishment orders". Source: `frontend/src/tabs/CommandCenterTab.tsx:476` → `recommended_order_value` (=246,761 via `/control-tower/kpis`).
- Inventory Planning Today's Plan banner (screens/invPlanning.png, top ribbon): **"At Risk $12K"** — bare label, no scope hint. Source: `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx:103-107` → `summary.financial_at_risk` (=12,112 via `/inv-planning/action-feed`, basis "7-day lost gross margin + proposed order value").
- The *Action Feed panel* on the same tab DID get the explicit sublabel in cycle 1 ("Financial Impact at Risk 7-day lost gross margin (open exceptions) + proposed order value $12.1K"). The fix simply didn't propagate up to the banner ribbon's "At Risk" chip.

**Root cause:** `TodaysPlanBanner.tsx` renders `<PriorityBadge label="At Risk" value={formatCompactCurrency(summary?.financial_at_risk)} />` with no basis annotation, while the sibling `ActionFeedPanel` and the Command Center tile both name their (different) basis. Two intentionally-different metrics ("7-day lost margin" vs "proposed order value") share the word "At Risk" with only one of three places explaining itself.

**Acceptance criterion:** The Today's Plan banner "At Risk" chip names its basis (e.g. label or tooltip "7-day lost margin at risk"), matching the Action Feed panel; OR the banner reuses the same scoped metric the Action Feed sublabel describes so the two figures on the inv-planning tab are self-consistent. A planner reading any one surface must be able to tell which dollar window/basis it represents without cross-referencing.

**Planner impact:** Moderate. Not a data bug — both numbers are individually correct — but unlabeled 20× divergence on the two primary triage landing surfaces erodes trust in the "$ at risk" column a planner ranks by.

---

## F2.2 — FVA "Forecast Value Ladder" Champion rung still reads "No data" while AI/Planner read as intentional "Coming Soon" [SEV P2] (carried from F1.1, still open)

**Workflow blocked:** Forecast Value Added review — answering "is our ML beating the ERP forecast?". The ladder stops at External (71.2%, +5.9 pts). Champion (Step 3) shows **"No data"** with no reserved treatment, sandwiched between AI Adjusted and Planner Adjusted which now show a clean **"Coming Soon / Reserved stage"** treatment. Champion reads as broken precisely because its neighbours read as deliberate.

**Evidence:**
- FVA tab (screens/fva.png): Step 1 Naive 65.3%, Step 2 External 71.2% (+5.9 pts), Step 3 Champion **"No data" / "Baseline / no prior delta"**, Step 4 AI Adjusted **"Coming Soon / Reserved stage"**, Step 5 Planner Adjusted **"Coming Soon / Reserved stage"**.
- `curl /fva/waterfall?months=12` → champion stage `{"state":"missing","accuracy_pct":null,"n_rows":0}`, top-level `"champion": null`. AI/Planner stages return `state:"planned"` (which the UI renders as "Coming Soon"); Champion returns `state:"missing"` (rendered "No data").
- Champion data exists: `fact_production_forecast` = 182,688 rows; the Aggregate Analysis tab simultaneously renders per-model accuracy (lgbm_cluster, mstl, nbeats…). So the two tabs disagree on whether champion accuracy is knowable.

**Root cause:** `api/routers/forecasting/fva.py` `fva_waterfall()` computes every stage's accuracy from `fact_external_forecast_monthly`, which contains only `model_id='external'` — no champion rows. `models.get("champion")` is always None → `_build_stage()` emits `state="missing"`. The cheap cycle-1 acceptance alternative (relabel Step 3 to a reserved "planned" state like AI/Planner so it stops reading as broken) was applied to AI/Planner but **not** to Champion.

**Acceptance criterion:** Either (a) the champion stage is computed from the champion forecast source (`fact_production_forecast` / backtest lag archive) so `state="actual"`, non-null `accuracy_pct`, `n_rows>0`; OR (b) Champion's stage state is changed to the same reserved/"planned" treatment as AI/Planner so the ladder presents three consistent reserved stages instead of one that reads as a failure. (Option (b) is a one-line change in `fva.py` and fully resolves the visual-trust issue.)

**Planner impact:** The FVA tab's core question ("did our modeling add value over the ERP forecast?") is unanswerable, and the inconsistent Step 3 treatment makes the screen look half-broken.

---

## F2.3 — Data Quality "Total Checks 166" vs catalog 83 denominator inflation persists [SEV P3] (carried from F1.3, partially defensible)

**Workflow blocked:** Data-quality monitoring / trust. Header tile "Total Checks **166**" is ~2× the catalog's "Check Catalog (**83**)".

**Evidence (screens/dataQuality.png + API):**
- Header tiles: Total **166**, Passed **116**, Failed **0**, Warnings **26**.
- `curl /data-quality/checks` → 83 checks; `last_status` breakdown `{pass:58, fail:13, warn:3, skip:9}`; severity `{warning:48, critical:28, info:7}`.
- The 13 `last_status=fail` rows are **all warning/info severity** (none critical) — e.g. `referential_integrity_forecast_to_sku` (971), `cross_column_inventory_snapshot_date_not_future` (796,460). So **"Failed 0" is now defensible** under the documented severity-aware rule (warning-severity fails roll into the amber Warnings tile, and the catalog badges them WARNING). The remaining mismatch is purely the **166 vs 83 denominator**: `/data-quality/dashboard` `domains[]` double-counts cross-domain referential pairs (`forecast_to_sku`, `inventory_to_item`, `sku_to_item`, …) that the catalog counts once.

**Root cause:** `frontend/src/tabs/DataQualityTab.tsx` derives header tiles by summing `dashboard.domains[].total/passed/...` (which enumerates per-domain + per-referential-pair entries → ~2× inflation) while the catalog table is fed by `/data-quality/checks` (83 distinct checks).

**Acceptance criterion:** The "Total Checks" tile equals the catalog count (83), OR both surfaces use the same denominator with a tooltip explaining the per-domain referential-pair expansion. "166 total" over an "83-check catalog" must not coexist unexplained.

**Planner impact:** Low — DQ is an observability surface, not a daily decision path — but the 2× mismatch invites "is this dashboard even right?" doubt.

---

## F2.4 — Cold `/customer-analytics/affinity` still ~11.6s; below-fold CA panels captured on "Loading…" [SEV P3] (carried from F1.4)

**Workflow blocked:** Customer/channel drill-down. On a cold/evicted cache the bottom CA panels sit on "Loading…".

**Evidence:**
- screens/customerAnalytics.png + digest: top-of-fold (map, concentration, KPIs) renders fine; **8× "Loading…"** at the bottom.
- `curl -w time` `/customer-analytics/affinity` = **11.6s** (HTTP 200, 12.9KB) this cycle — still slow. Sibling `/customer-analytics/channel-mix` = 0.64s. So the affinity self-join is the laggard gating the below-fold panels.

**Root cause:** The affinity endpoint (`api/routers/intelligence/customer_analytics/…`) runs an expensive co-occurrence self-join with no warm cache / no pre-aggregating MV on first hit. LazyPanel defers it below the fold; combined with the 11.6s cold query the capture snapshots it pre-resolution.

**Acceptance criterion:** Cold `GET /customer-analytics/affinity` returns < ~3s (index/pre-aggregate or MV), OR confirm it's acceptable to leave behind LazyPanel + Redis warm-up and treat as won't-fix. (It resolves correctly once cached — not a correctness defect.)

**Planner impact:** Minor first-visit friction only.

---

## Checked and OK (no finding)

- **Command Center / Control Tower / AI Planner** — render correctly; controlTower & aiPlanner are intentional redirects to commandCenter (`useUrlState.ts` TAB_REDIRECTS). CC tiles (Health 58/100, Open Exceptions 6141/2464 critical, Fill Rate 98.2%, Order Value at Risk $246.8K) all reconcile to `/control-tower/kpis`.
- **Aggregate Analysis** — KPIs (Accuracy 73.9%, WAPE 26.1%, Bias 6.6%), per-model heatmap, per-cluster comparison, lag-horizon curve all populate; `BEER <0%*` and `L2_4/5/6 <0%*` cells are honestly annotated (tiny actual base, WAPE>100%). Real over-forecast signal, correctly surfaced.
- **Demand History** — 50 series with plausible volumes; outlier MoM% (e.g. 520.9%*) flagged with `*`.
- **Item Analysis / Explorer** — real data; Item 15502 @ 1401-BULK external 12-mo accuracy 65.82%, full attribute grid renders.
- **Customer Analytics top-of-fold** — Total Demand 23.0M cases, Fill 98.0%, Lost Sales 461K, 32,469 customers; map + concentration treemap render.
- **AI Planner FVA Backtest** — honest run history (succeeded/failed with reason; failed run shows "AI returned no quantity for 2 recommendations — run skipped"). Two "succeeded … RECS 0" rows are legitimate empty backtests, not errors.
- **S&OP (0 cycles), FVA interventions (0), Clusters (no assignments)** — correct honest empty states.
- All exceptions at single location `1401-BULK` — data reality (all inventory & forecast at that one warehouse), not a join bug.

newActionableCount (NEW unresolved P0/P1/P2) = 1 (F2.1)
