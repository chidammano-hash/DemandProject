# Acceptance Test — Cycle 1 Findings (Senior Demand Planner)

Date: 2026-06-14
Reviewer: automated planner-acceptance pass
Environment: UI :5173 → API :8000 → Postgres :5440, Redis :6379

## Verdict

The product is in **broadly good shape**. All 14 captured tabs loaded with **zero console errors and zero 4xx/5xx** in the capture dump. Every endpoint I spot-curled returned HTTP 200. The major daily-workflow tabs (Command Center triage, Aggregate accuracy, Inventory Planning action feed, Demand History, Item Analysis, Customer Analytics) render real, plausible data with honest empty states where data genuinely doesn't exist yet (S&OP = 0 cycles, FVA interventions = 0, Clusters = no assignments).

No P0 or P1 defects found this cycle. Findings below are correctness/clarity items (P2–P3) that would erode a planner's trust in two specific numbers (FVA Champion lift, "$ at risk") and two cosmetic mismatches.

**Note (NOT a bug):** The capture lists `controlTower` and `aiPlanner` tabs rendering identical Command Center content. This is **intentional** — `frontend/src/hooks/useUrlState.ts` `TAB_REDIRECTS` retires those keys to `commandCenter`. The harness is exercising legacy URL params. Do not "fix" this.

---

## F1.1 — FVA "Forecast Value Ladder": Champion stage is structurally dead [SEV P2]

**Workflow blocked:** Forecast Value Added review. The ladder's whole purpose is to show lift Baseline → External → Champion → AI → Planner. The Champion rung — the centerpiece of "did our modeling add value over the ERP forecast?" — never populates, so the ladder stops at External (+5.9 pts) and the planner cannot see champion lift even though champion forecasts exist.

**Evidence:**
- FVA tab (screens/fva.png): "Champion … STEP 3 … No data".
- `curl /fva/waterfall?months=12` → champion stage `{"state":"missing","accuracy_pct":null,"n_rows":0}`, and top-level `"champion": null`.
- Champion data DOES exist: `fact_production_forecast` = 182,688 rows; `data/champion/` has promoted winners. The Aggregate Analysis tab simultaneously renders per-model accuracy (lgbm_cluster, mstl, nbeats, …) from the backtest lag archive — so the two tabs disagree on whether champion accuracy is knowable.

**Root cause:** `api/routers/forecasting/fva.py` `fva_waterfall()` computes accuracy for ALL stages (including `model_id="champion"`) from a single table: `dfu_filter = "FROM fact_external_forecast_monthly f …"`. That table contains only `model_id='external'` (confirmed: `SELECT model_id, count(*) FROM fact_external_forecast_monthly` → `external|138931`, no champion rows). Champion forecasts live in `fact_production_forecast` / the backtest lag archive, which the waterfall query never reads. So `models.get("champion")` is always None → `_build_stage(...)` emits state="missing".

**Acceptance criterion:** With champion forecasts present in `fact_production_forecast`, `GET /fva/waterfall?months=12` returns a `champion` stage with `state="actual"`, non-null `accuracy_pct`, and `n_rows > 0`; and the FVA tab Step 3 shows a percentage and a "+X.X pts vs prior" delta instead of "No data". (Or, if champion-in-ladder is deliberately deferred, relabel Step 3 as a reserved "Coming Soon" stage like AI/Planner, so it doesn't read as broken.)

**Planner impact:** Without the Champion rung the FVA tab can't answer "is our ML beating the ERP forecast?" — the single most important question this screen exists to answer.

---

## F1.2 — "Financial Impact at Risk" is 7-day lost-*margin* only, but labeled as total $ at risk [SEV P2]

**Workflow blocked:** Morning triage prioritization by financial impact (Command Center + Inventory Planning Action Feed). A planner ranks and escalates by dollars. The dollar figures are implausibly tiny and silently scoped, so the planner will either deprioritize real stockouts or stop trusting the column.

**Evidence:**
- Inventory Planning (screens/invPlanning.png): "Financial Impact at Risk **$12.1K**" across **6,214 actions / 2,537 critical**. Top critical stockout = **$572**.
- `curl /inv-planning/action-feed` → `summary.financial_at_risk = 12112.49`; top item financial_impact = 571.98.
- DB: `SELECT avg, max, sum FROM fact_replenishment_exceptions WHERE status='open'` → avg **$1.59**, max **$571.98**, sum **$9,765** across 6,141 open exceptions (2,464 critical stockouts). $1.59 average "impact" for a critical stockout is not credible to a planner.

**Root cause:** `scripts/inventory/generate_replenishment_exceptions.py` sets `financial_impact_total = loss_7d = daily_demand × margin × min(7, days_at_risk)` — i.e. **7 days of lost gross margin** (a fraction of revenue), for low-velocity single-location (1401-BULK) SKUs. The value is mathematically defensible but the UI labels it "Financial Impact at Risk" / "At Risk $12K" with no indication of the 7-day, margin-only scope. There is a `loss_of_sales_30d` column computed but not surfaced.

**Acceptance criterion:** Either (a) the tile/label states the scope explicitly (e.g. "7-day lost margin at risk") AND offers the 30-day figure, or (b) the metric is changed to extended at-risk demand value (qty × unit value over the exposure window). Concretely: the Action Feed and Command Center "$ at risk" copy must name the window and basis, and the displayed magnitude must be reconcilable to a documented formula a planner can defend in an S&OP meeting.

**Planner impact:** A "$12K total at risk" headline over 2,500 critical stockouts reads as a broken metric and undermines confidence in the whole triage surface.

---

## F1.3 — Data Quality summary counts don't reconcile with the catalog [SEV P3]

**Workflow blocked:** Data-quality monitoring / trust. The header KPIs and the catalog table below them report different totals, and "Failed 0" sits above a grid containing rows badged CRITICAL — a planner can't tell whether the pipeline is healthy.

**Evidence (screens/dataQuality.png + API):**
- Header tiles: "Total Checks **166**", "Passed **116**", "Failed **0**", "Warnings **26**".
- Catalog header: "Check Catalog (**83**)". `curl /data-quality/checks` → 83 checks, status breakdown `{pass:58, fail:13, warn:3, skip:9}`.
- So: Total 166 ≈ 2× the 83 actual checks (domain referential-pair double counting via `/data-quality/dashboard` `domains[]`), and 13 checks have status=fail while the "Failed" tile shows 0.

**Root cause:** `frontend/src/tabs/DataQualityTab.tsx` derives the tiles from `dashboard.domains[]` (sum of per-domain `total/passed/...`, which counts cross-domain referential pairs like `forecast_to_sku`, `inventory_to_item` separately → ~2× inflation) while the catalog table is fed by `/data-quality/checks` (83 distinct checks). The "Failed 0" is the documented severity-aware rule (F7.2: warning-severity fails roll into the amber Warnings tile), but the catalog still badges those rows "CRITICAL"/"WARNING", so the two surfaces visibly disagree.

**Acceptance criterion:** The "Total Checks" tile equals the catalog count (83), OR the catalog header and tiles use the same denominator and a tooltip explains the per-domain expansion; AND a check with catalog status=fail is reflected somewhere in the tiles (Failed or Warnings) such that "Failed 0 / 13 fail rows" can't both be true without explanation.

**Planner impact:** Low — DQ is a supporting/observability surface, not a daily decision path — but the mismatch invites "is this dashboard even right?" doubt.

---

## F1.4 — Customer Analytics below-fold panels show transient "Loading…" due to a cold ~11s affinity query [SEV P3]

**Workflow blocked:** Customer/channel drill-down. On a cold cache the bottom panels (Channel Mix / affinity-backed) sit on "Loading…" long enough to be captured mid-load.

**Evidence:**
- screens/customerAnalytics.png + digest: 8× "Loading…" at the bottom of the tab; KPIs and map rendered fine.
- `curl /customer-analytics/affinity` cold = **11.3s** (HTTP 200, 12.9KB); warm (Redis) = ~4ms. All other CA panel endpoints (channel-mix, treemap, heatmap, segment-trends, order-patterns, demand-flow, lifecycle) return 200 in <1s.

**Root cause:** The affinity endpoint (`api/routers/intelligence/customer_analytics/…`) runs an expensive self-join/co-occurrence aggregation with no warm cache on first hit. LazyPanel (IntersectionObserver) defers it below the fold; combined with the 11s cold query the capture snapshots it pre-resolution. Not a correctness defect — the panel resolves once cached.

**Acceptance criterion:** Cold `GET /customer-analytics/affinity` returns in < ~3s (index/pre-aggregate or MV), so below-fold CA panels resolve before a planner scrolls to them; OR confirm the affinity query is acceptable to leave behind LazyPanel + Redis warm-up and treat as won't-fix.

**Planner impact:** Minor friction on first visit only; cached thereafter.

---

## Checked and OK (no finding)

- **Command Center / Control Tower / AI Planner** — render correctly; controlTower & aiPlanner are intentional redirects to commandCenter (`useUrlState.ts` TAB_REDIRECTS).
- **Aggregate Analysis** — `BEER <0%*` cells are real (actual 5,078 vs forecast 27,294 → WAPE>100%), honestly annotated. A genuine over-forecast signal, correctly surfaced.
- **Demand History** — outlier MoM% (e.g. 520.9%*) are flagged with `*`; data plausible.
- **S&OP** (0 cycles), **FVA interventions** (0), **Clusters** (no assignments), **AI Planner FVA** (failed run shows honest "AI returned no quantity … run skipped") — all correct empty/error states.
- **Item Analysis, Explorer** — real data, KPIs populate (Item 18676 @ 1401-BULK, 75.4% accuracy).
- All exceptions at single location `1401-BULK` — data reality (all inventory & forecast at that one warehouse), not a join bug.

newActionableCount (NEW unresolved P0/P1/P2) = 2 (F1.1, F1.2)
