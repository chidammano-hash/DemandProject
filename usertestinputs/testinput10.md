# Demand Planner UX Findings — Cycle 10
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-10 live capture digest + dump (14 tabs, **all 14 `ok=true`, 0 console errors, no 500s in the dump**) + curl / Postgres / code confirmation. Branch: restructure._

## Summary

The product is in **very good shape** — and the one new defect from cycle 9 is now **fixed and live**. All 14 planner tabs load (14/14 `ok`, 0 console errors, no 500s). I found **ZERO genuinely new defects** this cycle. The carried items (F4.3 control-tower MV staleness, F4.5 Store Type taxonomy, F6.2 dead concentration route) re-verify unchanged and remain honestly handled.

**F9.1 (cycle-9 P2) is RESOLVED and confirmed live:** the FVA "Forecast Value Ladder" now renders real numbers at the 3-month and 6-month windows. `GET /fva/waterfall?months=3` returns `seasonal_naive` 65.82% / `external` 72.57% (`+6.8 pts`), both `state:"actual"`, `n_rows:10889` (was `state:"missing"`, `n_rows:0`). The captured FVA tab shows "External improves +5.9 pts vs Naive Seasonal" with populated Step 1/Step 2 tiles instead of "No data." The `current_date` → `get_planning_date()` anchor fix landed.

**newActionableCount (new unresolved P0/P1/P2) = 0.**

Because I found no new actionable defects, I am not inventing any. Below I (1) record the F9.1 resolution, (2) re-verify the long-standing carried items so they aren't mistaken for regressions, and (3) note one purely-cosmetic capture artifact so it isn't mis-read as a break.

---

## Resolved this cycle (verification)

### F9.1 — FVA "Forecast Value Ladder" empty at 3-/6-month windows  →  RESOLVED  [was P2]
- **Verified live:**
  - `curl '/fva/waterfall?months=3'` → `seasonal_naive` `state:"actual"`, `accuracy_pct:65.82`, `n_rows:10889`; `external` `state:"actual"`, `accuracy_pct:72.57`, `delta_vs_prev:6.8`, `n_rows:10889`. (Cycle 9: every stage `state:"missing"`, `n_rows:0`.)
  - Captured `fva` tab (3-month default in the capture) shows the full ladder: Naive Seasonal 65.3% / 92,926 rows, External 71.2% (+5.9 pts) / 92,926 rows, headline "External improves +5.9 pts vs Naive Seasonal." Champion still honest "No data"; AI/Planner honest "Coming Soon."
- **Root cause that was fixed:** `api/routers/forecasting/fva.py` was anchoring the window to the database wall-clock `current_date` (2026-06-14, ~2.5 months past the data horizon ending 2026-02-01) instead of `get_planning_date()`. Now bound to the planning date, so the 3-/6-month windows intersect real forecast rows.

---

## Carried items (re-verified this cycle — NOT new, already in ledger, no regression)

### F4.3 — Control Tower / Command Center Portfolio Health 0/100, Fill Rate "--", Portfolio Trend "No trend data available"  [SEV: P2]  (carried — 8th cycle, honestly bannered)
- **Re-verified live:** `curl /control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0}`, `fill_rate.portfolio_fill_rate_3m:null`. The amber "Portfolio health data unavailable — these zeros are not a sign of a healthy portfolio. Refresh the analytics views (run make refresh-mvs-tiered)…" banner is present on Command Center / Control Tower / AI Planner (confirmed in digest), so this is **not a trust hazard**. The exceptions block in the same payload is correct and live (6,142 open / 2,465 critical, source `fact_replenishment_exceptions`).
- **Root cause:** `api/routers/operations/control_tower.py` reads only the stale `mv_control_tower_kpis` for health/fill_rate/trend; only the exceptions block has the cycle-3 live fallback.
- **Acceptance criterion (unchanged):** After `make refresh-mvs-tiered` the tiles + 6M trend show non-zero numbers; OR a live fallback computes below-SS coverage / portfolio fill rate / 6M trend from base tables when the MV is stale (mirroring the exceptions fallback).

### F4.5 — Customer Analytics **Store Type** filter lists ~275 raw free-text taxonomy values; Channel is clean  [SEV: P2]  (carried; Channel half resolved)
- **Re-verified:** Store Type dropdown still lists raw taxonomy (`**OBSOLETE **`, single-letter-prefixed codes like `B INDEPENDENT GROCERY`, `UNKNOWN NO`, `User Defined`). Channel (~22 entries) and State (clean 2-letter codes) are fine. Map, Concentration treemap (colored fill-rate ramp + legend), Channel Mix, and Heatmap all render with real data — `curl /customer-analytics/channel-mix` and `/heatmap` and `/treemap` all 200 in <0.5s.
- **Root cause:** upstream source-data taxonomy; `fetchCustomerAnalyticsFilterOptions` (`frontend/src/api/queries/customer-analytics.ts`) cannot canonicalize ~275 genuinely-distinct strings without an upstream raw→canonical mapping table.
- **Acceptance criterion (unchanged):** Store Type shows ~10–15 canonical buckets or a searchable disclosure; at minimum the obvious junk (`**OBSOLETE **`, `UNKNOWN *`, `User Defined`) is dropped.

### F6.2 — `/customer-analytics/concentration` returns 404, but it is a dead unused route (no UI calls it)  [SEV: P3]  (carried context, not a user break)
- **Re-verified:** the "Customer Concentration" treemap is served by `/customer-analytics/treemap` (200, renders colored). The 404 `/concentration` route is dead cleanup only — no UI dependency.

---

## Capture artifact (NOT a product defect — flagged so it isn't mis-read)

- **Customer Map below-fold panels show "Loading…" in the digest text.** The captured `customerAnalytics` screenshot shows the above-fold content fully rendered (KPI strip, Demand Map, Concentration treemap). The eight "Loading…" lines in the digest are the **below-fold** panels (Item × State Heatmap, Channel Mix) that use `LazyPanel` (IntersectionObserver) and had not scrolled into view at snapshot time. Their endpoints are healthy and fast: `/customer-analytics/heatmap` 200 / 0.41s, `/channel-mix` 200 / 0.003s, `/treemap` 200 / 0.21s. Not a stall.
- **`Control Tower` and `AI Planner` dump entries render Command Center content** (textLen 5516, byte-identical to entry 0) — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries, same as prior cycles. Not additional broken tabs.

---

## Tabs working well (no action)
- **Command Center / Control Tower / AI Planner** — feed lists real critical replenishment exceptions (50+ CRITICAL rows: 627099 @ 1401-BULK $572, 664631 $292, 913305 $279 …); KPI tile (6142 / 2465 critical) and feed agree. Only health/fill/trend tiles are MV-stale (F4.3, honestly bannered).
- **Inventory Planning** — Today's Plan + Unified Action Feed reconcile (6,214 total / 4,252 critical / $12.1K at risk; "Showing top 20 of 6,214… KPIs above reflect the full population" — U9.1 full-population aggregate holds); compact-currency "$12K"/"$3.6K" consistent.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias; `dashboard/kpis?window_months=3` returns real data (correctly scoped — does NOT share the old `current_date` bug); KPI deltas color by good/bad direction; heatmap `<0%*` flooring + caption + cluster-slice flooring all hold.
- **Item Analysis** — defaults to Item 106811 @ 1401-BULK; chart, 6 staging models (Chronos Bolt, LightGBM, MSTL, N-BEATS, N-HiTS, Seasonal Naive), SHAP, Forecast KPIs (External 12mo 91.24% acc / 8.76% WAPE), DQ Corrections all present.
- **Demand History** — 50/50 workbench series with volume + LAST + labeled MoM% (520.9%* low-base footnote on PINNACLE VODKA holds), `#item_id` disambiguator suffix holds.
- **Customer Map** — 23.0M cases, 32,469 customers; State/Channel dropdowns clean; treemap renders colored fill-rate ramp + legend; KPI MoM deltas labeled. Store Type still dirty (F4.5).
- **Data Quality** — 166 checks, 80% overall, skipped/info accounting reconciles (Customer 100% / Forecast 100% with 4 skip / Sku 80% with 2 info); genuine fails (Forecast_to_sku, Inventory_to_item, Sourcing_to_location 0%) surfaced honestly; "Last Value (violations)" header + Run Checks Now CTA present.
- **FVA & ROI** — 3-/6-/12-month windows all render real ladders now (F9.1 fixed). Champion "No data", AI/Planner "Coming Soon" are honest reserved states.
- **AI Planner FVA Backtest** — run history populated; the one `failed` run surfaces its real pydantic validation error inline; zero-rec runs (2026-04-01, 0 recs) handled honestly.
- **S&OP / Clusters** — genuinely empty with honest empty states + create/run CTAs ("No active S&OP cycles. Create one via the API or CLI." / "Run Clustering Pipeline"). Not defects.
- **Explorer** — fast raw data across 9 domains; null sentinels render "-".

---

## Verdict
Cycle 10 is a **clean cycle**: the prior cycle's single defect (F9.1) is resolved and verified live, no regressions, and no new actionable issues. The remaining carried items are either genuine-empty/MV-staleness states that are honestly bannered (F4.3), an upstream-data taxonomy limitation that needs a source mapping table (F4.5), or dead-route cleanup (F6.2) — none block a daily demand-planning workflow.
