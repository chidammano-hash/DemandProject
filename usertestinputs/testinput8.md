# Demand Planner UX Findings — Cycle 8
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-8 live capture digest + dump (14 tabs, **0 console errors on every tab**, all 14 `ok=true`) + curl / Postgres / code confirmation. Branch: restructure._

## Summary

The product is in **very good shape** and **cleaner than any prior cycle**. All 14 tabs loaded successfully (14/14 `ok`, 0 console errors, no 500s in the dump). I found **NO new defects this cycle.**

The cycle-7 NEW item **F7.1 (Data Quality "Domain Health" skipped-check accounting) is verified RESOLVED**:
- `GET /data-quality/dashboard` now emits a `skipped` field per domain and the scores reconcile. Live curl: `forecast {score:100.0, passed:12, skipped:4, total:16}`, `item {score:100.0, passed:10, skipped:6, total:16}` — domains with zero failures/warnings now read **100%** instead of 75%/62.5%.
- The Data Quality summary bar now shows a **"18 Skipped"** tile alongside 116 Passed / 26 Failed / 6 Warnings, and each domain card carries its skip count. 166 total now fully reconciles (116+26+6+18=166). Verified in `screens/dataQuality.png`.

Everything else reconciles with prior cycles. The genuinely-failing domains in the DQ grid (`inventory` 66.7% with 8 real fails, `forecast_to_sku`/`inventory_to_item` 0% with 2 fails each) are **correct** — those are true referential-integrity check failures, not a scoring bug.

The only carried high-value item remains **F4.3** (Command Center Portfolio Health 0/100, Fill Rate "--", and the "No trend data available." Portfolio-Trend widget) — all three driven by the single stale `mv_control_tower_kpis` and **honestly mitigated** by the amber "Portfolio health data unavailable — these zeros are not a sign of a healthy portfolio" banner. This is a data-refresh state, not a code defect; the exceptions block in the same payload is correct and live (6,142 open / 2,465 critical / $246,723).

**No new P0/P1/P2/P3. newActionableCount (new unresolved P0/P1/P2) = 0.**

---

## NEW Findings

None this cycle. The product passes acceptance for every daily demand-planning workflow I exercised (morning triage, forecast accuracy & FVA review, exception/control-tower triage, inventory planning actions, demand history, S&OP prep, customer/item drill-down).

---

## Resolved since Cycle 7 (verified this cycle)

### F7.1 — Data Quality "Domain Health" sub-100% scores with all-green breakdowns — RESOLVED
- **Was:** Forecast 75% / Item 62.5% with "0 fail / 0 warn" because skipped checks diluted the denominator and were invisible.
- **Now (verified live):** `curl /data-quality/dashboard` returns `skipped` per domain; clean domains with skips score 100% (forecast 100% skipped:4, item 100% skipped:6, customer 100% skipped:0). The DataQualityTab summary KPI bar shows an "18 Skipped" tile so 116+26+6+18=166 reconciles, and per-domain cards display their skip counts. `screens/dataQuality.png` confirms the badges and the Skipped tile.

---

## Carried items (re-verified this cycle — NOT new, already in ledger)

### F4.3 — Command Center Portfolio Health 0/100, Fill Rate "--", Portfolio Trend "No trend data available" (stale `mv_control_tower_kpis`; no live fallback)  [SEV: P2]  (carried, honestly bannered — 6th cycle)
- **Re-verified live:** `curl /control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0}`, `fill_rate.portfolio_fill_rate_3m:null`, `warning:"mv_control_tower_kpis not yet refreshed"`. `curl /control-tower/trend?months=6` → `{"trend":[], "warning":"Upstream materialized view not yet refreshed"}`. The Command Center / Control Tower tiles show Portfolio Health 0/100, Fill Rate (3m) "--", and the collapsible Portfolio Trend (6M) widget shows "No trend data available."
- **Mitigation present:** the amber "Portfolio health data unavailable — these zeros are not a sign of a healthy portfolio" banner sits above the tiles, so this is **not a trust hazard**. Exceptions block in the same payload is correct/live (6,142 / 2,465 critical / $246,723, source `fact_replenishment_exceptions`).
- **Root cause:** `api/routers/operations/control_tower.py` `get_control_tower_kpis` (line 94, `FROM mv_control_tower_kpis`) and `get_control_tower_trend` (line 519) read only the stale MV for health + fill_rate + trend; only the exceptions block has the cycle-3 live fallback.
- **Minor honesty sub-note (not worth a separate finding):** `CommandCenterTab.tsx:581` renders the trend empty-state purely on `trend.length > 0`, ignoring the `warning` the endpoint already returns — so the trend widget shows a bare "No trend data available." rather than echoing the "refresh MVs" hint the tiles' banner shows. Low value; folds into F4.3's fix (live fallback) or a one-line warning surface.
- **Acceptance criterion:** After `make refresh-mvs-tiered` the tiles + trend show non-zero numbers; OR a live fallback computes below-SS coverage / portfolio fill rate / 6M trend from base tables when the MV is stale (mirroring the exceptions fallback).

### F4.5 — Customer Analytics **Store Type** filter still lists ~275 raw free-text taxonomy values; Channel is clean  [SEV: P2]  (carried; Channel half resolved)
- **Re-verified:** `customerAnalytics` Store Type dropdown still lists the raw taxonomy (`**OBSOLETE **`, single-letter-prefixed codes, `UNKNOWN *`, `User Defined`, etc.). Channel (~22 entries) is clean (cycle-4 `normalizeLabelOptions()` holds).
- **Root cause:** source-data taxonomy problem — `fetchCustomerAnalyticsFilterOptions` (`customer-analytics.ts`) cannot canonicalize ~275 genuinely-distinct strings; needs an upstream raw→canonical mapping table.
- **Acceptance criterion:** Store Type shows ~10–15 canonical buckets (mapping table) or a searchable disclosure; at minimum the obvious junk is dropped.

### F6.2 — `/customer-analytics/concentration` returns 404, but it is a dead unused route (no UI calls it)  [SEV: P3]  (carried context, not a user break)
- **Re-verified:** the "Customer Concentration" treemap is served by `/treemap` (200, renders colored — U6.3/U7.1 fixes hold). The 404 route is dead cleanup only.

---

## Tabs working well (no action)
- **Command Center / Control Tower / AI Planner** — feed lists real critical replenishment exceptions (50+ CRITICAL rows: 627099 @ 1401-BULK $572, 664631 $292, 913305 $279 …) reconciling 1:1 with the Inventory Action Feed; KPI tile (6142 / 2465 critical) and feed agree. Only health/fill/trend tiles are MV-stale (F4.3, honestly bannered).
- **Inventory Planning** — Today's Plan: 20 urgent, $4K at risk; Action Feed: 20 critical / $3.6K, SKUs reconciling with the 6,142-row exceptions table. Strongest tab.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias; forecast-vs-actual series; lag curve; heatmap negatives floored to `<0%*` (BEER `<0%*`, FOOD 89.1%/76.8%, etc.) with caption; KPI deltas color by good/bad direction (U6.1 holds).
- **Item Analysis** — defaults to Item 10053 @ 1401-BULK; full chart + 6 staging models (Chronos Bolt, LightGBM, MSTL, N-BEATS, N-HiTS, Seasonal Naive) + SHAP + Forecast KPIs + DQ Corrections.
- **Demand History** — Workbench lists 50/50 series with volume + LAST + MoM% (TITOS 1.36M ↑63.4%, SMIRNOFF 520K ↓6.1%); MoM labeled with low-base footnote (U6.5 holds).
- **Customer Map** — 23.0M cases, 32,469 customers; State/Channel dropdowns clean; treemap renders colored fill-rate ramp with legend (U7.1 holds); KPI strip + demand map + concentration all rich and fast. Store Type still dirty (F4.5).
- **Data Quality** — 69% health, 166 checks, 116 passed / 26 failed / 6 warnings / **18 skipped** (F7.1 fixed), Last Run 1h ago, Check Catalog populated with status-driven icons (F6.1 holds). Genuine fails surfaced honestly (inventory 66.7%, forecast_to_sku/inventory_to_item 0%).
- **FVA & ROI** — External 70.8% (+5.2 pts vs Naive Seasonal 65.6%, 76,995 rows); Champion "No data", AI/Planner "Coming Soon" are honest empty/reserved states.
- **AI Planner FVA Backtest** — run history populated; the one `failed` run (2025-12-01) shows its real error inline ("2 validation errors for Recommendation proposed_qty … input_value=None") and the zero-rec runs are handled (U7.2 holds). The raw pydantic error string is verbose but it is the honest, intentionally-surfaced failure detail accepted in cycle 7 — not a new defect.
- **Explorer** — fast raw data across 9 domains.
- **S&OP / Clusters** — genuinely empty with honest empty states + create-cycle / run-pipeline CTAs. Not defects.

---

## Harness note (carried, not a product defect)
`controlTower` and `aiPlanner` dump entries render Command Center content byte-identical to entry 0 — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries. Flagged so they aren't mistaken for additional broken tabs.
