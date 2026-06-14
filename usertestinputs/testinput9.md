# Demand Planner UX Findings — Cycle 9
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-9 live capture digest + dump (14 tabs, **0 console errors on every tab, all 14 `ok=true`**) + curl / Postgres / code confirmation. Branch: restructure._

## Summary

The product remains in **very good shape** — all 14 tabs load (14/14 `ok`, 0 console errors, no 500s in the dump). The full cycle-8 picture reconciles. I found **ONE genuinely new defect** this cycle plus the long-standing carried items.

**NEW: F9.1 (P2)** — The **FVA & ROI** "Forecast Value Ladder" goes fully blank ("No data" on Naive Seasonal, External, *and* Champion) the moment a planner narrows the window to **3 months**, and is materially thinned at **6 months**. Root cause is a planning-date violation: `fva.py` anchors the window to the real wall-clock `current_date` (2026-06-14) instead of `get_planning_date()` (2026-04-02). The forecast horizon ends 2026-02-01, so `current_date − 3mo (≥ 2026-03-14)` matches **0 rows**, while the correct planning-date window (`≥ 2026-01-02`) matches **13,679 rows**. The tab's default is 12 months, which masks the bug — but the 3-month recency window is the most natural FVA-review lens and it shows an empty, misleading ladder. This is the same class of error CLAUDE.md flags as `[FREQUENTLY VIOLATED]` and the only `current_date`-anchored window left in the forecasting routers.

Carried items (F4.3 control-tower MV staleness, F4.5 Store Type taxonomy, F6.2 dead concentration route) are re-verified unchanged and remain honestly handled. **newActionableCount (new unresolved P0/P1/P2) = 1.**

---

## NEW Findings

### F9.1 — FVA "Forecast Value Ladder" empty at 3-month window (and thinned at 6m) because it anchors to wall-clock `current_date`, not the planning date  [SEV: P2]
- **Workflow blocked:** Forecast accuracy & FVA review. A planner reviewing recent forecast value-add naturally selects the shortest, most-current window (3 months). Doing so collapses the entire ladder — Naive Seasonal, External, and Champion all read **"No data"**, the headline ("External improves +5.2 pts vs Naive Seasonal") disappears, and the Ceiling Benchmark shows "—". The baselines *do* have data; they are simply outside the wrong window.
- **Evidence:**
  - `curl '/fva/waterfall?months=3'` → every measurable stage `state:"missing"`, `accuracy_pct:null`, `n_rows:0`.
  - `curl '/fva/waterfall?months=6'` → seasonal_naive 64.19% / external 71.42%, but only **21,536 rows**.
  - `curl '/fva/waterfall?months=12'` (the UI default) → seasonal_naive 65.58% / external 70.81%, **76,995 rows** — this is why the captured `fva` screen looks healthy.
  - DB: `fact_external_forecast_monthly` startdate range is **2025-03-01 → 2026-02-01**; `current_date` in Postgres is **2026-06-14**; `get_planning_date()` is **2026-04-02**.
  - Row counts: `WHERE startdate >= current_date − 3mo (2026-03-14)` → **0 rows**; `WHERE startdate >= 2026-01-02` (planning-date − 3mo) → **13,679 rows**.
  - UI presents `state:"missing"` as the literal "No data" tile (`FVATab.tsx:160-184`, `stageValueLabel(stage)`), so the planner sees a baseline that genuinely exists reported as absent.
- **Root cause:** `api/routers/forecasting/fva.py:53` —
  `"WHERE f.startdate >= current_date - (%s * interval '1 month') "`
  uses the database `current_date` (the real system clock, ~2.5 months ahead of the demo data horizon) instead of the planning date. Sibling forecasting routers (`production_forecast.py:541`, `consensus_plan.py:74`) correctly use `get_planning_date()`. Line 254 (`/fva/roi` interventions window) has the identical `current_date` anchor and would mis-window once real interventions exist. This is the only `current_date`-anchored window left in `api/routers/forecasting/`.
- **Acceptance criterion:** With `PLANNING_DATE=2026-04-02`, `GET /fva/waterfall?months=3` returns `seasonal_naive` and `external` stages with `state:"actual"` and `n_rows > 0` (≈13,679-row scope), and the FVATab 3-month view renders accuracy numbers + the comparison headline instead of "No data" on the baseline stages. Implement by replacing `current_date` with a bound planning date (`%s::date − interval`, passing `get_planning_date()`), consistently at lines 53 and 254. Add an API test asserting `months=3` is non-empty under a pinned `PLANNING_DATE`.
- **Planner impact:** Medium. The 12-month default hides it, but FVA exists precisely to defend recent forecast-improvement claims in S&OP; a blank 3-month ladder reads as "we have no measured value-add lately," which is false and undermines trust in the tab. Single-file backend fix; no schema or UI change required.

---

## Carried items (re-verified this cycle — NOT new, already in ledger)

### F4.3 — Command Center / Control Tower Portfolio Health 0/100, Fill Rate "--", Portfolio Trend "No trend data available" (stale `mv_control_tower_kpis`, no live fallback)  [SEV: P2]  (carried — 7th cycle, honestly bannered)
- **Re-verified live:** `curl /control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0}`, `fill_rate.portfolio_fill_rate_3m:null`. `curl /control-tower/trend?months=6` → `{"trend":[], "warning":"Upstream materialized view not yet refreshed."}`. The amber "Portfolio health data unavailable — these zeros are not a sign of a healthy portfolio" banner is present (confirmed in `screens`/digest), so this is **not a trust hazard**. The exceptions block in the same payload is correct and live (6,142 open / 2,465 critical / $246,723.23, source `fact_replenishment_exceptions`).
- **Root cause:** `api/routers/operations/control_tower.py` reads only the stale MV for health/fill_rate/trend; only the exceptions block has the cycle-3 live fallback.
- **Acceptance criterion:** After `make refresh-mvs-tiered` the tiles + trend show non-zero numbers; OR a live fallback computes below-SS coverage / portfolio fill rate / 6M trend from base tables when the MV is stale (mirroring the exceptions fallback).

### F4.5 — Customer Analytics **Store Type** filter still lists ~275 raw free-text taxonomy values; Channel is clean  [SEV: P2]  (carried; Channel half resolved)
- **Re-verified:** Customer Map Store Type dropdown still lists raw taxonomy (`**OBSOLETE **`, single-letter-prefixed codes, `UNKNOWN *`, `User Defined`). Channel (~22 entries) is clean.
- **Root cause:** upstream source-data taxonomy; `fetchCustomerAnalyticsFilterOptions` (`customer-analytics.ts`) cannot canonicalize ~275 genuinely-distinct strings without an upstream raw→canonical mapping table.
- **Acceptance criterion:** Store Type shows ~10–15 canonical buckets or a searchable disclosure; at minimum the obvious junk is dropped.

### F6.2 — `/customer-analytics/concentration` returns 404, but it is a dead unused route (no UI calls it)  [SEV: P3]  (carried context, not a user break)
- **Re-verified:** the "Customer Concentration" treemap is served by `/treemap` (200, renders colored). The 404 route is dead cleanup only.

---

## Tabs working well (no action)
- **Command Center / Control Tower / AI Planner** — feed lists real critical replenishment exceptions (50+ CRITICAL rows: 627099 @ 1401-BULK $572, 664631 $292, 913305 $279 …); KPI tile (6142 / 2465 critical) and feed agree. Only health/fill/trend tiles are MV-stale (F4.3, honestly bannered).
- **Inventory Planning** — Today's Plan + Action Feed reconcile with the 6,142-row exceptions table; compact-currency formatting ("$3.6K") consistent (U8.1 holds).
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias; `dashboard/kpis?window_months=3` returns real data (correctly scoped to available actuals — does NOT share F9.1's `current_date` bug); KPI deltas color by good/bad direction (U6.1 holds).
- **Item Analysis** — defaults to Item 105430 @ 1401-BULK; chart axes, 6 staging models (Chronos Bolt, LightGBM, MSTL, N-BEATS, N-HiTS, Seasonal Naive), SHAP, Forecast KPIs, DQ Corrections all present.
- **Demand History** — Workbench series with volume + LAST + labeled MoM% low-base footnote (U6.5 holds).
- **Customer Map** — 23.0M cases, 32,469 customers; State/Channel dropdowns clean; treemap renders colored fill-rate ramp with legend (U7.1 holds). Store Type still dirty (F4.5).
- **Data Quality** — 166 checks, skipped/info accounting reconciles (F7.1/U8.3 hold); genuine fails surfaced honestly.
- **FVA & ROI (12m default)** — External 70.8% (+5.2 pts vs Naive Seasonal 65.6%, 76,995 rows). Champion "No data", AI/Planner "Coming Soon" are honest reserved states. **But the 3-month/6-month windows are broken — see F9.1.**
- **AI Planner FVA Backtest** — run history populated; the one `failed` run surfaces its real pydantic error inline; zero-rec runs handled (U7.2 holds).
- **S&OP / Clusters** — genuinely empty with honest empty states + create/run CTAs. Not defects.
- **Explorer** — fast raw data across 9 domains.

---

## Harness note (carried, not a product defect)
`Control Tower` and `AI Planner` dump entries render Command Center content byte-identical to entry 0 (`textLen` 5516 for all three) — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries. Flagged so they aren't mistaken for additional broken tabs.
