# Demand Planner UX Findings — Cycle 5
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-5 live capture digest + dump (14 tabs, 0 console errors on every tab) + curl / DB / code confirmation. Branch: restructure._

## Summary
The product is in good shape and the two NEW cycle-4 P1 items are both **verified RESOLVED** this cycle:

- **F4.1 (Command Center exception feed reads the empty `exception_queue`) — RESOLVED.** The Command Center / Control Tower / AI Planner feed now renders the real replenishment exceptions: 50+ CRITICAL rows (627099 @ 1401-BULK $572, 664631 $292, 913305 $279, …) that reconcile 1:1 with the Inventory Planning Action Feed. The "Exception data unavailable" empty-state is gone; the KPI tile (6142) and the feed below it now agree.
- **F4.2 (Data Quality "Last Run: Never" / "Check Catalog (0)") — RESOLVED.** The DQ page now shows **Last Run: 32m ago** and **Check Catalog (83)** with all 83 distinct checks listed (status/severity/domain/table/last value/last run), driven off `fact_dq_check_results` instead of the empty `dim_dq_check_catalog`.

The remaining items are all **carried** presentation/data-state items plus **one genuinely new P2 performance issue** (cold Item×State heatmap load). No 500s, no 404s the app actually calls, no broken charts, zero console errors on all 14 tabs.

There are **no new P0 or P1 issues this cycle.**

---

## NEW Findings

### F5.1 — Customer Map "Item × State Heatmap" cold load takes ~9.4 s; the panel sits on "Loading…" for the first planner of the day (and after every filter change)  [SEV: P2]  (NEW)
- **Workflow blocked:** Customer/item drill-down on the Customer Map. The Item×State heatmap is one of the four headline analytical panels; on a cold cache (first view of the day, or any State/Channel/Store-Type/date filter change) it blocks for ~9.4 s behind a "Loading…" spinner.
- **Evidence:** Tab `customerAnalytics` (`screens/customerAnalytics.png`, dump entry 13, 0 console errors). The digest shows the lower panels rendering "Loading… / Loading…" at capture time. Curl proof:
  - `curl /customer-analytics/heatmap` (cold) → **HTTP 200 in 9.42 s**; immediate 2nd call (cached) → 0.003 s.
  - Sibling MV-backed panels are fast: `/treemap` 0.002 s (rich `tree`: FL → Off Premise Chains → PUBLIX WAREHOUSE …), `/channel-mix` 0.002 s, `/map` 0.41 s, `/kpis` fast. So the heatmap is the sole slow panel.
- **Root cause:** `api/routers/intelligence/customer_analytics/geo.py` `customer_analytics_heatmap()` (line 254). Its `agg` CTE aggregates **the full `fact_customer_demand_monthly` table** JOIN `dim_customer` (and LEFT JOIN `dim_item`) with `GROUP BY item_id, item_desc, state`, then re-scans `agg` three more times (`top_items`, `top_states`, final JOIN). Unlike the other CA panels, this query does **not** route through `mv_customer_activity_monthly` (the fast path the memory notes call out for 9 of 16 CA endpoints) — it hits the raw fact table every cold call. `@_CA_CACHE` + `max_age=300` only hides it after the first hit; each distinct filter combination is a fresh ~9 s scan.
- **Acceptance criterion:** Cold `/customer-analytics/heatmap` (default params and with a State/Channel filter) returns in < 1.5 s — e.g. by sourcing the (item, state, demand, sales, customer_count) aggregate from `mv_customer_activity_monthly` (or a dedicated item×state MV) instead of `fact_customer_demand_monthly`, mirroring the treemap/map fast path. A perf assertion (or `make perf-script`) shows the cold call under the threshold. (Lower priority than the resolved P1s because the panel does eventually render correct data and is cached for subsequent views.)
- **Planner impact:** When I open Customer Map to investigate which states are short on a given item, the most useful matrix view stalls for ~10 s while everything else is instant — and it re-stalls every time I narrow by state/channel. Friction, not a blocker.

---

## Carried items (re-verified this cycle — NOT new)

### F4.5 — Customer Analytics **Store Type** filter still lists 275 raw free-text taxonomy values; Channel is now clean  [SEV: P2]  (carried; Channel half resolved, Store Type half remains)
- **Status update:** The cycle-4 `normalizeLabelOptions()` fix is working for **Channel** — verified live: the raw endpoint returns 33 channel strings (`'Off Premise Chains'`, `'Off Premise Chains            '`, `'OFF PREMISE CHAINS'`, `'null'`, …) and the normalizer correctly collapses them to **21** clean entries (exact-case + trailing-whitespace + `null` removed). Good.
- **Remaining problem:** **Store Type** normalizes only 293 → **275**, because the residual 275 are *genuinely distinct* free-text codes, not case/whitespace duplicates — so case-insensitive de-dup can't merge them. The dropdown is effectively unusable: 23 "bar" variants (`BAR`, `BARS`, `Sports Bar`, `R BAR/TAVERN/LOUNGE`, `TAV/BAR RB`, `B/W TAVERN/BAR`, …), 5 casino variants, 9 grocery variants (`A CHAIN GROCERY`, `CHAIN GROCERY`, `Chain Grocery Store`, `CHAIN SUPERMARKET/GROCERY`, …), 26 single-letter-prefixed codes (`A CHAIN GROCERY` … `Z NIGHT CLUB`), plus `**OBSOLETE **`, `UNKNOWN NO/SS/TR`, `User Defined`.
- **Evidence:** `customerAnalytics` dump Store Type dropdown; `curl /customer-analytics/filter-options` → `store_types` length 293; simulated `normalizeLabelOptions` → 275 distinct.
- **Root cause:** This is a **source-data taxonomy** problem, not a de-dup bug — `fetchCustomerAnalyticsFilterOptions` (`customer-analytics.ts:486`) cannot canonicalize 275 unrelated strings. It needs an upstream mapping table (raw store_type → ~12 canonical buckets) the same way Channel Mix already buckets into "Grocery Stores / Mass Merchandiser / Liquor / Clubs / …".
- **Acceptance criterion:** Store Type dropdown shows a small canonical set (≈10–15 buckets) backed by a raw→canonical mapping; the raw 275 are hidden behind that mapping (or shown as a flat searchable list with an explicit "raw codes" disclosure). At minimum, drop the obvious junk (`**OBSOLETE **`, `UNKNOWN *`, `User Defined`, single-letter-prefix codes).

### F4.3 — Command Center Portfolio Health 0/100 and Fill Rate "--" (health/fill-rate MVs unpopulated; no live fallback)  [SEV: P2]  (carried)
- **Evidence (re-verified):** `/control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0, below_ss_count:0}`, `fill_rate.portfolio_fill_rate_3m:null`, `warning:"mv_control_tower_kpis not yet refreshed. Run make refresh-mvs-tiered"`. Tiles show Portfolio Health 0/100, Fill Rate (3m) "--", "Portfolio Trend: No trend data available." The honest amber "Portfolio health data unavailable" banner is shown, so this is not a trust hazard.
- **Root cause:** `control_tower.py get_control_tower_kpis` reads only `mv_control_tower_kpis`; the cycle-3 live fallback covers the `exceptions` block only (now correct). There is still no equivalent live fallback for health (a `COUNT … GROUP BY health_tier` over inventory) or fill rate.
- **Acceptance criterion:** After `make refresh-mvs-tiered` the health/fill tiles show non-zero numbers; OR a live fallback computes below-SS coverage / fill rate from base tables when the MV is stale (mirroring the exceptions fallback).

### F4.4 — Cluster Accuracy Comparison **table** still renders raw negative accuracy (`-12.89%`, `-61.08%`, `-128.04%`)  [SEV: P2]  (carried; heatmap half was fixed in F3.2, table half remains)
- **Evidence (re-verified):** `aggregateAnalysis` "MODEL COMPARISON — 13 CLUSTER ASSIGNMENT BUCKETS": L2_4 ★-12.89% (WAPE 112.89%), L2_5 ★-61.08%, L2_6 ★-128.04% (WAPE 228.04%), L2_6S ★-18.19%, L2_99 ★-6.17%, biases up to ⚠-69.2%. The Accuracy **heatmap** correctly floors to `<0%*` with caption (F3.2 fixed); this comparison table prints raw `accuracy_pct`.
- **Acceptance criterion:** Low-base rows floor accuracy at 0% with a "low base — see WAPE" marker (consistent with the heatmap), or the table caption explains negative = forecast ≫ actual on a tiny base. Snapshot test covers a row with `accuracy_pct < 0`.

---

## Tabs working well (no action)
- **Command Center / Control Tower / AI Planner** — feed now lists the real critical exceptions (F4.1 RESOLVED); KPI tiles and feed agree. Only the health/fill tiles remain MV-stale (F4.3, honestly bannered).
- **Inventory Planning** — Action Feed: 20 critical actions, $3.6K at risk, individual SKUs reconciling with the 6,142-row exceptions table. Strongest tab.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias, forecast-vs-actual series, lag curve (63→75%), heatmap negatives floored. Modulo F4.4 table presentation.
- **Item Analysis** — defaults to Item 106811 @ 1401-BULK with a **healthy 91.24% accuracy / 8.76% WAPE / FCST 3.3K / ACTUAL 3.4K**, full chart + SHAP + Forecast KPIs + DQ Corrections.
- **Demand History** — Workbench lists 50/50 series with volume + sparkline + MoM% (large values are legitimate intermittent swings).
- **Customer Map** — 23.0M cases, 98% fill, 461K lost sales, 32.5K active customers; State dropdown clean, Channel clean (F4.5 Channel half), treemap/channel-mix/map render rich hierarchies. Only the heatmap panel is slow on cold load (F5.1) and Store Type dropdown is dirty (F4.5).
- **Data Quality** — 64% health, 166 checks, 116 passed / 26 failed / 6 warnings, Last Run 32m ago, Check Catalog (83) fully populated (F4.2 RESOLVED), 20 domain cards, recent issues panel.
- **Explorer** — fast raw data across 9 domains.
- **AI Planner FVA Backtest** — run history populated (succeeded/failed, provider/DFU/rec counts).
- **FVA & ROI** — External +5.2 pts vs Naive Seasonal (76,995 rows); Champion "No data" and AI/Planner "Coming Soon" are honest empty/reserved states (no defect; populates after a backtest is promoted).
- **Clusters / S&OP** — genuinely empty with honest empty states + run-pipeline / create-cycle CTAs. Not defects.

## Harness note (carried, not a product defect)
`controlTower` and `aiPlanner` routes still render Command Center content in the dump (entries 4 and 7 byte-identical to entry 0) — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries. Flagged so they aren't mistaken for additional broken tabs.

---

## Resolved since Cycle 4 (verified this cycle)
- **F4.1 (Command Center feed read empty `exception_queue`) — RESOLVED.** Feed now lists 50+ real CRITICAL replenishment exceptions matching the Inventory Action Feed; the "Exception data unavailable" empty-state is gone.
- **F4.2 (DQ "Last Run: Never" / "Check Catalog (0)") — RESOLVED.** Now shows "Last Run: 32m ago" and "Check Catalog (83)" with all 83 checks listed.
- **F4.5 (dirty CA dropdowns) — Channel half RESOLVED** (33→21 verified live). Store Type half remains (taxonomy mapping needed) → carried as F4.5.
