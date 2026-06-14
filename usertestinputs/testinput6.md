# Demand Planner UX Findings — Cycle 6
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-6 live capture digest + dump (14 tabs, **0 console errors on every tab**, all 14 `ok=true`) + curl / Postgres / code confirmation. Branch: restructure._

## Summary

The product is in **very good shape** this cycle. The cycle-5 NEW item is **verified RESOLVED**, and there are **no new P0 or P1 issues**.

- **F5.1 (Customer Map Item×State heatmap cold load ~9.4 s) — RESOLVED.** The cycle-5 `mv_ca_item_state` MV (sql/187) is live (`pg_matviews` confirms it alongside `mv_ca_demand_at_risk`, `mv_ca_order_patterns`, `mv_ca_segment_trends`). Cold `curl /customer-analytics/heatmap` now returns **HTTP 200 in 0.43 s** (was 9.42 s); cached 0.002 s; returns 25 real items × states. All sibling CA panels are fast: `/treemap` 0.003 s, `/channel-mix` 0.61 s, `/map` 0.39 s, `/kpis` 0.18 s.
- **F4.4 (Cluster Accuracy Comparison table raw negative accuracy) — RESOLVED.** The digest now shows the comparison table flooring low-base rows to `★<0%*` (L2_4, L2_5, L2_6, L2_6S, L2_99) with the caption "Accuracy = 100 − WAPE; <0%* = tiny actual base (forecast >> actual) — read WAPE instead." Consistent with the heatmap. 11 `<0%*` literals in the digest, none raw-negative in the table.

Every tab loaded clean (14/14 `ok`, 0 console errors). No 500s. The single 404 endpoint (`/customer-analytics/concentration`) is **not called by the frontend** — verified the "Customer Concentration" panel is served by `/treemap`, so it is a dead/unused route, not a user-facing break.

The remaining items are all **carried** presentation/data-state items plus **one genuinely new low-severity clarity issue** in the Data Quality Check Catalog. The Customer Map "Loading…" panels in the digest are a **capture-timing artifact** (below-fold `LazyPanel`/IntersectionObserver had not fired at snapshot time), not a defect — every panel endpoint returns 200 with data.

**No new P0/P1. One new P3. newActionableCount (new unresolved P0/P1/P2) = 0.**

---

## NEW Findings

### F6.1 — Data Quality "Check Catalog" renders the check **severity** ("CRITICAL"/"WARNING") as the most prominent badge next to a bare "Last Value 0.00", which reads as 28 failing critical checks when all 28 are actually passing  [SEV: P3]  (NEW)
- **Workflow blocked:** Data Quality review / S&OP data-readiness sign-off. None blocked — this is a clarity/trust nuance, not a break.
- **Evidence:** Tab `dataQuality` (`screens/dataQuality.png`, dump entry 10, 0 console errors). The Check Catalog lists rows like `CRITICAL  completeness_forecast_item_id  …  Last Value 0.00`, `CRITICAL uniqueness_customer … 0.00`, `CRITICAL completeness_inventory_loc … 0.00` — 28 critical-severity rows. Curl + DB proof:
  - `curl /data-quality/checks` → all 28 `severity:"critical"` rows have `last_status:"pass"` (status counts: 58 pass / 13 fail / 3 warn / 9 skip — healthy).
  - `fact_dq_check_results`: `completeness_forecast_item_id` → `status=pass, metric_value=0, details={"nulls":0,"total":138931,"null_pct":0.0}`. So `metric_value` (rendered as "Last Value") is a **defect/violation count** — `0.00` means **zero defects = PASS**, not "0% complete."
- **Why it's only P3:** The Status **icon** column (left-most) does render the green pass icon off `last_status`, so a careful reader can disambiguate. The header summary (116 passed / 26 failed / 6 warnings) is accurate. But the eye is drawn to the bold uppercase red/amber "CRITICAL"/"WARNING" severity badge and a context-free "0.00", which together read as "critical check, value zero = broken." A planner scanning the catalog will misjudge 28 healthy checks as failures.
- **Root cause:** `frontend/src/tabs/data-quality/CheckCatalogPanel.tsx`. The "Severity" column (lines 158-162) renders `c.severity` with `SEVERITY_STYLE` (bold uppercase pill) — this is the check's *configured severity-if-it-fails*, not its current outcome. The "Last Value" column (lines 167-169) prints `c.last_value.toFixed(2)` with no label/units and no pass/fail context. The pass/fail signal lives only in the small icon (lines 144-156). Backend `/data-quality/checks` returns both `severity` and `last_status` correctly; the panel just over-weights `severity`.
- **Acceptance criterion:** The Check Catalog's prominent status conveys the **outcome** (pass/fail/warn/skip from `last_status`), with the configured `severity` shown as secondary metadata (e.g. greyed, or only emphasized when `last_status !== "pass"`). "Last Value" gets a tooltip or suffix clarifying it is a defect/violation metric (e.g. "0.00 violations" or "null_pct 0.0%"), so a passing `completeness 0.00` no longer reads as alarming. A snapshot/RTL test asserts a `severity:"critical", last_status:"pass"` row does NOT render as a failing/critical-looking row.
- **Planner impact:** During data-readiness review before an S&OP demand sign-off I scan the catalog for red. Right now 28 rows show "CRITICAL … 0.00" even though they passed — I either waste time chasing non-issues or learn to ignore the catalog entirely, which is worse.

---

## Carried items (re-verified this cycle — NOT new)

### F4.3 — Command Center Portfolio Health 0/100 and Fill Rate "--" (health/fill-rate MVs unpopulated; no live fallback)  [SEV: P2]  (carried, honestly bannered)
- **Evidence (re-verified live):** `curl /control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0, below_ss_count:0}`, `fill_rate.portfolio_fill_rate_3m:null`, `warning:"mv_control_tower_kpis not yet refreshed. Run make refresh-mvs-tiered"`. Command Center / Control Tower / AI Planner tiles show Portfolio Health 0/100, Fill Rate (3m) "--", "Portfolio Trend: No trend data available." The amber "Portfolio health data unavailable — Health KPIs are stale and showing zeros, they are not a sign of a healthy portfolio" banner is shown, so this is **not a trust hazard**.
- **Contrast:** The `exceptions` block in the same payload is correct and live (`open_exceptions_total:6142, critical_exceptions:2465, recommended_order_value:$246,723, source:"fact_replenishment_exceptions"`) via the cycle-3 live fallback. Health and fill rate still have no equivalent live fallback.
- **Root cause:** `api/routers/.../control_tower.py get_control_tower_kpis` reads only `mv_control_tower_kpis` for the health + fill_rate blocks; the live fallback (`_exceptions_fallback`) covers exceptions only.
- **Acceptance criterion:** After `make refresh-mvs-tiered` the health/fill tiles show non-zero numbers; OR a live fallback computes below-SS coverage / portfolio fill rate from base tables when the MV is stale (mirroring the exceptions fallback). Either removes the "0/100" + "--" presentation.

### F4.5 — Customer Analytics **Store Type** filter still lists ~275 raw free-text taxonomy values; Channel is clean  [SEV: P2]  (carried; Channel half resolved)
- **Evidence (re-verified):** `customerAnalytics` digest Store Type dropdown lists the full raw taxonomy — `**OBSOLETE **`, `A CHAIN GROCERY`, `BAR`/`BARS`/`Sports Bar`/`R BAR/TAVERN/LOUNGE`/`TAV/BAR RB`/`B/W TAVERN/BAR`, 26 single-letter-prefixed codes (`A CHAIN GROCERY` … `Z NIGHT CLUB`), `UNKNOWN NO`/`UNKNOWN SS`/`UNKNOWN TR`, `User Defined`, etc. Channel is clean (21 entries: `Off Premise Chains`, `On premise`, `House Accounts`, …) — the cycle-4 `normalizeLabelOptions()` fix holds.
- **Root cause:** Source-data taxonomy problem, not a de-dup bug. `fetchCustomerAnalyticsFilterOptions` (`customer-analytics.ts`) cannot canonicalize ~275 genuinely-distinct strings; needs an upstream raw→canonical mapping (the way Channel Mix already buckets into Grocery / Mass / Liquor / Clubs).
- **Acceptance criterion:** Store Type dropdown shows ~10–15 canonical buckets backed by a mapping table, with raw codes hidden behind it or in a searchable disclosure; at minimum the obvious junk (`**OBSOLETE **`, `UNKNOWN *`, `User Defined`, single-letter-prefixed codes) is dropped.

### F6.2 — `/customer-analytics/concentration` returns 404, but it is a **dead unused route** (no UI calls it)  [SEV: P3]  (carried context, not a user break)
- **Evidence:** `curl /customer-analytics/concentration` → HTTP 404. Grep of `frontend/src/api/queries/` shows no fetcher hits `/concentration`; the "Customer Concentration" treemap panel is served by `/treemap` (HTTP 200, rich `tree`). So the 404 is never triggered in the app.
- **Acceptance criterion:** Either remove the dangling `/concentration` route reference, or implement it — but since nothing calls it, this is cleanup, not a defect. Recorded so future scans don't mistake the 404 for a broken panel.

---

## Tabs working well (no action)
- **Command Center / Control Tower / AI Planner** — feed lists the real critical replenishment exceptions (50+ CRITICAL rows: 627099 @ 1401-BULK $572, 664631 $292, 913305 $279 …) reconciling 1:1 with the Inventory Action Feed; KPI tile (6142 / 2465 critical) and feed agree. Only health/fill tiles are MV-stale (F4.3, honestly bannered).
- **Inventory Planning** — Action Feed: 20 critical actions, $3.6K at risk, individual SKUs reconciling with the 6,142-row exceptions table. Strongest tab.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias, forecast-vs-actual series, lag curve (63→75%), heatmap negatives floored to `<0%*`, **cluster comparison table negatives now floored to `<0%*` (F4.4 RESOLVED)**.
- **Item Analysis** — defaults to a healthy sample DFU (Item 105430 @ 1401-BULK, 77.03% accuracy / 22.97% WAPE / 0.21 bias / FCST 4K / ACTUAL 3.3K), full chart + SHAP + Forecast KPIs + DQ Corrections.
- **Demand History** — Workbench lists 50/50 series with volume + sparkline + MoM%.
- **Customer Map** — 23.0M cases, 98% fill, 461K lost sales, 32.5K active customers; State dropdown clean, Channel clean, treemap/channel-mix/map/heatmap all render rich data and are now **all fast** (F5.1 RESOLVED). Store Type dropdown still dirty (F4.5).
- **Data Quality** — 64% health, 166 checks, 116 passed / 26 failed / 6 warnings, Last Run 56m ago, Check Catalog (83) fully populated. Modulo the severity-vs-status badge clarity (F6.1).
- **Explorer** — fast raw data across 9 domains.
- **AI Planner FVA Backtest** — run history populated (succeeded/failed, provider/DFU/rec counts).
- **FVA & ROI** — External +5.2 pts vs Naive Seasonal (76,995 rows); Champion "No data" and AI/Planner "Coming Soon" are honest empty/reserved states (no defect).
- **Clusters / S&OP** — genuinely empty with honest empty states + run-pipeline / create-cycle CTAs. Not defects.

---

## Harness note (carried, not a product defect)
`controlTower` and `aiPlanner` routes still render Command Center content in the dump (entries 4 and 7 byte-identical to entry 0, all `textLen=5516`) — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries. Flagged so they aren't mistaken for additional broken tabs.

---

## Resolved since Cycle 5 (verified this cycle)
- **F5.1 (Item×State heatmap cold load ~9.4 s) — RESOLVED.** `mv_ca_item_state` MV live; cold `/customer-analytics/heatmap` 9.42 s → **0.43 s**, returns 25 items.
- **F4.4 (cluster comparison table raw negative accuracy) — RESOLVED.** Table now floors low-base rows to `★<0%*` with the WAPE caption, consistent with the heatmap.
