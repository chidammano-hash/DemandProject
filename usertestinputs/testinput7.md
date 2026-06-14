# Demand Planner UX Findings — Cycle 7
_Persona: senior demand planner. Date: 2026-06-14. Method: cycle-7 live capture digest + dump (14 tabs, **0 console errors on every tab**, all 14 `ok=true`) + curl / Postgres / code confirmation. Branch: restructure._

## Summary

The product remains in **very good shape**. All 14 tabs loaded clean (14/14 `ok`, 0 console errors, no 500s). The cycle-6 NEW item **F6.1 (Data Quality Check Catalog severity-vs-status badge) is verified RESOLVED** — the catalog header now reads "Last Value (violations)", the leftmost Status column drives off `last_status`, and passing critical rows no longer read as failures.

There is **one NEW P2 trust/clarity issue** I found this cycle, in the Data Quality **Domain Health** grid (a sibling panel to the one fixed in F6.1): domain cards show health scores below 100% with breakdowns that read as perfectly healthy — e.g. "Forecast **75%** — 12 pass / 0 fail / 0 warn" and "Item **62.5%** — 10 pass / 0 fail / 0 warn". The gap is caused by **skipped** checks counted in the score denominator but never surfaced anywhere in the UI. A planner doing data-readiness sign-off cannot reconcile "0 failures" with "62.5% health".

Everything else reconciles with prior cycles. The remaining items are all **carried** data-state / presentation items (F4.3 health/fill MV stale but honestly bannered; F4.5 Store-Type taxonomy; F6.2 dead `/concentration` route). The Customer Map "Loading…" panels in the digest are again a below-fold `LazyPanel`/IntersectionObserver capture-timing artifact, not a defect.

**One new P2. No new P0/P1. newActionableCount (new unresolved P0/P1/P2) = 1.**

---

## NEW Findings

### F7.1 — Data Quality "Domain Health" cards show sub-100% scores with all-green breakdowns ("Forecast 75% — 12 pass / 0 fail / 0 warn"), because **skipped checks are counted in the score denominator but never shown**  [SEV: P2]  (NEW)
- **Workflow blocked:** Data-quality / data-readiness review before an S&OP demand sign-off. Not hard-blocked, but actively misleading — the central health signal contradicts its own breakdown.
- **Evidence:** Tab `dataQuality` (`screens/dataQuality.png`, digest lines 2734–2870, dump entry 10, 0 console errors). Domain Health cards:
  - `Forecast` badge **75%**, breakdown "12 pass · 0 fail · 0 warn" — but 12+0+0 = 12 ≠ the 16 total used for the score.
  - `Item` badge **62.5%**, breakdown "10 pass · 0 fail · 0 warn" — 10 ≠ 16.
  - `Location` badge **75%**, "6 pass · 0 fail · 0 warn" — 6 ≠ 8.
  - `Sales` badge **72.7%**, "16 pass · 0 fail · 2 warn" (=18) ≠ 22.
  - Summary KPI bar: "166 Total · 116 Passed · 26 Failed · 6 Warnings" — 116+26+6 = 148, **18 checks unaccounted for**.
  - Curl + DB proof:
    - `curl /data-quality/dashboard` → `forecast: {score:75.0, passed:12, failed:0, warnings:0, total:16}`, `item: {score:62.5, passed:10, failed:0, warnings:0, total:16}`. There is **no `skipped` field in the payload at all.**
    - `fact_dq_check_results` (last 24h): forecast = 12 pass / 0 fail / 0 warn / **4 skip** / 16 total; item = 10 / 0 / 0 / **6 skip** / 16; location = 6 / 0 / 0 / **2 skip** / 8; sales = 16 / 0 / 2 / **4 skip** / 22. Overall: 116 pass / 26 fail / 6 warn / **18 skip** / 166.
- **Root cause:**
  - Backend `api/routers/platform/data_quality.py` `dq_dashboard()` (lines 16–46): score = `passed / total` where `total = count(*)` over **all** statuses including `skip` (lines 22–25, 35–36). The response object (lines 37–44) emits `passed/failed/warnings/total` but **never `skipped`**, so the count is invisible downstream.
  - Frontend `frontend/src/tabs/DataQualityTab.tsx`: the domain card renders only `{d.passed} pass / {d.failed} fail / {d.warnings} warn` (lines 228–230) with the score badge `{d.score}%` (lines 223–224); the summary bar (lines 107–110) sums only pass/fail/warn. Skipped checks exist in neither the badge math display nor the breakdown, so the score looks unexplained.
  - Confirmed by the existing test `tests/api/test_data_quality.py:40` which bakes in `score == 80.0 # 8/10 * 100` with no skip handling — skips silently dilute every domain score.
- **Why P2 (not P1):** No data is wrong and nothing 500s — failed/warn counts are accurate, and the catalog (F6.1, now fixed) lets a planner drill in. But the headline domain-health number is the first thing scanned during readiness review, and "62.5% with zero failures" reads as either a broken score or a hidden problem. It erodes trust in the health grid.
- **Acceptance criterion:**
  1. `GET /data-quality/dashboard` returns a `skipped` count per domain (and the summary surfaces total skipped), so the breakdown reconciles: `passed + failed + warnings + skipped == total`.
  2. The Domain Health card either (a) shows the skip count alongside pass/fail/warn, or (b) excludes skipped checks from the score denominator (score = `passed / (passed+failed+warnings)`), so a domain with zero failures and zero warnings reads **100%**, not 62.5%. The summary KPI bar likewise accounts for all 166 checks (e.g. adds a "Skipped" tile) so 148 ≠ 166 no longer leaves 18 checks unexplained.
  3. A backend test asserts a domain with pass=10, fail=0, warn=0, skip=6 returns the chosen, consistent score (either 100% with skip surfaced, or the documented diluted value with `skipped:6` present) — and the current `score == 80.0` assertion is updated to reflect the decision rather than silently dropping skips.
- **Planner impact:** When I sign off data readiness before locking the demand plan, I scan Domain Health for anything below green. Today Forecast and Item show 75% / 62.5% with **no failures and no warnings**, so I either waste time hunting a non-existent problem or learn to distrust the score entirely — both bad for a control surface whose whole job is trustworthy at-a-glance health.

---

## Carried items (re-verified this cycle — NOT new)

### F4.3 — Command Center Portfolio Health 0/100 and Fill Rate "--" (health/fill-rate MV unpopulated; no live fallback)  [SEV: P2]  (carried, honestly bannered — 5th cycle)
- **Re-verified live:** `curl /control-tower/kpis` → `health.{total_dfus:0, avg_health_score:0, below_ss_count:0}`, `fill_rate.portfolio_fill_rate_3m:null`, `warning:"mv_control_tower_kpis not yet refreshed. Run make refresh-mvs-tiered"`. Command Center / Control Tower / AI Planner tiles show Portfolio Health 0/100, Fill Rate (3m) "--". The amber "Portfolio health data unavailable — these zeros are not a sign of a healthy portfolio" banner is shown, so **not a trust hazard**.
- **Contrast:** the `exceptions` block in the same payload is correct and live (6142 open / 2465 critical / $246,723, source `fact_replenishment_exceptions`) via the cycle-3 fallback. Only health + fill_rate lack the equivalent live fallback.
- **Root cause:** `control_tower.py get_control_tower_kpis` reads only `mv_control_tower_kpis` for the health + fill_rate blocks.
- **Acceptance criterion:** After `make refresh-mvs-tiered` the tiles show non-zero numbers; OR a live fallback computes below-SS coverage / portfolio fill rate from base tables when the MV is stale (mirroring the exceptions fallback).

### F4.5 — Customer Analytics **Store Type** filter still lists ~275 raw free-text taxonomy values; Channel is clean  [SEV: P2]  (carried; Channel half resolved)
- **Re-verified:** `customerAnalytics` digest Store Type dropdown lists the full raw taxonomy — `**OBSOLETE **`, single-letter-prefixed codes (`A CHAIN GROCERY` … `Z NIGHT CLUB`), `BAR`/`BARS`/`Sports Bar`/`TAV/BAR RB`, `UNKNOWN NO`/`UNKNOWN SS`/`UNKNOWN TR`, `User Defined`, etc. Channel is clean (~22 entries) — the cycle-4 `normalizeLabelOptions()` fix holds.
- **Root cause:** source-data taxonomy problem, not a de-dup bug. `fetchCustomerAnalyticsFilterOptions` (`customer-analytics.ts`) cannot canonicalize ~275 genuinely-distinct strings; needs an upstream raw→canonical mapping table.
- **Acceptance criterion:** Store Type dropdown shows ~10–15 canonical buckets backed by a mapping table (or a searchable disclosure); at minimum the obvious junk (`**OBSOLETE **`, `UNKNOWN *`, `User Defined`, single-letter-prefixed codes) is dropped.

### F6.2 — `/customer-analytics/concentration` returns 404, but it is a **dead unused route** (no UI calls it)  [SEV: P3]  (carried context, not a user break)
- **Re-verified:** `curl /customer-analytics/concentration` → 404; no fetcher in `frontend/src/api/queries/` hits `/concentration`; the "Customer Concentration" treemap is served by `/treemap` (200). Cleanup, not a defect — recorded so future scans don't mistake it for a broken panel.

---

## Tabs working well (no action)
- **Command Center / Control Tower / AI Planner** — feed lists the real critical replenishment exceptions (50+ CRITICAL rows: 627099 @ 1401-BULK $572, 664631 $292, 913305 $279 …) reconciling 1:1 with the Inventory Action Feed; KPI tile (6142 / 2465 critical) and feed agree. Only health/fill tiles are MV-stale (F4.3, honestly bannered).
- **Inventory Planning** — Today's Plan: 20 urgent, $4K at risk; Action Feed: 20 critical actions / $3.6K, individual SKUs reconciling with the 6,142-row exceptions table. Strongest tab.
- **Portfolio / Aggregate Accuracy** — 73.9% acc / 26.1% WAPE / 6.6% bias, forecast-vs-actual series, lag curve (63→75%), heatmap negatives floored to `<0%*` with caption, cluster comparison table negatives floored to `★<0%*` with WAPE caption (F4.4 holds).
- **Item Analysis** — defaults to a healthy DFU (Item 107395 @ 1401-BULK, 73.16% accuracy / 26.84% WAPE / 0.21 bias / FCST 499 / ACTUAL 411), full chart + SHAP + Forecast KPIs + DQ Corrections.
- **Demand History** — Workbench lists 50/50 series with volume + LAST + MoM%; the small LAST values (e.g. MEIOMI 46.7, KEND JACK 15) are honest most-recent-month volumes, not a formatting bug; >200% MoM rows carry the `*` low-base footnote (U6.5 holds).
- **Customer Map** — 22.99M cases, 32,469 customers; State dropdown clean, Channel clean, treemap/channel-mix/map/heatmap render rich data and are fast (F5.1 holds). Store Type dropdown still dirty (F4.5).
- **Data Quality** — 64% health, 166 checks, 116 passed / 26 failed / 6 warnings, Last Run 1h ago, Check Catalog (83) populated with the F6.1 "Last Value (violations)" header + status-driven icons. Modulo the Domain Health skip-accounting issue (F7.1).
- **Explorer** — fast raw data across 9 domains.
- **AI Planner FVA Backtest** — run history populated.
- **FVA & ROI** — External +5.2 pts vs Naive Seasonal (76,995 rows); Champion "No data" and AI/Planner "Coming Soon" are honest empty/reserved states.
- **Clusters / S&OP** — genuinely empty with honest empty states + run-pipeline / create-cycle CTAs. Not defects.

---

## Resolved since Cycle 6 (verified this cycle)
- **F6.1 (DQ Check Catalog severity-vs-status badge) — RESOLVED.** Catalog header now reads "Last Value (violations)"; the leftmost Status column drives off `last_status`; passing critical rows (e.g. `completeness_forecast_item_id` critical/pass/0.00) no longer read as failing-critical. Digest lines 2790+ confirm the new header + per-row status icons.

---

## Harness note (carried, not a product defect)
`controlTower` and `aiPlanner` routes still render Command Center content in the dump (entries 4 and 7 byte-identical to entry 0) — the harness `?tab=` fallback to the default tab for non-clickable sidebar entries. Flagged so they aren't mistaken for additional broken tabs.
