# Cycle 3 — Fixes Applied

Source findings: `usability3.md` (no `testinput3.md` planner file exists this cycle, so all items are usability). Strict TDD (red → green → refactor → live-verify) per item.

Environment: React UI :5173 → FastAPI :8000 (uvicorn --reload) → Postgres :5440. Backend edits hot-reload.

---

## U3.1 (P1) — CA chart-panel toggle pills used bare `bg-gray-100 text-gray-600` (no dark variant) — FIXED

- **Wrong:** The metric/grain/group-by toggle pills inside six Customer-Analytics chart panels (9 call sites) plus the tab search clear-× used raw Tailwind grays with no `dark:` variant, rendering gray-on-gray (illegible) in Dark theme. The cycle-2 U2.3 fix reached the dropdown/ranking/Clear surfaces but not these pills.
- **Fix (files):**
  - `frontend/src/tabs/customer-analytics/togglePill.ts` — NEW shared `togglePillClass(active)` helper: inactive uses `bg-muted text-muted-foreground hover:bg-accent`, active uses `bg-primary text-primary-foreground font-medium` (single source of truth).
  - `CustomerDemandMap.tsx`, `CustomerHeatmap.tsx` (3 sites incl. Reset-Sort), `ChannelSunburst.tsx`, `OosImpactBubble.tsx`, `SegmentSparklines.tsx` — swapped bare-gray pill classes to `togglePillClass()`. SegmentSparklines row bg `bg-red-50`/`hover:bg-gray-50` → `bg-destructive/10`/`hover:bg-accent`.
  - `frontend/src/tabs/CustomerAnalyticsTab.tsx:191` — clear-× `text-gray-400 hover:text-gray-600` → `text-muted-foreground hover:text-foreground`.
- **Test:** `src/tabs/__tests__/CustomerAnalyticsTab.theme.test.ts` — extended the U2.3 source-guard with a `it.each` over the five panels asserting no bare `bg-gray-*`/`bg-white`/`text-gray-*`, plus a clear-× guard on the tab.
- **Red→Green:** RED: `it.each` panels + clear-× failed (`Expected [] but got ['text-gray-600'...]`, 6 failing assertions). GREEN: after the swaps, all guard assertions pass. (10 passed in the theme + aria files.)
- **Acceptance met:** YES — grep for bare grays on the toggle surfaces returns none; tokens adapt to Light/Soft/Dark.

## U3.3 (P2) — CA toggles encoded active state by color alone, no `aria-pressed` — FIXED

- **Wrong:** The map/heatmap/sunburst/bubble/sparkline toggle pills exposed selection only via `bg-indigo-600` vs gray — no `aria-pressed`/`aria-selected`. Screen-reader and color-blind users could not tell which metric was active.
- **Fix (files):** Added `aria-pressed={selected}` to every toggle `<button>` in `CustomerDemandMap.tsx` (group-by + metric), `CustomerHeatmap.tsx` (metric + value-mode), `ChannelSunburst.tsx`, `OosImpactBubble.tsx`, `SegmentSparklines.tsx`. The active pill also gains a non-color cue (`font-medium`, via `togglePillClass`).
- **Test:** `src/tabs/customer-analytics/__tests__/CustomerHeatmap.aria.test.tsx` — renders the heatmap (ECharts stubbed) and asserts the active "Demand" button has `aria-pressed="true"` and inactive "Customers" has `aria-pressed="false"`.
- **Red→Green:** RED: `expect(active).toHaveAttribute("aria-pressed","true")` → `Received: null`. GREEN: passes after adding `aria-pressed`.
- **Acceptance met:** YES.

## U3.2 (P2) — "Total Demand" rendered in two formats on one CA screen — FIXED

- **Wrong:** The KPI tile read "Total Demand 23.0M cases" (`formatCompactKMB`) while the Customer Demand Map footer directly below read "22,986,295 cases total demand" (`formatInt`). Same metric, same screen, two precisions.
- **Fix (files):** `frontend/src/tabs/customer-analytics/CustomerDemandMap.tsx` — footer total-demand now uses `formatCompactKMB` (imported as `fmtCompact`), matching the KPI tile ("23.0M"). Customer count stays `formatInt`. Merged the two formatter import lines.
- **Test:** `src/tabs/customer-analytics/__tests__/CustomerDemandMap.format.test.tsx` — mocks the map query with `total_demand: 22986295` and asserts the footer renders the compact `formatCompactKMB(...)` string and NOT `22,986,295`.
- **Red→Green:** RED (footer temporarily reverted to `formatInt`): `Unable to find element with text /23.0M cases total demand/`. GREEN: passes with `formatCompactKMB`. Verified the revert→restore cycle.
- **Acceptance met:** YES — both values derive from the same formatter helper.

## U3.4 (P3, root-caused) — CA "→ 0.0% MoM" badge conflated genuine-zero vs no-data — FIXED

- **Wrong:** `concentration_top10` and `order_demand_ratio` had NO month-over-month computed — the backend hardcoded `delta: 0.0`. The UI rendered the same neutral "→ 0.0% MoM" for a true zero change and for "no prior period to compare". Root cause was the fabricated backend zero, not just UI presentation.
- **Fix (files):**
  - `api/routers/intelligence/customer_analytics/kpis.py` — the two metrics now report `delta: None` (honest "no MoM anchor") instead of `0.0`.
  - `frontend/src/api/queries/customer-analytics.ts` — `KpiMetric.delta` is now `number | null`.
  - `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx` — `DeltaBadge` renders "— no prior period" (aria "No prior period to compare") for a null delta; numeric deltas keep the existing direction-aware badge.
- **Tests:**
  - Backend `tests/api/test_customer_analytics.py::test_kpis_concentration_and_ratio_have_null_delta` — asserts those two deltas are `None` while `total_demand` stays `25.0`.
  - Frontend `src/tabs/customer-analytics/__tests__/KpiSummaryCards.test.tsx` — null-delta metrics render two "no prior period" affordances and never a "0.0% MoM" badge.
- **Red→Green:** Backend RED: `assert 0.0 is None`. GREEN: 31 passed. Frontend RED: `expected length 2 of /no prior period/` (0 found). GREEN: 8 passed.
- **Live-verify:** `curl /customer-analytics/kpis` → `concentration_top10 -> null`, `order_demand_ratio -> null` (was `0.0`).
- **Acceptance met:** YES — null delta and zero delta now produce distinct text.

## U3.5 (P3, carried U2.7) — Item Analysis breadcrumb showed a bare numeric code — FIXED

- **Wrong:** The breadcrumb rendered `Item 185690` using only `item_id`, while the product ("DAMMANN JARDIN BLEU TEA(96CT)") is human-readable everywhere else. The description was never plumbed through `/sku/analysis`.
- **Fix (files):**
  - `api/routers/forecasting/analysis.py` — `/sku/analysis` now resolves `dim_item.item_desc` once per item (parameterized `%s`, `fetchone`) and returns it as a top-level `item_desc`.
  - `frontend/src/types/index.ts` — `SkuAnalysisPayload.item_desc?: string | null`.
  - `frontend/src/tabs/item-analysis/breadcrumb.ts` — NEW pure `itemBreadcrumbLabel(id, desc)` ("Item <id> — <desc>", falls back to bare code, de-dupes an id-prefixed desc).
  - `frontend/src/tabs/ItemAnalysisTab.tsx` — breadcrumb uses `itemBreadcrumbLabel(skuItem, skuData?.item_desc)`.
- **Tests:** Backend `test_analysis.py::test_sku_analysis_returns_item_desc`; frontend `item-analysis/__tests__/breadcrumb.test.ts` (3 cases: with desc, loading fallback, id-prefix de-dupe).
- **Red→Green:** Backend RED: `assert 'item_desc' in data` failed. GREEN: 17 passed. Frontend RED: module-not-found. GREEN: 3 passed.
- **Live-verify:** `curl /sku/analysis?item=185690&mode=item_at_all_locations` → `item_desc: 'DAMMANN JARDIN BLEU TEA(96CT)'`.
- **Acceptance met:** YES.

---

## Deferred

- **U3.6 / U2.5 / U1.7** (P2, simplification) — 7 tabs > 600 lines (CommandCenterTab 941). Mechanical multi-file splits, high churn / low correctness value; deferred again per prior cycles.
- **U2.6** (P2, IA) — retired tab keys (`?tab=aiPlanner`/`controlTower`) render Command Center with stale URL. Needs a router/redirect + product decision.
- **U1.3** (P2, consistency) — raw `fetch()` in 7 model-tuning panels. Pre-existing tsc errors in those files make a clean slice risky (carried).
- **U2.8** (P3) — sidebar vs page-heading naming; cosmetic.

## Risk / notes

- The `delta: number | null` change is API-shape-visible. Frontend `KpiMetric` and `DeltaBadge` were updated in the same change; the OpenAPI generated schema (`schema.ts`) is regenerated separately via `npm run gen:types` and was not touched (no runtime dependency on it for these handlers).
- `togglePill.ts` is a deliberate small shared module (not a backward-compat shim) consumed by 5 panels — single source of truth for the pill classes.
- Pre-existing, NOT introduced this cycle: 2 failing frontend tests (`AppSidebar`, `DemandReferencePanel`) and ~72 baseline tsc `--noEmit` errors (mostly `ExportButtons getData` generic mismatches). Verified via git-stash baseline: tests fail identically without my changes; tsc error count went 72 → 70 (my changes reduced, not increased, errors). The `zip(... )` ruff hint on `analysis.py:208` is on a pre-existing line, not my added query.
