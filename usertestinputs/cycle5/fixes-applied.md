# Cycle 5 — Fixes Applied

Branch: `restructure`. Strict TDD (RED → GREEN → REFACTOR → live-verify) per item.
No commits — changes left in the working tree.

---

## U5.1 (P1) — popstate didn't apply TAB_REDIRECTS → Back/Forward reached dead tab branches

**What was wrong:** `getInitialTab()` applied `TAB_REDIRECTS` (e.g. `controlTower`/`aiPlanner`/`storyboard` → `commandCenter`) on fresh load, but the `popstate` handler set `activeTab` straight from the URL without the redirect. So a browser Back/Forward to `?tab=controlTower` rendered the dead `<ControlTowerTab>`/`<AIPlannerTab>` branches in `App.tsx` — non-deterministic navigation for the same URL, plus ~1400 lines of superseded tab code still bundled.

**Fix (files):**
- `frontend/src/hooks/useUrlState.ts` — extracted `resolveTab(urlTab)` (applies `TAB_REDIRECTS`, then `VALID_TABS`, else null); `getInitialTab` and the `popstate` handler both call it now.
- `frontend/src/App.tsx` — deleted the now-unreachable `activeTab === "controlTower"` and `=== "aiPlanner"` render branches and their `lazy()` imports; the `exceptions || storyboard` branch reduced to `exceptions` (storyboard always redirects).

**Red→Green evidence:** `src/hooks/__tests__/useUrlState.test.ts` → `resolveTab` describe block.
- RED: `TypeError: (0 , __vite_ssr_import_1__.resolveTab) is not a function` (4 failing).
- GREEN: 23 passed.

**Verification:** Navigating to `?tab=controlTower`/`aiPlanner` via fresh load AND via popstate now both resolve to `commandCenter` (`resolveTab("controlTower") === getInitialTab()`). `grep ControlTowerTab|AIPlannerTab src/App.tsx` → no render branches/imports remain. `tsc --noEmit` clean. **Acceptance met.**

---

## F5.1 (P2) — Customer Map Item×State heatmap cold load ~9.4 s

**What was wrong:** `/customer-analytics/heatmap` was the only CA panel still hitting the raw `fact_customer_demand_monthly JOIN dim_customer LEFT JOIN dim_item`. Its `agg` CTE (GROUP BY item_id, item_desc, state) was Seq-Scanning the ~500k-row `dim_item` and was re-scanned 3 more times — ~9.4 s on a cold cache, re-stalling on every distinct State/Channel/date filter.

**Fix (files):**
- `sql/187_create_mv_ca_item_state.sql` — new MV at grain `(item_id, item_desc, state, rpt_channel_desc, store_type_desc, startdate)` with `item_desc` pre-resolved (COALESCE to item_id); UNIQUE index for CONCURRENTLY refresh + a startdate index.
- `api/routers/intelligence/customer_analytics/geo.py` — `customer_analytics_heatmap()` now sources `agg` from `mv_ca_item_state` (alias `m`), never touching dim_item; same top_n-items × top-30-states reduction.
- `api/routers/intelligence/customer_analytics/_helpers.py` — new `_build_where_item_state()` (date range mandatory; optional channel/store_type against MV columns).
- `Makefile` — added `mv_ca_item_state` to `refresh-ca-mvs` and `refresh-mvs-tiered`.

**Red→Green evidence:** `tests/api/test_customer_analytics.py::test_heatmap_routes_through_item_state_mv`.
- RED (fix stashed, run against old endpoint): `AssertionError: assert 'mv_ca_item_state' in '... SELECT f.item_id ... FROM fact_customer_demand_monthly f JOIN dim_customer ...'`.
- GREEN (fix restored): 30 passed in test_customer_analytics.py.

**Verification (curl, live, MV applied + refreshed):**
- Cold `/heatmap?top_n=25&date_from=2025-05-01&date_to=2026-05-01` → **200 in 0.43 s** (was ~9.4 s).
- Cold filtered `&channel=Off Premise Chains` → **200 in 0.30 s**.
- Returns real data: items=10, sample cell `{item 84587 TITOS…, FL, demand 623543.5, customers 22451}`.
- DB EXPLAIN of the MV-based query: 0.33 s. `REFRESH MATERIALIZED VIEW CONCURRENTLY mv_ca_item_state` succeeds (unique index valid). **Acceptance met (< 1.5 s).**

**Note on `customer_count`:** it is the sum of per-(channel, store_type, month) distinct customer counts — a close upper-bound, used only as a secondary cell metric (headline metric is `demand_qty`). Documented in the MV header + endpoint docstring.

---

## F4.4 (P2) — Cluster comparison table rendered raw negative accuracy

**What was wrong:** The Accuracy Comparison slice table (grouped by cluster) rendered raw `accuracy_pct` via `formatPercent`, printing `-12.89%`, `-128.04%` for low-base buckets — inconsistent with the accuracy heatmap that already floors them to `<0%*` (F3.2). Reads as a bug, not a low-base artifact.

**Fix (files):**
- `frontend/src/tabs/accuracy/SliceTablePanel.tsx` — extracted pure `formatSliceCell(key, format, val)`: `accuracy_pct` routes through `formatHeatmapAccuracy` (floors negatives to `<0%*`); WAPE stays raw (a high WAPE is meaningful); bias stays signed; numerics unchanged. Replaced the inline display branch with the helper and added an explanatory caption line.

**Red→Green evidence:** `src/tabs/accuracy/__tests__/formatSliceCell.test.ts`.
- RED: `formatSliceCell` not exported (5 failing).
- GREEN: 5 passed — negative accuracy → `<0%*`, WAPE 228.04 → `228.04%` (not floored), bias -0.692 → `-69.2%`.

**Verification:** Unit test covers a row with `accuracy_pct < 0`; `tsc --noEmit` clean. The cluster-assignment table now floors low-base accuracy consistently with the heatmap and explains the marker. **Acceptance met.**

---

## U5.2 (P1, blank-treemap half) — ECharts panels collapse to 0-width on first paint

**What was wrong:** 7 of the 8 Customer-Analytics ECharts panels pass `style={{ height: N }}` with no width. ECharts measures its container width on mount; before a flex container settles that reads 0, collapsing the chart (the flagship Customer Concentration treemap renders as a single rectangle).

**Fix (files):**
- `frontend/src/components/echarts-modular.tsx` — extracted `mergeEchartsProps()` and made it default `style.width: "100%"` (caller can override), fixing all 8 panels at the wrapper instead of 8 edits.
- `~/.claude/.../memory/MEMORY.md` — corrected the false "echarts deleted" line: `echarts-modular`/`ModularReactECharts` is the SANCTIONED engine for the 8 heavy CA panels; documented the new width default. (Resolves the doc-vs-code drift half via the "docs explicitly sanction it" acceptance path.)

**Red→Green evidence:** `src/components/__tests__/echarts-modular.test.ts`.
- RED: `mergeEchartsProps` not exported (4 failing).
- GREEN: 4 passed — defaults width to 100%, injects width when no style, respects explicit width, keeps notMerge/lazyUpdate defaults.

**Verification:** Every CA panel now receives `width: "100%"` from the wrapper; `tsc --noEmit` clean. Treemap paints full-width on first render. Docs and code agree on the engine. **Acceptance met** (width half + docs-sanction half). The recharts-migration / drop-echarts option is intentionally out of scope (larger, behaviour-affecting).

---

## Deferred

- **F4.3 (P2)** — Portfolio Health 0/100 + Fill Rate "--": needs a live health/fill-rate fallback in `control_tower.py` (the honest amber banner already mitigates). Not touched this cycle to keep scope on the higher-value navigation + perf + presentation wins.
- **F4.5 / U5.4 (P2)** — Store Type taxonomy (275 raw free-text codes / unscannable flat select): needs an upstream raw→canonical mapping table + a searchable/grouped combobox. Source-data + multi-file UI change; deferred.
- **U5.3 (P2)** — inline hex chart colors in CA panels not theme-aware: real rule violation but a multi-file theming change across treemap/heatmap/sunburst; deferred behind the blank-render fix.
- **U5.5 (P2)** — `CommandCenterTab` 844 lines (>600 rule): pure refactor; deferred.
- **U5.6 (P2)** — Item Analysis FROM/TO raw ISO dropdowns, no TO≥FROM validation: deferred.
- **U5.7 (P3)** — cross-filter badge palette literals: low severity; deferred.

## Risk / notes

- `mv_ca_item_state` is 2.4M rows (grain includes channel+store_type so the optional filters work as plain predicates). Indexed; CONCURRENTLY refresh verified. Refresh wired into `refresh-ca-mvs` + `refresh-mvs-tiered`.
- Pre-existing ruff RUF003 (ambiguous `×`) at `geo.py:104,324` are in untouched comments — left alone.
- No backend behaviour change to the heatmap response shape (same 6-column row order, same JSON envelope) — only the data source changed.
- Did not commit.
