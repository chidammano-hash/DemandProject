## frontend/src/ — Files Not Needed

_Read-only audit. Confidence: SAFE = zero importers (excluding own test); PROBABLE = zero but minor doubt; INVESTIGATE = retired-but-still-wired, or dynamic refs._

_Method: for every candidate, searched importers with `rg -F` across `src/**/*.ts(x)` (default + named import styles, `@/` alias and relative paths), excluding the file's own source and its paired `__tests__` file. Query modules additionally checked for named-export consumption since the barrel `export *` makes them path-reachable without naming the file. Generated `src/api/generated/schema.ts` excluded from "importer" credit (it only documents backend routes)._

### SAFE to delete

- `src/components/AlertPanel.tsx` (80 LoC) — exported `AlertPanel` component imported by nothing in the app; only its own test references it. — evidence: `rg AlertPanel` across `src/` returns only `AlertPanel.tsx` + `__tests__/AlertPanel.test.tsx` (no tab/component/registry/string ref). — paired test: `src/components/__tests__/AlertPanel.test.tsx` (61 LoC)

- `src/hooks/useTabVisibility.ts` (24 LoC) — hook imported by nothing; only its own test. — evidence: `rg TabVisibility` across `src/` returns only `useTabVisibility.ts` + `__tests__/useTabVisibility.test.ts`. — paired test: `src/hooks/__tests__/useTabVisibility.test.ts` (68 LoC)

- `src/tabs/inventory/ItemDetailPanel.tsx` (140 LoC) — exported `ItemDetailPanel` panel imported by nothing; has no test at all. — evidence: `rg ItemDetailPanel` across `src/` returns only its own definition (interface + function). No `src/.../ItemDetailPanel.test.*` exists. — paired test: none

- `src/tabs/model-tuning/PipelineConfigPanel.tsx` (439 LoC) — exported `PipelineConfigPanel` imported only by its own test. ModelTuningTab uses the unrelated query `fetchPipelineConfig`, not this panel; the panel is not registered in any pipeline-stage tab map. — evidence: `rg PipelineConfigPanel` across `src/` returns only the source + `__tests__/PipelineConfigPanel.test.tsx`; ModelTuningTab.tsx imports `fetchPipelineConfig` (query) and `PipelineStage` (type) but never the panel. — paired test: `src/tabs/__tests__/PipelineConfigPanel.test.tsx`

### PROBABLE

- `src/api/queries/expsys.ts` (62 LoC) — query module re-exported by the barrel (`export * from "./expsys"`) but none of its exports (`expSysKeys`, `fetchExpSysLagAccuracy`, `fetchExpSysStatus`, and the 4 `ExpSys*` interfaces) are consumed by any tab/component/hook or any test. The backend `/expsys/*` routes still exist in the generated schema, so this is a dangling client with no UI. — evidence: `rg expsys|ExpSys` across `src/` returns only `expsys.ts`, the barrel line, and `src/api/generated/schema.ts` (route docs only). Confidence not SAFE only because a barrel `export *` could theoretically be re-consumed; verified zero named-export consumers, so removal is low-risk. — paired test: none. Note: removing it requires deleting its line in `src/api/queries/index.ts`.

### INVESTIGATE  (retired-but-still-wired — migrate first)

- `src/components/EChartContainer.tsx` (174 LoC) + `src/components/__tests__/EChartContainer.test.tsx` (114 LoC) — `EChartContainer` is documented as RETIRED in CLAUDE.md / MEMORY ("EChartContainer is retired; do NOT reintroduce it") and the sanctioned engine is recharts (or `ModularReactECharts` for the 8 heavy CA panels). It is NOT yet unimported: it is the chart renderer used by `ForecastTrendChart.tsx`. Two tab tests (`ItemAnalysisTab.test.tsx`, `AggregateAnalysisTab.test.tsx`) still `vi.mock("@/components/EChartContainer")`, but the live `ItemAnalysisTab.tsx` no longer imports it (stale mock) — only the `ForecastTrendChart` chain keeps it alive. — must migrate first: migrate `ForecastTrendChart` (below) off `EChartContainer`, then drop the stale `vi.mock` in the two tab tests, then delete `EChartContainer` + its test.

- `src/components/ForecastTrendChart.tsx` (189 LoC) + `src/components/__tests__/ForecastTrendChart.test.tsx` (211 LoC) — renders via the retired `EChartContainer` and is STILL WIRED into a live tab: `AggregateAnalysisTab.tsx` (lazy-loaded in `App.tsx`) imports `ForecastTrendChart` and renders `<ForecastTrendChart data={trendQ.data?.months ?? []} />` at line 429. — must migrate first: re-implement the forecast-vs-actual trend chart in `AggregateAnalysisTab` with recharts (or `ModularReactECharts`), then delete `ForecastTrendChart` + `EChartContainer` together and update both tab tests' mocks. Until that migration lands, neither file is safe to remove.

### NOT dead (checked, keep)

- All other `src/api/queries/*.ts` modules — every one is consumed by at least one non-test app file (verified named-export usage, not just barrel reachability). Notable partial-use modules kept whole: `control-tower.ts` (3 of 9 exports used by `CommandCenterTab`, which absorbed the deleted ControlTower feature), `purchaseOrders.ts`, `sourcing.ts`, `accuracy-budget.ts`, `cluster-eda.ts`, `feature-lab.ts`, `fill-rate.ts`, `filter-meta.ts`, `tuning-chat.ts`, `inv-planning-rebalancing.ts` — these have unused individual exports (file-level dead-export cleanup, out of scope here) but the files are live.
- `demand-history.ts`, `integration.ts`, `integration_chain.ts`, `supply.ts` — initially suspected (not directly in barrel) but all are actively imported by panels (`demand-history/*`, `IntegrationTab` + `components/integration/*`, `inv-planning/OpenPOPanel`) or re-exported by `inv-planning.ts`.
- All `src/tabs/*.tsx` top-level tabs — every one is lazy-registered in `App.tsx` (`ClustersTab` is intentionally kept: "removed from sidebar — still importable via URL").
- All other `src/components/*.tsx` (AlertPanel/EChartContainer/ForecastTrendChart excepted above), all `src/hooks/*` (useTabVisibility excepted), all `src/lib/*`, `src/constants/*`, `src/types/*`, `src/context/*` — each has ≥1 non-test importer.
- The ~22 "filename-mismatch" test files (`no-retired-tabs.test.ts`, `severityBadgeMigration.test.ts`, `formatHeatmapAccuracy.test.ts`, `WhatIfScenarios.test.tsx`, `DemandWorkbenchTab.test.tsx`, `JobsTuningIntegration.test.tsx`, `*.theme.test.ts`, `no-raw-fetch.test.ts`, etc.) — NOT orphans: they are behavioral/integration/lint-style tests that exercise live components (e.g. `WhatIfScenarios.test.tsx` drives `ClustersTab`; `DemandWorkbenchTab.test.tsx` drives `DemandHistoryTab`) rather than sharing a 1:1 source filename.

---

### Counts
- SAFE source files: 4 (`AlertPanel.tsx`, `useTabVisibility.ts`, `ItemDetailPanel.tsx`, `PipelineConfigPanel.tsx`) + 3 paired tests = 7 files, ~772 LoC source.
- PROBABLE: 1 query module (`expsys.ts`, 62 LoC; also remove its barrel line).
- INVESTIGATE: 2 retired-but-wired components (`EChartContainer.tsx`, `ForecastTrendChart.tsx`) + their 2 tests — blocked on migrating `AggregateAnalysisTab` to recharts first.
