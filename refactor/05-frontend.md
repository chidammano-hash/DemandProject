## Frontend — Refactor Opportunities

_Scope: `frontend/src/` (tabs, components, api/queries, hooks, lib); generated `schema.ts` excluded. Read-only audit — no code changed._

### Quick wins
- Retired `EChartContainer` is still live: `components/EChartContainer.tsx:141` → `components/ForecastTrendChart.tsx:72` → rendered at `tabs/AggregateAnalysisTab.tsx:503`. Reintroduction is forbidden — migrate to recharts and delete both files (363 LoC).
- `theme=` passed as a chart prop at `components/ForecastTrendChart.tsx:72` and `components/EChartContainer.tsx:141` — the only two violations of the "charts never accept theme" rule (both in the dead path above).
- Duplicate private `fetchJson` copies at `api/queries/customer-analytics.ts:295` and `api/queries/demand-history.ts:118` lack the canonical `{detail}`/status parsing in `api/queries/core.ts:104` — replace with the canonical import.
- 16 `: any` in queries layer — `api/queries/feature-lab.ts` (91,94,108,111,128,155,188,191) and `api/queries/cluster-eda.ts` (66,70,87,98,106,122,135,146).
- `MODEL_PREFIX` constant duplicated 3× — `tabs/model-tuning/EnhancedComparisonPanel.tsx:79`, `ExperimentBuilder.tsx:67`, `EnhancedPromoteModal.tsx:52` — hoist to `api/queries/unified-model-tuning.ts`.
- 72 local `fmt*`/`format*` helpers across tabs/components despite `lib/formatters.ts` exporting `formatCurrency`/`formatPct`/`formatInt`/`formatDate` (e.g. `tabs/inv-planning/FinancialPlanPanel.tsx:27`, `ScenarioPlanningPanel.tsx:34`, `OpenPOPanel.tsx:20`, `storyboard/storyboardShared.ts:72`).
- `tabs/clusters/ClusterComparisonPanel.tsx` (600) and `tabs/InvPlanningTab.tsx` (705) at/over the 600-line tab limit.
- 240 inline hex literals in tabs/components; worst single file `tabs/item-analysis/UnifiedChartPanel.tsx` (31) — a hardcoded per-model color map (35-59) that belongs in a shared theme/color module.

### Ranked opportunities

1. **Systemic raw-fetch bypass of `fetchJson` in the queries layer**
   - Files: `api/queries/champion-experiments.ts` (15), `inv-planning-signals.ts` (10), `lgbm-tuning.ts` (6), `tuning-chat.ts` (6), `inv-planning-safety-stock.ts` (6), `accuracy-budget.ts` (5), `cluster-eda.ts` (5), `feature-lab.ts` (5), `sql-runner.ts` (3), `unified-model-tuning.ts` (3), `expsys.ts` (2), `sku-features.ts` (2), `cluster-experiments.ts` (1), `backtest-management.ts` (1) — 70+ raw `fetch(` across 14 modules
   - Problem: Bypass the canonical `fetchJson` (`core.ts:104`), so no FastAPI `{detail}` parsing or `status` attachment the global error handler/toast sanitizer relies on. Each reimplements `if (!res.ok) throw`.
   - Proposed change: Replace every `fetch(...)`+manual handling with `fetchJson<T>(url, init)`; delete the two local copies and import from `core.ts`.
   - Impact: High (consistent error UX, ~70 boilerplate blocks, enforces the rule) · Effort: M-H · Risk: M (POST/PUT bodies, non-JSON sql-runner responses need care)

2. **`feature-lab.ts` / `cluster-eda.ts`: replace `any` response shaping with typed interfaces**
   - Problem: Both fetch raw then `.map((f: any) => ...)` with `?? fallback` field aliasing (e.g. `f.shap_value ?? f.mean_abs_shap`), masking schema drift.
   - Proposed change: Mirror the backend Pydantic response as a TS interface (raw + normalized), fetch via `fetchJson<RawT>`, narrow with typed mappers; pin the unstable backend response.
   - Impact: High · Effort: M · Risk: L-M

3. **Split `tabs/forecast/ForecastPanel.tsx` (1215)** — worst offender, 2× the limit
   - Problem: 8 `useQuery`, 9 `useState`, 5 `useMemo`, 4 distinct Card sections (model selection ~500-665, training status ~842-1003, pipeline config ~1004-1037, generate ~1038-1130, recent jobs ~1131+).
   - Proposed change: Extract `ModelSelectionCard`, `TrainingStatusCard`, `PipelineConfigCard`, `GenerateForecastCard`, `RecentJobsCard`; lift the 8 queries into `useForecastPanelData()`.
   - Impact: High · Effort: M · Risk: Low

4. **Split `tabs/item-analysis/UnifiedChartPanel.tsx` (1121) + extract color map**
   - Problem: 31 inline hex, 9 `useState`/7 `useMemo`/3 `useEffect`; mixes a model-color map (35-59), localStorage helpers (117-126), three memoized sub-menus.
   - Proposed change: Move color maps to `lib/chartColors`/`useChartColors`; extract the defaults-menu cluster + localStorage helpers + a `useChartMeasureToggles()` hook.
   - Impact: High · Effort: M-H · Risk: M (localStorage + memo behavior)

5. **Consolidate ~72 ad-hoc formatters into `lib/formatters.ts`**
   - Files: `tabs/inv-planning/{FinancialPlanPanel,ScenarioPlanningPanel,OpenPOPanel,ProcurementPanel,DemandForecastPanel}.tsx`, `tabs/AiPlannerFvaTab.tsx:57-68`, `tabs/storyboard/storyboardShared.ts:72-77`, `tabs/accuracy/BiasCorrectionsPanel.tsx:56`, `tabs/ai-planner/aiPlannerShared.ts:131`, +~20
   - Proposed change: Delete locals; import from `lib/formatters.ts`; add the 2-3 genuinely missing variants there once.
   - Impact: High · Effort: M · Risk: Low

6. **Centralize inline hex colors (240) into theme/`useChartColors`**
   - Files: top offenders `UnifiedChartPanel.tsx` (31), `components/ScenarioCharts.tsx` (26), `tabs/customer-analytics/ChannelSunburst.tsx` (14), `tabs/lgbm-tuning/FeatureLabPanel.tsx` (13), `tabs/inventory/TrendChartPanel.tsx` (12), `tabs/inv-planning/ServiceLevelWaterfallPanel.tsx` (12), `tabs/dfu-analysis/DfuShapPanel.tsx` (10)
   - Proposed change: Define semantic palettes (status, model series, heatmap scale) once, read via `useChartColors()`/`useThemeContext()` (25 files already import it).
   - Impact: High · Effort: M-H · Risk: M (visual regressions; do per-file)

7. **Unify the Experiment Builder / Experiments Panel families**
   - Files: `tabs/champion/ChampionExperimentBuilder.tsx` (887), `tabs/clusters/ClusterExperimentBuilder.tsx` (392), `tabs/model-tuning/ExperimentBuilder.tsx` (585); `tabs/champion/ChampionExperimentsPanel.tsx` (489) vs `tabs/clusters/ClusterExperimentsPanel.tsx` (778)
   - Problem: Three near-parallel "builders" and two parallel "experiments list/promote" panels with the same submit/poll/promote/cancel lifecycle + table scaffolding.
   - Proposed change: Extract a shared `useExperimentLifecycle()` hook + generic `<ExperimentsTable>`/`<ExperimentForm>` parameterized by domain config. (Builder-by-builder; high risk if done at once.)
   - Impact: High (dedup across 3100+ LoC) · Effort: H · Risk: M-H

8. **Split `tabs/CommandCenterTab.tsx` (941)**
   - Proposed change: Move normalize/severity helpers (`mapRuleSeverity`, `normalizeAiInsight`, `dropRedundantIdentitySuffix`) to `tabs/command-center/exceptions.ts`; extract `KpiSummaryCard` (721) + `ExceptionFeedCard` (812); lift queries into `useCommandCenterData()`.
   - Impact: Med-High · Effort: M · Risk: Low

9. **Split `tabs/model-tuning/EnhancedComparisonPanel.tsx` (999) and dedup vs lgbm `ComparisonPanel`**
   - Files: `EnhancedComparisonPanel.tsx` (999), `tabs/lgbm-tuning/ComparisonPanel.tsx` (766)
   - Problem: 999 lines with self-contained `DeltaIndicator`/`MetricCard`/`VerdictBadge`/`ComparisonBarChart`/`ClusterDeltaTable` + a raw-fetch compare (line 98). Overlaps the older lgbm panel (model-tuning appears to be the successor).
   - Proposed change: Extract the 5 presentational components to `tabs/model-tuning/comparison/`; route compare via `unified-model-tuning.ts` `fetchTuningComparison2` (line 333). Evaluate retiring the lgbm-tuning tree.
   - Impact: Med-High · Effort: M (split) / H (lgbm retirement) · Risk: Low / Med

10. **Split remaining over-limit tabs/panels (15 files at/over 600)**
    - Files: `WorkbenchPanel.tsx` (842), `ClusterExperimentsPanel.tsx` (778), `lgbm-tuning/ComparisonPanel.tsx` (766), `inv-planning/SafetyStockPanel.tsx` (759), `ExceptionQueuePanel.tsx` (752), `InvPlanningTab.tsx` (705), `AccuracyBudgetPanel.tsx` (693), `DataQualityTab.tsx` (676), `StoryboardTab.tsx` (672), `AggregateAnalysisTab.tsx` (668), `SettingsTab.tsx` (649), `AiPlannerFvaTab.tsx` (637), `ClusterComparisonPanel.tsx` (600)
    - Proposed change: Standard subpanel-extraction per file; `WorkbenchPanel`/`ClusterExperimentsPanel` also warrant state-into-hook extraction.
    - Impact: Med · Effort: H (cumulative) · Risk: Low per file

11. **`ChampionExperimentBuilder` state explosion (15 `useState`, 3 `useEffect`)**
    - Proposed change: Collapse form state into a `useReducer` or typed `useExperimentForm()` hook; pass `state`/`dispatch`. Pairs with #7.
    - Impact: Med · Effort: M · Risk: M

12. **Inline color-map duplication for model series** (`UnifiedChartPanel.tsx:35-59`, `ScenarioCharts.tsx`, `DfuShapPanel.tsx`, `TrendChartPanel.tsx`)
    - Proposed change: One canonical `MODEL_SERIES_COLORS`/`MEASURE_COLORS` in a theme module via `useChartColors()`. (Subset of #6, callable independently.)
    - Impact: Med · Effort: L-M · Risk: Low

**Highest leverage:** #1, #2, #5, #6 enforce four stated conventions at scale and remove the most boilerplate; #3, #4, #8, #9 are the highest-value individual splits. Resolve the `EChartContainer`/`ForecastTrendChart` dead path (quick wins) first — it actively violates a "do not reintroduce" rule.
