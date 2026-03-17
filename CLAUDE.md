# CLAUDE.md — Demand Studio

## Project Overview

**Demand Studio** is a unified demand forecasting analytics platform. It ingests sales and forecast data, stores it in PostgreSQL, and serves a React UI for interactive analytics.

**Working directory for all dev work:** `mvp/demand/`

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | Python + FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| DB Driver | psycopg v3 |
| Frontend | React + Vite + TypeScript |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts + ECharts |
| Database | PostgreSQL 16 |
| ML / Clustering | scikit-learn, pandas, scipy, matplotlib, seaborn |
| ML Tracking | MLflow |
| Job Scheduling | APScheduler 3.11 (BackgroundScheduler + ThreadPoolExecutor) |
| E2E Testing | Playwright |
| Python packaging | uv |
| Build | Make |
| Containers | Docker Compose |

---

## Key Files

| File | Purpose |
|---|---|
| `mvp/demand/common/domain_specs.py` | Central config: all 8 datasets (dimensions + facts + inventory) with columns, types, keys |
| `mvp/demand/api/main.py` | FastAPI backend — primary endpoints + mounts routers (clusters, jobs, etc.) |
| `mvp/demand/frontend/src/App.tsx` | React UI — sidebar layout shell (~200 lines, lazy-loaded tabs) |
| `mvp/demand/Makefile` | All dev commands |
| `mvp/demand/docker-compose.yml` | 2-service infra cluster (Postgres + MLflow) |
| `mvp/demand/scripts/normalize_dataset_csv.py` | Generic ETL: CSV → clean CSV |
| `mvp/demand/scripts/load_dataset_postgres.py` | Generic loader: clean CSV → PostgreSQL |
| `mvp/demand/sql/` | DDL for all tables, indexes, materialized views |
| `mvp/demand/sql/017_create_fact_inventory_snapshot.sql` | Inventory snapshot table DDL, indexes, materialized view |
| `mvp/demand/sql/019_inventory_forecast_view.sql` | Inventory-forecast bridge materialized view (Feature 37) |
| `mvp/demand/scripts/normalize_inventory_csv.py` | Inventory ETL: merge 14 monthly CSVs → single clean CSV |
| `mvp/demand/scripts/generate_clustering_features.py` | Feature engineering: 14 core features across 6 dimensions (volume, trend, seasonality, periodicity, intermittency, lifecycle) from sales history |
| `mvp/demand/scripts/train_clustering_model.py` | KMeans clustering with combined Silhouette + Calinski-Harabasz K selection, 5% min cluster size, MLflow logging |
| `mvp/demand/scripts/label_clusters.py` | Priority-ordered taxonomy labeling: Intermittency → Periodicity → Seasonality → Trend → Volatility → Volume (5 tiers) |
| `mvp/demand/scripts/update_cluster_assignments.py` | Write cluster labels to `dim_dfu.cluster_assignment` in Postgres |
| `mvp/demand/config/clustering_config.yaml` | Clustering hyperparameters: time_window_months: 36, k_range [5,18], min_cluster_size_pct: 5.0, combined scoring, priority-ordered labeling thresholds |
| `mvp/demand/config/model_competition.yaml` | Champion model selection: competing models, metric, lag, strategy |
| `mvp/demand/scripts/run_champion_selection.py` | Per-DFU champion selection: best-of-models via configurable strategy |
| `mvp/demand/common/champion_strategies.py` | 5 champion strategies (expanding, rolling, decay, ensemble, meta_learner) + registry + leak guards |
| `mvp/demand/scripts/simulate_champion_strategies.py` | Diagnostic: simulate all strategies, compare accuracy vs ceiling |
| `mvp/demand/scripts/train_meta_learner.py` | Meta-learner training: ceiling labels as ground truth, temporal split |
| `mvp/demand/common/backtest_framework.py` | Shared backtest orchestrator: `run_tree_backtest()`, timeframes, data loading, output saving; `_fill_predict_nans()`, `_predict_single_month(models, predict_data, feature_cols)`, `recursive` param (Feature 43); config-driven via algorithm_config.yaml (Feature 44); ml_cluster is always a hard feature |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix: lag/rolling features, future masking, `cat_dtype` parameter; `update_grid_with_predictions()` for recursive inference write-back (Feature 43) |
| `mvp/demand/common/metrics.py` | Shared accuracy metrics: WAPE, bias, accuracy % |
| `mvp/demand/common/mlflow_utils.py` | Shared MLflow logging wrapper for backtest runs |
| `mvp/demand/common/db.py` | Shared DB connection parameters |
| `mvp/demand/common/utils.py` | Shared utilities: `_ts()` timestamp helper, `load_config()` thread-safe YAML config loader with caching, `reset_config()` for tests |
| `mvp/demand/common/planning_date.py` | Shared planning date: `get_planning_date()` replaces `date.today()` across all scripts and routers |
| `mvp/demand/config/planning_config.yaml` | Planning date config: `planning_date` (frozen dev date) + `use_system_date` flag |
| `mvp/demand/common/constants.py` | Shared constants: `CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`, output columns, thresholds |
| `mvp/demand/config/algorithm_config.yaml` | Per-algorithm config: cluster_strategy (per_cluster/global), recursive, shap_select, shap_threshold, shap_top_n, shap_sample_size, tune_inline, params_file, default hyperparameters (Feature 44) |
| `mvp/demand/scripts/run_backtest.py` | LGBM backtest: per-cluster and global training functions; reads config/algorithm_config.yaml (Feature 44); ml_cluster is a hard feature |
| `mvp/demand/scripts/run_backtest_catboost.py` | CatBoost backtest: per-cluster and global training functions; reads config/algorithm_config.yaml (Feature 44); ml_cluster is a hard feature |
| `mvp/demand/scripts/run_backtest_xgboost.py` | XGBoost backtest: per-cluster and global training functions; reads config/algorithm_config.yaml (Feature 44); ml_cluster is a hard feature |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load backtest predictions into Postgres (main + archive); supports `--model MODEL_ID`, `--all` (scan all subdirs), `--input PATH` |
| `mvp/demand/scripts/tune_hyperparams.py` | Bayesian hyperparameter tuning via Optuna: walk-forward CV, TPE sampler, early stopping, outputs `data/tuning/best_params_<model>.json` |
| `mvp/demand/common/tuning.py` | Shared tuning utilities: `generate_cv_month_splits`, `compute_wape_stabilised`, `suggest_params`, `save_best_params`, `load_best_params`, `best_rounds_to_n_estimators`, `tune_for_timeframe()` (per-timeframe causal tuning, PL-002), `TRAIN_FOLD_FNS` registry (`train_lgbm_fold`, `train_catboost_fold`, `train_xgboost_fold`) |
| `mvp/demand/config/hyperparameter_tuning.yaml` | Search spaces (LGBM 8 params, CatBoost 5 params, XGBoost 8 params), CV settings, trial budget, pruner config |
| `mvp/demand/common/shap_selector.py` | SHAP computation, cumulative-importance feature selection, cluster-pooled SHAP, CSV output helpers (Feature 42) |
| `mvp/demand/api/routers/shap.py` | SHAP REST API: 5 endpoints — models list, cross-timeframe summary, timeframes list, per-timeframe detail (Feature 42); on-demand per-DFU SHAP `GET /forecast/shap/{model_id}/dfu` (loads pkl, rebuilds feature matrix, runs SHAP, returns per-month signed values for historical + future months) |
| `mvp/demand/frontend/src/types/shap.ts` | TypeScript types for SHAP API responses: ShapFeatureSummary, ShapFeatureDetail, ShapTimeframeEntry, payload envelopes (Feature 42) |
| `mvp/demand/scripts/clean_backtest_models.py` | Selective cleanup of model predictions from Postgres + view refresh |
| `mvp/demand/scripts/clean_forecasts_by_date.py` | Date-range forecast cleanup: delete rows by time bucket from forecast + archive tables |
| `mvp/demand/sql/010_create_backtest_lag_archive.sql` | DDL for backtest all-lags archive table |
| `mvp/demand/sql/008_perf_indexes_and_agg.sql` | Performance indexes (B-tree, GIN trigram) + materialized views |
| `mvp/demand/frontend/src/api/queries.ts` | Thin re-export barrel: `export * from "./queries/index"` — all domain query modules in `queries/` subfolder |
| `mvp/demand/frontend/src/api/queries/` | Domain query modules: `core.ts`, `inv-planning.ts`, `ai-planner.ts`, `control-tower.ts`, `fill-rate.ts`, `storyboard.ts`, `platform.ts`, `evolution.ts`, `supply.ts`, `filter-meta.ts`, `production-forecast.ts`, `index.ts` |
| `mvp/demand/frontend/src/tabs/` | Extracted tab components (DashboardTab, ExplorerTab, AccuracyTab, ItemAnalysisTab, ClustersTab, MarketIntelTab, InvBacktestTab, ChatPanel, JobsTab, InvPlanningTab, ControlTowerTab, AIPlannerTab, StoryboardTab, SopTab, DataQualityTab, FVATab) |
| `mvp/demand/frontend/src/tabs/ItemAnalysisTab.tsx` | Unified Item Analysis tab: merged DFU Analysis + Inventory into single tab with checkbox toggle toolbar (7 panels: Forecast Chart, SHAP, Model KPIs, Inv KPIs, Position Table, Variability, Lead Time); state persisted in localStorage via `usePanelToggles` |
| `mvp/demand/frontend/src/hooks/usePanelToggles.ts` | localStorage-persisted panel toggle hook for ItemAnalysisTab checkbox toolbar |
| `mvp/demand/frontend/src/tabs/inv-planning/` | Inventory Planning panel components: 28 panels including ExceptionQueuePanel, PortfolioHealthPanel, EoqPanel, PolicyManagementPanel, FillRatePanel, AbcXyzPanel, SupplierPanel, IntramonthPanel, SafetyStockPanel, VariabilityPanel, LeadTimePanel, DemandSignalsPanel, SimulationPanel, InvestmentPanel, RebalancingPanel, DemandForecastPanel, BlendedDemandPanel, EchelonPanel, FinancialPlanPanel, EventCalendarPanel, ScenarioPlanningPanel, ReplenishmentPlanPanel, DemandPlanPanel, OverrideQueuePanel, ProcurementPanel, OpenPOPanel, ProjectionPanel, PlannedOrdersPanel |
| `mvp/demand/frontend/src/hooks/useTheme.ts` | Color mode management (light/dark) for the General theme |
| `mvp/demand/frontend/src/hooks/useUrlState.ts` | URL state synchronization (VALID_TABS includes `itemAnalysis`; backward compat redirects `dfuAnalysis`/`inventory` → `itemAnalysis`) |
| `mvp/demand/frontend/src/hooks/useKeyboardShortcuts.ts` | Keyboard shortcuts handler (1-9 tabs, sidebar, dark mode) |
| `mvp/demand/frontend/src/hooks/useSidebar.ts` | Sidebar collapse/expand state + mobile drawer |
| `mvp/demand/frontend/src/hooks/useGlobalFilters.ts` | Global filter state with debounced URL sync |
| `mvp/demand/frontend/src/context/GlobalFilterContext.tsx` | Global filter React context provider |
| `mvp/demand/frontend/src/context/ScenarioNotificationContext.tsx` | Cross-tab scenario notification context (Feature 38) |
| `mvp/demand/common/job_registry.py` | APScheduler-powered job engine: BackgroundScheduler, JobManager singleton, per-group concurrency, scheduling, pipelines (Feature 39) |
| `mvp/demand/api/routers/jobs.py` | Jobs REST API: 12 endpoints — CRUD, scheduling, pipelines, stats (Feature 39) |
| `mvp/demand/sql/020_create_job_history.sql` | DDL for persistent job_history table (Feature 39) |
| `mvp/demand/sql/021_alter_job_history_scheduling.sql` | DDL for scheduling columns + job_schedule table (Feature 39) |
| `mvp/demand/frontend/src/tabs/JobsTab.tsx` | Automation dashboard UI: KPI cards, grouped job cards, schedules, history (Feature 39) |
| `mvp/demand/frontend/src/types/jobs.ts` | TypeScript types: JobStats, JobSchedule, GROUP_CONFIG (Feature 39) |
| `mvp/demand/frontend/src/context/JobNotificationContext.tsx` | Cross-tab job notification context (Feature 39) |
| `mvp/demand/frontend/src/types/theme.ts` | TypeScript types for theme, sidebar, filters, dashboard |
| `mvp/demand/frontend/src/constants/themes/general.ts` | Single professional theme config (Demand Studio, light + dark) |
| `mvp/demand/frontend/src/components/AppSidebar.tsx` | Collapsible sidebar navigation (16 items, 5 sections) |
| `mvp/demand/frontend/src/components/ThemeSelector.tsx` | Light/dark mode toggle (sidebar footer) |
| `mvp/demand/frontend/src/components/GlobalFilterBar.tsx` | Cross-tab filter bar (brand, category, item, location, market, channel) |
| `mvp/demand/frontend/src/components/WidgetGrid.tsx` | CSS Grid dashboard layout (WidgetGrid + WidgetCard) |
| `mvp/demand/frontend/src/components/AlertPanel.tsx` | Severity-coded alert list |
| `mvp/demand/frontend/src/components/HeatmapGrid.tsx` | CSS Grid heatmap with color scale |
| `mvp/demand/frontend/src/components/TopMovers.tsx` | Period-over-period top movers list |
| `mvp/demand/frontend/src/components/ForecastTrendChart.tsx` | ECharts forecast vs actual trend chart |
| `mvp/demand/sql/018_dashboard_views.sql` | Materialized view for top movers |
| `mvp/demand/frontend/src/lib/formatters.ts` | Number/cell formatting utilities |
| `mvp/demand/frontend/src/lib/export.ts` | CSV export utility (papaparse) |
| `mvp/demand/frontend/src/components/DataTable.tsx` | Virtualized data grid (TanStack Table + Virtual) |
| `mvp/demand/frontend/src/components/Skeleton.tsx` | Loading skeleton placeholder |
| `mvp/demand/frontend/src/components/EChartContainer.tsx` | Theme-aware ECharts wrapper |
| `mvp/demand/frontend/vite.config.ts` | Vite dev server config: API proxy rules (CRITICAL — every new API path prefix must be added here) |
| `mvp/demand/frontend/vitest.config.ts` | Vitest test configuration |
| `mvp/demand/frontend/e2e/playwright.config.ts` | Playwright E2E test configuration |
| `mvp/demand/frontend/e2e/fixtures/base.ts` | Shared E2E fixtures: `navigateToTab()`, sidebar labels, filter bar tab lists |
| `mvp/demand/frontend/e2e/tests/` | E2E smoke tests: navigation, dashboard, accuracy, global-filters, inv-planning, ai-planner, control-tower, theme |
| `mvp/demand/frontend/tailwind.config.ts` | Tailwind config with custom `pulse-glow` animation |
| `mvp/demand/tests/` | Backend test suite (pytest): unit/ + api/ |
| `mvp/demand/tests/conftest.py` | Shared pytest fixtures (sample DataFrames) |
| `mvp/demand/tests/api/conftest.py` | API test fixtures: `make_pool(fetchall_return, fetchone_return)` factory + `mock_pool` fixture |
| `mvp/demand/frontend/src/**/__tests__/` | Frontend test suites (Vitest + RTL) |
| `docs/architecture-diagram.md` | Full-stack architecture diagram (layers, data flow, ML pipeline) |
| `docs/design-specs/` | Feature specs organized in 5 functional subfolders: 01-platform-infrastructure/, 02-forecasting-models/, 03-clustering-seasonality/, 04-inventory-planning/, 05-ui-automation/ |
| `mvp/demand/api/core.py` | Shared API utilities: connection pool, OpenAI client, SQL helpers used by router modules |
| `mvp/demand/api/auth.py` | Optional API key auth (`require_api_key` dependency; disabled when `API_KEY` env var unset) |
| `mvp/demand/api/routers/` | Modular FastAPI router modules: 53 active routers (all with OpenAPI tags; see inv_planning_* split below) |
| `mvp/demand/api/routers/inv_planning.py` | Thin compatibility shim — re-exports `router` from domain routers for backward compat |
| `mvp/demand/api/routers/inv_planning_eoq.py` | EOQ endpoints: summary, detail, sensitivity (IPfeature4) |
| `mvp/demand/api/routers/inv_planning_policy.py` | Policy CRUD + assignment + compliance endpoints (IPfeature5) |
| `mvp/demand/api/routers/inv_planning_health.py` | Health score endpoints: summary, detail, heatmap (IPfeature6) |
| `mvp/demand/api/routers/inv_planning_exceptions.py` | Exception queue endpoints: list, summary, acknowledge, status, generate (IPfeature7) |
| `mvp/demand/api/routers/inv_planning_safety_stock.py` | Safety stock endpoints: summary, detail, gap analysis (IPfeature3) |
| `mvp/demand/api/routers/inv_planning_abc_xyz.py` | ABC-XYZ classification endpoints (IPfeature11) |
| `mvp/demand/api/routers/inv_planning_supplier.py` | Supplier performance endpoints (IPfeature12) |
| `mvp/demand/api/routers/inv_planning_investment.py` | Investment plan endpoints (IPfeature13) |
| `mvp/demand/api/routers/inv_planning_intramonth.py` | Intramonth stockout endpoints (IPfeature14) |
| `mvp/demand/api/routers/inv_planning_demand_signals.py` | Demand signals endpoints (IPfeature9) |
| `mvp/demand/api/routers/inv_planning_simulation.py` | Safety stock simulation endpoints (IPfeature10) |
| `mvp/demand/api/routers/inv_planning_variability.py` | Demand variability endpoints |
| `mvp/demand/api/routers/inv_planning_lead_time.py` | Lead time analysis endpoints |
| `mvp/demand/api/routers/inv_planning_rebalancing.py` | Inventory rebalancing endpoints: KPIs, network, imbalances, plans, transfers, approval workflow (12 endpoints) |
| `mvp/demand/api/routers/inventory.py` | Inventory endpoints: position, KPIs, trend, item-detail |
| `mvp/demand/api/routers/inv_backtest.py` | Inventory backtest endpoints: summary, trend, root-cause, detail |
| `mvp/demand/api/routers/dashboard.py` | Dashboard endpoints: KPIs, alerts, top-movers, heatmap |
| `mvp/demand/frontend/src/context/ThemeContext.tsx` | Theme React context provider (replaces theme prop-drilling) |
| `mvp/demand/frontend/src/hooks/useChartColors.ts` | Chart color hook: returns chartColors + trendColors from theme context |
| `mvp/demand/frontend/src/components/ScenarioCharts.tsx` | Extracted scenario visualization charts (elbow, silhouette, radar, pie, gap) |
| `mvp/demand/config/seasonality_config.yaml` | Seasonality detection hyperparameters and profile labeling thresholds |
| `mvp/demand/scripts/detect_seasonality.py` | Compute seasonality metrics per DFU (strength, profile, peak/trough month) |
| `mvp/demand/scripts/update_seasonality_profiles.py` | Write seasonality profiles to `dim_dfu` in Postgres |
| `mvp/demand/scripts/run_clustering_scenario.py` | What-If clustering: run trial KMeans with custom params + promote flow |
| `mvp/demand/sql/013_add_composite_indexes.sql` | Composite B-tree indexes for multi-column query performance |
| `mvp/demand/sql/015_add_seasonality_columns.sql` | DDL: 6 seasonality columns on `dim_dfu` (Feature 30) |
| `mvp/demand/sql/016_add_seasonality_to_accuracy_views.sql` | DDL: seasonality joins in accuracy materialized views (Feature 32) |
| `mvp/demand/frontend/src/hooks/useDebounce.ts` | Generic debounce hook used by filter inputs |
| `mvp/demand/frontend/src/constants/colors.ts` | Shared color palette constants |
| `mvp/demand/frontend/src/constants/elements.ts` | Periodic table element definitions for loading overlay |
| `mvp/demand/frontend/src/components/KeyboardShortcutHelp.tsx` | Keyboard shortcut help modal (triggered by `?` shortcut) |
| `mvp/demand/frontend/src/components/KpiCard.tsx` | Reusable KPI metric card component |
| `mvp/demand/frontend/src/lib/utils.ts` | Shared utilities: `cn()` Tailwind class merger and misc helpers |
| `mvp/demand/scripts/compute_eoq.py` | EOQ computation: `compute_eoq()`, `compute_effective_eoq()`, `compute_eoq_metrics()`, `sensitivity_curve()`, `run()` (IPfeature4) |
| `mvp/demand/config/eoq_config.yaml` | EOQ config: ordering_cost, holding_cost_pct, moq, max_eoq_months_supply (IPfeature4) |
| `mvp/demand/sql/024_create_eoq_targets.sql` | DDL for `fact_eoq_targets` table (IPfeature4) |
| `mvp/demand/config/replenishment_policy_config.yaml` | 4 default replenishment policies + auto-assign rules by segment (IPfeature5) |
| `mvp/demand/sql/025_create_replenishment_policy.sql` | DDL for `dim_replenishment_policy` + `fact_dfu_policy_assignment` (IPfeature5) |
| `mvp/demand/scripts/assign_replenishment_policies.py` | Upsert policies from config + auto-assign DFUs by segment (--dry-run, --force-overwrite) (IPfeature5) |
| `mvp/demand/sql/026_create_inventory_health_score.sql` | DDL for stub `fact_safety_stock_targets` + `mv_inventory_health_score` materialized view (IPfeature6) |
| `mvp/demand/scripts/refresh_health_scores.py` | Refresh health score materialized view: `REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score` (IPfeature6) |
| `mvp/demand/sql/027_create_replenishment_exceptions.sql` | DDL for `fact_replenishment_exceptions` + 6 indexes (IPfeature7) |
| `mvp/demand/scripts/generate_replenishment_exceptions.py` | Detect exceptions from inventory + policy data, write to DB with --dry-run support (IPfeature7) |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Inventory Planning tab: two-column layout with fixed 220px grouped sidebar navigation (7 groups — Daily Operations, Optimize, Analytics, Planning, Sensing, Strategic, Supply — each with colored divider, group label, and icon-labeled tab buttons) + scrollable main content with per-panel header bar (title + description from PANEL_META). 27 panels total covering IPfeature4-14, rebalancing, and beyond |
| `mvp/demand/api/routers/inv_planning.py` | Inventory planning endpoints: EOQ (IP4), policy CRUD/assign/compliance (IP5), health (IP6), exceptions (IP7), demand-signals (IP9), simulation (IP10), abc-xyz (IP11), supplier-performance (IP12), investment (IP13), intramonth-stockouts (IP14) |
| `mvp/demand/api/routers/fill_rate.py` | Fill rate analytics endpoints: summary, trend, detail (IPfeature8) |
| `mvp/demand/api/routers/control_tower.py` | Control Tower endpoints: kpis, alerts, top-critical, trend (IPfeature15) |
| `mvp/demand/frontend/src/tabs/ControlTowerTab.tsx` | Unified Control Tower tab: KPI cards, alert panel, top-critical list, trend chart (IPfeature15) |
| `mvp/demand/sql/028_create_fill_rate_monthly.sql` | DDL for `mv_fill_rate_monthly` materialized view (IPfeature8) |
| `mvp/demand/sql/029_create_demand_signals.sql` | DDL for `fact_demand_signals` table (IPfeature9) |
| `mvp/demand/sql/030_create_ss_simulation_results.sql` | DDL for `fact_ss_simulation_results` table (IPfeature10) |
| `mvp/demand/sql/031_add_xyz_classification.sql` | DDL: XYZ classification columns on `dim_dfu` (IPfeature11) |
| `mvp/demand/sql/032_create_supplier_performance.sql` | DDL for `mv_supplier_performance` materialized view (IPfeature12) |
| `mvp/demand/sql/033_create_investment_plan.sql` | DDL for `fact_inventory_investment_plan` + `fact_efficient_frontier` (IPfeature13) |
| `mvp/demand/sql/034_create_intramonth_stockout.sql` | DDL for `mv_intramonth_stockout` materialized view (IPfeature14) |
| `mvp/demand/sql/035_create_control_tower_kpis.sql` | DDL for `mv_control_tower_kpis` materialized view (IPfeature15) |
| `mvp/demand/scripts/compute_demand_signals.py` | Compute short-horizon demand signals from sales velocity and inventory movement (IPfeature9) |
| `mvp/demand/scripts/run_ss_simulation.py` | Monte Carlo safety stock simulation: reads config/simulation_config.yaml, writes to fact_ss_simulation_results (IPfeature10) |
| `mvp/demand/scripts/classify_abc_xyz.py` | ABC-XYZ classification: combined volume (ABC) × variability (XYZ) segmentation, writes to dim_dfu (IPfeature11) |
| `mvp/demand/scripts/compute_investment_plan.py` | Capital investment optimization: efficient frontier computation, budget allocation (IPfeature13) |
| `mvp/demand/scripts/refresh_intramonth_stockout.py` | Refresh intramonth stockout materialized view: `REFRESH MATERIALIZED VIEW CONCURRENTLY mv_intramonth_stockout` (IPfeature14) |
| `mvp/demand/config/simulation_config.yaml` | Monte Carlo simulation config: n_simulations, random_seed (IPfeature10) |
| `mvp/demand/common/ai_planner.py` | AI Planning Agent core: `AIPlannerAgent` class, 10 tool functions (9 read-only SQL tools + `create_insight`), `CreateInsightInput` Pydantic validator, `log_ai_call` helper, MAX_TURNS=40 + TOKEN_BUDGET=100K circuit breaker, few-shot system prompt (IPAIfeature1) |
| `mvp/demand/config/ai_planner_config.yaml` | AI planner config: model, thresholds (DOS, WAPE, bias), severity rules, schedule (IPAIfeature1) |
| `mvp/demand/sql/036_create_ai_insights.sql` | DDL for `ai_insights` + `ai_planning_memos` tables (IPAIfeature1) |
| `mvp/demand/sql/039_create_ai_call_log.sql` | DDL for `ai_call_log` observability table: per-turn token usage, latency, tool call outcomes (IPAIfeature1 enhancement) |
| `mvp/demand/scripts/generate_ai_insights.py` | CLI batch job: portfolio scan or single DFU analysis, --dry-run support (IPAIfeature1) |
| `mvp/demand/api/routers/ai_planner.py` | AI planner endpoints: analyze DFU, portfolio-scan (202), insights list, status update, memos list, metrics (IPAIfeature1) |
| `mvp/demand/frontend/src/tabs/AIPlannerTab.tsx` | AI Planner tab: portfolio health bar, insight cards with confidence badge + causal reasoning, planning memo panel, last-scan timestamp, auto-dismiss banner (IPAIfeature1) |
| `mvp/demand/frontend/src/constants/design-tokens.ts` | Design tokens: semantic color palette (AI_COLOR=teal, severity colors), UX limits (MAX_PRIMARY_KPIS=4), AI confidence thresholds |
| `mvp/demand/frontend/src/types/ai-planner.ts` | TypeScript types: AiInsight, AiPlanningMemo, InsightSeverity, InsightStatus, InsightType (IPAIfeature1) |
| `mvp/demand/frontend/src/components/ui/select.tsx` | Minimal shadcn/ui Select API wrapper (Select, SelectContent, SelectItem, SelectTrigger, SelectValue) |
| `mvp/demand/common/exception_engine.py` | Exception detection engine: `ExceptionEngine` class, threshold evaluation, severity scoring, exception type classification (Feature 40) |
| `mvp/demand/scripts/generate_storyboard_exceptions.py` | CLI batch job: generate storyboard exceptions for all or specific DFUs, --dry-run support (Feature 40) |
| `mvp/demand/api/routers/storyboard.py` | Storyboard endpoints: exception list, summary, detail, acknowledge, generate (Feature 40) |
| `mvp/demand/frontend/src/tabs/StoryboardTab.tsx` | Storyboard tab: exception cards with causal chain, severity badges, resolution workflow (Feature 40) |
| `mvp/demand/frontend/src/types/storyboard.ts` | TypeScript types: StoryboardException, StoryboardSummary, ExceptionType, ResolutionStatus (Feature 40) |
| `mvp/demand/sql/038_create_storyboard.sql` | DDL for `fact_storyboard_exceptions` + indexes (Feature 40) |
| `mvp/demand/config/exception_config.yaml` | Exception detection thresholds by type and severity (Feature 40) |
| `mvp/demand/scripts/compute_safety_stock.py` | Safety stock computation: `compute_safety_stock()`, service-level Z-table lookup, demand/LT variability, writes to `fact_safety_stock_targets` (IPfeature3) |
| `mvp/demand/sql/037_create_safety_stock_targets.sql` | DDL for `fact_safety_stock_targets` + indexes (IPfeature3) |
| `mvp/demand/config/safety_stock_config.yaml` | Safety stock config: service levels by ABC class, Z-table, guard rails (IPfeature3) |
| `mvp/demand/scripts/compute_demand_variability.py` | Demand variability computation: CV, dispersion metrics, volatility profiles per DFU (IPfeature3/variability) |
| `mvp/demand/scripts/compute_lead_time_variability.py` | Lead time variability computation: LT CV, reliability metrics per item-location (IPfeature3/lead time) |
| `mvp/demand/config/variability_config.yaml` | Demand variability config: CV thresholds, history_months, volatility profile labels |
| `mvp/demand/config/lead_time_config.yaml` | Lead time variability config: LT CV thresholds, reliability bands |
| `mvp/demand/sql/022_create_demand_variability.sql` | DDL: demand variability columns on `dim_dfu` |
| `mvp/demand/sql/023_create_lead_time_profile.sql` | DDL: lead time profile table + indexes |
| `mvp/demand/frontend/src/tabs/inventory/` | Inventory panel components (used by ItemAnalysisTab): KpiSection, TrendChartPanel, PositionTablePanel, ItemDetailPanel |
| `mvp/demand/frontend/src/tabs/accuracy/` | Accuracy tab panel components: KpiSection, TrendChartPanel, SliceTablePanel, ChampionPanel, ShapPanel |
| `mvp/demand/frontend/src/tabs/jobs/` | Jobs tab panel components: KpiSection, JobGroupsPanel, ActiveJobsPanel, SchedulesPanel, JobHistoryPanel |
| `mvp/demand/frontend/src/tabs/clusters/` | Clusters tab panel components: ClusterOverviewPanel, WhatIfPanel, ScenarioResultsPanel, PastScenariosPanel |
| `mvp/demand/frontend/src/tabs/dfu-analysis/` | DFU Analysis panel components (used by ItemAnalysisTab): SelectorPanel, OverlayChartPanel, ModelKpiSection, DfuShapPanel |
| `mvp/demand/frontend/src/api/queries/` | Domain query modules: core.ts, inv-planning.ts (barrel), inv-planning-eoq.ts, inv-planning-policy.ts, inv-planning-health.ts, inv-planning-exceptions.ts, inv-planning-safety-stock.ts, inv-planning-signals.ts, inv-planning-abc.ts, inv-planning-supplier.ts, inv-planning-intramonth.ts, ai-planner.ts, control-tower.ts, fill-rate.ts, storyboard.ts |
| `mvp/demand/common/job_state.py` | Job in-memory state: `_active_jobs`, `_pending_queues`, `_cancel_flags`, state lock, status constants |
| `mvp/demand/common/job_scheduler.py` | APScheduler wrapper: scheduler initialization, cron/interval schedule CRUD, APScheduler-specific utilities |
| `mvp/demand/sql/039_create_production_forecast.sql` | DDL for `fact_production_forecast` + `fact_model_registry` tables (F1.1) |
| `mvp/demand/config/production_forecast_config.yaml` | Production forecast config: inference settings, model_selection, plan_version format, scheduler config (F1.1) |
| `mvp/demand/scripts/generate_production_forecasts.py` | Full inference pipeline: load_active_models, get_champion_assignments, build_inference_grid, generate_forecast_recursive, write_forecast, purge_old_versions (F1.1) |
| `mvp/demand/api/routers/production_forecast.py` | Production forecast endpoints: GET /forecast/production, /summary, /versions (F1.1) |
| `mvp/demand/frontend/src/api/queries/production-forecast.ts` | TypeScript interfaces + fetch functions for production forecast endpoints (F1.1) |
| `mvp/demand/frontend/src/tabs/inv-planning/DemandForecastPanel.tsx` | Demand Forecast panel: KPI cards, ABC breakdown chart, DFU drill-down chart with CI bands (F1.1) |
| `mvp/demand/sql/071_create_transfer_network.sql` | DDL for `dim_transfer_lane` network topology table |
| `mvp/demand/sql/072_create_rebalancing_plan.sql` | DDL for `fact_rebalancing_plan` + `fact_rebalancing_transfer` |
| `mvp/demand/sql/073_create_rebalancing_views.sql` | DDL for `mv_network_balance` materialized view |
| `mvp/demand/config/rebalancing_config.yaml` | Rebalancing config: solver, thresholds, costs, constraints, scheduling |
| `mvp/demand/scripts/compute_rebalancing.py` | Rebalancing computation: detect imbalances, build candidates, greedy/LP solvers, financial analysis |
| `mvp/demand/api/routers/inv_planning_rebalancing.py` | Inventory rebalancing endpoints: KPIs, network, imbalances, plans, transfers, approval workflow (12 endpoints) |
| `mvp/demand/api/routers/accuracy.py` | Accuracy analytics endpoints |
| `mvp/demand/api/routers/analysis.py` | DFU analysis endpoints |
| `mvp/demand/api/routers/bias_corrections.py` | Bias correction endpoints (F1.2/F3.1) |
| `mvp/demand/api/routers/blended_forecast.py` | Blended demand forecast endpoints (F3.4) |
| `mvp/demand/api/routers/clusters.py` | Clustering scenario endpoints |
| `mvp/demand/api/routers/collaboration.py` | Collaboration threads + annotations endpoints (08-05) |
| `mvp/demand/api/routers/competition.py` | Champion model competition endpoints |
| `mvp/demand/api/routers/consensus_plan.py` | Consensus plan endpoints (F4.2) |
| `mvp/demand/api/routers/data_quality.py` | Data quality dashboard + rule endpoints (08-01) |
| `mvp/demand/api/routers/domains.py` | Generic domain CRUD — catch-all `{domain}` path param (mounted last) |
| `mvp/demand/api/routers/echelon_planning.py` | Multi-echelon safety stock endpoints (F3.5) |
| `mvp/demand/api/routers/events.py` | Event calendar endpoints (F4.3) |
| `mvp/demand/api/routers/external_signals.py` | External demand signal endpoints (08-06) |
| `mvp/demand/api/routers/financial_plan.py` | Financial planning endpoints (F4.1) |
| `mvp/demand/api/routers/fva.py` | FVA tracking + ROI endpoints (08-07) |
| `mvp/demand/api/routers/inv_planning_projection.py` | Inventory projection endpoints (F1.2) |
| `mvp/demand/api/routers/inv_planning_replenishment.py` | Replenishment plan endpoints |
| `mvp/demand/api/routers/lead_time_learning.py` | Lead time learning endpoints |
| `mvp/demand/api/routers/notifications.py` | Notification channel + preference endpoints (08-04) |
| `mvp/demand/api/routers/reports.py` | Report generation + scheduling endpoints (08-08) |
| `mvp/demand/api/routers/service_level.py` | Service level actuals endpoints |
| `mvp/demand/api/routers/sop.py` | S&OP cycle endpoints (F4.2) |
| `mvp/demand/api/routers/supply.py` | Supply chain endpoints |
| `mvp/demand/api/routers/supply_scenarios.py` | Supply chain scenario endpoints (F4.4) |
| `mvp/demand/api/routers/auth_router.py` | RBAC authentication endpoints (08-02) |
| `mvp/demand/api/routers/users.py` | User management endpoints (08-02) |
| `mvp/demand/api/routers/webhooks.py` | Webhook registration + delivery endpoints (08-10) |
| `mvp/demand/common/cache.py` | Caching utilities (08-03) |
| `mvp/demand/common/dq_engine.py` | Data quality engine: rule evaluation, scoring (08-01) |
| `mvp/demand/common/forecast_ci.py` | Forecast confidence interval computation |
| `mvp/demand/common/notification_engine.py` | Notification dispatch engine (08-04) |
| `mvp/demand/common/query_tracker.py` | API query tracking + usage metrics |
| `mvp/demand/common/rate_limiter.py` | Token bucket rate limiter (08-09) |
| `mvp/demand/common/webhook_dispatcher.py` | Webhook event dispatcher (08-10) |
| `mvp/demand/scripts/compute_bias_corrections.py` | Bias correction computation (F1.2/F3.1) |
| `mvp/demand/scripts/compute_blended_forecast.py` | Alpha-weighted blended demand forecast (F3.4) |
| `mvp/demand/scripts/compute_echelon_targets.py` | Multi-echelon safety stock computation (F3.5) |
| `mvp/demand/scripts/compute_financial_plan.py` | Financial plan computation (F4.1) |
| `mvp/demand/scripts/compute_inventory_projection.py` | Forward inventory projection (F1.2) |
| `mvp/demand/scripts/compute_replenishment_plan.py` | Replenishment plan computation |
| `mvp/demand/scripts/compute_service_level_actuals.py` | Service level actuals computation |
| `mvp/demand/scripts/generate_consensus_plan.py` | Consensus plan generation (F4.2) |
| `mvp/demand/scripts/generate_planned_orders.py` | Planned order generation |
| `mvp/demand/scripts/generate_quantile_forecasts.py` | Quantile forecast generation |
| `mvp/demand/scripts/load_open_pos.py` | Open PO data loading |
| `mvp/demand/scripts/apply_event_adjustments.py` | Event-driven forecast adjustments (F4.3) |
| `mvp/demand/scripts/release_planned_orders.py` | Planned order release to procurement |
| `mvp/demand/scripts/run_sop_cycle.py` | S&OP cycle execution (F4.2) |
| `mvp/demand/scripts/run_supply_chain_scenario.py` | Supply chain scenario simulation (F4.4) |
| `mvp/demand/scripts/update_lead_time_actuals.py` | Lead time actuals update |
| `mvp/demand/config/bias_correction_config.yaml` | Bias correction thresholds and parameters (F1.2/F3.1) |
| `mvp/demand/config/consensus_config.yaml` | Consensus plan config (F4.2) |
| `mvp/demand/config/data_quality_config.yaml` | Data quality rules and thresholds (08-01) |
| `mvp/demand/config/echelon_config.yaml` | Multi-echelon SS config (F3.5) |
| `mvp/demand/config/event_planning_config.yaml` | Event calendar config (F4.3) |
| `mvp/demand/config/financial_plan_config.yaml` | Financial planning config (F4.1) |
| `mvp/demand/config/fva_config.yaml` | FVA tracking config (08-07) |
| `mvp/demand/config/notification_config.yaml` | Notification channel config (08-04) |
| `mvp/demand/config/projection_config.yaml` | Inventory projection config (F1.2) |
| `mvp/demand/config/reporting_config.yaml` | Reporting config (08-08) |
| `mvp/demand/config/sop_config.yaml` | S&OP cycle config (F4.2) |
| `mvp/demand/config/supply_scenario_config.yaml` | Supply chain scenario config (F4.4) |
| `mvp/demand/config/auth_config.yaml` | RBAC authentication config (08-02) |
| `mvp/demand/config/cache_config.yaml` | Caching config (08-03) |
| `mvp/demand/config/api_governance_config.yaml` | API governance config: rate limits, versioning (08-09) |
| `mvp/demand/frontend/src/tabs/DataQualityTab.tsx` | Data quality dashboard tab (08-01) |
| `mvp/demand/frontend/src/tabs/FVATab.tsx` | FVA & ROI tracking tab (08-07) |
| `mvp/demand/frontend/src/tabs/SopTab.tsx` | S&OP cycle stage machine tab (F4.2) |
| `mvp/demand/frontend/src/tabs/inv-planning/BlendedDemandPanel.tsx` | Blended demand panel (F3.4) |
| `mvp/demand/frontend/src/tabs/inv-planning/EchelonPanel.tsx` | Multi-echelon SS panel (F3.5) |
| `mvp/demand/frontend/src/tabs/inv-planning/FinancialPlanPanel.tsx` | Financial plan panel (F4.1) |
| `mvp/demand/frontend/src/tabs/inv-planning/EventCalendarPanel.tsx` | Event calendar panel (F4.3) |
| `mvp/demand/frontend/src/tabs/inv-planning/ScenarioPlanningPanel.tsx` | Scenario planning panel (F4.4) |
| `mvp/demand/frontend/src/tabs/inv-planning/RebalancingPanel.tsx` | Inventory rebalancing panel |
| `mvp/demand/frontend/src/tabs/inv-planning/ReplenishmentPlanPanel.tsx` | Replenishment plan panel |
| `mvp/demand/frontend/src/tabs/inv-planning/DemandPlanPanel.tsx` | Demand plan panel |
| `mvp/demand/frontend/src/tabs/inv-planning/OverrideQueuePanel.tsx` | Override queue panel |
| `mvp/demand/frontend/src/tabs/inv-planning/ProcurementPanel.tsx` | Procurement panel |
| `mvp/demand/frontend/src/tabs/inv-planning/OpenPOPanel.tsx` | Open PO panel |
| `mvp/demand/frontend/src/tabs/inv-planning/ProjectionPanel.tsx` | Inventory projection panel (F1.2) |
| `mvp/demand/frontend/src/tabs/inv-planning/PlannedOrdersPanel.tsx` | Planned orders panel |
| `mvp/demand/frontend/src/tabs/accuracy/BiasCorrectionsPanel.tsx` | Bias corrections panel (F3.1) |
| `mvp/demand/frontend/src/tabs/inv-planning/` | All 28 inventory planning panel components |
| `mvp/demand/frontend/src/api/queries/platform.ts` | Platform query keys + fetch functions (data quality, notifications, collaboration, FVA, reports, webhooks) |
| `mvp/demand/frontend/src/api/queries/evolution.ts` | Evolution-to-operations query keys + fetch functions (F3.1–F4.4) |
| `mvp/demand/frontend/src/api/queries/supply.ts` | Supply chain query keys + fetch functions |
| `mvp/demand/frontend/src/api/queries/filter-meta.ts` | Filter metadata query keys + fetch functions |
| `mvp/demand/frontend/src/api/queries/inv-planning-projection.ts` | Inventory projection query keys + fetch functions (F1.2) |
| `mvp/demand/frontend/src/api/queries/inv-planning-rebalancing.ts` | Inventory rebalancing query keys + fetch functions |
| `mvp/demand/frontend/src/api/queries/inv-planning-replenishment.ts` | Replenishment plan query keys + fetch functions |
| `mvp/demand/frontend/src/hooks/useFilteredQuery.ts` | Filtered query hook with global filter integration |
| `mvp/demand/frontend/src/components/EmptyState.tsx` | Empty state placeholder component |
| `mvp/demand/frontend/src/components/LoadingElement.tsx` | Chemistry-themed loading element |
| `mvp/demand/frontend/src/components/VrantisLogo.tsx` | Vrantis logo component with wordmark option |

---

## Common Commands

```bash
# One-time setup
make init              # Create .venv, install uv, sync dependencies

# Infrastructure
make up                # Start Docker services (Postgres, MLflow)
make down              # Stop all services
make db-apply-sql      # Apply DDL schemas to Postgres

# Data pipeline
make normalize-all     # Normalize all 8 datasets (CSV → clean CSV)
make load-all          # Load cleaned data into Postgres + refresh materialized views
make load-forecast-replace  # Reload external forecast only (preserves backtest data)
make load-forecast-replace-no-archive  # Reload external forecast, skip archive (fast)

# Inventory pipeline
make db-apply-inventory     # Create inventory table + indexes + materialized view (one-time)
make normalize-inventory    # Merge 14 monthly CSVs into single clean CSV
make load-inventory         # Load into Postgres + refresh agg view
make inventory-pipeline     # normalize + load + refresh (all-in-one)

# Inventory backtest (Feature 37)
make db-apply-inv-backtest  # Create mv_inventory_forecast_monthly materialized view
make refresh-inv-backtest   # Refresh inventory-forecast bridge view with current data

# Job scheduler (Feature 39)
make db-apply-jobs          # Create job_history + job_schedule tables + indexes

# Inv Planning schema setup (one-time, per environment)
make fill-rate-schema       # mv_fill_rate_monthly (IPfeature8)
make demand-signals-schema  # fact_demand_signals (IPfeature9)
make sim-schema             # fact_ss_simulation_results (IPfeature10)
make abc-xyz-schema         # XYZ classification columns (IPfeature11)
make supplier-perf-schema   # mv_supplier_performance (IPfeature12)
make investment-schema      # fact_inventory_investment_plan + fact_efficient_frontier (IPfeature13)
make intramonth-schema      # mv_intramonth_stockout (IPfeature14)
make control-tower-schema   # mv_control_tower_kpis (IPfeature15)

# Run services
make api               # Start FastAPI on :8000
make ui-init           # Install npm deps
make ui                # Start React dev server on :5173

# Validation
make check-db          # Table row counts in Postgres
make check-api         # Curl API health + sample endpoints
make check-all         # Full check: DB + API

# Chatbot
make db-apply-chat     # Apply pgvector + embeddings table DDL
make generate-embeddings  # Generate and store schema embeddings (requires OPENAI_API_KEY)

# Benchmarking

# Clustering pipeline
make cluster-features  # Generate clustering feature matrix from sales/DFU/item data
make cluster-train     # Train KMeans, select optimal K, log to MLflow
make cluster-label     # Assign business labels to clusters
make cluster-update    # Write cluster labels to dim_dfu in Postgres
make cluster-all       # Run full clustering pipeline (features → train → label → update)

# Backtesting (Feature 44 — config-driven)
# Edit config/algorithm_config.yaml to set cluster_strategy, SHAP, recursive, tune_inline, params_file, etc.
make backtest-lgbm          # Run LGBM backtest (reads config/algorithm_config.yaml; cluster_strategy: per_cluster or global)
make backtest-catboost      # Run CatBoost backtest
make backtest-xgboost       # Run XGBoost backtest
make backtest-all           # Run all three sequentially (LGBM → CatBoost → XGBoost)
make backtest-all-parallel  # Run all three in parallel (logs → data/backtest/logs/)

# Hyperparameter tuning (Feature 41)
make tune-lgbm              # Tune LGBM hyperparameters via Optuna (50 trials, ~20-40 min) → data/tuning/best_params_lgbm.json
make tune-catboost          # Tune CatBoost hyperparameters (~30-60 min) → data/tuning/best_params_catboost.json
make tune-xgboost           # Tune XGBoost hyperparameters (~25-50 min) → data/tuning/best_params_xgboost.json
make tune-all               # Run all three tuning jobs sequentially

# Backtest loading (shared across all models)
make backtest-load MODEL=lgbm_cluster   # Load one model from data/backtest/lgbm_cluster/
make backtest-load MODEL=catboost_cluster
make backtest-load MODEL=xgboost_cluster
make backtest-load-all      # Load ALL models found under data/backtest/*/

# Champion model selection
make champion-select        # Run per-DFU champion selection (best-of-models via configurable strategy)
make champion-simulate      # Simulate all strategies, compare accuracy vs ceiling
make champion-train-meta    # Train meta-learner classifier for champion selection
make champion-all           # train-meta + simulate + select (full pipeline)

# Seasonality pipeline (feature 30)
make seasonality-schema     # Apply DDL for seasonality columns on dim_dfu (one-time)
make seasonality-detect     # Detect seasonality patterns per DFU from sales history
make seasonality-update     # Write seasonality profiles back to dim_dfu
make seasonality-all        # Full pipeline: schema + detect + update

# EOQ & Inventory Planning (IPfeature4)
make eoq-schema             # Apply DDL for fact_eoq_targets table (one-time)
make eoq-compute            # Compute EOQ metrics from agg_inventory_monthly → fact_eoq_targets
make eoq-all                # eoq-schema + eoq-compute (full pipeline)

# Replenishment Policy Management (IPfeature5)
make policy-schema          # Apply DDL for dim_replenishment_policy + fact_dfu_policy_assignment (one-time)
make policy-assign          # Upsert policies from config + auto-assign DFUs by segment
make policy-all             # policy-schema + policy-assign (full pipeline)

# Inventory Health Score (IPfeature6)
make health-schema          # Apply DDL: stub fact_safety_stock_targets + mv_inventory_health_score (one-time)
make health-refresh         # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score
make health-all             # health-schema + health-refresh (full pipeline)

# Exception Queue (IPfeature7)
make exceptions-schema       # Apply DDL for fact_replenishment_exceptions (one-time)
make exceptions-generate     # Detect exceptions + write to DB
make exceptions-generate-dry # Preview exceptions without writing (--dry-run)

# Fill Rate Analytics (IPfeature8)
make fill-rate-schema        # Apply DDL for mv_fill_rate_monthly (one-time)
make fill-rate-refresh       # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_fill_rate_monthly
make fill-rate-all           # fill-rate-schema + fill-rate-refresh (full pipeline)

# Demand Sensing (IPfeature9)
make demand-signals-schema   # Apply DDL for fact_demand_signals (one-time)
make demand-signals-compute  # Compute demand signals → fact_demand_signals
make demand-signals-all      # demand-signals-schema + demand-signals-compute

# Safety Stock Monte Carlo Simulation (IPfeature10)
make sim-schema              # Apply DDL for fact_ss_simulation_results (one-time)
make sim-run                 # Run Monte Carlo safety stock simulation

# ABC-XYZ Classification (IPfeature11)
make abc-xyz-schema          # Apply DDL for XYZ classification columns (one-time)
make abc-xyz-classify        # Run ABC-XYZ classification + write to dim_dfu
make abc-xyz-all             # abc-xyz-schema + abc-xyz-classify

# Supplier Performance (IPfeature12)
make supplier-perf-schema    # Apply DDL for mv_supplier_performance (one-time)
make supplier-perf-refresh   # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_supplier_performance
make supplier-perf-all       # supplier-perf-schema + supplier-perf-refresh

# Capital Investment Optimization (IPfeature13)
make investment-schema       # Apply DDL for fact_inventory_investment_plan + fact_efficient_frontier (one-time)
make investment-plan         # Compute investment plan + efficient frontier
make investment-all          # investment-schema + investment-plan

# Intra-Month Stockout Detection (IPfeature14)
make intramonth-schema       # Apply DDL for mv_intramonth_stockout (one-time)
make intramonth-refresh      # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_intramonth_stockout
make intramonth-all          # intramonth-schema + intramonth-refresh

# Control Tower (IPfeature15)
make control-tower-schema    # Apply DDL for mv_control_tower_kpis (one-time)
make control-tower-refresh   # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_control_tower_kpis
make control-tower-all       # control-tower-schema + control-tower-refresh

# Inventory Rebalancing
make rebalancing-schema       # Apply DDL: dim_transfer_lane + fact_rebalancing_plan/transfer + mv_network_balance
make rebalancing-compute      # Run rebalancing computation (detect imbalances, solve, write plan)
make rebalancing-compute-dry  # Preview rebalancing without writing (--dry-run)
make rebalancing-refresh      # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_network_balance
make rebalancing-all          # rebalancing-schema + rebalancing-compute (full pipeline)

# AI Planning Agent (IPAIfeature1)
make ai-insights-schema     # Apply DDL for ai_insights + ai_planning_memos tables (one-time)
make ai-insights-scan       # Run portfolio AI scan → write insights to DB
make ai-insights-dfu        # Analyze single DFU: make ai-insights-dfu ITEM=100320 LOC=1401-BULK
make ai-insights-all        # ai-insights-schema + ai-insights-scan (full pipeline)

# Production Forecast Generation (F1.1)
make forecast-prod-schema   # Apply DDL for fact_production_forecast + fact_model_registry (one-time)
make forecast-generate      # Run full production forecast inference pipeline
make forecast-generate-dfu  # Generate forecast for single DFU: make forecast-generate-dfu ITEM=100320 LOC=1401-BULK
make forecast-generate-dry  # Preview forecast generation without writing (--dry-run)
make forecast-prod-all      # forecast-prod-schema + forecast-generate (full pipeline)

# Backtest cleanup
make backtest-list          # List model_id row counts in forecast + archive tables
make backtest-clean MODELS="lgbm_cluster catboost_cluster"  # Remove specific model predictions

# Forecast date-range cleanup
make forecast-clean-list                                             # List row counts by model + month
make forecast-clean ARGS="--before 2025-04-01 --model external"     # Delete external forecasts before Apr 2025
make forecast-clean ARGS="--between 2024-01-01 2024-07-01"          # Delete all models between Jan-Jun 2024
make forecast-clean ARGS="--months 2024-03 2024-06 2024-09"         # Delete specific months
make forecast-clean ARGS="--months 2025-01 --model external"        # Delete one month for one model
make forecast-clean ARGS="--before 2025-01-01 --dry-run"            # Preview without deleting
make forecast-clean ARGS="--after 2025-06-01 --forecast-only"       # Only clean main table
make forecast-clean ARGS="--before 2025-01-01 --date-column fcstdate --archive-only"  # Archive only, by fcstdate

# Testing
make test              # Run all backend pytest tests
make test-unit         # Backend unit tests only (common/ modules)
make test-api          # Backend API endpoint tests only
make test-cov          # Backend tests with coverage report
make ui-test           # Run frontend vitest unit tests
make test-all          # Run all backend + frontend tests

# E2E Testing (Playwright)
make e2e-install       # Install Playwright browsers (one-time)
make e2e               # Run Playwright E2E smoke tests (requires API on :8000)
make e2e-ui            # Run Playwright in interactive UI mode
make e2e-headed        # Run E2E with visible browser
make e2e-report        # Open last HTML test report
```

---

## Architecture

### Domain-Driven Generic Design

All datasets extend a single `DomainSpec` dataclass in `common/domain_specs.py`. Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

**8 Domains:**
- **Dimensions (read-only):** `item`, `location`, `customer`, `time`, `dfu`
- **Facts (time-series):** `sales`, `forecast`
- **Inventory:** `inventory` (dedicated normalize script + API endpoints + UI tab)

### Data Flow

```
Source CSV → normalize_dataset_csv.py → clean CSV
                                              ↓
                              load_dataset_postgres.py
                                              ↓
                                       PostgreSQL 16
                                              ↓
                                          FastAPI
                                              ↓
                                      React UI (:5173)
```

### API Pattern

- Generic domains served via: `GET /domains/{domain}/rows`, `GET /domains/{domain}/search`, etc.
- Inventory has dedicated endpoints: `GET /inventory/position`, `GET /inventory/kpis`, `GET /inventory/trend`, `GET /inventory/item-detail`
- Pagination: offset/limit (50–1000 rows)
- Reserved word workaround: `class` column aliased as `class_` in responses

---

## Data Models

### Dimension Tables
- Surrogate key `sk`, composite key `ck`, `load_ts`, `modified_ts`
- Full-text search on configured fields via `pg_trgm` trigram indexes

### Fact Tables
- `fact_sales_monthly`: grain = item + customer_group + location + month + type; measures = qty_shipped, qty_ordered, qty
- `fact_external_forecast_monthly`: grain = item + loc + forecast_date + actual_month; tracks lag 0–4 months; measures = base forecast + actual demand
- `fact_inventory_snapshot`: grain = item_no + loc + snapshot_date; measures = qty_on_hand, qty_on_hand_on_order, qty_on_order, mtd_sales, lead_time_days (~190M rows)

### Archive Tables
- `backtest_lag_archive`: All-lags (0–4) backtest predictions for accuracy analysis at any horizon. Grain = forecast_ck + model_id + lag. Includes `timeframe` column for traceability.

### Materialized Views
- `agg_sales_monthly`, `agg_forecast_monthly` — pre-aggregated for O(1) KPI queries
- `agg_inventory_monthly` — monthly EOM on-hand, EOM on-hand+on-order, avg on-hand, monthly sales (MAX of cumulative MTD), avg daily sales (via LAG CTE), snapshot days, latest lead time. Daily sales derived from cumulative `mtd_sales` via LAG() window function. Enables DOS, WOC, Turns, LT Coverage computation.
- `mv_inventory_forecast_monthly` — inventory-forecast bridge: joins `agg_inventory_monthly` + `fact_external_forecast_monthly` + `dim_dfu`. Grain = item_no + loc + month_start + model_id. Computed: forecast_error, abs_error, dos, is_stockout, is_excess, bias_direction. Enables stockout/excess root cause attribution by forecast model.
- `mv_fill_rate_monthly` — fill rate metrics aggregated by item-location-month from inventory snapshot data (IPfeature8).
- `mv_supplier_performance` — supplier delivery performance KPIs aggregated from inventory receipt data (IPfeature12).
- `mv_intramonth_stockout` — within-month stockout events detected from daily inventory snapshots before end-of-month (IPfeature14).
- `mv_control_tower_kpis` — cross-dimensional KPIs aggregating key supply chain health metrics for the Control Tower dashboard (IPfeature15).
- `mv_network_balance` — per-item network balance metrics (DOS CV, excess/shortage location counts) from agg_inventory_monthly + fact_safety_stock_targets, filtered to items with 2+ locations.

### Inventory Planning Fact Tables (IPfeature9-13)
- `fact_demand_signals`: grain = item_no + loc + signal_date; short-horizon demand signals from sales velocity and inventory movement.
- `fact_ss_simulation_results`: grain = simulation_id + item_no + loc; Monte Carlo safety stock simulation output with service level probability distributions.
- `fact_inventory_investment_plan`: grain = item_no + loc + plan_date; computed capital investment allocation per DFU.
- `fact_efficient_frontier`: efficient frontier data points (budget vs. service level trade-off curve) for investment optimization.

---

## Frontend Features

- Paginated data explorer with column filtering and sorting
- Type-aware column filters: `=exact` prefix for B-tree match, plain text for GIN trigram substring search
- Column-level typeahead suggestions via native HTML datalist (text columns only)
- Chemistry-themed loading overlay: periodic table element tile with pulse-glow animation (replaces invisible spinner)
- Approximate row count badge (`100,000+`) for large filtered queries
- KPI cards: Accuracy %, WAPE, MAPE, Bias, Total Forecast/Actual
- KPI window selector: 1–12 month rolling window
- Multi-metric trend charts (dual Y-axis: volume left, accuracy % right)
- Item/Location filter with typeahead suggestions
- Champion Selection panel: model competition config, run, and FVA model-wins visualization
- Market Intelligence tab: item/location selector with Google web search + GPT-4o narrative briefing
- Item Analysis tab (merged DFU Analysis + Inventory): unified tab with checkbox toggle toolbar (7 toggleable panels grouped as Demand: Forecast Chart, SHAP, Model KPIs; and Supply: Inv KPIs, Position Table, Variability, Lead Time). Toggle state persisted in localStorage via `usePanelToggles` hook. Includes: sales vs multi-model forecast overlay chart, 3 scope modes, per-model KPI cards, toggleable measures; **clickable forecast lines** (selected=thicker+full opacity, others=30% fade) + **per-DFU SHAP Panel** (`DfuShapPanel`) showing signed SHAP feature contributions per month as stacked bar chart (historical bars at 90% opacity, future at 45%); fallback to cluster-level summary SHAP on 404; inventory KPI cards, trend chart, paginated position table, item detail drill-down. Old `DfuAnalysisTab.tsx` and `InventoryTab.tsx` kept in repo but no longer imported.
- Collapsible sidebar navigation (16 items, 5 sections, mobile drawer, `[` toggle)
- Dashboard overview landing page: KPI sparkline cards, alert panel, heatmap, top movers, forecast trend chart
- Global filter bar: brand, category, item (searchable), location (searchable), market, channel multi-select dropdowns — applied to dashboard, accuracy, and auto-populated into tab-local inputs
- Single professional theme (Demand Studio) with light/dark modes via CSS variable palettes
- Keyboard shortcuts (1-9 tab switch — 5=Item Analysis, 6=exceptions, 7=invPlanning, 8=jobs; `[` sidebar, `d` dark mode, / search, Esc close, ? help, Ctrl+E fields)
- Lazy-loaded tab components with per-tab error boundaries
- TanStack Query caching (stale-while-revalidate, instant tab switching)
- Virtualized data grid with column resize, row selection, CSV export
- Print-ready CSS (@media print rules)
- ECharts integration for canvas-based charting
- Inventory backtest tab: model comparison (stockout/excess/service level/WAPE), root cause attribution (bias direction), monthly trend, DFU-level event detail table
- Clustering What-If Scenarios panel: parameter controls, scenario simulation, result charts, promote flow, background execution with runtime estimation, dashboard completion alerts, enhanced charts (elbow with optimal K, silhouette with quality zones, feature importance, cluster size pie, gap statistic), scenario queueing (queued status when group busy), "View Results" navigation from JobsTab, Past Scenarios history (last 10 completed runs with inline charts)
- Job Scheduler/Monitor tab (APScheduler-powered): automation dashboard with KPI cards (Total Jobs, Active Now, Success Rate, Avg Duration), grouped job type cards with category colors (blue=clustering, violet=backtest, emerald=seasonality, amber=champion), "Run Now" and schedule buttons, live active job monitoring with animated progress bars and elapsed timers, schedule dialog with presets (hourly/6h/daily 2AM/weekly), recurring schedules section with cron badges, expandable job history with params/results/errors, sidebar active job count badge, cross-tab alerts via `JobNotificationContext`, ClustersTab "Schedule Scenario Job" integration
- Feature Importance (SHAP) panel in Accuracy tab: collapsible card with model selector (populated from `/forecast/shap/models`), timeframe selector (cross-timeframe summary or individual timeframes A–J), horizontal bar chart with indigo=selected / gray=dropped feature coloring, `selected_count`/`n_timeframes` consistency indicator (Feature 42)
- Inventory Planning tab (IPfeature4-14): professional two-column layout with fixed 220px grouped sidebar navigation (7 groups with colored dividers, group labels, and icon-labeled buttons — Daily Operations/red: Exceptions, Health; Optimize/blue: EOQ, Policy, Rebalancing; Analytics/emerald: Fill Rate, ABC-XYZ, Supplier, Intramonth; Planning/violet: Safety Stock, Variability, Lead Time, Signals, Simulation, Investment, Repl. Plan, Demand Fcst; Sensing/teal: Blended Demand, Echelon SS; Strategic/amber: Financial Plan, Events, Scenarios; Supply/slate: Demand Plan, Override Queue, Procurement, Open POs, Projection, Planned Orders) + scrollable main content area with per-panel header bar showing title + description. All 27 panels; sidebar nav item "Inv. Planning"
- Control Tower tab (IPfeature15): unified operational command center with cross-dimensional KPI cards, active alert list, top-critical items list, and trend chart aggregating key supply chain health metrics
- AI Planner tab (IPAIfeature1): NOT a chatbot — a proactive exception work-queue. Portfolio health bar (4 KPI chips: Open Insights, Critical, High Priority, Total Financial Risk), insight cards (severity badge, DFU identity, ABC/cluster chips, 1-sentence summary, specific recommendation, metrics row, financial impact chip, collapsible AI reasoning chain), Acknowledge/Resolve action buttons, planning memo panel (latest portfolio narrative in markdown + model_version badge), "Generate Now" button triggers async portfolio scan; sidebar nav item "AI Planner" with `Sparkles` icon (16 items total)
- Vitest testing infrastructure
- Playwright E2E smoke tests: 8 test files covering navigation, dashboard, accuracy, global filters, inv planning, AI planner, control tower, theme toggle. Config in `frontend/e2e/playwright.config.ts`. Auto-starts Vite dev server; requires API on :8000.

---

## Mandatory Testing Rules

**Every new feature, endpoint, component, hook, or utility MUST include tests. Every removed feature MUST have its tests removed. Tests MUST pass before work is considered complete.**

### When adding functionality:
1. **New Python module in `common/`** → Add unit tests in `tests/unit/test_<module>.py`
2. **New API endpoint** → Add API tests in `tests/api/test_<feature>.py` using httpx AsyncClient with ASGI transport
3. **New React component** → Add component tests in `src/components/__tests__/<Component>.test.tsx`
4. **New React hook** → Add hook tests in `src/hooks/__tests__/<hook>.test.ts`
5. **New utility function** → Add tests in `src/lib/__tests__/<util>.test.ts`
6. **New tab component** → Add smoke tests in `src/tabs/__tests__/<Tab>.test.tsx`
7. **New sidebar tab** → Add E2E navigation test in `frontend/e2e/tests/navigation.spec.ts`
8. **New Inv. Planning sub-tab** → Add E2E sub-tab test in `frontend/e2e/tests/inv-planning.spec.ts`

### When removing functionality:
1. Delete the corresponding test files
2. Remove any fixtures that are no longer needed
3. Update `conftest.py` if shared fixtures were affected

### Test execution:
- Run `make test-all` after every change to verify no regressions
- Backend tests: `make test` (~0.7s, no infra needed — DB is mocked)
- Frontend tests: `make ui-test` (258 tests, ~1.5s)
- E2E tests: `make e2e` (requires API on :8000 + Vite auto-started; ~30s)
- Coverage: `make test-cov` for backend coverage report

### Test patterns:
- **Backend API tests:** Use `httpx.AsyncClient(transport=ASGITransport(app))` — no running server needed
- **Backend mocking:** Mock `pool` fixture in `tests/api/conftest.py` for DB; use `@patch.dict("sys.modules")` for imports inside functions
- **Frontend component tests:** Wrap with `QueryClientProvider` from `src/tabs/__tests__/test-utils.tsx`
- **Frontend mocking:** Use `vi.mock("../api/queries")` for API layer; mock `echarts-for-react` for chart components
- **E2E tests (Playwright):** Real browser tests against running dev server. Config in `frontend/e2e/playwright.config.ts`. Use `navigateToTab()` fixture from `e2e/fixtures/base.ts`. Assertions use Playwright auto-wait (`toBeVisible`, `toHaveURL`).
- **E2E selectors:** Use semantic selectors (`getByRole("button", { name })`, `getByText()`) — never CSS classes or fragile DOM paths

### Reference:
- Full testing strategy: `docs/design-specs/feature31.md`
- E2E testing spec: `docs/specs/06-ui-platform/06-07-e2e-visual-testing.md`
- Backend test directory: `mvp/demand/tests/`
- Frontend test directories: `mvp/demand/frontend/src/**/__tests__/`
- E2E test directory: `mvp/demand/frontend/e2e/tests/`

---

## Important Conventions

- **Config files for all modules:** Every module (script, job, computation pipeline) MUST externalize all configurable parameters into a YAML file under `mvp/demand/config/`. No magic numbers, thresholds, default values, or tuning knobs hardcoded inside Python scripts. Scripts load their config via `yaml.safe_load(open("config/<module>_config.yaml"))` at startup. Examples: `variability_config.yaml` (CV thresholds, history_months), `safety_stock_config.yaml` (service levels by ABC class, Z-table, guard rails), `eoq_config.yaml` (ordering cost, holding cost, MOQ source), `simulation_config.yaml` (n_simulations, random_seed), `replenishment_policy_config.yaml` (policy definitions, auto-assign rules). API endpoints that expose config values must read from the YAML file, not from hardcoded dicts.

- **Null normalization:** `''`, `'null'`, `'none'`, `'NA'` all treated as NULL during load
- **Type casting:** Integer/float/date fields auto-cast with null coercion in normalize scripts
- **Lag computation:** `month_diff` auto-computed during forecast normalization
- **Forecast accuracy formula:** `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias formula:** `(SUM(Forecast) / SUM(History)) - 1`
- **Sales filtering:** Only rows with `TYPE=1` are loaded into `fact_sales_monthly`
- **Time dimension:** Auto-generated 2020–2035, not sourced from a file
- **Forecast model_id:** Identifies the forecasting algorithm; default `'external'` for source-system forecasts. `UNIQUE(forecast_ck, model_id)` constraint prevents duplicates within a model. Not part of the business key.
- **Forecast execution-lag loading:** `load_dataset_postgres.py` uses a **dual-path insert with phase ordering** to preserve archive integrity. **Phase 3b** loads the archive FIRST from untouched staging data — each row's original `lag` is preserved as `execution_lag` in `backtest_lag_archive`. **Phase 3c** THEN mutates the staging table's `execution_lag` column from `dim_dfu` (matched DFUs get the DFU-level value, unmatched default to 0). **Phase 5** inserts into the main table with `WHERE lag = execution_lag` — only execution-lag rows enter `fact_external_forecast_monthly`. This ordering is critical: the archive must read staging before the mutation, otherwise all rows for a DFU would have the same `execution_lag` value, corrupting multi-horizon accuracy analysis. Use `--replace` flag (or `make load-forecast-replace`) to only delete+reload `model_id='external'` rows without truncating the whole table (preserves backtest/champion/ceiling data). Use `--skip-archive` flag (or `make load-forecast-replace-no-archive`) to skip the archive load entirely — only the execution-lag row is loaded into the main table, bypassing the 45M-row archive INSERT for faster external forecast reloads. See `feature2.md` § "Forecast Loading: Dual-Path with Execution-Lag Filtering" for a worked example.
- **Chat endpoint:** `POST /chat` — OpenAI-powered NL→SQL with pgvector context retrieval. Read-only execution with 5s timeout and 500-row limit. Requires `OPENAI_API_KEY` in `.env`.
- **DFU clustering:** KMeans-based clustering pipeline groups DFUs by demand patterns. Feature engineering extracts **14 core features across 6 dimensions** (volume, trend, seasonality, periodicity, intermittency, lifecycle) from sales history (default: 36-month window). New features include FFT periodicity strength, OLS seasonal R-squared, Croston ADI, scale-invariant trend slope, IQR, CAGR, recency ratio, and YoY correlation. Optimal K selection uses **combined Silhouette + Calinski-Harabasz scoring** (`0.5 * sil_norm + 0.5 * CH_norm`; gap statistic removed) with a **hard 5% minimum cluster size constraint** (k_range [5,18]). Log-transforms skewed volume features before StandardScaler. Post-hoc `merge_small_clusters()` merges any remaining small clusters into nearest large neighbor. Labeling uses a **priority-ordered taxonomy**: Intermittency -> Periodicity -> Seasonality -> Trend -> Volatility -> Volume (5 tiers: very_high/high/medium/low/very_low). Compound labels like `high_volume_seasonal_growing`. Cluster labels stored in `dim_dfu.cluster_assignment`. MLflow tracks experiments under `dfu_clustering`. Config in `config/clustering_config.yaml`.
- **Champion model selection:** Configurable per-DFU per-month champion selection via 5 strategies: expanding (cumulative WAPE), rolling (last N months), decay (exponential weighting), ensemble (blend top-K models), meta_learner (ML classifier). All strategies enforce **exec-lag-aware strict causality** — selection for month T with execution_lag=L uses ONLY data from months where `startdate < T − L` (i.e. `startdate < fcstdate`), implemented as `shift(exec_lag + 1)` per DFU-model group. This prevents using actuals that weren't available when the forecast was issued. With `exec_lag=0` the behaviour is identical to `shift(1)` (backward compatible). **Fallback model** (`fallback_model_id: lgbm_cluster` by default): DFU-months in the warm-up period (first `exec_lag + min_dfu_rows` months with insufficient prior history) are filled with the fallback model's forecast so every DFU-month always has a champion row. Strategy registry in `common/champion_strategies.py`. Config in `config/model_competition.yaml` controls competing models, metric, lag, `min_dfu_rows`, `fallback_model_id`, `strategy`, and `strategy_params`. Champion rows stored as `model_id='champion'` in `fact_external_forecast_monthly`. Ceiling (oracle) picks the best model per DFU per month with perfect foresight (after-the-fact), stored as `model_id='ceiling'`. Both at DFU-month granularity with consistent WAPE formula `SUM(|F-A|) / |SUM(A)|`. Gap-to-ceiling quantifies improvement opportunity. Meta-learner uses ceiling labels as ground truth with strict temporal train/test split. Simulation script (`scripts/simulate_champion_strategies.py`) runs all strategies and compares accuracy vs ceiling. UI panel in Accuracy tab shows champion + ceiling KPI cards, gap-to-ceiling indicator, and dual model wins bar charts.
- **Shared backtest framework (Feature 44):** All tree-based backtest scripts (LGBM, CatBoost, XGBoost) use `common/backtest_framework.py` as a shared orchestrator via `run_tree_backtest()`. Each script implements both `train_and_predict_per_cluster()` and `train_and_predict_global()` and selects which to pass based on the `cluster_strategy` key in `config/algorithm_config.yaml`. Algorithm options (cluster_strategy, recursive, shap_select, tune_inline, params_file, hyperparameters) are read from `config/algorithm_config.yaml` — not from CLI flags. Shared modules in `common/`: `backtest_framework.py`, `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`, `tuning.py` (CV splits, fold functions, per-timeframe inline tuner), `shap_selector.py` (SHAP extraction, feature selection, CSV output). `run_tree_backtest()` optional parameters: `inline_tuner_fn` (PL-002, per-timeframe causal tuning), `feature_selector_fn` (Feature 42, SHAP retraining), `recursive: bool = False` (Feature 43, recursive multi-step inference via `update_grid_with_predictions()`). `_predict_single_month(models, predict_data, feature_cols)` routes each recursive inference step to the correct cluster model. **`ml_cluster` is always a hard feature** — never stripped from `feature_cols` in either strategy.
- **Backtest output paths (model-scoped subdirectories):** Each backtest run writes output to `data/backtest/<model_id>/` (e.g., `data/backtest/lgbm_cluster/backtest_predictions.csv`). Multiple models can be run sequentially without overwriting each other. Load with `make backtest-load MODEL=<model_id>` or `make backtest-load-all` (scans all subdirs). See PL-001 in `docs/PARKING_LOT.md` for history.
- **Hyperparameter tuning (Feature 41):** Bayesian Optuna tuning for LGBM, CatBoost, XGBoost cluster models. Walk-forward CV with causal masking (`mask_future_sales()` inside every fold). WAPE stabilised with denominator floor. `n_estimators` determined by early stopping (not in search space). Outputs `data/tuning/best_params_<model>.json` with `best_params` + `best_n_estimators` + per-cluster WAPEs. Apply tuned params by setting `params_file: data/tuning/best_params_lgbm.json` in `config/algorithm_config.yaml` (Feature 44). Make targets: `tune-lgbm`, `tune-catboost`, `tune-xgboost`, `tune-all`. MLflow experiment: `hyperparameter_tuning`. Config: `config/hyperparameter_tuning.yaml`. Shared utilities: `common/tuning.py`. **Two-mode workflow:** (1) Production scoring: tune once on full history (`make tune-lgbm`), apply via `params_file` in algorithm config — fastest path, no future leakage for production use. (2) Honest backtesting: per-timeframe causal tuning via `tune_inline: true` in algorithm config (PL-002 fix) — each timeframe tunes on only the data available at that point in time; no future leakage into backtest accuracy metrics. `TRAIN_FOLD_FNS` registry in `common/tuning.py` exposes shared fold functions for both global tuning and inline tuner.
- **SHAP feature selection (Feature 42):** Per-timeframe automatic feature selection using SHAP values for LGBM, CatBoost, and XGBoost backtests. Activated by `shap_select: true` in `config/algorithm_config.yaml` (Feature 44). Flow per timeframe: (1) train initial model on all features, (2) compute SHAP via `common/shap_selector.py`, (3) select features covering 95% cumulative SHAP mass (or exactly `shap_top_n` features), (4) retrain final model on selected features. CatBoost uses native `get_feature_importance(type="ShapValues")`; LGBM/XGBoost use `shap.TreeExplainer` (requires `shap>=0.43.0`). SHAP is pooled across clusters weighted by cluster size; `ml_cluster` is excluded from the effective feature set. Output written to `data/backtest/<model_id>/shap/shap_timeframe_XX.csv` (per-timeframe) and `shap_summary.csv` (cross-timeframe). Served via 4 read-only REST endpoints under `/forecast/shap/` (no DB queries — CSV-based). Composable with `tune_inline` (PL-002) and `params_file` via config keys. Config keys: `shap_select`, `shap_top_n`, `shap_threshold` (default 0.95), `shap_sample_size` (default 500). Graceful fallback: if SHAP computation fails, all features are kept and the backtest continues.
- **Recursive multi-step forecasting (Feature 43):** Enabled via `recursive: true` in `config/algorithm_config.yaml` (Feature 44). In direct mode (default), all future months are predicted from the same lag-1-zero baseline (masked sales = 0 for months 2+). In recursive mode, each month in the prediction window is forecast one at a time; the model's prediction for month T is written back into the feature grid via `update_grid_with_predictions()` in `common/feature_engineering.py`, recomputing all lag and rolling features before month T+1 is scored. This gives `qty_lag_1` a real signal (model's own prior prediction) instead of zero. `_predict_single_month(models, predict_data, feature_cols)` in `common/backtest_framework.py` routes inference to the correct cluster model dict during the recursive loop (inference-only, no retraining). `_fill_predict_nans()` fills numeric NaNs per-month. `recursive: bool = False` is the parameter on `run_tree_backtest()`. Composable with `shap_select` and `tune_inline` via config keys. `"recursive": true` written to `backtest_metadata.json` for traceability. No API, frontend, or DB changes. Trade-off: richer near-horizon signal vs potential error compounding across months.
- **Market intelligence:** `POST /market-intelligence` — combines Google Custom Search API (product news/trends) + GPT-4o narrative synthesis for item + location pairs. Looks up item metadata (description, brand, category) from `dim_item` and location state from `dim_location`. Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in `.env`.
- **Backtest cleanup:** `scripts/clean_backtest_models.py` selectively removes model predictions from `fact_external_forecast_monthly` and `backtest_lag_archive` by `model_id`, then refreshes 5 materialized views. Supports `--list`, `--dry-run`, `--all-backtest` (excludes `external`). Make targets: `backtest-clean`, `backtest-list`.
- **Forecast date-range cleanup:** `scripts/clean_forecasts_by_date.py` deletes rows from `fact_external_forecast_monthly` and/or `backtest_lag_archive` by time bucket. Supports `--before`, `--after`, `--between` date range filters and `--months` for specific month(s) on `startdate` (default) or `fcstdate`, optional `--model` filter, `--forecast-only`/`--archive-only` scope, `--dry-run` preview, and `--list` for row counts by model+month. All dates normalized to month-start. Refreshes same 5 materialized views as `clean_backtest_models.py`. Make targets: `forecast-clean`, `forecast-clean-list`.
- **Inventory snapshots:** 14 monthly CSV files (`datafiles/Inventory_Snapshot_YYYY_MM.csv`, ~190M rows total) merged by `scripts/normalize_inventory_csv.py` into a single clean CSV. Loaded into `fact_inventory_snapshot` via generic loader. `qty_on_order` derived as `qty_on_hand_on_order - qty_on_hand` during normalization. Dedicated API endpoints (`/inventory/*`) and frontend Item Analysis tab (inventory panels). `agg_inventory_monthly` materialized view with daily sales derivation (LAG CTE), EOM snapshots, and proper monthly sales (MAX not SUM). `/inventory/kpis` uses two-query pattern: point-in-time totals from latest snapshot + trailing-month aggregates for supply chain KPIs (DOS, WOC, Inventory Turns, LT Coverage). KPI cards use severity color-coding (green/yellow/red thresholds). Trend chart renders 5 lines: On Hand, On Order, Monthly Sales, Lead Time, Days of Supply.
- **DFU seasonality detection:** Pipeline in `scripts/detect_seasonality.py` + `update_seasonality_profiles.py` computes seasonality metrics (strength, profile label, peak/trough month, peak-to-trough ratio, is_yearly_seasonal flag) from sales history and writes them to `dim_dfu`. Config in `config/seasonality_config.yaml`. DDL in `sql/015_add_seasonality_columns.sql`. Make targets: `seasonality-detect`, `seasonality-update`, `seasonality-all`. These 6 columns (`seasonality_profile`, `seasonality_strength`, `is_yearly_seasonal`, `peak_month`, `trough_month`, `peak_trough_ratio`) are now part of `DFU_SPEC` and are exposed by the generic Data Explorer.
- **What-If clustering scenarios:** `POST /clustering/scenario` runs a trial KMeans pipeline with custom `feature_params`, `model_params`, and `label_params` without overwriting production clustering. Returns HTTP 202 immediately and runs in background thread; `GET /clustering/scenario/{id}/status` polls for running/completed/failed. `GET /clustering/scenario/estimate` returns runtime estimate based on DFU count, K range, and gap flag. `POST /clustering/scenario/{id}/promote` applies the winning scenario to `dim_dfu.ml_cluster`. `ScenarioNotificationContext` tracks running/completed state across tabs; Dashboard injects completion alert. Enhanced charts: elbow with optimal K ReferenceLine, silhouette bar chart with quality zone thresholds (Strong/Reasonable/Weak/No structure), feature importance horizontal bars, cluster size pie chart, conditional gap statistic line chart. Requires `API_KEY` env var to be set for auth (disabled when unset). **Scenario queueing:** When a clustering job is already running, new scenarios are queued (`status="queued"`) instead of rejected with 409; queued jobs auto-dispatch via `_dispatch_next()` when the active job completes. **View Results:** "View Results" button in JobsTab navigates to ClustersTab with `?scenario_job=<id>` URL param; ClustersTab auto-loads result and renders ScenarioCharts. **Past Scenarios:** ClustersTab What-If panel shows last 10 completed scenario runs in an accordion with inline charts and promote buttons.
- **Modular API router architecture:** `api/routers/` contains 53 FastAPI `APIRouter` modules (see api/routers/ for full list). `main.py` is a ~149-line shell that only creates the app, adds middleware, and mounts all 53 routers via `app.include_router()`. All route handlers live in router modules — no inline routes in main.py. `domains.py` is mounted last because it has catch-all `{domain}` path parameters. All mutation endpoints require `require_api_key` auth when `API_KEY` env var is set.
- **Job scheduler (APScheduler):** `common/job_registry.py` provides `JobManager` singleton powered by APScheduler 3.11 (`BackgroundScheduler` + `ThreadPoolExecutor(max_workers=4)`). Thread-safe: `_state_lock` guards `_active_jobs`, `_pending_queues`, `_cancel_flags`; `_init_lock` with double-checked locking protects `_ensure_init()`. `JOB_TYPE_REGISTRY` maps 7 job types across 4 groups. Per-group concurrency control with FIFO queueing (one active job per group: clustering, backtest, seasonality, champion; busy groups queue jobs instead of rejecting). Job callables wrap existing scripts via `subprocess.run()`. Progress updates written to `job_history` table. `recover_stale_jobs()` re-enqueues queued jobs from DB on restart and marks running jobs as failed. Supports cron/interval scheduling (`POST /jobs/schedule`, `GET /jobs/schedules`), job pipelines (`POST /jobs/pipeline` — sequential chaining), retry logic with exponential backoff (`max_retries`), and dashboard stats (`GET /jobs/stats`). 12 REST API endpoints total. Route ordering in `jobs.py`: literal paths (`/jobs/schedules`, `/jobs/pipeline`) must come before parameterized `{job_id}` paths. Frontend polls `GET /jobs/active` every 2s, stats every 5s, history every 10s. `JobNotificationContext` provides cross-tab completion alerts. Sidebar shows active job count badge. ClustersTab uses "Schedule Scenario Job" button. Dependencies: `apscheduler>=3.10`, `tzlocal>=5.0`.
- **API key authentication:** `api/auth.py` provides `require_api_key` FastAPI dependency. Auth is disabled when the `API_KEY` env var is unset (development default). When set, mutation endpoints (`POST /clustering/scenario`, `PUT /competition/config`, `POST /competition/run`, `POST /chat`, `POST /market-intelligence`) require `X-API-Key` header.
- **Vite dev server proxy:** `frontend/vite.config.ts` proxies all API path prefixes (`/domains`, `/jobs`, `/clustering`, `/forecast`, `/inventory`, `/dashboard`, `/health`, `/chat`, `/dfu`, `/competition`, `/bench`, `/market-intelligence`, `/inv-planning`, `/fill-rate`, `/control-tower`, `/ai-planner`, `/storyboard`) to the FastAPI backend at `http://127.0.0.1:8000`. **CRITICAL:** When adding a new API path prefix, you MUST add a corresponding proxy entry in `vite.config.ts` or the frontend will receive HTML instead of JSON. Restart the Vite dev server (`make ui`) after changes.
- **Health endpoint DB pattern (IPfeature6):** All `inv_planning_*.py` router files use `get_conn()` directly (NOT `Depends(_get_pool)`). Using `Depends(_get_pool)` causes 422 errors when `api.main` is first imported inside a `patch("api.core._get_pool", ...)` test context — FastAPI inspects the MagicMock's signature as `(*args, **kwargs)` and turns them into required query params. All new endpoints in any `inv_planning_*.py` file must use `get_conn()`.
- **Shared utilities pattern:** `common/utils.py` provides `_ts()` (HH:MM:SS timestamp for console logging) and `load_config(name)` / `reset_config(name)` (thread-safe YAML config loader with per-file caching via double-checked locking). All `common/` modules that load YAML config use `load_config()` instead of their own `_config_cache` global. All modules and scripts that need a timestamp helper import `from common.utils import _ts` instead of defining their own. Module-level singletons (`get_cache()`, `get_rate_limiter()`, `get_tracker()`, `get_planning_date()`) use `threading.Lock()` with double-checked locking for thread safety.
- **Shared DB params pattern:** All scripts import `from common.db import get_db_params` — no inline `_get_db_params`/`_db_conn` functions. `common/db.py` is the canonical source.
- **Planning date pattern:** All date-sensitive operations import `from common.planning_date import get_planning_date` and call `get_planning_date()` instead of `date.today()`. Config in `config/planning_config.yaml` (`planning_date: "2026-02-24"`, `use_system_date: false`). Env var overrides: `PLANNING_DATE=2026-02-24` (specific date) or `USE_SYSTEM_DATE=true` (use real system date). Precedence: `USE_SYSTEM_DATE` env > `PLANNING_DATE` env > config file > `date.today()` fallback. Config is cached per-process; use `_reset_cache()` in tests to reset between test cases.
- **Shared test pool factory:** `tests/api/conftest.py` exports `make_pool(fetchall_return=None, fetchone_return=None)` module-level factory (NOT a pytest fixture). Defaults: fetchall=[], fetchone=(0,). Import with `from tests.api.conftest import make_pool as _make_pool`. For endpoints making multiple `fetchall()` calls, use `cursor.fetchall.side_effect = [list1, list2, ...]`; for single-call endpoints use `cursor.fetchall.return_value = [...]`.
- **Stub table pattern (IPfeature6):** When a materialized view depends on a table from a future feature (e.g., `fact_safety_stock_targets` for IPfeature6 depends on IPfeature3), create the stub table with the minimum required columns using `CREATE TABLE IF NOT EXISTS`. The LEFT JOIN produces NULL for all rows, causing score components to use neutral scores. When the real table is populated by the upstream feature, real scores flow automatically with zero code changes.
- **Single theme with light/dark modes:** Only the "General" (Demand Studio) product theme remains. `useTheme()` manages light/dark color mode. `ThemeSelector` in sidebar footer provides light/dark toggle. No theme cycling, no motifs.
- **Theme context (no prop-drilling):** Tab components access the current theme via `useThemeContext()` from `context/ThemeContext.tsx` or `useChartColors()` from `hooks/useChartColors.ts` — NOT via a `theme` prop from `App.tsx`. `ThemeProvider` wraps the app tree in `App.tsx`. `useChartColors()` returns `{ theme, chartColors, trendColors }` for Recharts styling. `ScenarioCharts` component extracted to `components/ScenarioCharts.tsx` (elbow, silhouette, radar, pie, gap charts).
- **Algorithm configuration (Feature 44):** All backtest algorithm options are controlled by `config/algorithm_config.yaml`, not CLI flags. Backtest scripts for LGBM, CatBoost, and XGBoost accept only `--config`, `--model-id`, and `--n-timeframes`. Features (cluster_strategy, recursive, shap_select, tune_inline, params_file, default hyperparameters) are set per-algorithm in the YAML file. Each algorithm has a `cluster_strategy` key (`per_cluster` or `global`): `per_cluster` trains one model per ml_cluster partition, `global` trains one model on all data. **`ml_cluster` is always a hard feature** — it is included in `feature_cols` for both strategies and is never stripped. In `per_cluster` mode it provides a constant identity signal; in `global` mode it provides inter-cluster discrimination. Each backtest script implements both `train_and_predict_per_cluster()` and `train_and_predict_global()` and selects which to pass to `run_tree_backtest()` based on the config. Prophet, StatsForecast, NeuralProphet, PatchTST, and DeepAR scripts were deleted.
- **Inventory Rebalancing:** `scripts/compute_rebalancing.py` detects cross-location inventory imbalances using DOS CV from `mv_network_balance`, builds transfer candidates between excess and shortage locations, and solves via greedy or LP solver (configurable in `config/rebalancing_config.yaml`). Output written to `fact_rebalancing_plan` (plan header) + `fact_rebalancing_transfer` (individual transfers). `dim_transfer_lane` defines the network topology (valid source→destination pairs with transit days and cost per unit). 12 REST endpoints in `inv_planning_rebalancing.py` cover KPIs, network view, imbalance list, plan CRUD, transfer detail, and approval workflow (draft→approved→in_transit→completed). Uses `get_conn()` directly (same as all `inv_planning_*.py` routers). Frontend panel "Rebalancing" in the Optimize group of InvPlanningTab.
- **AI Planning Agent (IPAIfeature1):** NOT a chatbot — a proactive exception work-queue. `AIPlannerAgent` in `common/ai_planner.py` uses `anthropic.Anthropic()` client with `tool_use` API (model: `claude-opus-4-6`). 10 tools: `get_dfu_full_context`, `get_forecast_performance`, `get_portfolio_exceptions`, `compute_bias_trend`, `get_inventory_trend`, `get_eoq_context`, `get_similar_dfus`, `check_stockout_history`, `get_portfolio_health_summary` (all read-only SQL), and `create_insight` (INSERT into `ai_insights`). Agentic loop is circuit-breaker guarded: MAX_TURNS=40, TOKEN_BUDGET=100_000 per run; both OpenAI and Anthropic loops terminate when either limit is hit. `create_insight` calls are validated by `CreateInsightInput` Pydantic model before any DB write — invalid `insight_type`, summary without a digit, or `financial_impact_estimate > $10M` are rejected and logged (return -1). `log_ai_call` writes per-turn token usage + tool call latency to `ai_call_log` table (best-effort, non-fatal). System prompt includes 2 worked few-shot examples anchoring summary/recommendation quality. `INTERVAL '%s months'` bug fixed → `INTERVAL '1 month' * %s`. Three async methods: `run_dfu_analysis(item_no, loc, scan_run_id)`, `run_portfolio_scan(scan_run_id)`, `generate_portfolio_memo(period, scan_run_id)`. POST /ai-planner/portfolio-scan returns 202 immediately; scan runs in background thread via `_executor.submit(...)`. GET /ai-planner/insights returns `{"insights": [...], "total": N}` (key is `insights` not `rows`). New endpoint: GET /ai-planner/metrics?days=7 — per-model token/latency/error aggregates from `ai_call_log`. Config in `config/ai_planner_config.yaml` — model, thresholds, schedule. Dependency: `anthropic>=0.40.0`. Frontend AIPlannerTab.tsx: confidence badge (HIGH/MED/LOW derived from WAPE + bias + financial impact), last-scan timestamp in header, auto-dismiss scan success banner (5s), context-aware empty state ("Portfolio looks healthy!" when no open exceptions). `frontend/src/constants/design-tokens.ts` defines semantic color palette and UX limits.

---

## Design Specs

Located in `docs/specs/` — 6 domains, 40 files, `DD-SS-descriptive-name.md` convention:

### 01-data-platform/
- `01-01-infrastructure.md` — Tech stack, Docker Compose, services, implemented-features master index
- `01-02-data-models.md` — Data architecture + ERD + dimension tables (Item/Location/Customer/Time/DFU) + fact tables (Sales, Forecast)
- `01-03-benchmarking.md` — *(removed — benchmarking feature deleted)*
- `01-03-planning-date.md` — Planning date configuration: `get_planning_date()`, frozen dev date, env var overrides, 22 production files migrated

### 02-forecasting/ (includes demand intelligence: clustering, seasonality, blended demand)
- `02-01-accuracy-kpis.md` — Accuracy metrics (WAPE/bias/accuracy%) + multi-dimensional slicing (agg_accuracy_by_dim, lag-curve)
- `02-02-multi-model-support.md` — model_id column, UNIQUE constraint, /models endpoint
- `02-03-backtest-framework.md` — Expanding window timeframes (A-J), dual-path storage, lag 0-4 archive
- `02-04-tree-model-implementations.md` — LGBM + CatBoost + XGBoost per-cluster backtests (shared feature engineering, model-specific sections)
- `02-05-champion-selection.md` — 5 strategies, exec-lag-aware causality, ceiling model, meta-learner, FVA
- `02-06-advanced-backtest.md` — Hyperparameter tuning (Optuna) + SHAP feature selection + recursive multi-step forecasting
- `02-07-algorithm-config.md` — algorithm_config.yaml, cluster_strategy (per_cluster/global), config keys reference
- `02-08-production-forecast.md` — Production inference pipeline, fact_production_forecast, versioning, CI bands
- `02-09-bias-correction.md` — Forward inventory projection (F1.2) + bias correction engine (F3.1)
- `02-09-forecast-ci-bands.md` — Forecast confidence interval bands
- `02-10-dfu-clustering.md` — KMeans engine + what-if scenario UI + enhancements (background exec, charts, queuing, past scenarios)
- `02-11-seasonality.md` — Seasonality detection pipeline + profile filtering in Accuracy/DFU analysis tabs
- `02-12-blended-demand.md` — Alpha-weighted sensing + statistical blend

### 03-inventory-planning/
- `03-01-inventory-snapshot.md` — Snapshot ingestion (190M rows, agg views) + backtest/attribution (stockout/excess root cause)
- `03-02-demand-variability.md` — Demand variability profiling (CV, MAD, skewness) + lead time variability (LT CV, reliability)
- `03-03-safety-stock.md` — Safety stock engine (Z-score, ROP) + Monte Carlo simulation (service level curves)
- `03-04-replenishment.md` — EOQ cycle stock + replenishment policies (4 types, auto-assign) + health score (4-component × 25pt)
- `03-05-exception-queue.md` — 6 exception types, severity ranking, 7-day dedup, recommendations
- `03-06-analytics.md` — Fill rate analytics + demand signals (short-horizon) + intramonth stockout detection
- `03-07-abc-xyz-supplier.md` — ABC-XYZ policy matrix (3×3) + supplier performance analytics
- `03-08-investment-optimization.md` — Efficient frontier, budget allocation, fact_inventory_investment_plan
- `03-09-inventory-planning-reference.md` — Original world-class design vision (reference; see 03-01 through 03-08 for implementation)
- `03-10-multi-echelon-ss.md` — Multi-echelon safety stock with cascade risk severity badges
- `03-11-replenishment-plan.md` — Replenishment plan computation and management
- `03-12-inventory-rebalancing.md` — Inventory rebalancing: network topology, imbalance detection, greedy/LP solvers, transfer plans, approval workflow

### 04-operations/
- `04-01-sop-cycle.md` — S&OP stage machine, cycle phases, approval workflow
- `04-02-financial-planning.md` — Inventory value, carrying cost, budget utilization
- `04-03-event-calendar.md` — Promotion & event calendar, approval status
- `04-04-scenario-planning.md` — Disruption scenarios, what-if planning, financial impact results

### 05-ai-platform/
- `05-01-ai-planning-agent.md` — Claude tool_use agent, 10 tools, proactive exception work-queue, ai_insights tables
- `05-02-chatbot-market-intel.md` — NL→SQL chatbot (GPT-4o + pgvector) + market intelligence (Google search + GPT-4o narrative)
- `05-03-control-tower.md` — mv_control_tower_kpis, cross-dimensional KPI cards, active alerts, top-critical
- `05-04-storyboard.md` — Exception-based planner workflow, causal chain cards, decision logging

### 06-ui-platform/
- `06-01-data-explorer.md` — Data explorer UX (type-aware filtering, GIN, typeahead) + DFU analysis tab (sales vs multi-model overlay)
- `06-02-ui-architecture.md` — Component architecture (Vite, TanStack Query, virtualization) + product-grade UI overhaul (sidebar, global filters, dashboard, theme)
- `06-03-theming.md` — Light/dark modes (single Demand Studio theme)
- `06-04-job-scheduler.md` — APScheduler 3.11, 12 endpoints, 4 job groups, pipelines, retry logic
- `06-05-testing-strategy.md` — Full-stack pytest + Vitest testing spec, mandatory test requirements
- `06-06-backtest-cleanup.md` — clean_backtest_models.py, selective model deletion, view refresh
- `06-07-e2e-visual-testing.md` — Playwright E2E smoke tests, testing pyramid, 8 test files

### 07-platform-integration/
- `07-01-integration-architecture.md` — Four bidirectional integration vectors: notifications (Slack/Teams/Email/PagerDuty), REST API consumers (CORS, rate limiting, webhooks), cloud data pipelines (Snowflake/BigQuery/S3/Databricks), ERP/WMS adapters (SAP/Oracle/NetSuite/Manhattan)
- `07-02-data-quality.md` — Data quality engine, rule evaluation, scoring dashboard (08-01)
- `07-03-rbac.md` — Role-based access control, user management (08-02)
- `07-04-caching.md` — Caching strategy, cache invalidation (08-03)
- `07-05-notifications.md` — Notification channels, preferences, dispatch engine (08-04)
- `07-06-collaboration.md` — Collaboration threads, annotations, shared views (08-05)
- `07-07-external-signals.md` — External demand signal ingestion, decomposition (08-06)
- `07-08-fva.md` — Forecast Value Add tracking, ROI measurement (08-07)
- `07-09-reporting.md` — Report templates, scheduling, delivery (08-08)
- `07-10-api-governance.md` — API rate limiting, versioning, usage metrics (08-09)
- `07-11-webhooks.md` — Webhook registration, event dispatch, delivery tracking (08-10)

### Archived specs (deleted code / superseded designs)
Located in `docs/archive/` — feature14 (transfer learning), feature19-21/24-25 (archived ML models), feature35 (deleted motif themes), theme-testing-strategy (orphaned), feature27 (Figma MCP, not started)

---

## Documentation Update Rules

**Whenever ANY code is added, changed, or deleted in the codebase, you MUST update ALL of the following documentation files to keep them in sync:**

1. **`mvp/demand/docs/ARCHITECTURE.md`** — Update architecture, component technologies, tables, or data flow if affected
2. **`mvp/demand/docs/README.md`** — Update stack, datasets, analytics behavior, quick start, or key paths if affected
3. **`mvp/demand/docs/RUNBOOK.md`** — Update setup steps, notes, or troubleshooting if affected
4. **`mvp/demand/docs/WORKFLOW.md`** — Update the end-to-end workflow if any pipeline phase is added, removed, or reordered — new Make targets, new schema steps, new scripts, or changes to the dependency chain between phases
5. **`mvp/demand/docs/product_summary.md`** — Update the product summary when a feature area is added, removed, or significantly changed — update the relevant Feature Area section, Data Scale row counts if tables change, UI tab counts, and test counts
6. **`docs/specs/<domain>/<DD-SS-name>.md`** — Create or update the design spec for the feature in the appropriate domain folder
7. **`docs/specs/01-data-platform/01-01-infrastructure.md`** — Add the feature to the "Implemented Features" list
8. **`CLAUDE.md`** (this file) — Update Key Files, Common Commands, Data Models, Frontend Features, Important Conventions, or Design Specs list if affected

**This applies to ALL changes — additions, modifications, AND deletions. When code is removed, the corresponding references in ALL documentation files above must also be removed or updated.**

**Additionally, you MUST write tests for every change and run `make test-all` to verify they pass:**

9. **`mvp/demand/tests/`** — Add or update backend tests for any new/modified Python modules or API endpoints
10. **`mvp/demand/frontend/src/**/__tests__/`** — Add or update frontend tests for any new/modified components, hooks, or utilities
11. **Run `make test-all`** — Verify all 1457+ tests pass (both backend and frontend) before considering the work complete

**What counts as changes requiring doc updates:**
- New feature implementation (new endpoints, UI panels, tables, scripts)
- Schema changes (new columns, tables, indexes, materialized views)
- New dependencies or infrastructure changes (docker images, pyproject.toml)
- New Make targets or CLI commands
- Changes to data flow or pipeline behavior — including new pipeline phases, reordered steps, or new schema/compute prerequisites
- Removal or renaming of any module, endpoint, component, table, or script
- Refactors that change architecture, file structure, or public interfaces

**What does NOT require doc updates:**
- Bug fixes that don't change behavior or interfaces
- Minor internal code refactors that don't change architecture or file structure
- Typo corrections within code (not in docs)

**What ALWAYS requires tests (even for bug fixes):**
- Any new Python function or class
- Any new API endpoint or modification to existing endpoint behavior
- Any new React component, hook, or utility
- Bug fixes that change behavior (add a regression test)

---

## Do Not

- Do not commit `__pycache__/`, `.pyc` files, or `.venv/`
- Do not modify `mvp/demand/data/*.csv` files manually — they are generated by normalize scripts
- Do not touch the `reference/` directory — it is archived code
