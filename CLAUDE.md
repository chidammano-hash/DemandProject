# CLAUDE.md — Demand Studio

## Project Overview

**Demand Studio** is a unified demand forecasting analytics platform. It ingests sales and forecast data, stores it in PostgreSQL (OLTP) and Apache Iceberg (lakehouse), and serves a React UI for interactive analytics.

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
| Lakehouse | Apache Iceberg via MinIO + Iceberg REST |
| Big Data | Apache Spark 3.5 |
| Query Engine | Trino |
| ML / Clustering | scikit-learn, pandas, scipy, matplotlib, seaborn, StatsForecast, NeuralProphet |
| ML Tracking | MLflow |
| Python packaging | uv |
| Build | Make |
| Containers | Docker Compose |

---

## Key Files

| File | Purpose |
|---|---|
| `mvp/demand/common/domain_specs.py` | Central config: all 8 datasets (dimensions + facts + inventory) with columns, types, keys |
| `mvp/demand/api/main.py` | FastAPI backend — primary endpoints + mounts `api/routers/clusters` for What-If and seasonality routes |
| `mvp/demand/frontend/src/App.tsx` | React UI — sidebar layout shell (~200 lines, lazy-loaded tabs) |
| `mvp/demand/Makefile` | All dev commands |
| `mvp/demand/docker-compose.yml` | 7-service infra cluster |
| `mvp/demand/scripts/normalize_dataset_csv.py` | Generic ETL: CSV → clean CSV |
| `mvp/demand/scripts/load_dataset_postgres.py` | Generic loader: clean CSV → PostgreSQL |
| `mvp/demand/scripts/spark_dataset_to_iceberg.py` | Spark job: clean CSV → Iceberg |
| `mvp/demand/sql/` | DDL for all tables, indexes, materialized views |
| `mvp/demand/sql/017_create_fact_inventory_snapshot.sql` | Inventory snapshot table DDL, indexes, materialized view |
| `mvp/demand/scripts/normalize_inventory_csv.py` | Inventory ETL: merge 14 monthly CSVs → single clean CSV |
| `mvp/demand/scripts/generate_clustering_features.py` | Feature engineering: sales history → clustering feature matrix |
| `mvp/demand/scripts/train_clustering_model.py` | KMeans clustering with optimal K selection + MLflow logging |
| `mvp/demand/scripts/label_clusters.py` | Assign business labels to clusters based on feature centroids |
| `mvp/demand/scripts/update_cluster_assignments.py` | Write cluster labels to `dim_dfu.cluster_assignment` in Postgres |
| `mvp/demand/config/clustering_config.yaml` | Clustering hyperparameters and labeling thresholds |
| `mvp/demand/config/model_competition.yaml` | Champion model selection: competing models, metric, lag |
| `mvp/demand/scripts/run_champion_selection.py` | Per-DFU champion selection: best-of-models via WAPE |
| `mvp/demand/common/backtest_framework.py` | Shared backtest orchestrator: `run_tree_backtest()`, timeframes, data loading, output saving |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix: lag/rolling features, future masking, `cat_dtype` parameter |
| `mvp/demand/common/metrics.py` | Shared accuracy metrics: WAPE, bias, accuracy % |
| `mvp/demand/common/mlflow_utils.py` | Shared MLflow logging wrapper for backtest runs |
| `mvp/demand/common/db.py` | Shared DB connection parameters |
| `mvp/demand/common/constants.py` | Shared constants: `CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`, output columns, thresholds |
| `mvp/demand/scripts/run_backtest.py` | LGBM backtest: model-specific training functions (uses shared framework) |
| `mvp/demand/scripts/run_backtest_catboost.py` | CatBoost backtest: model-specific training functions (uses shared framework) |
| `mvp/demand/scripts/run_backtest_xgboost.py` | XGBoost backtest: model-specific training functions (uses shared framework) |
| `mvp/demand/scripts/run_backtest_prophet.py` | Prophet backtest: per-DFU fitting with multiprocessing (uses shared utilities) |
| `mvp/demand/scripts/run_backtest_statsforecast.py` | StatsForecast backtest: vectorized AutoARIMA + AutoETS (uses shared utilities) |
| `mvp/demand/scripts/run_backtest_neuralprophet.py` | NeuralProphet backtest: PyTorch-based per-DFU fitting with GPU support (uses shared utilities) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load backtest predictions into Postgres (main + archive) |
| `mvp/demand/scripts/clean_backtest_models.py` | Selective cleanup of model predictions from Postgres + view refresh |
| `mvp/demand/sql/010_create_backtest_lag_archive.sql` | DDL for backtest all-lags archive table |
| `mvp/demand/sql/008_perf_indexes_and_agg.sql` | Performance indexes (B-tree, GIN trigram) + materialized views |
| `mvp/demand/frontend/src/api/queries.ts` | Centralized TanStack Query layer (all fetch functions + query keys) |
| `mvp/demand/frontend/src/tabs/` | Extracted tab components (DashboardTab, ExplorerTab, AccuracyTab, DfuAnalysisTab, ClustersTab, MarketIntelTab, ChatPanel) |
| `mvp/demand/frontend/src/hooks/useTheme.ts` | Product theme + color mode management (3 themes, light/dark) |
| `mvp/demand/frontend/src/hooks/useUrlState.ts` | URL state synchronization (9 tabs, overview default) |
| `mvp/demand/frontend/src/hooks/useKeyboardShortcuts.ts` | Keyboard shortcuts handler (1-7 tabs, sidebar, theme) |
| `mvp/demand/frontend/src/hooks/useSidebar.ts` | Sidebar collapse/expand state + mobile drawer |
| `mvp/demand/frontend/src/hooks/useGlobalFilters.ts` | Global filter state with debounced URL sync |
| `mvp/demand/frontend/src/context/GlobalFilterContext.tsx` | Global filter React context provider |
| `mvp/demand/frontend/src/types/theme.ts` | TypeScript types for themes, sidebar, filters, dashboard |
| `mvp/demand/frontend/src/constants/themes/` | Product theme configs (wineSpirits, general, obsidian) |
| `mvp/demand/frontend/src/components/AppSidebar.tsx` | Collapsible sidebar navigation (9 items, 5 sections) |
| `mvp/demand/frontend/src/components/ThemeSelector.tsx` | Theme + color mode picker (sidebar footer) |
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
| `mvp/demand/frontend/vitest.config.ts` | Vitest test configuration |
| `mvp/demand/frontend/tailwind.config.ts` | Tailwind config with custom `pulse-glow` animation |
| `mvp/demand/tests/` | Backend test suite (pytest): unit/ + api/ |
| `mvp/demand/tests/conftest.py` | Shared pytest fixtures (sample DataFrames) |
| `mvp/demand/tests/api/conftest.py` | API test fixtures (mock DB pool, async httpx client) |
| `mvp/demand/frontend/src/**/__tests__/` | Frontend test suites (Vitest + RTL) |
| `docs/design-specs/` | Feature specs (feature1–feature36) |
| `mvp/demand/api/core.py` | Shared API utilities: connection pool, OpenAI client, SQL helpers used by router modules |
| `mvp/demand/api/auth.py` | Optional API key auth (`require_api_key` dependency; disabled when `API_KEY` env var unset) |
| `mvp/demand/api/routers/` | Modular FastAPI router modules (clusters, accuracy, analysis, benchmark, chat, competition, domains, intel) |
| `mvp/demand/config/seasonality_config.yaml` | Seasonality detection hyperparameters and profile labeling thresholds |
| `mvp/demand/scripts/detect_seasonality.py` | Compute seasonality metrics per DFU (strength, profile, peak/trough month) |
| `mvp/demand/scripts/update_seasonality_profiles.py` | Write seasonality profiles to `dim_dfu` in Postgres |
| `mvp/demand/scripts/run_clustering_scenario.py` | What-If clustering: run trial KMeans with custom params + promote flow |
| `mvp/demand/scripts/run_backtest_patchtst.py` | PatchTST backtest: deep-learning transformer model (Apple MPS GPU) |
| `mvp/demand/scripts/run_backtest_deepar.py` | DeepAR backtest: LSTM probabilistic forecasting |
| `mvp/demand/sql/013_add_composite_indexes.sql` | Composite B-tree indexes for multi-column query performance |
| `mvp/demand/sql/015_add_seasonality_columns.sql` | DDL: 6 seasonality columns on `dim_dfu` (Feature 30) |
| `mvp/demand/sql/016_add_seasonality_to_accuracy_views.sql` | DDL: seasonality joins in accuracy materialized views (Feature 32) |
| `mvp/demand/frontend/src/hooks/useMotifTheme.ts` | Motif selection, localStorage + URL persistence, CSS data-attr injection |
| `mvp/demand/frontend/src/hooks/useDebounce.ts` | Generic debounce hook used by filter inputs |
| `mvp/demand/frontend/src/context/MotifContext.tsx` | React context provider for active motif theme |
| `mvp/demand/frontend/src/constants/motifRegistry.ts` | Registry mapping motif IDs to motif config objects |
| `mvp/demand/frontend/src/constants/motifs/` | 5 motif theme definitions: periodic, spirits, space, f1, zen |
| `mvp/demand/frontend/src/constants/colors.ts` | Shared color palette constants |
| `mvp/demand/frontend/src/constants/elements.ts` | Periodic table element definitions for loading overlay |
| `mvp/demand/frontend/src/types/motif.ts` | TypeScript types for motif system (MotifId, MotifThemeConfig, TileConfig) |
| `mvp/demand/frontend/src/components/MotifSettingsPanel.tsx` | Motif theme picker panel (opened by Ctrl+M) |
| `mvp/demand/frontend/src/components/KeyboardShortcutHelp.tsx` | Keyboard shortcut help modal (triggered by `?` shortcut) |
| `mvp/demand/frontend/src/components/KpiCard.tsx` | Reusable KPI metric card component |
| `mvp/demand/frontend/src/lib/utils.ts` | Shared utilities: `cn()` Tailwind class merger and misc helpers |

---

## Common Commands

```bash
# One-time setup
make init              # Create .venv, install uv, sync dependencies

# Infrastructure
make up                # Start Docker services (Postgres, MinIO, Spark, Trino, MLflow)
make down              # Stop all services
make db-apply-sql      # Apply DDL schemas to Postgres

# Data pipeline
make normalize-all     # Normalize all 8 datasets (CSV → clean CSV)
make load-all          # Load cleaned data into Postgres + refresh materialized views
make spark-all         # Publish datasets to Iceberg (optional)

# Inventory pipeline
make db-apply-inventory     # Create inventory table + indexes + materialized view (one-time)
make normalize-inventory    # Merge 14 monthly CSVs into single clean CSV
make load-inventory         # Load into Postgres + refresh agg view
make inventory-pipeline     # normalize + load + refresh (all-in-one)

# Run services
make api               # Start FastAPI on :8000
make ui-init           # Install npm deps
make ui                # Start React dev server on :5173

# Validation
make check-db          # Table row counts in Postgres
make check-api         # Curl API health + sample endpoints
make check-all         # Full check: DB + API + Trino

# Chatbot
make db-apply-chat     # Apply pgvector + embeddings table DDL
make generate-embeddings  # Generate and store schema embeddings (requires OPENAI_API_KEY)

# Benchmarking
make bench-compare DOMAIN=sales RUNS=7 ITEM=100320 LOCATION=1401-BULK

# Clustering pipeline
make cluster-features  # Generate clustering feature matrix from sales/DFU/item data
make cluster-train     # Train KMeans, select optimal K, log to MLflow
make cluster-label     # Assign business labels to clusters
make cluster-update    # Write cluster labels to dim_dfu in Postgres
make cluster-all       # Run full clustering pipeline (features → train → label → update)

# Backtesting (LGBM)
make backtest-lgbm          # Run global LGBM backtest (10 expanding timeframes)
make backtest-lgbm-cluster  # Run per-cluster LGBM backtest
make backtest-lgbm-transfer # Run LGBM transfer learning backtest

# Backtesting (CatBoost)
make backtest-catboost          # Run global CatBoost backtest (10 expanding timeframes)
make backtest-catboost-cluster  # Run per-cluster CatBoost backtest
make backtest-catboost-transfer # Run CatBoost transfer learning backtest

# Backtesting (XGBoost)
make backtest-xgboost          # Run global XGBoost backtest (10 expanding timeframes)
make backtest-xgboost-cluster  # Run per-cluster XGBoost backtest
make backtest-xgboost-transfer # Run XGBoost transfer learning backtest

# Backtesting (StatsForecast)
make backtest-statsforecast          # Run global StatsForecast backtest (AutoARIMA+AutoETS, ~100x faster)
make backtest-statsforecast-cluster  # Run per-cluster StatsForecast backtest
make backtest-statsforecast-pooled   # Run StatsForecast pooled cluster backtest

# Backtesting (NeuralProphet)
make backtest-neuralprophet          # Run global NeuralProphet backtest (PyTorch GPU)
make backtest-neuralprophet-cluster  # Run per-cluster NeuralProphet backtest
make backtest-neuralprophet-pooled   # Run NeuralProphet pooled cluster backtest

# Backtest loading (shared across all models)
make backtest-load          # Load backtest predictions into Postgres + refresh agg
make backtest-all           # backtest-lgbm + backtest-load

# Champion model selection
make champion-select        # Run per-DFU champion selection (best-of-models via WAPE)

# Seasonality pipeline (feature 30)
make seasonality-schema     # Apply DDL for seasonality columns on dim_dfu (one-time)
make seasonality-detect     # Detect seasonality patterns per DFU from sales history
make seasonality-update     # Write seasonality profiles back to dim_dfu
make seasonality-all        # Full pipeline: schema + detect + update

# Backtest cleanup
make backtest-list          # List model_id row counts in forecast + archive tables
make backtest-clean MODELS="lgbm_global deepar_global"  # Remove specific model predictions

# Testing
make test              # Run all backend pytest tests
make test-unit         # Backend unit tests only (common/ modules)
make test-api          # Backend API endpoint tests only
make test-cov          # Backend tests with coverage report
make ui-test           # Run frontend vitest unit tests
make test-all          # Run all backend + frontend tests
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
                          ┌───────────────────┴───────────────────┐
                          ▼                                         ▼
              load_dataset_postgres.py                spark_dataset_to_iceberg.py
                          ▼                                         ▼
                    PostgreSQL 16                          Apache Iceberg (MinIO)
                          ▼                                         ▼
                      FastAPI                                    Trino SQL
                          ▼
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
- `agg_inventory_monthly` — monthly avg on-hand, avg on-order, avg lead time, total MTD sales

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
- Postgres vs Iceberg latency benchmarking panel
- Champion Selection panel: model competition config, run, and FVA model-wins visualization
- Market Intelligence tab: item/location selector with Google web search + GPT-4o narrative briefing
- DFU Analysis tab: unified sales vs multi-model forecast overlay chart, 3 scope modes, per-model KPI cards, toggleable measures
- Collapsible sidebar navigation (9 items, 5 sections, mobile drawer, `[` toggle)
- Dashboard overview landing page: KPI sparkline cards, alert panel, heatmap, top movers, forecast trend chart
- Global filter bar: brand, category, item (searchable), location (searchable), market, channel multi-select dropdowns — applied to dashboard, accuracy, and auto-populated into tab-local inputs
- Three product themes: Wine & Spirits, General, Obsidian (CSS variable palettes, light/dark modes)
- Keyboard shortcuts (1-7 tab switch, `[` sidebar, `t` theme, `d` dark mode, / search, Esc close, ? help, Ctrl+E fields)
- Lazy-loaded tab components with per-tab error boundaries
- TanStack Query caching (stale-while-revalidate, instant tab switching)
- Virtualized data grid with column resize, row selection, CSV export
- Print-ready CSS (@media print rules)
- ECharts integration for canvas-based charting
- Inventory tab: KPI cards, trend chart, paginated position table, item detail drill-down
- Clustering What-If Scenarios panel: parameter controls, scenario simulation, result charts, promote flow
- Five motif themes: Periodic Table, Wine & Spirits, Space, Formula 1, Zen Garden — each with distinct tiles, icons, and loading animations; selectable independently of light/dark color mode
- `MotifSettingsPanel` accessible via Ctrl+M keyboard shortcut
- `?motif=<id>` URL parameter for deep-linking to a specific motif theme
- Vitest testing infrastructure

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

### When removing functionality:
1. Delete the corresponding test files
2. Remove any fixtures that are no longer needed
3. Update `conftest.py` if shared fixtures were affected

### Test execution:
- Run `make test-all` after every change to verify no regressions
- Backend tests: `make test` (~0.7s, no infra needed — DB is mocked)
- Frontend tests: `make ui-test` (218 tests, ~1.5s)
- Coverage: `make test-cov` for backend coverage report

### Test patterns:
- **Backend API tests:** Use `httpx.AsyncClient(transport=ASGITransport(app))` — no running server needed
- **Backend mocking:** Mock `pool` fixture in `tests/api/conftest.py` for DB; use `@patch.dict("sys.modules")` for imports inside functions
- **Frontend component tests:** Wrap with `QueryClientProvider` from `src/tabs/__tests__/test-utils.tsx`
- **Frontend mocking:** Use `vi.mock("../api/queries")` for API layer; mock `echarts-for-react` for chart components

### Reference:
- Full testing strategy: `docs/design-specs/feature31.md`
- Backend test directory: `mvp/demand/tests/`
- Frontend test directories: `mvp/demand/frontend/src/**/__tests__/`

---

## Important Conventions

- **Null normalization:** `''`, `'null'`, `'none'`, `'NA'` all treated as NULL during load
- **Type casting:** Integer/float/date fields auto-cast with null coercion in normalize scripts
- **Lag computation:** `month_diff` auto-computed during forecast normalization
- **Forecast accuracy formula:** `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias formula:** `(SUM(Forecast) / SUM(History)) - 1`
- **Sales filtering:** Only rows with `TYPE=1` are loaded into `fact_sales_monthly`
- **Time dimension:** Auto-generated 2020–2035, not sourced from a file
- **Forecast model_id:** Identifies the forecasting algorithm; default `'external'` for source-system forecasts. `UNIQUE(forecast_ck, model_id)` constraint prevents duplicates within a model. Not part of the business key.
- **Chat endpoint:** `POST /chat` — OpenAI-powered NL→SQL with pgvector context retrieval. Read-only execution with 5s timeout and 500-row limit. Requires `OPENAI_API_KEY` in `.env`.
- **DFU clustering:** KMeans-based clustering pipeline groups DFUs by demand patterns. Feature engineering extracts time series, item, and DFU features. Cluster labels (e.g., `high_volume_steady`, `seasonal_medium_volume`) stored in `dim_dfu.cluster_assignment`. MLflow tracks experiments under `dfu_clustering`. Config in `config/clustering_config.yaml`.
- **Champion model selection:** Rolling/expanding window per-DFU per-month champion selection via WAPE (Forecast Value Added). At each month, picks the model with lowest cumulative WAPE from prior months only (before-the-fact). Config in `config/model_competition.yaml` controls competing models, metric, lag, and `min_dfu_rows` (minimum prior months required). Champion rows stored as `model_id='champion'` in `fact_external_forecast_monthly`. Ceiling (oracle) picks the best model per DFU per month with perfect foresight (after-the-fact), stored as `model_id='ceiling'`. Both at DFU-month granularity with consistent WAPE formula `SUM(|F-A|) / |SUM(A)|`. Gap-to-ceiling quantifies improvement opportunity. UI panel in Accuracy tab shows champion + ceiling KPI cards, gap-to-ceiling indicator, and dual model wins bar charts.
- **Shared backtest framework:** All tree-based backtest scripts (LGBM, CatBoost, XGBoost) use `common/backtest_framework.py` as a shared orchestrator via `run_tree_backtest()`. Each script implements only model-specific training functions passed as callables. Prophet and NeuralProphet use shared utilities but orchestrate their own per-DFU fitting loops. StatsForecast uses shared utilities with vectorized batch fitting (no per-DFU loop, ~100x faster than Prophet). Shared modules in `common/`: `backtest_framework.py`, `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`.
- **Market intelligence:** `POST /market-intelligence` — combines Google Custom Search API (product news/trends) + GPT-4o narrative synthesis for item + location pairs. Looks up item metadata (description, brand, category) from `dim_item` and location state from `dim_location`. Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in `.env`.
- **Backtest cleanup:** `scripts/clean_backtest_models.py` selectively removes model predictions from `fact_external_forecast_monthly` and `backtest_lag_archive` by `model_id`, then refreshes 5 materialized views. Supports `--list`, `--dry-run`, `--all-backtest` (excludes `external`). Make targets: `backtest-clean`, `backtest-list`.
- **Benchmarking:** `GET /bench/compare` runs identical queries (count, page, trend) against Postgres and Trino/Iceberg, returning per-query latency stats (min/max/avg/p50/p95) with winner determination and speedup factor. Requires Docker services running. Make target: `bench-compare`.
- **Inventory snapshots:** 14 monthly CSV files (`datafiles/Inventory_Snapshot_YYYY_MM.csv`, ~190M rows total) merged by `scripts/normalize_inventory_csv.py` into a single clean CSV. Loaded into `fact_inventory_snapshot` via generic loader. `qty_on_order` derived as `qty_on_hand_on_order - qty_on_hand` during normalization. Dedicated API endpoints (`/inventory/*`) and frontend InventoryTab. `agg_inventory_monthly` materialized view for trend queries.
- **DFU seasonality detection:** Pipeline in `scripts/detect_seasonality.py` + `update_seasonality_profiles.py` computes seasonality metrics (strength, profile label, peak/trough month, peak-to-trough ratio, is_yearly_seasonal flag) from sales history and writes them to `dim_dfu`. Config in `config/seasonality_config.yaml`. DDL in `sql/015_add_seasonality_columns.sql`. Make targets: `seasonality-detect`, `seasonality-update`, `seasonality-all`. These 6 columns (`seasonality_profile`, `seasonality_strength`, `is_yearly_seasonal`, `peak_month`, `trough_month`, `peak_trough_ratio`) are now part of `DFU_SPEC` and are exposed by the generic Data Explorer.
- **What-If clustering scenarios:** `POST /clustering/scenario` runs a trial KMeans pipeline with custom `feature_params`, `model_params`, and `label_params` without overwriting production clustering. `POST /clustering/scenario/{id}/promote` applies the winning scenario to `dim_dfu.ml_cluster`. The UI panel is fully implemented in ClustersTab. Requires `API_KEY` env var to be set for auth (disabled when unset).
- **Modular API router architecture:** `api/routers/` contains 8 FastAPI `APIRouter` modules (clusters, accuracy, analysis, benchmark, chat, competition, domains, intel). These are imported and mounted via `app.include_router()` at the end of `main.py`. Existing inline `@app.get` routes registered earlier take precedence for duplicate paths. The `clusters` router provides the What-If scenario and seasonality-profile endpoints not in the inline routes.
- **API key authentication:** `api/auth.py` provides `require_api_key` FastAPI dependency. Auth is disabled when the `API_KEY` env var is unset (development default). When set, mutation endpoints (`POST /clustering/scenario`, `PUT /competition/config`, `POST /competition/run`, `POST /chat`, `POST /market-intelligence`) require `X-API-Key` header.
- **Motif theme system:** 5 visual motifs (periodic, spirits, space, f1, zen) defined in `frontend/src/constants/motifs/`. Selected motif persists in localStorage and URL (`?motif=<id>`). `useMotifTheme` hook manages state; `MotifContext` provides it app-wide. Ctrl+M opens `MotifSettingsPanel`. Motif identity is separate from product theme (wine & spirits, general, obsidian) and color mode (light/dark).

---

## Design Specs

Located in `docs/design-specs/`:
- `feature1.md` — Infrastructure & platform setup
- `feature2.md` — Internal data architecture & data contracts (includes ERD)
- `feature3.md` — Dimension tables (Item, Location, Customer, Time, DFU)
- `feature4.md` — Fact tables (Sales, External Forecast)
- `feature5.md` — Forecast accuracy KPIs
- `feature6.md` — Multi-model forecast support
- `feature7.md` — DFU clustering framework
- `feature8.md` — Backtesting framework (expanding window timeframes)
- `feature9.md` — LGBM backtesting implementation
- `feature10.md` — Multi-dimensional accuracy slicing
- `feature11.md` — Chatbot / natural language queries
- `feature12.md` — CatBoost backtesting implementation
- `feature13.md` — XGBoost backtesting implementation
- `feature14.md` — Transfer learning backtest strategy
- `feature15.md` — Champion model selection (rolling window best-of-models per DFU per month)
- `feature16.md` — Data Explorer performance & UX (type-aware filtering, GIN indexes, column typeahead, loading overlay)
- `feature17.md` — DFU Analysis tab (sales vs multi-model forecast overlay)
- `feature18.md` — Market intelligence (web search + LLM narrative briefings)
- `feature19.md` — PatchTST backtesting implementation (deep learning, Apple MPS GPU)
- `feature20.md` — DeepAR backtesting implementation (LSTM probabilistic forecasting)
- `feature21.md` — Prophet backtesting implementation (per-DFU time series)
- `feature22.md` — UI theming (dark mode + midnight theme)
- `feature23.md` — Backtest model cleanup utility (selective model removal + view refresh)
- `feature24.md` — StatsForecast backtesting implementation (vectorized AutoARIMA + AutoETS)
- `feature25.md` — NeuralProphet backtesting implementation (PyTorch-based Prophet with GPU)
- `feature26.md` — Postgres vs Trino/Iceberg benchmarking (latency comparison API)
- `feature27.md` — Figma MCP Integration: Design-to-Code & Code-to-Design Workflow *(not started)*
- `feature28.md` — UI Architecture & Performance (component decomposition, TanStack Query, lazy loading, error boundaries, keyboard shortcuts, testing)
- `feature29.md` — What-If / Scenario UI for Clustering (UI implemented; backend routes mounted via include_router)
- `feature30.md` — DFU Seasonality Detection & Profile Assignment (scripts + DDL implemented; Makefile targets added)
- `feature31.md` — Comprehensive Testing Strategy (full-stack testing spec, mandatory test requirements)
- `feature32.md` — Seasonality Profile Filtering (backend router written; frontend UI pending)
- `feature33.md` — Inventory Overlay in DFU Analysis *(not implemented)*
- `feature34.md` — Inventory Planning Module Phase 1 (snapshot pipeline, API, UI tab)
- `feature35.md` — Configurable Multi-Theme / Motif System (5 motifs implemented; `?motif=` URL param added)
- `feature36.md` — Product-Grade UI Overhaul (sidebar, themes, dashboard, global filters, widgets)
- `theme-testing-strategy.md` — Multi-Theme Testing Strategy (unit tests implemented; integration/a11y/perf tests pending)
- `docs/REFACTORING_RECOMMENDATIONS.md` — Comprehensive codebase refactoring roadmap

---

## Documentation Update Rules

**Whenever you implement a new feature or make significant changes, you MUST update the following files:**

1. **`docs/design-specs/feature<N>.md`** — Create or update the design spec for the feature
2. **`docs/design-specs/feature1.md`** — Add the feature to the "Implemented Features (MVP)" list
3. **`mvp/demand/docs/ARCHITECTURE.md`** — Update architecture, component technologies, tables, or data flow if affected
4. **`mvp/demand/docs/README.md`** — Update stack, datasets, analytics behavior, quick start, or key paths if affected
5. **`mvp/demand/docs/RUNBOOK.md`** — Update setup steps, notes, or troubleshooting if affected
6. **`CLAUDE.md`** (this file) — Update Key Files, Common Commands, Data Models, Frontend Features, Important Conventions, or Design Specs list if affected

**Additionally, you MUST write tests for every change and run `make test-all` to verify they pass:**

7. **`mvp/demand/tests/`** — Add or update backend tests for any new/modified Python modules or API endpoints
8. **`mvp/demand/frontend/src/**/__tests__/`** — Add or update frontend tests for any new/modified components, hooks, or utilities
9. **Run `make test-all`** — Verify all 485+ tests pass (both backend and frontend) before considering the work complete

**What counts as "significant changes":**
- New feature implementation (new endpoints, UI panels, tables, scripts)
- Schema changes (new columns, tables, indexes, materialized views)
- New dependencies or infrastructure changes (docker images, pyproject.toml)
- New Make targets or CLI commands
- Changes to data flow or pipeline behavior

**What does NOT require doc updates:**
- Bug fixes that don't change behavior or interfaces
- Minor code refactors that don't change architecture
- Typo corrections

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
- Do not run `make spark-all` unless Iceberg/MinIO is needed; Postgres path is sufficient for most dev work
