# Demand Studio — Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        DEMAND STUDIO PLATFORM                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      PRESENTATION LAYER                                             ║
║                              React 18 + Vite 5 + TypeScript (:5173)                                 ║
║                                                                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐    ║
║  │                                    App.tsx (Root Shell)                                      │    ║
║  │  ┌────────────────────────────────── Context Providers ──────────────────────────────────┐   │    ║
║  │  │  ThemeProvider → GlobalFilterProvider → ScenarioNotificationProvider → JobNotifProvider│   │    ║
║  │  └──────────────────────────────────────────────────────────────────────────────────────┘   │    ║
║  │                                                                                             │    ║
║  │  ┌──────────────┐  ┌─────────────────────────────────────────────────────────────────────┐  │    ║
║  │  │  AppSidebar   │  │                      Main Content Area                              │  │    ║
║  │  │  ───────────  │  │  ┌───────────────────────────────────────────────────────────────┐  │  │    ║
║  │  │  Overview     │  │  │                    GlobalFilterBar                             │  │  │    ║
║  │  │  Explorer     │  │  │  [Brand] [Category] [Item] [Location] [Market] [Channel]      │  │  │    ║
║  │  │  Clusters     │  │  └───────────────────────────────────────────────────────────────┘  │  │    ║
║  │  │  DFU Analysis │  │                                                                     │  │    ║
║  │  │  Accuracy     │  │  ┌─────────────── Lazy-Loaded Tabs (Suspense + ErrorBoundary) ───┐  │  │    ║
║  │  │  Intel        │  │  │                                                                │  │  │    ║
║  │  │  Inventory    │  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │  │  │    ║
║  │  │  Inv Backtest │  │  │  │ DashboardTab│ │ ExplorerTab │ │ ClustersTab │              │  │  │    ║
║  │  │  Jobs         │  │  │  │ KPIs/Alerts │ │ 8 Domains   │ │ What-If     │              │  │  │    ║
║  │  │  ───────────  │  │  │  │ Heatmap     │ │ Virtualized │ │ Scenarios   │              │  │  │    ║
║  │  │  Chat ◉       │  │  │  │ TopMovers   │ │ DataTable   │ │ ScenarioChts│              │  │  │    ║
║  │  │  [d] Dark Mode│  │  │  └─────────────┘ └─────────────┘ └─────────────┘              │  │  │    ║
║  │  │  [?] Shortcuts│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │  │  │    ║
║  │  │  Active: (3)  │  │  │  │DfuAnalysis  │ │ AccuracyTab │ │MarketIntel  │              │  │  │    ║
║  │  │               │  │  │  │ Multi-Model │ │ WAPE/Bias   │ │ Web Search  │              │  │  │    ║
║  │  │               │  │  │  │ Overlay     │ │ Champion    │ │ GPT-4o      │              │  │  │    ║
║  │  │               │  │  │  └─────────────┘ └─────────────┘ └─────────────┘              │  │  │    ║
║  │  │               │  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │  │  │    ║
║  │  │               │  │  │  │InventoryTab │ │InvBacktest  │ │  JobsTab    │              │  │  │    ║
║  │  │               │  │  │  │ Position    │ │ Model Comp  │ │ Scheduler   │              │  │  │    ║
║  │  │               │  │  │  │ KPIs/Trend  │ │ Root Cause  │ │ Monitor     │              │  │  │    ║
║  │  │               │  │  │  └─────────────┘ └─────────────┘ └─────────────┘              │  │  │    ║
║  │  │               │  │  │                                                                │  │  │    ║
║  │  └──────────────┘  │  └────────────────────────────────────────────────────────────────┘  │  │    ║
║  │                     │                                            ┌──────────┐              │  │    ║
║  │                     │                                            │ChatPanel │              │  │    ║
║  │                     │                                            │(always)  │              │  │    ║
║  │                     └────────────────────────────────────────────┴──────────┘──────────────┘  │    ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                                     ║
║  ┌──── Hooks ────────────────────┐  ┌──── State ───────────────────┐  ┌──── UI Libs ─────────────┐  ║
║  │ useTheme       useUrlState    │  │ TanStack Query (SWR cache)   │  │ Tailwind CSS + shadcn/ui │  ║
║  │ useSidebar     useDebounce    │  │ React Context (4 providers)  │  │ Recharts + ECharts       │  ║
║  │ useGlobalFilters              │  │ URL State Sync               │  │ TanStack Table + Virtual │  ║
║  │ useKeyboardShortcuts          │  │                              │  │ Radix UI primitives      │  ║
║  │ useChartColors                │  │ queries.ts (50+ query keys)  │  │ lucide-react icons       │  ║
║  └───────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
                                              │
                                    Vite Proxy (12 path prefixes)
                                    /domains /jobs /clustering /forecast
                                    /inventory /dashboard /health /chat
                                    /dfu /competition /bench /market-intelligence
                                              │
                                              ▼
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                       API LAYER                                                     ║
║                              FastAPI + Uvicorn (:8000)                                              ║
║                                                                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐    ║
║  │                               main.py (~65 lines)                                           │    ║
║  │  App creation → GZip middleware → CORS middleware → Mount 12 routers                        │    ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                                     ║
║  ┌──── core.py ──────────────┐  ┌──── auth.py ─────────────────────────────────────────────────┐    ║
║  │ Connection Pool (_pool)   │  │ require_api_key dependency (disabled when API_KEY unset)     │    ║
║  │ OpenAI Client (shared)    │  └──────────────────────────────────────────────────────────────┘    ║
║  │ SQL Helpers / Pagination  │                                                                     ║
║  │ Domain Spec Wrappers      │                                                                     ║
║  └───────────────────────────┘                                                                     ║
║                                                                                                     ║
║  ┌──────────────────────────── 12 Modular API Routers ────────────────────────────────────────┐     ║
║  │                                                                                            │     ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │     ║
║  │  │  accuracy.py  │ │ analysis.py  │ │benchmark.py  │ │   chat.py    │ │ clusters.py  │    │     ║
║  │  │  /forecast/*  │ │ /dfu/*       │ │ /bench/*     │ │ POST /chat   │ │ /clustering/*│    │     ║
║  │  │  Accuracy     │ │ DFU Analysis │ │ PG vs Trino  │ │ NL→SQL      │ │ What-If      │    │     ║
║  │  │  Slicing+Lags │ │ Multi-model  │ │ Latency      │ │ pgvector    │ │ Scenarios    │    │     ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    │     ║
║  │                                                                                            │     ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │     ║
║  │  │competition.py│ │ dashboard.py │ │   intel.py   │ │inv_backtest  │ │ inventory.py │    │     ║
║  │  │/competition/*│ │ /dashboard/* │ │/market-intel │ │/inventory/bt │ │ /inventory/* │    │     ║
║  │  │ Champion     │ │ KPIs/Alerts  │ │ Google+GPT4o │ │ Model Comp   │ │ Position     │    │     ║
║  │  │ Selection    │ │ Heatmap/Move │ │ Briefings    │ │ Root Cause   │ │ KPIs/Trend   │    │     ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    │     ║
║  │                                                                                            │     ║
║  │  ┌──────────────┐ ┌──────────────┐                                                        │     ║
║  │  │   jobs.py    │ │  domains.py  │  ← mounted LAST (catch-all {domain} paths)             │     ║
║  │  │  /jobs/*     │ │ /domains/*   │                                                        │     ║
║  │  │ APScheduler  │ │ Generic CRUD │                                                        │     ║
║  │  │ 12 endpoints │ │ 8 Domains    │                                                        │     ║
║  │  └──────────────┘ └──────────────┘                                                        │     ║
║  └────────────────────────────────────────────────────────────────────────────────────────────┘     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
           │                    │                    │                        │
           │                    │                    │                        │
           ▼                    ▼                    ▼                        ▼
╔══════════════════╗  ╔══════════════════╗  ╔══════════════════╗  ╔═══════════════════════╗
║  COMMON MODULES  ║  ║  JOB SCHEDULER   ║  ║  EXTERNAL APIs   ║  ║   ML TRACKING         ║
║                  ║  ║                  ║  ║                  ║  ║                       ║
║ domain_specs.py  ║  ║ job_registry.py  ║  ║ OpenAI GPT-4o   ║  ║  MLflow (:5003)       ║
║  8 DomainSpecs   ║  ║ APScheduler 3.11 ║  ║  Chat NL→SQL    ║  ║  Experiment tracking  ║
║                  ║  ║ BackgroundSched  ║  ║  Market Intel    ║  ║  Model registry       ║
║ backtest_        ║  ║ ThreadPool(4)    ║  ║                  ║  ║  Artifact storage     ║
║  framework.py    ║  ║ Per-group queues ║  ║ Google CSE API   ║  ║                       ║
║  run_tree_       ║  ║ Cron/Interval    ║  ║  Web search      ║  ╚═══════════════════════╝
║  backtest()      ║  ║ Pipelines        ║  ║                  ║
║                  ║  ║ Retry + backoff  ║  ╚══════════════════╝
║ champion_        ║  ║                  ║
║  strategies.py   ║  ║ 7 Job Types:     ║
║  5 strategies    ║  ║  clustering      ║
║  expanding       ║  ║  backtest_lgbm   ║
║  rolling         ║  ║  backtest_cat    ║
║  decay           ║  ║  backtest_xgb    ║
║  ensemble        ║  ║  seasonality     ║
║  meta_learner    ║  ║  champion        ║
║                  ║  ║  scenario        ║
║ feature_         ║  ╚══════════════════╝
║  engineering.py  ║
║  Lag/Rolling     ║
║                  ║
║ metrics.py       ║
║  WAPE/Bias/Acc   ║
║                  ║
║ mlflow_utils.py  ║
║ db.py            ║
║ constants.py     ║
║                  ║
║ tuning.py        ║
║  CV splits       ║
║  WAPE stable     ║
║  Optuna suggest  ║
║  save/load JSON  ║
╚══════════════════╝
           │
           ▼
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                     DATA PIPELINE LAYER                                             ║
║                                   31 Production Scripts                                             ║
║                                                                                                     ║
║  ┌─────────────── ETL ──────────────────┐  ┌──────────── Backtesting (8 Models) ────────────────┐   ║
║  │                                       │  │                                                    │   ║
║  │  normalize_dataset_csv.py  (generic)  │  │  ┌─ Tree-Based (shared framework) ──────────────┐ │   ║
║  │  normalize_inventory_csv.py (merge)   │  │  │  run_backtest.py         (LGBM)              │ │   ║
║  │  load_dataset_postgres.py  (generic)  │  │  │  run_backtest_catboost.py                    │ │   ║
║  │  spark_dataset_to_iceberg.py          │  │  │  run_backtest_xgboost.py                     │ │   ║
║  │                                       │  │  └──────────────────────────────────────────────┘ │   ║
║  │  Source CSV                           │  │  ┌─ Per-DFU Fitting ────────────────────────────┐ │   ║
║  │     │                                 │  │  │  run_backtest_prophet.py    (multiprocessing)│ │   ║
║  │     ▼                                 │  │  │  run_backtest_neuralprophet.py (PyTorch GPU) │ │   ║
║  │  normalize → clean CSV                │  │  └──────────────────────────────────────────────┘ │   ║
║  │     │                                 │  │  ┌─ Vectorized ────────────────────────────────┐ │   ║
║  │     ├──→ load_dataset_postgres.py     │  │  │  run_backtest_statsforecast.py (~100x faster)│ │   ║
║  │     │     (archive first, then mutate │  │  └──────────────────────────────────────────────┘ │   ║
║  │     │      staging, then exec-lag→main)│  │  ┌─ Deep Learning ─────────────────────────────┐ │   ║
║  │     │                                 │  │  │  run_backtest_patchtst.py   (Transformer)   │ │   ║
║  │     └──→ spark_dataset_to_iceberg.py  │  │  │  run_backtest_deepar.py     (LSTM)          │ │   ║
║  │              ▼                        │  │  └──────────────────────────────────────────────┘ │   ║
║  │         Iceberg (MinIO)               │  │                                                    │   ║
║  └───────────────────────────────────────┘  │  load_backtest_forecasts.py  (bulk load)           │   ║
║                                              │  clean_backtest_models.py   (selective cleanup)    │   ║
║                                              │  clean_forecasts_by_date.py (date-range cleanup)  │   ║
║  ┌──── Clustering ─────────┐  ┌──── Seasonality ───┐  └────────────────────────────────────────┘   ║
║  │ generate_clustering_    │  │ detect_             │  ┌──── Champion Selection ─────────────────┐  ║
║  │  features.py            │  │  seasonality.py     │  │ run_champion_selection.py                │  ║
║  │ train_clustering_       │  │ update_seasonality_ │  │ simulate_champion_strategies.py          │  ║
║  │  model.py               │  │  profiles.py        │  │ train_meta_learner.py                    │  ║
║  └─────────────────────────────────────────┘  ║
║  ┌──── Hyperparameter Tuning (Feature 41) ──┐  ║
║  │ tune_hyperparams.py  (Optuna TPE+Prune)  │  ║
║  │ Walk-forward CV + causal masking         │  ║
║  │ → data/tuning/best_params_<model>.json   │  ║
║  └──────────────────────────────────────────┘  ║
║  │ label_clusters.py       │  └─────────────────────┘  └─────────────────────────────────────────┘  ║
║  │ update_cluster_         │                                                                        ║
║  │  assignments.py         │  ┌──── Config (YAML) ──────────────────────────────────────────────┐   ║
║  │ run_clustering_         │  │ clustering_config.yaml    │ model_competition.yaml               │   ║
║  │  scenario.py            │  │ seasonality_config.yaml   │ hyperparameter_tuning.yaml           │   ║
║  └─────────────────────────┘  └──────────────────────────────────────────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
                                              │
                                              ▼
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      STORAGE LAYER                                                  ║
║                                                                                                     ║
║  ┌────────────────────────── PostgreSQL 16 (pgvector) (:5440) ──────────────────────────────────┐   ║
║  │                                                                                               │   ║
║  │  ┌──── Dimensions ────────────────┐  ┌──── Facts ─────────────────────────────────────────┐  │   ║
║  │  │  dim_item          (SK, CK)    │  │  fact_sales_monthly        (item+cust+loc+month)   │  │   ║
║  │  │  dim_location      (SK, CK)    │  │  fact_external_forecast    (item+loc+fdate+month)  │  │   ║
║  │  │  dim_customer      (SK, CK)    │  │  fact_inventory_snapshot   (item+loc+date, ~190M)  │  │   ║
║  │  │  dim_time          (auto-gen)  │  │  backtest_lag_archive      (fck+model+lag)         │  │   ║
║  │  │  dim_dfu           (SK, CK)    │  └────────────────────────────────────────────────────┘  │   ║
║  │  │   + cluster_assignment         │                                                           │   ║
║  │  │   + 6 seasonality columns      │  ┌──── Materialized Views ────────────────────────────┐  │   ║
║  │  └────────────────────────────────┘  │  agg_sales_monthly         (pre-aggregated KPIs)   │  │   ║
║  │                                       │  agg_forecast_monthly      (pre-aggregated KPIs)   │  │   ║
║  │  ┌──── Job Tables ────────────────┐  │  agg_inventory_monthly     (EOM + daily sales)     │  │   ║
║  │  │  job_history                   │  │  mv_inventory_forecast     (inv-forecast bridge)   │  │   ║
║  │  │  job_schedule                  │  │  mv_top_movers             (dashboard)             │  │   ║
║  │  └────────────────────────────────┘  │  mv_accuracy_*             (accuracy slices)       │  │   ║
║  │                                       └───────────────────────────────────────────────────┘   │   ║
║  │  ┌──── Indexes ───────────────────────────────────────────────────────────────────────────┐  │   ║
║  │  │  B-tree (composite)  │  GIN trigram (pg_trgm full-text)  │  pgvector (embeddings)     │  │   ║
║  │  └────────────────────────────────────────────────────────────────────────────────────────┘  │   ║
║  │                                                                                               │   ║
║  │  21 SQL Migration Files (sql/001 → sql/021)                                                   │   ║
║  └───────────────────────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                                     ║
║  ┌──────────── Lakehouse (Optional) ──────────────────────────────────────────────────────────┐     ║
║  │                                                                                             │     ║
║  │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐                      │     ║
║  │  │  Apache Iceberg   │    │   MinIO (:9200)   │    │  Trino (:8282)   │                      │     ║
║  │  │  REST Catalog     │◄──│  S3 Object Store   │──►│  SQL Query Engine │                      │     ║
║  │  │  (:8381)          │    │  Parquet files     │    │  Iceberg connector│                      │     ║
║  │  └──────────────────┘    └──────────────────┘    └──────────────────┘                      │     ║
║  │                                    ▲                                                        │     ║
║  │                                    │                                                        │     ║
║  │                          ┌──────────────────┐                                               │     ║
║  │                          │ Apache Spark 3.5  │                                               │     ║
║  │                          │ (:7277/:8280)     │                                               │     ║
║  │                          │ ETL to Iceberg    │                                               │     ║
║  │                          └──────────────────┘                                               │     ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────┘     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      BUILD & TOOLING                                                ║
║                                                                                                     ║
║  ┌──── Build ─────────────┐  ┌──── Testing ──────────────┐  ┌──── Infrastructure ────────────────┐  ║
║  │ Makefile (100+ targets)│  │ pytest (backend, ~0.7s)   │  │ Docker Compose (7 services)        │  ║
║  │ uv (Python packaging) │  │ vitest (frontend, ~1.5s)  │  │ Vite dev server + proxy            │  ║
║  │ Vite 5 (frontend)     │  │ 485+ total tests          │  │ MLflow tracking server             │  ║
║  │ TypeScript 5           │  │ httpx ASGI transport      │  │ APScheduler (in-process)           │  ║
║  │ Tailwind CSS           │  │ React Testing Library     │  │                                    │  ║
║  └────────────────────────┘  └───────────────────────────┘  └────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════════════════
                                   DATA FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════════════════════════════

  Source CSVs (8 domains)              ML Models (8 algorithms)
         │                                      │
         ▼                                      ▼
  ┌──────────────┐                    ┌──────────────────┐
  │  Normalize    │                    │  Backtest Scripts │
  │  (ETL clean)  │                    │  LGBM │ CatBoost │
  └──────┬───────┘                    │  XGB  │ Prophet  │
         │                            │  Stats│ Neural   │
         ▼                            │  Patch│ DeepAR   │
  ┌──────────────┐                    └────────┬─────────┘
  │  Clean CSVs   │                             │
  └──────┬───────┘                             ▼
         │                            ┌──────────────────┐
    ┌────┴────┐                       │  Load Forecasts   │
    │         │                       └────────┬─────────┘
    ▼         ▼                                │
┌────────┐ ┌──────────┐                       │
│Postgres│ │ Iceberg   │◄──────────────────────┘
│  (OLTP)│ │(Lakehouse)│
└────┬───┘ └─────┬────┘         ┌──────────────────┐
     │           │               │ Champion Select   │
     │           │               │ (best per DFU)    │
     │           │               └────────┬─────────┘
     │           │                        │
     ▼           ▼                        ▼
┌────────────────────────┐    ┌──────────────────┐
│  FastAPI (:8000)        │◄───│ Job Scheduler    │
│  12 Routers             │    │ (APScheduler)    │
│  Pydantic v2 Validation │    │ Cron/Pipelines   │
└──────────┬─────────────┘    └──────────────────┘
           │
    Vite Proxy (:5173)
           │
           ▼
┌────────────────────────┐
│  React UI               │
│  9 Tabs + Chat          │
│  TanStack Query Cache   │
│  ECharts + Recharts     │
│  Light/Dark Theme       │
└─────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                               ML / FORECASTING PIPELINE
═══════════════════════════════════════════════════════════════════════════════════════════

                        ┌─────────────────────────────────────────┐
                        │           Historical Sales Data          │
                        │         fact_sales_monthly (PG)          │
                        └─────────────────┬───────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
          ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
          │  Clustering      │  │  Seasonality      │  │  Feature Eng.    │
          │  Pipeline        │  │  Detection        │  │  Lag/Rolling     │
          │  KMeans + MLflow │  │  Strength/Profile │  │  Future Masking  │
          └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘
                   │                    │                      │
                   ▼                    ▼                      ▼
          ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐
          │  dim_dfu         │  │  dim_dfu          │  │  8 Backtesting Models        │
          │  cluster_assign  │  │  seasonality_*    │  │                              │
          └─────────────────┘  └──────────────────┘  │  Tree-Based (shared fw):     │
                                                      │    LGBM, CatBoost, XGBoost   │
                                                      │  Per-DFU:                    │
                                                      │    Prophet, NeuralProphet    │
                                                      │  Vectorized:                 │
                                                      │    StatsForecast (ARIMA+ETS) │
                                                      │  Deep Learning:              │
                                                      │    PatchTST, DeepAR          │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌──────────────────────────────┐
                                                      │  fact_external_forecast       │
                                                      │  backtest_lag_archive         │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌──────────────────────────────┐
                                                      │  Champion Selection           │
                                                      │                              │
                                                      │  5 Strategies:               │
                                                      │  ├─ expanding (cumul WAPE)   │
                                                      │  ├─ rolling (last N months)  │
                                                      │  ├─ decay (exp weighting)    │
                                                      │  ├─ ensemble (blend top-K)   │
                                                      │  └─ meta_learner (ML clf)    │
                                                      │                              │
                                                      │  Strict Causality:           │
                                                      │  month T uses data < T only  │
                                                      │                              │
                                                      │  Output: model_id='champion' │
                                                      │  Ceiling: model_id='ceiling' │
                                                      └──────────────────────────────┘
```
