# ML Forecast Pipeline Workflow Plan

> Complete operational reference for the Supply Chain Command Center ML forecast pipeline.
> Covers every stage from raw data to production forecast, including experimentation,
> configuration, error recovery, and gap analysis.

---

## Table of Contents

1. [Glossary](#glossary)
2. [Prerequisites](#prerequisites)
3. [Quick Start / Happy Path](#quick-start--happy-path)
4. [Pipeline Overview](#section-1-pipeline-overview)
5. [Per-Stage Detail](#section-2-per-stage-detail)
   - [Stage 1: Data Loading](#stage-1-data-loading--preparation)
   - [Stage 2: Seasonality & Variability Profiling](#stage-2-seasonality--variability-profiling)
   - [Stage 3: Lead Time & ABC-XYZ Classification](#stage-3-lead-time-profiling--abc-xyz-classification)
   - [Stage 4: Customer Features Generation](#stage-4-customer-features-generation)
   - [Stage 5: Clustering](#stage-5-clustering)
   - [Stage 6: Backtesting](#stage-6-backtesting)
   - [Stage 7: Backtest Loading](#stage-7-backtest-loading)
   - [Stage 8: Hyperparameter Tuning](#stage-8-hyperparameter-tuning)
   - [Stage 9: Champion Selection](#stage-9-champion-selection)
   - [Stage 10: Production Forecast Generation](#stage-10-production-forecast-generation)
6. [Experimentation Workflows](#section-3-experimentation-workflows)
7. [Operational Workflows](#section-4-operational-workflows)
8. [Expert Panel Testing](#section-5-expert-panel-testing)
9. [UI Surfaces Reference](#section-6-ui-surfaces-reference)
10. [Configuration Reference](#section-7-configuration-reference)
11. [Database Reference](#section-8-database-reference)
12. [Product Gaps](#section-9-product-gaps--missing-support)

---

## Glossary

| Term | Definition |
|---|---|
| **DFU** | Demand Forecast Unit -- the atomic forecasting grain: `item_id + location + customer_group`. Each DFU gets its own champion model and production forecast. |
| **WAPE** | Weighted Absolute Percentage Error: `SUM(\|F-A\|) / \|SUM(A)\|`. Primary accuracy metric for model comparison. Lower is better. |
| **Accuracy** | `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`. Higher is better. Complement of WAPE expressed as a percentage. |
| **Bias** | `(SUM(Forecast) / SUM(History)) - 1`. Positive = over-forecast, negative = under-forecast. |
| **Execution Lag** | The number of months between when a forecast is generated and the actual month being forecast. A forecast made in January for March has execution lag 2. Production forecasts use each DFU's configured `execution_lag` from `dim_sku`. |
| **Champion** | The model selected as best-performing for a given DFU based on historical backtest accuracy. Champion assignments are stored with `model_id = 'champion'` and `source_model_id` pointing to the winning algorithm. |
| **Backtest** | Expanding-window historical evaluation: train on data up to a cutoff, predict forward, compare to actuals. Repeated across N timeframes to produce robust accuracy estimates. |
| **Timeframe** | A single cutoff date in a backtest. With `n_timeframes: 10`, the backtest evaluates 10 different historical cutoff points. |
| **Cluster Strategy** | How tree models are trained: `per_cluster` trains a separate model per demand cluster; `global` trains one model across all DFUs. |
| **Meta-Learner** | An ML classifier (Random Forest) that predicts the best-performing model for each DFU based on demand features and historical performance. Used in advanced champion selection strategies. |
| **Promotion** | Moving an experimental configuration (hyperparameters, clusters, champion strategy) into the production pipeline config (`forecast_pipeline_config.yaml`). |
| **Plan Version** | A timestamped production forecast run, tagged as `YYYY-MM`. The system keeps the last N versions for comparison. |
| **Foundation Model** | Pre-trained time-series models (Chronos family) used zero-shot without task-specific training. |
| **Oracle / Ceiling** | The theoretical best accuracy achievable if the perfect model were selected for every DFU at every timeframe. Measures the upper bound of champion selection improvement. |

---

## Prerequisites

Before running any pipeline stage, ensure the following are in place:

### 1. Environment Setup

```bash
# Clone and enter the project
cd /path/to/DemandProject

# Create .env file (copy from template)
cp .env.example .env   # Edit with your Postgres credentials, API keys, etc.

# Install Python dependencies (uses uv)
make init              # Creates .venv, installs uv, syncs deps

# Install frontend dependencies
make ui-init           # Runs npm install in frontend/
```

### 2. Infrastructure

```bash
# Start Postgres + MLflow containers
make up

# Apply database schemas (one-time, or after DDL changes)
make db-apply-sql

# Verify services are running
make health
```

### 3. Input Data

Place raw CSV files in `data/input/`. Required files:
- `item.csv`, `location.csv`, `customer.csv` (dimensions)
- `sku.csv` (DFU dimension: item + location + customer_group)
- `sales.csv`, `forecast.csv` (facts)
- `inventory.csv`, `sourcing.csv`, `purchase_order.csv` (supply chain)
- `customer_demand.csv` (customer-level demand)

### 4. Verify

```bash
make check-all         # DB row counts + API health
```

---

## Quick Start / Happy Path

For a complete from-scratch setup (data + ML + planning + operations):

```bash
make setup-all         # ~4-6 hours, runs everything end-to-end
```

For a step-by-step approach, run these in order:

| Step | Command | Time | What it does |
|---|---|---|---|
| 1 | `make setup-data` | ~30 min | Normalize CSVs + load all 11 domains into Postgres |
| 2 | `make setup-features` | ~30 min | Clustering + seasonality + variability + lead time + ABC-XYZ + demand signals |
| 3 | `make customer-features` | ~10 min | Pre-compute 34 customer-derived features |
| 4 | `make tune-all` | ~1-4 hours | *(Optional)* Bayesian hyperparameter tuning — finds best params before backtesting |
| 5 | `make backtest-all` | ~2-5 hours | Backtest 4 core algorithms (uses tuned params if step 4 ran) |
| 6 | `make backtest-load-all-bulk` | ~20 min | Load predictions into Postgres |
| 7 | `make champion-all` | ~10 min | Train meta-learner + select champions |
| 8 | `make forecast-generate` | ~15-30 min | Generate production forecasts |

After the pipeline completes:

```bash
make api               # Start FastAPI on :8000
make ui                # Start React dev server on :5173
```

Open `http://localhost:5173` to access the UI. The Aggregate Analysis tab shows accuracy KPIs; the Model Tuning tab provides experimentation.

---

## Section 1: Pipeline Overview

### 1.1 Visual Flow Diagram

```
                         ┌──────────────────────────────┐
                         │   DATA LOADING & PREPARATION  │
                         │   (ETL: normalize + load)     │
                         │   make setup-data             │
                         │   ~30 min                     │
                         └──────────────┬───────────────┘
                                        │
          ┌─────────────┬───────────────┼───────────────┬─────────────┐
          ▼             ▼               ▼               ▼             ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌────────────┐
 │ SEASONALITY  │ │ VARIABILITY  │ │ LEAD TIME    │ │ ABC-XYZ     │ │ CUSTOMER   │
 │ DETECTION    │ │ PROFILING    │ │ PROFILING    │ │ CLASSIF.    │ │ FEATURES   │
 │ ~5 min       │ │ ~3 min       │ │ ~5 min       │ │ ~3 min      │ │ ~10 min    │
 └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ └──────┬─────┘
        │                │               │                │              │
        └────────────────┴───────┬───────┴────────────────┘              │
                                 ▼                                       │
                    ┌──────────────────────────────┐                     │
                    │       CLUSTERING             │                     │
                    │  Feature eng -> KMeans train │                     │
                    │  make cluster-all            │                     │
                    │  ~10 min                     │                     │
                    └──────────────┬───────────────┘                     │
                                   │                                     │
                                   ├─────────────────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  HYPERPARAMETER TUNING       │
                    │  (optional, improves params)  │
                    │  make tune-all               │
                    │  ~1-4 hours                  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       BACKTESTING            │
                    │  4 core algorithms x 10 TFs  │
                    │  (uses tuned params if avail) │
                    │  make backtest-all           │
                    │  ~2-5 hours                  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    BACKTEST LOADING          │
                    │  CSV -> Postgres (bulk COPY) │
                    │  make backtest-load-all-bulk │
                    │  ~20 min                     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   CHAMPION SELECTION         │
                    │   Best model per DFU         │
                    │   make champion-all          │
                    │   ~10 min                    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  PRODUCTION FORECAST         │
                    │  GENERATION                  │
                    │  make forecast-generate      │
                    │  ~15-30 min                  │
                    │  Scheduled: 0 6 2 * *        │
                    └──────────────────────────────┘

    Iterative improvement loop:
    tune → backtest → load → champion → evaluate accuracy → retune
```

### 1.2 Stage Dependencies (Formal DAG)

| Stage | Depends On | Blocking? |
|---|---|---|
| 1. Data Loading | (none) | Yes |
| 2. Seasonality & Variability | Data Loading | No (auxiliary, improves clustering) |
| 3. Lead Time & ABC-XYZ | Data Loading | No (auxiliary, informs planning) |
| 4. Customer Features | Data Loading | No (needed only for customer-enriched models) |
| 5. Clustering | Data Loading (+ optionally 2, 3) | Yes (for per-cluster tree models) |
| 6. Hyperparameter Tuning | Clustering | Optional (improves backtest accuracy) |
| 7. Backtesting | Clustering (+ optionally 4, 6) | Yes |
| 8. Backtest Loading | Backtesting | Yes |
| 9. Champion Selection | Backtest Loading | Yes |
| 10. Production Forecast | Champion Selection | Yes |

### 1.3 Estimated End-to-End Runtimes

| Scenario | Time |
|---|---|
| `make setup-all` (complete from scratch) | 4-6 hours |
| `make fresh-all` (truncate + rebuild) | 4-6 hours |
| `make fresh-champion` (data + ML pipeline) | 3-5 hours |
| `make backtest-all-parallel` (4 models concurrent) | 1-2 hours |
| Single model backtest (e.g., `make backtest-lgbm`) | 20-60 min |
| Champion selection only | 5-10 min |
| Production forecast generation | 15-30 min |

---

## Section 2: Per-Stage Detail

---

### STAGE 1: Data Loading & Preparation

**Purpose:** Normalize raw CSV input files into a canonical schema and load them into PostgreSQL. This is the foundation for all downstream ML and planning operations.

**Prerequisites:** Raw CSV files in `data/input/`. PostgreSQL database running (via `make up`).

**Config files:**
- `config/etl/etl_config.yaml` -- domain load order, MV refresh tiers, parallel workers, normalization settings

**Database tables written (11 tables):**

| Table | Purpose |
|---|---|
| `dim_item` | Item dimension: descriptions, hierarchy, attributes |
| `dim_location` | Location dimension: site, region, type |
| `dim_customer` | Customer dimension: customer_no, name, group |
| `dim_time` | Time dimension: auto-generated 2020-2035 calendar |
| `dim_sku` | DFU dimension: item_id + loc + customer_group, ABC class, region |
| `fact_sales_monthly` | Monthly sales actuals (type=1, qty shipped/ordered) |
| `fact_external_forecast_monthly` | External forecasts with execution-lag tracking |
| `fact_customer_demand_monthly` | Customer-level demand (monthly partitioned) |
| `fact_inventory_snapshot` | Inventory positions (monthly partitioned, ~198M rows) |
| `dim_sourcing` | Sourcing relationships (supplier, lead time) |
| `fact_purchase_orders` | Purchase order history |

**CLI commands:**
```bash
make pipeline-full          # Full parallel ETL (normalize all + load all)
make normalize-all          # Just normalize (no load)
make load-all               # Just load (assumes normalized)
make refresh-mvs-tiered     # Refresh materialized views after load
```

Individual domain targets:
```bash
make normalize-item && make load-item
make normalize-sales && make load-sales
make normalize-forecast && make load-forecast
make normalize-customer-demand && make load-customer-demand
```

**Scripts:**
- `scripts/etl/run_pipeline.py` -- orchestrator for parallel normalize + load
- `scripts/normalize_dataset_csv.py` -- unified normalizer (takes `--dataset` flag for item, sales, forecast, sku, etc.)
- `scripts/normalize_inventory_csv.py` -- dedicated inventory normalizer (handles large volumes)
- `scripts/etl/normalize_customer_demand_csv.py` -- customer demand normalizer
- `scripts/etl/load_dataset_postgres.py` -- generic COPY-based bulk loader

**Output artifacts:**
- Normalized CSVs in `data/normalized/`
- Populated dim/fact tables (11 tables)
- Refreshed materialized views (4 tiers)

**Error Recovery:**
- If normalize fails: check CSV format, encoding (UTF-8 expected), column names against `DomainSpec` in `common/core/domain_specs.py`
- If load fails: check Postgres connection (`make health`), verify table DDL applied (`make db-apply-sql`)
- Partial load recovery: re-run individual domain targets (e.g., `make load-sales`)
- `make pipeline-refresh` does incremental reload (detects changes, reloads only deltas)

**Validation:**
```bash
make check-db       # Verify row counts and table health
make check-all      # DB + API health check
```

---

### STAGE 2: Seasonality & Variability Profiling

**Purpose:** Compute per-SKU demand characteristics that inform clustering, model selection, and planning policies. Seasonality detection identifies yearly patterns (peak/trough months, strength). Variability profiling classifies each DFU as low/medium/high/lumpy based on CV and intermittency.

**Prerequisites:** `fact_sales_monthly` and `dim_sku` populated (Stage 1 complete).

**Config files:**
- `config/forecasting/forecast_domain_config.yaml` -- key settings:
  - `seasonality.min_months_history`: 24 -- **Why:** Need at least 2 full seasonal cycles to detect annual patterns
  - `seasonality.thresholds.low/medium/high`: 0.15 / 0.35 / 0.70 -- **When to change:** Adjust if too many/few DFUs are classified as seasonal
  - `seasonality.confirmation.yoy_correlation`: 0.40, `acf_lag12`: 0.30
  - `variability.cv_thresholds.low/medium/high`: 0.30 / 0.80 / 1.50 -- **When to change:** Tune if variability classes are imbalanced
  - `variability.intermittency_threshold.ratio`: 0.30 -- **Why:** DFUs with >30% zero-demand months are flagged as intermittent

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `fact_sales_monthly` | Read | Source sales history |
| `dim_sku` | Read + Write | Reads DFU attributes; writes seasonality and variability columns |

Seasonality columns written to `dim_sku`: `seasonality_profile`, `seasonality_strength`, `is_yearly_seasonal`, `peak_month`, `trough_month`, `peak_trough_ratio`.

Variability columns written to `dim_sku`: `demand_mean`, `demand_std`, `demand_cv`, `demand_mad`, `demand_p50`, `demand_p90`, `demand_skewness`, `demand_kurtosis`, `zero_demand_months`, `total_demand_months`, `intermittency_ratio`, `variability_class`, `demand_profile_ts`.

**CLI commands:**
```bash
make features-compute       # Unified SKU feature pipeline (seasonality + variability + lifecycle)

# Legacy aliases (all delegate to features-compute)
make seasonality-all        # alias → features-compute
make seasonality-detect     # alias → features-compute
make variability-all        # alias → features-compute
make variability-compute    # alias → features-compute
```

**UI actions:**
- Settings tab > "Forecast Domain" config card shows seasonality and variability thresholds
- Aggregate Analysis tab (`AggregateAnalysisTab.tsx`) has a `seasonality_profile` filter in the accuracy slice endpoint

**Scripts:**
- `scripts/ml/compute_sku_features.py` -- unified pipeline: computes volume, trend, seasonality (CV of monthly means, YoY correlation, ACF lag-12, peak/trough), variability (CV/MAD/skewness/kurtosis), intermittency, and lifecycle features in a single pass; writes all columns to `dim_sku`
- ~~`scripts/detect_seasonality.py`~~ -- **deleted**; use `compute_sku_features.py` (`make features-compute`) instead
- ~~`scripts/update_seasonality_profiles.py`~~ -- **deleted**; handled by `compute_sku_features.py`
- ~~`scripts/compute_demand_variability.py`~~ -- **deleted**; use `compute_sku_features.py` instead
- ~~`scripts/detect_drift.py`~~ -- **deleted**; drift signals are now derived from `dim_sku` features and the SKU feature pipeline

**Error Recovery:**
- If seasonality detection fails: check `min_months_history` threshold -- DFUs with insufficient history are skipped (not an error)
- If variability compute hangs: likely large dataset; add `--batch-size` or run subsets
- Re-run is idempotent (upserts to `dim_sku`)

**Validation:**
```sql
SELECT seasonality_profile, COUNT(*) FROM dim_sku GROUP BY 1;
SELECT variability_class, COUNT(*) FROM dim_sku GROUP BY 1;
```

---

### STAGE 3: Lead Time Profiling & ABC-XYZ Classification

**Purpose:** Compute lead time variability profiles per item-location and classify DFUs using ABC-XYZ methodology. Lead time profiles inform safety stock calculations. ABC-XYZ classification drives inventory policies.

**Prerequisites:** `dim_sku`, `dim_sourcing`, `fact_sales_monthly` populated (Stage 1 complete).

**Config files:**
- `config/inventory/inventory_planning_config.yaml` -- lead time profiling thresholds
- ABC-XYZ settings in `config/inventory/inventory_planning_config.yaml` -- ABC thresholds (revenue-based), XYZ thresholds (CV-based)

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `dim_item_lead_time_profile` | Write | Lead time mean, std, CV per item-location |
| `dim_sku` | Read + Write | Reads demand history; writes ABC/XYZ classification columns |
| `fact_sales_monthly` | Read | Revenue and volume for ABC classification |
| `dim_sourcing` | Read | Sourcing lead times for LT profiling |

**CLI commands:**
```bash
# Lead Time Profiling
make lt-profile-all         # schema + compute (full pipeline)
make lt-profile-schema      # Apply DDL only
make lt-profile-compute     # Compute LT variability profiles

# ABC-XYZ Classification
make abc-xyz-all            # schema + classify (full pipeline)
make abc-xyz-schema         # Apply DDL only
make abc-xyz-classify       # Run classification
make abc-xyz-classify-dry   # Dry run (no DB writes)
```

**Error Recovery:**
- Lead time profiling is safe to re-run (upserts)
- ABC-XYZ dry run (`make abc-xyz-classify-dry`) previews changes before committing
- If sourcing data is missing, lead time profiles will have NULLs (graceful degradation)

---

### STAGE 4: Customer Features Generation


**Prerequisites:** `fact_customer_demand_monthly` populated (Stage 1 complete).

**CLI commands:**
```bash
make customer-features          # SQL-based generation (recommended, faster)
make customer-features-python   # Python-based alternative
```

**Scripts:**
- `scripts/ml/generate_customer_features_sql.py` -- generates features via SQL aggregation
- `scripts/ml/generate_customer_features.py` -- Python-based alternative

**Error Recovery:**
- Re-run is idempotent (features table is recreated)
- If `fact_customer_demand_monthly` is empty, features will be empty (customer-enriched models fall back gracefully)

---

### STAGE 5: Clustering


**Prerequisites:** `fact_sales_monthly` and `dim_sku` populated. Seasonality/variability profiling is recommended but not strictly required.

**Config files:**
- `config/forecasting/forecast_pipeline_config.yaml` -- `clustering` section:
  - `enabled`: true -- **Master switch.** When `false`, all backtests fall back to `global` strategy regardless of per-algorithm settings.
  - `steps`: generate_features, train_model, label_clusters, update_db
- Clustering params are stored in the `cluster_experiment` table (promoted row):
  - `k_range`: [9, 18] -- **When to change:** Increase max if you have >50K DFUs with diverse demand patterns; decrease if <5K DFUs
  - `min_cluster_size_pct`: 2.0 -- **Why:** Prevents tiny clusters that overfit
  - `min_months_history`: 12
  - `time_window_months`: 36

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `dim_sku` | Read | SKU attributes and computed SKU features |
| `sku_cluster_assignment` | Write | Durable promoted ML cluster labels keyed by `(experiment_id, sku_ck)` |
| `current_sku_cluster_assignment` | Read | Current promoted `ml_cluster` labels for downstream pipelines |
| `fact_sales_monthly` | Read | Source time series for feature engineering |
| `cluster_experiment` | Write | Experiment lifecycle tracking (UI experiments) |

**CLI commands:**
```bash
make cluster-all       # Full pipeline: features -> train -> label -> update DB
# Note: the legacy step-1 script `scripts/ml/generate_clustering_features.py`
# has been deleted. SKU features (volume, trend, seasonality, periodicity,
# intermittency, lifecycle) are now produced by `make features-compute`
# (scripts/ml/compute_sku_features.py) and read from `dim_sku` by the
# clustering pipeline.
make features-compute  # Step 1: unified SKU feature computation (replaces generate_clustering_features.py)
make cluster-train     # Step 2: KMeans training (via run_cluster_pipeline.py)
# Step 3: cluster labeling is now inline in run_clustering_scenario.py
#         (assign_cluster_labels from common.ml.clustering.labeling) — the
#         standalone label_clusters.py script was removed.
# (cluster-update absorbed into unified pipeline via promote_scenario)
```

**UI actions (Cluster Experiments):**
- Model Tuning Tab > **Clustering** stage card > `ClusterExperimentsPanel`
- "New Experiment" opens builder with 7 templates: Production Baseline, High-K Granular, Low-K Broad, Seasonal Focus, Intermittent Specialist, PCA Compressed, Recent Data Focus
- Completed experiments can be **compared** (migration matrix, quality metrics, profile comparison)
- A completed experiment can be **promoted** to production (upserts `sku_cluster_assignment`;
  downstream reads use `current_sku_cluster_assignment`)

API endpoints (prefix `/cluster-experiments`):
- `POST /` -- create + launch experiment (combined, returns 201)
- `GET /` -- list experiments
- `GET /templates` -- list available templates
- `GET /completed` -- list completed experiments
- `GET /compare` -- pairwise comparison (query params: `baseline_id`, `candidate_id`)
- `GET /{id}` -- experiment detail
- `PATCH /{id}` -- update experiment metadata
- `DELETE /{id}` -- delete experiment
- `POST /{id}/promote` -- promote to production
- `GET /{id}/used-by` -- which tuning experiments reference this clustering

**Scripts:**
- ~~`scripts/ml/generate_clustering_features.py`~~ -- **deleted**; the 6 feature dimensions (volume, trend, seasonality, periodicity, intermittency, lifecycle) are now produced by `scripts/ml/compute_sku_features.py` (`make features-compute`) and read from `dim_sku`
- ~~`scripts/ml/train_clustering_model.py`~~ -- **deleted**; KMeans training is now driven by `scripts/ml/run_cluster_pipeline.py`
- `scripts/ml/run_cluster_pipeline.py` -- unified pipeline: features -> train -> label -> promote (KMeans with silhouette + Calinski-Harabasz combined scoring)
- ~~`scripts/ml/label_clusters.py`~~ -- **deleted**; hierarchical labeling taxonomy (e.g., `high_volume_seasonal_growing`) is now inline via `assign_cluster_labels` in `common/ml/clustering/labeling.py`, called by `run_clustering_scenario.py`
- Cluster assignment updates are handled by `promote_scenario()` in the unified pipeline

**Output artifacts:**
- `data/staged/clustering_features.csv`, `data/clustering/kmeans_model.pkl`, `data/clustering/cluster_labels.csv`
- `data/clustering/centroids.csv`, `data/clustering/cluster_summary.json`, `data/clustering/k_selection_plot.png`

**Error Recovery:**
- If clustering fails at train step: check `k_selection_plot.png` -- if silhouette scores are flat, data may lack structure; try reducing `k_range`
- If a cluster is below minimum size: the script auto-merges small clusters. Check logs for merge warnings.
- Re-run any individual step without re-running the full pipeline
- After promotion, downstream backtests must be re-run with new clusters

**Validation:**
```sql
SELECT ml_cluster, COUNT(*)
FROM current_sku_cluster_assignment
WHERE ml_cluster IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC;
```
Verify no cluster is below the minimum size threshold (2% of total).

---

### STAGE 6: Backtesting

**Purpose:** Evaluate every enabled algorithm using expanding-window historical backtesting. For each of N timeframes, train on all data up to a cutoff, predict the next 6 months, and compare to actuals. This produces the accuracy evidence used for champion selection.

**Prerequisites:** Clustering complete (for per-cluster tree models). `fact_sales_monthly` populated.

**Config files:**
- `config/forecasting/forecast_pipeline_config.yaml` -- `backtest` section:
  - `n_timeframes`: 10 -- **When to change:** Reduce to 5 for faster iteration during experimentation; use 10+ for production evaluation
  - `embargo_months`: 0 -- **Why:** Prediction already starts after training ends; zero preserves lag-0 evidence
  - `forecast_horizon`: 6
  - `early_stop_pct`: 0.03 -- **Why:** 3% patience for early stopping prevents overfitting
- `config/forecasting/forecast_pipeline_config.yaml` -- `algorithms` section (10 algorithms):
  - **Foundation models (1):** `chronos2_enriched` -- the only Chronos variant remaining; T5, Bolt,
  - **Deep learning models (2):** `nbeats`, `nhits`
  - Each algorithm has lifecycle flags: `enabled`, `tune`, `backtest`, `compete`, `forecast`, `expert`
- `config/forecasting/forecast_pipeline_config.yaml` -- `backtest_sampling` section:
  - `enabled`: true, `default_target_n`: 5000 -- **When to change:** Set to `false` or increase `default_target_n` for production-quality champion selection; keep sampled for experimentation speed
  - `default_method`: proportional, `min_per_cluster`: 10

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `dim_sku` | Read | Cluster assignments, execution_lag, attributes |
| `fact_sales_monthly` | Read | Historical actuals for train/test |

**CLI commands:**

The `backtest-all` target runs every `compete: true` model that trains cleanly on a rebuild -- the 4
operator-gated baselines) require separate targets.

```bash
# Core 4 (run by backtest-all)
make backtest-lgbm             # LightGBM (--parallel --workers 8, defaults to lgbm)
make backtest-chronos2e        # Chronos 2 Enriched (~6h) -- the only remaining foundation model

# Additional 6 (separate targets, cheap/operator-gated)
make backtest-mstl             # MSTL statistical
make backtest-nhits            # N-HiTS deep learning
make backtest-nbeats           # N-BEATS deep learning

# Composite targets
make backtest-all              # 4 core algorithms sequentially
make backtest-all-parallel     # 4 core algorithms concurrently (logs in data/backtest/logs/)

# Convenience targets (backtest + load combined)
make backtest-chronos2e-full   # Chronos 2 Enriched backtest + load
make backtest-mstl-full        # MSTL backtest + load
make backtest-nhits-full       # N-HiTS backtest + load
make backtest-nbeats-full      # N-BEATS backtest + load
```

**Scripts:**
  when no `--model` flag. Per-cluster training with SHAP feature selection, Tweedie objective for
  intermittent clusters, recursive multi-step prediction.
- `scripts/ml/run_backtest_chronos2_enriched.py` -- the sole remaining foundation-model backtest (Chronos 2 Enriched)
- `scripts/ml/run_backtest_dl.py` -- deep learning backtest (N-HiTS, N-BEATS) via NeuralForecast
- `scripts/ml/run_backtest_mstl.py` -- MSTL statistical backtest

**Output artifacts (per algorithm):**
- `data/backtest/<model_id>/backtest_predictions.csv` -- execution-lag rows for DB load
- `data/backtest/<model_id>/backtest_predictions_all_lags.csv` -- lag 0-4 archive
- `data/backtest/<model_id>/backtest_metadata.json` -- accuracy stats, timing
- `data/backtest/<model_id>/shap_values/` -- SHAP feature importance (tree models only)
- `data/models/<model_id>/<cluster>/model.pkl` -- trained model artifacts (tree models)

**Error Recovery:**
- If a backtest fails mid-run: check `data/backtest/<model_id>/` for partial output. Re-running overwrites previous output.
- If CUDA/GPU errors: set `DEMAND_GPU=off` to force CPU mode
- Memory errors on large datasets: reduce `--workers` count, or enable backtest sampling
- Individual algorithm failures do not block other algorithms

**Validation:**
```bash
# Check metadata.json for accuracy_pct, wape, n_predictions, n_dfus
cat data/backtest/lgbm_cluster/backtest_metadata.json | python -m json.tool
```

---

### STAGE 7: Backtest Loading

**Purpose:** Load backtest prediction CSVs into PostgreSQL for accuracy analysis and champion selection. Loads execution-lag rows into the main forecast table and all-lag rows (0-4) into the archive table.

**Prerequisites:** At least one backtest completed (CSVs exist in `data/backtest/<model_id>/`).

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `fact_external_forecast_monthly` | Write (COPY + upsert) | Main forecast table: execution-lag predictions |
| `backtest_lag_archive` | Write (COPY + upsert) | All-lag archive: lag 0-4 predictions with timeframe labels |

**CLI commands:**
```bash
make backtest-load-all-bulk    # Load all models (bulk: drop indexes -> COPY -> recreate)
make backtest-load-all         # Load all models (standard, per-model)
make backtest-load MODEL=lgbm_cluster  # Single model
make backtest-load-bulk        # 4 core models with single index cycle
make backtest-load-main-only MODELS="lgbm_cluster chronos2_enriched"  # Main table only
make backtest-load-archive-only MODELS="lgbm_cluster"       # Archive only

# Per-algorithm load targets (tree models load via `make backtest-load MODEL=<id>`;
# these dedicated targets exist for the non-tree baselines)
make backtest-load-chronos2e
make backtest-load-mstl
make backtest-load-nhits
make backtest-load-nbeats

# After loading, refresh accuracy materialized views
make refresh-accuracy-mvs
make accuracy-slice-refresh
```

**Scripts:**
- `scripts/etl/load_backtest_forecasts.py` -- COPY-based bulk loader. Supports `--model`, `--all`, `--replace`, `--bulk`, `--main-only`, `--archive-only`. Drops secondary indexes for fast bulk insert, recreates after. Dual-path insert: archive loaded BEFORE staging mutation.

**Error Recovery:**
- If bulk load fails mid-way: indexes may be dropped. Re-run `make backtest-load-all-bulk` -- it recreates indexes at the end.
- For single-model reload: `make backtest-load MODEL=lgbm_cluster` replaces only that model's rows
- After loading, always run `make refresh-accuracy-mvs` to update accuracy MVs

**Validation:**
```sql
SELECT model_id, COUNT(*) FROM fact_external_forecast_monthly GROUP BY 1 ORDER BY 2 DESC;
SELECT model_id, COUNT(*) FROM backtest_lag_archive GROUP BY 1 ORDER BY 2 DESC;
```

---

### STAGE 8: Hyperparameter Tuning

**Purpose:** Optimize tree model hyperparameters using Bayesian search (Optuna) or strategy-based grid search. Produces `best_params_<model>.json` files that can be promoted into `forecast_pipeline_config.yaml`.

**Prerequisites:** Backtest data loaded (used for walk-forward CV splits). Clustering complete.

**Config files:**
- `config/forecasting/hyperparameter_tuning.yaml`:
  - `n_trials`: 50 -- **When to change:** Increase to 100-200 for thorough search; reduce to 20 for quick exploration
  - `n_splits`: 5, `gap_months`: 1, `val_months_per_fold`: 3, `min_train_months`: 13
  - Per-model search spaces with `type`, `low`/`high` bounds, optional `log` flag
- `config/forecasting/tuning_templates.yaml` -- UI experiment templates (production_baseline + 4 expert templates per model)

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `fact_sales_monthly` | Read | Training data for CV |
| `dim_sku` | Read | Cluster assignments, features |
| `lgbm_tuning_run` | Write | Run-level summary (used for all tree models despite name) |
| `lgbm_tuning_timeframe` | Write | Per-timeframe breakdown |
| `lgbm_tuning_cluster` | Write | Per-cluster accuracy |
| `lgbm_tuning_month` | Write | Per-month accuracy |
| `lgbm_tuning_comparison` | Write | Pairwise comparisons between runs |
| `tuning_promotion_log` | Write | Audit trail for parameter promotions |

**CLI commands:**
```bash
# Bayesian tuning (Optuna)
make tune-lgbm              # 50 trials, walk-forward CV
make tune-all               # All 3 sequentially
make tune-cust-enriched-all # All 3 customer-enriched models

# Auto-tune (strategy grid)
make lgbm-auto-tune RUNS=5              # Run first 5 LGBM strategies
make lgbm-auto-tune-promote RUNS=10    # Run all + promote best to config
make lgbm-auto-tune-dry-run RUNS=10    # Dry run (print strategies)
```

**UI actions (Model Tuning Experiments):**
- View experiment **leaderboard** (table of all runs sorted by accuracy)
- **"New Experiment"** -> `ExperimentBuilder` dialog with template selection and hyperparameter adjustment
- Select 2 experiments -> **"Compare"** -> `EnhancedComparisonPanel`
- Select winning experiment -> **"Promote"** -> writes params to `forecast_pipeline_config.yaml`

API endpoints (prefix `/model-tuning/{model}`):
- `POST /{model}/experiments` -- create + launch experiment (combined, returns 201)
- `GET /{model}/experiments` -- list experiments
- `GET /{model}/experiments/{id}` -- experiment detail
- `GET /{model}/experiments/{id}/lags` -- per-lag breakdown
- `GET /{model}/experiments/{id}/clusters` -- per-cluster breakdown
- `GET /{model}/experiments/{id}/months` -- per-month breakdown
- `GET /{model}/experiments/{id}/logs` -- stream logs
- `GET /{model}/compare` -- compare two experiments (query params: `baseline_id`, `candidate_id`)
- `POST /{model}/experiments/{id}/promote` -- promote to config
- `POST /{model}/experiments/{id}/promote-results` -- promote results
- `GET /{model}/experiments/{id}/promote-results/status` -- promotion status
- `POST /{model}/experiments/{id}/cancel` -- cancel running experiment
- `DELETE /{model}/experiments/{id}` -- delete experiment
- `GET /{model}/templates` -- list templates
- `GET /{model}/promoted` -- get currently promoted config
- `GET /{model}/promotions` -- promotion history

**Scripts:**
- `scripts/tune_hyperparams.py` -- Optuna Bayesian optimization with walk-forward CV
- `scripts/ml/auto_tune.py` -- strategy-based grid search from `tune_strategies.yaml`
- `scripts/tune_cluster_hyperparams.py` -- Per-cluster Bayesian tuning (runs Optuna independently per `ml_cluster`)

**Per-cluster tuning CLI:**
```bash
make tune-lgbm-clusters       # Per-cluster LGBM tuning
make tune-clusters            # Per-cluster tuning for all tree models
```

Per-cluster tuning writes cluster-specific overrides to `config/forecasting/cluster_tuning_profiles.yaml` with `cluster_name` in `match_criteria`. During backtest, `resolve_cluster_params()` matches profiles: Phase 1 = exact `cluster_name` match, Phase 2 = statistical criteria fallback (mean_demand, cv_demand, zero_demand_pct, etc.).

**Error Recovery:**
- Optuna studies are resumable: `data/tuning/optuna_<model>.db` (SQLite). Re-running continues from where it stopped.
- If a tuning experiment fails: check logs via `GET /{model}/experiments/{id}/logs`
- Promotion is reversible: the previous config is preserved in `tuning_promotion_log`

**Validation:**
- Compare experiment accuracy vs. production baseline. `improvement_threshold_bps: 5` (0.05% improvement required for "improved" verdict).

---

### STAGE 9: Champion Selection

**Purpose:** For each DFU at each month, select the best-performing model using a configurable strategy (rolling WAPE, ensemble, meta-learner, etc.). The selected model's forecast rows are written with `model_id = 'champion'`.

**Prerequisites:** Backtest predictions loaded for all competing models.

**Config files:**
- `config/forecasting/forecast_pipeline_config.yaml` -- `champion` section:
  - `strategy`: rolling -- **When to change:** Try `ensemble_top3_inverse` or `meta_learner_rf` if single-model selection leaves accuracy on the table vs. oracle ceiling
  - `strategy_params.window_months`: 6
  - `metric`: wape, `lag`: execution
- `config/forecasting/champion_experiment_templates.yaml` -- **36 strategy templates** organized by category:
  - Core: expanding, rolling_6m, rolling_3m, decay_090/095
  - Ensemble: ensemble_top3_inverse, ensemble_top2_equal, adaptive_ensemble, ensemble_rolling_6m, cascade_ensemble, diverse_ensemble
  - Learning: meta_learner_rf, learned_blend_ridge, ridge_blend, shrinkage_blend
  - Segment: per_segment_sba, per_cluster, seasonal_quarter
  - Advanced: hybrid_meta_router, uncertainty_aware, dynamic_window, regime_adaptive, bayesian_model_avg, error_correcting
  - Bandit/RL: thompson_sampling, thompson_ensemble, linucb, exp3
  - Meta: dfu_strategy_router, stacked_strategies, cluster_regime_hybrid, hybrid_warmup

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `fact_external_forecast_monthly` | Read + Write | Read per-model predictions; write champion rows |
| `backtest_lag_archive` | Read | All-lag data for cross-horizon analysis |
| `dim_sku` | Read | DFU features for meta-learner, cluster info |
| `champion_experiment` | Write | Experiment lifecycle tracking |
| `champion_experiment_lag` | Write | Per-execution-lag breakdown |
| `champion_experiment_month` | Write | Per-month breakdown |
| `champion_experiment_comparison` | Write | Pairwise experiment comparison |
| `champion_promotion_log` | Write | Audit trail for promotions |

**CLI commands:**
```bash
make champion-all            # train-meta + simulate + select (full pipeline)
make champion-select         # Run champion selection only
make champion-simulate       # Simulate all strategies (diagnostic, no DB write)
make champion-train-meta     # Train meta-learner classifier
make seed-baselines-champion # Seed production baseline experiment row
```

**UI actions (Champion Experiments):**
- Model Tuning Tab > **Champion** stage card > `ChampionExperimentsPanel`
- "New Experiment" -> select from 36 templates, toggle competing models, configure params
- Compare 2 experiments -> per-lag, per-month, model distribution comparison
- **"Promote Config"** (stage 1) -> writes strategy to `forecast_pipeline_config.yaml`
- **"Load Results"** (stage 2) -> copies champion rows into `fact_external_forecast_monthly`
- `ChampionConfigPanel.tsx` (in Jobs tab): quick config for production competition

API endpoints (prefix `/champion-experiments`):
- `POST /` -- create + launch experiment (combined, returns 202)
- `GET /` -- list experiments
- `GET /templates` -- list templates
- `GET /promoted` -- get currently promoted experiment
- `GET /promotions` -- promotion history
- `GET /compare` -- compare two experiments (query params: `baseline_id`, `candidate_id`)
- `GET /{id}` -- experiment detail with breakdowns
- `GET /{id}/lags` -- per-lag breakdown
- `GET /{id}/months` -- per-month breakdown
- `GET /{id}/logs` -- stream logs
- `POST /{id}/promote` -- promote config (stage 1)
- `POST /{id}/promote-results` -- load champion rows (stage 2)
- `GET /{id}/promote-results/status` -- check promotion status
- `POST /{id}/cancel` -- cancel running experiment
- `DELETE /{id}` -- delete experiment

**MV refresh after champion run:** `api/routers/forecasting/competition.py` no longer
runs `REFRESH MATERIALIZED VIEW` synchronously inside the request handler; instead
it enqueues the `refresh_forecast_views` job via `common/services/job_registry.py`.
This unblocks the request and centralizes refresh ordering through the job
runner so concurrent champion runs serialize cleanly.

**Scripts:**
- `scripts/run_champion_selection.py` -- main champion selection pipeline
- `scripts/run_champion_experiment.py` -- experiment runner (called as subprocess)
- `scripts/simulate_champion_strategies.py` -- diagnostic tool for all strategies
- `scripts/train_meta_learner.py` -- trains Random Forest classifier

**Error Recovery:**
- If champion selection produces poor results: check model distribution. If one model dominates, try ensemble strategy.
- If meta-learner training fails: check that sufficient backtest data exists (needs multi-model coverage)
- Champion rows can be re-generated by re-running `make champion-select` (idempotent upsert)
- To revert a promotion: check `champion_promotion_log` for previous config, manually restore

**Validation:**
```sql
-- Check model distribution
SELECT source_model_id, COUNT(*) FROM fact_external_forecast_monthly
WHERE model_id = 'champion' GROUP BY 1 ORDER BY 2 DESC;
```
Compare champion accuracy to ceiling (oracle): gap_bps should be <100 bps.

---

### STAGE 10: Production Forecast Generation

**Purpose:** Generate forward-looking demand forecasts for future months using champion model assignments. Predictions flow into safety stock, replenishment, S&OP, and financial plans.

**Prerequisites:** Champion selection complete. Trained model artifacts exist in `data/models/`.

**Config files:**
- `config/forecasting/forecast_pipeline_config.yaml` -- `production_forecast` section:
  - `horizon_months`: 24 -- **When to change:** Reduce to 12 if only short-term planning needed; increase to 36 for long-range S&OP
  - `min_history_months`: 12 -- **Why:** DFUs below this threshold use cold-start model instead of champion
  - `cold_start_min_months`: 3 -- **Why:** Absolute floor; DFUs below this are skipped entirely (not enough data for any model)
  - `confidence_interval.z_lower/z_upper`: 1.282 (80% CI)
  - `scheduler.cron`: `0 6 2 * *` (2nd of every month at 6am)

**Database tables:**

| Table | Read/Write | Purpose |
|---|---|---|
| `fact_external_forecast_monthly` | Read | Champion assignments |
| `dim_sku` | Read | DFU attributes, cluster assignments |
| `fact_sales_monthly` | Read | Historical actuals for inference features |
| `fact_production_forecast` | Write | Future-month predictions |

**CLI commands:**
```bash
make forecast-generate         # Generate for all DFUs
make forecast-generate-dfu ITEM=100320 LOC=1401-BULK  # Single DFU
make forecast-generate-dry     # Dry run (no DB write)
make forecast-prod-all         # Schema + generate
```

**UI actions:**
- **Jobs tab > PipelineBuilderPanel** can trigger forecast generation as part of a pipeline sequence
- Aggregate Analysis tab shows production forecast trend charts and accuracy KPIs
- Settings tab shows production forecast parameters (horizon, min history, cold-start model)

**Scripts:**
- `scripts/forecasting/generate_production_forecasts.py` -- loads champion assignments, loads model artifacts, builds inference grid, generates predictions recursively, computes confidence intervals, writes to `fact_production_forecast`

**Error Recovery:**
- If forecast generation fails for some DFUs: check logs for specific model loading errors. Individual DFU failures are logged but do not block other DFUs.
- Dry run first: `make forecast-generate-dry` previews without writing to DB
- Plan versions are immutable once written; re-running creates a new version

**Validation:**
```sql
SELECT plan_version, COUNT(*), COUNT(DISTINCT item_id || loc) AS n_dfus
FROM fact_production_forecast GROUP BY 1;

SELECT horizon_months, COUNT(*)
FROM fact_production_forecast WHERE plan_version = '2026-04' GROUP BY 1 ORDER BY 1;
```

---

## Section 3: Experimentation Workflows

### 3.1 Clustering Experiments (UI Flow)

1. Navigate to **Model Tuning Tab** > **Clustering** stage card
2. View existing experiments in the `ClusterExperimentsPanel` table
3. Click **"New Experiment"**
4. Select a template or customize parameters (k_range, feature engineering, PCA, etc.)
5. Enter a label and optional notes
6. **Submit** -> creates experiment + launches background job (single API call: `POST /cluster-experiments/`)
7. Monitor status: queued -> running -> completed/failed
8. View results: optimal_k, silhouette_score, cluster_sizes, profiles
9. **Compare** two experiments: select two rows -> migration matrix, quality comparison
10. **Promote** winning experiment -> upserts `sku_cluster_assignment`
11. Downstream: re-run backtests and champion selection with new clusters

### 3.2 Tuning Experiments (UI Flow)

2. View the experiment leaderboard with accuracy_pct, wape, bias, status
3. Click **"New Experiment"**
4. In `ExperimentBuilder`: select template, adjust hyperparameters, configure training settings
5. **Submit** -> creates experiment + launches backtest (single API call: `POST /model-tuning/{model}/experiments`)
6. Monitor: running -> completed/failed
7. View results: accuracy, wape, bias, per-timeframe breakdown, per-cluster breakdown
8. **Compare** two experiments -> delta accuracy, per-timeframe deltas, verdict
9. **Promote** -> writes params to `forecast_pipeline_config.yaml`
10. After promotion: re-run backtest to apply new params across all timeframes

### 3.3 Champion Experiments (UI Flow)

1. Navigate to **Model Tuning Tab** > **Champion** stage card
2. View experiment list with champion_accuracy, ceiling_accuracy, gap_bps, model_distribution
3. Click **"New Experiment"**
4. Select template from 36 strategies, toggle competing models, configure strategy params
5. **Submit** -> creates experiment + launches (single API call: `POST /champion-experiments/`, returns 202)
6. Monitor: queued -> running -> completed
7. View results: champion accuracy, ceiling, gap to oracle (bps), model distribution
8. **Compare** two experiments -> per-lag delta, model distribution diff
9. **"Promote Config"** (stage 1) -> writes strategy to pipeline config
10. **"Load Results"** (stage 2) -> copies champion rows into forecast table

### 3.4 Promotion Workflow (Experiment to Production)

**Clustering promotion:**
1. Experiment completes -> user clicks "Promote" -> `sku_cluster_assignment` upserted
2. Downstream: re-run backtests with new clusters, re-run champion selection

**Tuning promotion:**
1. Experiment completes -> user clicks "Promote" in `EnhancedPromoteModal`
2. API writes params to `config/forecasting/forecast_pipeline_config.yaml`
3. Downstream: re-run backtest for that algorithm, reload, re-run champion

**Champion promotion (two stages):**
1. **Promote Config** (stage 1): writes strategy + params to config. Logs to `champion_promotion_log`.
2. **Promote Results** (stage 2): runs champion selection with promoted strategy, writes champion rows.

### 3.5 Post-Promotion Validation

After any promotion, validate that end-to-end accuracy has not regressed:

1. Re-run the affected backtest(s) with promoted config
2. Reload predictions: `make backtest-load MODEL=<model_id>`
3. Refresh accuracy MVs: `make refresh-accuracy-mvs`
4. Compare accuracy in Aggregate Analysis tab against pre-promotion baseline
5. If regression detected: revert config from promotion log and re-run

---

## Section 4: Operational Workflows

### 4.1 Data Refresh (Incremental)

When new monthly data arrives:

```bash
# Detect changes and reload only modified domains
make pipeline-refresh

# Or reload specific domain
make load-forecast-replace     # Reload external forecast only
make load-customer-demand-month MONTH=2026-03  # Single partition

# Refresh downstream views
make refresh-mvs-tiered
make refresh-accuracy-mvs
```

### 4.2 Full Pipeline Re-run

After major data corrections or schema changes:

```bash
make fresh-all      # Truncate everything + full rebuild (4-6 hours)
# OR step-by-step:
make fresh-load     # Data reload only
make fresh-features # Data + features
make fresh-backtest # Data + features + backtests
make fresh-champion # Full ML pipeline
```

### 4.3 Rollback After Bad Promotion

**Tuning rollback:**
1. Query `tuning_promotion_log` for previous parameters
2. Manually edit `config/forecasting/forecast_pipeline_config.yaml` to restore old params
3. Re-run backtest for affected model: `make backtest-lgbm && make backtest-load-lgbm`
4. Re-run champion selection: `make champion-select`

**Champion rollback:**
1. Query `champion_promotion_log` for previous strategy
2. Restore strategy in `config/forecasting/forecast_pipeline_config.yaml` champion section
3. Re-run: `make champion-select`

**Clustering rollback:**
- Use a previous cluster experiment's data to restore `sku_cluster_assignment`
- Or re-run `make cluster-all` with original config

### 4.4 Adding a New Algorithm

1. Register the algorithm in `config/forecasting/forecast_pipeline_config.yaml` under `algorithms`
2. Set lifecycle flags: `enabled`, `tune`, `backtest`, `compete`, `forecast`, `expert`
3. If tree model: add `params` section and entry in `config/forecasting/forecast_pipeline_config.yaml`
4. Create backtest script or add model to existing registry (`common/ml/model_registry.py`)
5. Add Makefile targets: `backtest-<name>`, `backtest-load-<name>`
6. Run backtest and load: `make backtest-<name> && make backtest-load-<name>`
7. Refresh accuracy MVs: `make refresh-accuracy-mvs`
8. Re-run champion selection to include new model: `make champion-select`

### 4.5 Exporting Results

```bash
# Export production forecasts
make sql-runner  # Use the SQL Runner tab in the UI to query and export

# Direct SQL export
psql -h localhost -p 5440 -U demand -d demand_mvp \
  -c "COPY (SELECT * FROM fact_production_forecast WHERE plan_version = '2026-04') TO STDOUT CSV HEADER" \
  > forecast_export.csv
```

The SQL Runner tab in the UI (`/sql-runner`) supports ad-hoc queries with CSV export.

### 4.6 Troubleshooting Quick Reference

| Problem | Cause | Fix |
|---------|-------|-----|
| Champion experiments fail immediately | No backtest data loaded | Run `make backtest-load-all-bulk` first |
| "Repository not found" for Chronos 2 Enriched | HuggingFace auth issue | Run `huggingface-cli login` |
| Backtest accuracy is 0% | No sales data for the period | Check `make health`; verify `fact_sales_monthly` has data |
| Production forecast all NaN | No trained `.pkl` models | Run backtest + `make train-production-all` first - tree models save `.pkl` artifacts during production training, not during backtest |
| Tuning converges in < 10 trials | Search space too narrow | Widen ranges in `config/forecasting/hyperparameter_tuning.yaml` |
| Cold-start items get no forecast | < 3 months of history | Lower `cold_start_min_months` in `production_forecast` config (min 1) |
| Clustering produces 1 cluster | Insufficient data diversity | Lower `min_months_history` or increase sample size in the clustering experiment config |

---

## Section 5: Expert Panel Testing

The Expert Panel is an offline testing framework (`algorithm_testing/` and `adv_algorithm_testing/`) that evaluates algorithm selection strategies outside the main pipeline. It is used for research and validation, not production.

### 5.1 Basic Expert Panel

Tests champion selection strategies on a sample of DFUs with multiple algorithms:

```bash
make expert-panel              # 5000 DFUs, 5 timeframes, ~30 min
make expert-panel-quick        # 1000 DFUs, 3 timeframes, ~8 min
make expert-panel-mini         # 200 DFUs, 2 timeframes, ~2 min
make expert-panel-loc LOC=1401-BULK  # All DFUs at one location
```

### 5.2 Advanced Expert Panel

Includes execution-lag accuracy, foundation models, deep learning, and statistical model upgrades:

```bash
make adv-expert-panel          # 5000 DFUs, 10 timeframes
make adv-expert-panel-quick    # 1000 DFUs, 5 timeframes
make adv-expert-panel-mini     # 200 DFUs, 2 timeframes
make adv-expert-panel-loc LOC=1401-BULK
```

### 5.3 Expert System Backtest

Full-population backtest with segment-assigned algorithms, loads results to DB:

```bash
make expsys-backtest           # Full population, ~4-5h
make expsys-backtest-dry       # Accuracy only, no DB load
```

### 5.4 Known Gotchas

- **Ridge alpha must be 100.0** (not 1.0) to avoid LinAlgWarning. Also: drop constant columns before fitting, use `solver='lsqr'`.
- **Oracle ceiling is ~75.9%**: Champion = ~75.24%, portfolio = ~72.4%. Very little headroom above champion for single-model selection; hybrid/ensemble strategies are needed.

---

## Section 6: UI Surfaces Reference

### 6.1 Model Tuning Tab (`ModelTuningTab.tsx`)

The primary experimentation interface. Shows a pipeline layout with stage cards:
- **Clustering** card -> `ClusterExperimentsPanel` (create, compare, promote cluster experiments)
- **Champion** card -> `ChampionExperimentsPanel` (strategy experiments, comparison, two-stage promotion)

### 6.2 Aggregate Analysis Tab (`AggregateAnalysisTab.tsx`)

The accuracy monitoring dashboard. Shows:
- Accuracy KPIs by model, lag, timeframe, cluster, seasonality profile
- Production forecast trend charts
- Coverage statistics (DFUs with backtest data vs total)
- Sliceable by multiple dimensions (item, location, cluster, variability class, etc.)

### 6.3 Jobs Tab (`JobsTab.tsx`)

- **PipelineBuilderPanel**: Visual pipeline builder that can chain stages (normalize -> load -> backtest -> champion -> forecast generation). Can trigger `generate_production_forecast` as a pipeline step.
- Job history and status monitoring
- `ChampionConfigPanel`: Quick config for production competition (model checkboxes, metric, lag)

### 6.4 Settings Tab

- Shows all YAML config files organized by category
- Editable config cards for each domain (forecast, clustering, inventory, etc.) — this is where pipeline/algorithm/backtest/champion/clustering YAML is edited (there is no separate Pipeline Config panel in the Model Tuning tab)

### 6.5 SQL Runner Tab

- Ad-hoc SQL query interface for data exploration and export
- Parameterized queries with CSV download

---

## Section 7: Configuration Reference

### 7.1 Complete Config File Table

| File | Purpose | Key Settings | Used By |
|---|---|---|---|
| `forecast_pipeline_config.yaml` | Master pipeline config | 16 algorithms, backtest, tuning, champion, production_forecast, clustering, sampling, tracking | ALL stages |
| `etl_config.yaml` | ETL pipeline config | Domain load order, MV refresh tiers, parallel workers | Stage 1 |
| `cluster_experiment` table | Clustering ML params | k_range, min_cluster_size_pct, labeling thresholds | Stage 5 |
| `hyperparameter_tuning.yaml` | Optuna search spaces | n_trials, search_space per model, pruner settings | Stage 8 |
| `tuning_templates.yaml` | UI experiment templates | Per-model: production_baseline + 4 expert templates | Stage 8 UI |
| `champion_experiment_templates.yaml` | Champion strategy templates | 36 strategies: expanding, rolling, decay, ensemble, meta_learner, bandit, etc. | Stage 9 UI |
| `cluster_experiment_templates.yaml` | Cluster experiment templates | 7 templates: baseline, high-K, low-K, seasonal, intermittent, PCA, recent | Stage 5 UI |
| `cluster_tuning_profiles.yaml` | Per-cluster tuning profiles | Cluster-specific hyperparameter overrides | Stage 8 |
| `forecast_domain_config.yaml` | Seasonality + variability + quantile + bias | seasonality thresholds, variability CV classes, quantile model | Stage 2 |
| `expert_system_backtest.yaml` | Expert system backtest config | Segment-algorithm routing, DFU classification rules | Expert Panel |
| `ext_ml_forecasts.yaml` | External ML forecast config | External model registration, source mapping | Stage 1 |
| `data_quality_config.yaml` | Data quality rules | Validation rules, outlier thresholds, completeness checks | Stage 1 |
| `cache_config.yaml` | API cache settings | TTL, max entries, cache-aside patterns | API layer |
| `auth_config.yaml` | Authentication config | API key settings, rate limits | API layer |
| `perf_config.yaml` | Performance profiling | Timing thresholds, profiling settings | All stages |
| `shared_constants.yaml` | Shared constants | service_levels_by_abc, z_table, financial_defaults | Cross-cutting |
| `inventory_planning_config.yaml` | Inventory planning | Safety stock, lead time, ABC-XYZ thresholds | Stages 3, planning |
| `safety_stock_config.yaml` | Safety stock calculation | Guard rails, service level targets | Planning |

### 7.2 Key Config Settings: When and Why to Change

| Setting | Default | When to Change | Impact |
|---|---|---|---|
| `clustering.enabled` | true | Set false if SKUs are homogeneous or <1000 | All backtests use global strategy |
| `backtest.n_timeframes` | 10 | Reduce to 5 for experimentation speed | Less robust accuracy estimates |
| `backtest_sampling.enabled` | true | Set false for production-quality evaluation | Full population backtest (slower) |
| `backtest_sampling.default_target_n` | 5000 | Increase if you need broader DFU coverage | Longer backtest, better champion quality |
| `champion.strategy` | rolling | Try ensemble if oracle ceiling >> champion | Different model selection approach |
| `production_forecast.horizon_months` | 24 | Reduce for short-term planning | Fewer forecast months generated |

---

## Section 8: Database Reference

### 8.1 Complete Table Reference

**Dimension Tables:**

| Table | Purpose | Populated By |
|---|---|---|
| `dim_item` | Item master (descriptions, hierarchy) | Stage 1 (load) |
| `dim_location` | Location master (site, region, type) | Stage 1 (load) |
| `dim_customer` | Customer master (customer_no, group) | Stage 1 (load) |
| `dim_time` | Calendar dimension (auto-generated 2020-2035) | Stage 1 (load) |
| `dim_sku` | DFU dimension (item+loc+customer_group) | Stage 1 (load), Stages 2-5 (enrichment) |
| `dim_sourcing` | Sourcing relationships (supplier, lead time) | Stage 1 (load) |
| `dim_item_lead_time_profile` | Lead time variability profiles | Stage 3 |

**Fact Tables:**

| Table | Purpose | Populated By |
|---|---|---|
| `fact_sales_monthly` | Monthly sales actuals (type=1) | Stage 1 |
| `fact_external_forecast_monthly` | Backtest predictions + champion + external forecasts | Stages 1, 7, 9 |
| `fact_customer_demand_monthly` | Customer-level demand (monthly partitioned) | Stage 1 |
| `fact_inventory_snapshot` | Inventory positions (monthly partitioned, ~198M rows) | Stage 1 |
| `fact_purchase_orders` | Purchase order history | Stage 1 |
| `fact_production_forecast` | Forward-looking production forecasts | Stage 10 |
| `backtest_lag_archive` | All-lag (0-4) backtest predictions | Stage 7 |

**Experiment & Tracking Tables:**

| Table | Purpose | Populated By |
|---|---|---|
| `lgbm_tuning_run` | Tuning experiment summary (all tree models) | Stage 8 |
| `lgbm_tuning_timeframe` | Per-timeframe tuning breakdown | Stage 8 |
| `lgbm_tuning_cluster` | Per-cluster tuning breakdown | Stage 8 |
| `lgbm_tuning_month` | Per-month tuning breakdown | Stage 8 |
| `lgbm_tuning_comparison` | Pairwise tuning experiment comparison | Stage 8 |
| `tuning_promotion_log` | Tuning promotion audit trail | Stage 8 |
| `cluster_experiment` | Cluster experiment lifecycle | Stage 5 |
| `cluster_experiment_comparison` | Cluster experiment pairwise comparison | Stage 5 |
| `champion_experiment` | Champion experiment lifecycle | Stage 9 |
| `champion_experiment_lag` | Per-lag champion breakdown | Stage 9 |
| `champion_experiment_month` | Per-month champion breakdown | Stage 9 |
| `champion_experiment_comparison` | Champion experiment pairwise comparison | Stage 9 |
| `champion_promotion_log` | Champion promotion audit trail | Stage 9 |
| `job_history` | Background job execution history | All stages (UI jobs) |

**Key Materialized Views:**

| View | Purpose | Refresh Command |
|---|---|---|
| `agg_sales_monthly` | Pre-aggregated sales KPIs | `refresh-mvs-tiered` (tier 1) |
| `agg_forecast_monthly` | Pre-aggregated forecast KPIs | `refresh-mvs-tiered` (tier 1; refreshed `CONCURRENTLY`) |
| `agg_inventory_monthly` | EOM on-hand, sales, DOS, lead time | `refresh-mvs-tiered` (tier 1) |
| `mv_inventory_forecast_monthly` | Inventory-forecast bridge | `refresh-mvs-tiered` (tier 2) |
| `mv_fill_rate_monthly` | Fill rate metrics | `refresh-mvs-tiered` (tier 2) |
| `mv_supplier_performance` | Supplier delivery performance | `refresh-mvs-tiered` (tier 2) |
| `mv_intramonth_stockout` | Intra-month stockout detection | `refresh-mvs-tiered` (tier 2) |
| `mv_control_tower_kpis` | Control tower aggregate KPIs | `refresh-mvs-tiered` (tier 3) |
| `mv_network_balance` | Network inventory balance | `refresh-mvs-tiered` (tier 3) |
| `agg_accuracy_by_dim` | Pre-aggregated accuracy by dimension | `refresh-accuracy-mvs` (refreshed `CONCURRENTLY`) |
| `agg_accuracy_lag_archive` | Pre-aggregated archive accuracy | `refresh-accuracy-mvs` (refreshed `CONCURRENTLY`) |
| `agg_dfu_coverage` | DFU backtest coverage stats | `refresh-accuracy-mvs` (refreshed `CONCURRENTLY`) |

---

## Section 9: Product Gaps & Missing Support

### Gap 1: Limited UI trigger for production forecast generation

**What exists:** The Jobs tab `PipelineBuilderPanel` CAN trigger forecast generation as part of a pipeline sequence. However, there is no standalone "Generate Forecasts" button in the Model Tuning Tab workflow.

**Impact:** Users who complete champion selection in the Model Tuning Tab must navigate to the Jobs tab to trigger forecast generation, or use the CLI. The workflow is not seamless end-to-end.

**Severity:** MEDIUM

**Suggested fix:** Add a "Generate Forecasts" action button directly in the Model Tuning Tab champion stage card, wired to the same job endpoint.

---

### Gap 2: No end-to-end experiment pipeline (cluster -> backtest -> champion)

**What's missing:** Each experiment type operates independently. There is no automated chain: "Run cluster experiment -> backtest with those clusters -> run champion selection."

**Impact:** After promoting a new clustering configuration, users must manually trigger backtest re-runs, then re-load, then re-run champion selection.

**Severity:** HIGH

**Suggested fix:** Add a "Full Pipeline Experiment" workflow that chains sub-stages as a single tracked experiment.

---

### Gap 3: Foundation and DL models cannot be tuned through the UI


**Impact:** Foundation model parameters (batch_size, num_samples) and DL hyperparameters (max_steps, learning_rate) require manual YAML edits.

**Severity:** MEDIUM

---

### Gap 4: No production forecast versioning UI

**What's missing:** `fact_production_forecast` has `plan_version` and `keep_last_n_versions: 3`, but no UI to view, compare, or manage plan versions.

**Severity:** MEDIUM

---

### Gap 5: Backtest sampling impacts champion selection accuracy without UI awareness


**Impact:** Champion selection quality is silently degraded. The UI does not surface sampling status or coverage percentage.

**Severity:** CRITICAL

**Suggested fix:** (a) Warning banner in champion experiments panel when sampling is active. (b) Show coverage percentage. (c) Consider full backtest for champion-eligible models, sampled for experimentation only.

---

### Gap 6: `lgbm_tuning_run` table name is misleading

**What's missing:** Tuning tables have "lgbm" prefix but are used for all three tree model types.

**Severity:** LOW

---

### Gap 7: No automated accuracy regression detection after promotion

**What's missing:** No automated guard validates that end-to-end accuracy did not regress after a promotion. The `tracking.improvement_threshold_bps: 5` setting exists for pairwise comparison but is not enforced as a gate.

**Severity:** MEDIUM

---

### Gap 8: Customer-enriched model inference gap

**What's missing:** If a customer-enriched model wins champion selection, production forecast generation may fail if customer features are not in the inference grid.

**Severity:** MEDIUM

---

### Gap 9: MV refresh not automated after backtest loading

**What's missing:** `backtest-load-all-bulk` does not automatically refresh accuracy MVs.

**Severity:** MEDIUM

**Suggested fix:** Chain `refresh-accuracy-mvs` into the load target.

---

### Gap 10: Foundation-model production-forecast inference path

**What's missing:** The sole remaining foundation model, `chronos2_enriched`, has `forecast: true`
(unlike the deleted zero-shot Chronos/Bolt variants, which had `forecast: false`). Production
inference for a `chronos2_enriched` champion routes through the shared non-tree
`generate_forecasts_statistical()` path (see [Production Forecast](./08-production-forecast.md) Step
2) rather than a `.pkl`-backed recursive pass, since foundation models never persist trained artifacts.

**Impact:** Worth re-verifying whether that shared statistical path reproduces Chronos-quality
forecasts for a `chronos2_enriched` champion, now that it is the only foundation model in the roster.

**Severity:** MEDIUM

---

### Critical File Paths

| Category | Path |
|---|---|
| Master pipeline config | `config/forecasting/forecast_pipeline_config.yaml` |
| Production forecast script | `scripts/forecasting/generate_production_forecasts.py` |
| Champion selection script | `scripts/run_champion_selection.py` |
| Tree model backtest | `scripts/ml/run_backtest.py` |
| Foundation model backtest | `scripts/ml/run_backtest_chronos2_enriched.py` |
| Deep learning backtest | `scripts/ml/run_backtest_dl.py` |
| Model Tuning UI | `frontend/src/tabs/ModelTuningTab.tsx` |
| Aggregate Analysis UI | `frontend/src/tabs/AggregateAnalysisTab.tsx` |
| Pipeline Builder UI | `frontend/src/tabs/jobs/PipelineBuilderPanel.tsx` |
| Config editing UI | `frontend/src/tabs/SettingsTab.tsx` (YAML config editor; the former `model-tuning/PipelineConfigPanel.tsx` was removed) |
| Cluster experiments API | `api/routers/forecasting/cluster_experiments.py` |
| Champion experiments API | `api/routers/forecasting/champion_experiments.py` |
| Model tuning API | `api/routers/forecasting/tuning/` (15-module package; mounted via `tuning/__init__.py`. See [Unified Model Tuning Studio](./11-unified-model-tuning-v2.md#router-layout) for the full sub-router map) |
| Champion strategies library | `common/ml/champion/` (9-module package: `registry.py`, `basic.py`, `blend.py`, `meta.py`, `bandit.py`, `segment.py`, `regime.py`, `routing.py`, `helpers.py`. See [Champion Selection](./07-champion-selection.md#module-layout)) |
| Expert panel (basic) | `algorithm_testing/run_expert_panel.py` |
| Expert panel (advanced) | `adv_algorithm_testing/run_adv_expert_panel.py` |
