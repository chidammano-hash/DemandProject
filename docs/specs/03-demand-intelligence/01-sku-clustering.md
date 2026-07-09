# SKU Clustering & Experimentation Studio

> Groups SKUs (item + location combinations) by shared demand patterns across 14 features and 6 dimensions, with automatic K selection, taxonomy labeling, a What-If scenario UI for planners, and a full experiment lifecycle for testing segmentation configurations (create, run, compare, promote) with cluster-aware algorithm tuning integration.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Clusters (Overview + Experiments sub-tabs) |
| **Key Files** | `common/ml/clustering/` (features, training, labeling, scenario), `scripts/ml/run_cluster_pipeline.py`, `api/routers/clusters.py`, `api/routers/forecasting/cluster_experiments.py`, `cluster_experiment` table |
| **API Prefixes** | `/clustering`, `/cluster-experiments` |
| **Frontend** | `frontend/src/tabs/clusters/ClusterExperimentsPanel.tsx`, `frontend/src/api/queries/cluster-experiments.ts` |
| **Tests** | `tests/api/test_cluster_experiments.py`, `frontend/src/tabs/__tests__/ClusterExperimentsPanel.test.tsx` |
| **Depends On** | Feature 44 (Resilient Jobs), Feature 46 (Unified Model Tuning Studio) |

---

## Table of Contents

1. [Overview](#overview)
2. [Problem](#problem)
3. [Prerequisites](#prerequisites)
4. [Architecture](#architecture)
5. [Feature Engineering](#feature-engineering)
6. [Training Pipeline](#training-pipeline)
7. [Taxonomy Labeling](#taxonomy-labeling)
8. [UI Workflows](#ui-workflows)
9. [Experimentation Studio](#experimentation-studio)
10. [Promotion & Downstream Impact](#promotion--downstream-impact)
11. [Backtest Cluster Override](#backtest-cluster-override)
12. [Algorithm Tuning Integration](#algorithm-tuning-integration)
13. [API Endpoints](#api-endpoints)
14. [Database Schema](#database-schema)
15. [Frontend Components](#frontend-components)
16. [Configuration](#configuration)
17. [Pipeline](#pipeline)
18. [Output Artifacts](#output-artifacts)
19. [Testing](#testing)
20. [Dependencies](#dependencies)

---

## Overview

Clustering groups SKUs (item + location combinations) by shared demand patterns. It *describes* demand behavior rather than *predicting* future values, making it a demand-intelligence capability rather than a forecasting one. Downstream consumers include inventory planning (safety stock by segment), forecasting (per-cluster model training), and exception detection.

The Cluster Experimentation Studio adds a persistent experiment lifecycle on top of the clustering engine. Users create labeled experiments from templates, run them as resilient jobs, compare results side by side with migration matrices and quality metric deltas, and promote winners to production. Completed cluster experiments can be referenced by algorithm tuning experiments, enabling accuracy-aware cluster promotion.

---

## Problem

A portfolio of 100K+ SKUs cannot be managed individually. Planners need coherent groups that share demand characteristics so they can apply segment-level policies, train ML models per cluster, and spot anomalies within groups.

Additionally, clustering and algorithm tuning were historically disconnected workflows. The What-If scenario engine produces cluster assignments but has no persistent tracking, no comparison infrastructure, and no mechanism to feed experimental clusters into algorithm backtests. Specific gaps addressed by the Experimentation Studio:

1. **No experiment lifecycle** -- Cluster scenarios run and produce artifacts but are not tracked in the database. There is no list view, no labeling, no history.
2. **No comparison** -- Two cluster configurations cannot be compared side by side. There is no migration matrix showing how SKUs move between cluster labels, no quality metric deltas.
3. **No accuracy feedback loop** -- When a user promotes a new clustering, there is no way to preview the accuracy impact on LGBM/CatBoost/XGBoost backtests before committing.
4. **No templates** -- Every scenario requires manually specifying all parameters. There are no preset starting points for common strategies (high-K granular, low-K broad, seasonal focus).

---

## Prerequisites

Before clustering can run, the following must be in place:

| Requirement | Source |
|---|---|
| Sales history loaded | `fact_sales_monthly` via `make load-all` |
| SKU dimension populated | `dim_sku` via `make load-all` |
| Planning date set | `config/planning_config.yaml` or `PLANNING_DATE` env var |
| Clustering enabled | `clustering.enabled: true` in `forecast_pipeline_config.yaml` |

Minimum data: each SKU needs `min_months_history` (default 12) months of sales within the `time_window_months` (default 36) window.

---

## Architecture

### Library Package: `common/ml/clustering/`

| Module | Exports | Purpose |
|---|---|---|
| `features.py` | `compute_time_series_features()` | 30+ time-series features per SKU |
| `training.py` | `CORE_FEATURES`, `find_optimal_k()`, `merge_small_clusters()` | KMeans training + K selection |
| `labeling.py` | `assign_cluster_labels()` | Business taxonomy labels |
| `scenario.py` | `generate_scenario_id()`, `promote_scenario()`, `get_scenario_result()`, `_load_config_defaults()` | Scenario infrastructure |

### Orchestration Scripts

| Script | Purpose |
|---|---|
| `scripts/run_clustering_scenario.py` | `run_scenario()` -- full pipeline orchestrator (used by both UI and CLI) |
| `scripts/ml/run_cluster_pipeline.py` | `run_unified_pipeline()` -- creates experiment row, runs, auto-promotes |

### System Diagram

```
+---------------------------------------------------------+
|  CLUSTER EXPERIMENTATION STUDIO                         |
|  (Independent, no model tied)                           |
|                                                         |
|  Create -> Run -> Compare -> Promote                    |
|  Table: cluster_experiment, cluster_experiment_comparison|
|  Router: cluster_experiments.py                         |
|  UI: ClustersTab > Experiments sub-tab                  |
+----------------------------+----------------------------+
                             | cluster_experiment_id (optional FK)
                             v
+---------------------------------------------------------+
|  ALGORITHM TUNING STUDIO (Feature 46)                   |
|  (Independent, its own lifecycle)                       |
|                                                         |
|  Cluster Source: Production | Experiment #X             |
|  Create -> Run -> Compare -> Promote                    |
|  Table: lgbm_tuning_run (extended with cluster_source)  |
|  Router: unified_model_tuning.py                        |
|  UI: ModelTuningTab > ExperimentBuilder                 |
+---------------------------------------------------------+
```

Connection is **one-directional**: algorithm experiments can reference a cluster experiment via `cluster_experiment_id` FK, but cluster experiments know nothing about algorithm experiments. The `used-by` endpoint provides a reverse lookup for informational purposes only.

### Data Flow

```
config/forecasting/cluster_experiment_templates.yaml
         |
         v
+------------------+    POST /cluster-experiments     +--------------+
| ExperimentBuilder|--------------------------------->| cluster_     |
| (ClusterTab UI)  |                                  | experiments  |
+------------------+                                  | .py router   |
                                                      +------+-------+
                                                             | submit cluster_scenario job
                                                             v
                                                      +--------------+
                                                      | job_state.py |
                                                      | _run_cluster |
                                                      | _scenario()  |
                                                      +------+-------+
                                                             | Popen -> run_cluster_pipeline.py
                                                             v
                                                      +--------------+
                                                      | Artifacts:   |
                                                      | cluster_     |
                                                      | labels.csv   |
                                                      | + profiles   |
                                                      +------+-------+
                                                             | write results to DB
                                                             v
                                                      +--------------+
                                                      | cluster_     |
                                                      | experiment   |
                                                      | table (DB)   |
                                                      +------+-------+
                                                             | optional: reference from
                                                             v algorithm tuning
                                                      +--------------+
                                                      | lgbm_tuning_ |
                                                      | run.cluster_ |
                                                      | experiment_id|
                                                      +--------------+
```

### Configuration Source

Clustering parameters are stored in the **`cluster_experiment` table** (promoted row). No YAML config file -- the promoted experiment's `feature_params`, `model_params`, and `label_params` JSONB columns are the single source of truth.

Hardcoded defaults (used when no experiment is promoted):
```
time_window_months: 36, min_months_history: 12
k_range: [9, 18], min_cluster_size_pct: 2.0, use_pca: false
volume_high: 0.75, volume_low: 0.25, cv_steady: 0.4, cv_volatile: 0.8
```

---

## Feature Engineering

`compute_time_series_features()` computes per-SKU features from monthly sales:

| Dimension | Core Features | Method |
|---|---|---|
| **Volume** | `mean_demand`, `cv_demand`, `iqr_demand` | Basic stats on demand array |
| **Trend** | `trend_slope_norm`, `trend_r2`, `cagr` | Linear regression + CAGR |
| **Seasonality** | `seasonal_amplitude`, `seasonal_r2`, `yoy_correlation` | STL-lite OLS + year-over-year pivot correlation |
| **Periodicity** | `periodicity_strength` | FFT dominant-frequency power ratio |
| **Intermittency** | `zero_demand_pct`, `adi` | Gap analysis + Croston ADI |
| **Lifecycle** | `months_available`, `recency_ratio` | Temporal splits |

**Two feature sets:**
- **Core** (14 features): Used by default -- the 14 features above
- **All** (~28 features): Includes median, std, total, max, trend_pct_change, seasonality_strength, etc.

Log-transforms applied to skewed volume features. StandardScaler normalization. Optional PCA reduction.

---

## Training Pipeline

### Step-by-Step Flow

```
fact_sales_monthly + dim_sku
    -> Filter (min_months_history, time_window)
    -> compute_time_series_features() per SKU
    -> Normalize (log-transform, scale, PCA optional)
    -> find_optimal_k() -- evaluate K range, score, select best
    -> KMeans training on optimal K
    -> merge_small_clusters() -- merge undersized clusters
    -> assign_cluster_labels() -- taxonomy labeling
    -> Build profiles + PCA scatter
    -> Save artifacts + update cluster_experiment table
```

### Optimal K Selection

1. Evaluate K in configured range (default [9, 18])
2. Score: `0.5 * silhouette_norm + 0.5 * calinski_harabasz_norm`
3. Discard K where smallest cluster < `min_cluster_size_pct`
4. Select K with highest combined score
5. `merge_small_clusters()` merges remaining undersized clusters into nearest neighbor

---

## Taxonomy Labeling

Priority-ordered evaluation (first match wins):

| Priority | Dimension | Example Label |
|---|---|---|
| 1 | Intermittency | `intermittent_sporadic` |
| 2 | Periodicity | `periodic_quarterly` |
| 3 | Seasonality | `high_volume_seasonal` |
| 4 | Trend | `growing_moderate` |
| 5 | Volatility | `volatile_erratic` |
| 6 | Volume (5 tiers) | `steady_high_volume` |

Compound labels (e.g., `high_volume_seasonal_growing`). Promoted labels are
stored in `sku_cluster_assignment` and exposed through
`current_sku_cluster_assignment`.

---

## UI Workflows

### Workflow 1: Experiment Studio (What-If)

```
User: "New Experiment" -> ClusterExperimentBuilder modal
  -> Select template (Production Baseline, High-K, Low-K, Custom, etc.)
  -> Customize params (K range, features, PCA, thresholds)
  -> Click "Launch Experiment"
    -> POST /cluster-experiments (body: params)
      -> Backend creates cluster_experiment row (status=queued)
      -> Submits cluster_scenario job to JobManager
      -> Returns 202 + experiment_id
    -> JobManager executes _run_cluster_scenario()
      -> Calls run_scenario() -- full pipeline
      -> Updates cluster_experiment row with results
  -> User sees results (profiles, PCA scatter, K selection chart)
  -> User compares experiments (config diff + metric comparison)
  -> User promotes best experiment
    -> POST /cluster-experiments/{id}/promote
      -> promote_scenario() upserts sku_cluster_assignment
      -> Refreshes cluster-dependent accuracy/coverage MVs
      -> Sets is_promoted=TRUE on experiment row
```

### Workflow 2: Production Pipeline (Run Pipeline)

```
User: "Run Pipeline" button in Cluster Overview
  -> submitJob("cluster_pipeline", {})
    -> JobManager executes _run_cluster_pipeline()
      -> Calls run_unified_pipeline()
        -> Reads params from promoted experiment (or defaults)
        -> Creates cluster_experiment row
        -> Runs full scenario
        -> AUTO-PROMOTES result
        -> Refreshes accuracy MVs
```

### Workflow 3: CLI Batch

```bash
make cluster-all
  -> python scripts/ml/run_cluster_pipeline.py --label "make cluster-all"
    -> Same as Workflow 2: create experiment -> run -> auto-promote
```

---

## Experimentation Studio

The Experimentation Studio supersedes ad-hoc What-If scenarios with a persistent, labeled experiment lifecycle.

### Experiment Builder

Users launch experiments from the ClustersTab Experiments sub-tab. A full-screen builder modal (`ClusterExperimentBuilder.tsx`) offers:

1. **Label & Notes** -- text input + textarea
2. **Template Selection** -- radio button grid:
   - Production Baseline (reads `cluster_experiment` table live)
   - High-K Granular (K=12-25, min_size=1.5%)
   - Low-K Broad (K=3-8, min_size=5%)
   - Seasonal Focus (48mo window, low threshold)
   - Intermittent Specialist (low zero_demand threshold)
   - PCA Compressed (all features + PCA)
   - Custom (start from defaults)
3. **Parameters** -- 3-column grid (reusable `ClusterParamsForm.tsx`):
   - **Data Scope**: time_window_months, min_months_history
   - **Model**: k_range [min, max], min_cluster_size_pct, use_pca, pca_components
   - **Labeling Thresholds**: volume_high, volume_low, cv_steady, cv_volatile, seasonality_threshold, zero_demand_threshold
4. **Estimate bar** -- shows estimated runtime + SKU count (from `/clustering/scenario/estimate`)
5. **Footer**: Cancel | Launch Experiment

**Clone support**: "Clone" action on each experiment row opens builder pre-populated with that experiment's params.

### Run

The existing `cluster_scenario` job type is reused. The `cluster_experiments.py` router creates the DB record in `cluster_experiment`, then submits the same `cluster_scenario` job with the `experiment_id` added to params. After `run_scenario()` completes, results (optimal_k, silhouette, profiles, cluster_sizes) are written to the `cluster_experiment` table.

### Compare

Two experiments can be compared side by side (`ClusterComparisonPanel.tsx`). The comparison panel shows:

1. **Quality Metrics Header** -- two-column showing each experiment's silhouette, inertia, K, SKUs with delta badges (arrows, green/red)
2. **Cluster Profile Comparison Table** -- merged table showing clusters from both experiments with side-by-side count, percentage, and delta columns
3. **SKU Migration Matrix** -- ECharts heatmap showing how SKUs moved between cluster labels (rows = Exp A, cols = Exp B, cells = count with color intensity). Includes an accessible `<table>` alternative view toggle ("Table view" button) for screen readers. `overflow-x-auto` with 500px min width.
4. **K Selection Overlay** -- line chart overlaying both experiments' elbow curves
5. **Promote Button** -- for the candidate experiment

Migration matrix computation: load both experiments' `cluster_labels.csv` from `artifacts_path`, join on `sku_ck`, compute pandas `crosstab`. Results are cached in `cluster_experiment_comparison` table.

Heatmap cells: tooltip on hover "2,345 SKUs moved from [source] to [target]"

### Compare Response Format

```json
{
    "experiment_a": { "..." : "full experiment object" },
    "experiment_b": { "..." : "full experiment object" },
    "quality_comparison": {
        "silhouette_delta": -0.032,
        "inertia_delta": -27000,
        "k_delta": 3,
        "verdict": "mixed"
    },
    "profile_comparison": {
        "clusters_only_in_a": ["low_volume_volatile"],
        "clusters_only_in_b": ["very_high_volume_seasonal"],
        "common_clusters": [{ "label": "high_volume_steady", "count_a": 3456, "count_b": 3120 }]
    },
    "migration_matrix": {
        "high_volume_steady": { "high_volume_steady": 2800, "very_high_volume_seasonal": 656 }
    },
    "total_skus_migrated": 4500,
    "total_skus_unchanged": 8200
}
```

---

## Promotion & Downstream Impact

The promotion flow performs:
1. Load per-SKU labels — from the working `cluster_labels.csv` if present, else from
   the durable gzip copy on the experiment row (`cluster_experiment.cluster_labels_gz`,
   sql/191), reconstructing the production CSV
2. Copy artifacts to `data/clustering/` (production location)
3. Bulk UPSERT `sku_cluster_assignment` for all SKUs
4. Refresh cluster-dependent accuracy/coverage materialized views after the
   promoted experiment flag is set:
   - `agg_accuracy_by_dim`
   - `agg_accuracy_lag_archive`
   - `agg_dfu_coverage`
   - `agg_dfu_coverage_lag_archive`
   - `agg_accuracy_by_dfu`

**Durable re-promotion:** on completion, `store_durable_labels()` saves the per-SKU
assignments (gzip) onto the experiment row, so any completed experiment can be
re-promoted later without re-running clustering — even after the working
`/tmp/clustering_scenarios/<id>/` artifacts are cleared. The promote endpoint rejects
(409) an experiment that has no cluster results.

### Cluster Experiment Promotion

`POST /cluster-experiments/{id}/promote` writes results to
`sku_cluster_assignment`; downstream readers use `current_sku_cluster_assignment`.

A warning modal (`ClusterPromoteModal.tsx`) displays:
- Downstream impact warnings (backtests, forecasts, inventory planning)
- Experiment details (K, silhouette, SKU count)
- Tip: "Re-run your algorithm experiments with the new clusters to see accuracy impact"
- Confirm / Cancel buttons

### Downstream Systems Using `ml_cluster`

| System | How it uses clusters |
|---|---|
| **Per-cluster backtests** | `lgbm_cluster`, `catboost_cluster`, `xgboost_cluster` train separate models per cluster |
| **Accuracy analysis** | `agg_accuracy_by_dim` slices accuracy by promoted `ml_cluster` |
| **Champion selection** | Champion model selected per cluster |
| **Production forecast** | Routes SKUs to cluster-specific models |
| **Cluster-aware tuning** | `lgbm_tuning_run.cluster_experiment_id` FK |
| **Safety stock** | Segment-level policies by cluster |

### Deletion Guards

Cluster experiments cannot be deleted if:
- `is_promoted = TRUE` (production config)
- `status IN ('running', 'queued')` (in progress)
- Referenced by `lgbm_tuning_run.cluster_experiment_id` (used by tuning)

---

## Backtest Cluster Override

### CLI Argument

`scripts/run_backtest.py` accepts `--cluster-override`:

```python
parser.add_argument("--cluster-override", type=str, default=None,
    help="CSV with sku_ck,cluster_label to override promoted ml_cluster assignments")
```

Read from config if not on CLI:
```python
cluster_override = args.cluster_override or algo_config.get("cluster_override_path")
```

### Override Mechanism

In `common/ml/backtest_framework.py` -- `load_backtest_data()`, after loading `sku_attrs` from DB, inject override:

```python
if cluster_override_path:
    override_df = pd.read_csv(cluster_override_path, usecols=["sku_ck", "cluster_label"])
    override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
    sku_attrs["ml_cluster"] = sku_attrs["sku_ck"].map(override_map).fillna(sku_attrs["ml_cluster"])
    logger.info(f"Cluster override: {len(override_map):,} SKUs remapped")
```

Minimal, non-invasive. Only affects the in-memory DataFrame. All downstream per-cluster training, feature engineering, and metrics work unchanged because they reference `sku_attrs["ml_cluster"]`.

---

## Algorithm Tuning Integration

The connection between the Cluster Experimentation Studio and the Algorithm Tuning Studio is one-directional and optional:

1. **Creating an algorithm experiment** -- The ExperimentBuilder UI shows a "Cluster Source" dropdown. Default is "Production Clusters". If completed cluster experiments exist, they appear in the dropdown.
2. **Running with experimental clusters** -- When `cluster_source=experimental`, the backtest subprocess receives `--cluster-override` pointing to the cluster experiment's `cluster_labels.csv`. The `ml_cluster` column in the in-memory DataFrame is remapped before training/prediction.
3. **Tracking lineage** -- `lgbm_tuning_run.cluster_source` and `cluster_experiment_id` record which cluster configuration was used. Responses include `cluster_experiment_label` via a LEFT JOIN.
4. **Deletion safety** -- `ON DELETE SET NULL` on the FK means deleting a cluster experiment does not delete algorithm experiments. The UI shows "Cluster experiment deleted" when `cluster_experiment_id` is NULL but `cluster_source` is `'experimental'`.

### Cluster Source Selector in ExperimentBuilder

In `frontend/src/tabs/model-tuning/ExperimentBuilder.tsx`, the Training Configuration section includes a "Cluster Source" selector:

```
Cluster Source                 | Cluster Strategy        | Recursive
[Production Clusters      v]  | [per_cluster        v]  | [x] Recursive
```

Dropdown options (shadcn/ui `Select`):
- **Production Clusters** (default) -- shows current production K and silhouette in secondary text
- Separator line
- **Completed cluster experiments** -- each showing: `[label] -- K=[n], Sil=[score]`

When experimental cluster selected:
- Info badge below dropdown: "Using clusters from experiment #3: High-K K=15"
- `config.cluster_source = "experimental"` and `config.cluster_experiment_id = 3` included in payload

### Algorithm Tuning API Changes

`api/routers/forecasting/unified_model_tuning.py` -- `CreateExperimentBody.config` gains:

```python
cluster_source: str = "production"        # "production" | "experimental"
cluster_experiment_id: int | None = None  # FK to cluster_experiment (required when experimental)
```

When `cluster_source == "experimental"`:
- Validate `cluster_experiment_id` exists and has `status='completed'`
- Read `artifacts_path` from `cluster_experiment` table
- Build `cluster_override_path` into the temp `forecast_pipeline_config.yaml`
- Store `cluster_source` and `cluster_experiment_id` on `lgbm_tuning_run`

---

## API Endpoints

### Clustering Scenario API (`/clustering`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/clustering/scenario` | Submit What-If scenario (202 async) |
| GET | `/clustering/scenario/{id}/status` | Poll scenario progress |
| GET | `/clustering/scenario/estimate` | Runtime estimate |
| GET | `/clustering/core-features` | List of 14 core feature names |
| GET | `/clustering/defaults` | Current default params (from promoted experiment) |

### Cluster Experiments API (`/cluster-experiments`)

Router: `api/routers/forecasting/cluster_experiments.py` -- mounted in `api/main.py` before `domains.py`.

| Method | Path | Description |
|---|---|---|
| GET | `/cluster-experiments` | List experiments (paginated, filterable by status) |
| GET | `/cluster-experiments/{id}` | Single experiment detail |
| POST | `/cluster-experiments` | Create + launch experiment (202 async) |
| PATCH | `/cluster-experiments/{id}` | Update label/notes |
| DELETE | `/cluster-experiments/{id}` | Delete (with guards: 409 if running/queued/promoted/referenced) |
| POST | `/cluster-experiments/{id}/promote` | Promote to production `sku_cluster_assignment` |
| GET | `/cluster-experiments/compare/{a}/{b}` | Compare two experiments |
| GET | `/cluster-experiments/templates` | Cluster experiment templates from YAML config |
| GET | `/cluster-experiments/completed` | Completed experiments only (for algorithm experiment dropdown) |
| GET | `/cluster-experiments/{id}/used-by` | Algorithm experiments that reference this cluster |

### Create Request Body

```json
{
    "label": "High-K Seasonal Focus",
    "notes": "Testing whether K=15 captures seasonal patterns better",
    "template": "seasonal_focus",
    "feature_params": { "time_window_months": 36, "min_months_history": 12 },
    "model_params": { "k_range": [10, 20], "min_cluster_size_pct": 2.0 },
    "label_params": { "seasonality_threshold": 0.2 }
}
```

---

## Database Schema

### File: `sql/101_cluster_experiments.sql`

#### Table: `cluster_experiment`

Tracks the full lifecycle of a cluster experimentation run.

```sql
CREATE TABLE IF NOT EXISTS cluster_experiment (
    experiment_id       SERIAL PRIMARY KEY,
    scenario_id         VARCHAR(30) NOT NULL UNIQUE,   -- sc_YYYYMMDD_HHMMSS_xxxx
    label               TEXT NOT NULL,
    notes               TEXT,
    template_id         VARCHAR(100),
    status              TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued','running','completed','failed','cancelled')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    runtime_seconds     NUMERIC(8,1),
    job_id              VARCHAR(100),

    -- Input config
    feature_params      JSONB,   -- {time_window_months, min_months_history}
    model_params        JSONB,   -- {k_range, min_cluster_size_pct, use_pca, ...}
    label_params        JSONB,   -- {volume_high, volume_low, cv_steady, ...}

    -- Results (populated on completion)
    optimal_k           INTEGER,
    silhouette_score    NUMERIC(8,6),
    inertia             NUMERIC(14,2),
    total_skus          INTEGER,
    n_clusters          INTEGER,
    cluster_sizes       JSONB,
    profiles            JSONB,
    k_selection_results JSONB,

    -- Promotion
    is_promoted         BOOLEAN NOT NULL DEFAULT FALSE,
    promoted_at         TIMESTAMPTZ,
    artifacts_path      TEXT     -- /tmp/clustering_scenarios/{scenario_id}
);

CREATE INDEX idx_cluster_exp_status ON cluster_experiment(status);
CREATE INDEX idx_cluster_exp_promoted ON cluster_experiment(is_promoted) WHERE is_promoted;
CREATE INDEX idx_cluster_exp_created ON cluster_experiment(created_at DESC);
```

**Key columns:**

| Column | Type | Purpose |
|--------|------|---------|
| `scenario_id` | VARCHAR(30) | Unique scenario identifier (`sc_YYYYMMDD_HHMMSS_xxxx`) |
| `feature_params` | JSONB | Data scope: `time_window_months`, `min_months_history` |
| `model_params` | JSONB | KMeans config: `k_range`, `min_cluster_size_pct`, `use_pca`, `pca_components` |
| `label_params` | JSONB | Labeling thresholds: `volume_high`, `volume_low`, `cv_steady`, `cv_volatile`, `seasonality_threshold`, `zero_demand_threshold` |
| `profiles` | JSONB | Cluster profiles (populated on completion) |
| `artifacts_path` | TEXT | Filesystem path to clustering artifacts (cluster_labels.csv, etc.) |
| `is_promoted` | BOOLEAN | Whether this experiment has been promoted to production |

#### Table: `cluster_experiment_comparison`

Caches pairwise comparison results (migration matrix, quality deltas) to avoid recomputation.

```sql
CREATE TABLE IF NOT EXISTS cluster_experiment_comparison (
    id                  SERIAL PRIMARY KEY,
    experiment_a_id     INTEGER NOT NULL REFERENCES cluster_experiment(experiment_id) ON DELETE CASCADE,
    experiment_b_id     INTEGER NOT NULL REFERENCES cluster_experiment(experiment_id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    migration_matrix    JSONB,     -- {from_label: {to_label: count}}
    quality_comparison  JSONB,
    profile_comparison  JSONB,
    UNIQUE (experiment_a_id, experiment_b_id)
);
```

#### Extension: `lgbm_tuning_run`

Adds cluster source reference to algorithm tuning experiments.

```sql
ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS cluster_source VARCHAR(20) NOT NULL DEFAULT 'production'
        CHECK (cluster_source IN ('production', 'experimental')),
    ADD COLUMN IF NOT EXISTS cluster_experiment_id INTEGER
        REFERENCES cluster_experiment(experiment_id) ON DELETE SET NULL;

CREATE INDEX idx_tuning_run_cluster_exp
    ON lgbm_tuning_run(cluster_experiment_id)
    WHERE cluster_experiment_id IS NOT NULL;
```

`ON DELETE SET NULL` ensures algorithm experiments survive cluster experiment deletion. The UI shows "Cluster experiment deleted" in gray when `cluster_experiment_id` is NULL but `cluster_source` is `'experimental'`.

### Data Model Summary

| Table / Column | Type | Purpose |
|---|---|---|
| `cluster_experiment` | Table | Experiment tracking (params, results, promotion) |
| `cluster_experiment.feature_params` | JSONB | Feature engineering params |
| `cluster_experiment.model_params` | JSONB | KMeans training params |
| `cluster_experiment.label_params` | JSONB | Labeling threshold params |
| `cluster_experiment.is_promoted` | BOOLEAN | Current production config flag |
| `sku_cluster_assignment.cluster_label` | TEXT | Production cluster label for a promoted experiment |
| `current_sku_cluster_assignment.ml_cluster` | TEXT | Current promoted cluster label view for downstream reads |
| `dim_sku.cluster_assignment` | TEXT | Production cluster label (alias) |
| `lgbm_tuning_run.cluster_experiment_id` | INTEGER FK | Links tuning runs to cluster experiments |
| `cluster_experiment_comparison` | Table | Cached comparison results (ON DELETE CASCADE) |

`ml_cluster` is excluded from model features (listed in `METADATA_COLS` in `constants.py`) to prevent data leakage. It is merged into the feature grid as a metadata column for per-cluster model partitioning — models are trained separately per cluster, but `ml_cluster` is never passed as an input feature.

---

## Frontend Components

### ClustersTab Sub-tabs

**Current:** Overview (implicit, single view)
**With Experimentation Studio:** Overview | **Experiments**

The WhatIfPanel (existing) is superseded by the Experiments sub-tab -- no more ad-hoc scenarios without labels/tracking.

### New Components in `frontend/src/tabs/clusters/`

#### `ClusterExperimentsPanel.tsx`

Two-column grid (`grid-cols-1 xl:grid-cols-2`):

**Left column -- Experiment list:**

KPI cards row:
- Best Silhouette (highest across completed)
- Production K (current promoted)
- Total Experiments
- Active Runs

Toolbar: Status filter dropdown + "New Experiment" button

Table columns:

| # | Label | Status | K | Silhouette | SKUs | Duration | Created | Actions |

Row click uses the same baseline/candidate selection pattern as ModelTuningTab (blue for baseline, green for candidate, toggle to deselect).

**Right column -- Comparison panel** (appears when 2 rows selected, empty-state guidance otherwise)

**Empty state** (when no experiments exist):
- Icon + "No cluster experiments yet"
- Explanatory text: "Cluster experiments let you test different SKU segmentation configurations before committing to production. Each experiment runs the full clustering pipeline with your chosen parameters."
- Primary "Create First Experiment" button

#### `ClusterExperimentBuilder.tsx`

Full-screen modal matching ExperimentBuilder pattern. See [Experimentation Studio](#experimentation-studio) section for full builder description.

#### `ClusterComparisonPanel.tsx`

See [Experimentation Studio -- Compare](#compare) section for full comparison panel description.

#### `ClusterPromoteModal.tsx`

Warning modal -- see [Promotion](#promotion--downstream-impact) section.

#### `ClusterParamsForm.tsx`

Reusable 3-column parameter form extracted for use by both `ClusterExperimentBuilder` and potential future clustering UI.

### Shared Utilities -- `frontend/src/components/shared-tuning-utils.tsx`

Extracted from ModelTuningTab for reuse by both ClusterExperimentsPanel and ModelTuningTab:
- `StatusBadge` component
- `formatDuration()` helper
- `timeAgo()` helper
- `SortIndicator` component
- `KpiCard` component

### Query Layer -- `frontend/src/api/queries/cluster-experiments.ts`

```typescript
// Types
interface ClusterExperiment {
    experiment_id: number;
    scenario_id: string;
    label: string;
    notes: string | null;
    template_id: string | null;
    status: "queued" | "running" | "completed" | "failed" | "cancelled";
    created_at: string;
    completed_at: string | null;
    runtime_seconds: number | null;
    feature_params: FeatureParams | null;
    model_params: ModelParams | null;
    label_params: LabelParams | null;
    optimal_k: number | null;
    silhouette_score: number | null;
    inertia: number | null;
    total_skus: number | null;
    n_clusters: number | null;
    profiles: ClusterProfile[] | null;
    is_promoted: boolean;
    promoted_at: string | null;
}

// Fetchers
fetchClusterExperiments(opts?) -> { experiments, total }
fetchClusterExperiment(id) -> ClusterExperiment
createClusterExperiment(body) -> { experiment_id, scenario_id, status, job_id }
deleteClusterExperiment(id) -> { deleted: boolean }
promoteClusterExperiment(id) -> { status, skus_updated }
fetchClusterComparison(aId, bId) -> ClusterExperimentComparison
fetchClusterTemplates() -> { templates }
fetchCompletedClusterExperiments() -> { experiments: ClusterExperiment[] }
fetchClusterExperimentUsedBy(id) -> { runs: TuningExperiment[] }
```

### Modifications to Existing Frontend Files

| File | Change |
|------|--------|
| `frontend/src/tabs/ClustersTab.tsx` | Add sub-tab navigation (Overview / Experiments) |
| `frontend/src/tabs/ModelTuningTab.tsx` | Add cluster source badge ("Exp #3" pill) to experiment table rows |
| `frontend/src/tabs/model-tuning/ExperimentBuilder.tsx` | Add Cluster Source selector dropdown |
| `frontend/src/tabs/model-tuning/EnhancedComparisonPanel.tsx` | Add "Cluster Source" row to config diffs |
| `frontend/src/api/queries/unified-model-tuning.ts` | Extend `TuningExperiment` with `cluster_source`, `cluster_experiment_id`, `cluster_experiment_label` |
| `frontend/vite.config.ts` | Add `/cluster-experiments` proxy entry |

---

## Configuration

### Cluster Pipeline Config

Source: `cluster_experiment` table (promoted row)

```yaml
time_window_months: 36
k_range: [5, 18]
min_cluster_size_pct: 5.0
scoring: combined          # 0.5 * silhouette + 0.5 * calinski_harabasz
labeling: priority_ordered # intermittency -> periodicity -> seasonality -> trend -> volatility -> volume
```

### Experiment Templates

File: `config/forecasting/cluster_experiment_templates.yaml`

```yaml
templates:
  - id: production_baseline
    label: "Production Baseline"
    description: "Current production parameters from cluster_experiment table"
    source: promoted_experiment
  - id: high_k_granular
    label: "High-K Granular"
    description: "More clusters (12-25) for finer segmentation"
    model_params: { k_range: [12, 25], min_cluster_size_pct: 1.5 }
  - id: low_k_broad
    label: "Low-K Broad"
    description: "Fewer clusters (3-8) for robust per-cluster training"
    model_params: { k_range: [3, 8], min_cluster_size_pct: 5.0 }
  - id: seasonal_focus
    label: "Seasonal Focus"
    description: "48-month window + low seasonality threshold"
    feature_params: { time_window_months: 48 }
    label_params: { seasonality_threshold: 0.2 }
  - id: intermittent_specialist
    label: "Intermittent Specialist"
    description: "Tuned for intermittent demand detection"
    label_params: { zero_demand_threshold: 0.1, cv_volatile: 0.6 }
  - id: pca_compressed
    label: "PCA Compressed"
    description: "All features with PCA dimensionality reduction"
    model_params: { all_features: true, use_pca: true, pca_components: 10 }
```

---

## Pipeline

```
make cluster-all    # features -> train -> label -> update (full pipeline)
```

Unified entry point: `scripts/ml/run_cluster_pipeline.py`

| Step | Module | Output |
|---|---|---|
| Feature engineering | `common/ml/clustering/features` | Feature matrix |
| Train + select K | `common/ml/clustering/training` | MLflow experiment `sku_clustering` |
| Label clusters | `common/ml/clustering/labeling` | Labeled cluster assignments |
| Write to DB | `common/ml/clustering/scenario` / `promote_scenario()` | `sku_cluster_assignment` upserted |

### Job Pipeline

No new job type registration needed. The existing `cluster_scenario` job type is reused. The `cluster_experiments.py` router creates the DB record in `cluster_experiment`, then submits the same `cluster_scenario` job with the `experiment_id` added to params.

In `common/services/job_state.py`:

- `_run_cluster_scenario`: After `run_scenario()` completes, writes results to `cluster_experiment` table (optimal_k, silhouette, profiles, cluster_sizes, status, runtime_seconds, completed_at).
- `_run_model_tuning_experiment`: When params include `cluster_experiment_id`, validates the experiment exists and has `status='completed'`, then passes `--cluster-override {artifacts_path}/cluster_labels.csv` to the backtest subprocess.

---

## Output Artifacts

| File | Location | Content |
|---|---|---|
| `clustering_features.csv` | scenario dir | Feature matrix per SKU |
| `cluster_centroids.csv` | scenario dir | Centroid values per cluster |
| `cluster_labels.csv` | scenario dir | SKU -> cluster_id -> cluster_label mapping |
| `scenario_result.json` | scenario dir | Full results (K, silhouette, profiles, PCA scatter) |
| `data/clustering/` | production | Promoted artifacts copied here |

---

## Testing

### Backend Tests

| Test File | Coverage |
|-----------|----------|
| `tests/api/test_cluster_experiments.py` | CRUD lifecycle (create, list, get, update, delete), compare endpoint (migration matrix, quality deltas), promote workflow, used-by reverse lookup, 409 on delete of running experiment, validation of completed status for cluster source |
| `tests/unit/test_backtest_cluster_override.py` | `--cluster-override` CLI arg parsing, `load_backtest_data()` override mechanism, partial override (some SKUs missing from CSV), empty override file |
| `tests/api/test_unified_model_tuning.py` (extend) | Create experiment with `cluster_source=experimental`, validation of `cluster_experiment_id`, response includes `cluster_experiment_label`, deleted experiment shows NULL ID |

### Frontend Tests

| Test File | Coverage |
|-----------|----------|
| `frontend/src/tabs/__tests__/ClusterExperimentsPanel.test.tsx` | List rendering, row selection (baseline/candidate), empty state, KPI cards, status filter |
| `frontend/src/tabs/__tests__/ClusterExperimentBuilder.test.tsx` | Template selection, parameter form, validation (required label), submit payload, clone pre-population |
| `frontend/src/tabs/__tests__/ClusterComparisonPanel.test.tsx` | Quality metrics rendering, profile comparison table, migration matrix heatmap mock, promote button |
| `frontend/src/tabs/__tests__/ExperimentBuilder.test.tsx` (extend) | Cluster source dropdown rendering, completed experiments in dropdown, empty dropdown state, payload includes cluster_source fields |

### Test Pattern

Backend tests follow the existing pattern:

```python
@pytest.mark.asyncio
async def test_list_cluster_experiments(tmp_path):
    with patch("api.core._get_pool", return_value=_make_pool()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/cluster-experiments")
    assert response.status_code == 200
```

Frontend tests use `TestQueryWrapper` and mock `@/api/queries` barrel.

---

## Dependencies

### Upstream

- `fact_sales_monthly`, `dim_sku`, `dim_item`
- Feature 44 (Resilient Jobs) -- subprocess isolation, PID tracking, log streaming
- Feature 46 (Unified Model Tuning) -- `lgbm_tuning_run` table extended with cluster source FK
- `cluster_experiment` table (promoted row) -- Production Baseline template reads current production parameters

### Downstream

- All backtest scripts (`ml_cluster` feature)
- Safety stock (segment policies)
- Champion selection, production forecast
- ABC-XYZ classification

### Libraries

scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, MLflow

### Config References

- `config/forecasting/cluster_experiment_templates.yaml` -- preset experiment templates
- `config/forecasting/cluster_tuning_profiles.yaml` -- per-cluster hyperparameter overrides for backtests

---

## See Also

- [02-sku-feature-engineering](../03-demand-intelligence/02-sku-feature-engineering.md) -- Seasonality detection + demand variability profiling (complementary demand pattern analysis)
- [../02-forecasting/19-forecast-pipeline-config](../02-forecasting/19-forecast-pipeline-config.md) -- `cluster_strategy` config key
- [../07-user-experience/04-job-scheduler](../07-user-experience/04-job-scheduler.md) -- Background scenario execution
- [../02-forecasting/11-unified-model-tuning-v2](../02-forecasting/11-unified-model-tuning-v2.md) -- Algorithm tuning studio (cluster source selector)
- [../02-forecasting/13-production-baseline-seeding](../02-forecasting/13-production-baseline-seeding.md) -- Auto-seeds production baselines
