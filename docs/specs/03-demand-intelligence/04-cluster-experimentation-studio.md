# 03-04 Cluster Experimentation Studio

> **Status:** Planned | **Feature:** 47

A standalone experiment lifecycle for testing DFU segmentation configurations (create, run, compare, promote) with a one-directional connection to algorithm tuning: algorithm experiments can reference a cluster experiment, but cluster experiments know nothing about algorithm experiments.

| | |
|---|---|
| **UI Tab** | Clusters (new "Experiments" sub-tab) |
| **API Prefix** | `/cluster-experiments` |
| **Router** | `api/routers/forecasting/cluster_experiments.py` |
| **Frontend** | `frontend/src/api/queries/cluster-experiments.ts`, `frontend/src/tabs/clusters/ClusterExperimentsPanel.tsx` |
| **Tests** | `tests/api/test_cluster_experiments.py`, `frontend/src/tabs/__tests__/ClusterExperimentsPanel.test.tsx` |
| **Depends On** | 03-01 (DFU Clustering), Feature 46 (Unified Model Tuning Studio), Feature 44 (Resilient Jobs) |

---

## Table of Contents

1. [Problem](#problem)
2. [Solution Overview](#solution-overview)
3. [Architecture](#architecture)
4. [Database Schema](#database-schema)
5. [API Design](#api-design)
6. [Frontend Components](#frontend-components)
7. [Job Pipeline Changes](#job-pipeline-changes)
8. [Backtest Cluster Override](#backtest-cluster-override)
9. [Algorithm Tuning Integration](#algorithm-tuning-integration)
10. [Configuration](#configuration)
11. [Test Strategy](#test-strategy)
12. [Implementation Sequence](#implementation-sequence)
13. [Acceptance Criteria](#acceptance-criteria)

---

## Problem

Clustering and algorithm tuning are currently disconnected workflows. The existing What-If scenario engine (`/clustering/scenario`) produces cluster assignments but has no persistent tracking, no comparison infrastructure, and no mechanism to feed experimental clusters into algorithm backtests. Users must promote clusters blindly -- without knowing their accuracy impact on downstream models.

Specific gaps:

1. **No experiment lifecycle** -- Cluster scenarios run and produce artifacts but are not tracked in the database. There is no list view, no labeling, no history.
2. **No comparison** -- Two cluster configurations cannot be compared side by side. There is no migration matrix showing how DFUs move between cluster labels, no quality metric deltas.
3. **No accuracy feedback loop** -- When a user promotes a new clustering, there is no way to preview the accuracy impact on LGBM/CatBoost/XGBoost backtests before committing.
4. **No templates** -- Every scenario requires manually specifying all parameters. There are no preset starting points for common strategies (high-K granular, low-K broad, seasonal focus).

---

## Solution Overview

Two independent but connected systems:

| System | Scope | Table | UI Location |
|--------|-------|-------|-------------|
| **Cluster Experimentation Studio** | Standalone experiment lifecycle for segmentation configurations | `cluster_experiment` | ClustersTab > Experiments sub-tab |
| **Cluster Source Selector** | Algorithm experiments choose which cluster configuration to use | `lgbm_tuning_run` (extended) | ModelTuningTab > ExperimentBuilder |

Connection is **one-directional**: algorithm experiments can reference a cluster experiment via `cluster_experiment_id` FK, but cluster experiments know nothing about algorithm experiments. The `used-by` endpoint provides a reverse lookup for informational purposes only.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CLUSTER EXPERIMENTATION STUDIO                             │
│  (Independent, no model tied)                               │
│                                                             │
│  Create → Run → Compare → Promote                          │
│  Table: cluster_experiment, cluster_experiment_comparison   │
│  Router: cluster_experiments.py                             │
│  UI: ClustersTab > Experiments sub-tab                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ cluster_experiment_id (optional FK)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  ALGORITHM TUNING STUDIO (Feature 46)                       │
│  (Independent, its own lifecycle)                           │
│                                                             │
│  Cluster Source: Production | Experiment #X                 │
│  Create → Run → Compare → Promote                          │
│  Table: lgbm_tuning_run (extended with cluster_source)      │
│  Router: unified_model_tuning.py                            │
│  UI: ModelTuningTab > ExperimentBuilder                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
config/cluster_experiment_templates.yaml
         │
         ▼
┌──────────────────┐    POST /cluster-experiments     ┌──────────────┐
│  ExperimentBuilder├──────────────────────────────────►│ cluster_     │
│  (ClusterTab UI)  │                                  │ experiments  │
└──────────────────┘                                   │ .py router   │
                                                       └──────┬───────┘
                                                              │ submit cluster_scenario job
                                                              ▼
                                                       ┌──────────────┐
                                                       │ job_state.py │
                                                       │ _run_cluster │
                                                       │ _scenario()  │
                                                       └──────┬───────┘
                                                              │ Popen → run_clustering_scenario.py
                                                              ▼
                                                       ┌──────────────┐
                                                       │ Artifacts:   │
                                                       │ cluster_     │
                                                       │ labels.csv   │
                                                       │ + profiles   │
                                                       └──────┬───────┘
                                                              │ write results to DB
                                                              ▼
                                                       ┌──────────────┐
                                                       │ cluster_     │
                                                       │ experiment   │
                                                       │ table (DB)   │
                                                       └──────┬───────┘
                                                              │ optional: reference from
                                                              ▼ algorithm tuning
                                                       ┌──────────────┐
                                                       │ lgbm_tuning_ │
                                                       │ run.cluster_ │
                                                       │ experiment_id│
                                                       └──────────────┘
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
    total_dfus          INTEGER,
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

---

## API Design

### Router: `api/routers/forecasting/cluster_experiments.py`

Prefix: `/cluster-experiments` -- mounted in `api/main.py` before `domains.py`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/cluster-experiments` | List experiments (paginated, filterable by status) |
| GET | `/cluster-experiments/{id}` | Single experiment detail |
| POST | `/cluster-experiments` | Create + launch experiment |
| PATCH | `/cluster-experiments/{id}` | Update label/notes |
| DELETE | `/cluster-experiments/{id}` | Delete (409 if running/queued) |
| POST | `/cluster-experiments/{id}/promote` | Promote to production `dim_sku.ml_cluster` |
| GET | `/cluster-experiments/compare` | Compare two experiments (`?a_id=X&b_id=Y`) |
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

### Compare Response

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
    "total_dfus_migrated": 4500,
    "total_dfus_unchanged": 8200
}
```

Migration matrix computation: load both experiments' `cluster_labels.csv` from `artifacts_path`, join on `sku_ck`, compute pandas `crosstab`. Results are cached in `cluster_experiment_comparison` table.

### Algorithm Tuning API Changes

**Modified file:** `api/routers/forecasting/unified_model_tuning.py`

The `CreateExperimentBody.config` gains two new fields:

```python
cluster_source: str = "production"        # "production" | "experimental"
cluster_experiment_id: int | None = None  # FK to cluster_experiment (required when experimental)
```

When `cluster_source == "experimental"`:
- Validate `cluster_experiment_id` exists and has `status='completed'`
- Read `artifacts_path` from `cluster_experiment` table
- Build `cluster_override_path` into the temp `algorithm_config.yaml`
- Store `cluster_source` and `cluster_experiment_id` on `lgbm_tuning_run`

List/detail/compare responses include `cluster_source`, `cluster_experiment_id`, and `cluster_experiment_label` (joined from `cluster_experiment.label`).

---

## Frontend Components

### ClustersTab Enhancement

**Current sub-tabs:** Overview (implicit, single view)
**New sub-tabs:** Overview | **Experiments**

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

| # | Label | Status | K | Silhouette | DFUs | Duration | Created | Actions |

Row click uses the same baseline/candidate selection pattern as ModelTuningTab (blue for baseline, green for candidate, toggle to deselect).

**Right column -- Comparison panel** (appears when 2 rows selected, empty-state guidance otherwise)

**Empty state** (when no experiments exist):
- Icon + "No cluster experiments yet"
- Explanatory text: "Cluster experiments let you test different DFU segmentation configurations before committing to production. Each experiment runs the full clustering pipeline with your chosen parameters."
- Primary "Create First Experiment" button

#### `ClusterExperimentBuilder.tsx`

Full-screen modal matching ExperimentBuilder pattern:

1. **Label & Notes** -- text input + textarea
2. **Template Selection** -- radio button grid:
   - Production Baseline (reads `clustering_config.yaml` live)
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
4. **Estimate bar** -- shows estimated runtime + DFU count (from `/clustering/scenario/estimate`)
5. **Footer**: Cancel | Launch Experiment

**Clone support**: "Clone" action on each experiment row opens builder pre-populated with that experiment's params.

#### `ClusterComparisonPanel.tsx`

Sections (first two expanded, last two collapsed by default):

1. **Quality Metrics Header** -- two-column showing each experiment's silhouette, inertia, K, DFUs with delta badges (arrows, green/red)
2. **Cluster Profile Comparison Table** -- merged table showing clusters from both experiments with side-by-side count, percentage, and delta columns
3. **DFU Migration Matrix** -- ECharts heatmap showing how DFUs moved between cluster labels (rows = Exp A, cols = Exp B, cells = count with color intensity). Includes an accessible `<table>` alternative view toggle ("Table view" button) for screen readers. `overflow-x-auto` with 500px min width.
4. **K Selection Overlay** -- line chart overlaying both experiments' elbow curves
5. **Promote Button** -- for the candidate experiment

Heatmap cells: tooltip on hover "2,345 DFUs moved from [source] to [target]"

#### `ClusterPromoteModal.tsx`

Warning modal with:
- Downstream impact warnings (backtests, forecasts, inventory planning)
- Experiment details (K, silhouette, DFU count)
- Tip: "Re-run your algorithm experiments with the new clusters to see accuracy impact"
- Confirm / Cancel buttons

#### `ClusterParamsForm.tsx`

Reusable 3-column parameter form extracted for use by both `ClusterExperimentBuilder` and potential future clustering UI.

### Shared Utilities -- `frontend/src/components/shared-tuning-utils.tsx`

Extract from ModelTuningTab for reuse by both ClusterExperimentsPanel and ModelTuningTab:
- `StatusBadge` component
- `formatDuration()` helper
- `timeAgo()` helper
- `SortIndicator` component
- `KpiCard` component

### Cluster Source Selector in ExperimentBuilder

**Modified file:** `frontend/src/tabs/model-tuning/ExperimentBuilder.tsx`

In the Training Configuration section, add a "Cluster Source" selector as the first row:

```
Cluster Source                 | Cluster Strategy        | Recursive
[Production Clusters      v]  | [per_cluster        v]  | [x] Recursive
```

Dropdown design (shadcn/ui `Select`):
- **Production Clusters** (default) -- shows current production K and silhouette in secondary text
- Separator line
- **Completed cluster experiments** -- each showing: `[label] -- K=[n], Sil=[score]`

When experimental cluster selected:
- Info badge below dropdown: "Using clusters from experiment #3: High-K K=15"
- `config.cluster_source = "experimental"` and `config.cluster_experiment_id = 3` included in payload

When dropdown is empty (no completed experiments):
- Shows placeholder: "No cluster experiments yet"
- Link below: "Create one in the Clusters tab" (navigates to ClustersTab Experiments sub-tab)

Data fetching: `fetchCompletedClusterExperiments()` with `staleTime: 5 min`

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
    total_dfus: number | null;
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
promoteClusterExperiment(id) -> { status, dfus_updated }
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

## Job Pipeline Changes

### Reuse Existing `cluster_scenario` Job Type

No new job type registration needed. The existing `cluster_scenario` job type is reused. The `cluster_experiments.py` router creates the DB record in `cluster_experiment`, then submits the same `cluster_scenario` job with the `experiment_id` added to params.

### Modify `common/services/job_state.py` -- `_run_cluster_scenario`

After `run_scenario()` completes, also write to `cluster_experiment` table:
- Insert/update the row with results (optimal_k, silhouette, profiles, cluster_sizes, etc.)
- Set `status='completed'` or `status='failed'`
- Write `runtime_seconds` and `completed_at`

### Modify `common/services/job_state.py` -- `_run_model_tuning_experiment`

When params include `cluster_experiment_id`:
- Validate the experiment exists and has `status='completed'`
- Pass `--cluster-override {artifacts_path}/cluster_labels.csv` to the backtest subprocess

### Modify `scripts/run_clustering_scenario.py`

The `run_scenario()` function gains an optional `experiment_id` parameter. When provided, writes results to the `cluster_experiment` table on completion.

---

## Backtest Cluster Override

### `scripts/run_backtest.py` -- CLI Argument (~5 lines)

```python
parser.add_argument("--cluster-override", type=str, default=None,
    help="CSV with sku_ck,cluster_label to override dim_sku.ml_cluster")
```

Read from config if not on CLI:
```python
cluster_override = args.cluster_override or algo_config.get("cluster_override_path")
```

### `common/ml/backtest_framework.py` -- `load_backtest_data()` (~8 lines)

After loading `dfu_attrs` from DB, inject override:

```python
if cluster_override_path:
    override_df = pd.read_csv(cluster_override_path, usecols=["sku_ck", "cluster_label"])
    override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
    dfu_attrs["ml_cluster"] = dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])
    logger.info(f"Cluster override: {len(override_map):,} DFUs remapped")
```

Minimal, non-invasive. Only affects the in-memory DataFrame. All downstream per-cluster training, feature engineering, and metrics work unchanged because they reference `dfu_attrs["ml_cluster"]`.

---

## Algorithm Tuning Integration

The connection between the Cluster Experimentation Studio and the Algorithm Tuning Studio is one-directional and optional:

1. **Creating an algorithm experiment** -- The ExperimentBuilder UI shows a "Cluster Source" dropdown. Default is "Production Clusters". If completed cluster experiments exist, they appear in the dropdown.
2. **Running with experimental clusters** -- When `cluster_source=experimental`, the backtest subprocess receives `--cluster-override` pointing to the cluster experiment's `cluster_labels.csv`. The `ml_cluster` column in the in-memory DataFrame is remapped before training/prediction.
3. **Tracking lineage** -- `lgbm_tuning_run.cluster_source` and `cluster_experiment_id` record which cluster configuration was used. Responses include `cluster_experiment_label` via a LEFT JOIN.
4. **Deletion safety** -- `ON DELETE SET NULL` on the FK means deleting a cluster experiment does not delete algorithm experiments. The UI shows "Cluster experiment deleted" when `cluster_experiment_id` is NULL but `cluster_source` is `'experimental'`.

---

## Configuration

### File: `config/cluster_experiment_templates.yaml`

```yaml
templates:
  - id: production_baseline
    label: "Production Baseline"
    description: "Current production parameters from clustering_config.yaml"
    source: clustering_config
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

## Test Strategy

### Backend Tests

| Test File | Coverage |
|-----------|----------|
| `tests/api/test_cluster_experiments.py` | CRUD lifecycle (create, list, get, update, delete), compare endpoint (migration matrix, quality deltas), promote workflow, used-by reverse lookup, 409 on delete of running experiment, validation of completed status for cluster source |
| `tests/unit/test_backtest_cluster_override.py` | `--cluster-override` CLI arg parsing, `load_backtest_data()` override mechanism, partial override (some DFUs missing from CSV), empty override file |
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

## Implementation Sequence

### Phase 1: Schema + Backend Core

1. `sql/101_cluster_experiments.sql` -- create tables, alter `lgbm_tuning_run`
2. `api/routers/forecasting/cluster_experiments.py` -- CRUD + compare + promote + templates (10 endpoints)
3. `api/main.py` -- mount router before `domains.py`
4. `scripts/run_clustering_scenario.py` -- add `experiment_id` param, write to DB
5. `common/services/job_state.py` -- update `_run_cluster_scenario` to populate DB
6. `tests/api/test_cluster_experiments.py`

### Phase 2: Backtest Cluster Override

7. `scripts/run_backtest.py` -- `--cluster-override` CLI arg
8. `common/ml/backtest_framework.py` -- override in `load_backtest_data()`
9. `api/routers/forecasting/unified_model_tuning.py` -- `cluster_source` in config
10. `common/services/job_state.py` -- pass override to model tuning job
11. `tests/unit/test_backtest_cluster_override.py`
12. `tests/api/test_unified_model_tuning.py` -- cluster source tests

### Phase 3: Frontend -- Cluster Experiments

13. `frontend/src/api/queries/cluster-experiments.ts`
14. `frontend/vite.config.ts` -- add `/cluster-experiments` proxy
15. `frontend/src/components/shared-tuning-utils.tsx` -- extract shared components
16. `frontend/src/tabs/clusters/` -- all 5 new components
17. `frontend/src/tabs/ClustersTab.tsx` -- sub-tab navigation
18. Frontend tests (3 new test files)

### Phase 4: Frontend -- Cluster Source Selector

19. `frontend/src/tabs/model-tuning/ExperimentBuilder.tsx` -- cluster source dropdown
20. `frontend/src/tabs/ModelTuningTab.tsx` -- cluster source badge
21. `frontend/src/tabs/model-tuning/EnhancedComparisonPanel.tsx` -- config diffs
22. `frontend/src/api/queries/unified-model-tuning.ts` -- extend types
23. Frontend tests (extend ExperimentBuilder.test.tsx)

### Phase 5: Config

24. `config/cluster_experiment_templates.yaml`

**Total: 14 new files, 14 modified files**

---

## Acceptance Criteria

1. **Unit**: `--cluster-override` correctly remaps `ml_cluster` in backtest DataFrame
2. **API**: CRUD lifecycle for cluster experiments (create/list/get/update/delete/promote)
3. **API**: Compare endpoint returns migration matrix and quality deltas
4. **API**: Algorithm experiment with `cluster_source=experimental` creates correct temp config
5. **API**: Deleted cluster experiment results in `cluster_experiment_id=NULL` on algorithm experiments (FK SET NULL)
6. **Frontend**: Cluster Experiments sub-tab renders list, builder modal, comparison panel
7. **Frontend**: ExperimentBuilder cluster source dropdown shows completed experiments
8. **Frontend**: Empty dropdown shows "Create one in the Clusters tab" link
9. **Integration**: Create cluster experiment -> completes -> select in algorithm ExperimentBuilder -> run backtest with overridden clusters -> accuracy metrics reflect new clustering
10. **`make test-all`** passes with no regressions

---

## Dependencies

| Dependency | Why |
|------------|-----|
| 03-01 DFU Clustering | Reuses `run_scenario()` function, clustering pipeline, and artifacts format |
| Feature 44 Resilient Jobs | Subprocess isolation, PID tracking, log streaming for cluster experiments |
| Feature 46 Unified Model Tuning | Algorithm experiment table (`lgbm_tuning_run`) extended with cluster source FK |
| `config/clustering_config.yaml` | Production Baseline template reads current production parameters |

---

## Reusable Components (no new code needed)

- `scripts/run_clustering_scenario.py` -> `run_scenario()` -- already orchestrates clustering with custom params
- `api/routers/clusters.py` -> `promote_scenario()` logic -- reuse for cluster experiment promotion
- `cluster_scenario` job type -- existing job type reused (no new job registration needed)
- `config/clustering_config.yaml` -> defaults for cluster params
- All algorithm comparison/breakdown infrastructure -- works unchanged
