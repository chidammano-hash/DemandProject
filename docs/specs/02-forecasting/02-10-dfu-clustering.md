<!-- SOURCE: feature7.md (DFU Clustering Framework) -->
# Feature 7: DFU Clustering Framework

## Objective
Build a robust clustering framework that groups DFUs by historical demand patterns, enabling better global LGBM model performance through homogeneous training segments.

## Scope
- Dataset: `dim_dfu` (cluster assignments stored in `cluster_assignment` column)
- Data Sources: `fact_sales_monthly`, `dim_dfu`, `dim_item`
- Output: Cluster labels assigned to DFUs based on time series, item, and DFU features

## Architecture Overview

The clustering pipeline consists of:

1. **Feature Engineering**: Extract time series, item, and DFU features from historical sales data
2. **Clustering Model**: Apply clustering algorithms with optimal K selection
3. **Cluster Labeling**: Assign meaningful business labels to clusters
4. **Storage & Integration**: Update `dim_dfu.cluster_assignment` and log to MLflow

## Feature Engineering

Feature engineering covers **6 dimensions with 14 core features** extracted from historical sales data (default: 36-month window, minimum 12 months history).

### 14 Core Features (used for clustering)

| Dimension | Feature | Description |
|---|---|---|
| **Volume** | `mean_demand` | Average monthly demand (log-transformed before scaling) |
| | `cv_demand` | Coefficient of variation (std/mean) |
| | `iqr_demand` | Interquartile range — robust spread measure (log-transformed) |
| **Trend** | `trend_slope_norm` | Scale-invariant OLS slope: `slope * n_months / mean_demand` |
| | `trend_r2` | R-squared from linear regression (demand vs time index) |
| | `cagr` | Compound Annual Growth Rate (%) from first-year to last-year average |
| **Seasonality** | `seasonal_amplitude` | (max monthly mean - min monthly mean) / overall mean |
| | `seasonal_r2` | OLS R-squared from 11 monthly dummy variables (STL-lite) |
| | `yoy_correlation` | Pearson correlation of year-over-year demand patterns |
| **Periodicity** | `periodicity_strength` | FFT dominant non-DC component power / total power |
| **Intermittency** | `zero_demand_pct` | Fraction of months with zero demand |
| | `adi` | Croston Average Demand Interval — mean gap between non-zero months |
| **Lifecycle** | `months_available` | Number of months with sales history |
| | `recency_ratio` | Last-6-month average / full-history average (>1 = accelerating) |

### Additional Features (computed but not used in clustering)

**Volume (extended):** `median_demand`, `std_demand`, `min_demand`, `max_demand`, `total_demand`
**Trend (extended):** `trend_slope` (raw OLS slope), `trend_pct_change`, `trend_direction`
**Seasonality (extended):** `peak_month`, `seasonal_index_std`
**Volatility:** `sparsity_score`, `demand_stability`, `outlier_count`
**Growth:** `growth_rate`, `acceleration`, `recent_vs_historical`

### Item Features (from dim_item)
- Categorical: `category`, `class`, `sub_class`, `brand_name`, `country`, `national_service_model`
- Numeric: `case_weight`, `item_proof`, `bpc`, `bottle_pack`, `pack_case`

### DFU Features (from dim_dfu)
- Categorical: `brand`, `region`, `state_plan`, `sales_div`, `prod_cat_desc`, `abc_vol`, `service_lvl_grp`, `supergroup`
- Numeric: `execution_lag`, `total_lt`, `alcoh_pct`, `proof`

## Clustering Approach

### Primary Method: KMeans with Optimal K Selection

**K Selection Methods:**
1. Elbow Method (WCSS vs. K) — visual aid
2. Silhouette Score — cluster separation quality
3. Calinski-Harabasz Score — between/within variance ratio
4. **Combined score**: `0.5 * silhouette_norm + 0.5 * calinski_norm` (min-max normalized to [0,1])
5. **Hard 5% minimum cluster size constraint** — K values where any cluster < 5% of DFUs are rejected

**Optimal K Range**: 5-18 clusters (configurable via `k_range` in `clustering_config.yaml`)

> **Note:** Gap statistic was removed in favor of the combined Silhouette + Calinski-Harabasz scoring, which is faster and more robust for tree-model-friendly cluster sizing.

### Feature Scaling & Preprocessing
- Log-transform highly skewed volume features (`mean_demand`, `median_demand`, `std_demand`, `total_demand`, `max_demand`, `iqr_demand`, `adi`) via `log1p` before scaling
- StandardScaler (mean=0, std=1) on the 14 core features only
- Optional PCA for dimensionality reduction (disabled by default)
- `merge_small_clusters()` post-hoc: any cluster below the minimum size threshold is merged into its nearest large neighbor by centroid distance

## Cluster Labeling Strategy

Labels are assigned using a **priority-ordered taxonomy** that evaluates pattern characteristics from most distinctive to least:

**Priority Order:**
1. **Intermittency** — checked first (ADI > 1.5 or zero_demand_pct > 15%)
2. **Periodicity** — non-12-month cycles (periodicity_strength > 0.25)
3. **Seasonality** — amplitude + seasonal_r2 thresholds
4. **Trend** — trend_r2 + CAGR together (growing/declining)
5. **Volatility** — cv_demand thresholds (volatile/very_volatile)
6. **Volume tier** — always appended (5 tiers: very_high/high/medium/low/very_low based on percentiles)

**Volume Tiers (5 levels):**
- `very_high` (top 10%), `high` (top 25%), `medium` (25th-75th pctile), `low` (bottom 25%), `very_low` (bottom 10%)

**Example Compound Labels:**
- `high_volume_seasonal_growing`, `medium_volume_steady_seasonal`
- `low_volume_intermittent`, `high_volume_periodic`
- `very_high_volume_growing`, `medium_volume_volatile`
- `high_volume_declining`, `very_low_volume_intermittent`

**Disambiguation:** Two-pass approach — first assigns base labels from volume + pattern, then disambiguates any remaining duplicates using secondary features (recency_ratio for accelerating/decelerating, trend for growing/declining).

## Implementation Components

### 1. Feature Engineering Script
`mvp/demand/scripts/generate_clustering_features.py`
- Parameters: `--min-months` (default: 12), `--time-window` (default: 36), `--output`
- Output: `data/clustering_features.csv`
- Computes 14 core features across 6 dimensions (volume, trend, seasonality, periodicity, intermittency, lifecycle)
- New features (vs original): FFT periodicity, OLS seasonal R-squared, Croston ADI, scale-invariant trend slope, IQR, CAGR, recency ratio, YoY correlation

### 2. Clustering Model Script
`mvp/demand/scripts/train_clustering_model.py`
- Parameters: `--input`, `--k-range` (default: 5 18), `--min-cluster-size-pct` (default: 5.0), `--use-pca`, `--output-dir`
- Output: `cluster_assignments.csv`, `cluster_centroids.csv`, `cluster_metadata.json`, visualization PNGs
- Uses only 14 CORE_FEATURES for clustering (not all computed features)
- Log-transforms skewed volume features before StandardScaler
- Combined score: 0.5 * silhouette_norm + 0.5 * calinski_norm (Calinski-Harabasz replaces gap statistic)
- Hard 5% minimum cluster size constraint; post-hoc merge of small clusters
- MLflow: experiment `dfu_clustering` with params, metrics, artifacts

### 3. Cluster Labeling Script
`mvp/demand/scripts/label_clusters.py`
- Parameters: `--centroids`, `--assignments`, `--metadata`, `--output`, `--config`
- Output: `cluster_labels.csv`, `cluster_profiles.json`

### 4. Assignment Update Script
`mvp/demand/scripts/update_cluster_assignments.py`
- Parameters: `--input`, `--dry-run`
- Updates `dim_dfu.cluster_assignment` column in PostgreSQL

## Makefile Targets

```makefile
cluster-features:   # Generate clustering feature matrix
cluster-train:      # Train KMeans, select optimal K
cluster-label:      # Assign business labels
cluster-update:     # Write labels to dim_dfu
cluster-all:        # Full pipeline
```

## Configuration
File: `mvp/demand/config/clustering_config.yaml`

```yaml
clustering:
  min_months_history: 12
  time_window_months: 36           # 3 years for better seasonality detection
  k_range: [5, 18]                 # 5% min = max 20 clusters; start at 5 for business meaningfulness
  min_cluster_size_pct: 5.0        # CRITICAL: each cluster must be >= 5% of DFUs
  feature_scaling: "standard"
  use_pca: false

  labeling:
    volume_thresholds:
      very_high: 0.90              # top 10%
      high: 0.75                   # top 25%
      low: 0.25                    # bottom 25%
      very_low: 0.10               # bottom 10%
    cv_thresholds:
      very_steady: 0.2
      steady: 0.4
      volatile: 0.8
      very_volatile: 1.2
    seasonality_threshold: 0.3
    seasonality_r2_threshold: 0.25
    periodicity_threshold: 0.25
    zero_demand_threshold: 0.15
    adi_threshold: 1.5
    trend_r2_threshold: 0.25
    cagr_growing: 5.0
    cagr_declining: -5.0
    recency_ratio_high: 1.2
    recency_ratio_low: 0.8
```

## API Integration
- `GET /domains/dfu/clusters` — returns cluster summary with counts, avg_demand, cv_demand
- Existing filter mechanism supports filtering by `cluster_assignment`

## Database Schema
No schema changes — `dim_dfu.cluster_assignment` already exists (TEXT column).
Index: `idx_dim_dfu_cluster_assignment` in `sql/005_create_dim_dfu.sql`.

## Dependencies
- Feature 3 (`dim_item`, `dim_dfu`)
- Feature 4 (`fact_sales_monthly`)
- scikit-learn, pandas, scipy, matplotlib, seaborn, mlflow, pyyaml

---

## Implementation Details

### Additional API Endpoints (in `api/routers/clusters.py`)
- `GET /domains/dfu/clusters/profiles` — cluster profiles with centroid features from JSON
- `GET /domains/dfu/clusters/visualization/{image_name}` — clustering visualization PNGs
- `GET /clustering/defaults` — current default parameters from YAML config
- `GET /clustering/scenario/estimate` — runtime estimation (DFU count, K range, gap flag)
- `POST /clustering/scenario` — trial clustering with custom params (HTTP 202, async via JobManager)
- `GET /clustering/scenario/{id}/status` — poll execution status
- `GET /clustering/scenario/{id}` — retrieve completed scenario result
- `POST /clustering/scenario/{id}/promote` — promote to production (`dim_dfu.ml_cluster`)
- `GET /domains/dfu/seasonality-profiles` — distinct seasonality profiles with DFU counts

### Clusters Endpoint Enhancement
- `source` parameter: `ml` (ml_cluster column) or `source` (cluster_assignment column)
- Response includes `pct_of_total` per cluster

### Training Script Enhancements
- `CORE_FEATURES` (14 features across 6 dimensions): mean_demand, cv_demand, iqr_demand, trend_slope_norm, trend_r2, cagr, seasonal_amplitude, seasonal_r2, yoy_correlation, periodicity_strength, zero_demand_pct, adi, months_available, recency_ratio
- `LOG_TRANSFORM_FEATURES`: mean_demand, median_demand, std_demand, total_demand, max_demand, iqr_demand, adi
- `merge_small_clusters()` post-processing: merges clusters below min_cluster_size_pct into nearest large neighbor by centroid distance
- Combined K scoring: `0.5 * silhouette_norm + 0.5 * calinski_norm` (gap statistic removed)
- Hard 5% minimum cluster size constraint during K selection (K values violating the constraint are rejected)

### Integration
- `POST /clustering/scenario` delegates to APScheduler-powered `JobManager` (Feature 39)
- Pydantic models: `FeatureParams`, `ModelParams`, `LabelParams`, `ClusteringScenarioRequest`
- `MAX_DFUS_FOR_TRAINING = 20,000` — samples for training if DFU count exceeds threshold


---

## Examples

### Example: Run full clustering pipeline

```bash
make cluster-all
# cluster-features → data/clustering_features.csv  (18,432 DFUs × 42 features)
# cluster-train   → best K=7, silhouette=0.68 (MLflow: experiment dfu_clustering)
# cluster-label   → high_volume_steady(3102), seasonal_medium_volume(2841), ...
# cluster-update  → Updated 18,432 DFU rows in dim_dfu
```

### Example: Verify cluster distribution

```sql
SELECT cluster_assignment, COUNT(*) AS n_dfus
FROM dim_dfu GROUP BY 1 ORDER BY 2 DESC LIMIT 5;
-- high_volume_steady       | 3102
-- seasonal_medium_volume   | 2841
-- medium_volume_steady     | 1620
-- seasonal_high_volume     | 1847
-- intermittent_low_volume  |  562
```

### Example: Clustering config YAML

```yaml
# config/clustering_config.yaml
clustering:
  time_window_months: 36
  k_range: [5, 18]
  min_cluster_size_pct: 5.0
  feature_scaling: standard
  labeling:
    volume_thresholds: {very_high: 0.90, high: 0.75, low: 0.25, very_low: 0.10}
    cv_thresholds: {very_steady: 0.2, steady: 0.4, volatile: 0.8, very_volatile: 1.2}
    seasonality_threshold: 0.3
    seasonality_r2_threshold: 0.25
    periodicity_threshold: 0.25
    zero_demand_threshold: 0.15
    adi_threshold: 1.5
    trend_r2_threshold: 0.25
    cagr_growing: 5.0
    cagr_declining: -5.0
```

### Example: What-If scenario — test K=5

```bash
curl -s -X POST http://localhost:8000/clustering/scenario \
  -H "Content-Type: application/json" \
  -d '{"model_params": {"k_range": [5, 5]}, "feature_params": {}, "label_params": {}}' \
  | jq '{id, status}'
# {"id": "scen_20260228_143021", "status": "running"}
```


---

<!-- SOURCE: feature29.md (What-If Scenario UI) -->
# Feature 29 — What-If / Scenario UI for Clustering

## Overview

Add an interactive **What-If Scenarios** panel to the Clusters tab that lets users adjust clustering parameters, run trial configurations against the backend pipeline, compare results side-by-side, and optionally promote a chosen scenario to production. This transforms clustering from a static, CLI-only workflow into a self-service analytics experience.

## Problem

Today the clustering pipeline is entirely CLI-driven:

1. **No parameter visibility** — users see the final cluster table and static PNGs but have no way to understand how K, time window, feature set, or labeling thresholds produced those results
2. **No experimentation** — changing a single parameter (e.g., K range, PCA toggle, CV threshold) requires editing `clustering_config.yaml`, re-running `make cluster-all`, and waiting for the full pipeline
3. **No comparison** — there is no way to compare two configurations side-by-side (e.g., "K=5 with PCA" vs "K=8 without PCA") to evaluate trade-offs
4. **No interactive charts** — the K-selection data (`k_values`, `inertias`, `silhouette_scores`, `ch_scores`, `combined_scores`) is already returned by the `/domains/dfu/clusters/profiles` API but is displayed only as static PNGs; users cannot hover, zoom, or overlay multiple runs
5. **No labeling tuning** — the volume/CV/seasonality thresholds in the labeling step directly control business labels but can only be changed in YAML

## Goals

- Expose all meaningful clustering parameters in an interactive UI
- Let users run multiple trial scenarios without affecting production clusters
- Provide rich interactive charts (Recharts) replacing the static PNGs
- Enable side-by-side scenario comparison with visual and quantitative diffs
- Allow promotion of a selected scenario to production (write to `dim_dfu.ml_cluster`)

## Non-Goals (Out of Scope)

- Changing the clustering algorithm itself (remains KMeans)
- Real-time streaming of pipeline progress (polling is sufficient)
- Undo/rollback of promoted scenarios (manual re-run via CLI serves this)
- Persisting scenario history across browser sessions (in-memory only; MLflow already tracks runs)

---

## Architecture

### Data Flow

```
User adjusts sliders/toggles in What-If panel
        |
        v
POST /clustering/scenario  (new endpoint)
        |
        v
Backend runs pipeline stages in a temp directory:
  1. generate_clustering_features.py --time-window X --min-months Y --output /tmp/scenario_<id>/
  2. train_clustering_model.py --k-range MIN MAX --use-pca --min-cluster-size-pct Z ...
  3. label_clusters.py (with custom thresholds)
        |
        v
Returns scenario result JSON:
  - cluster_assignments (count per cluster)
  - cluster_centroids (feature means)
  - cluster_profiles (label + centroid features)
  - k_selection_results (for interactive charts)
  - metadata (silhouette, inertia, optimal_k, runtime)
        |
        v
Frontend renders interactive charts + summary table
        |
        v (optional)
POST /clustering/scenario/<id>/promote
        |
        v
Runs label + update_cluster_assignments against dim_dfu
```

### API Endpoints (New)

#### `POST /clustering/scenario`

Runs a trial clustering pipeline with user-specified parameters. Results are stored server-side in a temp directory keyed by scenario ID. Does **not** modify `dim_dfu`.

**Request body:**

```json
{
  "feature_params": {
    "time_window_months": 24,
    "min_months_history": 6
  },
  "model_params": {
    "k_range": [3, 10],
    "min_cluster_size_pct": 2.0,
    "use_pca": false,
    "pca_components": null,
    "all_features": false
  },
  "label_params": {
    "volume_high": 0.75,
    "volume_low": 0.25,
    "cv_steady": 0.3,
    "cv_volatile": 0.8,
    "seasonality_threshold": 0.5,
    "zero_demand_threshold": 0.2
  }
}
```

All fields are optional — omitted fields use the defaults from `clustering_config.yaml`.

**Response:**

```json
{
  "scenario_id": "sc_20260222_143022_a1b2",
  "status": "completed",
  "runtime_seconds": 14.3,
  "params": { "...merged params with defaults..." },
  "result": {
    "optimal_k": 6,
    "silhouette_score": 0.412,
    "inertia": 8341.2,
    "n_clusters": 6,
    "total_dfus": 4821,
    "cluster_sizes": { "0": 1203, "1": 892 },
    "k_selection_results": {
      "k_values": [3, 4, 5, 6, 7, 8, 9, 10],
      "inertias": [18432, 14221, 11003, 8341, 7102, 6244, 5811, 5503],
      "silhouette_scores": [0.31, 0.35, 0.39, 0.41, 0.38, 0.36, 0.33, 0.31],
      "ch_scores": [1200, 1400, 1500, 1450, 1380, 1320, 1260, 1210],
      "combined_scores": [0.2, 0.45, 0.78, 0.85, 0.72, 0.55, 0.35, 0.2],
      "feasible_mask": [true, true, true, true, true, true, true, true]
    },
    "profiles": [
      {
        "cluster_id": 0,
        "label": "high_volume_steady",
        "count": 1203,
        "pct_of_total": 24.95,
        "mean_demand": 387.2,
        "cv_demand": 0.18,
        "seasonality_strength": 0.12,
        "trend_slope": 0.003,
        "growth_rate": 1.2,
        "zero_demand_pct": 0.01
      }
    ],
    "feature_importance": [
      { "feature": "mean_demand", "variance_ratio": 0.34 },
      { "feature": "cv_demand", "variance_ratio": 0.22 }
    ]
  }
}
```

#### `GET /clustering/scenario/<scenario_id>`

Retrieve a previously run scenario result. Returns 404 if the temp directory has been cleaned up.

#### `POST /clustering/scenario/<scenario_id>/promote`

Promotes a scenario to production:
1. Copies the scenario's `cluster_labels.csv` to `data/clustering/`
2. Runs `update_cluster_assignments.py` to write labels to `dim_dfu.ml_cluster`
3. Refreshes the cluster summary cache

**Response:**

```json
{
  "status": "promoted",
  "scenario_id": "sc_20260222_143022_a1b2",
  "dfus_updated": 4821,
  "cluster_distribution": { "high_volume_steady": 1203 }
}
```

#### `GET /clustering/defaults`

Returns the current default parameter values from `clustering_config.yaml` so the UI can populate sliders with baseline values.

**Response:**

```json
{
  "feature_params": {
    "time_window_months": 24,
    "min_months_history": 1
  },
  "model_params": {
    "k_range": [3, 12],
    "min_cluster_size_pct": 2.0,
    "use_pca": false,
    "pca_components": null,
    "all_features": false
  },
  "label_params": {
    "volume_high": 0.75,
    "volume_low": 0.25,
    "cv_steady": 0.3,
    "cv_volatile": 0.8,
    "seasonality_threshold": 0.5,
    "zero_demand_threshold": 0.2
  }
}
```

---

## UI Design

### Panel Location

The What-If panel lives within the existing **Clusters tab** (`Cl` element tile), rendered below the current cluster summary table and visualization section. It is collapsed by default behind a disclosure button: **"What-If Scenarios"**.

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Cl  Clusters                                                          │
│─────────────────────────────────────────────────────────────────────────│
│  Source: [ML Pipeline ▾]    Cluster: [All ▾]                           │
│  6 clusters, 4821 DFUs assigned                                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Cluster │ DFUs │   %  │ Avg Demand │  CV  │                     │   │
│  │─────────┼──────┼──────┼────────────┼──────│                     │   │
│  │ high_v… │ 1203 │ 24.9 │    387.2   │ 0.18 │  ← existing table  │   │
│  │ ...     │      │      │            │      │                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  K = 6   Silhouette = 0.4120   Inertia = 8341  ← existing badges      │
│                                                                         │
│  ▼ What-If Scenarios ─────────────────────────────────────────────────  │
│                                                                         │
│  ┌──────────────── Parameter Controls ─────────────────────────────┐   │
│  │                                                                  │   │
│  │  DATA SCOPE                                                      │   │
│  │  Time Window (months)   [══════●══════] 24                       │   │
│  │  Min History (months)   [●═════════════]  1                      │   │
│  │                                                                  │   │
│  │  MODEL                                                           │   │
│  │  K Range                [══●═══════●══] 3 – 12                   │   │
│  │  Min Cluster Size (%)   [══●══════════] 2.0                      │   │
│  │  Use PCA                [ ] Off                                  │   │
│  │  PCA Components         [══════════●══] auto  (disabled if off)  │   │
│  │  Skip Gap Statistic     [✓] On                                   │   │
│  │  Feature Set            (●) Core 8   ( ) All Features            │   │
│  │                                                                  │   │
│  │  LABELING THRESHOLDS                                             │   │
│  │  Volume High (pctl)     [═══════════●═] 0.75                     │   │
│  │  Volume Low (pctl)      [══●══════════] 0.25                     │   │
│  │  CV Steady (<)          [═══●═════════] 0.30                     │   │
│  │  CV Volatile (>)        [═══════════●═] 0.80                     │   │
│  │  Seasonality Threshold  [═══════●═════] 0.50                     │   │
│  │  Zero Demand Threshold  [═══●═════════] 0.20                     │   │
│  │                                                                  │   │
│  │  [  Reset to Defaults  ]          [ ▶ Run Scenario ]             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────── Scenario Results ───────────────────────────────┐   │
│  │                                                                  │   │
│  │  Scenario A (14.3s)              vs    Scenario B (11.8s)        │   │
│  │  K=6  Sil=0.412  Inertia=8341         K=5  Sil=0.391  Ine=11003│   │
│  │                                                                  │   │
│  │  ┌─ K-Selection Chart (interactive Recharts) ─────────────────┐ │   │
│  │  │                                                             │ │   │
│  │  │   Elbow (WCSS)           Silhouette Score                  │ │   │
│  │  │   ┌──────────┐           ┌──────────┐                      │ │   │
│  │  │   │  ╲       │           │    ╱╲    │     ── Scenario A    │ │   │
│  │  │   │   ╲__    │           │   ╱  ╲   │     -- Scenario B    │ │   │
│  │  │   │      ╲___│           │__╱    ╲__│                      │ │   │
│  │  │   └──────────┘           └──────────┘                      │ │   │
│  │  │   3  4  5  6  7  8       3  4  5  6  7  8                  │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  ┌─ Cluster Profile Radar Chart ──────────────────────────────┐ │   │
│  │  │          mean_demand                                        │ │   │
│  │  │             ╱╲                                              │ │   │
│  │  │  zero_pct ╱    ╲ cv_demand                                 │ │   │
│  │  │          ╱  C0  ╲                                           │ │   │
│  │  │ growth ─┤   C1   ├─ seasonality     (one polygon per       │ │   │
│  │  │          ╲  C2  ╱    cluster, hover to highlight)           │ │   │
│  │  │           ╲    ╱                                            │ │   │
│  │  │            ╲╱                                               │ │   │
│  │  │         trend_slope                                         │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  ┌─ Cluster Size Distribution (bar chart) ────────────────────┐ │   │
│  │  │  ████████████████  high_volume_steady (1203, 25%)          │ │   │
│  │  │  ████████████      medium_volume_seasonal (892, 19%)       │ │   │
│  │  │  ██████████        low_volume_intermittent (714, 15%)      │ │   │
│  │  │  █████████         medium_volume_steady (683, 14%)         │ │   │
│  │  │  ████████          high_volume_growing (641, 13%)          │ │   │
│  │  │  ███████           low_volume_declining (688, 14%)         │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  ┌─ Comparison Table ─────────────────────────────────────────┐ │   │
│  │  │ Metric           │ Scenario A │ Scenario B │  Delta        │ │   │
│  │  │──────────────────┼────────────┼────────────┼───────────────│ │   │
│  │  │ Optimal K        │     6      │     5      │  -1           │ │   │
│  │  │ Silhouette       │   0.412    │   0.391    │  -0.021 ▼     │ │   │
│  │  │ Inertia          │   8,341    │  11,003    │  +2,662 ▲     │ │   │
│  │  │ Total DFUs       │   4,821    │   4,821    │  —            │ │   │
│  │  │ Largest Cluster  │  25.0%     │  31.2%     │  +6.2pp       │ │   │
│  │  │ Smallest Cluster │  13.1%     │  14.8%     │  +1.7pp       │ │   │
│  │  │ Runtime          │  14.3s     │  11.8s     │  -2.5s        │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  [ ★ Promote Scenario A ]  [ ★ Promote Scenario B ]             │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Parameter Controls — Detail

#### Data Scope Section

| Control | Type | Range | Default | Step | Description |
|---------|------|-------|---------|------|-------------|
| Time Window (months) | Slider | 6 – 60, plus "All" toggle | 24 | 6 | Sales lookback window for feature engineering |
| Min History (months) | Slider | 1 – 24 | 1 | 1 | Minimum months of sales data required to include a DFU |

#### Model Section

| Control | Type | Range | Default | Step | Description |
|---------|------|-------|---------|------|-------------|
| K Range Min | Dual-thumb slider | 2 – 20 | 3 | 1 | Lower bound of K search |
| K Range Max | Dual-thumb slider | 2 – 20 | 12 | 1 | Upper bound of K search (must be > min) |
| Min Cluster Size (%) | Slider | 0.5 – 10.0 | 2.0 | 0.5 | Clusters smaller than this % get merged |
| Use PCA | Toggle switch | on/off | off | — | Enable PCA dimensionality reduction before KMeans |
| PCA Components | Slider | 2 – 8, plus "auto" | auto | 1 | Number of PCA components (disabled when PCA is off) |
| Skip Gap Statistic | Toggle switch | on/off | off | — | Skip gap stat calculation (faster but less info) |
| Feature Set | Radio group | Core 8 / All | Core 8 | — | Core 8 features vs all numeric features |

#### Labeling Thresholds Section

| Control | Type | Range | Default | Step | Description |
|---------|------|-------|---------|------|-------------|
| Volume High Percentile | Slider | 0.50 – 0.95 | 0.75 | 0.05 | Centroid mean_demand above this percentile = "high_volume" |
| Volume Low Percentile | Slider | 0.05 – 0.50 | 0.25 | 0.05 | Centroid mean_demand below this percentile = "low_volume" |
| CV Steady Threshold | Slider | 0.1 – 0.5 | 0.30 | 0.05 | CV below this = "steady" pattern |
| CV Volatile Threshold | Slider | 0.5 – 1.5 | 0.80 | 0.05 | CV above this = "volatile" pattern |
| Seasonality Threshold | Slider | 0.1 – 1.0 | 0.50 | 0.05 | seasonality_strength above this = "seasonal" |
| Zero Demand Threshold | Slider | 0.05 – 0.50 | 0.20 | 0.05 | zero_demand_pct above this = "intermittent" |

#### Validation Rules

- `k_range[0] < k_range[1]` (min < max)
- `volume_low < volume_high`
- `cv_steady < cv_volatile`
- `pca_components` disabled when `use_pca` is false
- Display inline validation errors below controls when violated

### Scenario Slots

The UI maintains up to **2 scenario slots** (A and B) for comparison:

- **First run** populates Slot A
- **Second run** populates Slot B (Slot A preserved for comparison)
- **Third run** replaces the oldest slot (user chooses which to replace, or auto-replaces A)
- Each slot displays a compact summary badge: `Scenario A: K=6, Sil=0.41, 14.3s`
- Active slot is highlighted; both are always visible in the comparison section

### Interactive Charts

All charts use **Recharts** (consistent with the rest of the app) and support:
- Hover tooltips with exact values
- Theme-aware colors via `CHART_COLORS[theme]`
- Legend toggle to show/hide individual series
- Responsive container for window resizing

#### 1. K-Selection Chart (replaces static PNG)

A **2-panel** line chart (or 3-panel if gap stats are available):

**Left panel — Elbow (WCSS/Inertia):**
- X-axis: K values
- Y-axis: Inertia (WCSS)
- Line per scenario (solid for A, dashed for B)
- Vertical dashed line at each scenario's optimal K
- Tooltip: `K=6, Inertia=8,341`

**Right panel — Silhouette Score:**
- X-axis: K values
- Y-axis: Silhouette score (0–1)
- Line per scenario
- Highlighted peak point (optimal K)
- Tooltip: `K=6, Silhouette=0.412`

#### 2. Cluster Profile Radar Chart

A **Recharts RadarChart** with one polygon per cluster:

- Axes: `mean_demand`, `cv_demand`, `seasonality_strength`, `trend_slope`, `growth_rate`, `zero_demand_pct`
- Values normalized to 0–1 range (min-max across all clusters in the scenario)
- One colored polygon per cluster (colors from `MODEL_COLORS` or `tab10`-style palette)
- Hover highlights individual cluster polygon and shows centroid values
- Legend: cluster labels with color swatches
- If comparing two scenarios: a tab/toggle to switch between Scenario A and B radar views

#### 3. Cluster Size Distribution (Horizontal Bar Chart)

A **Recharts BarChart** (horizontal layout):

- Y-axis: Cluster labels (e.g., `high_volume_steady`)
- X-axis: Count (or percentage of total)
- Bars colored by cluster
- If comparing: grouped bars (A and B side-by-side per cluster label)
- Tooltip: `high_volume_steady: 1,203 DFUs (25.0%)`

#### 4. Comparison Table

A standard HTML table (matching existing project table style):

| Metric | Scenario A | Scenario B | Delta |
|--------|-----------|-----------|-------|
| Optimal K | 6 | 5 | -1 |
| Silhouette Score | 0.4120 | 0.3910 | -0.0210 (red down arrow) |
| Inertia | 8,341 | 11,003 | +2,662 (red up arrow — lower is better) |
| Total DFUs | 4,821 | 4,821 | — |
| Largest Cluster % | 25.0% | 31.2% | +6.2pp |
| Smallest Cluster % | 13.1% | 14.8% | +1.7pp |
| Balance (std of sizes) | 184 | 211 | +27 |
| Runtime | 14.3s | 11.8s | -2.5s |

Delta coloring: green = better, red = worse, gray = neutral. Direction depends on the metric (lower inertia is better, higher silhouette is better).

### Loading State

When a scenario is running, the **Run Scenario** button shows a spinner and the results area displays a `LoadingElement` component (existing chemistry-themed periodic table tile with pulse-glow animation) with the message: **"Running clustering scenario..."**

Estimated runtime varies by parameters:
- Core features, skip gap: ~10–20s
- Core features, with gap: ~30–60s
- All features, with gap: ~60–120s

The UI displays an estimated time range based on the selected options.

### Promote Flow

When the user clicks **"Promote Scenario X"**:

1. A confirmation dialog appears:
   ```
   ┌─────────────────────────────────────────────┐
   │  Promote Scenario A to Production?           │
   │                                               │
   │  This will update ml_cluster for 4,821 DFUs  │
   │  in dim_dfu with the following configuration: │
   │                                               │
   │  K = 6 | Silhouette = 0.412                   │
   │  Time Window = 24 months                      │
   │  Features = Core 8 | PCA = Off                │
   │                                               │
   │         [ Cancel ]    [ Promote ]              │
   └─────────────────────────────────────────────┘
   ```

2. On confirm, `POST /clustering/scenario/<id>/promote` is called
3. Success toast: "Cluster assignments updated for 4,821 DFUs"
4. The main cluster summary table (above the What-If panel) auto-refreshes to reflect the new assignments
5. MLflow run is logged with scenario parameters + "promoted" tag

---

## State Management

### Frontend State (within Clusters tab)

```typescript
// What-If panel state
interface ScenarioParams {
  feature_params: {
    time_window_months: number | "all";
    min_months_history: number;
  };
  model_params: {
    k_range: [number, number];
    min_cluster_size_pct: number;
    use_pca: boolean;
    pca_components: number | null;
    all_features: boolean;
  };
  label_params: {
    volume_high: number;
    volume_low: number;
    cv_steady: number;
    cv_volatile: number;
    seasonality_threshold: number;
    zero_demand_threshold: number;
  };
}

interface ScenarioResult {
  scenario_id: string;
  status: "running" | "completed" | "failed";
  runtime_seconds: number;
  params: ScenarioParams;
  result: {
    optimal_k: number;
    silhouette_score: number;
    inertia: number;
    n_clusters: number;
    total_dfus: number;
    cluster_sizes: Record<string, number>;
    k_selection_results: {
      k_values: number[];
      inertias: number[];
      silhouette_scores: number[];
      ch_scores?: number[];
      combined_scores?: number[];
      feasible_mask?: boolean[];
    };
    profiles: ClusterProfile[];
    feature_importance: { feature: string; variance_ratio: number }[];
  } | null;
  error?: string;
}

// Component state
const [whatIfExpanded, setWhatIfExpanded] = useState(false);
const [scenarioParams, setScenarioParams] = useState<ScenarioParams>(defaults);
const [scenarioA, setScenarioA] = useState<ScenarioResult | null>(null);
const [scenarioB, setScenarioB] = useState<ScenarioResult | null>(null);
const [runningScenario, setRunningScenario] = useState(false);
const [defaults, setDefaults] = useState<ScenarioParams | null>(null);
```

### Backend State

- Scenario results stored in temp directories: `/tmp/clustering_scenario_<id>/`
- Each directory contains: `clustering_features.csv`, `cluster_assignments.csv`, `cluster_centroids.csv`, `cluster_metadata.json`, `cluster_labels.csv`, `cluster_profiles.json`
- Temp directories cleaned up after 1 hour (configurable) or on server restart
- No database writes until promote

---

## Interaction Flows

### Flow 1: First Scenario Run

1. User expands "What-If Scenarios" panel
2. UI fetches `GET /clustering/defaults` and populates all controls
3. User adjusts parameters (e.g., changes K range to 4–8, enables PCA)
4. User clicks "Run Scenario"
5. UI sends `POST /clustering/scenario` with current params
6. Loading state shown with `LoadingElement`
7. Response received — Scenario A populated
8. K-selection chart, radar chart, bar chart, and summary table rendered
9. No comparison shown (only one scenario exists)

### Flow 2: Comparison Run

1. With Scenario A already populated, user adjusts parameters
2. User clicks "Run Scenario" again
3. Response populates Scenario B
4. All charts now show both scenarios overlaid
5. Comparison table appears with deltas
6. "Promote" buttons appear for both scenarios

### Flow 3: Labeling-Only Rerun

Some parameters only affect labeling (volume/CV/seasonality/zero-demand thresholds) and do not require retraining the model. The UI detects this:

1. If only `label_params` changed (model_params and feature_params unchanged from the last run):
   - UI sends the request with a `relabel_only: true` flag
   - Backend skips feature generation and model training
   - Only re-runs `label_clusters.py` with new thresholds on the existing centroids
   - Response time: < 1 second (vs 10–120s for full run)
2. UI shows a badge: "Relabel only (instant)" on the results

### Flow 4: Promote

1. User clicks "Promote Scenario A"
2. Confirmation dialog shown with scenario summary
3. User confirms
4. `POST /clustering/scenario/<id>/promote` called
5. Backend writes to `dim_dfu.ml_cluster`
6. Success toast shown
7. Main cluster summary table refreshes
8. Promoted scenario badge changes to "Active"

---

## Backend Implementation Notes

### Scenario Runner

The backend scenario endpoint should:

1. Create a unique temp directory: `/tmp/clustering_scenario_<uuid>/`
2. Call the existing scripts as Python functions (not subprocess) where possible, with overridden output paths and parameters
3. Capture all output artifacts (CSVs, JSONs) in the temp dir
4. Return the aggregated result as JSON (no PNGs — the UI renders interactive charts)
5. Log the run to MLflow under experiment `dfu_clustering_whatif` with a `scenario_id` tag

### Concurrency

- Only one scenario can run at a time per server (clustering is CPU-intensive)
- If a second request arrives while one is running, return `409 Conflict` with estimated remaining time
- The UI disables the "Run Scenario" button while a run is in progress

### Relabel Shortcut

When `relabel_only: true`:
- Read existing centroids from the previous scenario's temp dir (or from `data/clustering/cluster_centroids.csv` if no scenario specified)
- Apply labeling logic with new thresholds
- Return updated profiles without re-running feature generation or model training

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mvp/demand/api/main.py` | Modify | Add `/clustering/scenario`, `/clustering/scenario/<id>`, `/clustering/scenario/<id>/promote`, `/clustering/defaults` endpoints |
| `mvp/demand/frontend/src/App.tsx` | Modify | Add What-If panel to Clusters tab with parameter controls, scenario slots, charts, comparison table, promote flow |
| `mvp/demand/scripts/run_clustering_scenario.py` | Create | Scenario runner that orchestrates feature gen + training + labeling in a temp dir |
| `mvp/demand/config/clustering_config.yaml` | No change | Read by `/clustering/defaults` endpoint; not modified by scenarios |
| `docs/design-specs/feature29.md` | Create | This spec |
| `docs/design-specs/feature1.md` | Modify | Add Feature 29 to implemented features list |
| `CLAUDE.md` | Modify | Add scenario endpoints and What-If UI to relevant sections |

## Dependencies

No new npm packages required. Uses existing:
- **Recharts** — `RadarChart`, `LineChart`, `BarChart` (already in the project)
- **shadcn/ui** — `Card`, `Badge`, `Button`, slider/toggle if available (or plain `<input type="range">`)
- **Tailwind CSS** — all styling via existing semantic tokens

Backend: no new Python packages. Uses existing `scikit-learn`, `pandas`, `yaml`, `mlflow`.

## Testing & Validation

### Manual Testing Checklist

1. **Defaults load** — expanding the panel populates all controls from `clustering_config.yaml` values
2. **Validation** — setting K min > K max shows inline error, "Run" button disabled
3. **First run** — produces Scenario A with all charts and summary table
4. **Second run** — produces Scenario B; comparison table and overlaid charts appear
5. **Relabel shortcut** — changing only labeling thresholds and running completes in < 1s
6. **Promote** — confirmation dialog shows, DFUs updated, main cluster table refreshes
7. **Concurrent block** — running two scenarios simultaneously returns 409 on the second
8. **Theme support** — all charts and controls render correctly in Light, Dark, and Midnight themes
9. **Reset** — "Reset to Defaults" button restores all controls to config values
10. **Error handling** — if the pipeline fails (e.g., insufficient data), error message shown in results area

### Automated Tests

- API unit tests for `/clustering/scenario` with mock pipeline
- API unit tests for `/clustering/defaults` returning correct YAML values
- Frontend: verify parameter controls render and update state
- Frontend: verify chart data transformation from API response to Recharts props

## Performance Considerations

- **Feature generation** is the slowest step (~5–15s) due to SQL queries and joins. Consider caching the base feature matrix and only regenerating when `time_window_months` or `min_months_history` changes.
- **Combined Silhouette + CH scoring** is fast (~0.002s per DFU per K) since it reuses the same KMeans fit.
- **Relabel shortcut** avoids the expensive steps entirely when only thresholds change.
- **Chart rendering** — Recharts handles up to 20 clusters and 20 K values without performance issues. No virtualization needed.

## Future Enhancements (Out of Scope for Feature 29)

1. **Algorithm selection** — add DBSCAN, Agglomerative, Gaussian Mixture as alternatives to KMeans
2. **Feature importance visualization** — SHAP-style feature contribution per cluster
3. **Scenario persistence** — save scenarios to database for cross-session comparison
4. **Auto-tune** — backend runs a grid of parameter combinations and returns the Pareto-optimal set
5. **Cluster stability analysis** — bootstrap resampling to show how stable cluster assignments are across random seeds
6. **Export** — download scenario results as CSV/PDF report
7. **DFU preview** — click a cluster in results to see sample DFUs and their sales time series

---

## Implementation Corrections

### Async Execution (major change from spec)
- `POST /clustering/scenario` returns **HTTP 202 Accepted** and runs asynchronously via `JobManager` (Feature 39)
- Response includes `job_id` for tracking in Jobs tab
- Frontend polls `GET /clustering/scenario/{id}/status` every 3 seconds

### Additional Endpoints (not in original spec)
- `GET /clustering/scenario/{id}/status` — returns `running` + `elapsed_seconds` or `completed`/`failed` with full result
- `GET /clustering/scenario/estimate` — runtime estimation based on DFU count, K range, gap flag

### File Location
- Endpoints in `api/routers/clusters.py` (not `api/main.py`), mounted via `include_router`

### Additional Request Fields
- `relabel_only: bool = False`
- `previous_scenario_id: str | None = None`

### Additional Response Fields
- `training_sample_size`, `sampled` (bool), `job_id`

### Large Dataset Sampling
- `MAX_DFUS_FOR_TRAINING = 20,000` — samples for training but predicts all DFUs

### Chart Differences
- Silhouette: **BarChart** with per-bar colors and quality zone ReferenceLines (not LineChart)
- Cluster sizes: **PieChart** (not horizontal BarChart)
- Feature importance: horizontal BarChart (top 10 features by variance ratio)
- Combined score: conditional BarChart (green=feasible, red=penalized, only when combined_scores present)

### Cross-Tab Notifications
- `ScenarioNotificationContext` (`context/ScenarioNotificationContext.tsx`) with `startScenario`, `completeScenario`, `failScenario`, `dismissNotification`
- Dashboard injects completion alert

### Backend State Directory
- `/tmp/clustering_scenarios/<scenario_id>/` (note plural "scenarios")

### Test Files
- `tests/unit/test_scenario_runner.py` (5 tests)
- `tests/api/test_clustering_scenario.py` (12 tests)
- `frontend/src/tabs/__tests__/ClustersTab.test.tsx` (3 tests)
- `frontend/src/tabs/__tests__/WhatIfScenarios.test.tsx` (7 tests)
- `frontend/src/context/__tests__/ScenarioNotificationContext.test.tsx` (4 tests)

---

## Examples

### Example: Full What-If scenario workflow

```bash
# Step 1: Get runtime estimate
curl -s "http://localhost:8000/clustering/scenario/estimate?k_max=8&gap=true" | jq .
# {"estimated_seconds": 72, "dfu_count": 18432, "k_range": 5}

# Step 2: Submit scenario (202 Accepted — runs in background)
SCENARIO_ID=$(curl -s -X POST http://localhost:8000/clustering/scenario \
  -H "Content-Type: application/json" \
  -d '{"model_params": {"k_range": [3, 8]}, "feature_params": {}, "label_params": {}}' \
  | jq -r '.id')
echo $SCENARIO_ID
# scen_20260228_143021

# Step 3: Poll for completion
curl -s "http://localhost:8000/clustering/scenario/$SCENARIO_ID/status" | jq .
# {"status": "completed", "elapsed_seconds": 68, "best_k": 6, "silhouette": 0.71}

# Step 4: Promote to production
curl -s -X POST "http://localhost:8000/clustering/scenario/$SCENARIO_ID/promote"
# {"promoted": true, "dfus_updated": 18432}
```

### Example: TypeScript status polling with TanStack Query

```typescript
const { data: statusData } = useQuery({
  queryKey: ['scenario-status', scenarioId],
  queryFn: () => fetchScenarioStatus(scenarioId),
  refetchInterval: (data) =>
    data?.status === 'running' ? 3000 : false,  // poll every 3s while running
  enabled: !!scenarioId,
})
```

### Example: Scenario queueing (when clustering group is busy)

```bash
# Second scenario submitted while first is running → status: "queued"
curl -s -X POST http://localhost:8000/clustering/scenario \
  -d '{"model_params": {"k_range": [4, 6]}, ...}' | jq .status
# "queued"  ← auto-dispatched when active job completes
```


---

<!-- SOURCE: feature38.md (Scenario Enhancements) -->
# Feature 38: Clustering What-If Scenario Enhancements — Background Execution, Runtime Estimation, Dashboard Alerts, Enhanced Charts

## Executive Summary

Feature 38 enhances the Clustering What-If Scenarios panel (Feature 29) with four improvements:
1. **Runtime Estimation** — display approximate execution time before running a scenario
2. **Background Execution** — non-blocking async POST with polling, so users can navigate away
3. **Dashboard Alerts** — notification on the Dashboard tab when a scenario completes
4. **Enhanced Charts** — richer elbow/silhouette/feature importance visualizations with quality indicators

## Key Features

- **Estimate Endpoint:** `GET /clustering/scenario/estimate` — returns estimated runtime based on DFU count, K range, and gap statistic flag
- **Async POST:** `POST /clustering/scenario` now returns HTTP 202 immediately with `scenario_id` and runs in background
- **Status Polling:** `GET /clustering/scenario/{id}/status` — returns `running` (with elapsed time), `completed` (with full result), or `failed`
- **ScenarioNotificationContext** — React context for cross-tab scenario state tracking
- **Dashboard Alert Injection** — scenario completion alert prepended to Dashboard AlertPanel with dismiss support
- **Enhanced Elbow Chart** — optimal K reference line with marker
- **Enhanced Silhouette Chart** — bar chart with quality zone thresholds (Strong/Reasonable/Weak/No structure), color-coded bars
- **Feature Importance Chart** — horizontal bar chart showing top 10 features by variance ratio
- **Cluster Size Pie Chart** — replaces basic bar chart with labeled pie chart
- **Gap Statistic Chart** — conditional line chart when gap stats are available

---

## API Endpoints

### `GET /clustering/scenario/estimate`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `k_min` | int | 3 | Min K value |
| `k_max` | int | 12 | Max K value |

**Response:**
```json
{ "estimated_seconds": 45, "dfu_count": 1200, "k_range": 10 }
```

### `POST /clustering/scenario` (Updated)

Now returns **HTTP 202** immediately instead of blocking:
```json
{ "scenario_id": "sc_20260226_abc1", "status": "running" }
```

### `GET /clustering/scenario/{id}/status`

**While running:**
```json
{ "scenario_id": "sc_123", "status": "running", "elapsed_seconds": 12 }
```

**When complete:**
```json
{ "scenario_id": "sc_123", "status": "completed", "runtime_seconds": 45, "result": { ... } }
```

---

## UI Changes

### ClustersTab
- **Estimate badge** next to Run button: "Est. ~45s (1.2K DFUs)"
- **Running indicator** with elapsed time and spinner animation
- **Background polling** every 3s via TanStack Query `refetchInterval`

### Dashboard AlertPanel
- Scenario completion alert with FlaskConical icon and dismiss button
- Alert text: "Scenario {label} Complete — finished in {time}s. View results in Clusters tab."

### Enhanced ScenarioCharts
- Elbow: ReferenceLine at optimal K
- Silhouette: Bar chart with quality zone reference lines, Cell color-coding
- Feature Importance: Horizontal bar chart (top 10)
- Cluster Size: PieChart with labels
- Gap Statistic: Conditional LineChart with optimal K marker

---

## Testing

### Backend Tests (14 total in test_clustering_scenario.py)
- 7 existing tests (updated: POST now returns 202)
- 7 new tests: estimate (3), status (3), conflict (1)

### Frontend Tests
- ScenarioNotificationContext: 4 tests (defaults, start, complete, dismiss)
- ClustersTab: 3 tests (smoke, cluster summary, what-if section)
- DashboardTab: 4 tests (updated with ScenarioNotificationProvider)

---

## Files

| File | Action |
|------|--------|
| `api/routers/clusters.py` | Edited — estimate endpoint, async POST, status endpoint |
| `scripts/run_clustering_scenario.py` | Unchanged — result saving already in place |
| `frontend/src/api/queries.ts` | Edited — estimate + status fetch functions, query keys |
| `frontend/src/tabs/ClustersTab.tsx` | Edited — estimation UI, polling, enhanced ScenarioCharts |
| `frontend/src/context/ScenarioNotificationContext.tsx` | **Created** — cross-tab notification context |
| `frontend/src/App.tsx` | Edited — ScenarioNotificationProvider wrapper |
| `frontend/src/types/theme.ts` | Edited — scenario_complete AlertType |
| `frontend/src/components/AlertPanel.tsx` | Edited — FlaskConical icon, dismiss button, click handler |
| `frontend/src/tabs/DashboardTab.tsx` | Edited — scenario alert injection |
| `tests/api/test_clustering_scenario.py` | Edited — 14 tests (7 new) |
| `frontend/src/context/__tests__/ScenarioNotificationContext.test.tsx` | **Created** — 4 tests |
| `frontend/src/tabs/__tests__/ClustersTab.test.tsx` | Edited — updated with provider |
| `frontend/src/tabs/__tests__/DashboardTab.test.tsx` | Edited — updated with provider |

---

## Implementation Corrections

### Estimate Response
2 additional fields not in spec:
- `training_sample` (int) — number of DFUs used for training (capped at 20,000)
- `sampled` (boolean) — whether DFU count exceeds sampling threshold
- Undocumented `scope` query parameter on estimate endpoint

### POST Scenario Response
- Additional field: `job_id` (for tracking in Jobs tab)

### JobManager Integration (Feature 39)
- POST handler delegates to `JobManager.submit_job("cluster_scenario", ...)` instead of running inline
- Maintains legacy state tracking for backward compatibility with status polling
- 409 conflict raised via `RuntimeError` from `manager.submit_job()` (not direct `_scenario_running` check)

### Runtime Estimation Formula
- `feature_gen_per_dfu = 0.001s`, `kmeans_per_dfu_per_k = 0.002s`
- `gap_multiplier = 2.5x`, `overhead_seconds = 10.0`
- `max_training_dfus = 20,000` sampling cap

### ScenarioNotificationContext
- `CompletedScenario` interface: `id`, `label`, `runtimeSeconds`, `result`
- `useScenarioNotification()` hook with error if used outside provider
- `failScenario()` method for error handling

### Additional Endpoint
- `GET /clustering/scenario/{scenario_id}` — retrieve scenario result directly (separate from `/status`)

### Pydantic Models (not in spec)
- `FeatureParams`: time_window_months, min_months_history
- `ModelParams`: k_range, min_cluster_size_pct, use_pca, pca_components, all_features
- `LabelParams`: volume_high, volume_low, cv_steady, cv_volatile, seasonality_threshold, zero_demand_threshold
- `ClusteringScenarioRequest`: feature_params, model_params, label_params, relabel_only, previous_scenario_id


---

## Examples

### Example: Full enhanced scenario workflow

```bash
# 1. Get runtime estimate
curl -s "http://localhost:8000/clustering/scenario/estimate?k_max=8" | jq .
# {"estimated_seconds": 72, "dfu_count": 18432, "k_range": 5}

# 2. Submit (202 Accepted — non-blocking)
SCENARIO_ID=$(curl -s -X POST http://localhost:8000/clustering/scenario \
  -H "Content-Type: application/json" \
  -d '{"model_params": {"k_range": [3, 8], "use_pca": false}, "feature_params": {}, "label_params": {}}' \
  | jq -r '.id')

# 3. Poll until complete (check every 3s)
until curl -s "http://localhost:8000/clustering/scenario/$SCENARIO_ID/status" | jq -e '.status == "completed"' > /dev/null; do
  sleep 3
done

# 4. Promote winning scenario to production
curl -s -X POST "http://localhost:8000/clustering/scenario/$SCENARIO_ID/promote"
# {"promoted": true, "dfus_updated": 18432, "best_k": 6, "silhouette": 0.71}
```

### Example: Enhanced chart descriptions

- **Elbow chart**: WCSS vs K with `ReferenceLine` at optimal K (red dashed vertical line)
- **Silhouette chart**: Bar chart with quality zones: Strong (≥0.7), Reasonable (0.5-0.7), Weak (0.25-0.5)
- **Feature importance**: Horizontal bars showing top 10 features driving cluster separation
- **Cluster size pie**: Pie chart with percentage labels, n_dfus per cluster

### Example: Scenario queueing when group is busy

```bash
# Submit 2 scenarios while one is running → second gets queued
curl -s -X POST http://localhost:8000/clustering/scenario -d '{"model_params": {}}' | jq .status
# "running"
curl -s -X POST http://localhost:8000/clustering/scenario -d '{"model_params": {"k_range": [4,6]}}' | jq .status
# "queued"  ← auto-dispatched when running job completes
```
