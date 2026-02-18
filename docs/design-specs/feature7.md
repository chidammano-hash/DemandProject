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

### Time Series Features (per DFU)

**Volume Metrics:**
- `mean_demand`, `median_demand`, `std_demand`, `cv_demand`, `min_demand`, `max_demand`, `total_demand`

**Trend Features:**
- `trend_slope`: Linear regression slope (demand vs. time)
- `trend_pct_change`: Percentage change from first to last period
- `trend_direction`: Positive/negative/stable indicator

**Seasonality Features:**
- `seasonality_strength`: Amplitude of seasonal pattern (via monthly means)
- `peak_month`, `seasonal_index_std`, `year_over_year_correlation`

**Volatility Features:**
- `zero_demand_pct`, `sparsity_score`, `demand_stability`, `outlier_count`

**Growth Patterns:**
- `growth_rate` (CAGR), `acceleration`, `recent_vs_historical`

### Item Features (from dim_item)
- Categorical: `category`, `class`, `sub_class`, `brand_name`, `country`, `national_service_model`
- Numeric: `case_weight`, `item_proof`, `bpc`, `bottle_pack`, `pack_case`

### DFU Features (from dim_dfu)
- Categorical: `brand`, `region`, `state_plan`, `sales_div`, `prod_cat_desc`, `abc_vol`, `service_lvl_grp`, `supergroup`
- Numeric: `execution_lag`, `total_lt`, `alcoh_pct`, `proof`

## Clustering Approach

### Primary Method: KMeans with Optimal K Selection

**K Selection Methods:**
1. Elbow Method (WCSS vs. K)
2. Silhouette Score
3. Gap Statistic
4. Business Constraints (minimum cluster size)

**Optimal K Range**: 3-12 clusters (configurable)

### Feature Scaling & Preprocessing
- StandardScaler (mean=0, std=1)
- Optional PCA for dimensionality reduction
- Low-variance feature removal (< 0.01)

## Cluster Labeling Strategy

**Composite Labels** based on volume tier + pattern type:
- `high_volume_steady`, `seasonal_high_volume`, `intermittent_low_volume`
- `high_volume_growing`, `low_volume_declining`, `seasonal_medium_volume`
- `medium_volume_steady`

**Label Priority Logic:**
1. Check volume tier (high/medium/low based on percentiles)
2. Check pattern type (seasonal/trending/intermittent/volatile/steady)
3. Combine into composite label

## Implementation Components

### 1. Feature Engineering Script
`mvp/demand/scripts/generate_clustering_features.py`
- Parameters: `--min-months` (default: 12), `--time-window` (default: 24), `--output`
- Output: `data/clustering_features.csv`

### 2. Clustering Model Script
`mvp/demand/scripts/train_clustering_model.py`
- Parameters: `--input`, `--k-range` (default: 3 12), `--min-cluster-size-pct`, `--use-pca`, `--output-dir`
- Output: `cluster_assignments.csv`, `cluster_centroids.csv`, `cluster_metadata.json`, visualization PNGs
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
  time_window_months: 24
  k_range: [3, 12]
  min_cluster_size_pct: 1.0
  feature_scaling: "standard"
  use_pca: false
  labeling:
    volume_thresholds: { high: 0.75, low: 0.25 }
    cv_thresholds: { steady: 0.3, volatile: 0.8 }
    seasonality_threshold: 0.5
    zero_demand_threshold: 0.2
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
