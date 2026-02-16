# Feature 13: DFU Clustering Framework

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
- `mean_demand`: Average monthly demand
- `median_demand`: Median monthly demand
- `std_demand`: Standard deviation
- `cv_demand`: Coefficient of variation (std/mean)
- `min_demand`, `max_demand`: Range indicators
- `total_demand`: Sum over history window

**Trend Features:**
- `trend_slope`: Linear regression slope (demand vs. time)
- `trend_pct_change`: Percentage change from first to last period
- `trend_direction`: Positive/negative/stable indicator

**Seasonality Features:**
- `seasonality_strength`: Amplitude of seasonal pattern (via monthly means)
- `peak_month`: Month with highest average demand
- `seasonal_index_std`: Standard deviation of monthly seasonal indices
- `year_over_year_correlation`: Correlation between same months across years

**Volatility Features:**
- `zero_demand_pct`: Percentage of months with zero demand
- `sparsity_score`: Measure of demand intermittency
- `demand_stability`: Inverse of coefficient of variation
- `outlier_count`: Number of months > 2 standard deviations from mean

**Growth Patterns:**
- `growth_rate`: Compound annual growth rate (CAGR)
- `acceleration`: Second derivative (change in growth rate)
- `recent_vs_historical`: Ratio of last 6 months to previous 6 months

### Item Features (from dim_item)

**Categorical:**
- `category`, `class`, `sub_class`
- `brand_name`, `country`
- `national_service_model`

**Numeric:**
- `case_weight`, `item_proof`
- `bpc`, `bottle_pack`, `pack_case`

### DFU Features (from dim_dfu)

**Categorical:**
- `brand`, `region`, `state_plan`, `sales_div`
- `prod_cat_desc`, `prod_class_desc`, `subclass_desc`
- `abc_vol` (A/B/C classification)
- `service_lvl_grp`, `supergroup`

**Numeric:**
- `execution_lag`, `total_lt`
- `alcoh_pct`, `proof`
- `vintage` (if applicable)

## Clustering Approach

### Primary Method: KMeans with Optimal K Selection

**K Selection Methods:**
1. **Elbow Method**: Plot within-cluster sum of squares (WCSS) vs. K
2. **Silhouette Score**: Maximize average silhouette coefficient
3. **Gap Statistic**: Compare log(WCSS) to null reference distribution
4. **Business Constraints**: Ensure minimum cluster size (e.g., 1% of total DFUs)

**Optimal K Range**: 3-12 clusters (configurable)

### Feature Scaling & Preprocessing

- **StandardScaler**: Normalize all numeric features to mean=0, std=1
- **PCA** (optional): Reduce dimensionality if feature count > 50
- **Feature Selection**: Remove low-variance features (< 0.01 variance)

## Cluster Labeling Strategy

### Automated Label Assignment

Each cluster receives a primary label based on feature centroids:

**Volume-Based Labels:**
- `high_volume`: mean_demand > 75th percentile
- `medium_volume`: mean_demand between 25th-75th percentile
- `low_volume`: mean_demand < 25th percentile

**Pattern-Based Labels:**
- `steady_demand`: cv_demand < 0.3, trend_slope near zero
- `seasonal`: seasonality_strength > threshold, clear peak_month pattern
- `trending_up`: trend_slope > 0, growth_rate > 5%
- `trending_down`: trend_slope < 0, growth_rate < -5%
- `intermittent`: zero_demand_pct > 20%, sparsity_score > threshold
- `volatile`: cv_demand > 0.8, high outlier_count

**Composite Labels:**
- `high_volume_steady`: High volume + steady pattern
- `seasonal_high_volume`: Seasonal + high volume
- `intermittent_low_volume`: Intermittent + low volume
- `high_volume_growing`: High volume + trending up
- `low_volume_declining`: Low volume + trending down
- `seasonal_medium_volume`: Medium volume + seasonal
- `medium_volume_steady`: Medium volume + steady

### Label Priority Logic

1. Check volume tier (high/medium/low)
2. Check pattern type (seasonal/trending/intermittent/volatile/steady)
3. Combine into composite label if both dimensions are strong
4. Default to volume-based label if pattern is weak

## Implementation Components

### 1. Feature Engineering Script

**File**: `mvp/demand/scripts/generate_clustering_features.py`

**Responsibilities:**
- Query sales history for each DFU (configurable time window, default: 24 months)
- Compute all time series features
- Join with `dim_dfu` and `dim_item` for attribute features
- Output feature matrix: CSV format
- Handle missing data (exclude DFUs with insufficient history)

**Parameters:**
- `--min-months`: Minimum history required (default: 12)
- `--time-window`: Months to include (default: 24, or "all")
- `--output`: Output file path (default: `data/clustering_features.csv`)

**Output Format:**
- CSV file with one row per DFU
- Columns: `dfu_ck` (key), time series features, DFU attributes, item attributes
- DFUs with insufficient history (< min_months) are excluded
- Missing numeric values are filled with 0

### 2. Clustering Model Script

**File**: `mvp/demand/scripts/train_clustering_model.py`

**Responsibilities:**
- Load feature matrix
- Scale features (StandardScaler)
- Run K selection analysis (elbow, silhouette, gap statistic)
- Train KMeans with optimal K
- Generate cluster assignments
- Compute cluster statistics (centroids, sizes, feature distributions)
- Log to MLflow: parameters, metrics, cluster profiles, visualization artifacts

**MLflow Logging:**
- Parameters: `k`, `min_months`, `time_window`, `scaling_method`
- Metrics: `silhouette_score`, `inertia`, `n_clusters`, `cluster_sizes`
- Artifacts: Elbow plot, silhouette plot, cluster feature distributions, PCA visualization (if used)

**Output Files** (saved to `--output-dir`):
- `cluster_assignments.csv`: DFU assignments (`dfu_ck`, `cluster_id`)
- `cluster_metadata.json`: K selection results, optimal K, metrics, cluster sizes
- `cluster_centroids.csv`: Feature centroids per cluster (`cluster_id` + feature columns)
- `k_selection_plots.png`: Combined visualization of elbow, silhouette, and gap statistic plots
- `cluster_visualization.png`: 2D PCA projection of clusters (if not using PCA, temporary PCA applied for visualization)

**Parameters:**
- `--input`: Input feature matrix (default: `data/clustering_features.csv`)
- `--k-range`: K range to test (default: 3 12)
- `--min-cluster-size-pct`: Minimum cluster size percentage (default: 0.01)
- `--use-pca`: Use PCA for dimensionality reduction
- `--pca-components`: Number of PCA components (auto if not specified)
- `--output-dir`: Output directory (default: `data/clustering`)

### 3. Cluster Labeling Script

**File**: `mvp/demand/scripts/label_clusters.py`

**Responsibilities:**
- Load cluster assignments and centroids
- Apply labeling logic based on feature thresholds
- Generate cluster profiles (summary statistics per cluster)
- Validate label uniqueness and coverage
- Output labeled assignments: `dfu_ck` -> `cluster_label`

**Parameters:**
- `--centroids`: Cluster centroids file (default: `data/clustering/cluster_centroids.csv`)
- `--assignments`: Cluster assignments file (default: `data/clustering/cluster_assignments.csv`)
- `--metadata`: Cluster metadata file (default: `data/clustering/cluster_metadata.json`)
- `--output`: Output file (default: `data/clustering/cluster_labels.csv`)
- `--config`: Configuration file (default: `config/clustering_config.yaml`)

**Output Files:**
- `cluster_labels.csv`: Labeled assignments (`dfu_ck`, `cluster_id`, `cluster_label`)
- `cluster_profiles.json`: Summary statistics per cluster (mean_demand, cv_demand, seasonality_strength, etc.)

**Labeling Logic:**
- Volume thresholds computed from actual centroid mean_demand values (percentile-based)
- Pattern detection uses absolute thresholds from config
- Composite labels prioritized when both volume and pattern dimensions are strong
- Warning logged if duplicate labels detected

### 4. Assignment Update Script

**File**: `mvp/demand/scripts/update_cluster_assignments.py`

**Responsibilities:**
- Load labeled cluster assignments
- Update `dim_dfu.cluster_assignment` column in PostgreSQL
- Validate updates (counts, referential integrity)
- Generate summary report

**Parameters:**
- `--input`: Labeled cluster assignments file (default: `data/clustering/cluster_labels.csv`)
- `--dry-run`: Show what would be updated without making changes

**Behavior:**
- Updates `dim_dfu.cluster_assignment` and `modified_ts` columns
- Validates that DFUs exist in database before updating
- Reports cluster distribution before and after update
- Warns about DFUs with missing labels or DFUs not found in database
- Shows count of DFUs still without cluster assignment after update

## Makefile Targets

```makefile
cluster-features:
	$(UV) python scripts/generate_clustering_features.py --min-months 12 --time-window 24

cluster-train:
	$(UV) python scripts/train_clustering_model.py --k-range 3 12

cluster-label:
	$(UV) python scripts/label_clusters.py

cluster-update:
	$(UV) python scripts/update_cluster_assignments.py

cluster-all: cluster-features cluster-train cluster-label cluster-update
```

## Configuration

**File**: `mvp/demand/config/clustering_config.yaml`

```yaml
clustering:
  min_months_history: 12
  time_window_months: 24  # or "all"
  k_range: [3, 12]
  min_cluster_size_pct: 1.0
  feature_scaling: "standard"
  use_pca: false
  pca_components: null  # auto if use_pca=true
  
  labeling:
    volume_thresholds:
      high: 0.75
      low: 0.25
    cv_thresholds:
      steady: 0.3
      volatile: 0.8
    seasonality_threshold: 0.5
    zero_demand_threshold: 0.2
```

## Database Schema

No schema changes required — `dim_dfu.cluster_assignment` already exists (TEXT column).

**Index** (added for performance):
```sql
CREATE INDEX IF NOT EXISTS idx_dim_dfu_cluster_assignment 
  ON dim_dfu (cluster_assignment);
```

This index is created in `sql/005_create_dim_dfu.sql` to optimize queries filtering by cluster assignment.

## API Integration

### New Endpoint: `GET /domains/dfu/clusters`

Returns cluster summary:
```json
{
  "domain": "dfu",
  "total_assigned": 8500,
  "clusters": [
    {
      "cluster_id": "high_volume_steady",
      "label": "high_volume_steady",
      "count": 1250,
      "pct_of_total": 14.7,
      "avg_demand": 4500.5,
      "cv_demand": 0.18
    },
    ...
  ]
}
```

### Filter Enhancement

Existing `/domains/dfu/page` endpoint already supports filtering by `cluster_assignment` column (via existing filter mechanism).

**Example Usage:**
```bash
# Get DFUs in high_volume_steady cluster
curl "http://localhost:8000/domains/dfu/page?filters={\"cluster_assignment\":\"=high_volume_steady\"}&limit=100"

# Get cluster summary
curl "http://localhost:8000/domains/dfu/clusters"
```

## MLflow Integration

**Experiment Name**: `dfu_clustering`

**Run Tags:**
- `model_type`: `clustering`
- `feature_set`: `time_series_item_dfu`
- `version`: `v1.0`

**Logged Artifacts:**
- `cluster_assignments.csv`: DFU -> cluster_id mapping
- `cluster_metadata.json`: K selection results, optimal K, metrics, cluster sizes
- `cluster_centroids.csv`: Feature centroids per cluster
- `k_selection_plots.png`: Combined visualization (elbow, silhouette, gap statistic)
- `cluster_visualization.png`: 2D PCA projection of clusters (always generated for visualization)

**MLflow Tracking URI**: Configured via `MLFLOW_TRACKING_URI` environment variable (default: `http://localhost:5003`)

## Dependencies

Added to `mvp/demand/pyproject.toml`:
- `scikit-learn>=1.3.0` - Clustering algorithms (KMeans)
- `numpy>=1.24.0` - Numerical computations
- `pandas>=2.0.0` - Data manipulation
- `scipy>=1.11.0` - Statistical functions and gap statistic computation
- `mlflow>=2.8.0` - Experiment tracking and model registry
- `matplotlib>=3.7.0` - Visualization (elbow plots, cluster visualizations)
- `seaborn>=0.12.0` - Enhanced plotting
- `pyyaml>=6.0.0` - Configuration file parsing

## Validation & Quality Checks

1. **Cluster Balance**: No cluster < 1% of total DFUs (configurable via `--min-cluster-size-pct`)
2. **Feature Coverage**: All DFUs with sufficient history (>= min_months) receive assignments
3. **Label Uniqueness**: Each cluster has exactly one primary label (warning logged if duplicates)
4. **Data Quality**: 
   - DFUs with insufficient history excluded from clustering
   - Missing numeric features filled with 0
   - Low-variance features (< 0.01) removed before clustering
5. **Business Validation**: Sample DFUs per cluster should be reviewed for label accuracy

## Error Handling

- **Insufficient Data**: DFUs with < min_months history are excluded from feature generation
- **Missing Files**: Scripts validate input files exist before processing
- **Database Errors**: Connection errors and SQL errors are caught and reported
- **MLflow Connection**: If MLflow is unavailable, clustering still proceeds but logging is skipped
- **Empty Clusters**: Minimum cluster size validation prevents clusters that are too small

## Usage Workflow

1. **Initial Clustering**:
   ```bash
   # Run full pipeline
   make cluster-all
   
   # Or run steps individually
   make cluster-features  # Generate features
   make cluster-train      # Train model and select optimal K
   make cluster-label      # Assign business labels
   make cluster-update     # Update database
   ```

2. **Periodic Re-clustering** (e.g., quarterly):
   - Re-run feature generation with updated sales history
   - Retrain clustering model (may select different optimal K)
   - Compare new assignments to previous (track drift via MLflow)
   - Review cluster profiles and label changes
   - Update `dim_dfu.cluster_assignment` if significant changes
   - Use `--dry-run` flag to preview changes before updating database

3. **LGBM Integration**:
   - Use `cluster_assignment` as a categorical feature in global LGBM models
   - Train separate models per cluster (if cluster-specific models desired)
   - Filter training data by cluster for homogeneous segments
   - Cluster labels provide interpretable segments for model explainability

## File Structure

```
mvp/demand/
├── scripts/
│   ├── generate_clustering_features.py
│   ├── train_clustering_model.py
│   ├── label_clusters.py
│   └── update_cluster_assignments.py
├── config/
│   └── clustering_config.yaml
├── data/
│   ├── clustering_features.csv          # Feature matrix
│   └── clustering/
│       ├── cluster_assignments.csv      # dfu_ck -> cluster_id
│       ├── cluster_centroids.csv        # Feature centroids
│       ├── cluster_metadata.json        # K selection results
│       ├── cluster_labels.csv           # dfu_ck -> cluster_label
│       ├── cluster_profiles.json        # Cluster statistics
│       ├── k_selection_plots.png        # Visualization
│       └── cluster_visualization.png     # PCA visualization
└── sql/
    └── 005_create_dim_dfu.sql          # Includes cluster_assignment index
```

## Future Enhancements

- **Incremental Updates**: Update assignments for new/changed DFUs without full re-clustering
- **Hierarchical Clustering**: Multi-level clusters (e.g., volume tier -> pattern type)
- **Dynamic K Selection**: Auto-adjust K based on data characteristics
- **Cluster Stability Metrics**: Track assignment changes over time
- **UI Visualization**: Cluster explorer in React frontend showing cluster characteristics and DFU memberships
