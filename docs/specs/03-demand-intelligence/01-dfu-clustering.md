# 03-01 DFU Clustering

> **Status:** Implemented | **Features:** 7, 29, 38

## Why This Moved Here

Clustering groups DFUs (Demand Forecast Units -- item + location combinations) by shared demand patterns. It *describes* demand behavior rather than *predicting* future values, making it a demand-intelligence capability rather than a forecasting one. Downstream consumers include inventory planning (safety stock by segment), forecasting (per-cluster model training), and exception detection.

---

## Problem

A portfolio of 100K+ DFUs cannot be managed individually. Planners need coherent groups that share demand characteristics so they can apply segment-level policies, train ML models per cluster, and spot anomalies within groups.

---

## Solution

KMeans clustering over 14 engineered features across 6 demand dimensions, with automatic K selection, minimum cluster size enforcement, and human-readable taxonomy labels. A What-If scenario UI lets planners experiment with parameters before promoting results to production.

---

## How It Works

### Feature Engineering

| Dimension | Features | Source |
|---|---|---|
| Volume | total_qty, avg_monthly_qty, qty_iqr | `fact_sales_monthly` |
| Trend | ols_trend_slope, cagr | OLS regression, CAGR formula |
| Seasonality | cv_monthly, yoy_corr, ols_seasonal_r2 | CV of monthly means, YoY correlation, OLS seasonal R-squared |
| Periodicity | fft_periodicity_strength | FFT dominant frequency magnitude |
| Intermittency | zero_month_pct, croston_adi | Zero-month ratio, Croston ADI (average demand interval) |
| Lifecycle | recency_ratio | Recent vs total volume ratio |

Log-transforms are applied to skewed volume features before StandardScaler normalization. Default time window is 36 months.

### Optimal K Selection

1. Evaluate K in range [5, 18] via KMeans
2. Score each K: `0.5 * silhouette_norm + 0.5 * calinski_harabasz_norm`
3. Discard any K where the smallest cluster is < 5% of DFUs
4. Select K with highest combined score
5. Post-hoc `merge_small_clusters()` merges any remaining undersized clusters into nearest large neighbor

### Taxonomy Labeling

Priority-ordered evaluation (first match wins):

| Priority | Dimension | Example Label |
|---|---|---|
| 1 | Intermittency | `intermittent_sporadic` |
| 2 | Periodicity | `periodic_quarterly` |
| 3 | Seasonality | `high_volume_seasonal` |
| 4 | Trend | `growing_moderate` |
| 5 | Volatility | `volatile_erratic` |
| 6 | Volume (5 tiers) | `steady_high_volume` |

Labels are compound (e.g., `high_volume_seasonal_growing`). Stored in `dim_dfu.cluster_assignment`.

### What-If Scenarios

Planners submit custom parameters via `POST /clustering/scenario`. The system:

1. Returns HTTP 202 immediately; runs KMeans in a background thread via JobManager
2. Supports per-group concurrency -- new requests queue (FIFO) instead of being rejected
3. `GET /clustering/scenario/{id}/status` polls progress
4. `POST /clustering/scenario/{id}/promote` writes results to `dim_dfu.ml_cluster`

Enhanced charts: elbow with optimal-K marker, silhouette bar chart with quality zones, feature importance bars, cluster size pie, gap statistic line.

---

## Data Model

| Table / Column | Type | Purpose |
|---|---|---|
| `dim_dfu.cluster_assignment` | TEXT | Production cluster label |
| `dim_dfu.ml_cluster` | INTEGER | Numeric cluster ID (used as ML feature) |

`ml_cluster` is always a hard feature in backtest models -- never stripped in either per-cluster or global training mode.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/clustering/scenario` | Submit What-If scenario (202 async) |
| GET | `/clustering/scenario/{id}/status` | Poll scenario progress |
| GET | `/clustering/scenario/estimate` | Runtime estimate |
| POST | `/clustering/scenario/{id}/promote` | Apply scenario to production |

---

## Pipeline

```
make cluster-all    # features -> train -> label -> update (full pipeline)
```

| Step | Script | Output |
|---|---|---|
| Feature engineering | `scripts/generate_clustering_features.py` | CSV feature matrix |
| Train + select K | `scripts/train_clustering_model.py` | MLflow experiment `dfu_clustering` |
| Label clusters | `scripts/label_clusters.py` | Labeled cluster assignments |
| Write to DB | `scripts/update_cluster_assignments.py` | `dim_dfu.cluster_assignment` updated |

---

## Configuration

File: `config/clustering_config.yaml`

```yaml
time_window_months: 36
k_range: [5, 18]
min_cluster_size_pct: 5.0
scoring: combined          # 0.5 * silhouette + 0.5 * calinski_harabasz
labeling: priority_ordered # intermittency -> periodicity -> seasonality -> trend -> volatility -> volume
```

---

## Dependencies

- **Upstream:** `fact_sales_monthly`, `dim_dfu`, `dim_item`
- **Downstream:** All backtest scripts (ml_cluster feature), safety stock (segment policies), ABC-XYZ classification
- **Libraries:** scikit-learn, pandas, scipy, matplotlib, seaborn, MLflow

---

## See Also

- [02-seasonality](02-seasonality.md) -- Seasonality detection (complementary demand pattern analysis)
- [../02-forecasting/02-07-algorithm-config](../02-forecasting/02-07-algorithm-config.md) -- `cluster_strategy` config key
- [../06-ui-platform/06-04-job-scheduler](../06-ui-platform/06-04-job-scheduler.md) -- Background scenario execution
