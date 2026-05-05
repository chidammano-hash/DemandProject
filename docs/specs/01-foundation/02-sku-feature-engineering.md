# SKU Feature Engineering

> Unified per-DFU time-series feature computation pipeline. Reads monthly sales history from `fact_sales_monthly`, computes ~35 statistical features across six dimensions (volume, trend, seasonality, periodicity, intermittency, lifecycle), derives classification labels (seasonality profile, variability class), and persists all results to `dim_sku`. These features are the foundation for clustering, backtest feature engineering, champion selection, ABC-XYZ segmentation, safety stock formulas, and inventory policy assignment.

| | |
|---|---|
| **Status** | Implemented |
| **Spec version** | 2.0 (2026-04-05) |
| **UI Tab** | SKU Features (dedicated explorer), Inventory Planning (variability detail), DFU Analysis (filter dimension) |
| **API prefix** | `/sku-features` |
| **Config** | `config/forecasting/sku_features_config.yaml` |
| **DDL** | `sql/005_create_dim_dfu.sql` (base), `sql/022_add_demand_variability_columns.sql` (variability), `sql/120_add_unified_feature_columns.sql` (extended) |
| **Pipeline entry** | `scripts/ml/compute_sku_features.py` (`make features-compute`) |
| **Core library** | `common/ml/sku_features/` (orchestration), `common/ml/clustering/features.py` (computation) |

---

## Table of Contents

1. [Problem](#problem)
2. [Solution Overview](#solution-overview)
3. [Feature Catalog](#feature-catalog)
4. [Derived Classifications](#derived-classifications)
5. [Architecture](#architecture)
6. [Pipeline](#pipeline)
7. [Configuration](#configuration)
8. [Database Schema](#database-schema)
9. [API Endpoints](#api-endpoints)
10. [UI Tab](#ui-tab)
11. [Downstream Consumers](#downstream-consumers)
12. [Column Name Mapping](#column-name-mapping)
13. [Dependencies](#dependencies)
14. [See Also](#see-also)

---

## Problem

Supply chain planners cannot manually inspect thousands of DFUs for demand behavior patterns. Without automated feature engineering:

- **Seasonal items get uniform treatment.** Safety stock and replenishment logic does not distinguish strong-seasonal from steady-demand items, leading to over-stock in troughs and stock-outs at peaks.
- **Safety stock formulas lack inputs.** Demand variability (sigma-D, CV) is a required input for statistical safety stock computation. Without per-DFU profiling, planners rely on blanket safety factors across the entire portfolio, misallocating buffer stock.
- **ABC-XYZ classification is blocked.** The XYZ dimension requires a coefficient of variation (CV) per DFU to classify demand predictability.
- **Clustering operates blind.** ML-based DFU clustering needs numeric features (CV, seasonality strength, trend slope, intermittency, etc.) as input vectors. Without pre-computed features, clustering must recompute them from scratch on every run.
- **Backtest feature engineering is duplicated.** Each backtest model (LGBM, CatBoost, XGBoost) independently computes DFU-level statistics. Centralizing computation eliminates redundancy and guarantees consistent feature definitions across all models.

---

## Solution Overview

A unified feature computation pipeline processes all DFUs in a single pass:

1. **Load** monthly sales history from `fact_sales_monthly` joined with `dim_sku` (configurable lookback window, default 36 months).
2. **Compute** ~35 time-series features per DFU using the shared library in `common/ml/clustering/features.py`.
3. **Classify** each DFU into categorical labels (seasonality profile, variability class) using configurable thresholds.
4. **Persist** all numeric features and classification labels to `dim_sku` columns via bulk COPY + UPDATE.
5. **Export** a backward-compatible CSV (`data/clustering_features.csv`) consumed by the clustering pipeline.

The pipeline is idempotent (UPDATE-based, not INSERT), parallelized across CPU cores, and completes in under 2 minutes for typical portfolios (~18,000 DFUs).

---

## Feature Catalog

All features are computed by `compute_time_series_features()` in `common/ml/clustering/features.py`. The function accepts a DataFrame with `startdate` (datetime) and `qty` (float) columns and returns a `pd.Series` of named scalar features.

### Volume Features (8)

Core demand magnitude and dispersion statistics.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `mean_demand` | `np.mean(qty)` | Average monthly demand (including zeros) | 1 month |
| `median_demand` | `np.median(qty)` | Median monthly demand | 1 month |
| `std_demand` | `np.std(qty)` | Standard deviation of monthly demand | 2 months |
| `cv_demand` | `std / mean` | Coefficient of variation (scale-free volatility) | 2 months |
| `iqr_demand` | `P75 - P25` | Interquartile range (robust spread) | 1 month |
| `min_demand` | `np.min(qty)` | Minimum monthly demand observed | 1 month |
| `max_demand` | `np.max(qty)` | Maximum monthly demand observed | 1 month |
| `total_demand` | `np.sum(qty)` | Total demand across the window | 1 month |

### Statistical Features (5)

Higher-order statistics and distribution shape descriptors.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `demand_mad` | `mean(abs(qty - mean))` | Mean absolute deviation from the mean | 1 month |
| `demand_p90` | `np.percentile(qty, 90)` | 90th percentile of demand | 1 month |
| `demand_skewness` | `scipy.stats.skew(qty)` | Skewness (tail asymmetry); 0.0 if scipy unavailable | 3 months |
| `demand_kurtosis` | `scipy.stats.kurtosis(qty)` | Excess kurtosis (tail heaviness); 0.0 if scipy unavailable | 3 months |
| `outlier_count` | `sum(abs(qty - mean) > 2*std)` | Count of months exceeding 2-sigma from mean | 2 months |

### Trend Features (7)

Linear trend direction, magnitude, and growth dynamics.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `trend_slope` | `np.polyfit(x, y, 1)[0]` | Raw OLS slope (units per month) | 2 months |
| `trend_slope_norm` | `slope / mean` | Scale-invariant normalized slope | 2 months |
| `trend_r2` | `(1 - SS_res/SS_tot) * sign(slope)` | Signed R-squared of linear fit (positive = uptrend, negative = downtrend) | 3 months |
| `trend_pct_change` | `(last - first) / first * 100` | Percentage change from first to last observation | 2 months |
| `trend_direction` | `1 if slope > 0.01, -1 if < -0.01, else 0` | Categorical trend direction (integer) | 2 months |
| `cagr` | `((mean_H2 / mean_H1)^(1/periods) - 1) * 100` | Compound annual growth rate (half-split) | 12 months |
| `acceleration` | `g2 - g1` (growth rate of thirds) | Second-derivative approximation: accelerating vs decelerating | 12 months |

### Seasonality Features (9)

Seasonal pattern detection, timing, and confirmation signals.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `seasonality_strength` | CV of monthly means across years | Higher CV = stronger seasonal signal (legacy metric) | 12 months |
| `seasonal_amplitude` | `(max_monthly - min_monthly) / mean` | Amplitude ratio: peak-to-trough swing relative to overall mean | 12 months |
| `seasonal_r2` | 12-period seasonal dummy OLS R-squared | Fraction of variance explained by monthly indicators (STL-lite) | 24 months |
| `seasonal_index_std` | `std(monthly_means)` | Standard deviation of monthly average demands | 12 months |
| `peak_month` | `argmax(monthly_means)` | Month (1-12) with highest average demand | 12 months |
| `trough_month` | `argmin(monthly_means)` | Month (1-12) with lowest average demand | 12 months |
| `peak_trough_ratio` | `max(monthly_means) / min(monthly_means)` | Ratio of peak to trough monthly averages (0.0 if trough is zero) | 12 months |
| `yoy_correlation` | Mean pairwise Pearson correlation between yearly month-vectors | Measures repeatability of the yearly pattern | 24 months |
| `acf_lag12` | `np.corrcoef(qty[:-12], qty[12:])[0,1]` | Autocorrelation at 12-month lag (confirms annual periodicity) | 24 months |

**Note:** `year_over_year_correlation` is a backward-compatible alias for `yoy_correlation`.

### Periodicity Features (1)

FFT-based dominant frequency detection.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `periodicity_strength` | `max_fft_power / total_fft_power` (DC excluded) | Fraction of FFT spectral power in the dominant non-DC component. High value = strong single periodic signal (e.g., annual cycle). | 12 months |

### Intermittency Features (3)

Demand occurrence patterns and sparsity indicators.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `zero_demand_pct` | `count(qty == 0) / n` | Fraction of months with zero demand (0.0 to 1.0) | 1 month |
| `adi` | Mean gap between nonzero demand periods (Croston's Average Demand Interval) | Average number of months between demand occurrences. Higher = more intermittent. | 1 month |
| `demand_stability` | `1.0 / (1.0 + cv_demand)` | Inverse-CV stability score (bounded 0-1); higher = more stable | 2 months |

**Note:** `sparsity_score` is a backward-compatible alias for `zero_demand_pct`.

### Lifecycle Features (2)

History depth and recency of demand activity.

| Feature | Formula / Method | Description | Minimum data |
|---|---|---|---|
| `months_available` | `len(qty)` | Number of months of demand history available in the window | 1 month |
| `recency_ratio` | `mean(last_6m) / mean(prior)` | Ratio of recent demand (last 6 months) to earlier history. >1 = growing, <1 = declining. | 12 months |

**Note:** `growth_rate` is an alias for `cagr`; `recent_vs_historical` is an alias for `recency_ratio`. Both are kept for backward compatibility.

### Complete Feature Summary

| Dimension | Count | Features |
|---|---|---|
| Volume | 8 | mean_demand, median_demand, std_demand, cv_demand, iqr_demand, min_demand, max_demand, total_demand |
| Statistical | 5 | demand_mad, demand_p90, demand_skewness, demand_kurtosis, outlier_count |
| Trend | 7 | trend_slope, trend_slope_norm, trend_r2, trend_pct_change, trend_direction, cagr, acceleration |
| Seasonality | 9 | seasonality_strength, seasonal_amplitude, seasonal_r2, seasonal_index_std, peak_month, trough_month, peak_trough_ratio, yoy_correlation, acf_lag12 |
| Periodicity | 1 | periodicity_strength |
| Intermittency | 3 | zero_demand_pct, adi, demand_stability |
| Lifecycle | 2 | months_available, recency_ratio |
| **Total** | **35** | Plus 4 backward-compat aliases (sparsity_score, year_over_year_correlation, growth_rate, recent_vs_historical) |

---

## Derived Classifications

Classification columns are computed *after* the numeric features and stored alongside them in `dim_sku`. They are derived by thresholding the numeric features according to configurable rules.

### Seasonality Profile

Column: `seasonality_profile` (TEXT)

Determines how strong and repeatable the seasonal pattern is. Classification uses `seasonal_amplitude` as the primary signal, with `yoy_correlation` and `seasonal_r2` as confirmation gates.

| Profile | Criteria | Interpretation |
|---|---|---|
| `strong` | amplitude >= 0.70 AND (yoy_corr >= 0.40 OR seasonal_r2 >= 0.30) | Clear, repeatable seasonal pattern |
| `moderate` | amplitude >= 0.35 AND (yoy_corr >= 0.40 OR seasonal_r2 >= 0.30) | Detectable seasonal component, not dominant |
| `low` | amplitude >= 0.15 (no confirmation required) | Weak seasonal signal |
| `none` | amplitude < 0.15 | No meaningful seasonality detected |

The classifier in `common/ml/sku_features/classifiers.py` (`classify_seasonality_profile()`) accepts custom thresholds as parameters.

**Legacy `is_yearly_seasonal` flag:** Set when `yoy_corr >= 0.5 AND acf_lag12 >= 0.3`. This predates the four-tier profile classification and is retained for backward compatibility.

### Variability Class

Column: `variability_class` (TEXT)

Four-class taxonomy following the Syntetos-Boylan framework, using `cv_demand` and `zero_demand_pct` as inputs.

| Class | Criteria | Interpretation |
|---|---|---|
| `smooth` | CV < 0.30 AND zero_pct < intermittency_threshold | Low variability, regular demand |
| `erratic` | 0.30 <= CV < 0.80 AND zero_pct < intermittency_threshold | High variability but regular timing |
| `intermittent` | zero_pct >= 0.30 AND CV < 0.80 | Low variability but sporadic timing |
| `lumpy` | CV >= 1.50, OR (zero_pct >= 0.30 AND CV >= 0.80) | Both high variability and sporadic timing |

The classifier in `common/ml/sku_features/classifiers.py` (`classify_variability_class()`) handles edge cases: extreme CV forces `lumpy` regardless of intermittency; moderate CV with regular timing maps to `erratic`.

**Note:** The pipeline script `compute_sku_features.py` also contains inline classifier functions (`_classify_seasonality_profile`, `_classify_variability`) with slightly different thresholds used during the pipeline run. The `classifiers.py` module provides the canonical reusable implementation for downstream callers.

### Dependent Classifications (not computed here)

The following classifications in `dim_sku` depend on features computed by this pipeline but are assigned by separate downstream pipelines:

| Column | Source | Dependency |
|---|---|---|
| `xyz_class` | ABC-XYZ pipeline (`scripts/inventory/compute_abc_xyz.py`) | Reads `demand_cv` from `dim_sku` |
| `abc_xyz_segment` | ABC-XYZ pipeline | Combines ABC (revenue-based) with XYZ (CV-based) |
| `abc_xyz_policy_id` | ABC-XYZ pipeline | Policy assignment from segment |
| `ml_cluster` | Clustering pipeline (`scripts/ml/run_clustering_scenario.py`) | Reads pre-computed features from `data/clustering_features.csv` |

---

## Architecture

### Module Layout

```
common/ml/
  clustering/
    features.py              # Core: compute_time_series_features()
                             #   _seasonal_r2(), _periodicity_strength(), _adi()
                             #   _compute_features_for_group()  <-- multiprocessing entry

  sku_features/              # Orchestration package
    __init__.py              # Re-exports public API
    compute.py               # load_sales_from_db(), compute_all_sku_features()
    classifiers.py           # classify_seasonality_profile(), classify_variability_class()
    persistence.py           # write_features_to_dim_sku() — COPY + UPDATE bulk write

scripts/ml/
  compute_sku_features.py    # CLI entry point: run_pipeline(), main()
                             # Loads config, orchestrates load -> compute -> classify -> persist

scripts/                     # Deprecated shims (backward-compat only)
  detect_seasonality.py      # Delegates to unified pipeline
  compute_demand_variability.py  # Delegates to unified pipeline
  update_seasonality_profiles.py # Delegates to unified pipeline
```

### Computation Flow

```
                            +--------------------------+
                            | fact_sales_monthly       |
                            | (item_id, customer_group,|
                            |  loc, startdate, qty)    |
                            +-------------|------------+
                                          |
                                 JOIN dim_sku (sku_ck)
                                          |
                            +-------------|------------+
                            | load_sales_from_db()     |
                            | common/ml/sku_features/  |
                            | compute.py               |
                            +-------------|------------+
                                          |
                           DataFrame: sku_ck, startdate, qty
                                          |
                            +-------------|------------+
                            | compute_all_sku_features |
                            |                          |
                            |  groupby sku_ck          |
                            |  multiprocessing.Pool    |
                            |  -> _compute_features_   |
                            |     for_group()          |
                            +-------------|------------+
                                          |
                   per-group: compute_time_series_features(df)
                              common/ml/clustering/features.py
                                          |
                            +-------------|------------+
                            | ~35 features per SKU     |
                            | returned as pd.Series    |
                            +-------------|------------+
                                          |
                            +-------------|------------+
                            | _apply_classifiers()     |
                            | scripts/ml/              |
                            | compute_sku_features.py  |
                            |                          |
                            | -> seasonality_profile   |
                            | -> variability_class     |
                            +-------------|------------+
                                          |
                      +---------|-----------|----------+
                      |                     |          |
              +-------|-------+    +--------|--------+ |
              | persistence.py|    | clustering_     | |
              | write_features|    | features.csv    | |
              | _to_dim_sku() |    | (backward compat| |
              |               |    |  CSV export)    | |
              | COPY -> temp  |    +-----------------+ |
              | UPDATE dim_sku|                         |
              +---------------+                         |
                                                        |
                                              +---------|-------+
                                              | Optional:       |
                                              | --output-csv    |
                                              | --dry-run       |
                                              +-----------------+
```

### Parallelism Strategy

- For **>500 SKUs**: uses `multiprocessing.Pool` with `min(cpu_count, 8)` workers and auto-tuned chunk size (`n_items / (workers * 4)`).
- For **<=500 SKUs**: serial computation to avoid process spawning overhead.
- Worker count configurable via `--workers` CLI flag, config YAML, or defaults.

### Persistence Strategy

`write_features_to_dim_sku()` in `persistence.py` uses a three-step approach:

1. **CREATE TEMP TABLE** with TEXT columns (auto-dropped on commit).
2. **COPY** staging DataFrame into the temp table via psycopg3 binary COPY protocol.
3. **UPDATE dim_sku SET ... FROM temp_table** with explicit type casts (TEXT -> NUMERIC, INTEGER, TIMESTAMPTZ).

This approach avoids row-by-row UPDATEs and achieves near-constant performance regardless of portfolio size.

---

## Pipeline

### Execution

```bash
# Primary entry point (recommended)
make features-compute

# With options
uv run python scripts/ml/compute_sku_features.py
uv run python scripts/ml/compute_sku_features.py --dry-run
uv run python scripts/ml/compute_sku_features.py --output-csv data/sku_features.csv
uv run python scripts/ml/compute_sku_features.py --workers 4 --time-window 24

# Legacy aliases (all delegate to features-compute)
make seasonality-all        # alias -> features-compute
make seasonality-detect     # alias -> features-compute
make variability-all        # alias -> features-compute
make variability-compute    # alias -> features-compute
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--dry-run` | false | Compute features but skip DB writes |
| `--output-csv PATH` | none | Write feature DataFrame to CSV |
| `--workers N` | from config (4) | Number of parallel workers |
| `--time-window N` | from config (36) | Lookback window in months |

### Pipeline Steps

| Step | Component | Input | Output |
|---|---|---|---|
| 1. Load sales | `load_sales_from_db()` | `fact_sales_monthly` + `dim_sku` | DataFrame: sku_ck, startdate, qty |
| 2. Compute features | `compute_all_sku_features()` | Sales DataFrame | DataFrame: sku_ck + 35 feature columns |
| 3. Apply classifiers | `_apply_classifiers()` | Feature DataFrame + config thresholds | Adds `seasonality_profile`, `variability_class` columns |
| 4. Write to DB | `write_features_to_dim_sku()` | Feature DataFrame | `dim_sku` columns updated |
| 5. Write CSV | pandas `to_csv()` | Feature DataFrame | `data/clustering_features.csv` (backward-compat) |
| 6. Optional CSV | pandas `to_csv()` | Feature DataFrame | User-specified path via `--output-csv` |

### Pipeline Position in Full Setup

```
make setup-data           # Load all 11 domains into Postgres
    |
make features-compute     # THIS PIPELINE: compute SKU features -> dim_sku
    |
make cluster-all          # Clustering reads features from clustering_features.csv
    |
make abc-xyz-all          # ABC-XYZ reads demand_cv from dim_sku
    |
make setup-inv-planning   # Safety stock, EOQ, policies read from dim_sku
```

Abbreviated: `make setup-features` runs the full chain: `setup-data -> features-compute -> cluster-all -> lt-profile-all -> abc-xyz-all -> demand-signals-all`.

### Idempotency

The pipeline is fully idempotent. Running `make features-compute` multiple times produces identical results (subject to planning date). The DB write is an UPDATE (not INSERT), so re-execution overwrites previous values without creating duplicates.

---

## Configuration

### File: `config/forecasting/sku_features_config.yaml`

```yaml
# Unified SKU Feature Computation Configuration
# Single source of truth for all feature computation thresholds

history:
  time_window_months: 36          # months of sales history to use
  min_months_history: 12          # minimum months for a SKU to be included

seasonality:
  amplitude_threshold: 0.3        # seasonal_amplitude > this -> seasonal
  r2_threshold: 0.25              # seasonal_r2 > this -> confirmed seasonal
  yoy_correlation_threshold: 0.3  # year-over-year correlation confirmation
  peak_trough_min_ratio: 1.5      # peak/trough ratio for strong seasonality

variability:
  cv_thresholds:
    smooth: 0.3                   # CV < this -> smooth
    erratic: 0.8                  # CV > this -> erratic
  intermittency_threshold: 0.15   # zero_demand_pct > this -> intermittent

trend:
  slope_threshold: 0.01           # |trend_slope_norm| > this -> trending
  r2_threshold: 0.25              # trend_r2 > this -> meaningful trend
  cagr_growing: 5.0               # CAGR > this -> growing
  cagr_declining: -5.0            # CAGR < this -> declining

parallelism:
  max_workers: 8                  # multiprocessing pool size
```

### Built-in Defaults

If the config file is absent, the pipeline falls back to hardcoded defaults in `compute_sku_features.py`:

```python
_DEFAULTS = {
    "time_window_months": 36,
    "min_months_history": 12,
    "workers": 4,
    "classifiers": {
        "seasonality_profile": {
            "high_threshold": 0.4,
            "moderate_threshold": 0.15,
        },
        "variability_class": {
            "low_cv": 0.3,
            "moderate_cv": 0.6,
            "high_cv": 1.0,
        },
    },
}
```

### Legacy Config Files (deprecated)

The following config files predate the unified pipeline. They are no longer authoritative:

- `config/seasonality_config.yaml` -- legacy seasonality thresholds
- `config/variability_config.yaml` -- legacy variability thresholds

All new work should reference `config/forecasting/sku_features_config.yaml` exclusively.

---

## Database Schema

### Base Columns on `dim_sku` (from `sql/005_create_dim_dfu.sql`)

Seasonality columns (6):

| Column | Type | Example |
|---|---|---|
| `seasonality_profile` | TEXT | `strong` |
| `seasonality_strength` | NUMERIC(10,4) | 0.72 |
| `is_yearly_seasonal` | BOOLEAN | true |
| `peak_month` | INTEGER | 7 |
| `trough_month` | INTEGER | 2 |
| `peak_trough_ratio` | NUMERIC(10,4) | 3.4 |

Demand variability columns (13):

| Column | Type | Example |
|---|---|---|
| `demand_mean` | NUMERIC(15,4) | 500.0 |
| `demand_std` | NUMERIC(15,4) | 225.0 |
| `demand_cv` | NUMERIC(10,6) | 0.45 |
| `demand_mad` | NUMERIC(15,4) | 120.5 |
| `demand_p50` | NUMERIC(15,4) | 480.0 |
| `demand_p90` | NUMERIC(15,4) | 800.0 |
| `demand_skewness` | NUMERIC(10,6) | 0.8 |
| `demand_kurtosis` | NUMERIC(10,6) | 1.2 |
| `zero_demand_months` | INTEGER | 3 |
| `total_demand_months` | INTEGER | 36 |
| `intermittency_ratio` | NUMERIC(10,6) | 0.08 |
| `variability_class` | TEXT | `smooth` |
| `demand_profile_ts` | TIMESTAMPTZ | 2026-04-01T... |

### Extended Columns (from `sql/120_add_unified_feature_columns.sql`)

Volume features (5 new):

| Column | Type |
|---|---|
| `iqr_demand` | NUMERIC(15,4) |
| `median_demand` | NUMERIC(15,4) |
| `min_demand` | NUMERIC(15,4) |
| `max_demand` | NUMERIC(15,4) |
| `total_demand` | NUMERIC(15,4) |

Trend features (5):

| Column | Type |
|---|---|
| `trend_slope` | NUMERIC(15,6) |
| `trend_slope_norm` | NUMERIC(15,6) |
| `trend_r2` | NUMERIC(10,6) |
| `trend_pct_change` | NUMERIC(15,4) |
| `trend_direction` | SMALLINT |

Seasonality features (4 beyond base):

| Column | Type |
|---|---|
| `seasonal_amplitude` | NUMERIC(10,6) |
| `seasonal_r2` | NUMERIC(10,6) |
| `yoy_correlation` | NUMERIC(10,6) |
| `seasonal_index_std` | NUMERIC(15,4) |

Periodicity (1):

| Column | Type |
|---|---|
| `periodicity_strength` | NUMERIC(10,6) |

Intermittency (1 new):

| Column | Type |
|---|---|
| `adi` | NUMERIC(10,4) |

Lifecycle (3):

| Column | Type |
|---|---|
| `cagr` | NUMERIC(10,4) |
| `recency_ratio` | NUMERIC(10,6) |
| `acceleration` | NUMERIC(10,6) |

Other (2):

| Column | Type |
|---|---|
| `outlier_count` | INTEGER |
| `acf_lag12` | NUMERIC(10,6) |
| `features_computed_ts` | TIMESTAMPTZ |

### Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_dim_sku_seasonality_profile ON dim_sku (seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_dim_sku_variability_class   ON dim_sku (variability_class);
```

### DDL Application Order

For fresh installs, `sql/005_create_dim_dfu.sql` includes all base columns. The migration files are idempotent (`ADD COLUMN IF NOT EXISTS`) for upgrades:

1. `sql/005_create_dim_dfu.sql` -- base table with seasonality + variability columns
2. `sql/022_add_demand_variability_columns.sql` -- idempotent migration for variability columns
3. `sql/120_add_unified_feature_columns.sql` -- extended feature columns (trend, periodicity, lifecycle, etc.)

---

## API Endpoints

Router: `api/routers/forecasting/sku_features.py`
Prefix: `/sku-features`
Tags: `sku-features`
Mounted in: `api/main.py`
Vite proxy: `/sku-features` in `frontend/vite.config.ts`

### GET /sku-features/summary

Aggregate statistics for the SKU feature portfolio.

**Response:**
```json
{
  "total_skus": 18432,
  "features_computed_ts": "2026-04-01T14:30:00+00:00",
  "seasonality_distribution": [
    {"profile": "none", "count": 8200},
    {"profile": "low", "count": 4100},
    {"profile": "moderate", "count": 3800},
    {"profile": "strong", "count": 2332}
  ],
  "variability_distribution": [
    {"class": "smooth", "count": 7500},
    {"class": "erratic", "count": 5200},
    {"class": "intermittent", "count": 3100},
    {"class": "lumpy", "count": 2632}
  ],
  "trend_distribution": [
    {"direction": -1, "count": 4000},
    {"direction": 0, "count": 9000},
    {"direction": 1, "count": 5432}
  ],
  "averages": {
    "cv_demand": 0.6823,
    "seasonal_amplitude": 0.3412,
    "trend_r2": 0.1284,
    "zero_demand_pct": 0.1123,
    "adi": 2.34,
    "cagr": 3.21,
    "demand_mean": 523.45,
    "seasonality_strength": 0.2891
  }
}
```

Cache: `public, max-age=120`

### GET /sku-features/list

Paginated, sortable, filterable list of per-SKU feature rows.

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int (1-1000) | 50 | Page size |
| `offset` | int (>=0) | 0 | Offset |
| `sort_by` | string | `item_id` | Column to sort by (validated against allowlist) |
| `sort_dir` | `asc` / `desc` | `asc` | Sort direction |
| `seasonality_profile` | string | none | Filter by profile (none/low/moderate/strong) |
| `variability_class` | string | none | Filter by class (smooth/erratic/intermittent/lumpy) |
| `trend_direction` | int | none | Filter by direction (-1/0/1) |
| `search` | string (max 200) | none | ILIKE search on `item_id` |

**Response:**
```json
{
  "total": 18432,
  "limit": 50,
  "offset": 0,
  "rows": [
    {
      "sku_ck": "ITEM001__LOC01",
      "item_id": "ITEM001",
      "loc": "LOC01",
      "ml_cluster": "C3",
      "seasonality_profile": "moderate",
      "variability_class": "erratic",
      "trend_direction": 1,
      "features_computed_ts": "2026-04-01T14:30:00+00:00",
      "demand_cv": 0.65,
      "seasonal_amplitude": 0.42,
      ...
    }
  ]
}
```

All 35+ feature columns from `dim_sku` are included in each row. The sortable column allowlist includes all feature columns plus `item_id`, `loc`, `ml_cluster`, `seasonality_profile`, `variability_class`, `trend_direction`, and `features_computed_ts`.

Cache: `public, max-age=120`

### GET /sku-features/distributions

Histogram distributions for key continuous features, computed server-side using PostgreSQL `width_bucket()`.

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `bins` | int (5-100) | 20 | Number of equal-width histogram bins |

**Histogram features:** `demand_cv`, `seasonal_amplitude`, `trend_r2`, `intermittency_ratio`, `adi`, `cagr`.

**Response:**
```json
{
  "bins": 20,
  "distributions": {
    "demand_cv": [
      {"bin_start": 0.0, "bin_end": 0.15, "count": 3200},
      {"bin_start": 0.15, "bin_end": 0.30, "count": 4100},
      ...
    ],
    "seasonal_amplitude": [...],
    ...
  }
}
```

Cache: `public, max-age=300`

### Variability Detail (legacy endpoint)

The inventory planning variability endpoints remain available:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/variability/summary` | Portfolio variability distribution |
| GET | `/inv-planning/variability/detail` | Per-DFU variability metrics |

Router: `api/routers/inv_planning_variability.py`

---

## UI Tab

### SKU Features Tab (`frontend/src/tabs/SkuFeaturesTab.tsx`)

A dedicated feature explorer tab with four visual sections:

**Section 1 -- Summary Cards (top row, 4 cards):**
- Total SKUs (with features computed)
- Last Computed (relative timestamp)
- Avg CV Demand (coefficient of variation)
- Avg Seasonal Amplitude

**Section 2 -- Distribution Charts (3 horizontal bar charts):**
- Seasonality Profile distribution (none/low/moderate/strong)
- Variability Class distribution (smooth/erratic/intermittent/lumpy)
- Trend Direction distribution (declining/flat/growing)

Color-coded: seasonality uses slate/blue/amber/red; variability uses emerald/orange/violet/rose; trend uses red/slate/green.

**Section 3 -- Feature Histograms (2x3 grid):**
- CV Demand, Seasonal Amplitude, Trend R-squared, Zero Demand %, ADI, CAGR
- Server-side binned via `width_bucket()`, rendered as vertical bar charts

**Section 4 -- Feature Table:**
- Sortable columns: SKU, Item, Location, Cluster, Seasonality, Variability, Trend, CV Demand, Seasonal Amp, Zero %, CAGR, Recency
- Filterable by: seasonality profile, variability class, trend direction, item_id search
- Paginated (50 rows per page)
- Classification columns rendered as color-coded badge pills

### API Query Module

`frontend/src/api/queries/sku-features.ts` provides:
- TypeScript types: `SkuFeaturesSummary`, `SkuFeatureRow`, `FeatureDistribution`, `SkuFeaturesListParams`
- Query key factory: `skuFeatureKeys` (summary, list, distributions)
- Stale times: summary 5min, list 1min, distributions 5min
- Fetch functions: `fetchSkuFeaturesSummary()`, `fetchSkuFeaturesList()`, `fetchSkuFeaturesDistributions()`

---

## Downstream Consumers

The features computed by this pipeline are read by multiple downstream subsystems. This section documents each consumer and the specific features it depends on.

### 1. Clustering Pipeline

**Consumer:** `scripts/ml/run_clustering_scenario.py`, `common/ml/clustering/`
**Mechanism:** Reads `data/clustering_features.csv` (written by Step 5 of the pipeline).
**Features used:** All numeric features are available as clustering input dimensions. The clustering algorithm selects a subset based on the scenario configuration. Typical inputs: `cv_demand`, `mean_demand`, `seasonality_strength`, `yoy_correlation`, `zero_demand_pct`, `trend_slope_norm`.
**Column mapping:** The clustering scenario script maps between naming conventions (see [Column Name Mapping](#column-name-mapping) below).

### 2. Backtest Feature Engineering

**Consumer:** `common/ml/backtest_framework.py` (`compute_dfu_features()`)
**Mechanism:** Computes a lightweight 4-feature vector per DFU during backtest runs (mean_demand, cv_demand, zero_demand_pct, seasonal_amplitude). This is a fast inline version; the full 35-feature set from `dim_sku` is also available to tree models as additional features.
**Features used:** `mean_demand`, `cv_demand`, `zero_demand_pct`, `seasonal_amplitude`.

### 3. Safety Stock Computation

**Consumer:** `scripts/compute_safety_stock.py`
**Mechanism:** Queries `dim_sku` directly for per-DFU demand statistics.
**Features used:** `demand_mean`, `demand_std`, `demand_cv`. The safety stock formula requires demand CV as the sigma-D input for statistical safety stock calculation. Each DFU's safety stock is calibrated to its specific demand volatility rather than a blanket factor.

### 4. ABC-XYZ Classification

**Consumer:** ABC-XYZ pipeline (`scripts/inventory/compute_abc_xyz.py`)
**Mechanism:** Reads `demand_cv` from `dim_sku` to assign XYZ class.
**Features used:** `demand_cv` -> `xyz_class` (X: low CV, Y: moderate, Z: high). The ABC dimension (revenue-based) is independent. The combination produces `abc_xyz_segment` (e.g., AX, BY, CZ).

### 5. Accuracy Views (Slicing)

**Consumer:** Accuracy materialized views (`agg_accuracy_by_dim`)
**Mechanism:** JOIN with `dim_sku` on seasonality columns.
**Features used:** `seasonality_profile` as a filter/group-by dimension. Enables planners to answer "How accurate are our forecasts for strong-seasonal items?"

### 6. Inventory Policy Assignment

**Consumer:** Replenishment policy engine
**Mechanism:** Reads `variability_class` and `abc_xyz_segment` from `dim_sku`.
**Features used:** `variability_class` influences review period, reorder point method, and safety stock multipliers. Lumpy items may use Croston-based methods; smooth items use standard (Q,R) policies.

### 7. Inventory Planning Variability Views

**Consumer:** `api/routers/inv_planning_variability.py`
**Mechanism:** Queries `dim_sku` for variability metrics.
**Features used:** `demand_cv`, `demand_mad`, `demand_mean`, `demand_std`, `demand_skewness`, `demand_kurtosis`, `demand_p50`, `demand_p90`, `zero_demand_months`, `total_demand_months`, `intermittency_ratio`, `variability_class`.

---

## Column Name Mapping

The feature computation library (`compute_time_series_features()`) and the `dim_sku` table use different naming conventions for some columns. The persistence layer in `persistence.py` handles this mapping.

| Feature name (compute output) | dim_sku column(s) | Notes |
|---|---|---|
| `mean_demand` | `demand_mean` | Volume: mean monthly demand |
| `std_demand` | `demand_std` | Volume: standard deviation |
| `cv_demand` | `demand_cv` | Volume: coefficient of variation |
| `median_demand` | `demand_p50` | Volume: 50th percentile |
| `seasonality_strength` | `seasonality_strength` | Same name (no mapping needed) |
| `peak_month` | `peak_month` | Same name |
| `zero_demand_pct` | `intermittency_ratio` | Intermittency: fraction of zero months |
| `months_available` | `total_demand_months` | Lifecycle: history depth |

Features not listed in the persistence mapping (e.g., `trend_slope`, `iqr_demand`, `adi`, `cagr`, etc.) are written to `dim_sku` columns that share the exact same name as the feature -- no mapping required.

The clustering scenario script (`scripts/run_clustering_scenario.py`) maintains a reverse mapping for reading `dim_sku` column names back into the feature convention:

```python
{"demand_mean": "mean_demand", "demand_std": "std_demand",
 "demand_cv": "cv_demand", "demand_mad": "demand_mad",
 "demand_p50": "median_demand"}
```

---

## Dependencies

### Upstream

| Dependency | Role |
|---|---|
| `fact_sales_monthly` | Source of monthly demand quantities (item_id, customer_group, loc, startdate, qty) |
| `dim_sku` | Target table for feature persistence; also provides sku_ck via JOIN during data load |
| `get_planning_date()` | Determines the lookback window end date |

### Libraries

| Library | Used by | Purpose |
|---|---|---|
| pandas | All modules | DataFrame manipulation, groupby aggregation |
| numpy | `features.py` | Statistical computations, FFT, polyfit, corrcoef |
| scipy.stats | `features.py` | Skewness, kurtosis (optional -- graceful fallback when absent) |
| psycopg (v3) | `compute.py`, `persistence.py` | Database connectivity, COPY protocol |
| multiprocessing | `compute.py` | Parallel feature computation |

### Config

| File | Purpose |
|---|---|
| `config/forecasting/sku_features_config.yaml` | Thresholds for classification, history window, parallelism |
| `config/planning_config.yaml` | Planning date (via `get_planning_date()`) |

---

## See Also

- [01-sku-clustering](01-sku-clustering.md) -- Clustering consumes pre-computed features from `clustering_features.csv`
- [../02-forecasting/01-accuracy-kpis](../02-forecasting/01-accuracy-kpis.md) -- Accuracy slicing by seasonality profile
- [../02-forecasting/03-backtest-framework](../02-forecasting/03-backtest-framework.md) -- Backtest inline feature computation (lightweight version)
- [../04-inventory/02-demand-variability](../04-inventory/02-demand-variability.md) -- Lead time variability (complementary to demand variability)
- [../04-inventory/03-safety-stock](../04-inventory/03-safety-stock.md) -- Primary consumer of `demand_cv`, `demand_mean`, `demand_std`
- [../04-inventory/07-abc-xyz-supplier](../04-inventory/07-abc-xyz-supplier.md) -- XYZ classification derived from `demand_cv`
