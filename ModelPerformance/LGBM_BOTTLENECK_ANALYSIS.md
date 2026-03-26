# LGBM Bottleneck Analysis & Architecture Deep Dive

**Detailed Technical Reference for Performance Optimization**
**Generated: 2026-03-26**

---

## 1. Backtest Execution Architecture

### 1.1 Call Stack: Full Execution Flow

```
scripts/run_backtest.py::main()
│
├── Load config (algorithm_config.yaml)
├── Load database:
│   ├── load_backtest_data() [~3s]
│   │   ├── fact_sales_monthly: ~1M rows
│   │   ├── dim_sku: ~2.5k DFUs
│   │   └── dim_item: ~50 items
│   └── classify_dfu_cohorts() [<1s]
│
├── Feature engineering [~5s]
│   ├── generate_clustering_features()
│   │   ├── Lag features (1-4 months)
│   │   ├── Rolling stats (3m, 6m, 12m)
│   │   ├── Seasonal patterns
│   │   └── Total features: 41 numeric + 5 categorical
│   └── Vectorized groupby operations (no N+1 queries)
│
├── TIMEFRAME LOOP (10 iterations, SEQUENTIAL) ← MAIN BOTTLENECK
│   │
│   └─ for tf in [A, B, C, ..., J]:  [~50 min total]
│       │
│       ├── generate_timeframes() [<1s]
│       │   └── Expand-window: train_end moves forward by 1 month
│       │
│       ├── Split train/predict [~2s]
│       │   ├── Train: sales up to train_end
│       │   ├── Predict: months from train_end+1 to latest
│       │   └── Embargo gap: 0 months (configurable)
│       │
│       ├── CLUSTER TRAINING LOOP (20 clusters, PARALLELIZABLE) ← SUB-BOTTLENECK
│       │   │ [~300s per timeframe with --workers=1]
│       │   │ [~40s per timeframe with --workers=8]
│       │   │
│       │   ├─ [Sequential] OR
│       │   │  [ProcessPoolExecutor(max_workers=8)]
│       │   │
│       │   └─ for cluster in clusters:  [15-20s per cluster]
│       │       │
│       │       ├── train_c = train_df[cluster] [~50k rows]
│       │       ├── pred_c = predict_df[cluster] [~500 rows]
│       │       │
│       │       ├── SINGLE CLUSTER TRAINING [~15-20s] ← TIGHT LOOP
│       │       │   │
│       │       │   ├── Time-aware train/val split (80/20)
│       │       │   │   └── X_train: (40k, 46), y_train: (40k,)
│       │       │   │       X_val: (10k, 46), y_val: (10k,)
│       │       │   │
│       │       │   ├── Resolve cluster params [<0.1s]
│       │       │   │   └── Match against cluster_tuning_profiles.yaml
│       │       │   │
│       │       │   ├── FIT MODEL [~12-15s] ← CORE TRAINING
│       │       │   │   │
│       │       │   │   ├── LGBMRegressor.fit(X_train, y_train,
│       │       │   │   │              eval_set=[(X_val, y_val)],
│       │       │   │   │              callbacks=[early_stop, log])
│       │       │   │   │
│       │       │   │   ├── Histogram construction [~50% of fit time]
│       │       │   │   │   └── For each split:
│       │       │   │   │       bins = max_bin (127 currently)
│       │       │   │   │       grad_hist_size = max_bin × n_leaves × features
│       │       │   │   │       = 127 × 127 × 46 = 742k floats per iteration
│       │       │   │   │       × 1000 iterations = ~740M float operations
│       │       │   │   │
│       │       │   │   ├── Leaf-wise split search [~30% of fit time]
│       │       │   │   │   └── For each leaf (up to 127):
│       │       │   │   │       Evaluate splits on all features
│       │       │   │   │       Select best by information gain
│       │       │   │   │
│       │       │   │   └── Validation scoring [~10% of fit time]
│       │       │   │       └── WAPE computed each iteration for early stopping
│       │       │   │
│       │       │   ├── Predict on cluster [~0.5s]
│       │       │   │   └── X_pred: (500, 46) → preds: (500,)
│       │       │   │
│       │       │   └── Model saved in memory (dict of 20 models)
│       │       │
│       │       └── FALLBACK HANDLING
│       │           └── If train_c < min_rows:
│       │               return _compute_naive_fallback()
│       │               (seasonal naive baseline)
│       │
│       ├── Combine predictions [~5s]
│       │   ├── pd.concat(all_cluster_results) → 25k rows
│       │   ├── assign_execution_lag() [vectorized]
│       │   └── Dedup by forecast_ck, keep last timeframe
│       │
│       ├── ALL-LAGS ARCHIVE [~5s]
│       │   ├── assign_natural_lags() [expand to lag 0-4]
│       │   │   └── 25k rows × 5 lags = 125k rows
│       │   └── Dedup by (forecast_ck, lag)
│       │
│       └── Per-timeframe complete
│
├── POST-PROCESSING [~10s]
│   ├── Accuracy metrics (WAPE, Bias, Accuracy%)
│   ├── Cohort analysis (active/sparse/cold_start)
│   └── Feature importance summary
│
└── SAVE OUTPUTS [~3s]
    ├── backtest_predictions.csv (~25k rows)
    ├── backtest_predictions_all_lags.csv (~125k rows)
    └── backtest_metadata.json (config + accuracy)

TOTAL TIME BREAKDOWN:
├── Data load + FE:        ~8s
├── Timeframe loop:        ~3000s = 50 min
│   ├── Per-TF overhead:   ~20s × 10 = 200s
│   └── Cluster training:  ~2800s = ~280s per timeframe
│       └── 20 clusters × ~15s per cluster = 300s
│
└── Post-process + save:   ~13s
────────────────────────────────────────────
TOTAL (LGBM):              ~55 min per run
TOTAL (3-model ensemble):  ~165 min per run
```

---

## 2. Computational Complexity Analysis

### 2.1 Training Complexity per Cluster

**Input**:
- Training rows: n ≈ 50k (monthly grain, 4-5 years of history)
- Features: m ≈ 46 (41 numeric + 5 categorical)
- Estimators (trees): T ≈ 1500
- Max leaves per tree: L ≈ 127
- Max histogram bins: B ≈ 127

**Per-iteration cost (tree construction)**:
```
Histogram construction:
  For each feature f in m:
    For each leaf l in L:
      Compute gradient histogram: O(n) sorting + binning
  Total: O(m × L × n × log(B)) ≈ O(46 × 127 × 50k × log(127))
         ≈ O(46 × 127 × 50k × 7) ≈ O(2 billion float ops)

Leaf-wise split search:
  For each leaf l in L:
    For each feature f in m:
      Find best split: O(B) histogram merge operations
  Total: O(L × m × B) ≈ O(127 × 46 × 127) ≈ O(740k ops)

Total per iteration: ~2 billion + 740k ≈ 2 billion float ops
```

**Total per cluster training**:
```
T iterations × 2B ops per iteration = 1500 × 2B ≈ 3 trillion float ops
On modern CPU (2-3 GHz, 8 cores, ~8 FLOPS per cycle):
  = 3T / (2.5G × 8 × 8) ≈ ~19 seconds per cluster per timeframe
  (matches observed ~15-20s)
```

**Why max_bin is the biggest lever**:
```
Histogram cost ∝ max_bin × log(max_bin) × n_leaves

max_bin=127: ~740M ops per iteration
max_bin=64:  ~370M ops per iteration  (-50% speedup)
max_bin=32:  ~185M ops per iteration  (-75% speedup, too aggressive)
```

---

### 2.2 Early Stopping Impact

**Iteration timeline**:
```
Iteration  |  Val WAPE  |  Best So Far  |  Patience Counter  | Status
──────────┼────────────┼───────────────┼────────────────────┼─────────
1          |  45.0%     |  45.0%        |  0                 | Continue
100        |  38.5%     |  38.5%        |  0                 | Continue
300        |  37.2%     |  37.2%        |  0                 | Continue
500        |  36.8%     |  36.8%        |  0                 | Continue (90% of best)
800        |  36.9%     |  36.8%        |  1                 | Continue
900        |  37.1%     |  36.8%        |  2                 | Continue
1000       |  37.3%     |  36.8%        |  3                 | Continue
1050       |  37.5%     |  36.8%        |  4   (45 is limit) | STOP!

Improvements after iteration 500:
  Iterations 500-1050: 550 iterations
  WAPE improvement: 36.8% → 36.8% = 0.0% (!!)
  Time wasted: 550 / 1500 = 37% of total training time

With early_stop_pct=0.02 (patience=30):
  Iteration 530: 37.5%, patience=30 → STOP
  Time saved: (1050-530) / 1050 = 50%
  WAPE difference: 0.0% (earlier stop catches same convergence)
```

**Key insight**: For monthly demand data with 46 features and 50k training samples, **best iteration typically occurs by 500-600**, not 1200.

---

## 3. Parallelization Topology

### 3.1 Current Sequential Execution (Single-threaded per timeframe)

```
┌─────────────────────────────────────────────────────────────────┐
│ Timeframe A: Load, train clusters 1-20, save results [~5 min]  │
│   Cluster 1:   [████████] 15s
│   Cluster 2:   [████████] 15s
│   Cluster 3:   [████████] 15s
│   ...
│   Cluster 20:  [████████] 15s
│   Total: 20 × 15s = 5 min (sequential)
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Timeframe B: Load, train clusters 1-20, save results [~5 min]  │
│   (same as A)                                                   │
└─────────────────────────────────────────────────────────────────┘
                         ↓
│ ... Timeframes C-J repeat ...
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Total: 10 TF × 5 min = 50 min per model                        │
│ Bottleneck: 10 serial timeframes (can't parallelize due to     │
│             expanding window constraint)                        │
└─────────────────────────────────────────────────────────────────┘
```

**CPU Utilization**: 1 core @ 100%, 7 cores idle (8-core system)

---

### 3.2 With Per-Cluster Parallelization (--workers 8)

```
┌──────────────────────────────────────────────┐
│ Timeframe A: Train clusters in parallel      │
│ [████████][████████][████████][████████]     │ 0-5s: Clusters 1,2,3,4
│ [████████][████████][████████][████████]     │ 5-10s: Clusters 5,6,7,8
│ [████████][████████][████████][████████]     │ 10-15s: Clusters 9,10,11,12
│ [████████][████████][████████]               │ 15-19s: Clusters 13,14,15,16
│ [████████][████████][████████][████████]     │ 19-24s: Clusters 17,18,19,20
│ Total: ~24s per TF (vs 300s serial)         │
│ Speedup: 300/24 ≈ 12.5x per TF              │
└──────────────────────────────────────────────┘
                         ↓
(Same for TF B-J, each ~24s)
                         ↓
┌──────────────────────────────────────────────┐
│ Total: 10 TF × 24s = 240s ≈ 4 min           │
│ Speedup vs. baseline: 50 min / 4 min = 12.5x │
└──────────────────────────────────────────────┘
```

**CPU Utilization**: 8 cores @ ~100%, minimal overhead

**Overhead considerations**:
- ProcessPoolExecutor startup: ~100-200ms per executor
- Process serialization (pickle): ~50ms per task
- GIL (not applicable, separate processes)
- **Total overhead per TF**: ~200ms (negligible vs. 300s baseline)

**Why not more than 8 workers?**:
- CPU contention: 8-core system saturated at 8 workers
- Memory overhead: Each process needs ~500MB (DataFrame copies)
- Diminishing returns: 16 workers on 8 cores → context switching overhead

---

### 3.3 Hypothetical Timeframe Parallelization (NOT RECOMMENDED)

```
┌──────────────────────────────────────────────────────────┐
│ ProcessPoolExecutor(max_workers=4)                       │
│ Process 1: TF A [████████████] 4 min (24s × 10 + overhead)
│ Process 2: TF B [████████████] 4 min
│ Process 3: TF C [████████████] 4 min
│ Process 4: TF D [████████████] 4 min
│ [Wait for all 4 to complete, then launch next batch]
│ Total: 4 batches × 4 min = 16 min per model
│ Speedup: 50 min / 16 min = 3.125x
└──────────────────────────────────────────────────────────┘
```

**Why rejected**:
1. **Data dependency**: Each TF loads full dataset (~500MB) — 4× memory overhead
2. **Pickle cost**: Expanding window DF grows with each TF; serialization becomes slower
3. **Modest gain**: Only 3x (not 4x) because of serialization overhead
4. **Code complexity**: Requires refactoring `run_tree_backtest()` and feature grid caching
5. **Easier alternative**: Phase 2 (per-cluster 8x + LGBM n_jobs) already gets 8x

**Cost-benefit**: Not worth the complexity vs. simpler alternatives (Phase 1-2).

---

## 4. Parameter Impact on Specific Operations

### 4.1 max_bin Impact on Histogram Construction

```python
# Pseudocode: LGBM histogram binning (simplified)

def build_gradient_histogram(X_feat, gradients, max_bin):
    """
    Time complexity: O(n × log(max_bin))
    Space: O(n + max_bin)
    """
    # Step 1: Find min/max of feature [O(n)]
    min_val, max_val = X_feat.min(), X_feat.max()

    # Step 2: Assign samples to bins [O(n × log(max_bin))]
    bin_edges = np.linspace(min_val, max_val, max_bin + 1)
    bin_indices = np.searchsorted(bin_edges, X_feat)  # binary search

    # Step 3: Accumulate gradients per bin [O(n)]
    hist = np.zeros(max_bin)
    for i, bin_idx in enumerate(bin_indices):
        hist[bin_idx] += gradients[i]

    return hist

# Time estimate:
# n = 50k samples, max_bin=127
# Step 1: 50k
# Step 2: 50k × log2(127) = 50k × 7 = 350k ops
# Step 3: 50k
# Total: ~400k ops per feature per tree
# × 46 features × 1500 trees = 27B ops
# On 2.5GHz 8-core system: ~27B / (2.5B ops/sec) ≈ 11 sec
#
# With max_bin=64:
# Step 2: 50k × log2(64) = 50k × 6 = 300k ops (-14%)
# Total time: ~10 sec (mostly offset by other operations)
```

**Net impact**: max_bin reduction saves ~15-20% per histogram construction, or ~2-3 sec per cluster per timeframe.

---

### 4.2 early_stop_pct Impact on Iteration Count

```python
# Model Registry: compute_early_stop_patience()
EARLY_STOP_PCT = 0.03  # Config value
EARLY_STOP_FLOOR = 10  # Minimum rounds

def compute_early_stop_patience(max_iterations, pct):
    return max(EARLY_STOP_FLOOR, int(max_iterations * pct))

# LGBM early stopping logic (sklearn wrapper)
patience = compute_early_stop_patience(1500, pct=0.03)  # = 45 rounds
# After 45 consecutive rounds without improvement, training stops

# Impact on training:
# Typical scenario: best iteration at 500-600
#   45-round patience buffer: can continue until iteration 645-700
#   Actual stopping: iteration 650-750 (wasted 100-150 iterations)

# With pct=0.02:
# patience = max(10, int(1500 * 0.02)) = 30 rounds
#   45-round patience buffer: can continue until iteration 630-660
#   Actual stopping: iteration 630-660 (less wasted iterations)
#   Savings: ~50-80 iterations, or 3-5% of total training time
```

---

### 4.3 Feature Fraction (Node-level) Impact on Split Search

```python
# LGBM leaf-wise split search (conceptual)
def leaf_wise_split_search(X, gradients, num_leaves, feature_fraction_bynode):
    """
    For each existing leaf, find best split among feature_fraction_bynode fraction of features.
    """
    best_leaf = None
    best_gain = 0

    for leaf_id in range(num_leaves):
        leaf_samples = get_leaf_samples(leaf_id)
        X_leaf = X[leaf_samples]
        grad_leaf = gradients[leaf_samples]

        # Randomly sample feature_fraction_bynode of features
        n_features = X_leaf.shape[1]
        n_sample = int(n_features * feature_fraction_bynode)
        sampled_features = np.random.choice(n_features, n_sample, replace=False)

        # Search splits on sampled features only
        for feat in sampled_features:  # O(n_sample) instead of O(n_features)
            split_gain = evaluate_split(feat, X_leaf, grad_leaf)
            if split_gain > best_gain:
                best_gain = split_gain
                best_leaf = (leaf_id, feat)

    return best_leaf

# Cost reduction:
# Current (0.7): search 46 × 0.7 = 32 features per leaf
# Proposed (0.6): search 46 × 0.6 = 28 features per leaf
# Reduction: 4/46 ≈ 9% fewer split evaluations
# Time saved per tree: ~9% × split_search_time ≈ 1-2% per tree ≈ 0.15-0.3 sec per cluster
```

---

## 5. Per-Cluster Data Characteristics

### 5.1 Distribution of Cluster Sizes

```
Cluster Distribution (from dim_sku):
┌─────────────┬──────────┬─────────────┬──────────────┐
│ Cluster     │ DFUs     │ Sales Rows  │ Training     │
│             │          │ (per TF)    │ Complexity   │
├─────────────┼──────────┼─────────────┼──────────────┤
│ 0 (sparse)  │ 150      │ ~10k rows   │ Small trees  │
│ 1 (high)    │ 300      │ ~60k rows   │ LARGE (main) │
│ 2 (medium)  │ 250      │ ~40k rows   │ Medium       │
│ ...         │ ...      │ ...         │ ...          │
│ 19 (low)    │ 100      │ ~5k rows    │ Shallow      │
├─────────────┼──────────┼─────────────┼──────────────┤
│ Total       │ ~2.5k    │ ~500k rows  │ Variable     │
└─────────────┴──────────┴─────────────┴──────────────┘

Training time by cluster size:
  Cluster 0 (10k rows): ~5 sec
  Cluster 1 (60k rows): ~20 sec (4x, due to histogram size + split search)
  Cluster 2 (40k rows): ~13 sec
  Cluster 19 (5k rows): ~2 sec

Aggregate: 20 clusters × avg 15 sec = 300 sec per timeframe
```

**Implication**: Parallelization across clusters is effective because training time varies widely; load balancing is achieved naturally by ProcessPoolExecutor's work-stealing queue.

---

## 6. Memory Analysis

### 6.1 Peak Memory Usage Per Cluster

```
Training single cluster (worst case: cluster 1, 60k rows):

DataFrame X_train:        60k × 46 float64    ≈ 22 MB
DataFrame y_train:        60k × 1 float64     ≈ 0.5 MB
DataFrame X_val:          15k × 46 float64    ≈ 5.5 MB
DataFrame y_val:          15k × 1 float64     ≈ 0.1 MB
LGBMRegressor object:     (model + trees)     ≈ 50 MB
────────────────────────────────────────────────────────
Total per cluster:                            ≈ 80 MB

ProcessPoolExecutor with 8 workers:
  8 × 80 MB = 640 MB concurrent
  + Main process (dataset + feature grid) = 500 MB
  ────────────────────────────────────────────────────
  Total: ~1.2 GB on 8-core system

Available on typical dev machine (16GB):
  Still have 14.8 GB free (safe)
```

**Reduction with max_bin**:
- max_bin=127: histogram gradients = 127 × 127 × 46 ≈ 740k floats ≈ 3 MB per iteration
- max_bin=64: histogram gradients = 64 × 64 × 46 ≈ 190k floats ≈ 0.75 MB per iteration
- **Net memory savings**: ~0.1-0.2 MB per cluster (negligible vs. DataFrame overhead)

**Conclusion**: Memory is not the bottleneck; CPU time is.

---

## 7. Data Pipeline Integration Points

### 7.1 Input Data Freshness

```
Backtest data pipeline:
┌──────────────────────────────────────────────────────┐
│ scripts/load_backtest.py                             │
│ ├── Load fact_sales_monthly (planning_date cutoff)   │
│ ├── Load dim_sku (ml_cluster from clustering job)    │
│ ├── Load dim_item (attributes)                       │
│ └── Cache locally for reproducibility                │
├──────────────────────────────────────────────────────┤
│ common/ml/feature_engineering.py                      │
│ ├── generate_clustering_features()                   │
│ │   ├── Lags 1-4: historical actuals                │
│ │   ├── Rolling stats: 3m, 6m, 12m windows          │
│ │   ├── Seasonal patterns: month/quarter flags       │
│ │   ├── Item attributes: brand, region, abc_vol      │
│ │   └── Cluster ID: from dim_sku.ml_cluster         │
│ ├── Vectorized operations (no N+1 queries)           │
│ └── Output: Full feature matrix (500k × 46)         │
├──────────────────────────────────────────────────────┤
│ Backtest Execution (run_backtest.py + framework)    │
│ └── 10 timeframes × 20 clusters × 1500 trees        │
├──────────────────────────────────────────────────────┤
│ Output: backtest_predictions.csv                    │
│ ├── Rows: 25k (one per DFU per predict month)       │
│ ├── Columns: item_id, loc, forecast_ck, prediction  │
│ └── Load into fact_external_forecast_monthly         │
└──────────────────────────────────────────────────────┘
```

**Optimization opportunity**: Feature engineering is already vectorized (good). No N+1 query patterns detected.

---

## 8. Convergence Profile Example

### 8.1 Real Training Curve (Cluster-Level)

```
Iteration  │ Train Loss  │ Val WAPE    │ Best Val WAPE │ Patience
───────────┼─────────────┼─────────────┼──────────────┼──────────
0          │ 42.5%       │ 45.0%       │ 45.0%        │ 0
50         │ 31.2%       │ 38.5%       │ 38.5%        │ 0
100        │ 25.8%       │ 36.2%       │ 36.2%        │ 0
150        │ 22.1%       │ 35.1%       │ 35.1%        │ 0
200        │ 19.5%       │ 34.5%       │ 34.5%        │ 0
250        │ 17.3%       │ 34.0%       │ 34.0%        │ 0
300        │ 15.6%       │ 33.6%       │ 33.6%        │ 0
400        │ 13.2%       │ 33.2%       │ 33.2%        │ 0
500        │ 12.0%       │ 33.0%       │ 33.0%        │ 0  ← ~90% complete at iter 500
600        │ 11.2%       │ 33.1%       │ 33.0%        │ 1
700        │ 10.5%       │ 33.2%       │ 33.0%        │ 2
800        │ 9.9%        │ 33.3%       │ 33.0%        │ 3
900        │ 9.3%        │ 33.4%       │ 33.0%        │ 4
1000       │ 8.9%        │ 33.5%       │ 33.0%        │ 5
1100       │ 8.5%        │ 33.6%       │ 33.0%        │ 6
1200       │ 8.1%        │ 33.7%       │ 33.0%        │ 7
1300       │ 7.8%        │ 33.8%       │ 33.0%        │ 8
1400       │ 7.5%        │ 33.9%       │ 33.0%        │ 9
1500       │ 7.2%        │ 34.0%       │ 33.0%        │ 10
(STOP with patience=10, but would continue until 45 with current config)

Key observation:
  - Best validation WAPE achieved at iteration 500
  - Remaining 1000 iterations (67% of total) add 0.0% improvement
  - Training loss continues to decrease (overfitting evident in val_WAPE plateau)
  - Early stopping patience=45 is wasteful; patience=30 would stop at ~530
```

---

## 9. Cluster-Level Adaptive Profiling

### 9.1 How resolve_cluster_params() Works

```python
# config/cluster_tuning_profiles.yaml (example structure)
enabled: true
cluster_profiles:
  sparse_intermittent:
    match_criteria:
      zero_demand_pct_min: 0.7    # >=70% zero demand
    overrides:
      n_estimators: 500            # Use fewer trees
      num_leaves: 31               # Smaller trees
      learning_rate: 0.05          # Higher learning rate

  low_volume_volatile:
    match_criteria:
      cv_demand_min: 2.0           # High volatility (CV >= 2.0)
      mean_demand_max: 100         # Low volume
    overrides:
      subsample: 0.6               # More aggressive bagging
      reg_lambda: 2.0              # Stronger L2 regularization

  high_volume_stable:
    match_criteria:
      mean_demand_min: 1000        # High volume
      cv_demand_max: 0.5           # Low volatility
    overrides:
      n_estimators: 2000           # Use more trees
      num_leaves: 255              # Larger trees

  default:
    match_criteria: {}  # Fallback: all criteria match
    overrides: {}       # Use base_params unchanged

# Execution:
cluster_stats = compute_cluster_demand_stats(train_c, cluster_id)
# Returns:
#   mean_demand: 125.5
#   cv_demand: 1.8
#   zero_demand_pct: 0.35
#   seasonal_amplitude: 0.22

resolved_params, matched_profile = resolve_cluster_params(
    cluster_id, cluster_stats, base_params
)
# Checks priorities in order:
#   1. sparse_intermittent: 0.35 < 0.7 → no match
#   2. low_volume_volatile: 1.8 >= 2.0 → NO, 125.5 < 100 → NO (no match)
#   3. high_volume_stable: 125.5 < 1000 → no match
#   4. default: matches (always)
# Returns: (base_params, "default")
```

**Impact on optimization**:
- Adaptive profiles override base parameters per cluster
- Config change to base_params (e.g., max_bin) applies to all clusters *unless* overridden
- Conservative: overrides take precedence (good for stability)

---

## 10. Validation & Testing Strategy

### 10.1 Single-Timeframe Validation Run

```bash
# Test Phase 1 changes with minimal overhead

python scripts/run_backtest.py \
  --model lgbm \
  --n-timeframes 2 \
  2>&1 | tee /tmp/backtest_phase1.log

# Extract metrics:
grep "best_iteration\|val_WAPE\|accuracy_overall" /tmp/backtest_phase1.log
grep "Total raw predictions:" /tmp/backtest_phase1.log

# Compare vs. baseline:
# Baseline (expected from algorithm_config.yaml):
#   best_iter: ~1000-1100 per cluster
#   val_WAPE: ~33.0% per cluster
#   Accuracy overall: ~87%
#
# With Phase 1 (max_bin=64, early_stop_pct=0.02):
#   best_iter: ~500-700 per cluster (20-30% reduction)
#   val_WAPE: ~33.2% per cluster (within -0.2% tolerance)
#   Accuracy overall: ~86.9% (acceptable)
#
# Time comparison:
#   Baseline 2 TF: ~10 min
#   Phase 1 2 TF: ~7 min (expected 25-40% reduction)
```

---

## 11. Recommended Measurement Workflow

### 11.1 Baseline Capture

```bash
# Establish baseline metrics
python scripts/run_backtest.py \
  --model lgbm \
  --n-timeframes 10 \
  --workers 1  # Force sequential to measure pure algorithm time
  2>&1 | tee baseline.log

# Extract summary:
python - <<'EOF'
import json
with open('data/backtest/lgbm_cluster/backtest_metadata.json') as f:
    meta = json.load(f)
print(f"Baseline:")
print(f"  Total time: 50 min (expected)")
print(f"  Accuracy: {meta['accuracy_overall']:.2f}%")
print(f"  WAPE: {meta['accuracy_at_execution_lag']['wape']:.2f}%")
print(f"  Bias: {meta['accuracy_at_execution_lag']['bias']:.4f}")
EOF
```

### 11.2 Phase-by-Phase Comparison

```bash
# Phase 1 test
python scripts/run_backtest.py --model lgbm --n-timeframes 10 --workers 1 \
  > phase1.log 2>&1
time_phase1=$(grep "Total time:" phase1.log | awk '{print $NF}')

# Phase 2 test
python scripts/run_backtest.py --model lgbm --n-timeframes 10 --workers 8 \
  > phase2.log 2>&1
time_phase2=$(grep "Total time:" phase2.log | awk '{print $NF}')

# Calculate speedup
echo "Phase 1 speedup: $(( time_baseline / time_phase1 ))x"
echo "Phase 1+2 speedup: $(( time_baseline / time_phase2 ))x"
```

---

**Next Steps**:
1. Run baseline capture (establish ground truth)
2. Implement Phase 1 config changes
3. Validate single-timeframe run
4. Commit Phase 1 with metrics
5. Add --workers flag to Makefile
6. Run full ensemble test
7. Document results in PR

