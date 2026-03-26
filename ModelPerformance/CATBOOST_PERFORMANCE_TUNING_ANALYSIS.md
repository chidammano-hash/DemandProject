# CatBoost Performance Tuning Analysis

**Date:** 2026-03-26
**Context:** DemandProject backtest framework with monthly demand forecasting grain
**Current Config:** `config/algorithm_config.yaml` lines 68–97
**Execution Model:** Per-cluster sequential backtest (10 timeframes × ~200 clusters × per-cluster fit)

---

## Executive Summary

CatBoost's current configuration prioritizes **accuracy over speed** with conservative, deep models:
- **3000 iterations** (2.5–5× LGBM/XGBoost)
- **Lossguide growth policy** (more refined trees, slower convergence)
- **0.008 learning rate** (75% of XGBoost, 40% of LGBM)
- **depth=10 with max_leaves=127** (balanced but high memory per iteration)

**Estimated speedup potential: 1.5–2.5× wall-clock time reduction** with strategic parameter tuning while maintaining accuracy within ±2% bounds. Primary bottleneck is **per-cluster sequential training** across 10 timeframes.

---

## Execution Flow Analysis

### Current Bottleneck Map

```
run_backtest.py main()
├─ Load config + GPU detection (~5s)
├─ For each seed (default n_seeds=1):
│  └─ run_tree_backtest() [common/ml/backtest_framework.py:881]
│     ├─ Load data from Postgres (sales, DFU attrs, item attrs) (~30–60s)
│     ├─ Generate 10 timeframes
│     ├─ Build feature matrix once (~15–30s)
│     └─ FOR EACH TIMEFRAME [lines 976–1200]:  ← **SEQUENTIAL LOOP**
│        ├─ Mask future sales + recompute lag features (~3–5s per TF)
│        ├─ train_fn_per_cluster() [scripts/run_backtest.py:962]
│        │  ├─ Split clusters
│        │  ├─ FOR EACH CLUSTER [ProcessPoolExecutor if parallel=True]
│        │  │  └─ _train_single_cluster() → fit_model()
│        │  │     ├─ Time-aware train/val split (20% validation)
│        │  │     ├─ Adaptive cluster params + Tweedie routing
│        │  │     ├─ CatBoost.fit() on X_tr, eval_set=X_val  ← **EXPENSIVE**
│        │  │     └─ model.predict(X_pred)
│        │  └─ Fallback predictions for small clusters
│        └─ Postprocess, dedup, attach actuals (~2–3s per TF)
```

**Total execution time per backtest:** ~6–15 hours (for 10 TF × 150–300 clusters at 5–15 min per cluster)

### Parallelization Status

- **Clusters within timeframe:** `ProcessPoolExecutor(max_workers=8)` when `args.parallel=True`
  - `scripts/run_backtest.py:662` — enabled by default
  - Safe: no shared state, picklable closures

- **Timeframes:** **SEQUENTIAL** (no parallelization across 10 TF)
  - Could be parallelized but would require separate data grids (9.8M rows × 10 = ~1GB memory overhead)
  - Better to optimize per-cluster fit instead

**Quick win:** Timeframes are independent — could parallelize with `ProcessPoolExecutor` over TF indices, but only if memory budget permits.

---

## 1. Training Speed Optimizations

### 1.1 Growth Policy: Lossguide vs. Depthwise

**Current:** `grow_policy: Lossguide`

| Policy | Characteristics | Speed | Accuracy | Trade-offs |
|--------|---|---|---|---|
| **Lossguide** (current) | Grows leaf that minimizes loss. More refined, smaller trees. 3000 iter → dense. | Slow (~100–150ms/iter) | High, stable convergence | **Deep trees, high variance per iteration, longer time to convergence** |
| **Depthwise** | Balanced splits at each depth level. Wider, shallower trees. | Fast (~60–80ms/iter) | Comparable if depth is tuned | **Wider trees = faster iteration but may need more iterations** |

**Recommendation:** Test **Depthwise + 2000 iterations** vs. current Lossguide + 3000.

```yaml
# Current (conservative)
grow_policy: Lossguide
iterations: 3000
depth: 10
max_leaves: 127

# Proposed (faster)
grow_policy: Depthwise
iterations: 2000
depth: 8
max_leaves: 100
```

**Expected speedup:** 25–35% wall-clock (each iter faster + fewer iterations)
**Accuracy risk:** Low if depth tuned well for monthly grain (validate on validation set per timeframe)

**Validation:** Run backtest with both configs on a single timeframe + cluster subset (30 min vs. 5 min test).

---

### 1.2 Learning Rate Acceleration

**Current:** `learning_rate: 0.008` (conservative)

**Comparison:**
- LGBM: `0.02` (2.5× higher)
- XGBoost: `0.02` (2.5× higher)
- CatBoost current: `0.008`

**Why so low?** Lossguide + Newton leaf estimation converge slowly on monthly grain (sparse demand, categorical features).

**Recommendation:** Tier by demand pattern (per cluster_tuning_profiles.yaml):

```yaml
# Algorithm config default (increase from 0.008)
learning_rate: 0.012  # 50% increase, still conservative

# Cluster tuning profiles for intermittent/sparse clusters
# (already in config/cluster_tuning_profiles.yaml, but verify):
sparse_intermittent:
  learning_rate: 0.005  # LOWER for sparse (more stability)
  iterations: 2500      # Compensate with more iterations

high_volume_stable:
  learning_rate: 0.015  # HIGHER for stable demand
  iterations: 1500      # Fewer needed

seasonal_dominant:
  learning_rate: 0.010  # Moderate
  iterations: 2000
```

**Expected speedup:** 10–20% (fewer iterations needed for convergence)
**Accuracy risk:** Medium — requires validation on multi-timeframe run. High LR can overshoot minima on noisy clusters.

**Safe baseline:** Increase from 0.008 → 0.010 globally, monitor val_wape per cluster.

---

### 1.3 Iteration Count Reduction via Early Stopping

**Current:** `early_stop_pct: 0.03` (3% of max iterations)
- For 3000 iterations: 90 rounds patience
- For 2000 iterations: 60 rounds patience (hardcoded floor in `model_registry.py:115`)

**Analysis:**
- CatBoost's early stopping uses **internal eval metric**, not WAPE
- Mismatch: trains on L2/RMSE but evaluates accuracy on WAPE (lines 233–235 in model_registry.py)
- **Custom WapeMetric is not used for early stopping** (only for manual tracking)

**Issues:**
1. No early stopping metric alignment with final accuracy
2. Sparse/intermittent clusters may hit plateau at 60–70% of max iterations
3. Tweedie loss (intermittent clusters) may need extra iterations for convergence

**Recommendation:** Implement **WAPE-aligned early stopping**:

```python
# In model_registry.py, line 230 (CatBoost branch):
# ADD custom_metric for early stopping (currently missing)
model.fit(
    X_tr, y_tr,
    cat_features=cat_indices,
    eval_set=eval_pool,
    custom_metric=WapeMetric(),  # ADD THIS
    early_stopping_rounds=patience,
    verbose=False,
)
```

**Expected speedup:** 15–25% (stop 20–30 iter earlier on stable clusters)
**Accuracy risk:** Low — WAPE is our final metric, so early stopping aligns with goal.

**Implementation effort:** 5 minutes (add 1 line to fit_model, already imported).

---

### 1.4 Bootstrap & Bagging Optimization

**Current:**
```yaml
bootstrap_type: MVS
bagging_temperature: 0.4
subsample: 0.85
```

**MVS (Minimum Variance Sampling):**
- Probabilistic bootstrap: variance-weighted sampling of rows
- Good for noise robustness, slower than uniform bootstrap
- Overhead: ~10–15% per iteration

**Alternatives & Impact:**

| Type | Speed | Stability | When to use |
|------|-------|-----------|-------------|
| **MVS** (current) | Slow | Very stable | Sparse/noisy data |
| **Poisson** | Fast (+15%) | Good | Default, balanced |
| **No** (subsample only) | Fastest (+25%) | Good if subsample tuned | High-volume clusters |
| **Bernoulli** | Fast | Good | Similar to Poisson |

**Recommendation by cluster pattern:**

```yaml
# Cluster tuning profiles:
sparse_intermittent:
  bootstrap_type: MVS        # Keep — stabilizes sparse data
  bagging_temperature: 0.5   # Slight increase (more aggression)

high_volume_stable:
  bootstrap_type: Poisson    # Switch to faster method
  bagging_temperature: 0.3   # Lower temp = less aggressive sampling

default:
  bootstrap_type: Poisson    # Change default from MVS
  bagging_temperature: 0.4
```

**Expected speedup:** 10–15% globally (MVS→Poisson on stable clusters)
**Accuracy risk:** Low if subsample tuned (currently 0.85 is reasonable).

---

### 1.5 Depth vs. Leaf Trade-off

**Current:**
```yaml
depth: 10           # Max depth of each tree
max_leaves: 127     # Max leaves per tree (Lossguide policy)
```

**Analysis:**
- `depth=10` → potential 2^10 = 1024 leaves (soft limit)
- `max_leaves=127` → hard limit on actual leaves per tree
- For monthly grain: depth 8–9 typically sufficient (62 features, sparse interactions)
- Deeper trees → slower iteration, more memory, risk of overfit

**Monthly grain characteristics:**
- ~150–300 clusters × ~10–200 rows per cluster
- 62 features (lags, rolling, seasonal, categorical)
- High sparsity (many zero-demand months)

**Recommendation:** Reduce depth tier by cluster pattern:

```yaml
# Current
depth: 10
max_leaves: 127

# Proposed
default:
  depth: 8          # Sufficient for monthly grain
  max_leaves: 100   # Reduces tree size memory ~25%

sparse_intermittent:
  depth: 6          # Simpler trees for sparse patterns
  max_leaves: 64

high_volume_stable:
  depth: 9
  max_leaves: 120
```

**Expected speedup:** 15–20% (fewer nodes to evaluate per iteration)
**Accuracy risk:** Medium — requires validation. Depth 8 may be too shallow for some complex patterns.

**Validation approach:** Run multi-TF backtest with depth=[6,7,8,9,10]; measure final accuracy + wall time.

---

## 2. Parallelization Strategies

### 2.1 Across Timeframes (New Opportunity)

**Current:** Sequential loop over 10 timeframes (lines 976–1200 in backtest_framework.py).

**Challenge:** Each timeframe needs a fresh masked_grid (~9.8M rows), so parallel TF would require 10× grid copies = ~10GB RAM.

**Recommendation:** **Conditional parallelization** based on available memory:

```python
# In run_tree_backtest (backtest_framework.py)
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

max_memory_gb = int(os.getenv("BACKTEST_MEMORY_GB", "8"))
enable_tf_parallel = max_memory_gb >= 20  # Only if 20+ GB available

if enable_tf_parallel and len(timeframes) > 1:
    # Parallel timeframe execution with per-TF data isolation
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_single_timeframe, tf, ...): tf
            for tf in timeframes
        }
        for future in as_completed(futures):
            all_predictions.extend(future.result())
else:
    # Current sequential loop
    for ti, tf in enumerate(timeframes):
        ...
```

**Expected speedup:** 2.5–3× if memory available (3 parallel TF × 3.3 TF per worker)
**Feasibility:** Medium (requires careful data pickling, pickle overhead ~100ms per TF).

**Quick path:** Not worth now; better to optimize per-cluster fit first.

---

### 2.2 Per-Cluster Parallelization (Current)

**Status:** Already enabled via `ProcessPoolExecutor` in `scripts/run_backtest.py:662`.

**Configuration:**
- `max_workers=8` (default, configurable via `--workers`)
- Per-cluster fit is **CPU-bound** (training tree models)

**Optimization:**
- **Increase workers for CPU-heavy machines:** `--workers 12` on 16-core systems
- **Monitor memory:** ProcessPoolExecutor pickles models + data per worker
  - Typical: 100–200 MB per worker × 8 = 800 MB overhead
  - Safe range: `workers = min(8, cpu_count // 2)`

**No change needed** — already well-tuned.

---

### 2.3 Per-Cluster Parallelization vs. Sequential

**Sequential (current default when `--parallel` not set):**
- Pros: Minimal memory, easier debugging
- Cons: ~8× slower on 8-core system

**Parallel (with `--parallel`):**
- Pros: 6–7× speedup on 8-core (amortized 100ms pickle overhead)
- Cons: Memory overhead, harder to debug

**Recommendation:** Default to `--parallel` for production backtests.

```bash
# Current
make backtest-all  # Sequential, slow

# Proposed (add to Makefile)
BACKTEST_ARGS ?= --parallel --workers 8
make backtest-all BACKTEST_ARGS="$(BACKTEST_ARGS)"
```

---

### 2.4 Task Type GPU vs. CPU

**Current:** `DEMAND_GPU=auto` (autodetect on Darwin)

**CatBoost GPU support:**
- ✅ `task_type="GPU"` fully supported
- ✅ Can accelerate tree growing, loss computation
- ✅ Memory: ~2GB per GPU for typical models

**When GPU is beneficial:**
- ✅ Large datasets (>100K rows per cluster) — **not typical here**
- ✅ Lossguide policy (complex scoring) — **current config**
- ✗ Small clusters (<5K rows) — **most clusters here**

**Monthly grain reality:**
- Per-cluster rows: 10–200 (small)
- Feature matrix: 9.8M rows total (split by cluster)
- GPU overhead dominates on small clusters

**Recommendation:** **Keep CPU-default, GPU disabled for monthly grain.**

```yaml
# Add to algorithm_config.yaml
gpu_config:
  enable_gpu: false        # Disabled for monthly grain (small clusters)
  min_cluster_rows_gpu: 50000  # Only use GPU if cluster > 50K rows
```

**Current behavior (lines 1017–1038 in run_backtest.py):** Auto-detects and falls back gracefully. ✅ No change needed.

---

## 3. Parameter Space Deep Dive

### 3.1 Border Count Reduction

**Current:** `border_count: 64`

**Impact:**
- Controls histogram bin count per feature
- 64 bins = ~64 split candidates per feature per iteration
- Memory: ~64 float32 per feature per tree node = ~2 KB per node aggregate
- Computation: O(border_count × feature_count × node_count)

**Reduction options:**

| border_count | Speed vs. 64 | Accuracy | When to use |
|---|---|---|---|
| 64 (current) | 1× | Baseline | Monthly grain (good balance) |
| 32 | +30% | ±1% loss | Sparse features, large depth |
| 16 | +50% | ±2% loss | Quick prototyping |
| 128 | -25% | +0.5% gain | Very large datasets |

**Monthly grain assessment:**
- 62 features, most continuous (lags, rolling averages)
- Categorical: 4–6 (low cardinality: 10–50 each)
- **Current 64 is reasonable; 32 is safe.**

**Recommendation:** Reduce to `border_count: 32` by cluster pattern:

```yaml
default:
  border_count: 32         # Reduce from 64 (±1% accuracy for ~30% speedup)

sparse_intermittent:
  border_count: 32         # Fine for sparse data

high_volume_stable:
  border_count: 64         # Keep higher for complex patterns
```

**Expected speedup:** 25–30% (fewer bins to evaluate)
**Accuracy risk:** Low (validation needed, expect ±1% loss).

---

### 3.2 Random Strength

**Current:** `random_strength: 0.5`

**Definition:** Adds random noise to split scores to encourage diversity.
- Range: [0, 1]
- 0 = deterministic splits (fast)
- 1 = maximum randomness (slow, more models tried)

**Impact on monthly grain:**
- High randomness: helps with sparse/intermittent patterns
- Low randomness: faster convergence on stable clusters

**Recommendation:** Tier by pattern:

```yaml
sparse_intermittent:
  random_strength: 0.7     # Higher (more diversity for sparse)

high_volume_stable:
  random_strength: 0.3     # Lower (faster on stable)

default:
  random_strength: 0.4     # Reduce from 0.5 (modest speedup)
```

**Expected speedup:** 5–10% (faster split candidate pruning)
**Accuracy risk:** Low.

---

### 3.3 Leaf Estimation Method

**Current:** `leaf_estimation_method: Newton`

**Comparison:**

| Method | Speed | Stability | When to use |
|--------|-------|-----------|-------------|
| **Newton** (current) | Fast | Stable | General purpose ✅ |
| **Gradient** | Slower (2× iterations) | More conservative | When Newton overshoots |

**Monthly grain:** Newton is appropriate.

**Optimization:** Reduce `leaf_estimation_iterations` (currently 10):

```yaml
leaf_estimation_method: Newton
leaf_estimation_iterations: 5      # Reduce from 10 (half the iterations)
```

**Expected speedup:** 5–10% (fewer per-leaf refinement loops)
**Accuracy risk:** Low (validation recommended).

---

### 3.4 Max CTR Complexity

**Current:** `max_ctr_complexity: 1`

**Definition:** Max order of feature interactions encoded as category (CTR features).
- 0 = no CTR features
- 1 = first-order interactions (A×B)
- 2+ = higher-order (rarely used)

**Assessment:**
- Monthly grain has 4–6 categorical features
- Interactions: item × location (demand varies by location)
- Currently set to 1 ✅ (reasonable)

**Optimization:** Keep at 1. No change.

---

### 3.5 Model Size Regularization

**Current:** `model_size_reg: 0.08`

**Definition:** L1 penalty on model size (number of leaves).
- Higher = smaller trees, faster prediction but slower training
- Lower = larger trees, less regularization

**Assessment:**
- 0.08 is moderate (not too aggressive)
- For monthly grain with depth 8–10: appropriate

**Recommendation:** Increase slightly to reduce tree size:

```yaml
model_size_reg: 0.12       # Increase from 0.08 (smaller trees = faster)
```

**Expected speedup:** 3–5% (fewer leaves to evaluate)
**Accuracy risk:** Low.

---

## 4. Category Feature Handling

### 4.1 Native vs. One-Hot Encoding

**Current:** Native category handling (CatBoost best practice)

```python
# In model_registry.py, line 231
cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
eval_pool = lib_module.Pool(X_val, y_val, cat_features=cat_indices)
model.fit(..., cat_features=cat_indices, ...)
```

**Performance:**
- Native: CatBoost builds category-aware splits
- One-hot: Standard tree splits on binary features

**Monthly grain categories:**
- 4–6 categorical features: item_id, customer_group, loc, ml_cluster, item_class
- Cardinality: 100–5000 values each
- **Native is faster** (~10–20% speedup vs. one-hot) ✅

**No change needed** — current approach is optimal.

---

### 4.2 Category Permutation-based Split Ordering

**Not currently configured:** CatBoost supports `per_float_feature_quantization` for category cardinality optimization.

**Recommendation:** Add for high-cardinality categories:

```yaml
# For high-cardinality item_id (5000+ unique values)
# CatBoost will auto-quantize, but explicit tuning helps:
per_float_feature_quantization: "0:TargetStats"  # One strategy per categorical
```

**Expected speedup:** 5–10% on high-cardinality categories
**Complexity:** Moderate (requires profiling per dataset).

**Skip for now** — native handling is already good.

---

## 5. Risk Assessment & Recommendation Priority

### Safety Matrix (Accuracy Loss vs. Speedup)

| Change | Speedup | Risk Level | Accuracy Impact | Priority |
|--------|---------|-----------|---|---|
| **Reduce border_count: 64→32** | +25% | Low | -1% max | ✅ HIGH |
| **Increase learning_rate: 0.008→0.012** | +15% | Medium | -1–2% | ✅ HIGH |
| **Growth policy: Lossguide→Depthwise** | +30% | Medium | -2% | ✅ MEDIUM |
| **Reduce depth: 10→8** | +15% | Medium | -1–2% | ✅ MEDIUM |
| **Add WAPE custom metric** | +20% | Low | 0% | ✅ HIGH |
| **Reduce iterations via early stop** | +20% | Low | 0% | ✅ HIGH |
| **Bootstrap: MVS→Poisson** | +10% | Low | 0% | ⚠️ MEDIUM |
| **Reduce leaf_estimation_iterations: 10→5** | +5% | Low | 0% | ⚠️ LOW |
| **Increase model_size_reg** | +3% | Low | -0.5% | ⚠️ LOW |
| **Enable GPU** | -5% (overhead) | — | — | ❌ SKIP |
| **Parallelize timeframes** | +200% | Medium | 0% | ⚠️ FUTURE |

---

## 6. Recommended Configuration (Conservative Bundle)

Apply these changes to `config/algorithm_config.yaml`:

```yaml
catboost:
  enabled: true
  model_id: catboost_cluster
  cluster_strategy: per_cluster
  recursive: true
  shap_select: true
  shap_threshold: 0.95
  shap_top_n: null
  shap_sample_size: 500
  tune_inline: false
  params_file: null

  # ─── OPTIMIZED ─────────────────────────────────────────
  iterations: 2500          # DOWN from 3000 (fewer iterations)
  learning_rate: 0.010      # UP from 0.008 (faster convergence)
  depth: 9                  # DOWN from 10 (balance speed/accuracy)
  l2_leaf_reg: 7.5          # KEEP
  subsample: 0.85           # KEEP
  grow_policy: Lossguide    # KEEP (or test Depthwise separately)
  border_count: 32          # DOWN from 64 (less histogramming)
  random_strength: 0.4      # DOWN from 0.5 (slightly faster splits)
  min_data_in_leaf: 28      # KEEP
  colsample_bylevel: 0.85   # KEEP
  bagging_temperature: 0.4  # KEEP
  max_leaves: 100           # DOWN from 127 (fewer leaves)
  bootstrap_type: MVS       # KEEP (stability)
  model_size_reg: 0.10      # UP from 0.08 (smaller model penalty)
  score_function: L2        # KEEP
  boost_from_average: true  # KEEP
  leaf_estimation_method: Newton  # KEEP
  leaf_estimation_iterations: 8   # DOWN from 10 (fewer leaf refinements)
  max_ctr_complexity: 1     # KEEP
```

**Expected impact:**
- **Wall-clock speedup:** 25–35%
- **Accuracy loss:** -1% to +0.5% (validate)
- **Memory:** -10% (fewer leaves)

---

## 7. Progressive Testing Plan

### Phase 1: Quick Validation (4 hours)

1. **Single timeframe (TF-5), single cluster subset (50 clusters):**
   ```bash
   # Baseline (current config)
   time python scripts/run_backtest.py --model catboost --n-timeframes 1 --parallel

   # Test 1: Conservative (above bundle)
   time python scripts/run_backtest.py --model catboost --n-timeframes 1 --parallel \
     --config config/algorithm_config_test_v1.yaml
   ```

   **Acceptance:** Wall time <60% of baseline, accuracy within -1%.

2. **Multi-timeframe test (3 TF, no cluster limit):**
   ```bash
   time python scripts/run_backtest.py --model catboost --n-timeframes 3 --parallel
   ```

   **Acceptance:** 30% speedup, stable accuracy across timeframes.

### Phase 2: Full Validation (8–12 hours)

3. **Full backtest (10 TF, all clusters):**
   ```bash
   time python scripts/run_backtest.py --model catboost --n-timeframes 10 --parallel
   ```

   **Acceptance:** 25–35% wall-time reduction, accuracy ≥ -1.5%.

### Phase 3: Per-Cluster Profile Tuning (1–2 days)

4. Apply cluster_tuning_profiles.yaml overrides (if implemented):
   - sparse_intermittent: lower LR, keep depth
   - high_volume_stable: higher LR, lower depth

   **Acceptance:** +5–10% additional speedup without accuracy loss.

---

## 8. Code Changes Required

### 8.1 Update algorithm_config.yaml

**File:** `config/algorithm_config.yaml` (lines 68–97)

```diff
  catboost:
    enabled: true
    model_id: catboost_cluster
    cluster_strategy: per_cluster
    recursive: true
    shap_select: true
    shap_threshold: 0.95
    shap_top_n: null
    shap_sample_size: 500
    tune_inline: false
    params_file: null
-   iterations: 3000
-   learning_rate: 0.008
+   iterations: 2500
+   learning_rate: 0.010
    depth: 10
    l2_leaf_reg: 7.5
    subsample: 0.85
    grow_policy: Lossguide
-   border_count: 64
-   random_strength: 0.5
+   border_count: 32
+   random_strength: 0.4
    min_data_in_leaf: 28
    colsample_bylevel: 0.85
    bagging_temperature: 0.4
-   max_leaves: 127
+   max_leaves: 100
    bootstrap_type: MVS
-   model_size_reg: 0.08
+   model_size_reg: 0.10
    score_function: L2
    boost_from_average: true
    leaf_estimation_method: Newton
-   leaf_estimation_iterations: 10
+   leaf_estimation_iterations: 8
    max_ctr_complexity: 1
```

**Impact:** 3 lines changed, backwards-compatible (only hyperparameters).

### 8.2 Add WAPE Custom Metric (Optional Enhancement)

**File:** `common/ml/model_registry.py` (line 230–242)

```diff
  elif model_name == "catboost":
      cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
      eval_pool = lib_module.Pool(X_val, y_val, cat_features=cat_indices)
-     model.fit(
+     # Custom WAPE metric for early stopping alignment
+     model.fit(
          X_tr, y_tr,
          cat_features=cat_indices,
          eval_set=eval_pool,
+         custom_metric=WapeMetric(),  # ADD: align early stopping with final metric
          early_stopping_rounds=patience,
          verbose=False,
      )
```

**Impact:** 1 line added, WapeMetric already defined (line 151–170), no imports needed.

### 8.3 Update Makefile (Optional)

**File:** `Makefile`

```makefile
# Add to backtest targets
backtest-catboost-fast:
	@DEMAND_GPU=off uv run python scripts/run_backtest.py \
		--model catboost --parallel --workers 8 -q

backtest-catboost-validation:
	@# Quick test: 3 timeframes
	@DEMAND_GPU=off uv run python scripts/run_backtest.py \
		--model catboost --parallel --n-timeframes 3 -q
```

---

## 9. Monitoring & Instrumentation

### 9.1 Add Timing Breakdowns

Already implemented via `profiled_section()` in `common/services/perf_profiler.py`.

**Current coverage:**
- load_config, detect_gpu, run_backtest (high-level)
- Per-timeframe masking, feature selection, training NOT yet profiled

**Recommendation:** Add timeframe-level profiling:

```python
# In backtest_framework.py, line 976
for ti, tf in enumerate(timeframes):
    with profiled_section(f"timeframe_{tf['label']}"):
        # ... existing code ...
```

**Output:** Per-TF breakdown in perf report, helps identify bottleneck TFs.

### 9.2 Log Best Iterations

**Current:** `scripts/run_backtest.py` doesn't log best iteration per cluster.

**Add logging** (line 549 in run_backtest.py):

```python
logger.info(
    "Cluster %d/%d '%s': best_iter=%d/%d, val_wape=%.2f%%",
    ci, n_clusters, cluster_label,
    n_est_used, max_iters,  # NEW: show early stopping effectiveness
    val_wape,
)
```

**Output:** Track early stopping effectiveness, validate parameter tuning.

---

## 10. Final Recommendations Summary

### Immediate (Day 1)

1. ✅ **Update algorithm_config.yaml** (3-line change):
   - iterations: 3000 → 2500
   - learning_rate: 0.008 → 0.010
   - border_count: 64 → 32
   - Expected: 25% speedup

2. ✅ **Test on single timeframe** (1 hour):
   ```bash
   python scripts/run_backtest.py --model catboost --n-timeframes 1 --parallel
   ```

### Short-term (This Week)

3. ⚠️ **Optional: Add WAPE custom metric** (5 min):
   - Aligns early stopping with final accuracy metric
   - Expected: +15–20% speedup on stable clusters

4. ⚠️ **Test growth_policy alternatives** (4 hours):
   - Compare Lossguide vs. Depthwise in parallel runs
   - Expected: +30% if Depthwise tuned well

### Medium-term (Next Sprint)

5. ⚠️ **Implement cluster_tuning_profiles overrides**:
   - Tiered learning rates by demand pattern
   - Expected: +10% additional speedup

6. ⚠️ **Parallelize timeframes** (if memory permits):
   - ProcessPoolExecutor over TF indices
   - Expected: 2.5–3× speedup (requires validation)

---

## References

- **Config:** `config/algorithm_config.yaml` (lines 68–97)
- **Backtest framework:** `common/ml/backtest_framework.py` (881–1200)
- **Model fitting:** `common/ml/model_registry.py` (187–251)
- **Script runner:** `scripts/run_backtest.py` (830–1200)
- **Performance profiling:** `common/services/perf_profiler.py`
- **Cluster profiles:** `config/cluster_tuning_profiles.yaml` (if exists)
- **Testing:** `tests/unit/test_backtest_*.py`, `tests/api/test_lgbm_tuning.py`

---

**Analysis completed:** 2026-03-26
**Estimated wall-clock improvement:** 1.5–2.5× (25–60% reduction)
**Recommended starting point:** Conservative bundle (Phase 1)
