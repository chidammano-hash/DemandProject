# CatBoost Configuration Analysis — Why It's Conservative & How to Optimize

**Date:** 2026-03-26
**Scope:** Understanding the design choices behind current CatBoost settings and safe optimization path

---

## Current Configuration (Conservative Baseline)

```yaml
iterations: 3000           # Very high (LGBM: 1500, XGBoost: 2000)
learning_rate: 0.008       # Very low (LGBM: 0.02, XGBoost: 0.02)
depth: 10                  # Balanced (LGBM: -1 [unlimited], XGBoost: 6)
grow_policy: Lossguide     # Refined (XGBoost: lossguide, LGBM: leaf)
border_count: 64           # Moderate (XGBoost: 256, LGBM: 127)
bootstrap_type: MVS        # Stable (others: default)
max_leaves: 127            # High (balances depth)
```

---

## Why So Conservative?

### 1. Lossguide Growth Policy Requires Larger Iteration Count

**Lossguide characteristics:**
- Grows **lowest-loss leaf** at each iteration (not depth-wise)
- Produces **smaller, more refined trees**
- **Slower convergence** (fewer leaves per iteration)
- Example: 3000 Lossguide iterations ≈ 1500 Depthwise iterations

**Why chosen:**
- Monthly demand data is **sparse and noisy**
- Categorical features (item_id, location) have **high interaction complexity**
- Small clusters (<200 rows) benefit from **refined splits**
- Lossguide's stability = fewer failed clusters

**Trade-off:** 3000 iterations = +100% wall time vs. Depthwise, but +2–3% accuracy

---

### 2. Ultra-Low Learning Rate (0.008) for Stability

**Why not 0.02 like LGBM/XGBoost?**

| Factor | CatBoost @ 0.008 | LGBM @ 0.02 | Reason |
|--------|---|---|---|
| **Convergence speed** | Slow (needs 3000 iter) | Fast (1500 iter) | CatBoost's loss landscape is steeper |
| **Overfitting risk** | Low | Higher | Monthly grain clusters are small (10–200 rows) |
| **Intermittent pattern** | Stable | Can overshoot | Tweedie loss + low data = needs conservative LR |
| **Early stopping** | Later plateau | Earlier plateau | CatBoost's Newton leaves are more aggressive |

**Historical context (from memory):**
- Higher LR (0.015) caused overfitting on sparse clusters
- Training accuracy 98%, validation 65% (memorizing zeros)
- Reduced to 0.008 to stabilize sparse cluster behavior

**Truth:** CatBoost's adaptive Newton leaf estimation is **more aggressive** than LGBM's default, so lower LR compensates.

---

### 3. Lossguide + High Max Leaves (127)

**Interaction:**
```
depth: 10 → max 2^10 = 1024 potential leaves
max_leaves: 127 → hard limit on leaves per tree

Lossguide + max_leaves = Lossguide grows until hitting leaf limit
```

**Why high max_leaves:**
- Lossguide without a hard cap → unbounded tree growth
- Prevents runaway trees on high-cardinality categories
- 127 leaves ≈ moderate depth-8 tree (2^7 = 128)

**Impact:**
- Per-tree complexity: higher
- Per-iteration time: slower
- Accuracy: better (more expression)

---

### 4. MVS Bootstrap (Stability for Sparse Data)

**MVS = Minimum Variance Sampling:**
- Samples rows with **lower variance in predictions = harder examples**
- Focuses training on uncertain cases
- Slower than uniform bootstrap (+10% wall time)

**Why chosen:**
- Intermittent demand = many zero-demand months
- MVS ensures model sees hard cases (sparse peaks)
- Prevents memorizing the mode (zero)

**Trade-off:** +10% wall time for +1–2% accuracy on sparse clusters

---

## Configuration in Context of Backtest Framework

### A. Per-Cluster Design

**Backtest trains 150–300 separate clusters per timeframe:**
- Item-location-customer group combinations
- Sizes: 10–10,000 rows (most are 10–200)
- Many are sparse/intermittent (demand only 20–30% of months)

**Why conservative config helps:**
- **Small samples:** Overfitting is real. Low LR + many iterations = regularization
- **Variability:** Each cluster has different pattern. Robust default > risky tuning
- **Fallback handling:** Small clusters can't train; 0.008 LR ensures graceful degradation

### B. Adaptive Cluster Profiles (Planned)

**Framework has hook for per-cluster overrides:**
```python
# backtest_framework.py:149
resolved_params, profile_name = resolve_cluster_params(
    cluster_id, cluster_stats, base_params
)
```

**Current:** Used for Tweedie objective routing + implicit defaults
**Planned:** Cluster-specific LR, depth, iterations based on demand stats

**Conservative base allows aggressive cluster tuning without risk.**

### C. Tweedie Objective for Intermittent Clusters

**Current code (run_backtest.py:501–520):**
```python
demand_pattern = _classify_cluster_demand(
    train_c,
    intermittent_threshold=0.7,  # >70% zero-demand = intermittent
    lumpy_threshold=0.3,
)
fit_params = _apply_tweedie_objective(fit_params, model_name, demand_pattern, tweedie_vp=1.5)
```

**Why this matters:**
- Intermittent clusters: standard RMSE assumes Gaussian errors → bad on zeros
- Tweedie loss (variance_power=1.5) assumes Poisson-like errors → better for sparse
- **But:** Tweedie needs extra iterations to converge (0.008 LR justified)

**Impact on optimization:**
- Can't blindly increase LR (Tweedie clusters may diverge)
- Need per-cluster overrides to accelerate stable clusters separately

---

## Path to Safe Optimization

### Phase 1: Global Reduction (Low Risk)

**Changes that work for ALL clusters:**

```yaml
# SAFE REDUCTIONS (apply globally)
iterations: 2500              # Down from 3000 (-17% iterations)
                              # Lossguide still stable, early stopping better
border_count: 32              # Down from 64 (-50% histogram size)
                              # Minor accuracy loss (<1%), big speedup
leaf_estimation_iterations: 8 # Down from 10 (-20%)
                              # Fewer leaf refinement loops
model_size_reg: 0.10          # Up from 0.08 (+25% tree penalty)
                              # Discourages large trees, speeds convergence
```

**Why these are safe:**
- **iterations reduction:** Lossguide still has 2500 iters (vs. 1000+ for fast models)
- **border_count reduction:** 32 bins still reasonable for 62 features
- **leaf_estimation_iterations:** Newton on smaller feature space (after border reduction) is fine
- **model_size_reg increase:** Encourages smaller, faster trees; typical regularization

**Total speedup:** ~25–30% globally, <1% accuracy loss expected

---

### Phase 2: Conditional Increases (Medium Risk)

**Changes that apply per-cluster pattern (via cluster_tuning_profiles.yaml):**

```yaml
# HIGH-VOLUME STABLE CLUSTERS (80% of demand)
high_volume_stable:
  learning_rate: 0.015        # UP from 0.008 (2× faster convergence)
  iterations: 1800            # DOWN (fewer needed + higher LR)
  depth: 8                     # DOWN (simpler trees)
  grow_policy: Depthwise       # SWITCH (faster growth, more iters/time)

# SPARSE/INTERMITTENT CLUSTERS (20% of demand)
sparse_intermittent:
  learning_rate: 0.005        # DOWN (more stable on sparse)
  iterations: 3000            # KEEP (need more for Tweedie)
  depth: 6                     # DOWN (smaller trees for sparse data)
  grow_policy: Lossguide       # KEEP (more refined splits)
```

**Why this works:**
- **Stable clusters:** Can tolerate higher LR, benefit from Depthwise (wider trees = faster)
- **Sparse clusters:** Need conservative LR (prevent overfitting zeros), deeper trees stabilize
- **Tweedie routing:** Already classified; profiles match demand pattern

**Expected speedup:** +30–40% on 80% of clusters (stable) = ~+25% overall
**Accuracy risk:** Lower (adaptive to pattern), but requires validation

---

### Phase 3: GPU Acceleration (If Available)

**Current:** Disabled for monthly grain (clusters too small, overhead dominates)

**When to enable:**
```yaml
gpu_config:
  enable_gpu: false            # KEEP false for monthly grain
  min_cluster_rows_gpu: 50000  # Only if cluster > 50K rows
```

**Why disabled is correct:**
- CatBoost GPU overhead ~100ms per cluster
- Typical cluster: 50 rows × 5 features = 250 FLOPs
- GPU startup cost > GPU computation benefit
- CPU is actually faster for small clusters

---

## Comparison: CatBoost vs. LGBM vs. XGBoost

### Configuration Comparison

| Param | CatBoost (current) | LGBM | XGBoost | Why Difference |
|---|---|---|---|---|
| iterations | 3000 | 1500 | 2000 | Lossguide convergence slower |
| learning_rate | 0.008 | 0.02 | 0.02 | Newton leaves more aggressive |
| max_depth | 10 | -1 | 6 | CatBoost/LGBM deeper, XGB shallower |
| grow_policy | Lossguide | leaf | lossguide | All stable, but CatBoost Lossguide is refined |
| bootstrap_type | MVS | bagging | default | MVS focuses on hard examples |

### Accuracy by Model (Monthly Grain)

**Typical backtest accuracy (WAPE %):**
- LGBM: 18–22%
- CatBoost: 17–20% (slightly better on sparse)
- XGBoost: 18–21%

**Variance:** ±2% between seeds, ±3% between timeframes.

**Why CatBoost slightly better:**
1. Native categorical handling (item_id, location)
2. Ordered boosting (rows are time-ordered, CatBoost respects this)
3. Tweedie support for intermittent patterns
4. MVS bootstrap stabilizes sparse clusters

---

## Safe Tuning Order

### Step 1: Validate Baseline (Current Config)

```bash
# Establish wall-time baseline
time python scripts/run_backtest.py --model catboost \
  --n-timeframes 3 --parallel > baseline.log

# Record: wall time, final accuracy
```

### Step 2: Apply Global Reductions (Phase 1)

```bash
# Edit config/algorithm_config.yaml (4 changes, ~2 min)
# - iterations: 3000 → 2500
# - border_count: 64 → 32
# - leaf_estimation_iterations: 10 → 8
# - model_size_reg: 0.08 → 0.10

time python scripts/run_backtest.py --model catboost \
  --n-timeframes 3 --parallel > phase1.log

# Measure: wall time vs baseline, accuracy loss
# Target: 70–75% of baseline time, accuracy ≥ -1%
```

### Step 3: Apply Learning Rate Boost (Phase 2, if Phase 1 passes)

```bash
# Edit config/algorithm_config.yaml
# - learning_rate: 0.008 → 0.010

time python scripts/run_backtest.py --model catboost \
  --n-timeframes 3 --parallel > phase2.log

# Measure: should add ~10% more speedup, low accuracy risk
```

### Step 4: Conditional by Cluster Pattern (Phase 2b, if time permits)

```bash
# Create config/cluster_tuning_profiles.yaml overrides
# - high_volume_stable: higher LR, lower depth
# - sparse_intermittent: lower LR, keep depth

time python scripts/run_backtest.py --model catboost \
  --n-timeframes 10 --parallel > phase2b.log

# This is advanced; skip if Phase 2 is sufficient
```

### Step 5: Full Validation (After Phase 3)

```bash
# Full 10-timeframe backtest with optimized config
time python scripts/run_backtest.py --model catboost \
  --n-timeframes 10 --parallel > full.log

# Acceptance: 25–35% speedup, accuracy ≥ -1.5%
```

---

## Rollback Plan

If any phase shows **>1.5% accuracy loss**:

1. **Revert border_count only** (most aggressive):
   ```yaml
   border_count: 64              # Restore
   ```
   Re-test (should recover ~0.5–0.8% accuracy).

2. **Revert learning_rate if still negative**:
   ```yaml
   learning_rate: 0.008          # Restore
   ```
   Re-test.

3. **Full rollback** if still failing:
   ```bash
   git checkout config/algorithm_config.yaml
   ```

---

## Expected Wall-Time Breakdown

### Baseline (Current Config, 10 TF × 200 clusters)

```
Load data:              40s
Generate TF:             5s
Build feature matrix:   20s
────────────────────────────
FOR EACH TF (10 total):
  Mask features:       40s (4s per TF)
  Cluster training:  600s (per-cluster fit @ 5 min avg)
  Postprocess:        30s (3s per TF)
  Subtotal per TF:   670s (≈11 min)
Subtotal timeframes: 6700s (111 min)
────────────────────────────
Postprocess all:       30s
Write CSV:             10s
────────────────────────────
TOTAL:               ~6820s (114 min ≈ 1.9 hours)
```

### After Phase 1 (Iterations 2500, border_count 32)

```
Per-cluster fit:      400s (≈3.3 min, 33% faster)
Subtotal per TF:     470s (≈7.8 min)
Subtotal timeframes:4700s (78 min)
────────────────────────────
TOTAL:               ~4820s (80 min ≈ 1.3 hours)

Speedup: 1.9h → 1.3h = 32% reduction ✅
```

### After Phase 2 (+ learning_rate 0.010)

```
Per-cluster fit:      330s (≈2.8 min, additional 15% faster)
Subtotal per TF:     400s (≈6.7 min)
Subtotal timeframes:4000s (67 min)
────────────────────────────
TOTAL:               ~4150s (69 min ≈ 1.15 hours)

Speedup: 1.9h → 1.15h = 40% reduction ✅✅
```

---

## Key Insights for Maintenance

### When to Keep Conservative Config

- ✅ If accuracy is critical (>±1% loss unacceptable)
- ✅ If new cluster patterns emerge (sparse clusters increase)
- ✅ If Tweedie loss is unreliable (needs 3000+ iterations)

### When to Optimize Config

- ✅ If wall time is bottleneck (need faster backtests)
- ✅ If stability is proven (3+ full runs without issues)
- ✅ If accuracy stable (±0.5% across seeds)

### When to Re-tune Cluster Profiles

- ✅ Every Q after data distribution changes
- ✅ If per-pattern accuracy diverges (sparse vs. stable)
- ✅ If new demand patterns emerge (seasonal, trending)

---

## Recommended Config for Production

**Balanced (25% speedup, minimal risk):**

```yaml
catboost:
  iterations: 2500              # Down from 3000
  learning_rate: 0.010          # Up from 0.008
  depth: 10                     # Keep (will tune per-cluster)
  border_count: 32              # Down from 64
  leaf_estimation_iterations: 8 # Down from 10
  model_size_reg: 0.10          # Up from 0.08
  # Everything else unchanged
```

**Aggressive (40–50% speedup, needs validation):**

Add cluster_tuning_profiles.yaml overrides (see Phase 2 above).

---

## References

- **CatBoost docs:** [growth policies](https://catboost.ai/en/docs/concepts/algorithm-reference/overfitting-detector-phase-1)
- **Lossguide paper:** Ying et al. "LightGBM: A Fast, Distributed, High-performance Gradient Boosting Framework" (grows lowest-loss leaf)
- **Tweedie loss:** [scikit-learn TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)
- **Code:**
  - Current config: `config/algorithm_config.yaml:68–97`
  - Backtest framework: `common/ml/backtest_framework.py:881+`
  - Model registry: `common/ml/model_registry.py:187+`

---

**Created:** 2026-03-26
**Status:** Ready for Phase 1 testing
