# CatBoost Performance Tuning — Quick Reference

**TL;DR:** Apply 3-line config change for 25% speedup. Test 1 timeframe in 1 hour.

---

## Current Bottleneck
Sequential per-cluster training: **10 timeframes × 150–300 clusters × 5–15 min each = 6–15 hours**

---

## Quick Wins (Ranked by Impact/Effort)

### 1️⃣ HIGHEST PRIORITY: Update 3 Config Lines (25% speedup, 5 min)

Edit `config/algorithm_config.yaml` lines 79–93:

```yaml
iterations: 2500              # ← Change from 3000
learning_rate: 0.010          # ← Change from 0.008
border_count: 32              # ← Change from 64
```

**Why:** Fewer iterations + faster convergence + less histogramming.

**Test:**
```bash
python scripts/run_backtest.py --model catboost --n-timeframes 1 --parallel
# Expected: ~60% of baseline time
```

---

### 2️⃣ HIGH PRIORITY: Add WAPE Metric (15–20% speedup, 1 min)

Edit `common/ml/model_registry.py` line 236:

```python
model.fit(
    X_tr, y_tr,
    cat_features=cat_indices,
    eval_set=eval_pool,
    custom_metric=WapeMetric(),  # ← ADD THIS LINE
    early_stopping_rounds=patience,
    verbose=False,
)
```

**Why:** Aligns early stopping with final accuracy metric (WAPE).

---

### 3️⃣ MEDIUM PRIORITY: Reduce Max Leaves (15% speedup, no code change)

Edit `config/algorithm_config.yaml` line 90:

```yaml
max_leaves: 100               # ← Change from 127
```

---

### 4️⃣ MEDIUM PRIORITY: Tune Per-Cluster Learning Rates

If `config/cluster_tuning_profiles.yaml` exists, add:

```yaml
sparse_intermittent:
  learning_rate: 0.005        # Lower for sparse patterns

high_volume_stable:
  learning_rate: 0.015        # Higher for stable demand
```

---

## Safety Levels

| Change | Speedup | Risk | Validate? |
|--------|---------|------|-----------|
| border_count 64→32 | 25% | 🟢 Low | Yes, -1% accuracy OK |
| learning_rate 0.008→0.010 | 15% | 🟡 Medium | Yes, -1% accuracy risk |
| max_leaves 127→100 | 15% | 🟢 Low | Optional |
| WAPE metric | 15% | 🟢 Low | No (better metric alignment) |
| growth_policy Depthwise | 30% | 🟡 Medium | Yes (needs depth tuning) |

---

## Testing Checklist

- [ ] Edit `config/algorithm_config.yaml` (3 lines)
- [ ] Run single timeframe: `python scripts/run_backtest.py --model catboost --n-timeframes 1 --parallel`
- [ ] Compare wall time: target <60% of baseline
- [ ] Check accuracy loss: validate ≤-1%
- [ ] Run 3-timeframe test if time permits
- [ ] Commit config change if validated

---

## Rollback (if accuracy loss > 1%)

Revert to `config/algorithm_config.yaml`:

```yaml
iterations: 3000              # Restore
learning_rate: 0.008          # Restore
border_count: 64              # Restore
```

---

## Expected Results

**Conservative bundle (3 config lines + WAPE metric):**
- ⏱️ Wall time: 25–35% reduction
- 📊 Accuracy: -0.5% to +0.5% change (likely neutral)
- 💾 Memory: -10%

**Full optimization (if all changes applied + profiling):**
- ⏱️ Wall time: 50–60% reduction possible
- 📊 Accuracy: validate to ±1.5%

---

## Per-Cluster Training Phases

```
Current (per TF):
├─ Mask future sales       ~3–5s
├─ FOR each cluster       ~5–15 min (ProcessPoolExecutor 8 workers)
│  └─ CatBoost.fit()       ← TARGET OPTIMIZATION HERE
└─ Postprocess            ~2–3s

With 3-line config change:
├─ Mask future sales       ~3–5s
├─ FOR each cluster       ~3–10 min (25% faster fits)
│  └─ CatBoost.fit()       ← Fewer iters, less histogramming
└─ Postprocess            ~2–3s

Speedup per TF: 5–7 min saved
```

---

## When NOT to Apply

❌ If you need **maximum accuracy** (trade off speed instead)
❌ If **high-volume clusters** (>10K rows) are your priority (they're fast already)
❌ If you're still profiling your baseline (establish baseline first)

---

## Debugging Bad Accuracy Loss

If validation shows >1% accuracy loss:

1. **Revert border_count first** (most aggressive change):
   ```yaml
   border_count: 48              # Middle ground
   ```

2. **Keep learning_rate increase** (safer, more consistent):
   ```yaml
   learning_rate: 0.010          # Generally safe
   ```

3. **Reduce iterations less aggressively:**
   ```yaml
   iterations: 2750              # Intermediate step
   ```

4. **Re-test single timeframe** after each change.

---

## Profile Your Bottleneck

To see where time is spent:

```bash
# Add logging
DEMAND_LOG_LEVEL=DEBUG python scripts/run_backtest.py \
  --model catboost --n-timeframes 1 --parallel 2>&1 | grep -E "(Cluster|Mask|postprocess)"

# Look for:
# - "Masking done (Xs)" → feature recomputation cost
# - "Cluster X/Y: X.XXs" → individual cluster training time
# - "Postprocess (Xs)" → dedup/accuracy cost
```

---

## Common Questions

**Q: Will this hurt accuracy?**
A: -0.5% to +0.5% expected. Test on single TF first (1 hour).

**Q: Is GPU enabled?**
A: No (and shouldn't be — clusters are too small). `DEMAND_GPU=auto` checks and disables.

**Q: Can I parallelize timeframes?**
A: Yes, but requires 20GB+ RAM (10 TF × 1GB each). Not recommended for now.

**Q: Should I use Depthwise instead of Lossguide?**
A: Maybe (+30% speedup possible). Requires separate validation run.

**Q: Why is learning_rate so low (0.008)?**
A: Lossguide + monthly grain (sparse, categorical). Conservative default. 0.010 is reasonable middle ground.

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `config/algorithm_config.yaml` | 79, 80, 85 | iterations, learning_rate, border_count |
| `common/ml/model_registry.py` | 236 | Add custom_metric (optional) |

---

## Next Steps

1. **Day 1:** Apply 3-line config change + test single TF
2. **Day 2:** Run multi-TF validation (3 timeframes)
3. **Day 3:** Full 10-TF backtest if confident
4. **Week 2:** Explore depth/growth_policy tuning if needed

---

See [CATBOOST_PERFORMANCE_TUNING_ANALYSIS.md](./CATBOOST_PERFORMANCE_TUNING_ANALYSIS.md) for full details.
