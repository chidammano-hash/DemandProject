# LGBM Performance Optimization: Quick Start Guide

**Fast-track implementation of 25-40% speedup (Phase 1)**

---

## 🚀 One-Minute Summary

Current LGBM backtest: **~50 minutes per model** (3 models = 165 min total)

**After Phase 1 (10 minutes of changes)**: ~30-37 minutes per model
**After Phase 2 (1 line in Makefile)**: ~5-6 minutes per model (3 models = 20 min)

---

## Phase 1: Config Changes (10 min implementation)

### Step 1: Update algorithm_config.yaml

```bash
cd /Users/manoharchidambaram/projects/DemandProject
nano config/algorithm_config.yaml
```

**Find line 67 (max_bin):**
```yaml
# BEFORE:
max_bin: 127

# AFTER:
max_bin: 64
```

**Find line 63 (feature_fraction_bynode):**
```yaml
# BEFORE:
feature_fraction_bynode: 0.7

# AFTER:
feature_fraction_bynode: 0.6
```

### Step 2: Update backtest config

```bash
nano config/algorithm_config.yaml
```

**Find line 11 (early_stop_pct):**
```yaml
# BEFORE:
early_stop_pct: 0.03

# AFTER:
early_stop_pct: 0.02
```

### Step 3: Validate Changes

```bash
python scripts/run_backtest.py --model lgbm --n-timeframes 2
```

**Expected output**:
```
Training global lgbm on 500000 rows, 46 features...
Cluster 0/20 'sparse': val_WAPE=33.1%, best_iter=550
Cluster 1/20 'high': val_WAPE=32.9%, best_iter=580
...
Accuracy at execution lag (25000 rows): WAPE=33.2%, Bias=0.0012, Accuracy=86.5%
```

**Success criteria**:
- ✅ best_iter between 400-800 (not 1000+)
- ✅ WAPE within ±0.2% of baseline
- ✅ Script completes without errors

### Step 4: Measure Speedup

```bash
# Record time for 2-timeframe run:
time python scripts/run_backtest.py --model lgbm --n-timeframes 2

# Expected:
#   Before: ~10 min
#   After: ~7 min
#   Speedup: 30% ✓
```

---

## Phase 2: Parallelization (1 line of code)

### Step 1: Update Makefile

```bash
nano Makefile
```

**Find the `backtest-all` target** (search for `backtest-all:`):

```makefile
# BEFORE:
backtest-all:
	@~/.local/bin/uv run python scripts/run_backtest.py --model lgbm
	@~/.local/bin/uv run python scripts/run_backtest_catboost.py
	@~/.local/bin/uv run python scripts/run_backtest_xgboost.py

# AFTER:
backtest-all:
	@~/.local/bin/uv run python scripts/run_backtest.py --model lgbm --parallel --workers 8
	@~/.local/bin/uv run python scripts/run_backtest_catboost.py --parallel --workers 8
	@~/.local/bin/uv run python scripts/run_backtest_xgboost.py --parallel --workers 8
```

### Step 2: Test Full Ensemble

```bash
make backtest-all
```

**Expected output**:
```
Parallel cluster training enabled (max_workers=8)
Cluster 1/20 'high': val_WAPE=32.9%, best_iter=580
Cluster 2/20 'medium': val_WAPE=33.5%, best_iter=620
Cluster 3/20 'low': val_WAPE=34.1%, best_iter=550
...
Parallel cluster training enabled (max_workers=8)  [CatBoost]
...
Parallel cluster training enabled (max_workers=8)  [XGBoost]
...
```

**Success criteria**:
- ✅ No crashes in ProcessPoolExecutor
- ✅ All 3 models complete
- ✅ Accuracy unchanged (vs. Phase 1 baseline)
- ✅ Total time < 30 min (was 165 min)

### Step 3: Verify Speedup

```bash
# Time the full ensemble:
time make backtest-all

# Expected:
#   Before optimization: 165 min
#   Phase 1 only: 100 min (40% speedup)
#   Phase 1+2: 20-25 min (8.5x speedup)
```

---

## Rollback Plan (If Issues Occur)

### If accuracy drops significantly (>0.5%)

```bash
# Revert all changes:
git checkout config/algorithm_config.yaml
git checkout Makefile

# Test baseline:
python scripts/run_backtest.py --model lgbm --n-timeframes 2
```

### If ProcessPoolExecutor crashes

```bash
# Revert just the parallelization:
git checkout Makefile

# Keep Phase 1 config changes (safe)

# Test with sequential clusters:
python scripts/run_backtest.py --model lgbm --n-timeframes 2
```

---

## Next Steps (Optional, Moderate Risk)

### Phase 3: Fine-tune Hyperparameters

Only proceed if Phase 1+2 is working well:

```yaml
# In config/algorithm_config.yaml (advanced):
algorithms:
  lgbm:
    subsample: 0.7          # was 0.8
    bagging_freq: 2         # was 1
    min_gain_to_split: 0.01 # was 0.005
    reg_alpha: 0.2          # was 0.1
```

**Before committing**, validate:
```bash
python scripts/run_backtest.py --model lgbm --n-timeframes 2
# Check WAPE is within -0.2% of Phase 1 baseline
```

---

## Common Issues & Troubleshooting

### Issue: "ProcessPoolExecutor crashed" or "Pickling error"

**Cause**: Complex object serialization in multiprocessing

**Fix**:
```bash
# Revert to sequential training:
git checkout Makefile

# Or use fewer workers:
python scripts/run_backtest.py --model lgbm --parallel --workers 4
```

### Issue: "Accuracy dropped >0.5%"

**Cause**: One of the config parameters too aggressive

**Fix**:
```bash
# Revert Phase 1:
git checkout config/algorithm_config.yaml

# Re-apply conservatively:
# - Keep max_bin=64 (safest change)
# - Keep early_stop_pct=0.025 (more conservative than 0.02)
# - Revert feature_fraction_bynode to 0.7

# Test again:
python scripts/run_backtest.py --model lgbm --n-timeframes 2
```

### Issue: "Out of memory" with --workers 8

**Cause**: System has <8GB free RAM during training

**Fix**:
```bash
# Use fewer workers:
python scripts/run_backtest.py --model lgbm --parallel --workers 4

# Or monitor memory:
watch -n 1 'free -h | grep Mem'
```

---

## Performance Expectations by Phase

| Phase | Config Changes | Parallelization | Time/Model | Total (3 models) |
|-------|---|---|---|---|
| Baseline | None | Sequential | 50 min | 165 min |
| Phase 1 | max_bin, early_stop | Sequential | 30-35 min | 95-105 min |
| Phase 1+2 | Phase 1 + Feature% | 8-worker clusters | 6-7 min | 20-25 min |
| Phase 1+2+3 | Phase 1+2 + subsampling | 8-worker clusters | 5-6 min | 15-20 min |

---

## Validation Checklist

- [ ] Phase 1 config changes applied
- [ ] Single timeframe test passes (2 TF run)
- [ ] Accuracy within -0.2% of baseline
- [ ] Best iteration in 400-800 range
- [ ] Phase 1 speedup measured (~30%)
- [ ] Makefile updated with --parallel --workers 8
- [ ] Full ensemble backtest completes (<30 min)
- [ ] All 3 models have consistent accuracy
- [ ] No ProcessPoolExecutor errors
- [ ] Phase 1+2 speedup measured (8-9x)

---

## Git Workflow

```bash
# Create feature branch
git checkout -b perf/lgbm-optimization

# Apply Phase 1
# (edit files as per instructions above)
git add config/algorithm_config.yaml
git commit -m "Performance: Reduce LGBM max_bin & early_stop_pct

- max_bin: 127 → 64 (histogram memory -50%)
- early_stop_pct: 0.03 → 0.02 (earlier convergence)
- feature_fraction_bynode: 0.7 → 0.6 (fewer evaluations)
- Expected speedup: 25-40% with no accuracy loss"

# Test Phase 1
python scripts/run_backtest.py --model lgbm --n-timeframes 2
# Verify accuracy, commit results

# Apply Phase 2
git add Makefile
git commit -m "Performance: Enable LGBM cluster parallelization

- Add --parallel --workers 8 to backtest-all targets
- Expected speedup: 8x per model (total 20 min for 3 models)
- Safe: ProcessPoolExecutor, independent cluster training"

# Run full test
make backtest-all

# Push and create PR
git push origin perf/lgbm-optimization
```

---

## Performance Measurement Template

```bash
# Baseline (before any changes)
echo "=== BASELINE ===" > perf_results.txt
time python scripts/run_backtest.py --model lgbm >> perf_results.txt 2>&1
grep "accuracy_overall\|WAPE" data/backtest/lgbm_cluster/backtest_metadata.json >> perf_results.txt

# Phase 1 (after config changes)
echo "=== PHASE 1 ===" >> perf_results.txt
time python scripts/run_backtest.py --model lgbm >> perf_results.txt 2>&1
grep "accuracy_overall\|WAPE" data/backtest/lgbm_cluster/backtest_metadata.json >> perf_results.txt

# Phase 1+2 (after Makefile parallelization)
echo "=== PHASE 1+2 ===" >> perf_results.txt
time make backtest-all >> perf_results.txt 2>&1
grep "accuracy_overall\|WAPE" data/backtest/lgbm_cluster/backtest_metadata.json >> perf_results.txt

# Summary
cat perf_results.txt
```

---

## Questions?

Refer to:
- **LGBM_PERFORMANCE_TUNING_GUIDE.md** - Full analysis & rationale
- **LGBM_BOTTLENECK_ANALYSIS.md** - Technical deep dive
- **CLAUDE.md** - Critical rules & patterns

