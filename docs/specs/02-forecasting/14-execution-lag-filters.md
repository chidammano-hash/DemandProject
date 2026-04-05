# Execution Lag Filters for Experiment Tabs

> Adds a LagFilterBar segmented control to the Algorithm Experiments and Champion Experiments tabs so planners can evaluate model accuracy at each forecast horizon (lags 0–4) rather than only at the portfolio level.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Model Tuning, Champion Experiments |
| **Key Files** | `scripts/run_champion_experiment.py`, `common/ml/backtest_framework.py`, `frontend/src/components/LagFilterBar.tsx` |

---

**Affects:** Algorithm Experiments (Model Tuning), Champion Experiments
**Does not affect:** Cluster Experiments (no forecast accuracy dimension)

---

## 1. Concepts

### 1.1 Execution Lag (DFU property)

Every DFU (item + customer + location) has a single **execution lag** stored in `dim_sku.execution_lag`. It represents the planning horizon — how many months before the demand month that orders must be committed, driven by total lead time:

| Lead time | Execution lag | Meaning |
|---|---|---|
| < 30 days | 0 | Orders placed same month; 1-month-ahead forecast |
| < 60 days | 1 | 2-month-ahead forecast |
| < 90 days | 2 | 3-month-ahead forecast |
| < 120 days | 3 | 4-month-ahead forecast |
| >= 120 days | 4 | 5-month-ahead forecast |

A DFU has **exactly one** execution lag. It never changes between lags.

### 1.2 Forecast Lag (prediction property)

When the backtest framework generates predictions across 10 timeframes (A-J), each timeframe has a different training data cutoff. The **natural lag** of a prediction is computed from how far ahead the model was predicting: `lag = months(startdate - train_end) - 1`.

`assign_natural_lags()` produces up to **5 rows per DFU-month** — one for each forecast lag 0 through 4. Each row is a **genuinely different prediction** from a different timeframe:

- `lag = 0` → prediction from the most recent timeframe (1-month-ahead, highest accuracy)
- `lag = 4` → prediction from an earlier timeframe (5-month-ahead, lowest accuracy)

For example, Feb 2026 predictions:
- Lag 0: from Timeframe J (trained through Jan 2026)
- Lag 1: from Timeframe I (trained through Dec 2025)
- Lag 2: from Timeframe H (trained through Nov 2025)
- Lag 3: from Timeframe G (trained through Oct 2025)
- Lag 4: from Timeframe F (trained through Sep 2025)

### 1.3 Execution-Lag-Matched Prediction

The **production-relevant** prediction for a DFU is the one where `lag == execution_lag`. This is the only prediction that matters for ordering decisions.

- `assign_execution_lag()` in the backtest framework creates exactly these rows
- Portfolio accuracy in tuning runs is computed from these rows only
- Champion selection operates only on these rows (`WHERE lag::text = execution_lag::text`)

---

## 2. Current State

### 2.1 Algorithm Experiments (Model Tuning) — CORRECT semantics

**Data source:** `lgbm_tuning_lag` table, populated by `seed_production_baselines.py`

Population code:
```python
# seed_production_baselines.py
df = pd.read_csv(all_lags_path)  # all-lags archive (5 rows per DFU-month)
for lag_val, grp in df.groupby("lag"):  # groups by forecast lag 0-4
    metrics = _compute_accuracy(grp)
    INSERT INTO lgbm_tuning_lag (run_id, exec_lag, ...) VALUES (...)
```

| UI button | DB query | What it actually shows |
|---|---|---|
| **Exec Lag** (Portfolio) | `lgbm_tuning_run` portfolio columns | Accuracy at each DFU's own execution lag (production-relevant) |
| **Lag 0** | `lgbm_tuning_lag WHERE exec_lag = 0` | **ALL DFUs** measured at forecast lag 0 |
| **Lag 1** | `lgbm_tuning_lag WHERE exec_lag = 1` | **ALL DFUs** measured at forecast lag 1 |
| **Lag 2** | `lgbm_tuning_lag WHERE exec_lag = 2` | **ALL DFUs** measured at forecast lag 2 |
| **Lag 3** | `lgbm_tuning_lag WHERE exec_lag = 3` | **ALL DFUs** measured at forecast lag 3 |
| **Lag 4** | `lgbm_tuning_lag WHERE exec_lag = 4` | **ALL DFUs** measured at forecast lag 4 |

**Semantics are correct.** "Exec Lag" shows production accuracy. "Lag 0-4" shows cross-sectional accuracy at each horizon for all DFUs. Each lag has a genuinely different `basefcst_pref` because the backtest archive uses `assign_natural_lags()` — each lag comes from a different timeframe with a different training data cutoff.

**Note:** The column in `lgbm_tuning_lag` is named `exec_lag` but stores the **forecast lag** (0-4). This naming inconsistency is inherited; a future rename to `forecast_lag` would improve clarity.

### 2.2 Champion Experiments — WRONG semantics

**Data source:** `champion_experiment_lag` table, populated by `run_champion_experiment.py`

Population code:
```python
# run_champion_experiment.py
# monthly_errors_df loaded with lag_mode="execution"
# → SQL: WHERE lag::text = execution_lag::text
# → only rows where forecast lag matches DFU's execution lag

for lag_val, lag_group in monthly_errors_df.groupby("execution_lag"):
    # lag_val = 0, 1, 2, 3, 4
    # lag_group = DFUs whose execution_lag == lag_val
    lag_winners = strategy_fn(lag_group, **strat_kwargs)
```

| UI button | DB query | What it actually shows |
|---|---|---|
| **Exec Lag** (Portfolio) | `champion_experiment` portfolio columns | Champion accuracy across all DFUs at their own execution lag |
| **Lag 0** | `champion_experiment_lag WHERE exec_lag = 0` | Champion accuracy for **only DFUs with execution_lag=0** (short-lead-time items) |
| **Lag 3** | `champion_experiment_lag WHERE exec_lag = 3` | Champion accuracy for **only DFUs with execution_lag=3** (long-lead-time items) |

**This is fundamentally different from Algorithm Experiments:**
- Algorithm "Lag 0" = ALL DFUs at forecast horizon 0
- Champion "Lag 0" = only DFUs whose execution lag IS 0 (a subset of DFUs)

The champion per-lag breakdown is actually a **cohort breakdown by DFU lead-time class**, not a cross-sectional view by forecast horizon.

---

## 3. Target State

### 3.1 Unified Semantics

Both experiment types should show the same thing when a lag button is pressed:

| Button | Meaning (both tabs) |
|---|---|
| **Exec Lag** (Portfolio) | Each DFU measured at its own execution lag. Production-relevant accuracy. |
| **Lag 0** | ALL DFUs measured at forecast lag 0 (1-month-ahead). |
| **Lag 1** | ALL DFUs measured at forecast lag 1 (2-month-ahead). |
| **Lag 2** | ALL DFUs measured at forecast lag 2 (3-month-ahead). |
| **Lag 3** | ALL DFUs measured at forecast lag 3 (4-month-ahead). |
| **Lag 4** | ALL DFUs measured at forecast lag 4 (5-month-ahead). |

### 3.2 Algorithm Experiments — No changes needed

The current implementation already has the correct semantics. `seed_production_baselines.py` groups the all-lags archive by `lag` (forecast lag), so `lgbm_tuning_lag WHERE exec_lag = 0` already returns all DFUs at forecast lag 0.

### 3.3 Champion Experiments — Needs fix

The champion runner must be changed to compute per-lag breakdowns using **all-lags data**, not execution-lag-filtered data.

#### 3.3.1 Script change: `run_champion_experiment.py`

**Current** (step 8, line ~300):
```python
# Only has execution-lag-matched rows
for lag_val, lag_group in monthly_errors_df.groupby("execution_lag"):
    lag_winners = strategy_fn(lag_group, **strat_kwargs)
```

**Target:**
```python
# Load all-lags data for per-lag breakdown
all_lags_df = load_monthly_errors_df(db, models, lag_mode="all")
for lag_val, lag_group in all_lags_df.groupby("lag"):  # forecast lag 0-4, ALL DFUs
    lag_winners = strategy_fn(lag_group, **strat_kwargs)
```

This requires:
1. A new `lag_mode="all"` option in `load_monthly_errors_df()` that loads from `backtest_lag_archive` with no lag filter (all 5 lags per DFU-month)
2. The champion strategy runs independently on each lag slice (all DFUs at that forecast horizon)
3. Store results in `champion_experiment_lag` as before

#### 3.3.2 Data source: `load_monthly_errors_df` changes

Add a new lag mode:

```python
def load_monthly_errors_df(db, models, lag_mode):
    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"  # existing
    elif lag_mode == "all":
        lag_cond = "TRUE"  # no lag filter — returns 5 rows per DFU-month
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))
```

#### 3.3.3 Portfolio metric stays the same

The overall champion accuracy (stored in `champion_experiment` columns) remains computed from execution-lag-matched data. Only the per-lag breakdown changes.

```
Overall:     strategy_fn(monthly_errors_df)        # lag_mode="execution"
Per-lag:     strategy_fn(all_lags_df[lag == X])     # lag_mode="all", grouped by forecast lag
```

---

## 4. API Changes

### 4.1 Algorithm Experiments (no changes)

Endpoints already work correctly:
- `GET /model-tuning/{model}/experiments?exec_lag=N` — overrides portfolio KPIs with `lgbm_tuning_lag WHERE exec_lag = N`
- `GET /model-tuning/{model}/compare?exec_lag=N` — same override for comparison

### 4.2 Champion Experiments (already implemented, semantics fix pending)

Endpoints already accept `exec_lag` param (implemented in prior change):
- `GET /champion-experiments?exec_lag=N` — overrides portfolio KPIs with `champion_experiment_lag WHERE exec_lag = N`
- `GET /champion-experiments/{id}?exec_lag=N` — same for detail
- `GET /champion-experiments/compare?exec_lag=N` — same for comparison

**No API changes needed** — once the script populates `champion_experiment_lag` with forecast-lag-based data (section 3.3), the existing API endpoints will return the correct values.

---

## 5. Frontend Changes

### 5.1 Already Implemented

- `LagFilterBar` renamed "All" to "Exec Lag" (sub-label "Portfolio")
- `ChampionExperimentsPanel` now renders `<LagFilterBar>` and passes `execLag` to fetch + comparison
- `ChampionComparisonPanel` accepts `execLag` prop and passes to compare API
- `champion-experiments.ts` fetchers accept `exec_lag` param

### 5.2 No Additional Frontend Changes Needed

The UI is already wired. The semantic fix is entirely in the backend script (`run_champion_experiment.py`) that populates the lag table.

---

## 6. DB Schema

### 6.1 No Schema Changes

`champion_experiment_lag` table already has the right structure:
```sql
CREATE TABLE champion_experiment_lag (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    exec_lag INTEGER NOT NULL,       -- will now store forecast lag (0-4)
    champion_accuracy NUMERIC(8, 4),
    ceiling_accuracy NUMERIC(8, 4),
    gap_bps NUMERIC(8, 2),
    n_dfu_months INTEGER,
    model_distribution JSONB,
    UNIQUE (experiment_id, exec_lag)
);
```

**Note:** The column name `exec_lag` is used in both `lgbm_tuning_lag` and `champion_experiment_lag`. In both cases it will store the **forecast lag** (0-4, all DFUs at that horizon). The naming is inherited from the model tuning table. A future rename to `forecast_lag` would improve clarity but is not required for correctness.

---

## 7. Implementation Steps

| Step | File(s) | Description | Status |
|---|---|---|---|
| 1 | `common/ml/backtest_framework.py` | Replace `expand_to_all_lags()` with `assign_natural_lags()` — computes true forecast lag from timeframe training cutoff (`lag = months(startdate - train_end) - 1`) | Done |
| 2 | `common/ml/backtest_framework.py` | Update `postprocess_predictions()` to accept `timeframes` list and use natural lags for archive | Done |
| 3 | `scripts/run_champion_selection.py` | Add `lag_mode="all"` support to `load_monthly_errors_df()` — queries `backtest_lag_archive`, no lag filter | Done |
| 4 | `scripts/run_champion_experiment.py` | Load all-lags data separately for per-lag breakdown; group by `lag` column (not `execution_lag`) | Done |
| 5 | `tests/unit/test_natural_lags.py` | 12 unit tests: verify natural lag formula, timeframe mapping, lag coverage, fcstdate, forecast_ck | Done |
| 6 | `tests/unit/test_champion_lag_modes.py` | Unit tests: verify correct table routing and SQL per lag_mode | Done (5 tests) |
| 7 | Re-run backtests (`make backtest-all`) | Regenerate archive CSVs with natural lags (genuinely different predictions per lag) | Manual step |
| 8 | Re-run `make champion-all` | Regenerate champion lag data using corrected archive | Manual step |

### 7.1 Backward Compatibility

- Old `champion_experiment_lag` rows (from before the fix) will have cohort-based data. They are still valid — just interpreted differently.
- No migration needed. Re-running an experiment overwrites via `ON CONFLICT ... DO UPDATE`.
- The API response shape is unchanged.

---

## 8. Verification

After implementing, verify with a completed champion experiment:

1. **Lag 0 should have the highest accuracy** — 1-month-ahead predictions are easiest
2. **Lag 4 should have the lowest accuracy** — 5-month-ahead predictions are hardest
3. **n_dfu_months should be the same across all lags** — every DFU has predictions at every lag
4. **Exec Lag (portfolio) accuracy should match** — unchanged, still uses execution-lag-matched data

If n_dfu_months varies by lag, the old (broken) behavior is still active.

---

## 9. Summary

| Component | Current | Target | Change needed |
|---|---|---|---|
| LagFilterBar label | ~~All~~ → Exec Lag | Exec Lag | Done |
| Algorithm Experiments per-lag | All DFUs at forecast lag N | Same | None |
| Champion Experiments per-lag | DFUs with execution_lag=N only | All DFUs at forecast lag N | Done |
| Champion API exec_lag param | Not present | Present | Done |
| Champion UI LagFilterBar | Not present | Present | Done |
| Champion Experiments per-lag | DFUs with execution_lag=N only | All DFUs at forecast lag N | Fix script |
| Champion API exec_lag param | Not present | Present | Done |
| Champion UI LagFilterBar | Not present | Present | Done |
