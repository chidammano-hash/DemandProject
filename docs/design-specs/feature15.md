# Feature 15 — Champion Model Selection (Best-of-Models)

## Overview

Automatically identify the best-performing forecasting model **per DFU per month** using a causally-correct expanding window of prior performance, and create a composite "champion" forecast. This implements the industry-standard **Forecast Value Added (FVA)** approach used in demand planning — the champion simulates what a planner would do: at each month, pick the model that has performed best historically (before-the-fact).

The ceiling (oracle) picks the best model per DFU per month using that month's actual error — the theoretical upper bound with perfect foresight. The gap between champion and ceiling quantifies how close the rolling selection is to optimal.

---

## Problem

With 13+ forecasting models, users need an automated way to:
1. Compare model performance at the DFU level (not just aggregate)
2. Select the best model for each DFU at each month based on prior performance
3. Create a composite "champion" forecast from the per-DFU per-month winners
4. Fill warm-up gaps (early months with insufficient history) with a reliable fallback model
5. Track which models win the most DFU-months (forecast value added analysis)
6. Benchmark against a theoretical ceiling (oracle) to quantify improvement opportunity

---

## Architecture

### Champion Selection Flow

```
config/model_competition.yaml
          ↓
  run_champion_selection.py
          ↓
  ┌────────────────────────────────────────────────┐
  │ For each DFU × month:                          │
  │   Compute cumulative WAPE per model from       │
  │   causally-available prior months only         │
  │   (startdate < fcstdate = T - exec_lag)        │
  │   → Pick model with lowest prior WAPE          │
  │     (before-the-fact, exec-lag-aware)          │
  └──────────────────┬─────────────────────────────┘
                     ↓
  ┌────────────────────────────────────────────────┐
  │ Warm-up months (< exec_lag + min_dfu_rows):    │
  │   Insert fallback model (default: lgbm_cluster)│
  │   so every DFU-month has a champion row        │
  └──────────────────┬─────────────────────────────┘
                     ↓
  fact_external_forecast_monthly
    (model_id = 'champion')
             ↓
  ┌────────────────────────────────────────────────┐
  │ For each DFU × month:                          │
  │   Pick model with lowest absolute error FOR    │
  │   that month (after-the-fact oracle)           │
  └──────────────────┬─────────────────────────────┘
                     ↓
  fact_external_forecast_monthly
    (model_id = 'ceiling')
             ↓
  Refresh materialized views
             ↓
  Champion + Ceiling auto-appear
  in all accuracy comparison views
```

### Key Design Decision: model_id = 'champion' / 'ceiling'

Both forecasts are stored in the same `fact_external_forecast_monthly` table using `model_id = 'champion'` and `model_id = 'ceiling'`. Because all existing accuracy views, lag curves, and KPI computations are model_id-aware, they automatically appear in every comparison with **zero downstream changes**.

---

## Configuration

### `config/model_competition.yaml`

```yaml
competition:
  name: "default"
  metric: "wape"              # wape (lowest wins) or accuracy_pct (highest wins)
  lag: "execution"            # "execution" (per-DFU) or 0, 1, 2, 3, 4
  min_dfu_rows: 3             # min prior months required before champion can be selected
  champion_model_id: "champion"
  fallback_model_id: "lgbm_cluster"   # used for warm-up months without enough history
  ceiling_model_id: "ceiling"         # oracle ceiling (best model per DFU per month)
  models:
    - lgbm_cluster
    - catboost_cluster
    - xgboost_cluster
```

| Field | Description |
|-------|-------------|
| `metric` | `wape` (lowest wins) or `accuracy_pct` (highest wins). WAPE is the industry default. |
| `lag` | `execution` uses each DFU's own execution lag; integers 0-4 for fixed horizon |
| `min_dfu_rows` | Minimum **causally-available** prior months required before champion can be selected |
| `champion_model_id` | The model_id used for champion rows (default: `champion`) |
| `fallback_model_id` | The model used when insufficient prior history; defaults to `lgbm_cluster` |
| `ceiling_model_id` | The model_id used for ceiling/oracle rows (default: `ceiling`) |
| `models` | List of model_ids to compete; configurable from the UI |

---

## Execution-Lag Causality

### The Problem

Each DFU has an `execution_lag` — the number of months in advance that its forecast is issued. For a DFU with `execution_lag = L`:

- The forecast for month **T** (startdate = T) is **issued in month T − L** (fcstdate = T − L)
- At issuance time, the planner has actuals only for months with `startdate < T − L`
- Using actuals from months `T−L`, `T−L+1`, …, `T−1` would be a **data leak** — those actuals were not available when the forecast was issued

The original champion selection used `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING ORDER BY startdate`, which includes all months before T — including months in the `[T−L, T−1]` window that weren't available at issuance time.

### The Fix

For each DFU-model group sorted by `startdate`, apply `shift(exec_lag + 1)` before any cumulative or rolling computation. This skips the last `exec_lag + 1` rows so that:

- Month T's champion uses only months with `startdate < T − L` (= `startdate < fcstdate`)
- With `exec_lag = 0` the fix is identical to `shift(1)` — **fully backward compatible**
- With `exec_lag = L > 0`, the last L months of prior data are correctly excluded

---

## Concrete Illustration: April 2025 Champion Pick (exec_lag = 1)

### Setup

- DFU: `ITEM-X / GRP-A / LOC-1`
- `execution_lag = 1` — forecasts are issued **one month in advance**
- Two competing models: `lgbm_cluster` (A), `catboost_cluster` (B)
- `min_dfu_rows = 3`

### Raw Backtest Errors (execution-lag rows)

| startdate | fcstdate | Model | Forecast | Actual | |F−A| |
|-----------|----------|-------|----------|--------|------|
| Jan 2025  | Dec 2024 | A     | 110      | 100    | 10   |
| Jan 2025  | Dec 2024 | B     | 95       | 100    | 5    |
| Feb 2025  | Jan 2025 | A     | 115      | 100    | 15   |
| Feb 2025  | Jan 2025 | B     | 90       | 100    | 10   |
| Mar 2025  | Feb 2025 | A     | 105      | 100    | 5    |
| Mar 2025  | Feb 2025 | B     | 115      | 100    | 15   |
| Apr 2025  | Mar 2025 | A     | 108      | 100    | 8    |
| Apr 2025  | Mar 2025 | B     | 98       | 100    | 2    |

### When is the April 2025 champion selected?

The forecast for April 2025 is **issued in March 2025** (fcstdate = Mar 2025). At that moment:
- Jan 2025 actuals ✅ available (startdate < Mar 2025)
- Feb 2025 actuals ✅ available (startdate < Mar 2025)
- Mar 2025 actuals ❌ **not yet available** (startdate = Mar 2025, not < Mar 2025)

With `shift(exec_lag + 1) = shift(2)`, the prior window for April is **Jan + Feb only**.

### Prior WAPE Computation for April 2025

**Model A (lgbm_cluster):** prior window = Jan + Feb
- cum_abs_err = 10 + 15 = 25
- cum_actual  = 100 + 100 = 200
- prior_WAPE  = 25 / 200 = **0.125**

**Model B (catboost_cluster):** prior window = Jan + Feb
- cum_abs_err = 5 + 10 = 15
- cum_actual  = 100 + 100 = 200
- prior_WAPE  = 15 / 200 = **0.075**

**Winner for April 2025: Model B** (prior WAPE 0.075 < 0.125)

### Champion Row Inserted

```
model_id   = 'champion'
startdate  = 2025-04-01
basefcst_pref = 98  (Model B's April forecast)
tothist_dmd   = 100
```

### What if exec_lag = 0?

With `shift(1)` the prior window for April would include Jan + Feb + **Mar** — and Mar was the last available actual before April. For `exec_lag = 0` the forecast is issued **in the same month** as the target, so March actuals are indeed available at issuance time. The `shift(exec_lag + 1)` formula handles both cases correctly.

### Warm-Up Period

With `exec_lag = 1` and `min_dfu_rows = 3`, the first qualifying month is the **4th row** (after shifting by 2, you need 3 non-null priors → startdate = Apr 2025 is the earliest qualifying month in this example). Jan, Feb, Mar 2025 have no champion selection — those get the **fallback model** (`lgbm_cluster` by default).

| startdate | Champion selected? | Source |
|-----------|-------------------|--------|
| Jan 2025  | No (0 prior months) | fallback → lgbm_cluster |
| Feb 2025  | No (0 prior months) | fallback → lgbm_cluster |
| Mar 2025  | No (1 prior month, need 3) | fallback → lgbm_cluster |
| Apr 2025  | **Yes** (Jan+Feb = 2 prior months) | winner (model B) |

Wait — with `shift(2)` on Jan through Apr sorted:
- Jan: shifted from -∞ → 0 non-null priors (no champion)
- Feb: shifted from -∞ → 0 non-null priors (no champion)
- Mar: shifted Jan → 1 non-null prior (< 3, no champion)
- Apr: shifted Jan+Feb → 2 non-null priors (< 3, no champion)
- May: shifted Jan+Feb+Mar → 3 non-null priors (**qualifies!**)

So with `exec_lag = 1, min_dfu_rows = 3`, the first qualifying month is **May 2025**. Jan–Apr all get the fallback model.

---

## Selection Algorithm

### Champion (Exec-Lag-Aware Expanding Window — Before-the-Fact)

1. **Load** per-model per-DFU per-month errors at the configured lag, including `execution_lag` and `fcstdate` from the database
2. **Apply shift**: For each (DFU, model) group sorted by `startdate`, shift errors by `exec_lag + 1` before cumulative sums. This makes the prior window for month T = months with `startdate < fcstdate = T − exec_lag`.
3. **Compute prior WAPE** per model: `cum_abs_err / |cum_actual|` from the shifted expanding window
4. **Filter** models with fewer than `min_dfu_rows` non-null shifted priors
5. **Rank** models within each DFU-month by prior WAPE ascending
6. **Select** winner (rank = 1) per DFU-month → insert with `model_id = 'champion'`
7. **Fill gaps**: For DFU-months where no champion was selected (warm-up period), insert the fallback model's forecast with `model_id = 'champion'` (NOT EXISTS guard + ON CONFLICT DO NOTHING)

### Fallback Model

When a DFU-month cannot qualify for champion selection (insufficient causally-available prior history), the `fallback_model_id` is used instead. Default: `lgbm_cluster`.

This ensures every DFU-month with backtest data always has a champion row — important for complete accuracy analysis across the full history.

The fallback insert is **idempotent**: it only fills gaps (NOT EXISTS sub-select) and uses ON CONFLICT DO NOTHING for safety.

### Ceiling / Oracle (After-the-Fact — Perfect Foresight)

1. **Compute absolute error** per row: `ABS(basefcst_pref - tothist_dmd)`
2. **Rank** models within each DFU-month by absolute error ascending
3. **Select** winner (rank = 1) per DFU-month
4. **Insert** with `model_id = 'ceiling'`

The ceiling is **not deployable** — it uses actuals retroactively. It serves as a benchmark for the champion gap.

### Champion vs Ceiling

| Aspect | Champion | Ceiling |
|--------|----------|---------|
| Decision basis | Prior months only (before-the-fact, exec-lag-aware) | Current month actuals (after-the-fact) |
| Granularity | Per DFU per month | Per DFU per month |
| Oracle? | No — deployable strategy | Yes — theoretical upper bound |
| WAPE formula | `SUM(|F-A|) / |SUM(A)|` | `SUM(|F-A|) / |SUM(A)|` |
| Warm-up gaps | Filled by fallback model | Covered by all models |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/competition/config` | GET | Return current config + available model_ids from DB |
| `/competition/config` | PUT | Update config (writes YAML to disk) |
| `/competition/run` | POST | Execute champion selection, return summary |
| `/competition/summary` | GET | Return last run summary from disk |

### POST /competition/run Response

```json
{
  "config": {
    "metric": "wape", "lag": "execution",
    "fallback_model_id": "lgbm_cluster", ...
  },
  "total_dfus": 5432,
  "total_dfu_months": 43000,
  "total_champion_rows": 43000,
  "fallback_rows_inserted": 12500,
  "model_wins": {
    "lgbm_cluster": 15000,
    "catboost_cluster": 12000,
    ...
  },
  "overall_champion_wape": 28.5,
  "overall_champion_accuracy_pct": 71.5,
  "total_ceiling_rows": 54000,
  "ceiling_model_wins": {
    "lgbm_cluster": 18000,
    "catboost_cluster": 15000,
    ...
  },
  "overall_ceiling_wape": 18.2,
  "overall_ceiling_accuracy_pct": 81.8,
  "run_ts": "2026-02-25T10:30:00+00:00"
}
```

---

## Frontend UI

### Champion Selection Panel (Accuracy Tab)

Located below the Accuracy Comparison table in the Accuracy tab:

1. **Model Checkboxes**: Toggle which models compete (excludes `champion` and `ceiling`)
2. **Metric Selector**: WAPE (Lowest Wins) or Accuracy % (Highest Wins)
3. **Lag Selector**: Execution Lag or fixed lag 0-4
4. **Save Config**: Persists checkbox/metric/lag changes to YAML
5. **Run Competition**: Executes champion selection + ceiling computation (auto-saves config first)
6. **Champion Results**: DFUs evaluated, DFU-months, champion accuracy/WAPE, champion rows count, fallback rows count, and champion model wins bar chart (indigo)
7. **Ceiling Results**: Ceiling accuracy/WAPE (emerald green), ceiling rows count, gap-to-ceiling indicator (amber, in percentage points), and ceiling model wins bar chart (emerald)

### Gap to Ceiling

The "Gap to Ceiling" KPI card shows how many percentage points the champion accuracy is below the ceiling accuracy. A small gap means the rolling champion selection is near-optimal; a large gap suggests room for improvement in the selection algorithm.

---

## CLI

```bash
make champion-select  # Run champion selection from config
```

Equivalent to:
```bash
uv run python scripts/run_champion_selection.py --config config/model_competition.yaml
```

---

## Key Files

| File | Purpose |
|------|---------|
| `config/model_competition.yaml` | Competition configuration (models, strategy, fallback) |
| `scripts/run_champion_selection.py` | Main champion selection script |
| `common/champion_strategies.py` | 5 strategy functions + exec-lag-aware shift helpers |
| `api/routers/competition.py` | API endpoints for config, run, summary |
| `frontend/src/tabs/AccuracyTab.tsx` | Champion Selection UI panel |
| `data/champion/champion_summary.json` | Last run summary (generated) |
| `tests/unit/test_champion_selection.py` | Unit tests: summary, config, fallback logic |
| `tests/unit/test_champion_strategies.py` | Strategy unit tests including exec-lag causality tests |

---

## Design Rationale

| Decision | Why |
|----------|-----|
| `shift(exec_lag + 1)` causal prior | Prevents using actuals not yet available at forecast issuance. With exec_lag=0 identical to previous behavior. |
| Fallback model for warm-up gaps | Ensures complete champion coverage for all DFU-months, enabling full-history accuracy analysis |
| Rolling/expanding window champion (before-the-fact) | Simulates what a planner would do: pick best model based on available history. No data leak. |
| Ceiling as separate oracle model | Provides theoretical upper bound; quantifies gap to perfect foresight |
| Both at DFU-month granularity | Makes gap-to-ceiling directly comparable (same denominator) |
| WAPE as selection metric | Volume-weighted, handles zero-demand months; industry standard |
| `SUM(|F-A|) / |SUM(A)|` for both | Consistent formula makes gap calculation meaningful |
| Store as model_id='champion'/'ceiling' | Reuses existing fact table + views; zero downstream changes |
| Fallback insert with NOT EXISTS + ON CONFLICT DO NOTHING | Idempotent; only fills gaps; safe to re-run |
| YAML config (not DB table) | Matches clustering config pattern; simple and git-trackable |
| DELETE + INSERT for champion/ceiling | Idempotent full replace; consistent with backtest loading |

---

## Enhanced Selection Strategies (v2)

### Strategy Registry

All strategies are defined in `common/champion_strategies.py` via a `STRATEGY_REGISTRY` pattern:

```python
STRATEGY_REGISTRY: dict[str, Callable] = {}

@register_strategy("expanding")
def strategy_expanding(df, *, min_prior_months=3, **kwargs): ...
```

### 5 Strategies

| Strategy | Key Idea | Exec-Lag Implementation |
|----------|----------|-----------------------|
| `expanding` | Cumulative WAPE, all prior months equal weight | `shift(exec_lag+1).expanding()` per DFU-model group |
| `rolling` | Last N months only (default N=6) — adapts faster to regime changes | `shift(exec_lag+1).rolling(N)` per DFU-model group |
| `decay` | Exponential decay (recent months weighted more, `decay_factor=0.9`) | Iterates `months[:i - exec_lag]` |
| `ensemble` | Blend top-K models by inverse-WAPE weights | Weights from `shift(exec_lag+1).expanding()` WAPE |
| `meta_learner` | ML classifier predicts best model from DFU features + performance stats | Rolling features via `shift(exec_lag+1)` per group |

All strategies share the helper `_get_exec_lag(group)` which returns the DFU's execution_lag (defaults to 0 if column absent for backward compatibility).

### Configuration

```yaml
competition:
  strategy: expanding          # expanding | rolling | decay | ensemble | meta_learner
  strategy_params:
    window_months: 6           # rolling window size
    decay_factor: 0.90         # exponential decay factor
    top_k: 3                   # ensemble top-K models
    weight_method: inverse_wape  # ensemble weighting: inverse_wape or equal
    performance_window: 6      # meta-learner feature window
  meta_learner:
    model_type: random_forest
    n_estimators: 200
    max_depth: 15
    test_months: 3
    performance_window: 6
```

### Meta-Learner

The meta-learner is a supervised classifier that predicts which model will perform best for a given DFU-month based on:

- **Static DFU features:** ml_cluster, abc_vol, execution_lag, total_lt, brand, region, seasonality_profile, seasonality_strength, is_yearly_seasonal, peak_month, trough_month, peak_trough_ratio
- **Recent model performance (per model):** rolling WAPE from strictly prior N months via `shift(exec_lag+1).rolling(N)`
- **Calendar features:** month, quarter, month_sin, month_cos
- **Demand stats:** mean_qty, cv_demand from `shift(exec_lag+1).expanding()` only

**Training:** Uses ceiling (oracle) labels as ground truth. Strict temporal train/test split — no random splitting.

**Output:** `data/champion/meta_learner.joblib`

### Simulation Framework

`scripts/simulate_champion_strategies.py` runs all strategies on historical data and compares accuracy vs ceiling:

```
Strategy              Accuracy   WAPE    Gap to Ceil  DFU-months
expanding             71.50%    28.50%   10.30 pp     43,000
rolling_6m            73.20%    26.80%    8.60 pp     42,500
decay_090             73.80%    26.20%    8.00 pp     43,000
ensemble_top3         75.40%    24.60%    6.40 pp     43,000
meta_learner          76.80%    23.20%    5.00 pp     43,000
```

### CLI

```bash
make champion-select       # Run champion selection using configured strategy
make champion-simulate     # Simulate all strategies, compare accuracy vs ceiling
make champion-train-meta   # Train meta-learner classifier
make champion-all          # train-meta + simulate + select (full pipeline)
```

---

## Exec-Lag Causality Fix Details

### SQL Fallback (compute_champion_winners)

The SQL path uses a self-join instead of a window function, since window functions can't vary their frame per row based on `execution_lag`:

```sql
FROM monthly_errors a
LEFT JOIN monthly_errors b
    ON  a.dmdunit  = b.dmdunit
    AND a.dmdgroup = b.dmdgroup
    AND a.loc      = b.loc
    AND a.model_id = b.model_id
    AND b.startdate < a.fcstdate   -- causal cutoff: prior to issuance
```

`fcstdate` = startdate − execution_lag months, so `b.startdate < a.fcstdate` is equivalent to `b.startdate < T − L`, enforcing the exec-lag causal cutoff correctly in SQL.

### Python Strategies (champion_strategies.py)

```python
def _get_exec_lag(group: pd.DataFrame) -> int:
    """Return execution_lag for a DFU group; defaults to 0 if column absent."""
    if "execution_lag" in group.columns and len(group) > 0:
        val = group["execution_lag"].iloc[0]
        return int(val) if pd.notna(val) else 0
    return 0

def _expanding_stats(df: pd.DataFrame) -> pd.DataFrame:
    for _, group in df.groupby(["dmdunit","dmdgroup","loc","model_id"], sort=False):
        g = group.sort_values("startdate").copy()
        shift_n = _get_exec_lag(g) + 1          # ← exec-lag-aware shift
        shifted_err = g["abs_err"].shift(shift_n)
        shifted_act = g["tothist_dmd"].shift(shift_n)
        g["cum_abs_err"] = shifted_err.expanding(min_periods=1).sum()
        g["cum_actual"]  = shifted_act.expanding(min_periods=1).sum()
        g["prior_count"] = shifted_err.expanding(min_periods=1).count()
        ...
```

### Backward Compatibility

- `_get_exec_lag()` defaults to 0 when `execution_lag` column is absent → existing callers without the column continue to work identically
- For `exec_lag = 0`: `shift(1)` is the same as the previous implementation

---

## Archive Lag Preservation

Champion selection reads per-model per-DFU errors from `backtest_lag_archive`, which stores all 5 lags (0–4) per forecast row. The archive preserves each row's **original lag as its `execution_lag`** because it is loaded BEFORE the staging table is mutated.

This is critical because the archive's `execution_lag` column tells the champion selection algorithm which lag each row represents. If all archive rows for a DFU had the same `execution_lag` (the DFU-level value from `dim_dfu`), multi-horizon accuracy analysis would be corrupted — every row would appear to be the same lag.

The dual-path loading uses **phase ordering**:
1. **Phase 3b — Archive load**: reads untouched staging data — all 5 rows enter `backtest_lag_archive` with their original `lag` and `execution_lag` values
2. **Phase 3c — Staging mutation**: UPDATEs `execution_lag` from `dim_dfu` (overwrites all rows for each DFU to the DFU-level value)
3. **Phase 5 — Main table insert**: `WHERE lag = execution_lag` — only the execution-lag row enters `fact_external_forecast_monthly`

See `feature2.md` § "Forecast Loading: Dual-Path with Phase Ordering" for a concrete worked example.

---

## Dependencies

- `pyyaml>=6.0.0`
- `psycopg`
- `scikit-learn>=1.3.0` (for meta-learner Random Forest)
- `joblib>=1.3.0` (for meta-learner model serialization)
- Existing `fact_external_forecast_monthly` table with `UNIQUE(forecast_ck, model_id)` constraint and `execution_lag`, `fcstdate` columns
- Existing materialized views (`agg_accuracy_by_dim`, `agg_forecast_monthly`, `agg_dfu_coverage`)


---

## Examples

### Example: Champion competition YAML config

```yaml
# config/model_competition.yaml
competing_models:
  - lgbm_global
  - lgbm_cluster
  - catboost_global
  - external
metric: wape
strategy: rolling
strategy_params:
  window: 3
exec_lag: 2
min_dfu_rows: 6
fallback_model_id: lgbm_cluster
```

### Example: Run champion selection

```bash
make champion-select
# Selects best model per DFU per month using rolling 3-month WAPE
# exec-lag-aware: selection for month T uses only data where startdate < T - exec_lag
# Warm-up gaps filled with fallback model (lgbm_cluster)
# Stored as model_id='champion' in fact_external_forecast_monthly

# Also compute ceiling (oracle best-in-hindsight):
# model_id='ceiling' — picks best model AFTER seeing actuals
```

### Example: Simulate all strategies to find best approach

```bash
make champion-simulate
# Strategy  | Accuracy | WAPE   | Gap to Ceiling
# expanding  |  71.50%  | 28.50% |      10.30 pp
# rolling    |  73.20%  | 26.80% |       8.60 pp
# decay      |  72.80%  | 27.20% |       9.00 pp
# ensemble   |  74.10%  | 25.90% |       7.70 pp
# meta_learner| 75.40%  | 24.60% |       6.40 pp
```

### Example: Champion vs ceiling API comparison

```bash
curl -s "http://localhost:8000/competition/results?lag=2" | jq \
  '[.rows[] | select(.model_id == "champion" or .model_id == "ceiling") | {model_id, accuracy_pct}]'
# [{"model_id": "champion", "accuracy_pct": 73.2},
#  {"model_id": "ceiling",  "accuracy_pct": 81.8}]
# Gap-to-ceiling: 8.6 percentage points improvement opportunity
```


---

## Additional Examples

#### Example — API endpoints: read config and update

```bash
# GET current competition config + available model_ids from DB
curl -s http://localhost:8000/competition/config | jq .
# {
#   "config": {"metric": "wape", "lag": "execution", "strategy": "rolling",
#              "strategy_params": {"window_months": 6}, "min_dfu_rows": 3,
#              "fallback_model_id": "lgbm_cluster", "models": [...]},
#   "available_models": ["lgbm_global","lgbm_cluster","catboost_cluster","external","ceiling","champion"]
# }

# PUT — update config (writes YAML to disk, requires API key when set)
curl -s -X PUT http://localhost:8000/competition/config \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"metric":"wape","lag":"execution","strategy":"ensemble",
       "strategy_params":{"top_k":3,"weight_method":"inverse_wape"},
       "min_dfu_rows":3,"fallback_model_id":"lgbm_cluster",
       "models":["lgbm_cluster","catboost_cluster","xgboost_cluster"]}' \
  | jq '{status}'
# {"status": "config updated"}

# GET last run summary from disk (does not re-run selection)
curl -s http://localhost:8000/competition/summary | jq '{total_dfus, overall_champion_wape, overall_ceiling_wape}'
# {"total_dfus": 5432, "overall_champion_wape": 26.8, "overall_ceiling_wape": 18.2}
```

#### Example — Frontend Champion Selection panel walkthrough

```
1. Open Accuracy tab → scroll to "Champion Model Selection" panel
2. Check models to compete: lgbm_cluster, catboost_cluster, xgboost_cluster
   (champion and ceiling are automatically excluded from the checkbox list)
3. Set Metric = WAPE (Lowest Wins)
4. Set Lag = Execution Lag (uses each DFU's own execution_lag from dim_dfu)
5. Click "Save Config" → PUT /competition/config writes YAML
6. Click "Run Competition" → POST /competition/run (takes ~15–30 seconds)
7. Results panel shows:
     Champion: 5,432 DFUs  |  43,000 DFU-months  |  Accuracy: 73.2%  |  WAPE: 26.8%
     Fallback rows: 12,500 (warm-up gaps filled by lgbm_cluster)
     Model wins bar chart (indigo): lgbm_cluster=15k, catboost_cluster=12k, xgboost_cluster=9k
     Ceiling: Accuracy: 81.8%  |  WAPE: 18.2%
     Gap to Ceiling: 8.6 pp (amber KPI card)
     Ceiling model wins (emerald): lgbm_cluster=18k, catboost_cluster=15k, xgboost_cluster=10k
```

#### Example — 5 strategy comparison

```python
from common.champion_strategies import STRATEGY_REGISTRY

# expanding: cumulative WAPE from all prior causal months
champion_expanding = STRATEGY_REGISTRY["expanding"](df, min_prior_months=3)

# rolling: only last 6 months — adapts faster to demand regime changes
champion_rolling = STRATEGY_REGISTRY["rolling"](df, window_months=6, min_prior_months=3)

# decay: exponential weighting — recent errors count more
champion_decay = STRATEGY_REGISTRY["decay"](df, decay_factor=0.90, min_prior_months=3)

# ensemble: blend top-3 models weighted by inverse-WAPE
champion_ensemble = STRATEGY_REGISTRY["ensemble"](df, top_k=3, weight_method="inverse_wape")

# meta_learner: ML classifier predicts best model from DFU features
# (requires pre-trained meta_learner.joblib from make champion-train-meta)
champion_meta = STRATEGY_REGISTRY["meta_learner"](df, performance_window=6)

# All strategies use exec-lag-aware shift internally:
# shift_n = _get_exec_lag(group) + 1  (defaults to shift(1) when exec_lag=0)
```

#### Example — Meta-learner training and evaluation

```bash
# Step 1: Train meta-learner (uses ceiling labels as ground truth)
make champion-train-meta
# Temporal split: train on months up to -3 months, test on last 3 months
# Features: DFU static attrs + rolling WAPE per model (shift-based, causal)
# Model: RandomForestClassifier(n_estimators=200, max_depth=15)
# Output: data/champion/meta_learner.joblib
# Test accuracy: ~76.8%  (vs random baseline: 33.3% for 3 models)

# Step 2: Inspect meta-learner report
cat data/champion/meta_learner_report.json | jq '{test_accuracy, feature_importances}'
# {"test_accuracy": 0.768,
#  "feature_importances": {"ml_cluster": 0.18, "rolling_wape_lgbm_cluster": 0.15,
#                           "seasonality_strength": 0.12, "mean_qty": 0.10, ...}}

# Step 3: Run champion selection using meta_learner strategy
make champion-select   # uses strategy from config/model_competition.yaml
```

#### Example — Strategy selection guide

```
| Scenario                              | Recommended strategy |
|---------------------------------------|---------------------|
| Stable demand, long history           | expanding            |
| Volatile / frequent regime changes    | rolling (window=3)   |
| Want to reward recent improvement     | decay (factor=0.85)  |
| No single model dominates             | ensemble (top_k=3)   |
| Rich DFU features + seasonality data  | meta_learner         |
```
