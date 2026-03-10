# F2.1 — Confidence Interval Bands for Production Forecast

**Phase:** Evolution to Operations — Phase 2
**Feature Number:** F2.1
**Status:** Not Started
**Priority:** High (enables replenishment safety buffer calibration in F2.2)

---

## 1. Problem Statement

`fact_production_forecast` stores a single point estimate per DFU per horizon month. Point forecasts carry implicit uncertainty that grows with the forecast horizon — a T+6 prediction for a volatile SKU carries far more uncertainty than a T+1 prediction for a stable one — but today every row has `forecast_qty_lower = NULL` and `forecast_qty_upper = NULL`.

This gap causes two downstream failures:

**1. Replenishment under/over-buffering.** The safety stock computation in F2.2 derives a demand sigma from the CI width (`sigma = (upper - lower) / (2 * 1.282)`). With NULL CI bands that formula cannot execute, so F2.2 falls back to a hard-coded global sigma that is wrong for most DFUs.

**2. Planner trust deficit.** When the Demand Forecast panel shows a single line with no uncertainty envelope, planners cannot distinguish a confident near-term prediction from a highly uncertain long-range one. They mentally add their own buffer, usually inconsistently.

### Concrete Failure Scenario

It is March 2026. The champion model forecasts 490 units in April for Item 100320 at LOC 1401-BULK. In reality this DFU has a backtest RMSE of 85 units. Without CI bands:

- The planner sees "490" and orders exactly to cover that.
- The safety stock engine cannot derive sigma, defaults to a global 40-unit buffer.
- The true P10/P90 range (around 380–600 units) is invisible.
- If April actual comes in at 580, the site stocks out — and the signal to adjust the buffer was always in the residuals, never surfaced.

---

## 2. Design Philosophy

### Why Residual-Based, Not Quantile Regression

The obvious alternative is to train separate quantile-regression models (e.g., LGBM with `objective="quantile"`) for each percentile. This approach has two problems in the current architecture:

1. **Backtest artifacts exist.** For every DFU and model, `backtest_lag_archive` already contains side-by-side forecast and actual values across multiple years. This is a free, historically honest source of uncertainty — no new training runs are required.
2. **Quantile models need retraining per target percentile.** Adding P10 and P90 quantile models would roughly triple the backtest compute cost and double the model registry storage footprint.

Residual-based empirical CIs leverage what is already computed. The RMSE from backtest residuals is the natural measure of how wrong the model has historically been at a given horizon, and it is already available.

### Three-Level Fallback Hierarchy

Not every DFU has enough backtest history to produce a reliable DFU-level sigma. The hierarchy ensures every row gets a CI:

```
Level 1: DFU-level sigma     — requires ≥ min_residual_months observations
Level 2: Cluster-level sigma  — pooled from DFU sigmas within the same cluster
Level 3: Global sigma         — median of all cluster sigmas
```

A DFU with < `min_residual_months` residual observations uses its cluster's sigma. A cluster with zero DFU-level sigmas (e.g., a newly created cluster) uses the global fallback. This means CI coverage is always ≥ 0 and in practice exceeds 90% for any production dataset with at least one backtest run.

---

## 3. Residual Data Source

### Where Residuals Come From

Backtest residuals are stored in `backtest_lag_archive` (DDL in `sql/010_create_backtest_lag_archive.sql`). Each row represents one model's prediction for one DFU at one time horizon, alongside the actual demand that was observed.

| Column | Role in CI computation |
|---|---|
| `dmdunit` | DFU item identifier |
| `loc` | DFU location identifier |
| `startdate` | The month being predicted (actuals month) |
| `basefcst_pref` | The model's forecast quantity for that month |
| `tothist_dmd` | The actual demand observed for that month |
| `model_id` | Which algorithm produced this row (e.g., `lgbm_cluster`) |
| `lag` | Execution lag: number of months between forecast issue and actuals month |

**Residual definition:**

```
residual_i = basefcst_pref_i - tothist_dmd_i
```

Positive residual = model over-forecast. Negative residual = model under-forecast. The sign is irrelevant for sigma computation; only the magnitude (via RMSE) matters.

### Which Lag to Use

The `residual_lag` config key (default `0`) controls which lag rows are filtered from the archive. Lag 0 means the forecast was issued the same month as it was meant to cover — the tightest possible horizon, representing execution-period accuracy. For CI bands on a 12-month forward horizon, lag 0 is a conservative (narrow) choice. Teams that want horizon-aware widening should use lag-scaled sigma via `horizon_scaling`.

Only rows where `model_id IN (source_model_ids)` are used — backtest champion/ceiling/external rows are excluded.

### Champion Model Linkage

The CI is calibrated on the champion model's own historical residuals. The champion model for each DFU is identified from `fact_external_forecast_monthly` where `model_id = 'champion'` and `source_model_id` identifies the underlying algorithm (`lgbm_cluster`, `catboost_cluster`, `xgboost_cluster`). The lookup in `build_sigma_lookup` joins this to filter only residuals from the algorithm that is currently the champion for that DFU — preventing a DFU whose champion is CatBoost from being calibrated on LGBM residuals.

---

## 4. Formula Specification

### 4.1 DFU-Level Sigma (RMSE)

For a DFU with residuals `r_1, r_2, ..., r_n` from `backtest_lag_archive`:

```
sigma_dfu = sqrt( mean( r_i^2 ) )   for n >= min_residual_months
```

RMSE is preferred over MAE because it penalises large errors more heavily, producing wider bands for volatile DFUs where large errors occasionally occur. It is also directly interpretable as the standard deviation of the forecast error distribution under the assumption of zero-mean errors.

### 4.2 Cluster-Level Sigma (Weighted Pool)

For a cluster `k` containing DFUs `D_k`:

```
sigma_cluster_k = sum( n_i * sigma_i   for i in D_k ) / sum( n_i   for i in D_k )
```

where `n_i` is the number of residual observations for DFU `i`. DFUs with more data receive proportionally more weight. Only DFUs with `n_i >= min_residual_months` contribute to the cluster pool.

### 4.3 Global Sigma (Median Fallback)

```
sigma_global = median( sigma_cluster_k   for all k )
```

The median is chosen over the mean to prevent a single high-volatility cluster from inflating the global fallback used by all uncharacterised DFUs.

### 4.4 Guard Rails

**Floor:** Every resolved sigma is bounded below by `sigma_floor` (default `1.0`). This prevents zero-width CI bands for DFUs that historically had perfect backtest accuracy (which can happen by chance on small datasets).

**Cap:** A per-DFU cap is applied after the level-selection step:

```
sigma_cap = sigma_cap_multiplier * median( sigma_cluster_k   for all k )
sigma_effective = min( sigma_resolved, sigma_cap )
```

The cap (`sigma_cap_multiplier = 3.0` by default) prevents extreme outlier DFUs with pathological backtest residuals from producing unrealistically wide CI bands that confuse planners.

### 4.5 Horizon Scaling

Forecast uncertainty grows with the horizon. The `horizon_scaling` config key controls how sigma is scaled for each horizon step `h`:

| Mode | Scale factor applied to sigma |
|---|---|
| `sqrt` | `sqrt(h)` — assumes errors accumulate like a random walk |
| `linear` | `h` — conservative; uncertainty grows proportionally with horizon |
| `none` | `1` — flat sigma across all horizons (disable scaling) |

The `sqrt` mode is the statistical default for processes with independent increments (consistent with the recursive inference write-back where each step's error propagates into the next).

### 4.6 CI Bound Computation

For horizon `h` with effective sigma `sigma_eff` and point forecast `f_h`:

```
scale_h       = horizon_scale(h, scaling_mode)
lower_raw_h   = f_h - z_lower * sigma_eff * scale_h
upper_raw_h   = f_h + z_upper * sigma_eff * scale_h

lower_h       = max(0.0, lower_raw_h)   # never negative
upper_h       = max(f_h, upper_raw_h)   # always >= point forecast
```

Default `z_lower = z_upper = 1.282` gives an 80% confidence interval (P10/P90). At the P80 level the bands are interpretable to planners as "we expect 8 out of 10 months to fall inside this range."

---

## 5. Data Model Changes

### 5.1 Existing Table: `fact_production_forecast`

No schema changes are required. The `forecast_qty_lower` and `forecast_qty_upper` columns already exist (DDL in `sql/039_create_production_forecast.sql`). This feature populates them where previously they were left NULL.

| Column | Was | Now |
|---|---|---|
| `forecast_qty_lower` | Always NULL | Populated when `confidence_interval.enabled: true` |
| `forecast_qty_upper` | Always NULL | Populated when `confidence_interval.enabled: true` |

Backward compatibility: if CI is disabled in config, these columns remain NULL. No downstream code breaks on NULL — it simply cannot use the CI formula for sigma derivation.

### 5.2 No New Tables

All inputs come from existing tables: `backtest_lag_archive`, `fact_external_forecast_monthly` (champion assignments), and `dim_dfu` (cluster assignment). No new DDL is needed.

---

## 6. Configuration

### 6.1 Addition to `config/production_forecast_config.yaml`

```yaml
confidence_interval:
  enabled: true

  # Percentile targets for the CI bands (informational — actual bounds use z-scores)
  lower_percentile: 0.10
  upper_percentile: 0.90

  # Z-scores for the lower and upper bounds
  # 1.282 = P10/P90 (80% CI)
  # 1.645 = P5/P95  (90% CI)
  # 1.960 = P2.5/P97.5 (95% CI)
  z_lower: 1.282
  z_upper: 1.282

  # Minimum number of residual observations required to use DFU-level sigma
  # DFUs below this threshold fall back to cluster-level sigma
  min_residual_months: 6

  # Horizon scaling mode: sqrt | linear | none
  horizon_scaling: sqrt

  # Guard rails
  sigma_floor: 1.0               # Minimum sigma (prevents zero-width bands)
  sigma_cap_multiplier: 3.0      # Cap = multiplier × global median sigma

  # Which backtest model IDs to source residuals from
  source_model_ids:
    - lgbm_cluster
    - catboost_cluster
    - xgboost_cluster

  # Which lag from backtest_lag_archive to use for residuals
  # 0 = execution-period accuracy (tightest); increase for horizon-matched calibration
  residual_lag: 0
```

### 6.2 Parameter Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch; when false, CI columns are NULL |
| `lower_percentile` | float | `0.10` | Target lower percentile (documentation only) |
| `upper_percentile` | float | `0.90` | Target upper percentile (documentation only) |
| `z_lower` | float | `1.282` | Z-score for lower bound computation |
| `z_upper` | float | `1.282` | Z-score for upper bound computation |
| `min_residual_months` | int | `6` | Minimum DFU residual rows to use DFU sigma |
| `horizon_scaling` | string | `sqrt` | Horizon scaling mode (`sqrt` / `linear` / `none`) |
| `sigma_floor` | float | `1.0` | Minimum effective sigma after all adjustments |
| `sigma_cap_multiplier` | float | `3.0` | Cap = this × global median cluster sigma |
| `source_model_ids` | list | see above | Backtest model IDs to pull residuals from |
| `residual_lag` | int | `0` | Lag value to filter from `backtest_lag_archive` |

---

## 7. Implementation Files

| File | Action | Description |
|---|---|---|
| `common/forecast_ci.py` | New | 5 functions for sigma computation and CI bound application |
| `config/production_forecast_config.yaml` | Modified | Add `confidence_interval` section |
| `scripts/generate_production_forecasts.py` | Modified | Load sigma lookup before inference loop; populate CI columns |
| `api/routers/production_forecast.py` | Modified | Add `ci_coverage_pct` and `avg_ci_width` to summary endpoint |
| `tests/unit/test_forecast_ci.py` | New | 13+ unit tests for all functions in `common/forecast_ci.py` |
| `tests/api/test_production_forecast_ci.py` | New | 3 API tests for CI-populated summary responses |

---

## 8. Module API — `common/forecast_ci.py`

### 8.1 `load_champion_residuals`

```python
def load_champion_residuals(
    conn,
    source_model_ids: list[str],
    lag: int,
) -> pd.DataFrame:
```

**Purpose:** Load raw backtest residuals from `backtest_lag_archive` and compute the per-row residual value.

**Inputs:**
- `conn` — active psycopg3 connection
- `source_model_ids` — list of algorithm model IDs to include (e.g., `["lgbm_cluster", "catboost_cluster"]`)
- `lag` — integer lag value to filter on (default `0`)

**SQL executed:**
```sql
SELECT
    a.dmdunit,
    a.loc,
    a.startdate,
    a.basefcst_pref,
    a.tothist_dmd,
    a.model_id,
    f.source_model_id AS champion_algo
FROM backtest_lag_archive a
JOIN fact_external_forecast_monthly f
    ON f.dmdunit = a.dmdunit
    AND f.loc    = a.loc
    AND f.model_id = 'champion'
WHERE a.model_id = ANY(%s)
  AND a.lag      = %s
  AND a.tothist_dmd IS NOT NULL
  AND a.basefcst_pref IS NOT NULL
  -- Only include residuals from the algorithm that IS the champion for this DFU
  AND a.model_id = f.source_model_id
```

**Output:** DataFrame with columns `[dmdunit, loc, startdate, basefcst_pref, tothist_dmd, model_id, champion_algo, residual]` where `residual = basefcst_pref - tothist_dmd`.

**Edge cases:**
- Returns empty DataFrame if `backtest_lag_archive` has no rows matching the filter (e.g., no backtest has been run). Caller must handle this gracefully — `compute_dfu_sigma` returns an empty frame, CI computation falls back entirely to global sigma.
- Rows where `tothist_dmd = 0` are included; they represent real zero-demand periods and the residual is valid.

---

### 8.2 `compute_dfu_sigma`

```python
def compute_dfu_sigma(
    residuals: pd.DataFrame,
    min_residual_months: int,
) -> pd.DataFrame:
```

**Purpose:** Compute per-DFU RMSE from the residuals DataFrame produced by `load_champion_residuals`.

**Inputs:**
- `residuals` — output of `load_champion_residuals`; must have `dmdunit`, `loc`, `residual` columns
- `min_residual_months` — minimum number of residual rows for a DFU to receive a DFU-level sigma

**Computation:**
```
Grouped by (dmdunit, loc):
    n_months  = count(residual)
    sigma_dfu = sqrt( mean( residual^2 ) )

Filter: keep only rows where n_months >= min_residual_months
```

**Output:** DataFrame with columns `[dmdunit, loc, sigma, n_months]`. DFUs below the threshold are absent from this frame — their absence is the signal to fall back to cluster-level.

**Edge cases:**
- If `residuals` is empty, returns an empty DataFrame with the correct schema.
- A DFU with all zero residuals (model was perfect) gets `sigma = 0.0` before the floor is applied. The floor (`sigma_floor`) is applied later in `build_sigma_lookup`, not here.

---

### 8.3 `compute_cluster_sigma`

```python
def compute_cluster_sigma(
    dfu_sigma: pd.DataFrame,
    cluster_map: dict[tuple[str, str], str],
) -> dict[str, float]:
```

**Purpose:** Aggregate per-DFU sigmas into per-cluster weighted means.

**Inputs:**
- `dfu_sigma` — output of `compute_dfu_sigma`; columns `[dmdunit, loc, sigma, n_months]`
- `cluster_map` — dict mapping `(dmdunit, loc)` → `cluster_label` (string, e.g., `"high_volume_steady"`)

**Computation:**
```
For each cluster_label k:
    D_k = {(dmdunit, loc) : cluster_map[(dmdunit, loc)] == k}
          intersected with DFUs present in dfu_sigma

    sigma_cluster_k = sum( n_i * sigma_i  for i in D_k )
                    / sum( n_i             for i in D_k )

    (if D_k is empty: cluster k has no entry in the returned dict)
```

**Output:** `dict[str, float]` mapping cluster label → weighted mean sigma. Clusters with no DFU-level sigma data are absent from the dict — their absence triggers the global fallback in `build_sigma_lookup`.

**Edge cases:**
- `cluster_map` is built from `dim_dfu.cluster_assignment`. If a DFU is not in `dim_dfu` (not yet clustered), it is excluded from cluster pooling.
- If all DFUs in a cluster have fewer than `min_residual_months` observations, that cluster has no entry in the output dict.

---

### 8.4 `build_sigma_lookup`

```python
def build_sigma_lookup(
    conn,
    config: dict,
    cluster_map: dict[tuple[str, str], str],
) -> dict[tuple[str, str], float]:
```

**Purpose:** Main orchestrator — calls the three lower-level functions, applies guard rails, and returns a flat lookup from `(item_no, loc)` to the effective capped sigma.

**Inputs:**
- `conn` — active psycopg3 connection
- `config` — the `confidence_interval` sub-dict from `production_forecast_config.yaml`
- `cluster_map` — `(dmdunit, loc)` → cluster label, built by the caller from `dim_dfu`

**Steps executed:**
1. Call `load_champion_residuals(conn, source_model_ids, residual_lag)` → `residuals`
2. Call `compute_dfu_sigma(residuals, min_residual_months)` → `dfu_sigma`
3. Call `compute_cluster_sigma(dfu_sigma, cluster_map)` → `cluster_sigmas`
4. Compute global sigma: `sigma_global = median(cluster_sigmas.values())` — or `sigma_floor` if no cluster sigmas exist (full cold-start)
5. Compute cap: `sigma_cap = sigma_cap_multiplier * sigma_global`
6. For each `(dmdunit, loc)` in `cluster_map`:
   - Level 1: look up `dfu_sigma` — use if present
   - Level 2: look up `cluster_sigmas[cluster_map[(dmdunit, loc)]]` — use if present
   - Level 3: use `sigma_global`
   - Apply floor: `sigma = max(sigma, sigma_floor)`
   - Apply cap: `sigma = min(sigma, sigma_cap)`
7. Return `{(dmdunit, loc): sigma_effective}`

**Output:** `dict[tuple[str, str], float]` — one entry per DFU in `cluster_map`. Every entry has a valid positive float; no NULLs.

**Edge cases:**
- If CI is disabled (`config["enabled"] == False`), returns an empty dict immediately. The caller checks for empty dict and skips CI column population.
- If `backtest_lag_archive` is entirely empty (no backtest run yet), all DFUs resolve to `sigma_floor` after the global fallback chain.
- Computation is done once per inference run, before the per-DFU forecast loop. The lookup is O(1) per DFU during inference.

---

### 8.5 `compute_ci_bounds`

```python
def compute_ci_bounds(
    point_forecast: float,
    sigma: float,
    horizon: int,
    z_lower: float,
    z_upper: float,
    scaling: str,
) -> tuple[float, float]:
```

**Purpose:** Given a point forecast, sigma, and horizon index, compute the lower and upper CI bounds for a single forecast row.

**Inputs:**
- `point_forecast` — the model's mean prediction for this month (already clamped ≥ 0)
- `sigma` — the effective sigma from `build_sigma_lookup` for this DFU
- `horizon` — integer horizon step (1 = T+1, 2 = T+2, ..., 12 = T+12)
- `z_lower` — Z-score for lower bound
- `z_upper` — Z-score for upper bound
- `scaling` — one of `"sqrt"`, `"linear"`, `"none"`

**Computation:**

```python
if scaling == "sqrt":
    scale = math.sqrt(horizon)
elif scaling == "linear":
    scale = float(horizon)
else:  # "none"
    scale = 1.0

lower = max(0.0, point_forecast - z_lower * sigma * scale)
upper = max(point_forecast, point_forecast + z_upper * sigma * scale)

return round(lower, 2), round(upper, 2)
```

**Output:** `(lower, upper)` — both are floats rounded to 2 decimal places. `lower >= 0`, `upper >= point_forecast`.

**Edge cases:**
- `sigma = 0` — returns `(point_forecast, point_forecast)` (degenerate zero-width band); the floor in `build_sigma_lookup` prevents this in practice.
- `point_forecast = 0` — lower remains `0.0`, upper = `z_upper * sigma * scale`.
- `horizon = 1` with `sqrt` scaling — `scale = 1.0`, so T+1 CI is identical to the `none` mode (no scaling at the shortest horizon).

---

## 9. Integration into `generate_production_forecasts.py`

### 9.1 Changes to the Inference Pipeline

The existing script runs a per-DFU loop that calls `generate_forecast_recursive` and then `write_forecast`. The CI integration adds two steps before the loop and one step inside it:

**Before the DFU loop:**
```
1. Load cluster_map from dim_dfu (already done for champion routing)
2. sigma_lookup = build_sigma_lookup(conn, ci_config, cluster_map)
   (only if ci_config["enabled"] == True)
```

**Inside the per-DFU, per-horizon loop:**
```
For each forecast row at horizon h:
    sigma = sigma_lookup.get((item_no, loc), ci_config["sigma_floor"])
    lower, upper = compute_ci_bounds(
        point_forecast = row["forecast_qty"],
        sigma          = sigma,
        horizon        = h,
        z_lower        = ci_config["z_lower"],
        z_upper        = ci_config["z_upper"],
        scaling        = ci_config["horizon_scaling"],
    )
    row["forecast_qty_lower"] = lower
    row["forecast_qty_upper"] = upper
```

When CI is disabled (`enabled: false`), `sigma_lookup` is an empty dict, the inner block is skipped, and both CI columns remain `None` → written as NULL to Postgres.

### 9.2 No Changes to `write_forecast`

The `write_forecast` function already includes `forecast_qty_lower` and `forecast_qty_upper` in its INSERT statement (from the F1.1 DDL). The values are simply non-NULL now. No changes to the DB write path are required.

---

## 10. API Changes

### 10.1 `GET /forecast/production` — No Change

The per-DFU detail endpoint already returns `forecast_qty_lower` and `forecast_qty_upper` in each forecast row. With this feature implemented, those fields are populated rather than null. No code change is required — the response schema is already correct.

**Before this feature:**
```json
{ "forecast_qty": 490.0, "forecast_qty_lower": null, "forecast_qty_upper": null }
```

**After this feature:**
```json
{ "forecast_qty": 490.0, "forecast_qty_lower": 328.4, "forecast_qty_upper": 651.6 }
```

### 10.2 `GET /forecast/production/summary` — Extended Response

**File:** `api/routers/production_forecast.py`

Add two fields to the summary response body:

| Field | Type | Description |
|---|---|---|
| `ci_coverage_pct` | float or null | Percentage of forecast rows in this plan version where `forecast_qty_lower IS NOT NULL` |
| `avg_ci_width` | float or null | Average of `(forecast_qty_upper - forecast_qty_lower)` across all rows with non-null CI bands |

**Updated response schema:**
```json
{
  "plan_version": "2026-03",
  "horizon_months": 3,
  "total_dfu_count": 8412,
  "total_forecast_qty": 4182930.0,
  "ci_coverage_pct": 94.2,
  "avg_ci_width": 312.6,
  "by_abc_class": [
    {"abc_class": "A", "dfu_count": 842, "forecast_qty": 3104500.0},
    {"abc_class": "B", "dfu_count": 2514, "forecast_qty": 812200.0}
  ]
}
```

When CI is disabled, `ci_coverage_pct` and `avg_ci_width` are both `null`.

**SQL addition to the summary query:**

```sql
SELECT
    COUNT(*) FILTER (WHERE forecast_qty_lower IS NOT NULL) * 100.0 / COUNT(*) AS ci_coverage_pct,
    AVG(forecast_qty_upper - forecast_qty_lower) FILTER (WHERE forecast_qty_lower IS NOT NULL) AS avg_ci_width,
    -- ... existing aggregations
FROM fact_production_forecast
WHERE plan_version = %s
  AND horizon_months <= %s
  -- ... existing filters
```

---

## 11. Worked Example: End-to-End

**Item:** 100320, **Loc:** 1401-BULK, **Date:** March 2026
**Champion model:** `lgbm_cluster`, **Cluster:** `high_volume_steady`
**Config:** `z_lower = z_upper = 1.282`, `horizon_scaling = sqrt`, `sigma_floor = 1.0`

### Step 1 — Load Residuals

Query `backtest_lag_archive` for `model_id = 'lgbm_cluster'` and `lag = 0`.
DFU 100320 / 1401-BULK has 24 residual rows (2 years of backtest history at lag 0).

| Month | Forecast | Actual | Residual |
|---|---|---|---|
| Jan 2024 | 440 | 390 | +50 |
| Feb 2024 | 410 | 435 | -25 |
| Mar 2024 | 480 | 502 | -22 |
| ... | ... | ... | ... |
| Dec 2025 | 455 | 460 | -5 |

### Step 2 — Compute DFU-Level Sigma

```
n_months  = 24  (>= min_residual_months = 6 → DFU-level eligible)
residuals = [50, -25, -22, ..., -5]
sigma_dfu = sqrt( mean( [2500, 625, 484, ..., 25] ) )
          = sqrt( 1820.4 )
          ≈ 42.7
```

### Step 3 — Apply Guard Rails

```
sigma_cap_multiplier = 3.0
sigma_global (median of all cluster sigmas) ≈ 38.0
sigma_cap    = 3.0 * 38.0 = 114.0

sigma_effective = min( max(42.7, 1.0), 114.0 )
                = 42.7
```

### Step 4 — Compute CI Bounds for Each Horizon

| Horizon | Month | Point Forecast | Scale (√h) | Lower (−1.282×σ×scale) | Upper (+1.282×σ×scale) |
|---|---|---|---|---|---|
| T+1 | Apr 2026 | 490.0 | 1.000 | max(0, 490 − 54.7) = **435.3** | 490 + 54.7 = **544.7** |
| T+2 | May 2026 | 512.0 | 1.414 | max(0, 512 − 77.3) = **434.7** | 512 + 77.3 = **589.3** |
| T+3 | Jun 2026 | 478.0 | 1.732 | max(0, 478 − 94.7) = **383.3** | 478 + 94.7 = **572.7** |
| T+6 | Sep 2026 | 411.0 | 2.449 | max(0, 411 − 133.9) = **277.1** | 411 + 133.9 = **544.9** |
| T+12 | Mar 2027 | 390.0 | 3.464 | max(0, 390 − 189.4) = **200.6** | 390 + 189.4 = **579.4** |

Note how the CI width grows from ±54.7 at T+1 to ±189.4 at T+12 — a 3.46× widening consistent with the `sqrt(12)` scaling factor. This is the key visual signal that helps planners understand that near-term forecasts are actionable and long-range forecasts are directional only.

### Step 5 — Rows Written to `fact_production_forecast`

```sql
INSERT INTO fact_production_forecast
    (plan_version, item_no, loc, forecast_month, forecast_qty,
     forecast_qty_lower, forecast_qty_upper, model_id, horizon_months, ...)
VALUES
    ('2026-03', '100320', '1401-BULK', '2026-04-01', 490.00, 435.30, 544.70, 'lgbm_cluster', 1, ...),
    ('2026-03', '100320', '1401-BULK', '2026-05-01', 512.00, 434.70, 589.30, 'lgbm_cluster', 2, ...),
    ('2026-03', '100320', '1401-BULK', '2026-06-01', 478.00, 383.30, 572.70, 'lgbm_cluster', 3, ...);
    -- ... through T+12
```

---

## 12. Frontend — CI Bands in `DemandForecastPanel`

The `DemandForecastPanel` component (`frontend/src/tabs/inv-planning/DemandForecastPanel.tsx`) already renders `forecast_qty_lower` and `forecast_qty_upper` from the API response into a Recharts `Area` layer. With this feature implemented, those values are non-null and the uncertainty envelope becomes visible.

No component code changes are required. The only frontend impact is visual: the light-blue shaded band between lower and upper bounds appears on the chart when CI data is present. When CI is disabled, the band remains invisible (React does not render the `Area` if `dataKey` values are null).

**Chart rendering with CI enabled:**
```
Qty
650 │                          ╱╲── upper band (P90)
600 │                    ╱╲──╱   ╲
550 │              ╱╲──╱            ╲── forecast line
490 │────────────╱
435 │                ╲── lower band (P10)
    └────────────────────────────────
      Apr  May  Jun  Jul  Aug  Sep
```

The band narrows at T+1 and progressively widens through T+12, giving planners an immediate visual cue about forecast confidence.

---

## 13. Downstream Impact: Replenishment Safety Stock (F2.2)

The primary consumer of CI bands is the safety stock computation in Feature F2.2. The formula:

```
sigma_replenishment = (forecast_qty_upper - forecast_qty_lower) / (2 * z_lower)
```

For the T+1 row of Item 100320:
```
sigma_replenishment = (544.7 - 435.3) / (2 * 1.282)
                    = 109.4 / 2.564
                    ≈ 42.7
```

This recovers the original DFU-level sigma that was used to generate the bands — as expected. The safety stock engine in F2.2 can treat each DFU's CI width as a directly calibrated uncertainty estimate without needing access to `backtest_lag_archive` itself.

This decoupling is intentional: `common/forecast_ci.py` encapsulates all residual logic in the inference pipeline. F2.2 reads only `fact_production_forecast` — a clean boundary between the forecasting platform and the replenishment planning platform.

---

## 14. Dependencies

| Dependency | Status | Notes |
|---|---|---|
| `backtest_lag_archive` | Exists | Populated by `make backtest-load-all`. Must have at least one backtest run. |
| `fact_external_forecast_monthly` (champion rows) | Exists | Used to identify which algorithm is champion per DFU. |
| `dim_dfu.cluster_assignment` | Exists | Used to map DFUs to clusters for fallback sigma. |
| `fact_production_forecast` | Exists (F1.1) | Target table with CI columns already in DDL. |
| `generate_production_forecasts.py` | Exists (F1.1) | Modified by this feature to call `build_sigma_lookup`. |
| F1.1 (Production Forecast Pipeline) | Required | This feature only populates columns within F1.1's table. |
| F2.2 (Replenishment Order Recommendation) | Downstream | Consumes CI bands to derive demand sigma for safety stock. |

---

## 15. Out of Scope

- **Quantile regression models.** Training separate P10/P90 LGBM/CatBoost/XGBoost models per DFU is not part of this feature. Residual-based empirical CIs are used throughout.
- **Asymmetric CI bands.** `z_lower` and `z_upper` can be set to different values in config, but the current formula applies the same sigma to both sides. Asymmetric residual distributions (e.g., skewed demand) are not modelled.
- **Lag-specific sigma by horizon.** Using lag-0 residuals for T+1 and lag-5 residuals for T+6 (horizon-matched calibration) is architecturally possible but not implemented in this version. All horizons use `residual_lag` from config.
- **CI bands for external ERP forecast.** Only `fact_production_forecast` rows are affected. External forecast rows in `fact_external_forecast_monthly` have no CI columns.
- **UI controls for CI width.** No UI slider to adjust confidence level at runtime. Percentile targets are set in config only.
- **Forecast override CI recalculation.** When a planner overrides a forecast value, the CI bands are not recalculated — they remain anchored to the model's original point forecast.

---

## 16. Success Criteria

1. **Coverage:** ≥ 90% of forecast rows in `fact_production_forecast` have non-NULL `forecast_qty_lower` and `forecast_qty_upper` after running with `enabled: true`.
2. **Monotone widening:** For any DFU with `horizon_scaling = "sqrt"`, the CI width at T+6 must be at least 2× the width at T+1 (√6 / √1 ≈ 2.45×).
3. **Non-negative lower bound:** `forecast_qty_lower >= 0` for every row, without exception.
4. **Upper bound contract:** `forecast_qty_upper >= forecast_qty` for every row.
5. **Backward compatibility:** Setting `enabled: false` leaves all CI columns NULL; no existing downstream queries break.
6. **Guard rail enforcement:** No DFU's sigma exceeds `sigma_cap_multiplier × sigma_global`, verified via the summary endpoint's `avg_ci_width`.
7. **Cluster fallback activation:** DFUs with fewer than `min_residual_months` backtest observations receive a cluster-level sigma (non-null band), not a null CI.

---

## 17. Test Requirements

### Backend Unit Tests (`tests/unit/test_forecast_ci.py`)

Each test runs against an in-memory DataFrame or mock connection — no live DB required.

| Test | What Is Verified |
|---|---|
| `test_load_champion_residuals_empty_archive` | Returns empty DataFrame when no backtest rows exist |
| `test_load_champion_residuals_filters_by_lag` | Only rows matching `residual_lag` are returned |
| `test_load_champion_residuals_filters_by_model` | Rows from non-source-model IDs are excluded |
| `test_compute_dfu_sigma_basic` | RMSE formula is correct for a known residual set |
| `test_compute_dfu_sigma_below_threshold` | DFU with n < min_residual_months is absent from output |
| `test_compute_dfu_sigma_all_zero_residuals` | Returns sigma = 0.0 (floor applied downstream) |
| `test_compute_cluster_sigma_weighted` | DFU with more observations receives proportionally more weight |
| `test_compute_cluster_sigma_empty_dfu_sigma` | Returns empty dict when no DFUs qualify |
| `test_build_sigma_lookup_dfu_level` | DFU with sufficient history uses its own sigma |
| `test_build_sigma_lookup_cluster_fallback` | Sparse DFU falls back to cluster sigma |
| `test_build_sigma_lookup_global_fallback` | DFU in cluster with no sigma uses global fallback |
| `test_build_sigma_lookup_floor_applied` | Sigma never falls below sigma_floor |
| `test_build_sigma_lookup_cap_applied` | Outlier DFU sigma is capped at multiplier × global |
| `test_build_sigma_lookup_disabled` | Returns empty dict when `enabled: false` |
| `test_compute_ci_bounds_sqrt_scaling` | T+4 width is 2× T+1 width (√4/√1 = 2) |
| `test_compute_ci_bounds_linear_scaling` | T+4 width is 4× T+1 width |
| `test_compute_ci_bounds_none_scaling` | Width is constant across all horizons |
| `test_compute_ci_bounds_lower_clamp` | Lower bound is 0.0 when point forecast is very small |
| `test_compute_ci_bounds_upper_contract` | Upper bound >= point_forecast when forecast = 0.0 |
| `test_compute_ci_bounds_zero_sigma` | Returns (point_forecast, point_forecast) for sigma = 0 |

### Backend API Tests (`tests/api/test_production_forecast_ci.py`)

| Test | What Is Verified |
|---|---|
| `test_summary_ci_coverage_populated` | `ci_coverage_pct` and `avg_ci_width` are floats when rows exist with CI bands |
| `test_summary_ci_coverage_null_when_disabled` | Both fields are null when all CI columns are null |
| `test_summary_ci_coverage_partial` | `ci_coverage_pct` correctly reflects partial CI population (some rows have CI, some do not) |
