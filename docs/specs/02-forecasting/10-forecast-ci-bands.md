# Forecast Confidence Interval Bands

> Wraps each point forecast with an uncertainty range so planners know the difference between a confident near-term prediction and a speculative long-range one.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Demand Forecast panel) |
| **Key Files** | `common/ml/forecast_ci.py`, `config/forecasting/forecast_pipeline_config.yaml` (production_forecast.confidence_interval section), `scripts/forecasting/generate_production_forecasts.py`, `frontend/src/tabs/inv-planning/DemandForecastPanel.tsx` |

---

## Problem

A point forecast of "490 units in April" tells planners nothing about uncertainty. Is this a confident prediction with a tight range (450-530) or a wild guess (200-780)? Without confidence intervals, planners cannot distinguish near-term actionable forecasts from long-range directional ones. The safety stock engine also needs a demand sigma to compute buffer quantities -- without CI bands, it falls back to a hard-coded global sigma that is wrong for most items.

## Solution

Residual-based empirical confidence intervals use the model's own backtest history to measure how wrong it has been. RMSE (Root Mean Square Error) from backtest residuals becomes the sigma (standard deviation) for each DFU. A three-level fallback hierarchy ensures every DFU gets a CI band even with limited history. Horizon scaling widens the bands for longer-range forecasts, giving planners an immediate visual cue about forecast confidence.

## How It Works

1. Load backtest residuals from `backtest_lag_archive` for the champion model's algorithm
2. Compute per-DFU RMSE (sigma) from forecast-minus-actual residuals
3. Apply a three-level fallback hierarchy:
   - **Level 1:** DFU-level sigma (if >= 6 residual observations)
   - **Level 2:** Cluster-level sigma (weighted average of DFU sigmas in the same cluster)
   - **Level 3:** Global sigma (median of all cluster sigmas)
4. Apply guard rails: floor (minimum sigma = 1.0) and cap (3x global median)
5. Scale sigma by horizon: `sqrt(h)` by default (uncertainty grows like a random walk)
6. Compute bounds: `lower = max(0, forecast - z * sigma * scale)`, `upper = forecast + z * sigma * scale`
7. Write bounds to `forecast_qty_lower` and `forecast_qty_upper` in `fact_production_forecast`

### Why Residual-Based, Not Quantile Regression

Training separate P10/P90 models would triple compute cost. Backtest residuals already exist for free in `backtest_lag_archive` -- they provide historically honest uncertainty with no additional training runs.

The former standalone synthetic quantile-regression generator was removed. Residual-based bands documented here are the production CI source; foundation-model quantile support and generic downstream quantile math remain available without adding a sixth production forecasting algorithm.

### Horizon Scaling

| Mode | Scale Factor | Description |
|------|-------------|-------------|
| `sqrt` (default) | `sqrt(h)` | Errors accumulate like a random walk |
| `linear` | `h` | Conservative -- uncertainty grows proportionally |
| `none` | 1 | Flat sigma across all horizons |

With `sqrt` scaling, T+1 has a CI width of +/- 55 units while T+12 has +/- 189 units (3.46x wider). This is the key visual signal that near-term forecasts are actionable and long-range ones are directional.

## Data Model

No new tables. The existing `fact_production_forecast` columns `forecast_qty_lower` and `forecast_qty_upper` (previously NULL) are now populated. Setting `enabled: false` in config leaves them NULL -- no downstream queries break.

## API

No new endpoints. The existing `GET /forecast/production` already returns `forecast_qty_lower` and `forecast_qty_upper` in each forecast row. The summary endpoint adds:

| Field | Description |
|-------|-------------|
| `ci_coverage_pct` | Percentage of rows with non-NULL CI bands |
| `avg_ci_width` | Average CI width across all rows |

## Pipeline

No dedicated pipeline. CI computation runs automatically during `make forecast-generate` when `confidence_interval.enabled: true` in the production forecast config.

## Configuration

In `config/forecasting/forecast_pipeline_config.yaml` under `production_forecast`:

```yaml
production_forecast:
  confidence_interval:
    enabled: true
    z_lower: 1.282             # P10 bound (80% CI)
    z_upper: 1.282             # P90 bound (80% CI)
    min_residual_months: 6     # Minimum observations for DFU-level sigma
    horizon_scaling: sqrt      # sqrt | linear | none
    sigma_floor: 1.0           # Minimum sigma (prevents zero-width bands)
    sigma_cap_multiplier: 3.0  # Cap = multiplier x global median sigma
    source_model_ids:
      - lgbm_cluster
      - chronos2_enriched
      - mstl
      - nbeats
      - nhits
    residual_lag: 0             # Which lag from archive to use
```

| Z-Score | Interval | Interpretation |
|---------|----------|---------------|
| 1.282 | P10/P90 (80%) | "8 out of 10 months fall inside this range" |
| 1.645 | P5/P95 (90%) | Wider bands, higher coverage |
| 1.960 | P2.5/P97.5 (95%) | Statistical standard, very wide |

## Frontend

The `DemandForecastPanel` already renders `forecast_qty_lower` and `forecast_qty_upper` as a Recharts `Area` layer. With CI bands populated, the light-blue uncertainty envelope becomes visible. No component code changes needed -- the band appears when data is non-null and is invisible when null.

## Dependencies

- [Production Forecast](./08-production-forecast.md) -- target table and inference pipeline
- [Backtest Framework](./03-backtest-framework.md) -- provides `backtest_lag_archive` residuals
- [Champion Selection](./07-champion-selection.md) -- identifies which algorithm's residuals to use

## See Also

- [Bias Correction](./09-bias-correction.md) -- complementary feature for systematic error detection
- [Production Forecast](./08-production-forecast.md) -- parent feature
