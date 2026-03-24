# Production Forecast Generation

> Turns backtest-trained models into real forward-looking forecasts that planners can use for purchasing decisions.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Demand Forecast panel) |
| **Key Files** | `scripts/generate_production_forecasts.py`, `api/routers/production_forecast.py`, `config/production_forecast_config.yaml`, `sql/039_create_production_forecast.sql`, `frontend/src/tabs/inv-planning/DemandForecastPanel.tsx` |

---

## Problem

Backtest models train on historical data, evaluate accuracy, and then discard the trained weights. The models can tell you they would have forecast 490 units for April -- but they never actually produce a forecast for a future month. Planners see either the ERP's flat forecast or nothing at all for upcoming months. The ML signal exists but is never materialized into an actionable prediction.

## Solution

The production forecast pipeline persists trained model weights during backtesting, loads the champion model for each DFU, and generates recursive forward-looking predictions for the next 12 months. Predictions are written to a dedicated `fact_production_forecast` table with version management, confidence interval bands, and full traceability. A scheduled job runs monthly after sales data closes.

## How It Works

1. During backtest, model weights are saved as `.pkl` files to `data/models/<model_id>/cluster_<N>.pkl` and saved to `data/models/<model_id>/` directory
2. The inference script loads the champion assignment for each DFU from the forecast table
3. For each DFU, it loads the appropriate cluster model from the registry
4. It builds a feature matrix for future months using the last known actuals
5. Recursive inference predicts month by month: T+1 uses actual lags, T+2 uses T+1's prediction as `qty_lag_1`, and so on
6. Confidence interval bands are computed from backtest residuals (see [Forecast CI Bands](./10-forecast-ci-bands.md))
7. Results are written to `fact_production_forecast` with a versioned `plan_version` (e.g., `"2026-03"`)
8. Old versions are purged (keep last 3 by default)

## Data Model

### `fact_production_forecast`

| Column | Type | Description |
|--------|------|-------------|
| `plan_version` | VARCHAR(30) | Version label (e.g., "2026-03") |
| `item_id` | VARCHAR(50) | Item identifier |
| `loc` | VARCHAR(50) | Location code |
| `forecast_month` | DATE | Future month being forecast |
| `forecast_qty` | NUMERIC(12,2) | Point forecast |
| `forecast_qty_lower` | NUMERIC(12,2) | Lower CI bound (P10) |
| `forecast_qty_upper` | NUMERIC(12,2) | Upper CI bound (P90) |
| `model_id` | VARCHAR(100) | Algorithm that produced this row |
| `cluster_id` | INTEGER | ml_cluster used for inference |
| `horizon_months` | SMALLINT | 1=T+1, 2=T+2, ... 12=T+12 |
| `is_recursive` | BOOLEAN | Whether recursive inference was used |
| `lag_source` | VARCHAR(20) | "actual" (T+1) or "predicted" (T+2+) |
| `run_id` | UUID | Ties rows to a single inference run |

**Grain:** `(plan_version, item_id, loc, forecast_month)`

Tracks persisted model weights so the inference pipeline can reload them.

| Column | Type | Description |
|--------|------|-------------|
| `model_id` | VARCHAR(100) | Algorithm identifier |
| `cluster_id` | INTEGER | Cluster this model covers |
| `model_path` | TEXT | Path to `.pkl` file |
| `feature_cols` | TEXT[] | Ordered feature column names |
| `is_active` | BOOLEAN | TRUE = use for inference |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/production` | Forecast rows for a specific DFU + plan version |
| GET | `/forecast/production/summary` | Aggregate forecast by ABC class for a plan version |
| GET | `/forecast/production/versions` | List available plan versions with metadata |

## Pipeline

| Target | Description |
|--------|-------------|
| `make forecast-prod-schema` | Create tables (one-time) |
| `make forecast-generate` | Run full inference for all DFUs |
| `make forecast-generate-sku ITEM=100320 LOC=1401-BULK` | Single DFU inference |
| `make forecast-generate-dry` | Preview without writing |
| `make forecast-prod-all` | Schema + generate |

**Scheduler:** Runs as `generate_production_forecast` job type on the 2nd of each month at 06:00 UTC (after sales close).

## Configuration

### `config/production_forecast_config.yaml`

```yaml
inference:
  horizon_months: 12
  recursive: true
  confidence_interval: true
  ci_lower_quantile: 0.10
  ci_upper_quantile: 0.90

model_selection:
  strategy: champion
  fallback_model_id: lgbm_cluster

plan_version:
  format: "%Y-%m"
  keep_last_n_versions: 3

model_registry:
  base_path: "data/models"

scheduler:
  job_type: generate_production_forecast
  cron: "0 6 2 * *"
```

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- produces the trained models
- [Champion Selection](./07-champion-selection.md) -- determines which model to use per DFU
- [Forecast CI Bands](./10-forecast-ci-bands.md) -- populates confidence interval columns
- Clustering (in `03-demand-intelligence/`) -- routes DFUs to correct cluster model

## See Also

- [Bias Correction](./09-bias-correction.md) -- consumes production forecasts for projection
- [Forecast CI Bands](./10-forecast-ci-bands.md) -- companion feature for uncertainty quantification
