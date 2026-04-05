# Production Forecast Generation

> Turns backtest-trained models into real forward-looking forecasts that planners can use for purchasing decisions.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Demand Forecast panel) |
| **Key Files** | `scripts/generate_production_forecasts.py`, `api/routers/production_forecast.py`, `config/forecast_pipeline_config.yaml` (production_forecast section), `sql/039_create_production_forecast.sql`, `frontend/src/tabs/inv-planning/DemandForecastPanel.tsx` |

---

## Problem

Backtest models train on historical data, evaluate accuracy, and then discard the trained weights. The models can tell you they would have forecast 490 units for April -- but they never actually produce a forecast for a future month. Planners see either the ERP's flat forecast or nothing at all for upcoming months. The ML signal exists but is never materialized into an actionable prediction.

## Solution

The production forecast pipeline persists trained model weights during backtesting, loads the champion model for each DFU, and generates recursive forward-looking predictions for the next 24 months using 36 months of lookback history. DFUs with fewer than 12 months of history are routed to a cold-start model (rolling_mean), and DFUs with fewer than 3 months are skipped entirely. Predictions are written to a dedicated `fact_production_forecast` table with version management, confidence interval bands, and full traceability. A scheduled job runs monthly after sales data closes.

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

### `config/forecast_pipeline_config.yaml` (production_forecast section)

> **Note:** Production forecast settings live in `config/forecast_pipeline_config.yaml` under the `production_forecast` section. The legacy `config/production_forecast_config.yaml` has been deleted.

```yaml
production_forecast:
  horizon_months: 24               # Forecast T+1 through T+24 (was 12 in legacy config)
  lookback_months: 36              # Months of sales history loaded (was 24)
  min_history_months: 12           # Below this -> cold-start model routing
  cold_start_model_id: rolling_mean  # Fallback for DFUs with 3-11 months history
  cold_start_min_months: 3         # Absolute floor -- DFUs with < 3 months skipped
  fallback_model_id: lgbm_cluster  # Default for mature DFUs without champion assignment
  recursive: true
  plan_version_format: "%Y-%m"
  keep_last_n_versions: 3
  confidence_interval:
    enabled: true
    source_model_ids: [lgbm_cluster, catboost_cluster, xgboost_cluster]
    residual_lag: 0
    min_residual_months: 6
    z_lower: 1.282                 # P10
    z_upper: 1.282                 # P90
    horizon_scaling: sqrt
    sigma_floor: 1.0
    sigma_cap_multiplier: 3.0
  model_registry:
    base_path: "data/models"
  scheduler:
    job_type: generate_production_forecast
    cron: "0 6 2 * *"
```

### Cold-Start Routing

| History Length | Routing | Rationale |
|---|---|---|
| >= 12 months | Champion model (normal path) | Sufficient history for tree model features |
| 3-11 months | `cold_start_model_id` (rolling_mean) | Too little for tree models, enough for simple average |
| < 3 months | Skipped entirely | Not enough signal to produce meaningful forecast |

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- produces the trained models
- [Champion Selection](./07-champion-selection.md) -- determines which model to use per DFU
- [Forecast CI Bands](./10-forecast-ci-bands.md) -- populates confidence interval columns
- Clustering (in `03-demand-intelligence/`) -- routes DFUs to correct cluster model

## See Also

- [Bias Correction](./09-bias-correction.md) -- consumes production forecasts for projection
- [Forecast CI Bands](./10-forecast-ci-bands.md) -- companion feature for uncertainty quantification
