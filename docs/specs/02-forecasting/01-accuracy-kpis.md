# Forecast Accuracy KPIs

> Measures how close forecasts are to actual demand, so planners know which models and products to trust.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `common/services/metrics.py`, `api/routers/forecasting/accuracy.py`, `frontend/src/tabs/AggregateAnalysisTab.tsx`, `sql/011_create_accuracy_slice_views.sql` |

---

## Problem

Supply chain teams need to know how accurate their forecasts are before making purchasing decisions. Without standardized accuracy metrics, there is no way to compare models, identify problem items, or track improvement over time. Planners end up guessing which forecast to trust.

## Solution

Three core metrics are computed across the platform: WAPE (Weighted Absolute Percentage Error -- measures forecast accuracy weighted by volume), Bias (whether forecasts systematically over- or under-predict), and Accuracy % (the inverse of WAPE). These metrics can be sliced by 11 dimensions (cluster, brand, region, ABC class, model, lag, seasonality profile, and more) and viewed through a rolling window of 1-12 months. Pre-aggregated materialized views make slicing instantaneous.

## How It Works

1. The system computes three metrics from forecast vs. actual pairs: WAPE, Bias, and Accuracy %
2. WAPE = `100 * SUM(|Forecast - Actual|) / |SUM(Actual)|` -- a volume-weighted error measure
3. Bias = `(SUM(Forecast) / SUM(Actual)) - 1` -- positive means over-forecasting, negative means under-forecasting
4. Accuracy % = `100 - WAPE` -- the headline metric planners see first
5. The Accuracy tab lets users select a rolling window (1-12 months) and slice by any of 11 dimensions
6. A lag curve shows how accuracy degrades as the forecast ages (lag 0 is most recent, lag 4 is 4 months old)
7. "Common DFUs Only" mode restricts comparison to items present in ALL selected models, ensuring a fair comparison

## Data Model

### Materialized Views

| View | Source | Grain | Purpose |
|------|--------|-------|---------|
| `agg_accuracy_by_dim` | `fact_external_forecast_monthly` + `dim_sku` | model_id, lag, month, cluster, supplier, abc_vol, region, brand, seasonality_profile | Fast accuracy slicing |
| `agg_accuracy_lag_archive` | `backtest_lag_archive` + `dim_sku` | Same + timeframe | Lag-horizon accuracy curves |
| `agg_dfu_coverage` | `fact_external_forecast_monthly` | model_id, lag | DFU count per model |
| `agg_dfu_coverage_lag_archive` | `backtest_lag_archive` | model_id, lag | DFU count per model (archive) |

All four views include `seasonality_profile` (added in `sql/016`).

### Stored Aggregates

Each view stores `SUM(forecast)`, `SUM(actual)`, `SUM(|forecast - actual|)` so KPIs can be derived at query time without scanning the full fact table.

### dim_sku join grain (required)

`dim_sku.sku_ck = (item_id, customer_group, loc)`, and `customer_group` is **not** unique per `(item_id, loc)`. Any endpoint that joins the forecast fact to `dim_sku` MUST match on all three keys — `item_id AND customer_group AND loc`. Joining on `(item_id, loc)` only fans every fact row out across all customer_groups, inflating `SUM(|F−A|)`, `SUM(A)`, and `COUNT(DISTINCT sku_ck)` and corrupting WAPE / accuracy / bias.

> **Change note (2026-06-20):** `accuracy_budget.py` (all 11 dim_sku joins + both oracle-ceiling `ROW_NUMBER` partitions) and `fva.py` (FVA-waterfall `dfu_filter`) were fixed to join on the full `(item_id, customer_group, loc)` key, matching `accuracy.py`. This shifts the reported numbers on the Accuracy-Budget and FVA panels (they were previously over-counted). The same round also bound the `blended_forecast` summary window to `get_planning_date()` instead of SQL `CURRENT_DATE`.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/accuracy` | KPI cards for a DFU or portfolio (with window, model, lag filters) |
| GET | `/forecast/accuracy/slice` | Accuracy grouped by any of 11 dimensions |
| GET | `/forecast/accuracy/lag-curve` | Accuracy at each lag (0-4) per model -- shows forecast aging |

### Slice Parameters

| Parameter | Description |
|-----------|-------------|
| `group_by` | Dimension to group by (default: `cluster_assignment`). 11 options: `cluster_assignment`, `ml_cluster`, `supplier_desc`, `abc_vol`, `region`, `brand_desc`, `dfu_execution_lag`, `month_start`, `lag`, `model_id`, `seasonality_profile` |
| `models` | Comma-separated model IDs to compare |
| `lag` | -1 = execution lag per DFU, 0-4 for fixed horizon |
| `common_dfus` | When true with 2+ models, only includes DFUs present in ALL models |
| `month_from`, `month_to` | Date range filter |

## Pipeline

| Target | Description |
|--------|-------------|
| `make accuracy-slice-refresh` | Refresh all 4 materialized views |
| `make accuracy-slice-check` | Verify accuracy endpoints return data |

Views are also auto-refreshed by `make backtest-load`.

## Configuration

No dedicated config file. Accuracy formulas are defined in `common/services/metrics.py` and reused across all backtest scripts, champion selection, and API endpoints.

## Dependencies

- [Multi-Model Support](./02-multi-model.md) -- `model_id` column enables per-model accuracy
- [Backtest Framework](./03-backtest-framework.md) -- populates `backtest_lag_archive` for lag curves
- Clustering (in `03-demand-intelligence/`) -- provides `cluster_assignment` for slicing
- [SKU Feature Engineering](../03-demand-intelligence/02-sku-feature-engineering.md) -- provides `seasonality_profile` and `variability_class` for filtering

## Probabilistic & weekly metrics

Beyond point accuracy (WAPE / bias / accuracy%), the platform ships:

- **FM quantile → safety-stock bridge** — `common/ml/fm_quantile_bridge.py` turns foundation-model quantile output into a demand distribution for downstream safety stock.
- **Weekly granularity + rolling 13-week view** — `agg_sales_weekly` (`sql/150`) plus a rolling-13-week analytics endpoint.

## See Also

- [Champion Selection](./07-champion-selection.md) -- uses WAPE to pick the best model per DFU
- [Advanced Backtest](./05-advanced-backtest.md) -- SHAP panel also lives in the Accuracy tab
