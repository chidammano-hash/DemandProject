# Feature 4: Fact Tables

## Objective
Define the two core fact tables for demand analytics: monthly sales history and external forecast archive.

---

## 4A. Sales Fact (`fact_sales_monthly`)

### Purpose
Monthly shipped and ordered quantities from `dfu_lvl2_hist.txt` for analytics and UI exploration.

### Grain
- One row per `dmdunit` + `dmdgroup` + `loc` + `startdate` + `type`
- `startdate` is monthly grain and must be first day of month (`YYYYMM01`)
- Only `TYPE=1` is loaded for MVP

### Table
`fact_sales_monthly`

### Internal Fields
- `sales_sk`
- `sales_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc` + `_` + `startdate` + `_` + `type`)
- `load_ts`
- `modified_ts`

### Required Fields
- `dmdunit`, `dmdgroup` (mostly `ALL`), `loc`, `startdate`, `type`

### Measures
- `qty_shipped` (cases shipped)
- `qty_ordered` (cases ordered / demand)
- `qty` (as provided in source)

### Source Mapping
Source file: `datafiles/dfu_lvl2_hist.txt` (pipe-delimited)

- `DMDUNIT` -> `dmdunit`, `DMDGROUP` -> `dmdgroup`, `LOC` -> `loc`
- `STARTDATE` (`YYYYMMDD`) -> `startdate` (`YYYY-MM-DD`)
- `TYPE` -> `type` (load only value `1`)
- `U_QTY_SHIPPED` -> `qty_shipped`, `U_QTY_ORDERED` -> `qty_ordered`, `QTY` -> `qty`

### MVP Pipeline
1. Normalize: `make normalize-sales` -> `data/dfu_lvl2_hist_clean.csv` (keep TYPE=1 only, parse dates, reject non-month-start)
2. Load: `make load-sales` -> Postgres `fact_sales_monthly`
3. Publish: `make spark-sales` -> `iceberg.silver.fact_sales_monthly`
4. API: `/domains/sales`, `/domains/sales/page`

### Technology
- Ingest/normalize: Python (`csv`), `uv`, Make
- OLTP sink: PostgreSQL + `psycopg` bulk copy
- Lakehouse sink: Spark 3.5 + Apache Iceberg + MinIO
- Query engine: Trino
- Serving/UI: FastAPI + React/Vite

---

## 4B. External Forecast Fact (`fact_external_forecast_monthly`)

### Purpose
Archived external statistical forecasts for lag-based forecast accuracy analysis.

### Grain
- One row per `dmdunit` + `dmdgroup` + `loc` + `fcstdate` + `startdate` + `model_id`
- `fcstdate` is forecast generation month (month-start)
- `startdate` is forecasted demand month (month-start)
- `lag` is computed as month difference between `startdate` and `fcstdate`
- Only lags `0..4` are stored in MVP
- `model_id` identifies the forecasting algorithm (default `'external'`); see Feature 6

### Table
`fact_external_forecast_monthly`

### Internal Fields
- `forecast_sk`
- `forecast_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc` + `_` + `fcstdate` + `_` + `startdate`)
- `load_ts`
- `modified_ts`

### Business Fields
- `dmdunit`, `dmdgroup`, `loc`
- `fcstdate`, `startdate`
- `lag`, `execution_lag`
- `model_id` (forecasting algorithm identifier; default `'external'`)
- `basefcst_pref` (base statistical forecast)
- `tothist_dmd` (actual sales for the month)

### Constraint
`UNIQUE(forecast_ck, model_id)` — each business key appears once per model.

### Source Mapping
Source file: `datafiles/dfu_stat_fcst.txt` (pipe-delimited)

### MVP Pipeline
1. Normalize: `make normalize-forecast` (enforce month-start, compute lag, keep lag 0-4)
2. Load: `make load-forecast` -> Postgres `fact_external_forecast_monthly`
3. Publish: `make spark-forecast` -> `iceberg.silver.fact_external_forecast_monthly`
4. API: `/domains/forecast`, `/domains/forecast/page`

---

## Materialized Views

### `agg_sales_monthly`
Pre-aggregated sales for O(1) KPI queries.

### `agg_forecast_monthly`
Pre-aggregated forecasts including `model_id` in GROUP BY for per-model analytics.

---

## Shared Conventions
- Surrogate key `_sk`, composite business key `_ck`
- `load_ts` and `modified_ts` audit timestamps
- Null normalization: `''`, `'null'`, `'none'`, `'NA'` treated as NULL during load
- Type casting: integer/float/date fields auto-cast with null coercion
- All domains served via generic API with pagination (offset/limit)

---

## Implementation Details

### fact_sales_monthly
- Additional column: `file_dt DATE`
- CHECK constraints: `type = 1` (enforced at DB level), `startdate = date_trunc('month', startdate)` (month-start grain)
- Indexes: 4 B-tree (`dmdunit`, `loc`, `startdate`, `type`) + 2 composite (`(dmdunit, loc, startdate)`, `startdate`) + 3 GIN trigram (`dmdunit`, `loc`, `dmdgroup`)
- `pg_trgm` extension created for trigram-based substring search

### fact_external_forecast_monthly
- 4 CHECK constraints: lag 0-4, fcst month-start alignment, start month-start alignment, lag-matches-dates
- UNIQUE constraint: `(forecast_ck, model_id)`
- Indexes: 6 B-tree + 2 composite from `008` + 4 GIN trigram + 2 composite from `013` (`(dmdunit, dmdgroup, loc)`, `(model_id, lag)`)

### Additional Fact Tables (not in original spec)
- **backtest_lag_archive** (`sql/010`): All-lags (0-4) backtest predictions. Grain = `forecast_ck + model_id + lag`. Includes `timeframe` column. 4 CHECK constraints, `UNIQUE(forecast_ck, model_id, lag)`, 4 B-tree + 2 composite indexes.
- **fact_inventory_snapshot** (`sql/017`): Grain = `item_no + loc + snapshot_date`. Columns: `inventory_sk`, `inventory_ck`, `item_no`, `loc`, `snapshot_date`, `lead_time_days`, `qty_on_hand`, `qty_on_hand_on_order`, `qty_on_order`, `mtd_sales`. 4 B-tree + 2 GIN trigram indexes.

### Materialized Views (full list — 9 views)
| View | Grain | Key Measures | DDL |
|------|-------|-------------|-----|
| `agg_sales_monthly` | month_start, dmdunit, loc | row_count, qty_shipped, qty_ordered, qty | sql/008 |
| `agg_forecast_monthly` | month_start, dmdunit, loc, model_id | row_count, basefcst_pref, tothist_dmd | sql/008 |
| `agg_inventory_monthly` | month_start, item_no, loc | eom_qty_on_hand, monthly_sales, avg_daily_sls, latest_lead_time_days | sql/017 |
| `agg_accuracy_by_dim` | model_id, lag, month_start + 8 dims | sum_forecast, sum_actual, sum_abs_error | sql/011, sql/016 |
| `agg_accuracy_lag_archive` | model_id, lag, month_start + dims | sum_forecast, sum_actual, sum_abs_error | sql/011, sql/016 |
| `agg_dfu_coverage` | model_id, lag | dfu_count | sql/012 |
| `agg_dfu_coverage_lag_archive` | model_id, lag | dfu_count | sql/012 |
| `mv_top_movers` | dmdunit | current/prior period qty, pct_change | sql/018 |
| `mv_inventory_forecast_monthly` | item_no, loc, month_start, model_id | forecast, actual, error, dos, stockout/excess flags | sql/019 |


---

## Examples

### Example: Sales fact — recent months for item 100320

```sql
SELECT startdate, qty_shipped, qty_ordered
FROM fact_sales_monthly
WHERE dmdunit='100320' AND loc='1401-BULK' AND type=1
ORDER BY startdate DESC LIMIT 4;
-- 2026-01-01 | 788 | 801
-- 2025-12-01 | 910 | 928
-- 2025-11-01 | 842 | 860
-- 2025-10-01 | 875 | 891
```

### Example: Forecast fact — 5 lags for one forecast date

```sql
SELECT fcstdate, startdate, lag, model_id, basefcst_pref, tothist_dmd
FROM fact_external_forecast_monthly
WHERE dmdunit='100320' AND loc='1401-BULK'
  AND fcstdate='2025-01-01' AND model_id='external'
ORDER BY lag;
-- 2025-01-01 | 2025-01-01 | 0 | external | 920 | 921
-- 2025-01-01 | 2025-02-01 | 1 | external | 905 | 895
-- 2025-01-01 | 2025-03-01 | 2 | external | 910 | 875  ← execution-lag row
-- 2025-01-01 | 2025-04-01 | 3 | external | 895 | 842
-- 2025-01-01 | 2025-05-01 | 4 | external | 880 | 788
```

### Example: Load sales and forecast

```bash
make load-sales                       # load fact_sales_monthly
make load-forecast-replace            # reload external forecast, preserve backtest
make load-forecast-replace-no-archive # faster: skip 45M-row archive insert
```
