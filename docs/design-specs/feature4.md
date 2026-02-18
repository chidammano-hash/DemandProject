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
