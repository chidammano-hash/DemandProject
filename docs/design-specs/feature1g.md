# Feature G: External Forecast Archive Fact (Monthly)

## Purpose
Store archived external statistical forecasts for lag-based forecast accuracy analysis.

## Source
`datafiles/dfu_stat_fcst.txt` (pipe-delimited)

## Grain
- one row per `dmdunit` + `dmdgroup` + `loc` + `fcstdate` + `startdate`
- `fcstdate` is forecast generation month (month-start)
- `startdate` is forecasted demand month (month-start)
- `lag` is computed as month difference between `startdate` and `fcstdate`
- only lags `0..4` are stored in MVP

## Table
`fact_external_forecast_monthly`

## Internal Fields
- `forecast_sk`
- `forecast_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc` + `_` + `fcstdate` + `_` + `startdate`)
- `load_ts`
- `modified_ts`

## Business Fields
- `dmdunit`
- `dmdgroup`
- `loc`
- `fcstdate`
- `startdate`
- `lag`
- `execution_lag`
- `basefcst_pref` (base statistical forecast)
- `tothist_dmd` (actual sales for the month)

## Source Mapping
- `dmdunit` -> `dmdunit`
- `dmdgroup` -> `dmdgroup`
- `loc` -> `loc`
- `fcstdate` -> `fcstdate`
- `startdate` -> `startdate`
- `lag` -> recomputed from dates in normalize step
- `execution_lag` -> `execution_lag`
- `basefcst_pref` -> `basefcst_pref`
- `tothist_dmd` -> `tothist_dmd`

## MVP Pipeline (Source to Sink to UI)
1. Normalize:
   - `make -C mvp/demand normalize-forecast`
   - output: `mvp/demand/data/dfu_stat_fcst_clean.csv`
   - rules:
     - enforce month-start for `fcstdate` and `startdate`
     - compute `lag = month_diff(startdate, fcstdate)`
     - keep only lag range `0..4`
2. Load to Postgres:
   - `make -C mvp/demand load-forecast`
   - sink table: `fact_external_forecast_monthly`
3. Publish to Iceberg:
   - `make -C mvp/demand spark-forecast`
   - sink table: `iceberg.silver.fact_external_forecast_monthly`
4. Query and UI:
   - API: `/domains/forecast`, `/domains/forecast/page`, `/forecasts`, `/forecasts/page`
   - UI: `http://127.0.0.1:5173/?domain=forecast`

## Technology by Component
- Ingest/normalize: Python (`csv`), `uv`, Make
- Relational sink: PostgreSQL (`psycopg` COPY-based load)
- Lakehouse sink: Spark 3.5 + Apache Iceberg + MinIO
- Query: Trino
- Serving/UI: FastAPI + React/Vite shared UI
