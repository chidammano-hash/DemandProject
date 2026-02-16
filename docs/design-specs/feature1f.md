# Feature F: Sales Fact (Monthly, Item-CustomerGroup-Location)

## Purpose
Define monthly shipped and ordered quantities from `dfu_lvl2_hist.txt` for analytics and UI exploration.

## Grain
- one row per `dmdunit` + `dmdgroup` + `loc` + `startdate` + `type`
- `startdate` is monthly grain and must be first day of month (`YYYYMM01`)
- only `TYPE=1` is loaded for MVP

## Table
`fact_sales_monthly`

## Internal Fields
- `sales_sk`
- `sales_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc` + `_` + `startdate` + `_` + `type`)
- `load_ts`
- `modified_ts`

## Required Fields
- `dmdunit`
- `dmdgroup` (currently expected mostly `ALL`)
- `loc`
- `startdate`
- `type`

## Measures
- `qty_shipped` (cases shipped)
- `qty_ordered` (cases ordered / demand)
- `qty` (as provided in source)

## Source Mapping (MVP)
Source file: `datafiles/dfu_lvl2_hist.txt` (pipe-delimited)

- `DMDUNIT` -> `dmdunit`
- `DMDGROUP` -> `dmdgroup`
- `LOC` -> `loc`
- `STARTDATE` (`YYYYMMDD`) -> `startdate` (`YYYY-MM-DD`)
- `TYPE` -> `type` (load only value `1`)
- `U_QTY_SHIPPED` -> `qty_shipped`
- `U_QTY_ORDERED` -> `qty_ordered`
- `QTY` -> `qty`
- `FILE_DT` (`YYYYMMDD`) -> `file_dt` (`YYYY-MM-DD`)

Ignored in MVP:
- `U_LVL`

## MVP Pipeline (Source to Sink to UI)
1. Normalize source file:
   - `make -C mvp/demand normalize-sales`
   - output: `mvp/demand/data/dfu_lvl2_hist_clean.csv`
   - rules:
     - keep only `TYPE=1`
     - parse `STARTDATE`/`FILE_DT` to ISO date
     - reject rows where `STARTDATE` is not month-start
2. Load to Postgres:
   - `make -C mvp/demand load-sales`
   - sink table: `fact_sales_monthly`
3. Publish to Iceberg:
   - `make -C mvp/demand spark-sales`
   - sink table: `iceberg.silver.fact_sales_monthly`
4. Query and UI:
   - API: `/domains/sales`, `/domains/sales/page`, `/sales`, `/sales/page`
   - UI: `http://127.0.0.1:5173/?domain=sales`

## Technology by Component
- Ingest/normalize: Python (`csv`), `uv`, Make
- OLTP sink: PostgreSQL + `psycopg` bulk copy
- Lakehouse sink: Spark 3.5 + Apache Iceberg + MinIO (S3 API)
- Query engine: Trino (`iceberg.silver.fact_sales_monthly`)
- Serving/UI: FastAPI + React/Vite shared UI
