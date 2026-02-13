# Item MVP (Feature0 + Feature1 + Feature1A)

This is an item-only MVP using:
- Iceberg + MinIO
- Spark
- Trino
- MLflow
- Postgres
- FastAPI + Pydantic
- uv

## What this MVP covers
- Standard item dimension (`dim_item`) in Postgres
- Item API (`/items`) from FastAPI
- Spark job to write `silver.dim_item` to Iceberg
- Trino query path for Iceberg table

## Prerequisites
- Docker + Docker Compose
- Python 3.11+
- uv

## Quick Start (Makefile)
```bash
cd mvp/item
make init
make up
make normalize
make load
```

Run API in another terminal:
```bash
cd mvp/item
make api
```

Write Iceberg table and verify:
```bash
cd mvp/item
make spark
make check-all
```

Detailed execution and validation guide:
- `docs/RUNBOOK.md`

## Notes
- `item_ck = item_no` per `feature1a.md`.
- `itemdata.csv` has duplicate `size` header in source; `normalize_item_csv.py` handles header normalization.
- Default Spark image is `spark:3.5.7-java17`; override with `SPARK_IMAGE=spark:<valid-tag>` in `mvp/item/.env` if needed.
