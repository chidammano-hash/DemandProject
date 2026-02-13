# Item MVP Runbook

## 1) Prerequisites
- Docker + Docker Compose
- Python 3.11+
- `uv`

Run from:
```bash
cd mvp/item
```

## 2) Initialize
```bash
make init
```
What it does:
- creates `.env` from `.env.example` (if missing)
- creates virtual env
- installs Python dependencies

## 3) Start Infrastructure
```bash
make up
```
Services started:
- Postgres (`5432`)
- MLflow (`5000`)
- MinIO (`9000`, console `9001`)
- Iceberg REST (`8181`)
- Spark master (`7077`, UI `8080`)
- Trino (`8081`)

## 4) Prepare Item Data
Normalize source CSV:
```bash
make normalize
```
Expected output:
- `mvp/item/data/itemdata_clean.csv` created

Load into Postgres:
```bash
make load
```

## 5) Run API
In a new terminal:
```bash
cd mvp/item
make api
```

## 6) Write Iceberg Table
```bash
make spark
```
Expected result:
- Iceberg table `iceberg.silver.dim_item` created/replaced

## 7) Validate End-to-End
Postgres check:
```bash
make check-db
```

API checks:
```bash
make check-api
```

Trino check:
```bash
make trino-check
```

Run all checks:
```bash
make check-all
```

## 8) Stop
```bash
make down
```

## Troubleshooting
- API fails DB connection:
  - ensure `make up` is running
  - verify `.env` values
- Spark write fails:
  - check `item-mvp-spark` logs
  - rerun after `make normalize`
- Trino check fails:
  - run `make spark` first to create Iceberg table

