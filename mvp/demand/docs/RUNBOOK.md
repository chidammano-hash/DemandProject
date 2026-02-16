# Demand Unified Runbook

## 1) Initialize
```bash
cd mvp/demand
make init
```

## 2) Start infra
```bash
make up
```

## 3) Ingest datasets
```bash
make normalize-all
make load-all
```

Sales fact only:
```bash
make normalize-sales
make load-sales
```

Forecast fact only:
```bash
make normalize-forecast
make load-forecast
```

## 3b) Setup chatbot (requires OPENAI_API_KEY in .env)
```bash
make generate-embeddings
```
This parses all domain specs, generates OpenAI embeddings for schema metadata, and stores them in the `chat_embeddings` pgvector table. Re-run after adding new datasets or changing schema.

## 3c) Run DFU clustering (optional, for LGBM model support)
```bash
make cluster-all
```

Or run steps individually:
```bash
make cluster-features  # Generate time series + attribute features
make cluster-train     # Train KMeans model with optimal K selection
make cluster-label     # Assign business labels to clusters
make cluster-update    # Update dim_dfu.cluster_assignment in database
```

This groups DFUs by historical demand patterns for improved global LGBM model performance. Results are logged to MLflow experiment `dfu_clustering`. Cluster assignments can be filtered via `/domains/dfu/page` using the `cluster_assignment` filter, or viewed via `/domains/dfu/clusters` endpoint.

## 4) Start API + UI
```bash
make api
```

In another terminal:
```bash
make ui-init
make ui
```

Open:
- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

Notes:
- `make ui-init` requires internet access for npm package download.
- UI analytics is enabled only for `sales` and `forecast` (dimensions are table-only).
- `sales` and `forecast` analytics include Item (`dmdunit`) and Location (`loc`) filters.
- Use item/location filters (exact match) to focus charts and KPIs on one item-location pair.
- Item/Location filters show autocomplete suggestions as you type.
- Trend chart supports multiple measures via `Trend Measures` checkboxes.
- Forecast domain has a **Model selector** dropdown to filter by `model_id` (e.g., `external`, `arima`).
- **Chat panel** (below analytics grid) lets you ask natural language questions. Requires `OPENAI_API_KEY`.

## 5) Validate
```bash
make check-api
make check-db
```

Optional Iceberg path:
```bash
make spark-item
make spark-location
make spark-customer
make spark-time
make spark-dfu
make spark-sales
make spark-forecast
make trino-check-item
make trino-check-location
make trino-check-customer
make trino-check-time
make trino-check-dfu
make trino-check-sales
make trino-check-forecast
```

## 6) Stop
```bash
make down
```

## Troubleshooting
- `make up` fails on bucket creation:
  - rerun `make minio-bucket`
- API returns DB errors:
  - verify `.env` DB values and `make up` status
- Spark fails:
  - run `make normalize-all` first
  - inspect `demand-mvp-spark` and `demand-mvp-iceberg-rest` logs
- pgvector extension not found:
  - ensure `docker-compose.yml` uses `pgvector/pgvector:pg16` (not `postgres:16`)
  - run `make down && docker volume rm demand_pg_data && make up` to rebuild
- Chat endpoint errors:
  - verify `OPENAI_API_KEY` is set in `.env`
  - verify embeddings exist: `make generate-embeddings`
  - check API logs for OpenAI rate limit or connection errors
- Forecast load fails with "missing data for column model_id":
  - re-normalize forecast: `make normalize-forecast && make load-forecast`
- Clustering fails:
  - Ensure sales data is loaded: `make load-sales`
  - Check minimum history requirement (default: 12 months)
  - Verify MLflow is running: `docker ps | grep mlflow`
  - Check feature matrix output: `ls -lh data/clustering_features.csv`
  - Review cluster output: `ls -lh data/clustering/`
- Cluster assignments not updating:
  - Use `--dry-run` flag to preview changes: `make cluster-update` (with dry-run in script)
  - Verify DFU keys match: check `dfu_ck` format in assignments vs database
  - Check PostgreSQL connection: verify `.env` DB values
