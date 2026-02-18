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

## 3d) Run LGBM backtesting (optional, requires clustering)

Global model (one LGBM for all DFUs, `ml_cluster` as feature):
```bash
make backtest-lgbm          # Global LGBM backtest (10 timeframes)
make backtest-load          # Load predictions into Postgres
```

Per-cluster model (separate LGBM per cluster):
```bash
make backtest-lgbm-cluster  # Per-cluster LGBM backtest
make backtest-load          # Load predictions into Postgres
```

Or run global + load in one shot:
```bash
make backtest-all           # backtest-lgbm + backtest-load
```

Each backtest run produces two CSV files:
- `data/backtest/backtest_predictions.csv` — execution-lag only (loaded into `fact_external_forecast_monthly`)
- `data/backtest/backtest_predictions_all_lags.csv` — lag 0–4 archive (loaded into `backtest_lag_archive`)

`make backtest-load` has `--replace` built in. It only deletes rows matching the `model_id` in the CSV, so running the per-cluster backtest does **not** affect existing global results in Postgres.

Note: each backtest run overwrites the CSV files on disk. If you need both models loaded, run global first, load, then run per-cluster and load again.

Predictions are stored in `fact_external_forecast_monthly` with `model_id = lgbm_global` or `lgbm_cluster`. All-lag predictions are archived in `backtest_lag_archive` for accuracy reporting at any horizon. Results appear automatically in the forecast model selector UI and accuracy KPIs.

`make backtest-load` also automatically refreshes the accuracy slice views (`agg_accuracy_by_dim`, `agg_accuracy_lag_archive`) after loading — no additional step needed.

Verify archive data:
```bash
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
```

## 3e) Multi-dimensional accuracy comparison (feature16)

After running `make backtest-load`, the accuracy slice views are automatically populated. To view accuracy sliced by DFU attributes:

1. Open the Forecast domain in the UI.
2. Click **Accuracy Comparison** (collapsible card below the main analytics section).
3. Select a **Slice by** dimension (e.g., Cluster, Supplier, ABC Volume, Region).
4. Optionally select a **Lag Filter** (Execution Lag, or specific lag 0–4).
5. Optionally filter **Models** (comma-separated, e.g., `lgbm_global,external`).

The panel shows:
- **Model comparison table**: side-by-side Accuracy %, WAPE, Bias per model for each bucket. Best model highlighted in teal, high-bias cells in red.
- **Lag curve chart**: accuracy degradation from lag 0 → lag 4, one line per model.

To refresh the views manually (e.g., after `load-forecast` without backtest):
```bash
make accuracy-slice-refresh
```

To verify the slice endpoint:
```bash
make accuracy-slice-check
```

API endpoints:
- `GET /forecast/accuracy/slice?group_by=cluster_assignment&models=lgbm_global,external`
- `GET /forecast/accuracy/lag-curve?models=lgbm_global,lgbm_cluster,external`

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
- **MLflow not running (Connection refused on port 5003)**:
  - MLflow is a Docker Compose service and only runs when the stack is up.
  - Start the full stack: `make up` (this starts Postgres, MinIO, MLflow, Iceberg REST, Spark, Trino).
  - Check that the MLflow container is up: `docker ps | grep mlflow` (expect `demand-mvp-mlflow`).
  - MLflow UI: `http://localhost:5003` (or the port in `MLFLOW_HOST_PORT` in `.env`).
  - Clustering still completes if MLflow is down; it skips logging and saves outputs to disk.
- Clustering fails:
  - Ensure sales data is loaded: `make load-sales`
  - Check minimum history requirement (default: 12 months)
  - Verify MLflow is running (optional): `docker ps | grep mlflow`
  - Check feature matrix output: `ls -lh data/clustering_features.csv`
  - Review cluster output: `ls -lh data/clustering/`
- Backtest fails:
  - Ensure clustering has been run first: `make cluster-all`
  - Ensure sales data is loaded: `make load-sales`
  - Install lightgbm: `uv sync`
  - Check output: `ls -lh data/backtest/`
- Cluster assignments not updating:
  - Use `--dry-run` flag to preview changes: `make cluster-update` (with dry-run in script)
  - Verify DFU keys match: check `dfu_ck` format in assignments vs database
  - Check PostgreSQL connection: verify `.env` DB values
