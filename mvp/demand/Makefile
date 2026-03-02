SHELL := /bin/zsh

DC := docker compose
UV := uv run

.PHONY: help init init-pip up down logs minio-bucket db-apply-sql db-apply-chat db-apply-inventory db-apply-inv-backtest generate-embeddings api ui-init ui ui-test normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory normalize-all load-item load-location load-customer load-time load-dfu load-sales load-forecast load-forecast-replace load-forecast-replace-no-archive load-inventory load-all refresh-agg-sales refresh-agg-forecast refresh-agg-inventory refresh-agg refresh-inv-backtest inventory-pipeline spark-item spark-location spark-customer spark-time spark-dfu spark-sales spark-forecast trino-check-item trino-check-location trino-check-customer trino-check-time trino-check-dfu trino-check-sales trino-check-forecast bench-compare check-api check-db check-all cluster-features cluster-train cluster-label cluster-update cluster-all seasonality-schema seasonality-detect seasonality-update seasonality-all backtest-lgbm backtest-lgbm-cluster backtest-lgbm-transfer backtest-catboost backtest-catboost-cluster backtest-catboost-transfer backtest-xgboost backtest-xgboost-cluster backtest-xgboost-transfer backtest-prophet backtest-prophet-cluster backtest-prophet-pooled backtest-patchtst backtest-patchtst-cluster backtest-patchtst-transfer backtest-deepar backtest-deepar-cluster backtest-deepar-transfer backtest-statsforecast backtest-statsforecast-cluster backtest-statsforecast-pooled backtest-neuralprophet backtest-neuralprophet-cluster backtest-neuralprophet-pooled backtest-load backtest-load-all backtest-all backtest-clean backtest-list forecast-clean forecast-clean-list accuracy-slice-refresh accuracy-slice-check champion-select champion-simulate champion-train-meta champion-all tune-lgbm tune-catboost tune-xgboost tune-all backtest-lgbm-cluster-tuned backtest-catboost-cluster-tuned backtest-xgboost-cluster-tuned db-apply-jobs commit test test-unit test-api test-cov test-all

help:
	@echo "Targets:"
	@echo "  init                 - copy env, create uv env, install deps"
	@echo "  up/down/logs         - manage docker services (Postgres, MinIO, MLflow, Iceberg, Spark, Trino)"
	@echo "  db-apply-sql         - apply dataset DDL into running Postgres"
	@echo "  normalize-item       - normalize itemdata.csv"
	@echo "  normalize-location   - normalize locationdata.csv"
	@echo "  normalize-customer   - normalize customerdata.csv"
	@echo "  normalize-time       - generate timedata_clean.csv (2020-2035)"
	@echo "  normalize-dfu        - normalize dfu.txt"
	@echo "  normalize-sales      - normalize dfu_lvl2_hist.txt (TYPE=1)"
	@echo "  normalize-forecast   - normalize dfu_stat_fcst.txt (lags 0-4)"
	@echo "  normalize-inventory  - normalize inventory snapshot CSV"
	@echo "  normalize-all        - normalize all configured datasets"
	@echo "  load-item            - load dim_item into Postgres"
	@echo "  load-location        - load dim_location into Postgres"
	@echo "  load-customer        - load dim_customer into Postgres"
	@echo "  load-time            - load dim_time into Postgres"
	@echo "  load-dfu             - load dim_dfu into Postgres"
	@echo "  load-sales           - load fact_sales_monthly into Postgres"
	@echo "  load-forecast        - load fact_external_forecast_monthly into Postgres"
	@echo "  load-forecast-replace - reload external forecast only (preserves backtest data)"
	@echo "  load-forecast-replace-no-archive - reload external, skip archive (fast)"
	@echo "  load-inventory       - load fact_inventory_snapshot into Postgres"
	@echo "  load-all             - load all configured datasets"
	@echo "  refresh-agg          - refresh monthly aggregate materialized views"
	@echo "  db-apply-inventory   - apply inventory snapshot DDL into Postgres"
	@echo "  db-apply-inv-backtest - apply inventory-forecast bridge MV DDL"
	@echo "  refresh-inv-backtest  - refresh inventory-forecast materialized view"
	@echo "  inventory-pipeline   - normalize + load inventory data"
	@echo "  api                  - run unified FastAPI service on :8000"
	@echo "  ui-init              - install frontend dependencies"
	@echo "  ui                   - run shadcn React UI on :5173"
	@echo "  spark-item           - write iceberg.silver.dim_item"
	@echo "  spark-location       - write iceberg.silver.dim_location"
	@echo "  spark-customer       - write iceberg.silver.dim_customer"
	@echo "  spark-time           - write iceberg.silver.dim_time"
	@echo "  spark-dfu            - write iceberg.silver.dim_dfu"
	@echo "  spark-sales          - write iceberg.silver.fact_sales_monthly"
	@echo "  spark-forecast       - write iceberg.silver.fact_external_forecast_monthly"
	@echo "  bench-compare        - compare Postgres vs Trino query latency"
	@echo "  backtest-lgbm        - run LGBM global backtest (10 timeframes)"
	@echo "  backtest-lgbm-cluster- run LGBM per-cluster backtest"
	@echo "  backtest-lgbm-transfer - run LGBM transfer learning backtest"
	@echo "  backtest-catboost    - run CatBoost global backtest (10 timeframes)"
	@echo "  backtest-catboost-cluster - run CatBoost per-cluster backtest"
	@echo "  backtest-catboost-transfer - run CatBoost transfer learning backtest"
	@echo "  backtest-xgboost     - run XGBoost global backtest (10 timeframes)"
	@echo "  backtest-xgboost-cluster - run XGBoost per-cluster backtest"
	@echo "  backtest-xgboost-transfer - run XGBoost transfer learning backtest"
	@echo "  backtest-prophet     - run Prophet global backtest (per-DFU fits)"
	@echo "  backtest-prophet-cluster - run Prophet per-cluster backtest"
	@echo "  backtest-prophet-pooled - run Prophet pooled cluster backtest"
	@echo "  backtest-patchtst    - run PatchTST global backtest (Apple MPS GPU)"
	@echo "  backtest-patchtst-cluster - run PatchTST per-cluster backtest"
	@echo "  backtest-patchtst-transfer - run PatchTST transfer learning backtest"
	@echo "  backtest-deepar      - run DeepAR global backtest"
	@echo "  backtest-deepar-cluster - run DeepAR per-cluster backtest"
	@echo "  backtest-deepar-transfer - run DeepAR transfer learning backtest"
	@echo "  backtest-statsforecast - run StatsForecast global backtest (AutoARIMA+AutoETS)"
	@echo "  backtest-statsforecast-cluster - run StatsForecast per-cluster backtest"
	@echo "  backtest-statsforecast-pooled - run StatsForecast pooled cluster backtest"
	@echo "  backtest-neuralprophet - run NeuralProphet global backtest (PyTorch GPU)"
	@echo "  backtest-neuralprophet-cluster - run NeuralProphet per-cluster backtest"
	@echo "  backtest-neuralprophet-pooled - run NeuralProphet pooled cluster backtest"
	@echo "  backtest-load        - load one model: make backtest-load MODEL=lgbm_cluster"
	@echo "  backtest-load-all    - load ALL models from data/backtest/*/ (run after multiple backtests)"
	@echo "  backtest-clean       - remove model predictions (MODELS='lgbm_global deepar_global')"
	@echo "  backtest-list        - list model_id row counts in database"
	@echo "  forecast-clean       - delete forecasts by date range (ARGS='--before 2025-04-01 --model external')"
	@echo "  forecast-clean-list  - list forecast row counts by model + month"
	@echo "  backtest-all         - backtest-lgbm + load lgbm_global"
	@echo "  accuracy-slice-refresh - refresh agg_accuracy_by_dim + agg_accuracy_lag_archive"
	@echo "  accuracy-slice-check - curl accuracy slice + lag-curve endpoints"
	@echo "  champion-select      - run champion model selection (best-of-models per DFU)"
	@echo "  champion-simulate    - simulate all strategies, compare accuracy vs ceiling"
	@echo "  champion-train-meta  - train meta-learner classifier for champion selection"
	@echo "  champion-all         - train-meta + simulate + select (full pipeline)"
	@echo "  tune-lgbm            - Bayesian hyperparameter tuning for LGBM (50 trials)"
	@echo "  tune-catboost        - Bayesian hyperparameter tuning for CatBoost (50 trials)"
	@echo "  tune-xgboost         - Bayesian hyperparameter tuning for XGBoost (50 trials)"
	@echo "  tune-all             - Run all three tuning jobs sequentially"
	@echo "  seasonality-schema   - apply DDL for seasonality columns on dim_dfu (one-time)"
	@echo "  seasonality-detect   - run seasonality detection pipeline (detect + profile)"
	@echo "  seasonality-update   - write seasonality profiles to dim_dfu"
	@echo "  seasonality-all      - seasonality-schema + detect + update (full pipeline)"
	@echo "  test                 - run all Python tests"
	@echo "  test-unit            - run Python unit tests only"
	@echo "  test-api             - run Python API tests only"
	@echo "  test-cov             - run Python tests with coverage report"
	@echo "  test-all             - run all tests (Python + frontend)"
	@echo "  check-all            - run DB/API/Trino checks"

init:
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv is not installed."; \
		echo "Install: brew install uv"; \
		echo "Or use fallback: make init-pip"; \
		exit 1; \
	}
	uv venv
	uv sync

init-pip:
	@if [ ! -f .env ]; then cp .env.example .env; fi
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install fastapi uvicorn pydantic psycopg[binary] python-dotenv pytest

up:
	$(DC) up -d
	$(MAKE) minio-bucket
	$(MAKE) db-apply-sql

minio-bucket:
	docker exec demand-mvp-minio sh -lc '\
		until mc alias set local http://127.0.0.1:9000 minio minio123 >/dev/null 2>&1; do \
			sleep 1; \
		done; \
		mc mb --ignore-existing local/warehouse >/dev/null \
	'

db-apply-sql:
	docker exec demand-mvp-postgres sh -lc '\
		until pg_isready -U demand -d demand_mvp >/dev/null 2>&1; do \
			sleep 1; \
		done \
	'
	cat sql/001_create_dim_item.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/002_create_dim_location.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/003_create_dim_customer.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/004_create_dim_time.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/005_create_dim_dfu.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/006_create_fact_sales_monthly.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/007_create_fact_external_forecast_monthly.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/008_perf_indexes_and_agg.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/009_create_chat_embeddings.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/010_create_backtest_lag_archive.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/011_create_accuracy_slice_views.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 -c "ALTER TABLE IF EXISTS dim_customer ALTER COLUMN customer_name DROP NOT NULL;" >/dev/null

down:
	$(DC) down

logs:
	$(DC) logs -f

normalize-item:
	$(UV) python scripts/normalize_dataset_csv.py --dataset item

normalize-location:
	$(UV) python scripts/normalize_dataset_csv.py --dataset location

normalize-customer:
	$(UV) python scripts/normalize_dataset_csv.py --dataset customer

normalize-time:
	$(UV) python scripts/normalize_dataset_csv.py --dataset time

normalize-dfu:
	$(UV) python scripts/normalize_dataset_csv.py --dataset dfu

normalize-sales:
	$(UV) python scripts/normalize_dataset_csv.py --dataset sales

normalize-forecast:
	$(UV) python scripts/normalize_dataset_csv.py --dataset forecast

normalize-inventory:
	$(UV) python scripts/normalize_inventory_csv.py

normalize-all: normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory

load-item:
	$(UV) python scripts/load_dataset_postgres.py --dataset item

load-location:
	$(UV) python scripts/load_dataset_postgres.py --dataset location

load-customer:
	$(UV) python scripts/load_dataset_postgres.py --dataset customer

load-time:
	$(UV) python scripts/load_dataset_postgres.py --dataset time

load-dfu:
	$(UV) python scripts/load_dataset_postgres.py --dataset dfu

load-sales:
	$(UV) python scripts/load_dataset_postgres.py --dataset sales
	$(MAKE) refresh-agg-sales

load-forecast:
	$(UV) python scripts/load_dataset_postgres.py --dataset forecast
	$(MAKE) refresh-agg-forecast

load-forecast-replace:
	$(UV) python scripts/load_dataset_postgres.py --dataset forecast --replace
	$(MAKE) refresh-agg-forecast

load-forecast-replace-no-archive:
	$(UV) python scripts/load_dataset_postgres.py --dataset forecast --replace --skip-archive
	$(MAKE) refresh-agg-forecast

load-inventory:
	$(UV) python scripts/load_dataset_postgres.py --dataset inventory --fast
	$(MAKE) refresh-agg-inventory

load-all:
	$(UV) python scripts/load_dataset_postgres.py --dataset item
	$(UV) python scripts/load_dataset_postgres.py --dataset location
	$(UV) python scripts/load_dataset_postgres.py --dataset customer
	$(UV) python scripts/load_dataset_postgres.py --dataset time
	$(UV) python scripts/load_dataset_postgres.py --dataset dfu
	$(UV) python scripts/load_dataset_postgres.py --dataset sales
	$(UV) python scripts/load_dataset_postgres.py --dataset forecast
	$(UV) python scripts/load_dataset_postgres.py --dataset inventory --fast
	$(MAKE) refresh-agg

refresh-agg-sales:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_sales_monthly;" >/dev/null

refresh-agg-forecast:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_forecast_monthly;" >/dev/null

refresh-agg-inventory:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp \
		-c "REFRESH MATERIALIZED VIEW agg_inventory_monthly;"

refresh-agg: refresh-agg-sales refresh-agg-forecast refresh-agg-inventory

api:
	$(UV) uvicorn api.main:app --reload --port 8000

ui-init:
	cd frontend && npm install

ui:
	cd frontend && [ -x node_modules/.bin/vite ] || npm install
	cd frontend && npm run dev -- --host --port 5173

ui-test:
	cd frontend && npx vitest run

spark-item:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset item

spark-location:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset location

spark-customer:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset customer

spark-time:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset time

spark-dfu:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset dfu

spark-sales:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset sales

spark-forecast:
	docker exec -i \
		-e HOME=/tmp \
		-e AWS_ACCESS_KEY_ID=minio \
		-e AWS_SECRET_ACCESS_KEY=minio123 \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e SPARK_SUBMIT_OPTS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2 -Divy.cache.dir=/tmp/.ivy2/cache" \
		demand-mvp-spark /opt/spark/bin/spark-submit \
		--conf spark.jars.ivy=/tmp/.ivy2 \
		--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0,org.apache.iceberg:iceberg-aws-bundle:1.6.0,org.apache.hadoop:hadoop-aws:3.3.4 \
		/opt/mvp/scripts/spark_dataset_to_iceberg.py --dataset forecast

trino-check-item:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.dim_item;"

trino-check-location:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.dim_location;"

trino-check-customer:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.dim_customer;"

trino-check-time:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.dim_time;"

trino-check-dfu:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.dim_dfu;"

trino-check-sales:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.fact_sales_monthly;"

trino-check-forecast:
	docker exec -i demand-mvp-trino trino --execute "SELECT count(*) AS cnt FROM iceberg.silver.fact_external_forecast_monthly;"

db-apply-chat:
	cat sql/009_create_chat_embeddings.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null

db-apply-inventory:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp < sql/017_create_fact_inventory_snapshot.sql

db-apply-inv-backtest:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp < sql/019_inventory_forecast_view.sql

db-apply-jobs:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp < sql/020_create_job_history.sql
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp < sql/021_alter_job_history_scheduling.sql

refresh-inv-backtest:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp \
		-c "REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly;"

inventory-pipeline: normalize-inventory load-inventory

generate-embeddings:
	$(UV) python scripts/generate_embeddings.py

bench-compare:
	curl -sS "http://localhost:8000/bench/compare?domain=$(or $(DOMAIN),sales)&runs=$(or $(RUNS),5)&warmup=$(or $(WARMUP),1)&limit=$(or $(LIMIT),200)&points=$(or $(POINTS),24)&trino_catalog=$(or $(TRINO_CATALOG),iceberg)&trino_schema=$(or $(TRINO_SCHEMA),silver)$(if $(ITEM),&item=$(ITEM),)$(if $(LOCATION),&location=$(LOCATION),)$(if $(START_DATE),&start_date=$(START_DATE),)$(if $(END_DATE),&end_date=$(END_DATE),)" | python3 -m json.tool

check-api:
	curl -s http://localhost:8000/health && echo
	curl -s "http://localhost:8000/items?limit=3" && echo
	curl -s "http://localhost:8000/locations?limit=3" && echo
	curl -s "http://localhost:8000/customers?limit=3" && echo
	curl -s "http://localhost:8000/times?limit=3" && echo
	curl -s "http://localhost:8000/dfus?limit=3" && echo
	curl -s "http://localhost:8000/sales?limit=3" && echo
	curl -s "http://localhost:8000/forecasts?limit=3" && echo

check-db:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -c "SELECT 'dim_item' AS table_name, count(*) AS cnt FROM dim_item UNION ALL SELECT 'dim_location' AS table_name, count(*) AS cnt FROM dim_location UNION ALL SELECT 'dim_customer' AS table_name, count(*) AS cnt FROM dim_customer UNION ALL SELECT 'dim_time' AS table_name, count(*) AS cnt FROM dim_time UNION ALL SELECT 'dim_dfu' AS table_name, count(*) AS cnt FROM dim_dfu UNION ALL SELECT 'fact_sales_monthly' AS table_name, count(*) AS cnt FROM fact_sales_monthly UNION ALL SELECT 'fact_external_forecast_monthly' AS table_name, count(*) AS cnt FROM fact_external_forecast_monthly;"

cluster-features:
	$(UV) python scripts/generate_clustering_features.py --time-window 24

cluster-train:
	$(UV) python scripts/train_clustering_model.py --k-range 3 12 --skip-gap

cluster-label:
	$(UV) python scripts/label_clusters.py

cluster-update:
	$(UV) python scripts/update_cluster_assignments.py

cluster-all: cluster-features cluster-train cluster-label cluster-update

# ---------------------------------------------------------------------------
# Seasonality pipeline (feature 30)
# ---------------------------------------------------------------------------
seasonality-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/015_add_seasonality_columns.sql').read()); conn.close(); print('Seasonality DDL applied')"

seasonality-detect:
	$(UV) python scripts/detect_seasonality.py --config config/seasonality_config.yaml

seasonality-update:
	$(UV) python scripts/update_seasonality_profiles.py --config config/seasonality_config.yaml

seasonality-all: seasonality-schema seasonality-detect seasonality-update

backtest-lgbm:
	$(UV) python scripts/run_backtest.py --cluster-strategy global

backtest-lgbm-cluster:
	$(UV) python scripts/run_backtest.py --cluster-strategy per_cluster

backtest-lgbm-transfer:
	$(UV) python scripts/run_backtest.py --cluster-strategy transfer

backtest-catboost:
	$(UV) python scripts/run_backtest_catboost.py --cluster-strategy global

backtest-catboost-cluster:
	$(UV) python scripts/run_backtest_catboost.py --cluster-strategy per_cluster

backtest-catboost-transfer:
	$(UV) python scripts/run_backtest_catboost.py --cluster-strategy transfer

backtest-xgboost:
	$(UV) python scripts/run_backtest_xgboost.py --cluster-strategy global

backtest-xgboost-cluster:
	$(UV) python scripts/run_backtest_xgboost.py --cluster-strategy per_cluster

backtest-xgboost-transfer:
	$(UV) python scripts/run_backtest_xgboost.py --cluster-strategy transfer

backtest-prophet:
	$(UV) python scripts/run_backtest_prophet.py --cluster-strategy global

backtest-prophet-cluster:
	$(UV) python scripts/run_backtest_prophet.py --cluster-strategy per_cluster

backtest-prophet-pooled:
	$(UV) python scripts/run_backtest_prophet.py --cluster-strategy pooled

backtest-patchtst:
	$(UV) python scripts/run_backtest_patchtst.py --cluster-strategy global

backtest-patchtst-cluster:
	$(UV) python scripts/run_backtest_patchtst.py --cluster-strategy per_cluster

backtest-patchtst-transfer:
	$(UV) python scripts/run_backtest_patchtst.py --cluster-strategy transfer

backtest-deepar:
	$(UV) python scripts/run_backtest_deepar.py --cluster-strategy global

backtest-deepar-cluster:
	$(UV) python scripts/run_backtest_deepar.py --cluster-strategy per_cluster

backtest-deepar-transfer:
	$(UV) python scripts/run_backtest_deepar.py --cluster-strategy transfer

backtest-statsforecast:
	$(UV) python scripts/run_backtest_statsforecast.py --cluster-strategy global

backtest-statsforecast-cluster:
	$(UV) python scripts/run_backtest_statsforecast.py --cluster-strategy per_cluster

backtest-statsforecast-pooled:
	$(UV) python scripts/run_backtest_statsforecast.py --cluster-strategy pooled

backtest-neuralprophet:
	$(UV) python scripts/run_backtest_neuralprophet.py --cluster-strategy global

backtest-neuralprophet-cluster:
	$(UV) python scripts/run_backtest_neuralprophet.py --cluster-strategy per_cluster

backtest-neuralprophet-pooled:
	$(UV) python scripts/run_backtest_neuralprophet.py --cluster-strategy pooled

backtest-load:
	$(UV) python scripts/load_backtest_forecasts.py --model $(MODEL) --replace

backtest-load-all:
	$(UV) python scripts/load_backtest_forecasts.py --all --replace

backtest-all: backtest-lgbm
	$(UV) python scripts/load_backtest_forecasts.py --model lgbm_global --replace

backtest-clean:
	$(UV) python scripts/clean_backtest_models.py $(MODELS)

backtest-list:
	$(UV) python scripts/clean_backtest_models.py --list

# ---------------------------------------------------------------------------
# Forecast date-range cleanup
# ---------------------------------------------------------------------------
forecast-clean:
	$(UV) python scripts/clean_forecasts_by_date.py $(ARGS)

forecast-clean-list:
	$(UV) python scripts/clean_forecasts_by_date.py --list

accuracy-slice-refresh:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 \
		-c "REFRESH MATERIALIZED VIEW agg_accuracy_by_dim; REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive; REFRESH MATERIALIZED VIEW agg_dfu_coverage; REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive;"

accuracy-slice-check:
	curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=cluster_assignment" | python3 -m json.tool | head -60
	curl -s "http://localhost:8000/forecast/accuracy/lag-curve" | python3 -m json.tool | head -40

champion-select:
	$(UV) python scripts/run_champion_selection.py --config config/model_competition.yaml

champion-simulate:
	$(UV) python scripts/simulate_champion_strategies.py --config config/model_competition.yaml

champion-train-meta:
	$(UV) python scripts/train_meta_learner.py --config config/model_competition.yaml

champion-all: champion-train-meta champion-simulate champion-select

tune-lgbm:
	$(UV) python scripts/tune_hyperparams.py --model lgbm

tune-catboost:
	$(UV) python scripts/tune_hyperparams.py --model catboost

tune-xgboost:
	$(UV) python scripts/tune_hyperparams.py --model xgboost

tune-all: tune-lgbm tune-catboost tune-xgboost

# Causal inline-tuned backtests (PL-002 fix — per-timeframe tuning, no future leakage)
backtest-lgbm-cluster-tuned:
	$(UV) python scripts/run_backtest.py --cluster-strategy per_cluster --tune-inline $(ARGS)

backtest-catboost-cluster-tuned:
	$(UV) python scripts/run_backtest_catboost.py --cluster-strategy per_cluster --tune-inline $(ARGS)

backtest-xgboost-cluster-tuned:
	$(UV) python scripts/run_backtest_xgboost.py --cluster-strategy per_cluster --tune-inline $(ARGS)

commit:
	@if [ -z "$(MSG)" ]; then echo "Usage: make commit MSG=\"your message\""; exit 1; fi
	cd ../.. && git add -A && (git diff --staged --quiet || git commit -m "$(MSG)"); git push

test:
	$(UV) pytest tests/ -v --tb=short

test-unit:
	$(UV) pytest tests/unit/ -v --tb=short

test-api:
	$(UV) pytest tests/api/ -v --tb=short

test-cov:
	$(UV) pytest tests/ --cov=api --cov=common --cov-report=term-missing

test-all: test ui-test

check-all: check-db check-api trino-check-item trino-check-location trino-check-customer trino-check-time trino-check-dfu trino-check-sales trino-check-forecast
