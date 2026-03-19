SHELL := /bin/zsh

DC := docker compose
UV := uv run

.PHONY: help init init-pip up down logs db-apply-sql db-apply-chat db-apply-inventory db-apply-inv-backtest generate-embeddings api ui-init ui ui-test normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory normalize-all load-item load-location load-customer load-time load-dfu load-sales load-forecast load-forecast-replace load-forecast-replace-no-archive load-inventory load-all refresh-agg-sales refresh-agg-forecast refresh-agg-inventory refresh-agg refresh-inv-backtest inventory-pipeline check-api check-db check-all cluster-features cluster-train cluster-label cluster-update cluster-all seasonality-schema seasonality-detect seasonality-update seasonality-all variability-schema variability-compute variability-all lt-profile-schema lt-profile-compute lt-profile-all eoq-schema eoq-compute eoq-all policy-schema policy-assign policy-all health-schema health-refresh health-all exceptions-schema exceptions-generate exceptions-generate-dry ss-schema ss-compute ss-compute-dry ss-all ai-insights-schema ai-insights-scan ai-insights-scan-dry ai-insights-dfu ai-insights-all storyboard-schema storyboard-generate storyboard-generate-dry storyboard-all forecast-prod-schema forecast-generate forecast-generate-dfu forecast-generate-dry forecast-prod-all replplan-schema replplan-compute replplan-compute-dry replplan-all backtest-lgbm backtest-catboost backtest-xgboost backtest-load backtest-load-all backtest-all backtest-all-parallel backtest-clean backtest-list forecast-clean forecast-clean-list accuracy-slice-refresh accuracy-slice-check champion-select champion-simulate champion-train-meta champion-all tune-lgbm tune-catboost tune-xgboost tune-all db-apply-jobs commit test test-unit test-api test-cov test-all e2e-install e2e e2e-ui e2e-headed e2e-report quantile-schema quantile-train quantile-train-dfu quantile-dry quantile-all consensus-schema consensus-generate consensus-generate-dry consensus-all procurement-schema procurement-export procurement-send-erp procurement-all fva-schema sop-seed sop-all dq-schema dq-populate dq-run dq-all medallion-schema medallion-load-sales medallion-load-sales-fix medallion-load-all medallion-load-all-fix medallion-prune medallion-all

help:
	@echo "Targets:"
	@echo "  init                 - copy env, create uv env, install deps"
	@echo "  up/down/logs         - manage docker services (Postgres, MLflow)"
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
	@echo "  backtest-lgbm        - run LGBM per-cluster backtest (settings from algorithm_config.yaml)"
	@echo "  backtest-catboost    - run CatBoost per-cluster backtest (settings from algorithm_config.yaml)"
	@echo "  backtest-xgboost     - run XGBoost per-cluster backtest (settings from algorithm_config.yaml)"
	@echo "  backtest-all         - run all three backtests sequentially (LGBM → CatBoost → XGBoost)"
	@echo "  backtest-all-parallel- run all three backtests in parallel (logs in data/backtest/logs/)"
	@echo "  backtest-load        - load one model: make backtest-load MODEL=lgbm_cluster"
	@echo "  backtest-load-all    - load ALL models from data/backtest/*/ (run after backtest-all)"
	@echo "  backtest-clean       - remove model predictions (MODELS='lgbm_cluster catboost_cluster')"
	@echo "  backtest-list        - list model_id row counts in database"
	@echo "  forecast-clean       - delete forecasts by date range (ARGS='--before 2025-04-01 --model external')"
	@echo "  forecast-clean-list  - list forecast row counts by model + month"
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
	@echo "  NOTE: recursive, SHAP, and tuning are configured via config/algorithm_config.yaml"
	@echo "  seasonality-schema   - apply DDL for seasonality columns on dim_dfu (one-time)"
	@echo "  seasonality-detect   - run seasonality detection pipeline (detect + profile)"
	@echo "  seasonality-update   - write seasonality profiles to dim_dfu"
	@echo "  seasonality-all      - seasonality-schema + detect + update (full pipeline)"
	@echo "  variability-schema   - apply DDL for variability columns on dim_dfu (one-time)"
	@echo "  variability-compute  - compute demand variability stats per DFU"
	@echo "  variability-all      - variability-schema + variability-compute (full pipeline)"
	@echo "  lt-profile-schema    - create dim_item_lead_time_profile table (one-time)"
	@echo "  lt-profile-compute   - compute LT variability profiles per item-location"
	@echo "  lt-profile-all       - lt-profile-schema + lt-profile-compute (full pipeline)"
	@echo "  policy-schema        - create dim_replenishment_policy + fact_dfu_policy_assignment tables (one-time)"
	@echo "  policy-assign        - upsert policies + auto-assign DFUs from config"
	@echo "  policy-all           - policy-schema + policy-assign (full pipeline)"
	@echo "  health-schema        - create mv_inventory_health_score materialized view (one-time)"
	@echo "  health-refresh       - refresh mv_inventory_health_score with current data"
	@echo "  health-all           - health-schema + health-refresh (full pipeline)"
	@echo "  exceptions-schema    - create fact_replenishment_exceptions table (one-time)"
	@echo "  exceptions-generate  - run exception detection + insert into queue"
	@echo "  exceptions-generate-dry - preview exceptions without inserting"
	@echo "  ss-schema            - create fact_safety_stock_targets table (one-time)"
	@echo "  ss-compute           - compute SS targets + upsert into DB"
	@echo "  ss-compute-dry       - preview SS computation without writing"
	@echo "  ss-all               - ss-schema + ss-compute (full pipeline)"
	@echo "  ai-insights-schema   - create ai_insights + ai_planning_memos tables (one-time)"
	@echo "  ai-insights-scan     - run portfolio scan to generate AI insights"
	@echo "  ai-insights-scan-dry - preview scan without writing to DB"
	@echo "  ai-insights-dfu      - run single-DFU analysis (ITEM=<item> LOC=<loc>)"
	@echo "  ai-insights-all      - ai-insights-schema + ai-insights-scan (full pipeline)"
	@echo "  replplan-schema      - create fact_replenishment_plan table (one-time)"
	@echo "  replplan-compute     - compute forward replenishment plan from production forecast CI bands"
	@echo "  replplan-compute-dry - preview replenishment plan without writing to DB"
	@echo "  replplan-all         - replplan-schema + replplan-compute (full pipeline)"
	@echo "  test                 - run all Python tests"
	@echo "  test-unit            - run Python unit tests only"
	@echo "  test-api             - run Python API tests only"
	@echo "  test-cov             - run Python tests with coverage report"
	@echo "  test-all             - run all tests (Python + frontend)"
	@echo "  e2e-install          - install Playwright browsers (one-time)"
	@echo "  e2e                  - run Playwright E2E smoke tests"
	@echo "  e2e-ui               - run Playwright in interactive UI mode"
	@echo "  e2e-headed           - run E2E tests with visible browser"
	@echo "  e2e-report           - open last HTML test report"
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
	$(MAKE) db-apply-sql


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

# ---------------------------------------------------------------------------
# E2E Testing (Playwright) & Visual Regression (Percy)
# ---------------------------------------------------------------------------
e2e-install:
	cd frontend && npx playwright install chromium

e2e:
	cd frontend && npx playwright test --config=e2e/playwright.config.ts

e2e-ui:
	cd frontend && npx playwright test --config=e2e/playwright.config.ts --ui

e2e-headed:
	cd frontend && npx playwright test --config=e2e/playwright.config.ts --headed

e2e-report:
	cd frontend && npx playwright show-report e2e/playwright-report


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
	$(UV) python scripts/generate_clustering_features.py

cluster-train:
	$(UV) python scripts/train_clustering_model.py

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

# ---------------------------------------------------------------------------
# Demand Variability pipeline (IPfeature1)
# ---------------------------------------------------------------------------
variability-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/022_add_demand_variability_columns.sql').read()); conn.close(); print('Variability DDL applied')"

variability-compute:
	$(UV) python scripts/compute_demand_variability.py --config config/variability_config.yaml

variability-all: variability-schema variability-compute

# ---------------------------------------------------------------------------
# Lead Time Variability pipeline (IPfeature2)
# ---------------------------------------------------------------------------
lt-profile-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/023_create_lead_time_profile.sql').read()); conn.close(); print('Lead time profile DDL applied')"

lt-profile-compute:
	$(UV) python scripts/compute_lead_time_variability.py --config config/lead_time_config.yaml

lt-profile-all: lt-profile-schema lt-profile-compute

# ---------------------------------------------------------------------------
# EOQ & Cycle Stock pipeline (IPfeature4)
# ---------------------------------------------------------------------------
eoq-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/024_create_eoq_targets.sql').read()); conn.close(); print('EOQ targets DDL applied')"

eoq-compute:
	$(UV) python scripts/compute_eoq.py --config config/eoq_config.yaml

eoq-all: eoq-schema eoq-compute

# ---------------------------------------------------------------------------
# Replenishment Policy pipeline (IPfeature5)
# ---------------------------------------------------------------------------
policy-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/025_create_replenishment_policy.sql').read()); conn.close(); print('Replenishment policy DDL applied')"

policy-assign:
	$(UV) python scripts/assign_replenishment_policies.py --config config/replenishment_policy_config.yaml

policy-all: policy-schema policy-assign

health-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/026_create_inventory_health_score.sql').read()); conn.close(); print('Health score DDL applied')"

health-refresh:
	$(UV) python scripts/refresh_health_scores.py

health-all: health-schema health-refresh

# ---------------------------------------------------------------------------
# Exception Queue (IPfeature7)
# ---------------------------------------------------------------------------

exceptions-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/027_create_replenishment_exceptions.sql').read()); conn.close(); print('Exceptions DDL applied')"

exceptions-generate:
	$(UV) python scripts/generate_replenishment_exceptions.py

exceptions-generate-dry:
	$(UV) python scripts/generate_replenishment_exceptions.py --dry-run

# ---------------------------------------------------------------------------
# Safety Stock Engine (IPfeature3)
# ---------------------------------------------------------------------------

ss-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/037_create_safety_stock_targets.sql').read()); conn.close(); print('Safety Stock DDL applied')"

ss-compute:
	$(UV) python scripts/compute_safety_stock.py

ss-compute-dry:
	$(UV) python scripts/compute_safety_stock.py --dry-run

ss-all: ss-schema ss-compute

# ---------------------------------------------------------------------------
# AI Planning Agent (IPAIfeature1)
# ---------------------------------------------------------------------------

ai-insights-schema:
	$(UV) psql $(DATABASE_URL) -f sql/036_create_ai_insights.sql
	$(UV) psql $(DATABASE_URL) -f sql/039_create_ai_call_log.sql
	$(UV) psql $(DATABASE_URL) -f sql/040_create_ai_recommendation_outcomes.sql

ai-insights-scan:
	$(UV) python scripts/generate_ai_insights.py --portfolio

ai-insights-scan-dry:
	$(UV) python scripts/generate_ai_insights.py --portfolio --dry-run

ai-insights-dfu:
	$(UV) python scripts/generate_ai_insights.py --item $(ITEM) --loc $(LOC)

ai-insights-all: ai-insights-schema ai-insights-scan

# ---------------------------------------------------------------------------
# Backtesting (LGBM / CatBoost / XGBoost — per-cluster only)
# Options (recursive, SHAP, tuning, params) are set in config/algorithm_config.yaml
# ---------------------------------------------------------------------------
backtest-lgbm:
	$(UV) python scripts/run_backtest.py $(ARGS)

backtest-catboost:
	$(UV) python scripts/run_backtest_catboost.py $(ARGS)

backtest-xgboost:
	$(UV) python scripts/run_backtest_xgboost.py $(ARGS)

backtest-all: backtest-lgbm backtest-catboost backtest-xgboost

backtest-all-parallel:
	@mkdir -p data/backtest/logs
	@echo "[parallel] Starting LGBM, CatBoost, XGBoost concurrently — logs in data/backtest/logs/"
	$(UV) python scripts/run_backtest.py $(ARGS) > data/backtest/logs/lgbm.log 2>&1 & \
	$(UV) python scripts/run_backtest_catboost.py $(ARGS) > data/backtest/logs/catboost.log 2>&1 & \
	$(UV) python scripts/run_backtest_xgboost.py $(ARGS) > data/backtest/logs/xgboost.log 2>&1 & \
	wait && echo "[parallel] All three backtests complete. Check data/backtest/logs/ for output."

backtest-load:
	$(UV) python scripts/load_backtest_forecasts.py --model $(MODEL) --replace

backtest-load-all:
	$(UV) python scripts/load_backtest_forecasts.py --all --replace

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

check-all: check-db check-api

# ---------------------------------------------------------------------------
# IPfeature8: Fill Rate Analytics
# ---------------------------------------------------------------------------
fill-rate-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/028_create_fill_rate_monthly.sql').read()); conn.close(); print('Fill rate DDL applied')"

fill-rate-refresh:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; populated=conn.execute("SELECT relispopulated FROM pg_class WHERE relname='mv_fill_rate_monthly'").fetchone(); conn.execute('REFRESH MATERIALIZED VIEW ' + ('CONCURRENTLY ' if populated and populated[0] else '') + 'mv_fill_rate_monthly'); conn.close(); print('mv_fill_rate_monthly refreshed')"

fill-rate-all: fill-rate-schema fill-rate-refresh

# ---------------------------------------------------------------------------
# IPfeature9: Demand Sensing
# ---------------------------------------------------------------------------
demand-signals-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/029_create_demand_signals.sql').read()); conn.close(); print('Demand signals DDL applied')"

demand-signals-compute:
	$(UV) python scripts/compute_demand_signals.py

demand-signals-dry:
	$(UV) python scripts/compute_demand_signals.py --dry-run

demand-signals-all: demand-signals-schema demand-signals-compute

# ---------------------------------------------------------------------------
# IPfeature10: Monte Carlo Simulation
# ---------------------------------------------------------------------------
sim-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/030_create_ss_simulation_results.sql').read()); conn.close(); print('Simulation DDL applied')"

sim-run:
	$(UV) python scripts/run_ss_simulation.py --item $(ITEM) --loc $(LOC)

# ---------------------------------------------------------------------------
# IPfeature11: ABC-XYZ Classification
# ---------------------------------------------------------------------------
abc-xyz-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/031_add_xyz_classification.sql').read()); conn.close(); print('ABC-XYZ DDL applied')"

abc-xyz-classify:
	$(UV) python scripts/classify_abc_xyz.py

abc-xyz-classify-dry:
	$(UV) python scripts/classify_abc_xyz.py --dry-run

abc-xyz-all: abc-xyz-schema abc-xyz-classify

# ---------------------------------------------------------------------------
# IPfeature12: Supplier Performance
# ---------------------------------------------------------------------------
supplier-perf-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/032_create_supplier_performance.sql').read()); conn.close(); print('Supplier performance DDL applied')"

supplier-perf-refresh:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; populated=conn.execute("SELECT relispopulated FROM pg_class WHERE relname='mv_supplier_performance'").fetchone(); conn.execute('REFRESH MATERIALIZED VIEW ' + ('CONCURRENTLY ' if populated and populated[0] else '') + 'mv_supplier_performance'); conn.close(); print('mv_supplier_performance refreshed')"

supplier-perf-all: supplier-perf-schema supplier-perf-refresh

# ---------------------------------------------------------------------------
# IPfeature13: Investment Optimization
# ---------------------------------------------------------------------------
investment-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/033_create_investment_plan.sql').read()); conn.close(); print('Investment plan DDL applied')"

investment-plan:
	$(UV) python scripts/compute_investment_plan.py

investment-all: investment-schema investment-plan

# ---------------------------------------------------------------------------
# IPfeature14: Intra-Month Stockout Detection
# ---------------------------------------------------------------------------
intramonth-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/034_create_intramonth_stockout.sql').read()); conn.close(); print('Intramonth stockout DDL applied')"

intramonth-refresh:
	$(UV) python scripts/refresh_intramonth_stockout.py

intramonth-all: intramonth-schema intramonth-refresh

# ---------------------------------------------------------------------------
# IPfeature15: Control Tower
# ---------------------------------------------------------------------------
control-tower-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/035_create_control_tower_kpis.sql').read()); conn.close(); print('Control tower DDL applied')"

control-tower-refresh:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; populated=conn.execute("SELECT relispopulated FROM pg_class WHERE relname='mv_control_tower_kpis'").fetchone(); conn.execute('REFRESH MATERIALIZED VIEW ' + ('CONCURRENTLY ' if populated and populated[0] else '') + 'mv_control_tower_kpis'); conn.close(); print('mv_control_tower_kpis refreshed')"

control-tower-all: control-tower-schema control-tower-refresh

# ---------------------------------------------------------------------------
# Feature 40: Planner Storyboard (Exception-Based Value Workflow)
# ---------------------------------------------------------------------------
storyboard-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/038_create_storyboard.sql').read()); conn.close(); print('Storyboard DDL applied')"

storyboard-generate:
	$(UV) python scripts/generate_storyboard_exceptions.py

storyboard-generate-dry:
	$(UV) python scripts/generate_storyboard_exceptions.py --dry-run

storyboard-all: storyboard-schema storyboard-generate

# ---------------------------------------------------------------------------
# F1.1: Production Forecast Generation Pipeline
# ---------------------------------------------------------------------------

forecast-prod-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/039_create_production_forecast.sql').read()); conn.execute(open('sql/041_add_source_model_id.sql').read()); conn.close(); print('Production forecast DDL applied')"

forecast-generate:
	$(UV) python scripts/generate_production_forecasts.py

forecast-generate-dfu:
	$(UV) python scripts/generate_production_forecasts.py --dfu $(ITEM) $(LOC)

forecast-generate-dry:
	$(UV) python scripts/generate_production_forecasts.py --dry-run

forecast-prod-all: forecast-prod-schema forecast-generate

# ---------------------------------------------------------------------------
# Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)
# ---------------------------------------------------------------------------

replplan-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/041_create_replenishment_plan.sql').read()); conn.close(); print('Replenishment plan schema applied')"

replplan-compute:
	$(UV) python scripts/compute_replenishment_plan.py

replplan-compute-dry:
	$(UV) python scripts/compute_replenishment_plan.py --dry-run

replplan-all: replplan-schema replplan-compute

# ---------------------------------------------------------------------------
# Open PO Integration (F1.3)
# ---------------------------------------------------------------------------

po-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; [conn.execute(open(f).read()) for f in ['sql/042_create_supplier_master.sql','sql/043_create_open_purchase_orders.sql','sql/044_create_po_receipts.sql']]; conn.close(); print('PO schema applied')"

po-load:
	$(UV) python scripts/load_open_pos.py

po-load-file:
	$(UV) python scripts/load_open_pos.py --file $(FILE)

po-load-dry:
	$(UV) python scripts/load_open_pos.py --dry-run

po-receipts-load:
	$(UV) python scripts/load_open_pos.py --receipts

po-all: po-schema po-load

# ---------------------------------------------------------------------------
# Forward Inventory Projection (F1.2)
# ---------------------------------------------------------------------------

projection-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/045_create_inventory_projection.sql').read()); conn.close(); print('Projection schema applied')"

projection-compute:
	$(UV) python scripts/compute_inventory_projection.py --horizon 90

projection-compute-dfu:
	$(UV) python scripts/compute_inventory_projection.py --dfu $(ITEM) $(LOC) --horizon 90

projection-dry:
	$(UV) python scripts/compute_inventory_projection.py --dry-run --horizon 90

projection-all: projection-schema projection-compute

# ---------------------------------------------------------------------------
# Order Recommendation Engine (F2.1)
# ---------------------------------------------------------------------------

planned-orders-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/046_create_planned_orders.sql').read()); conn.close(); print('Planned orders schema applied')"

planned-orders-generate:
	$(UV) python scripts/generate_planned_orders.py

planned-orders-generate-dfu:
	$(UV) python scripts/generate_planned_orders.py --dfu $(ITEM) $(LOC)

planned-orders-dry:
	$(UV) python scripts/generate_planned_orders.py --dry-run

planned-orders-all: planned-orders-schema planned-orders-generate

# ---------------------------------------------------------------------------
# F2.2 — Multi-Horizon Quantile Demand Plan
# ---------------------------------------------------------------------------
VERSION ?= $(shell date +%Y-%m-%d)_production

quantile-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/047_create_demand_plan.sql').read()); conn.commit(); conn.close(); print('quantile schema applied')"

quantile-train:
	$(UV) scripts/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION)

quantile-train-dfu:
	$(UV) scripts/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION) --dfu $(ITEM) $(LOC)

quantile-dry:
	$(UV) scripts/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION) --dry-run

quantile-all: quantile-schema quantile-train

## F2.3 — Consensus Forecasting & Planner Overrides
consensus-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/048_create_consensus_plan.sql').read()); conn.commit(); conn.close(); print('consensus schema applied')"

consensus-generate:
	$(UV) scripts/generate_consensus_plan.py --plan-version $(VERSION) --months-ahead 12

consensus-generate-dry:
	$(UV) scripts/generate_consensus_plan.py --plan-version $(VERSION) --months-ahead 12 --dry-run

consensus-all: consensus-schema consensus-generate

## F2.4 — Procurement Workflow & Order Release
procurement-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/049_create_procurement_workflow.sql').read()); conn.commit(); conn.close(); print('procurement schema applied')"

procurement-export:
	$(UV) scripts/release_planned_orders.py --action export_csv --po-numbers $(PO_NUMBERS) --output-dir data/po_exports/

procurement-send-erp:
	$(UV) scripts/release_planned_orders.py --action send_erp --po-numbers $(PO_NUMBERS) --integration-id $(INTEGRATION_ID)

procurement-all: procurement-schema


# ---------------------------------------------------------------------------
# F3.1 — Forecast Bias Correction Engine
# ---------------------------------------------------------------------------
bias-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/050_create_bias_corrections.sql').read()); conn.commit(); conn.close(); print('bias corrections schema applied')"

bias-compute:
	$(UV) python scripts/compute_bias_corrections.py --plan-version $(VERSION)

bias-compute-dry:
	$(UV) python scripts/compute_bias_corrections.py --plan-version $(VERSION) --dry-run

bias-all: bias-schema bias-compute

# ---------------------------------------------------------------------------
# F3.2 — Service Level Actuals vs Targets
# ---------------------------------------------------------------------------
service-level-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/051_create_service_level_tracking.sql').read()); conn.commit(); conn.close(); print('service level schema applied')"

service-level-compute:
	$(UV) python scripts/compute_service_level_actuals.py

service-level-dry:
	$(UV) python scripts/compute_service_level_actuals.py --dry-run

service-level-all: service-level-schema service-level-compute

# ---------------------------------------------------------------------------
# F3.3 — Supplier Lead Time Learning
# ---------------------------------------------------------------------------
lead-time-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/052_create_lead_time_learning.sql').read()); conn.commit(); conn.close(); print('lead time schema applied')"

lead-time-update:
	$(UV) python scripts/update_lead_time_actuals.py

lead-time-dry:
	$(UV) python scripts/update_lead_time_actuals.py --dry-run

lead-time-all: lead-time-schema lead-time-update

# ---------------------------------------------------------------------------
# F3.4 — Demand Sensing / Blended Forecast
# ---------------------------------------------------------------------------
blended-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/053_create_blended_forecast.sql').read()); conn.commit(); conn.close(); print('blended forecast schema applied')"

blended-compute:
	$(UV) python scripts/compute_blended_forecast.py

blended-compute-dfu:
	$(UV) python scripts/compute_blended_forecast.py --item-no $(ITEM) --loc $(LOC)

blended-dry:
	$(UV) python scripts/compute_blended_forecast.py --dry-run

blended-all: blended-schema blended-compute

# ---------------------------------------------------------------------------
# F3.5 — Multi-Echelon Planning
# ---------------------------------------------------------------------------
echelon-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/054_create_echelon_planning.sql').read()); conn.commit(); conn.close(); print('echelon planning schema applied')"

echelon-compute:
	$(UV) python scripts/compute_echelon_targets.py

echelon-compute-item:
	$(UV) python scripts/compute_echelon_targets.py --item-no $(ITEM)

echelon-dry:
	$(UV) python scripts/compute_echelon_targets.py --dry-run

echelon-all: echelon-schema echelon-compute

# ---------------------------------------------------------------------------
# F4.1 — Financial Inventory Plan
# ---------------------------------------------------------------------------
financial-plan-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/055_create_financial_plan.sql').read()); conn.commit(); conn.close(); print('financial plan schema applied')"

financial-plan-compute:
	$(UV) python scripts/compute_financial_plan.py

financial-plan-dry:
	$(UV) python scripts/compute_financial_plan.py --dry-run

financial-plan-all: financial-plan-schema financial-plan-compute

# ---------------------------------------------------------------------------
# F4.2 — Sales & Operations Planning (S&OP)
# ---------------------------------------------------------------------------
sop-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/056_create_sop_module.sql').read()); conn.commit(); conn.close(); print('S&OP schema applied')"

sop-create:
	$(UV) python scripts/run_sop_cycle.py --action create --cycle-month $(CYCLE_MONTH)

sop-advance:
	$(UV) python scripts/run_sop_cycle.py --action advance --cycle-id $(CYCLE_ID)

sop-populate:
	$(UV) python scripts/run_sop_cycle.py --action populate-demand --cycle-id $(CYCLE_ID)

sop-seed:
	$(UV) python -c "from datetime import date; m=date.today().replace(day=1).isoformat(); import subprocess, sys; subprocess.run([sys.executable, 'scripts/run_sop_cycle.py', '--action', 'create', '--cycle-month', m], check=True)"

sop-all: sop-schema sop-seed

# ---------------------------------------------------------------------------
# F4.3 — Promotion & Event Planning
# ---------------------------------------------------------------------------
events-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/057_create_event_planning.sql').read()); conn.commit(); conn.close(); print('event planning schema applied')"

events-apply:
	$(UV) python scripts/apply_event_adjustments.py

events-apply-dry:
	$(UV) python scripts/apply_event_adjustments.py --dry-run

events-all: events-schema events-apply

# ---------------------------------------------------------------------------
# F4.4 — Supply Chain Scenario Planning
# ---------------------------------------------------------------------------
scenarios-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/058_create_supply_scenarios.sql').read()); conn.commit(); conn.close(); print('supply scenarios schema applied')"

scenarios-list:
	$(UV) python scripts/run_supply_chain_scenario.py --action list

scenarios-run:
	$(UV) python scripts/run_supply_chain_scenario.py --action run --scenario-id $(SCENARIO_ID)

scenarios-run-dry:
	$(UV) python scripts/run_supply_chain_scenario.py --action run --scenario-id $(SCENARIO_ID) --dry-run

scenarios-all: scenarios-schema

# ---------------------------------------------------------------------------
# Inventory Rebalancing — Cross-Location Transfer Optimization
# ---------------------------------------------------------------------------
rebalancing-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); cur=conn.cursor(); cur.execute(open('sql/071_create_transfer_network.sql').read()); cur.execute(open('sql/072_create_rebalancing_plan.sql').read()); cur.execute(open('sql/073_create_rebalancing_views.sql').read()); conn.commit(); conn.close(); print('Rebalancing DDL applied')"

rebalancing-compute:
	$(UV) python scripts/compute_rebalancing.py

rebalancing-compute-dry:
	$(UV) python scripts/compute_rebalancing.py --dry-run

rebalancing-refresh:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.autocommit=True; populated=conn.execute(\"SELECT relispopulated FROM pg_class WHERE relname='mv_network_balance'\").fetchone(); conn.execute('REFRESH MATERIALIZED VIEW ' + ('CONCURRENTLY ' if populated and populated[0] else '') + 'mv_network_balance'); conn.close(); print('mv_network_balance refreshed')"

rebalancing-all: rebalancing-schema rebalancing-compute

# ---------------------------------------------------------------------------
# 08-07 — FVA (Forecast Value Add) Tracking
# ---------------------------------------------------------------------------
fva-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/068_create_fva_tracking.sql').read()); conn.commit(); conn.close(); print('FVA tracking schema applied')"

# ---------------------------------------------------------------------------
# 08-01 — Data Quality
# ---------------------------------------------------------------------------
dq-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/063_create_data_quality.sql').read()); conn.commit(); conn.close(); print('Data quality schema applied')"

dq-populate:
	$(UV) python scripts/populate_dq_checks.py

dq-run:
	$(UV) python -c "from common.dq_engine import DQEngine; e=DQEngine(); results=e.run_all_checks(); print(f'Ran {len(results)} checks')"

dq-all: dq-schema dq-populate dq-run

# ---------------------------------------------------------------------------
# Medallion Pipeline (Bronze → Silver → Gold)
# ---------------------------------------------------------------------------
medallion-schema:
	@echo "Applying medallion DDL (080-086) ..."
	cat sql/080_create_medallion_infrastructure.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/081_create_bronze_tables.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/082_create_silver_tables.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/083_create_silver_quarantine.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/084_create_dq_corrections_audit.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/085_create_row_lineage.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	cat sql/086_create_fact_sales_original.sql | docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 >/dev/null
	@echo "Medallion schema applied (7 DDL files)"

medallion-load-sales:
	$(UV) python scripts/load_dataset_postgres.py --dataset sales --medallion

medallion-load-sales-fix:
	$(UV) python scripts/load_dataset_postgres.py --dataset sales --medallion --apply-fixes

medallion-load-all:
	@for ds in item location customer time dfu sales forecast inventory; do \
		$(UV) python scripts/load_dataset_postgres.py --dataset $$ds --medallion; \
	done

medallion-load-all-fix:
	@for ds in item location customer time dfu sales forecast inventory; do \
		$(UV) python scripts/load_dataset_postgres.py --dataset $$ds --medallion --apply-fixes; \
	done

medallion-prune:
	$(UV) python -c "import psycopg; from common.db import get_db_params; from common.medallion import prune_old_batches; conn=psycopg.connect(**get_db_params()); cur=conn.cursor(); r=prune_old_batches(cur); conn.commit(); conn.close(); print(f'Pruned: bronze={r[\"bronze_deleted\"]}, silver={r[\"silver_deleted\"]}')"

medallion-all: medallion-schema medallion-load-all-fix refresh-agg
