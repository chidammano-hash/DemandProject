SHELL := /bin/zsh

DC := docker compose
UV := uv run
POSTGRES_SERVICE := postgres
PG_EXEC := $(DC) exec -T $(POSTGRES_SERVICE)
PSQL := $(PG_EXEC) psql -U demand -d demand_mvp

.PHONY: help init init-pip up down logs db-apply-sql db-apply-chat db-apply-inventory db-apply-inv-backtest generate-embeddings api ui-init ui ui-test normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory normalize-all load-item load-location load-customer load-time load-dfu load-sales load-forecast load-forecast-replace load-forecast-replace-no-archive load-inventory load-all refresh-agg-sales refresh-agg-forecast refresh-agg-inventory refresh-agg refresh-inv-backtest inventory-pipeline check-api check-db check-all ai-sync-check cluster-features cluster-train cluster-label cluster-update cluster-all seasonality-schema seasonality-detect seasonality-update seasonality-all variability-schema variability-compute variability-all lt-profile-schema lt-profile-compute lt-profile-all eoq-schema eoq-compute eoq-all policy-schema policy-assign policy-all health-schema health-refresh health-all exceptions-schema exceptions-generate exceptions-generate-dry ss-schema ss-compute ss-compute-dry ss-all ai-insights-schema ai-insights-scan ai-insights-scan-dry ai-insights-dfu ai-insights-all storyboard-schema storyboard-generate storyboard-generate-dry storyboard-all forecast-prod-schema forecast-generate forecast-generate-dfu forecast-generate-dry forecast-prod-all replplan-schema replplan-compute replplan-compute-dry replplan-all backtest-lgbm backtest-catboost backtest-xgboost backtest-seasonal-naive backtest-rolling-mean backtest-mstl backtest-nhits backtest-nbeats backtest-baselines backtest-load backtest-load-all backtest-load-all-bulk backtest-load-bulk backtest-load-main-only backtest-load-archive-only backtest-all backtest-all-parallel backtest-clean backtest-list forecast-clean forecast-clean-list accuracy-slice-refresh accuracy-slice-check champion-select champion-simulate champion-train-meta champion-all tune-lgbm tune-catboost tune-xgboost tune-all db-apply-jobs commit test test-unit test-api test-cov test-all e2e-install e2e e2e-ui e2e-headed e2e-report quantile-schema quantile-train quantile-train-dfu quantile-dry quantile-all consensus-schema consensus-generate consensus-generate-dry consensus-all procurement-schema procurement-export procurement-send-erp procurement-all fva-schema sop-seed sop-all dq-schema dq-populate dq-run dq-all pipeline-full pipeline-refresh pipeline-inventory pipeline-inventory-refresh setup-data setup-features setup-backtest setup-inv-planning setup-demand-planning setup-ops setup-planning setup-all perf-report perf-script perf-api perf-pipeline lgbm-tuning-list lgbm-tuning-compare lgbm-tuning-backup lgbm-tuning-run lgbm-auto-tune lgbm-auto-tune-promote lgbm-auto-tune-dry-run lgbm-auto-tune-list seed-baselines seed-baselines-tuning seed-baselines-champion seed-baselines-clustering db-truncate-data clean-artifacts refresh-mvs-tiered refresh-accuracy-mvs fresh-load fresh-features fresh-backtest fresh-champion fresh-all dev fresh test-quick lint format type-check health audit-routers new-router expert-panel expert-panel-quick expert-panel-mini adv-expert-panel adv-expert-panel-quick adv-expert-panel-mini load-ext-lgbm load-ext-cat load-ext-xg load-ext-best load-ext-all

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
dev: up api ui
fresh: fresh-all
test-quick: test ui-test
lint:
	$(UV) ruff check api/ common/ scripts/ --fix
format:
	$(UV) ruff format api/ common/ scripts/
type-check:
	$(UV) mypy api/ common/ --ignore-missing-imports
health: check-all

# ---------------------------------------------------------------------------
# Developer tooling
# ---------------------------------------------------------------------------
audit-routers:
	@echo "=== Router Audit ==="
	@echo "Router files in api/routers/:"
	@find api/routers -name '*.py' ! -name '__init__.py' | wc -l
	@echo "include_router() calls in main.py:"
	@grep -c 'app.include_router' api/main.py
	@echo ""
	@echo "=== Vite Proxy Check ==="
	@echo "Checking for API prefixes missing from vite.config.ts..."
	@python3 scripts/tools/audit_routes.py 2>/dev/null || echo "Run: python3 scripts/tools/audit_routes.py"

new-router:
	@python3 scripts/tools/scaffold_router.py --domain $(DOMAIN) --name $(NAME)

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
	@echo "  backtest-chronos     - run Chronos T5 foundation model backtest"
	@echo "  backtest-chronos-full- run Chronos T5 backtest + load predictions"
	@echo "  backtest-bolt        - run Chronos Bolt foundation model backtest"
	@echo "  backtest-bolt-full   - run Chronos Bolt backtest + load predictions"
	@echo "  backtest-chronos2    - run Chronos 2 foundation model backtest"
	@echo "  backtest-chronos2-full - run Chronos 2 backtest + load predictions"
	@echo "  backtest-all         - run all six backtests sequentially"
	@echo "  backtest-all-parallel- run all six backtests in parallel (logs in data/backtest/logs/)"
	@echo "  backtest-load        - load one model: make backtest-load MODEL=lgbm_cluster"
	@echo "  backtest-load-all    - load ALL models from data/backtest/*/ (run after backtest-all)"
	@echo "  backtest-load-all-bulk - load ALL models with single index cycle (~4x faster)"
	@echo "  backtest-load-bulk   - load 4 core models (lgbm, catboost, xgboost, chronos) in bulk"
	@echo "  backtest-load-main-only - load specific models to main table only (MODELS='...')"
	@echo "  backtest-load-archive-only - load specific models to archive only (MODELS='...')"
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
	@echo "  expert-panel         - Expert Panel algorithm selection test (5000 DFUs, 5 TFs, ~30 min)"
	@echo "  expert-panel-quick   - Quick Expert Panel test (1000 DFUs, 3 TFs, ~8 min)"
	@echo "  expert-panel-mini    - Minimal Expert Panel test (200 DFUs, 2 TFs, ~2 min)"
	@echo "  expert-panel-loc     - Expert Panel for all DFUs at one location: make expert-panel-loc LOC=1401-BULK"
	@echo "  adv-expert-panel     - Advanced Expert Panel (execution-lag accuracy, foundation models + DL + stat upgrades)"
	@echo "  adv-expert-panel-quick - Quick Advanced Expert Panel (execution-lag accuracy, 1000 DFUs, 5 TFs)"
	@echo "  adv-expert-panel-mini  - Minimal Advanced Expert Panel (200 DFUs, 2 TFs)"
	@echo "  adv-expert-panel-loc   - Advanced Expert Panel for all DFUs at one location: make adv-expert-panel-loc LOC=1401-BULK"
	@echo "  expsys-backtest        - Expert System Backtest: full population, segment-assigned algo, loads to DB (~4-5h)"
	@echo "  expsys-backtest-dry    - ExpSys accuracy only, no DB load (--skip-load)"
	@echo "  expsys-backtest-replace - ExpSys: delete existing rows then reload"
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
	@echo "  lgbm-auto-tune       - auto-tune LGBM with N strategies (RUNS=3 default, max 10)"
	@echo "  lgbm-auto-tune-promote - auto-tune + promote best params to algorithm_config.yaml"
	@echo "  lgbm-auto-tune-dry-run - preview all strategies without running backtests"
	@echo "  lgbm-auto-tune-list  - list available auto-tune strategies"
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
	@echo "  ai-sync-check        - verify Claude and Codex guidance stay wired to shared repo scripts"
	@echo ""
	@echo "  === Full Pipeline (input CSVs -> ready app) ==="
	@echo "  setup-all            - EVERYTHING: data + ML + planning + ops (~4-6 hours)"
	@echo "  setup-data           - data only: normalize + load all 10 domains (~30 min)"
	@echo "  setup-planning       - data + inventory planning, no ML (~1 hour)"
	@echo "  setup-features       - data + clustering + seasonality + variability"
	@echo "  setup-backtest       - features + 3 backtests + champion selection"
	@echo "  setup-inv-planning   - inventory planning (SS, EOQ, policies, exceptions)"
	@echo "  setup-demand-planning - forecasts + projections + orders + replenishment"
	@echo "  setup-ops            - S&OP + events + financial + storyboard + DQ"
	@echo ""
	@echo "  === Database Cleanup & Fresh Recreate ==="
	@echo "  fresh-all            - FULL RESET: truncate + clean + load + ML + champion + baseline planning"
	@echo "  fresh-champion       - load + features + backtests + champion (no truncate)"
	@echo "  fresh-backtest       - load + features + backtests (no champion)"
	@echo "  fresh-features       - load + clustering + seasonality + variability + LT"
	@echo "  fresh-load           - normalize + load + refresh MVs only"
	@echo "  db-truncate-data     - truncate non-config data/history while preserving configuration masters"
	@echo "  clean-artifacts      - remove stale CSVs, backtest, tuning, clustering, champion files"
	@echo "  refresh-mvs-tiered   - refresh all MVs in dependency order"
	@echo "  refresh-accuracy-mvs - refresh accuracy MVs (after backtest load)"

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
	$(PG_EXEC) sh -lc '\
		until pg_isready -U demand -d demand_mvp >/dev/null 2>&1; do \
			sleep 1; \
		done \
	'
	@for f in $$(ls sql/*.sql | sort); do \
		$(PSQL) -v ON_ERROR_STOP=1 < "$$f" >/dev/null; \
	done
	$(PSQL) -v ON_ERROR_STOP=1 -c "ALTER TABLE IF EXISTS dim_customer ALTER COLUMN customer_name DROP NOT NULL;" >/dev/null
	@echo "Applied $$(ls sql/*.sql | wc -l | tr -d ' ') SQL migration files"

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
	$(UV) python scripts/normalize_dataset_csv.py --dataset sku

normalize-sales:
	$(UV) python scripts/normalize_dataset_csv.py --dataset sales

normalize-forecast:
	$(UV) python scripts/normalize_dataset_csv.py --dataset forecast

normalize-inventory:
	$(UV) python scripts/normalize_inventory_csv.py

normalize-sourcing:
	$(UV) python scripts/normalize_dataset_csv.py --dataset sourcing

normalize-purchase-order:
	$(UV) python scripts/normalize_dataset_csv.py --dataset purchase_order

normalize-all: normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory normalize-sourcing normalize-purchase-order normalize-customer-demand

load-item:
	$(UV) python scripts/load_dataset_postgres.py --dataset item

load-location:
	$(UV) python scripts/load_dataset_postgres.py --dataset location

load-customer:
	$(UV) python scripts/load_dataset_postgres.py --dataset customer

load-time:
	$(UV) python scripts/load_dataset_postgres.py --dataset time

load-dfu:
	$(UV) python scripts/load_dataset_postgres.py --dataset sku

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
	$(UV) python scripts/load_dataset_postgres.py --dataset inventory
	$(MAKE) refresh-agg-inventory

load-sourcing:
	$(UV) python scripts/load_dataset_postgres.py --dataset sourcing

load-purchase-order:
	$(UV) python scripts/load_dataset_postgres.py --dataset purchase_order

normalize-customer-demand:  ## Normalize customer demand CSVs
	$(UV) python scripts/etl/normalize_customer_demand_csv.py

load-customer-demand:  ## Load customer demand (full replace)
	$(UV) python scripts/etl/load_customer_demand_postgres.py --replace

load-customer-demand-month:  ## Load single month: make load-customer-demand-month MONTH=2026-01
	$(UV) python scripts/etl/load_customer_demand_postgres.py --month $(MONTH)

pipeline-customer-demand: normalize-customer-demand load-customer-demand  ## Full customer demand pipeline

load-all:
	$(UV) python scripts/load_dataset_postgres.py --dataset item
	$(UV) python scripts/load_dataset_postgres.py --dataset location
	$(UV) python scripts/load_dataset_postgres.py --dataset customer
	$(UV) python scripts/load_dataset_postgres.py --dataset time
	$(UV) python scripts/load_dataset_postgres.py --dataset sku
	$(UV) python scripts/load_dataset_postgres.py --dataset sales
	$(UV) python scripts/load_dataset_postgres.py --dataset forecast
	$(UV) python scripts/load_dataset_postgres.py --dataset inventory
	$(UV) python scripts/load_dataset_postgres.py --dataset sourcing
	$(UV) python scripts/load_dataset_postgres.py --dataset purchase_order
	$(UV) python scripts/etl/load_customer_demand_postgres.py --replace
	$(MAKE) refresh-agg

refresh-agg-sales:
	$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_sales_monthly;" >/dev/null

refresh-agg-forecast:
	$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_forecast_monthly;" >/dev/null

refresh-agg-inventory:
	$(PSQL) \
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
	$(PSQL) -v ON_ERROR_STOP=1 < sql/009_create_chat_embeddings.sql >/dev/null

db-apply-inventory:
	$(PSQL) < sql/017_create_fact_inventory_snapshot.sql

db-apply-inv-backtest:
	$(PSQL) < sql/019_inventory_forecast_view.sql

db-apply-jobs:
	$(PSQL) < sql/020_create_job_history.sql
	$(PSQL) < sql/021_alter_job_history_scheduling.sql

refresh-inv-backtest:
	$(PSQL) \
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
	$(PSQL) -c "SELECT 'dim_item' AS table_name, count(*) AS cnt FROM dim_item UNION ALL SELECT 'dim_location' AS table_name, count(*) AS cnt FROM dim_location UNION ALL SELECT 'dim_customer' AS table_name, count(*) AS cnt FROM dim_customer UNION ALL SELECT 'dim_time' AS table_name, count(*) AS cnt FROM dim_time UNION ALL SELECT 'dim_dfu' AS table_name, count(*) AS cnt FROM dim_dfu UNION ALL SELECT 'fact_sales_monthly' AS table_name, count(*) AS cnt FROM fact_sales_monthly UNION ALL SELECT 'fact_external_forecast_monthly' AS table_name, count(*) AS cnt FROM fact_external_forecast_monthly;"

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
	$(UV) python scripts/update_seasonality_profiles.py

seasonality-all: seasonality-schema seasonality-detect seasonality-update

# ---------------------------------------------------------------------------
# Demand Variability pipeline (IPfeature1)
# ---------------------------------------------------------------------------
variability-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/022_add_demand_variability_columns.sql').read()); conn.close(); print('Variability DDL applied')"

variability-compute:
	$(UV) python scripts/compute_demand_variability.py

variability-all: variability-schema variability-compute

# ---------------------------------------------------------------------------
# Lead Time Variability pipeline (IPfeature2)
# ---------------------------------------------------------------------------
lt-profile-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/023_create_lead_time_profile.sql').read()); conn.close(); print('Lead time profile DDL applied')"

lt-profile-compute:
	$(UV) python scripts/compute_lead_time_variability.py

lt-profile-all: lt-profile-schema lt-profile-compute

# ---------------------------------------------------------------------------
# EOQ & Cycle Stock pipeline (IPfeature4)
# ---------------------------------------------------------------------------
eoq-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/024_create_eoq_targets.sql').read()); conn.close(); print('EOQ targets DDL applied')"

eoq-compute:
	$(UV) python scripts/compute_eoq.py

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
# Backtesting (LGBM / CatBoost / XGBoost / Chronos — per-cluster only)
# Options (recursive, SHAP, tuning, params) are set in config/algorithm_config.yaml
# ---------------------------------------------------------------------------
backtest-lgbm:
	$(UV) python scripts/run_backtest.py --parallel --workers 8 $(ARGS)

backtest-catboost:
	$(UV) python scripts/run_backtest_catboost.py --parallel --workers 8 $(ARGS)

backtest-xgboost:
	$(UV) python scripts/run_backtest_xgboost.py --parallel --workers 8 $(ARGS)

backtest-chronos:
	$(UV) python -m scripts.run_backtest_chronos

backtest-load-chronos:
	$(UV) python -m scripts.load_backtest_forecasts --model chronos --replace

backtest-chronos-full: backtest-chronos backtest-load-chronos

backtest-bolt:
	$(UV) python -m scripts.run_backtest_chronos_bolt

backtest-load-bolt:
	$(UV) python -m scripts.load_backtest_forecasts --model chronos_bolt --replace

backtest-bolt-full: backtest-bolt backtest-load-bolt

backtest-bolt-hier:
	$(UV) python -m scripts.run_backtest_bolt_hierarchical

backtest-load-bolt-hier:
	$(UV) python -m scripts.load_backtest_forecasts --model bolt_hierarchical --replace

backtest-bolt-hier-full: backtest-bolt-hier backtest-load-bolt-hier

customer-features:
	$(UV) python -m scripts.ml.generate_customer_features_sql

customer-features-python:
	$(UV) python -m scripts.ml.generate_customer_features

backtest-lgbm-cust:
	$(UV) python -m scripts.run_backtest --model lgbm --model-id lgbm_cust_enriched

backtest-catboost-cust:
	$(UV) python -m scripts.run_backtest --model catboost --model-id catboost_cust_enriched

backtest-xgboost-cust:
	$(UV) python -m scripts.run_backtest --model xgboost --model-id xgboost_cust_enriched

backtest-cust-enriched-all: backtest-lgbm-cust backtest-catboost-cust backtest-xgboost-cust

backtest-load-cust-enriched:
	$(UV) python -m scripts.load_backtest_forecasts --model lgbm_cust_enriched --replace
	$(UV) python -m scripts.load_backtest_forecasts --model catboost_cust_enriched --replace
	$(UV) python -m scripts.load_backtest_forecasts --model xgboost_cust_enriched --replace

backtest-chronos2:
	$(UV) python -m scripts.run_backtest_chronos2

backtest-load-chronos2:
	$(UV) python -m scripts.load_backtest_forecasts --model chronos2 --replace

backtest-chronos2-full: backtest-chronos2 backtest-load-chronos2

backtest-chronos2e:
	$(UV) python -m scripts.run_backtest_chronos2_enriched

backtest-load-chronos2e:
	$(UV) python -m scripts.load_backtest_forecasts --model chronos2_enriched --replace

backtest-chronos2e-full: backtest-chronos2e backtest-load-chronos2e

backtest-seasonal-naive:
	$(UV) python scripts/run_backtest.py --model seasonal_naive $(ARGS)

backtest-load-seasonal-naive:
	$(UV) python scripts/load_backtest_forecasts.py --model seasonal_naive --replace

backtest-rolling-mean:
	$(UV) python scripts/run_backtest.py --model rolling_mean $(ARGS)

backtest-load-rolling-mean:
	$(UV) python scripts/load_backtest_forecasts.py --model rolling_mean --replace

backtest-mstl:
	$(UV) python scripts/run_backtest_mstl.py $(ARGS)

backtest-load-mstl:
	$(UV) python scripts/load_backtest_forecasts.py --model mstl --replace

backtest-mstl-full: backtest-mstl backtest-load-mstl

backtest-nhits:
	$(UV) python scripts/run_backtest_dl.py --model nhits $(ARGS)

backtest-load-nhits:
	$(UV) python scripts/load_backtest_forecasts.py --model nhits --replace

backtest-nhits-full: backtest-nhits backtest-load-nhits

backtest-nbeats:
	$(UV) python scripts/run_backtest_dl.py --model nbeats $(ARGS)

backtest-load-nbeats:
	$(UV) python scripts/load_backtest_forecasts.py --model nbeats --replace

backtest-nbeats-full: backtest-nbeats backtest-load-nbeats

backtest-baselines: backtest-seasonal-naive backtest-rolling-mean

backtest-all: backtest-lgbm backtest-catboost backtest-xgboost backtest-chronos backtest-bolt backtest-chronos2 backtest-chronos2e

backtest-all-parallel:
	@mkdir -p data/backtest/logs
	@echo "[parallel] Starting LGBM, CatBoost, XGBoost, Chronos, Bolt, Chronos2 concurrently — logs in data/backtest/logs/"
	$(UV) python scripts/run_backtest.py $(ARGS) > data/backtest/logs/lgbm.log 2>&1 & \
	$(UV) python scripts/run_backtest_catboost.py $(ARGS) > data/backtest/logs/catboost.log 2>&1 & \
	$(UV) python scripts/run_backtest_xgboost.py $(ARGS) > data/backtest/logs/xgboost.log 2>&1 & \
	$(UV) python -m scripts.run_backtest_chronos > data/backtest/logs/chronos.log 2>&1 & \
	$(UV) python -m scripts.run_backtest_chronos_bolt > data/backtest/logs/chronos_bolt.log 2>&1 & \
	$(UV) python -m scripts.run_backtest_chronos2 > data/backtest/logs/chronos2.log 2>&1 & \
	wait && echo "[parallel] All six backtests complete. Check data/backtest/logs/ for output."

backtest-load:
	$(UV) python scripts/load_backtest_forecasts.py --model $(MODEL) --replace

backtest-load-all:
	$(UV) python scripts/load_backtest_forecasts.py --all --replace

backtest-load-all-bulk:
	$(UV) python scripts/load_backtest_forecasts.py --all --replace --bulk

backtest-load-bulk:  ## Load 4 core models with single index cycle (~4x faster)
	$(UV) python scripts/load_backtest_forecasts.py --models lgbm_cluster catboost_cluster xgboost_cluster chronos --replace --bulk

backtest-load-main-only:  ## Load specific models to main table only (skip archive). Usage: make backtest-load-main-only MODELS="lgbm_cluster chronos"
	$(UV) python scripts/load_backtest_forecasts.py --models $(MODELS) --replace --bulk --main-only

backtest-load-archive-only:  ## Load specific models to archive only (skip main). Usage: make backtest-load-archive-only MODELS="lgbm_cluster chronos"
	$(UV) python scripts/load_backtest_forecasts.py --models $(MODELS) --replace --bulk --archive-only

# ---------------------------------------------------------------------------
# External ML forecast loading (ext_lgbm, ext_cat, ext_xg, ext_best)
# ---------------------------------------------------------------------------
load-ext-lgbm:   ## Load ext_lgbm from data/input/df_ml_lgbm_l2_extract.csv
	$(UV) python scripts/etl/load_ext_ml_forecasts.py --model ext_lgbm --replace

load-ext-cat:    ## Load ext_cat from data/input/df_ml_cat_l2_extract.csv
	$(UV) python scripts/etl/load_ext_ml_forecasts.py --model ext_cat --replace

load-ext-xg:     ## Load ext_xg from data/input/df_ml_xg_l2_extract.csv
	$(UV) python scripts/etl/load_ext_ml_forecasts.py --model ext_xg --replace

load-ext-best:   ## Load ext_best from data/input/df_ml_best.csv
	$(UV) python scripts/etl/load_ext_ml_forecasts.py --model ext_best --replace

load-ext-all: load-ext-lgbm load-ext-cat load-ext-xg load-ext-best  ## Load all 4 external ML forecast models
	@echo "All external ML forecasts loaded."

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
	$(PSQL) -v ON_ERROR_STOP=1 \
		-c "REFRESH MATERIALIZED VIEW agg_accuracy_by_dim; REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive; REFRESH MATERIALIZED VIEW agg_dfu_coverage; REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive;"

accuracy-slice-check:
	curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=cluster_assignment" | python3 -m json.tool | head -60
	curl -s "http://localhost:8000/forecast/accuracy/lag-curve" | python3 -m json.tool | head -40

champion-select:
	$(UV) python scripts/run_champion_selection.py

champion-simulate:
	$(UV) python scripts/simulate_champion_strategies.py

champion-train-meta:
	$(UV) python scripts/train_meta_learner.py

champion-all: champion-train-meta champion-simulate champion-select

tune-lgbm:
	$(UV) python scripts/tune_hyperparams.py --model lgbm

tune-catboost:
	$(UV) python scripts/tune_hyperparams.py --model catboost

tune-xgboost:
	$(UV) python scripts/tune_hyperparams.py --model xgboost

tune-all: tune-lgbm tune-catboost tune-xgboost

# ── Production Baseline Seeding ──────────────────────────────────────────────
seed-baselines:          ## Seed production baselines into experiment tables
	$(UV) python scripts/seed_production_baselines.py

seed-baselines-tuning:
	$(UV) python scripts/seed_production_baselines.py --scope tuning

seed-baselines-champion:
	$(UV) python scripts/seed_production_baselines.py --scope champion

seed-baselines-clustering:
	$(UV) python scripts/seed_production_baselines.py --scope clustering

# ── LGBM Tuning ──────────────────────────────────────────────────────────────
lgbm-tuning-list:
	$(UV) python scripts/ml/compare_backtest_runs.py --list

lgbm-tuning-compare:
	$(UV) python scripts/ml/compare_backtest_runs.py --baseline $(BASELINE) --candidate $(CANDIDATE)

lgbm-tuning-backup:
	$(UV) python scripts/ml/compare_backtest_runs.py --backup $(RUN)

lgbm-tuning-run:
	$(UV) python scripts/run_backtest.py --model lgbm
	$(UV) python scripts/ml/compare_backtest_runs.py --register-latest --auto-compare

lgbm-auto-tune:
	$(UV) python scripts/ml/auto_tune.py --runs $(or $(RUNS),3)

lgbm-auto-tune-promote:
	$(UV) python scripts/ml/auto_tune.py --runs $(or $(RUNS),3) --promote

lgbm-auto-tune-dry-run:
	$(UV) python scripts/ml/auto_tune.py --runs $(or $(RUNS),10) --dry-run

lgbm-auto-tune-list:
	$(UV) python scripts/ml/auto_tune.py --list-strategies

# ── Expert Panel Algorithm Selection ────────────────────────────────────────
expert-panel:            ## Run Expert Panel test (5000 DFUs, 5 timeframes, ~30 min)
	$(UV) python -m algorithm_testing.run_expert_panel

expert-panel-quick:      ## Quick Expert Panel test (1000 DFUs, 3 timeframes, ~8 min)
	$(UV) python -m algorithm_testing.run_expert_panel --n-dfus 1000 --n-timeframes 3

expert-panel-mini:       ## Minimal Expert Panel test (200 DFUs, 2 timeframes, ~2 min)
	$(UV) python -m algorithm_testing.run_expert_panel --n-dfus 200 --n-timeframes 2

expert-panel-loc:        ## Run Expert Panel for all DFUs at a specific location: make expert-panel-loc LOC=1401-BULK
	@if [ -z "$(LOC)" ]; then echo "Usage: make expert-panel-loc LOC=1401-BULK"; exit 1; fi
	$(UV) python -m algorithm_testing.run_expert_panel --loc $(LOC)

# ── Advanced Expert Panel (Foundation Models + Deep Learning) ───────────────
adv-expert-panel:        ## Advanced Expert Panel (5000 DFUs, 10 TFs, execution-lag accuracy, foundation+DL+stat upgrades)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m adv_algorithm_testing.run_adv_expert_panel --n-timeframes 10

adv-expert-panel-quick:  ## Quick Advanced Expert Panel (1000 DFUs, 5 TFs, execution-lag accuracy)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m adv_algorithm_testing.run_adv_expert_panel --n-dfus 1000 --n-timeframes 5

adv-expert-panel-mini:   ## Minimal Advanced Expert Panel (200 DFUs, 2 TFs)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m adv_algorithm_testing.run_adv_expert_panel --n-dfus 200 --n-timeframes 2

adv-expert-panel-loc:    ## Advanced Expert Panel for all DFUs at a specific location: make adv-expert-panel-loc LOC=1401-BULK
	@if [ -z "$(LOC)" ]; then echo "Usage: make adv-expert-panel-loc LOC=1401-BULK"; exit 1; fi
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m adv_algorithm_testing.run_adv_expert_panel --loc $(LOC)

route-analysis:          ## Compare per-DFU routing strategies on saved predictions (no retraining, ~2 min)
	$(UV) python -m adv_algorithm_testing.route_analysis

route-analysis-min3:     ## Same as route-analysis but require 3+ timeframes of history per DFU
	$(UV) python -m adv_algorithm_testing.route_analysis --min-history 3

# ── Expert System Backtest (full population, segment-assigned algorithm) ─────
expsys-backtest:         ## Full ExpSys backtest: all DFUs, 10 TFs, loads to DB (~4-5h)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_expert_system_backtest

expsys-backtest-dry:     ## ExpSys accuracy only — no DB loading (--skip-load)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_expert_system_backtest --skip-load

expsys-backtest-replace: ## ExpSys: delete existing rows first, then reload
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_expert_system_backtest --replace

commit:
	@if [ -z "$(MSG)" ]; then echo "Usage: make commit MSG=\"your message\""; exit 1; fi
	git add -A && (git diff --staged --quiet || git commit -m "$(MSG)") && git push -u origin HEAD

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

ai-sync-check:
	bash scripts/ai_checks/sync_check.sh

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

# ── Unified Pipeline Orchestrator ──────────────────────────────

pipeline-full:
	$(UV) python scripts/etl/run_pipeline.py --mode full --parallel

pipeline-refresh:
	$(UV) python scripts/etl/run_pipeline.py --mode refresh

pipeline-inventory:
	$(UV) python scripts/etl/run_pipeline.py --mode full --domains inventory

pipeline-inventory-refresh:
	$(UV) python scripts/etl/run_pipeline.py --mode refresh --domains inventory

# ── Full Application Setup (input CSVs → ready-to-use app) ────
# Usage: make setup-all        (everything including ML, ~4-6 hours)
#        make setup-data        (data only, no ML, ~30 min)
#        make setup-planning    (data + inv planning, no ML, ~1 hour)

setup-data:
	$(UV) python scripts/etl/run_pipeline.py --mode full --parallel
	@echo "✓ Phase 1 complete: all data loaded into Postgres (parallel pipeline)"

setup-features: setup-data cluster-all seasonality-all variability-all lt-profile-all abc-xyz-all demand-signals-all
	@echo "✓ Phase 2 complete: clustering, seasonality, variability, lead time, ABC-XYZ, demand signals"

setup-backtest: setup-features backtest-all backtest-load-all accuracy-slice-refresh champion-all seed-baselines
	@echo "✓ Phase 3 complete: backtests, champion selection"

setup-inv-planning: eoq-all policy-all ss-all exceptions-generate fill-rate-all health-all supplier-perf-all investment-all intramonth-all control-tower-all rebalancing-all
	@echo "✓ Phase 4 complete: inventory planning (safety stock, EOQ, policies, exceptions, health)"

setup-demand-planning: forecast-prod-all projection-all po-all quantile-all consensus-all planned-orders-all replplan-all bias-all blended-all service-level-all lead-time-all echelon-all
	@echo "✓ Phase 5 complete: demand planning (forecasts, projections, orders, replenishment)"

setup-ops: sop-all events-all financial-plan-all storyboard-all scenarios-all dq-all
	@echo "✓ Phase 6 complete: operations (S&OP, events, financial plan, storyboard, DQ)"

setup-planning: setup-data setup-inv-planning
	@echo "✓ Data + Inventory Planning complete (no ML)"

setup-all: setup-backtest setup-inv-planning setup-demand-planning setup-ops
	@echo ""
	@echo "============================================================"
	@echo "  Setup complete. Start the application:"
	@echo "    make api    # FastAPI on :8000"
	@echo "    make ui     # React UI on :5173"
	@echo "============================================================"

# ── Performance Profiling ────────────────────────────────────────────────────
perf-report:                           ## Full system perf report (read-only, safe for prod)
	$(UV) python scripts/ops/run_perf_analysis.py --mode report

perf-script:                           ## Profile a script: make perf-script SCRIPT=compute_safety_stock (read-only)
	$(UV) python scripts/ops/run_perf_analysis.py --mode script --script $(SCRIPT)

perf-script-full:                      ## Profile with REAL writes (use on staging only): make perf-script-full SCRIPT=X
	$(UV) python scripts/ops/run_perf_analysis.py --mode script --script $(SCRIPT) --no-readonly

perf-api:                              ## API endpoint performance analysis (read-only)
	$(UV) python scripts/ops/run_perf_analysis.py --mode api

perf-pipeline:                         ## ETL pipeline performance analysis (read-only)
	$(UV) python scripts/ops/run_perf_analysis.py --mode pipeline

perf-clean:                            ## Truncate all perf profiling history from DB
	psql "$(DATABASE_URL)" -c "TRUNCATE perf_suggestion, perf_query, perf_section, perf_run CASCADE;"

# ── Database Cleanup & Fresh Recreate ────────────────────────────────────────
# Full wipe-and-reload: clears non-config data/history, reloads from data/input/,
# and refreshes the core ML + baseline planning outputs while preserving configs.
# See docs/RUNBOOK.md "Database Cleanup & Fresh Recreate" for details.

db-truncate-data:                      ## Truncate non-config data/history (preserves configuration masters)
	@echo "Truncating non-config data, history, and experiment tables (preserving configuration masters)..."
	@printf '%s\n' \
	  'BEGIN;' \
	  'TRUNCATE TABLE ai_recommendation_outcomes CASCADE;' \
	  'TRUNCATE TABLE ai_insights CASCADE;' \
	  'TRUNCATE TABLE ai_planning_memos CASCADE;' \
	  'TRUNCATE TABLE ai_call_log CASCADE;' \
	  'TRUNCATE TABLE chat_embeddings CASCADE;' \
	  'TRUNCATE TABLE planner_decisions CASCADE;' \
	  'TRUNCATE TABLE exception_queue CASCADE;' \
	  'TRUNCATE TABLE fact_sop_approved_plan CASCADE;' \
	  'TRUNCATE TABLE fact_sop_gaps CASCADE;' \
	  'TRUNCATE TABLE fact_sop_supply_constraints CASCADE;' \
	  'TRUNCATE TABLE fact_sop_demand_review CASCADE;' \
	  'TRUNCATE TABLE fact_sop_cycles CASCADE;' \
	  'TRUNCATE TABLE fact_event_conflicts CASCADE;' \
	  'TRUNCATE TABLE fact_event_performance CASCADE;' \
	  'TRUNCATE TABLE fact_event_adjusted_forecast CASCADE;' \
	  'TRUNCATE TABLE fact_event_calendar CASCADE;' \
	  'TRUNCATE TABLE fact_scenario_results CASCADE;' \
	  'TRUNCATE TABLE fact_supply_scenarios CASCADE;' \
	  'TRUNCATE TABLE fact_po_approval_log CASCADE;' \
	  'TRUNCATE TABLE fact_po_receipts CASCADE;' \
	  'TRUNCATE TABLE fact_open_purchase_orders CASCADE;' \
	  'TRUNCATE TABLE fact_purchase_orders CASCADE;' \
	  'TRUNCATE TABLE fact_consensus_plan CASCADE;' \
	  'TRUNCATE TABLE fact_forecast_overrides CASCADE;' \
	  'TRUNCATE TABLE fact_rebalancing_transfer CASCADE;' \
	  'TRUNCATE TABLE fact_rebalancing_plan CASCADE;' \
	  'TRUNCATE TABLE backtest_lag_archive CASCADE;' \
	  'TRUNCATE TABLE fact_external_forecast_monthly CASCADE;' \
	  'TRUNCATE TABLE fact_production_forecast CASCADE;' \
	  'TRUNCATE TABLE fact_blended_demand_plan CASCADE;' \
	  'TRUNCATE TABLE fact_demand_plan CASCADE;' \
	  'TRUNCATE TABLE fact_demand_plan_weekly CASCADE;' \
	  'TRUNCATE TABLE fact_bias_corrections CASCADE;' \
	  'TRUNCATE TABLE fact_bias_correction_history CASCADE;' \
	  'TRUNCATE TABLE fact_inventory_snapshot CASCADE;' \
	  'TRUNCATE TABLE fact_inventory_projection CASCADE;' \
	  'TRUNCATE TABLE fact_sales_monthly CASCADE;' \
	  'TRUNCATE TABLE fact_sales_monthly_original CASCADE;' \
	  'TRUNCATE TABLE fact_customer_demand_monthly CASCADE;' \
	  'TRUNCATE TABLE fact_ss_simulation_results CASCADE;' \
	  'TRUNCATE TABLE fact_safety_stock_targets CASCADE;' \
	  'TRUNCATE TABLE fact_eoq_targets CASCADE;' \
	  'TRUNCATE TABLE fact_demand_signals CASCADE;' \
	  'TRUNCATE TABLE fact_replenishment_plan CASCADE;' \
	  'TRUNCATE TABLE fact_replenishment_exceptions CASCADE;' \
	  'TRUNCATE TABLE fact_planned_orders CASCADE;' \
	  'TRUNCATE TABLE fact_plan_versions CASCADE;' \
	  'TRUNCATE TABLE fact_financial_inventory_plan CASCADE;' \
	  'TRUNCATE TABLE fact_inventory_investment_plan CASCADE;' \
	  'TRUNCATE TABLE fact_efficient_frontier CASCADE;' \
	  'TRUNCATE TABLE fact_echelon_reorder_points CASCADE;' \
	  'TRUNCATE TABLE fact_echelon_ss_targets CASCADE;' \
	  'TRUNCATE TABLE fact_service_level_performance CASCADE;' \
	  'TRUNCATE TABLE fact_lead_time_actuals CASCADE;' \
	  'TRUNCATE TABLE fact_lt_review_triggers CASCADE;' \
	  'TRUNCATE TABLE fact_external_signal CASCADE;' \
	  'TRUNCATE TABLE fact_dq_corrections CASCADE;' \
	  'TRUNCATE TABLE fact_dq_check_results CASCADE;' \
	  'TRUNCATE TABLE fact_annotation CASCADE;' \
	  'TRUNCATE TABLE fact_shared_view CASCADE;' \
	  'TRUNCATE TABLE fact_intervention_metrics CASCADE;' \
	  'TRUNCATE TABLE fact_notification_log CASCADE;' \
	  'TRUNCATE TABLE fact_webhook_delivery CASCADE;' \
	  'TRUNCATE TABLE fact_report_delivery CASCADE;' \
	  'TRUNCATE TABLE fact_audit_log CASCADE;' \
	  'TRUNCATE TABLE fact_query_performance CASCADE;' \
	  'TRUNCATE TABLE audit_load_batch CASCADE;' \
	  'TRUNCATE TABLE job_history CASCADE;' \
	  'TRUNCATE TABLE perf_suggestion CASCADE;' \
	  'TRUNCATE TABLE perf_query CASCADE;' \
	  'TRUNCATE TABLE perf_section CASCADE;' \
	  'TRUNCATE TABLE perf_run CASCADE;' \
	  'TRUNCATE TABLE tuning_chat_message CASCADE;' \
	  'TRUNCATE TABLE tuning_chat_session CASCADE;' \
	  'TRUNCATE TABLE tuning_promotion_log CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_lag_cluster CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_lag CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_comparison CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_month CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_cluster CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_timeframe CASCADE;' \
	  'TRUNCATE TABLE lgbm_tuning_run CASCADE;' \
	  'TRUNCATE TABLE cluster_experiment_comparison CASCADE;' \
	  'TRUNCATE TABLE cluster_experiment CASCADE;' \
	  'TRUNCATE TABLE champion_experiment CASCADE;' \
	  'TRUNCATE TABLE dim_sku CASCADE;' \
	  'TRUNCATE TABLE dim_item CASCADE;' \
	  'TRUNCATE TABLE dim_location CASCADE;' \
	  'TRUNCATE TABLE dim_customer CASCADE;' \
	  'TRUNCATE TABLE dim_time CASCADE;' \
	  'TRUNCATE TABLE dim_sourcing CASCADE;' \
	  'TRUNCATE TABLE dim_item_lead_time_profile CASCADE;' \
	  'TRUNCATE TABLE dim_lead_time_profile CASCADE;' \
	  'TRUNCATE TABLE mv_demand_decomposition CASCADE;' \
	  'COMMIT;' \
	  | $(PSQL) -v ON_ERROR_STOP=1
	@echo "✓ Reset complete. Configuration masters preserved."

clean-artifacts:                       ## Remove stale intermediate files (clean CSVs, backtest, tuning, clustering, champion)
	rm -f data/*_clean.csv data/inventory_clean.csv
	rm -rf data/backtest/lgbm_cluster/ data/backtest/catboost_cluster/ data/backtest/xgboost_cluster/ data/backtest/chronos/ data/backtest/chronos_bolt/ data/backtest/chronos2/ data/backtest/chronos2_enriched/
	rm -rf data/backtest/logs/ data/backtest/tuning_archive/ data/tuning/ data/perf_reports/
	rm -rf data/clustering/ data/champion/ data/models/
	rm -f data/seasonality_results.csv data/clustering_features.csv
	@echo "✓ Intermediate artifacts cleaned."

refresh-mvs-tiered:                    ## Refresh all MVs in dependency order (4 tiers)
	@echo "Refreshing materialized views (tier-ordered)..."
	@echo "  Tier 1: Base aggregates (no dependencies)"
	@echo "  Tier 2: Derived views (depend on Tier 1)"
	@echo "  Tier 3: Cross-domain views (depend on Tier 2)"
	@echo "  Tier 4: Dashboard views (depend on all above)"
	$(PSQL) -v ON_ERROR_STOP=1 -c " \
	  /* --- Tier 1: Base aggregates (source: fact tables only) --- */ \
	  REFRESH MATERIALIZED VIEW agg_sales_monthly; \
	  REFRESH MATERIALIZED VIEW agg_forecast_monthly; \
	  REFRESH MATERIALIZED VIEW agg_inventory_monthly; \
	  /* --- Tier 2: Derived views (depend on Tier 1 aggregates) --- */ \
	  REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly; \
	  REFRESH MATERIALIZED VIEW mv_fill_rate_monthly; \
	  REFRESH MATERIALIZED VIEW mv_intramonth_stockout; \
	  /* --- Tier 3: Supplier / procurement views --- */ \
	  REFRESH MATERIALIZED VIEW mv_supplier_performance; \
	  REFRESH MATERIALIZED VIEW mv_supplier_po_performance; \
	  REFRESH MATERIALIZED VIEW mv_po_lead_time_analysis; \
	  REFRESH MATERIALIZED VIEW agg_accuracy_by_dim; \
	  REFRESH MATERIALIZED VIEW agg_dfu_coverage; \
	  /* --- Tier 4: Dashboard / composite views (depend on all above) --- */ \
	  REFRESH MATERIALIZED VIEW mv_inventory_health_score; \
	  REFRESH MATERIALIZED VIEW mv_control_tower_kpis; \
	"
	@echo "✓ All materialized views refreshed."

refresh-accuracy-mvs:                  ## Refresh accuracy MVs (after backtest load)
	$(PSQL) -v ON_ERROR_STOP=1 -c " \
	  REFRESH MATERIALIZED VIEW agg_accuracy_by_dim; \
	  REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive; \
	  REFRESH MATERIALIZED VIEW agg_dfu_coverage; \
	  REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive; \
	"
	@echo "✓ Accuracy materialized views refreshed."

fresh-load: normalize-all load-all refresh-mvs-tiered  ## Normalize + load + refresh MVs (from input CSVs)
	@echo "✓ Fresh data load complete."

fresh-features: fresh-load cluster-all seasonality-all variability-all lt-profile-all  ## Load + clustering + seasonality + variability + LT
	@echo "✓ Fresh load + feature engineering complete."

fresh-backtest: fresh-features backtest-all backtest-load-all refresh-accuracy-mvs  ## Load + features + backtests + accuracy refresh
	@echo "✓ Fresh load + features + backtests complete."

fresh-champion: fresh-backtest champion-all  ## Load + features + backtests + champion selection (full ML pipeline)
	@echo ""
	@echo "============================================================"
	@echo "  Fresh recreate complete (data → champion selection)."
	@echo "  Run 'make check-all' to validate, then start services."
	@echo "============================================================"

fresh-all: db-truncate-data clean-artifacts fresh-champion seed-baselines policy-all ss-all eoq-all health-all  ## Full cleanup & recreate: truncate → clean → load → ML → champion → baselines + baseline planning
	@echo ""
	@echo "============================================================"
	@echo "  Full database cleanup & recreate complete."
	@echo "  Core data, champion forecasts, and baseline planning outputs refreshed."
	@echo "  Start the application:"
	@echo "    make api    # FastAPI on :8000"
	@echo "    make ui     # React UI on :5173"
	@echo "============================================================"
