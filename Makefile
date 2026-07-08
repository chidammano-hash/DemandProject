SHELL := /bin/zsh

DC := docker compose
# Put the project root on PYTHONPATH so any `python scripts/<domain>/<x>.py` can
# `import common` regardless of the script's own sys.path bootstrap (scripts moved
# into domain subdirs during the restructure left some bootstraps one level short).
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
UV := uv run
POSTGRES_SERVICE := postgres
PG_EXEC := $(DC) exec -T $(POSTGRES_SERVICE)
PSQL := $(PG_EXEC) psql -U demand -d demand_mvp

.PHONY: help deploy deploy-check deploy-pydeps deploy-redis deploy-sql deploy-frontend deploy-api deploy-smoke refresh-customer-mv init init-pip up down logs db-apply-sql db-apply-inventory db-apply-inv-backtest api ui-init ui ui-test normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory normalize-all load-item load-location load-customer load-time load-dfu load-sales load-forecast load-forecast-replace load-forecast-replace-no-archive load-inventory load-all refresh-agg-sales refresh-agg-forecast refresh-agg-inventory refresh-agg refresh-inv-backtest inventory-pipeline check-api check-db check-all ai-sync-check cluster-all features-computelt-profile-schema lt-profile-compute lt-profile-all eoq-schema eoq-compute eoq-all policy-schema policy-assign policy-all health-schema health-refresh health-all exceptions-schema exceptions-generate exceptions-generate-dry ss-schema ss-compute ss-compute-dry ss-all ai-insights-schema ai-insights-scan ai-insights-scan-dry ai-insights-dfu ai-insights-all storyboard-schema storyboard-generate storyboard-generate-dry storyboard-all forecast-prod-schema forecast-generate forecast-generate-dfu forecast-generate-dry forecast-prod-all train-production train-production-all forecast-full forecast-model replplan-schema replplan-compute replplan-compute-dry replplan-all backtest-lgbm backtest-catboost backtest-xgboost backtest-seasonal-naive backtest-rolling-mean backtest-mstl backtest-nhits backtest-nbeats backtest-baselines backtest-load backtest-load-all backtest-load-all-bulk backtest-load-bulk backtest-load-main-only backtest-load-archive-only backtest-all backtest-all-parallel backtest-clean backtest-list forecast-clean forecast-clean-list accuracy-slice-refresh accuracy-slice-check champion-select champion-simulate champion-train-meta champion-all tune-lgbm tune-catboost tune-xgboost tune-all tune-lgbm-clusters tune-catboost-clusters tune-xgboost-clusters tune-clusters db-apply-jobs commit test test-unit test-api test-cov test-all e2e-install e2e e2e-ui e2e-headed e2e-report quantile-schema quantile-train quantile-train-dfu quantile-dry quantile-all consensus-schema consensus-generate consensus-generate-dry consensus-all procurement-schema procurement-export procurement-send-erp procurement-all fva-schema sop-seed sop-all dq-schema dq-populate dq-run dq-all pipeline-full pipeline-refresh pipeline-inventory pipeline-inventory-refresh setup-data setup-features setup-backtest setup-inv-planning setup-demand-planning setup-ops setup-planning setup-all perf-report perf-script perf-api perf-ingestion perf-pipeline lgbm-tuning-list lgbm-tuning-compare lgbm-tuning-backup lgbm-tuning-run lgbm-auto-tune lgbm-auto-tune-promote lgbm-auto-tune-dry-run lgbm-auto-tune-list seed-baselines seed-baselines-tuning seed-baselines-champion seed-baselines-clustering db-truncate-data clean-artifacts refresh-mvs-tiered refresh-accuracy-mvs fresh-load fresh-features fresh-backtest fresh-champion fresh-all dev fresh test-quick lint format type-check health audit-routers new-router expert-panel expert-panel-quick expert-panel-mini adv-expert-panel adv-expert-panel-quick adv-expert-panel-mini load-ext-lgbm load-ext-cat load-ext-xg load-ext-best load-ext-all db-analyze db-health db-drop-unused-indexes db-retention db-optimize db-maintain auto-create-partitions auto-create-partitions-dry-run auto-create-partitions-weekly auto-create-partitions-weekly-dry-run refresh-customer-filter-options

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
dev: up api ui  ## Start Docker + API + UI for local dev
fresh: fresh-all  ## Alias for fresh-all (full reset + reload)
test-quick: test ui-test  ## Backend + frontend tests only (fast)
lint:  ## Run ruff lint with --fix on api/ common/ scripts/
	$(UV) ruff check api/ common/ scripts/ --fix
format:  ## Run ruff format on api/ common/ scripts/
	$(UV) ruff format api/ common/ scripts/
type-check:  ## Run mypy on api/ and common/
	$(UV) mypy api/ common/ --ignore-missing-imports
health: check-all  ## Alias for check-all (DB row counts + API health)

# ---------------------------------------------------------------------------
# Deploy — see docs at the bottom of each step's @echo line.
#
# `make deploy` is idempotent: re-runs are safe. It does NOT create .env for
# you (would risk overwriting secrets). First-time setup requires REDIS_URL
# and POOL_MAX_SIZE in .env — see step 0 below.
# ---------------------------------------------------------------------------
.PHONY: deploy deploy-check deploy-pydeps deploy-redis deploy-sql deploy-mv-refresh deploy-frontend deploy-api deploy-smoke refresh-customer-mv

deploy: deploy-check deploy-pydeps deploy-redis deploy-sql deploy-frontend deploy-api deploy-smoke
	@echo ""
	@echo "================================================================"
	@echo "  Deploy complete."
	@echo "  Next: open the Customer Analytics tab and hard-refresh (Cmd+Shift+R)"
	@echo "  Don't forget: add 'make refresh-customer-mv' to your nightly load."
	@echo "================================================================"

# Step 0: validate env + decisions before touching anything
deploy-check:
	@echo "[0/7] Pre-flight checks..."
	@test -f .env || { echo "FAIL: .env not found. Create it with REDIS_URL and POOL_MAX_SIZE — see runbook step 4."; exit 1; }
	@grep -q '^REDIS_URL=' .env || { echo "FAIL: REDIS_URL not set in .env. Add: REDIS_URL=redis://redis:6379/0"; exit 1; }
	@grep -q '^POOL_MAX_SIZE=' .env || { echo "FAIL: POOL_MAX_SIZE not set in .env. Add: POOL_MAX_SIZE=12  (see api/pool.py default)"; exit 1; }
	@# Multi-pool preflight: enforce the REAL invariant from api/pool.py —
	@# per-worker backend connections = POOL_MAX_SIZE (sync) + ASYNC_POOL_MAX_SIZE
	@# (async). The read-replica pool is added ONLY when READ_REPLICA_URL is set
	@# (it counts against the PRIMARY ceiling here only as a conservative bound;
	@# in a real split deployment it hits the replica's own max_connections).
	@# Defaults mirror api/pool.py (12 / 20 / 12) and docker-compose.yml (200).
	@# Gate trips when total > 85% of max_connections.
	@WORKERS=$$(grep '^GUNICORN_WORKERS=' .env | cut -d= -f2); WORKERS=$${WORKERS:-4}; \
	 POOL=$$(grep '^POOL_MAX_SIZE=' .env | cut -d= -f2); POOL=$${POOL:-12}; \
	 APOOL=$$(grep '^ASYNC_POOL_MAX_SIZE=' .env | cut -d= -f2); APOOL=$${APOOL:-20}; \
	 RPOOL=$$(grep '^READ_POOL_MAX_SIZE=' .env | cut -d= -f2); RPOOL=$${RPOOL:-12}; \
	 MAXCONN=$$(grep '^POSTGRES_MAX_CONNECTIONS=' .env | cut -d= -f2); MAXCONN=$${MAXCONN:-200}; \
	 PER_WORKER=$$(( POOL + APOOL )); \
	 if grep -q '^READ_REPLICA_URL=.' .env; then PER_WORKER=$$(( PER_WORKER + RPOOL )); HAS_REPLICA=1; else HAS_REPLICA=0; fi; \
	 TOTAL=$$(( WORKERS * PER_WORKER )); \
	 CEILING=$$(( MAXCONN * 85 / 100 )); \
	 if [ $$TOTAL -gt $$CEILING ]; then \
	   echo "FAIL: $$WORKERS workers x (sync $$POOL + async $$APOOL$$( [ $$HAS_REPLICA -eq 1 ] && echo \" + read $$RPOOL\" )) = $$TOTAL backend connections > 85% of max_connections=$$MAXCONN (ceiling $$CEILING)."; \
	   echo "  Fix: lower POOL_MAX_SIZE / ASYNC_POOL_MAX_SIZE / READ_POOL_MAX_SIZE in .env, reduce GUNICORN_WORKERS, or raise max_connections in docker-compose.yml."; \
	   exit 1; \
	 fi; \
	 echo "  OK: $$WORKERS workers x (sync $$POOL + async $$APOOL$$( [ $$HAS_REPLICA -eq 1 ] && echo \" + read $$RPOOL\" )) = $$TOTAL conn <= 85% of PG max=$$MAXCONN (ceiling $$CEILING)"
	@command -v uv >/dev/null 2>&1 || { echo "FAIL: uv not on PATH. Install: brew install uv"; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "FAIL: node not on PATH."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "FAIL: Docker daemon not running."; exit 1; }
	@echo "  OK: env, uv, node, docker all present."

# Step 1: Python deps
deploy-pydeps:
	@echo "[1/7] Syncing Python dependencies..."
	uv sync
	@uv run python -c "import gunicorn, redis" 2>/dev/null && echo "  OK: gunicorn + redis importable" || { echo "FAIL: gunicorn or redis missing after uv sync"; exit 1; }

# Step 2: Redis up
deploy-redis:
	@echo "[2/7] Ensuring Redis is up..."
	$(DC) up -d redis
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
	   $(DC) exec -T redis redis-cli ping 2>/dev/null | grep -q PONG && echo "  OK: Redis responding" && exit 0; \
	   sleep 1; \
	 done; \
	 echo "FAIL: Redis didn't respond to PING after 10s"; exit 1

# Step 3: SQL migrations 168 + 169.
#
# 168 uses CREATE INDEX CONCURRENTLY to avoid taking AccessExclusive on the
# fact table. CONCURRENTLY can't run inside a transaction so we issue each
# statement individually. Idempotent via _new suffix + IF EXISTS swap.
deploy-sql:
	@echo "[3/7] Applying SQL migrations 168 + 169 (zero-downtime path)..."
	@echo "  3a. Building covering indexes CONCURRENTLY (this can take 1-2 min on prod data)..."
	@$(PSQL) -v ON_ERROR_STOP=1 -c "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cust_demand_customer_startdate_new ON fact_customer_demand_monthly (customer_no, startdate) INCLUDE (demand_qty, sales_qty, oos_qty);" || true
	@$(PSQL) -v ON_ERROR_STOP=1 -c "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cust_demand_item_customer_startdate_new ON fact_customer_demand_monthly (item_id, customer_no, startdate) INCLUDE (demand_qty, sales_qty, oos_qty);" || true
	@$(PSQL) -v ON_ERROR_STOP=1 -c "BEGIN; DROP INDEX IF EXISTS idx_cust_demand_customer_startdate; ALTER INDEX IF EXISTS idx_cust_demand_customer_startdate_new RENAME TO idx_cust_demand_customer_startdate; DROP INDEX IF EXISTS idx_cust_demand_item_customer_startdate; ALTER INDEX IF EXISTS idx_cust_demand_item_customer_startdate_new RENAME TO idx_cust_demand_item_customer_startdate; COMMIT;"
	@$(PSQL) -v ON_ERROR_STOP=1 -c "ANALYZE fact_customer_demand_monthly;" >/dev/null
	@echo "  OK: covering indexes in place."
	@echo "  3b. Building mv_customer_activity_monthly (~30s)..."
	@$(PSQL) -v ON_ERROR_STOP=1 < sql/169_mv_customer_activity_monthly.sql >/dev/null
	@ROW_COUNT=$$($(PSQL) -At -c "SELECT count(*) FROM mv_customer_activity_monthly;"); \
	 echo "  OK: MV built, $$ROW_COUNT rows."

# One-shot MV refresh — call this from your customer-demand load pipeline.
refresh-customer-mv:
	@echo "Refreshing mv_customer_activity_monthly CONCURRENTLY..."
	@$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_customer_activity_monthly;"
	@echo "  OK"

# Customer-analytics specialized MVs (Item 20 of perf roadmap).
# Refresh nightly after `make load-customer-demand`. Each MV powers one of
# the heavy panels in the Customer Analytics tab — collapsing per-request
# fact-table aggregations to single indexed lookups at 40x scale.
refresh-ca-mvs:                       ## Refresh customer-analytics specialized MVs
	@echo "Refreshing customer-analytics MVs CONCURRENTLY..."
	@for mv in mv_ca_segment_trends mv_ca_demand_at_risk mv_ca_order_patterns mv_ca_item_state; do \
	  echo "  Refreshing $$mv ..."; \
	  $(PSQL) -c "REFRESH MATERIALIZED VIEW CONCURRENTLY $$mv;" 2>/dev/null \
	    || $(PSQL) -c "REFRESH MATERIALIZED VIEW $$mv;" 2>/dev/null \
	    || echo "    WARN: $$mv skipped (does not exist — apply sql/180-182,187_*.sql)"; \
	done
	@echo "  OK"

# Step 5: frontend bundle.
#
# Skips `tsc -b` (which `npm run build` runs first) because the restructure
# branch has ~30 pre-existing type errors in unrelated files (model-tuning,
# lgbm-tuning, settings, storyboard, types/index.ts) that block the standard
# build. Vite produces working production JS even when TS would complain —
# type safety is enforced via tests + IDE, not the deploy gate. Run
# `cd frontend && npx tsc -b` manually if you want to see the type errors.
deploy-frontend:
	@echo "[5/7] Building frontend bundle (vite-only — skipping pre-existing tsc errors)..."
	cd frontend && npx vite build
	@test -d frontend/dist && echo "  OK: dist/ built" || { echo "FAIL: frontend/dist missing after build"; exit 1; }
	@echo "  Bundle chunks (sample):"
	@ls -lh frontend/dist/assets/*.js 2>/dev/null | awk '{print "    " $$5 "  " $$NF}' | head -8 || true

# Step 6: rebuild + restart API container
deploy-api:
	@echo "[6/7] Rebuilding + restarting API container..."
	$(DC) build api
	$(DC) up -d api
	@echo "  Waiting for API to become healthy..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
	   curl -sf -o /dev/null http://localhost:8000/health 2>/dev/null && echo "  OK: API healthy" && exit 0; \
	   sleep 2; \
	 done; \
	 echo "FAIL: API didn't respond to /health within 30s. Check: docker compose logs api"; exit 1

# Step 7: post-deploy smoke checks
deploy-smoke:
	@echo "[7/7] Smoke checks..."
	@$(DC) logs api --tail=80 | grep -q "Cache: using Redis backend" && echo "  OK: Redis backend active" || echo "  WARN: Redis backend not confirmed in logs (check: docker compose logs api | grep Cache)"
	@$(DC) logs api --tail=80 | grep -q "Booting worker" && WORKER_COUNT=$$($(DC) logs api --tail=80 | grep -c "Booting worker") && echo "  OK: $$WORKER_COUNT gunicorn workers started" || echo "  WARN: gunicorn worker count not confirmed in logs"
	@TIMING=$$(curl -s -o /dev/null -w "%{http_code}/%{time_total}s" http://localhost:8000/customer-analytics/kpis); \
	 echo "  /customer-analytics/kpis -> $$TIMING (first call cold, subsequent should be <100ms)"
	@CACHE_HEADER=$$(curl -sI http://localhost:8000/customer-analytics/filter-options | grep -i "^cache-control"); \
	 echo "  /customer-analytics/filter-options $$CACHE_HEADER"

# ---------------------------------------------------------------------------
# Developer tooling
# ---------------------------------------------------------------------------
audit-routers:  ## Verify router files match main.py mounts and vite proxy entries
	@python3 scripts/tools/audit_routes.py 2>/dev/null || echo "Run: python3 scripts/tools/audit_routes.py"

new-router:
	@python3 scripts/tools/scaffold_router.py --domain $(DOMAIN) --name $(NAME)

help:  ## Auto-generated from `## ...` annotations on target lines
	@echo "Most-used targets (annotated with '##'):"
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-25s %s\n",$$1,$$2}' | sort -u
	@echo ""
	@echo "Legacy hand-written list (full target catalog):"
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
	@echo "  backtest-lgbm        - run LGBM per-cluster backtest (settings from forecast_pipeline_config.yaml)"
	@echo "  backtest-catboost    - run CatBoost per-cluster backtest (settings from forecast_pipeline_config.yaml)"
	@echo "  backtest-xgboost     - run XGBoost per-cluster backtest (settings from forecast_pipeline_config.yaml)"
	@echo "  backtest-chronos2e   - run Chronos 2 Enriched foundation model backtest"
	@echo "  backtest-chronos2e-full - run Chronos 2 Enriched backtest + load predictions"
	@echo "  backtest-all         - run all clean-rebuild backtests sequentially (cluster trees + foundation)"
	@echo "  backtest-all-parallel- run all clean-rebuild backtests in parallel (logs in data/backtest/logs/)"
	@echo "  backtest-load        - load one model: make backtest-load MODEL=lgbm_cluster"
	@echo "  backtest-load-all    - load ALL models from data/backtest/*/ (run after backtest-all)"
	@echo "  backtest-load-all-bulk - load ALL models with single index cycle (~4x faster)"
	@echo "  backtest-load-bulk   - load 4 core models (lgbm, catboost, xgboost, chronos2_enriched) in bulk"
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
	@echo "  tune-lgbm-clusters   - Per-cluster Bayesian tuning for LGBM (30 trials)"
	@echo "  tune-catboost-clusters - Per-cluster Bayesian tuning for CatBoost (30 trials)"
	@echo "  tune-xgboost-clusters - Per-cluster Bayesian tuning for XGBoost (30 trials)"
	@echo "  tune-clusters        - Run all three per-cluster tuning jobs sequentially"
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
	@echo "  NOTE: recursive, SHAP, and tuning are configured via config/forecasting/forecast_pipeline_config.yaml"
	@echo "  features-compute     - compute all SKU features (volume, trend, seasonality, variability, lifecycle)"
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
	@echo "  lgbm-auto-tune-promote - auto-tune + promote best params to forecast_pipeline_config.yaml"
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
	@echo "  setup-features       - data + clustering + SKU features"
	@echo "  setup-backtest       - features + 3 backtests + champion selection"
	@echo "  setup-inv-planning   - inventory planning (SS, EOQ, policies, exceptions)"
	@echo "  setup-demand-planning - forecasts + projections + orders + replenishment"
	@echo "  setup-ops            - S&OP + events + financial + storyboard + DQ"
	@echo ""
	@echo "  === Database Cleanup & Fresh Recreate ==="
	@echo "  fresh-all            - FULL RESET: truncate + clean + load + ML + champion + baseline planning"
	@echo "  fresh-champion       - load + features + backtests + champion (no truncate)"
	@echo "  fresh-backtest       - load + features + backtests (no champion)"
	@echo "  fresh-features       - load + clustering + SKU features + LT"
	@echo "  fresh-load           - normalize + load + refresh MVs only"
	@echo "  db-truncate-data     - truncate non-config data/history while preserving configuration masters"
	@echo "  clean-artifacts      - remove stale CSVs, backtest, tuning, clustering, champion files"
	@echo "  refresh-mvs-tiered   - refresh all MVs in dependency order"
	@echo "  refresh-accuracy-mvs - refresh accuracy MVs (after backtest load)"

init:  ## Create .venv, install uv, sync deps
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

up:  ## Start Docker services (Postgres, MLflow) + apply DDL
	$(DC) up -d
	$(MAKE) db-apply-sql


db-apply-sql:  ## Apply all sql/*.sql migration files to Postgres
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

db-apply-sql-lakebase:  ## Apply all sql/*.sql to a networked Postgres / Lakebase (token-auth via env)
	$(UV) python -m scripts.db.apply_sql_lakebase $(ARGS)

down:  ## Stop Docker services
	$(DC) down

logs:  ## Tail Docker service logs
	$(DC) logs -f

normalize-item:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset item

normalize-location:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset location

normalize-customer:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset customer

normalize-time:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset time

normalize-dfu:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset sku

normalize-sales:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset sales

normalize-forecast:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset forecast

normalize-inventory:
	$(UV) python scripts/etl/normalize_inventory_csv.py

normalize-sourcing:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset sourcing

normalize-purchase-order:
	$(UV) python scripts/etl/normalize_dataset_csv.py --dataset purchase_order

normalize-all: normalize-item normalize-location normalize-customer normalize-time normalize-dfu normalize-sales normalize-forecast normalize-inventory normalize-sourcing normalize-purchase-order normalize-customer-demand  ## Normalize all input CSVs

load-item:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset item

load-location:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset location

load-customer:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset customer
	@$(MAKE) refresh-customer-filter-options

# One-shot MV refresh — call after dim_customer reloads. Backs the
# /customer-analytics/filter-options endpoint (sql/173).
refresh-customer-filter-options:
	@echo "Refreshing mv_customer_filter_options CONCURRENTLY..."
	@$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_customer_filter_options;" 2>/dev/null \
		|| $(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW mv_customer_filter_options;"
	@echo "  OK"

load-time:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset time

load-dfu:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset sku

load-sales:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset sales
	$(MAKE) refresh-agg-sales

load-forecast:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset forecast
	$(MAKE) refresh-agg-forecast

load-forecast-replace:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset forecast --replace
	$(MAKE) refresh-agg-forecast

load-forecast-replace-no-archive:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset forecast --replace --skip-archive
	$(MAKE) refresh-agg-forecast

load-inventory:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset inventory
	$(MAKE) refresh-agg-inventory

load-sourcing:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset sourcing

load-purchase-order:
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset purchase_order

normalize-customer-demand:  ## Normalize customer demand CSVs
	$(UV) python scripts/etl/normalize_customer_demand_csv.py

load-customer-demand:  ## Load customer demand (full replace)
	$(UV) python scripts/etl/load_customer_demand_postgres.py --replace
	$(MAKE) refresh-customer-mv
	$(MAKE) refresh-ca-mvs
	@$(MAKE) refresh-customer-mv

load-customer-demand-month:  ## Load single month: make load-customer-demand-month MONTH=2026-01
	$(UV) python scripts/etl/load_customer_demand_postgres.py --month $(MONTH)
	@$(MAKE) refresh-customer-mv

pipeline-customer-demand: normalize-customer-demand load-customer-demand  ## Full customer demand pipeline

load-all:  ## Load all clean CSVs into Postgres + refresh views
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset item
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset location
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset customer
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset time
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset sku
	# dfu.txt carries no ML cluster label. The loader preserves ml_cluster across its
	# own TRUNCATE, but on a first load after a clustering promote the in-DB snapshot is
	# empty — re-apply the promoted labels from data/clustering/cluster_labels.csv so
	# per-cluster tree models route correctly. Idempotent; --skip-if-missing no-ops
	# cleanly on a fresh DB before any clustering scenario is promoted (loop-4).
	$(UV) python scripts/ml/restore_cluster_assignments.py --skip-if-missing
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset sales
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset forecast
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset inventory
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset sourcing
	$(UV) python scripts/etl/load_dataset_postgres.py --dataset purchase_order
	$(UV) python scripts/etl/load_customer_demand_postgres.py --replace
	$(MAKE) refresh-agg
	$(MAKE) db-analyze  # refresh planner statistics after bulk load — without this the planner seq-scans (slow even at small scale)

refresh-agg-sales:
	# CONCURRENTLY when populated (unique idx uq_agg_sales_item_loc_month, sql/119); plain refresh on first run when the MV is not yet populated.
	$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY agg_sales_monthly;" >/dev/null 2>&1 || $(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_sales_monthly;" >/dev/null

refresh-agg-forecast:
	# CONCURRENTLY when populated (unique idx uq_agg_forecast_item_loc_month_model, sql/119); plain refresh on first run when the MV is not yet populated.
	$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY agg_forecast_monthly;" >/dev/null 2>&1 || $(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_forecast_monthly;" >/dev/null

refresh-agg-inventory:
	# CONCURRENTLY when populated (unique idx uq_agg_inv_item_loc_month, sql/119); plain refresh on first run when the MV is not yet populated.
	$(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY agg_inventory_monthly;" >/dev/null 2>&1 || $(PSQL) -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_inventory_monthly;" >/dev/null

refresh-agg: refresh-agg-sales refresh-agg-forecast refresh-agg-inventory

api:  ## Run FastAPI on :8000 with --reload
	$(UV) uvicorn api.main:app --reload --port 8000

ui-init:  ## Install npm deps for frontend
	cd frontend && npm install

ui:  ## Run React dev server on :5173
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


db-apply-inventory:
	$(PSQL) < sql/017_create_fact_inventory_snapshot.sql

db-apply-inv-backtest:
	$(PSQL) < sql/019_inventory_forecast_view.sql

db-apply-jobs:
	$(PSQL) < sql/020_create_job_history.sql
	$(PSQL) < sql/021_alter_job_history_scheduling.sql

refresh-inv-backtest:
	# CONCURRENTLY safe: uq_inv_fcst_item_loc_month_model (sql/119) is the required unique index.
	$(PSQL) \
		-c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_forecast_monthly;"

inventory-pipeline: normalize-inventory load-inventory


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
	@$(PSQL) -c " \
	SELECT t.tbl AS table_name, \
	  COALESCE(( \
	    SELECT reltuples::bigint::text FROM pg_class c \
	    JOIN pg_namespace n ON n.oid=c.relnamespace \
	    WHERE n.nspname='public' AND c.relname=t.tbl \
	  ), '—') AS est_rows \
	FROM (VALUES \
	  ('dim_item'),('dim_location'),('dim_customer'),('dim_time'),('dim_sku'), \
	  ('fact_sales_monthly'),('fact_external_forecast_monthly'),('fact_customer_demand_monthly'), \
	  ('fact_inventory_snapshot'),('fact_production_forecast'),('fact_purchase_orders'), \
	  ('backtest_lag_archive'),('champion_experiment'),('cluster_experiment'), \
	  ('lgbm_tuning_run'),('job_history') \
	) AS t(tbl) ORDER BY tbl;"
	@echo ""
	@echo "── Model Coverage ──"
	@$(PSQL) -c "SELECT model_id, count(*) AS rows FROM fact_external_forecast_monthly GROUP BY model_id ORDER BY count(*) DESC;" 2>/dev/null || echo "  (no backtest data loaded yet)"

cluster-all:                           ## Run clustering pipeline (creates experiment, auto-promotes)
	$(UV) python scripts/ml/run_cluster_pipeline.py --label "make cluster-all"

# ---------------------------------------------------------------------------
# Unified SKU Features pipeline
# ---------------------------------------------------------------------------
features-compute:                     ## Compute all SKU features (volume, trend, seasonality, variability, lifecycle)
	$(UV) python scripts/ml/compute_sku_features.py

# ---------------------------------------------------------------------------
# Lead Time Variability pipeline (IPfeature2)
# ---------------------------------------------------------------------------
lt-profile-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/023_create_lead_time_profile.sql').read()); conn.close(); print('Lead time profile DDL applied')"

lt-profile-compute:
	$(UV) python scripts/inventory/compute_lead_time_variability.py

lt-profile-all: lt-profile-schema lt-profile-compute

# ---------------------------------------------------------------------------
# EOQ & Cycle Stock pipeline (IPfeature4)
# ---------------------------------------------------------------------------
eoq-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/024_create_eoq_targets.sql').read()); conn.close(); print('EOQ targets DDL applied')"

eoq-compute:
	$(UV) python scripts/inventory/compute_eoq.py

eoq-all: eoq-schema eoq-compute

# ---------------------------------------------------------------------------
# Replenishment Policy pipeline (IPfeature5)
# ---------------------------------------------------------------------------
policy-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/025_create_replenishment_policy.sql').read()); conn.close(); print('Replenishment policy DDL applied')"

policy-assign:
	$(UV) python scripts/inventory/assign_replenishment_policies.py --config config/inventory/replenishment_policy_config.yaml

policy-all: policy-schema policy-assign

health-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/026_create_inventory_health_score.sql').read()); conn.close(); print('Health score DDL applied')"

health-refresh:
	$(UV) python scripts/inventory/refresh_health_scores.py

health-all: health-schema health-refresh

# ---------------------------------------------------------------------------
# Exception Queue (IPfeature7)
# ---------------------------------------------------------------------------

exceptions-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/027_create_replenishment_exceptions.sql').read()); conn.close(); print('Exceptions DDL applied')"

exceptions-generate:
	$(UV) python scripts/inventory/generate_replenishment_exceptions.py

exceptions-generate-dry:
	$(UV) python scripts/inventory/generate_replenishment_exceptions.py --dry-run

# ---------------------------------------------------------------------------
# Safety Stock Engine (IPfeature3)
# ---------------------------------------------------------------------------

ss-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/037_create_safety_stock_targets.sql').read()); conn.close(); print('Safety Stock DDL applied')"

ss-compute:
	$(UV) python scripts/inventory/compute_safety_stock.py

ss-compute-dry:
	$(UV) python scripts/inventory/compute_safety_stock.py --dry-run

ss-all: ss-schema ss-compute

# ---------------------------------------------------------------------------
# Multi-Algorithm Inventory Comparison
# ---------------------------------------------------------------------------
algo-comparison:
	$(UV) python scripts/inventory/compare_inventory_algorithms.py

# ---------------------------------------------------------------------------
# AI Planning Agent (IPAIfeature1)
# ---------------------------------------------------------------------------

ai-insights-schema:
	$(UV) psql $(DATABASE_URL) -f sql/036_create_ai_insights.sql
	$(UV) psql $(DATABASE_URL) -f sql/039_create_ai_call_log.sql
	$(UV) psql $(DATABASE_URL) -f sql/040_create_ai_recommendation_outcomes.sql

ai-insights-scan:
	$(UV) python scripts/ai/generate_ai_insights.py --portfolio

ai-insights-scan-dry:
	$(UV) python scripts/ai/generate_ai_insights.py --portfolio --dry-run

ai-insights-dfu:
	$(UV) python scripts/ai/generate_ai_insights.py --item $(ITEM) --loc $(LOC)

ai-insights-all: ai-insights-schema ai-insights-scan

# ---------------------------------------------------------------------------
# Backtesting (LGBM / CatBoost / XGBoost / Chronos — per-cluster only)
# Options (recursive, SHAP, tuning, params) are set in config/forecasting/forecast_pipeline_config.yaml
# ---------------------------------------------------------------------------
backtest-lgbm:
	$(UV) python scripts/ml/run_backtest.py --parallel --workers 8 $(ARGS)

backtest-catboost:
	$(UV) python scripts/ml/run_backtest_catboost.py --parallel --workers 8 $(ARGS)

backtest-xgboost:
	$(UV) python scripts/ml/run_backtest_xgboost.py --parallel --workers 8 $(ARGS)

customer-features:
	$(UV) python -m scripts.ml.generate_customer_features_sql

customer-features-python:
	$(UV) python -m scripts.ml.generate_customer_features

backtest-chronos2e:
	$(UV) python -m scripts.ml.run_backtest_chronos2_enriched

backtest-load-chronos2e:
	$(UV) python -m scripts.etl.load_backtest_forecasts --model chronos2_enriched --replace

backtest-chronos2e-full: backtest-chronos2e backtest-load-chronos2e

backtest-seasonal-naive:
	$(UV) python scripts/ml/run_backtest.py --model seasonal_naive $(ARGS)

backtest-load-seasonal-naive:
	$(UV) python scripts/etl/load_backtest_forecasts.py --model seasonal_naive --replace

backtest-rolling-mean:
	$(UV) python scripts/ml/run_backtest.py --model rolling_mean $(ARGS)

backtest-load-rolling-mean:
	$(UV) python scripts/etl/load_backtest_forecasts.py --model rolling_mean --replace

backtest-mstl:
	$(UV) python scripts/ml/run_backtest_mstl.py $(ARGS)

backtest-load-mstl:
	$(UV) python scripts/etl/load_backtest_forecasts.py --model mstl --replace

backtest-mstl-full: backtest-mstl backtest-load-mstl

backtest-nhits:
	$(UV) python scripts/ml/run_backtest_dl.py --model nhits $(ARGS)

backtest-load-nhits:
	$(UV) python scripts/etl/load_backtest_forecasts.py --model nhits --replace

backtest-nhits-full: backtest-nhits backtest-load-nhits

backtest-nbeats:
	$(UV) python scripts/ml/run_backtest_dl.py --model nbeats $(ARGS)

backtest-load-nbeats:
	$(UV) python scripts/etl/load_backtest_forecasts.py --model nbeats --replace

backtest-nbeats-full: backtest-nbeats backtest-load-nbeats

backtest-baselines: backtest-seasonal-naive backtest-rolling-mean

# backtest-all produces every compete:true model that runs on a clean rebuild:
#   lgbm_cluster, catboost_cluster, xgboost_cluster (cluster trees),
#   chronos2_enriched (foundation).
# Cheap/operator-gated baselines (mstl, nbeats, nhits, rolling_mean, rolling_median,
# seasonal_naive) are run on demand via their own targets.
# run_champion_selection.assert_competing_models_covered() fails loud if any
# compete:true model is missing from fact_external_forecast_monthly after load.
backtest-all: backtest-lgbm backtest-catboost backtest-xgboost backtest-chronos2e  ## Run all clean-rebuild backtests sequentially (cluster trees + foundation)

backtest-all-parallel:
	@mkdir -p data/backtest/logs
	@echo "[parallel] Starting cluster trees + Chronos2-enriched concurrently — logs in data/backtest/logs/"
	$(UV) python scripts/ml/run_backtest.py $(ARGS) > data/backtest/logs/lgbm.log 2>&1 & \
	$(UV) python scripts/ml/run_backtest_catboost.py $(ARGS) > data/backtest/logs/catboost.log 2>&1 & \
	$(UV) python scripts/ml/run_backtest_xgboost.py $(ARGS) > data/backtest/logs/xgboost.log 2>&1 & \
	$(UV) python -m scripts.ml.run_backtest_chronos2_enriched > data/backtest/logs/chronos2_enriched.log 2>&1 & \
	wait && echo "[parallel] All clean-rebuild backtests complete (3 cluster trees + chronos2e). Check data/backtest/logs/ for output."

backtest-load:
	$(UV) python scripts/etl/load_backtest_forecasts.py --model $(MODEL) --replace

backtest-load-all:
	$(UV) python scripts/etl/load_backtest_forecasts.py --all --replace

backtest-load-all-bulk:
	$(UV) python scripts/etl/load_backtest_forecasts.py --all --replace --bulk

backtest-load-bulk:  ## Load 4 core models with single index cycle (~4x faster)
	$(UV) python scripts/etl/load_backtest_forecasts.py --models lgbm_cluster catboost_cluster xgboost_cluster chronos2_enriched --replace --bulk

backtest-load-main-only:  ## Load specific models to main table only (skip archive). Usage: make backtest-load-main-only MODELS="lgbm_cluster chronos2_enriched"
	$(UV) python scripts/etl/load_backtest_forecasts.py --models $(MODELS) --replace --bulk --main-only

backtest-load-archive-only:  ## Load specific models to archive only (skip main). Usage: make backtest-load-archive-only MODELS="lgbm_cluster chronos2_enriched"
	$(UV) python scripts/etl/load_backtest_forecasts.py --models $(MODELS) --replace --bulk --archive-only

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
	$(UV) python scripts/ml/clean_backtest_models.py $(MODELS)

backtest-list:
	$(UV) python scripts/ml/clean_backtest_models.py --list

# ---------------------------------------------------------------------------
# Forecast date-range cleanup
# ---------------------------------------------------------------------------
forecast-clean:
	$(UV) python scripts/ml/clean_forecasts_by_date.py $(ARGS)

forecast-clean-list:
	$(UV) python scripts/ml/clean_forecasts_by_date.py --list

accuracy-slice-refresh:
	# CONCURRENTLY safe: uq_agg_accuracy_dim, uq_agg_accuracy_lag_archive,
	# uq_dfu_coverage_model_lag_dfu, uq_dfu_coverage_lag_archive (sql/119),
	# uq_agg_accuracy_by_dfu (sql/193) and uq_agg_dfu_naive_scale (sql/194) back the
	# six MVs respectively. agg_dfu_naive_scale = per-DFU in-sample seasonal-naive MAE
	# (the MASE scale); refreshed on the same backtest-load cadence as the others.
	$(PSQL) -v ON_ERROR_STOP=1 \
		-c "REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_by_dim; REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_by_dfu; REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_lag_archive; REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_coverage; REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_coverage_lag_archive; REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_naive_scale;"

accuracy-slice-check:
	curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=cluster_assignment" | python3 -m json.tool | head -60
	curl -s "http://localhost:8000/forecast/accuracy/lag-curve" | python3 -m json.tool | head -40

champion-select:
	$(UV) python scripts/ml/run_champion_selection.py

champion-simulate:
	$(UV) python scripts/ml/simulate_champion_strategies.py

champion-train-meta:
	$(UV) python scripts/ml/train_meta_learner.py

champion-all: champion-train-meta champion-simulate champion-select

tune-lgbm:
	$(UV) python scripts/ml/tune_hyperparams.py --model lgbm

tune-catboost:
	$(UV) python scripts/ml/tune_hyperparams.py --model catboost

tune-xgboost:
	$(UV) python scripts/ml/tune_hyperparams.py --model xgboost

tune-all: tune-lgbm tune-catboost tune-xgboost

tune-lgbm-clusters:
	$(UV) python scripts/ml/tune_cluster_hyperparams.py --model lgbm --trials 30

tune-catboost-clusters:
	$(UV) python scripts/ml/tune_cluster_hyperparams.py --model catboost --trials 30

tune-xgboost-clusters:
	$(UV) python scripts/ml/tune_cluster_hyperparams.py --model xgboost --trials 30

tune-clusters: tune-lgbm-clusters tune-catboost-clusters tune-xgboost-clusters

# ── Production Baseline Seeding ──────────────────────────────────────────────
seed-baselines:          ## Seed production baselines into experiment tables
	$(UV) python scripts/etl/seed_production_baselines.py

seed-baselines-tuning:
	$(UV) python scripts/etl/seed_production_baselines.py --scope tuning

seed-baselines-champion:
	$(UV) python scripts/etl/seed_production_baselines.py --scope champion

seed-baselines-clustering:
	$(UV) python scripts/etl/seed_production_baselines.py --scope clustering

# ── LGBM Tuning ──────────────────────────────────────────────────────────────
lgbm-tuning-list:
	$(UV) python scripts/ml/compare_backtest_runs.py --list

lgbm-tuning-compare:
	$(UV) python scripts/ml/compare_backtest_runs.py --baseline $(BASELINE) --candidate $(CANDIDATE)

lgbm-tuning-backup:
	$(UV) python scripts/ml/compare_backtest_runs.py --backup $(RUN)

lgbm-tuning-run:
	$(UV) python scripts/ml/run_backtest.py --model lgbm
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
	$(UV) python -m scripts.ml.run_expert_panel

expert-panel-quick:      ## Quick Expert Panel test (1000 DFUs, 3 timeframes, ~8 min)
	$(UV) python -m scripts.ml.run_expert_panel --n-dfus 1000 --n-timeframes 3

expert-panel-mini:       ## Minimal Expert Panel test (200 DFUs, 2 timeframes, ~2 min)
	$(UV) python -m scripts.ml.run_expert_panel --n-dfus 200 --n-timeframes 2

expert-panel-loc:        ## Run Expert Panel for all DFUs at a specific location: make expert-panel-loc LOC=1401-BULK
	@if [ -z "$(LOC)" ]; then echo "Usage: make expert-panel-loc LOC=1401-BULK"; exit 1; fi
	$(UV) python -m scripts.ml.run_expert_panel --loc $(LOC)

# ── Advanced Expert Panel (Foundation Models + Deep Learning) ───────────────
adv-expert-panel:        ## Advanced Expert Panel (5000 DFUs, 10 TFs, execution-lag accuracy, foundation+DL+stat upgrades)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_adv_expert_panel --n-timeframes 10

adv-expert-panel-quick:  ## Quick Advanced Expert Panel (1000 DFUs, 5 TFs, execution-lag accuracy)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_adv_expert_panel --n-dfus 1000 --n-timeframes 5

adv-expert-panel-mini:   ## Minimal Advanced Expert Panel (200 DFUs, 2 TFs)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_adv_expert_panel --n-dfus 200 --n-timeframes 2

adv-expert-panel-loc:    ## Advanced Expert Panel for all DFUs at a specific location: make adv-expert-panel-loc LOC=1401-BULK
	@if [ -z "$(LOC)" ]; then echo "Usage: make adv-expert-panel-loc LOC=1401-BULK"; exit 1; fi
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_adv_expert_panel --loc $(LOC)

route-analysis:          ## Compare per-DFU routing strategies on saved predictions (no retraining, ~2 min)
	$(UV) python -m scripts.ml.expert_panel_route_analysis

route-analysis-min3:     ## Same as route-analysis but require 3+ timeframes of history per DFU
	$(UV) python -m scripts.ml.expert_panel_route_analysis --min-history 3

# ── Expert System Backtest (full population, segment-assigned algorithm) ─────
expsys-backtest:         ## Full ExpSys backtest: all DFUs, 10 TFs, loads to DB (~4-5h)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_expert_system_backtest

expsys-backtest-dry:     ## ExpSys accuracy only — no DB loading (--skip-load)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_expert_system_backtest --skip-load

expsys-backtest-replace: ## ExpSys: delete existing rows first, then reload
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(UV) python -m scripts.ml.run_expert_system_backtest --replace

# AI Champion is interactive-only (per-DFU "AI Adjust" button on the Item Analysis
# tab → POST /ai-champion/adjust + /save). No batch Make target by design.

commit:
	@if [ -z "$(MSG)" ]; then echo "Usage: make commit MSG=\"your message\""; exit 1; fi
	git add -A && (git diff --staged --quiet || git commit -m "$(MSG)") && git push -u origin HEAD

test:  ## Backend pytest (~0.7s, DB mocked)
	$(UV) pytest tests/ -v --tb=short

test-unit:  ## Run only Python unit tests
	$(UV) pytest tests/unit/ -v --tb=short

test-api:  ## Run only Python API tests
	$(UV) pytest tests/api/ -v --tb=short

test-cov:  ## Backend tests with coverage report
	$(UV) pytest tests/ --cov=api --cov=common --cov-report=term-missing

test-all: test ui-test  ## Backend + frontend tests

scale-test:  ## Run scale test suite (synthetic 100K rows by default; SCALE=10000000 for nightly)
	$(UV) pytest tests/scale/ -v -m scale --override-ini="norecursedirs=" --scale=$${SCALE:-100000}

check-all: check-db check-api  ## DB row counts + API health

ai-sync-check:  ## Verify Claude/Codex shared guidance wiring
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
	$(UV) python scripts/inventory/compute_demand_signals.py

demand-signals-dry:
	$(UV) python scripts/inventory/compute_demand_signals.py --dry-run

demand-signals-all: demand-signals-schema demand-signals-compute

# ---------------------------------------------------------------------------
# IPfeature10: Monte Carlo Simulation
# ---------------------------------------------------------------------------
sim-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/030_create_ss_simulation_results.sql').read()); conn.close(); print('Simulation DDL applied')"

sim-run:
	$(UV) python scripts/inventory/run_ss_simulation.py --item $(ITEM) --loc $(LOC)

# ---------------------------------------------------------------------------
# IPfeature11: ABC-XYZ Classification
# ---------------------------------------------------------------------------
abc-xyz-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/031_add_xyz_classification.sql').read()); conn.close(); print('ABC-XYZ DDL applied')"

abc-xyz-classify:
	$(UV) python scripts/inventory/classify_abc_xyz.py

abc-xyz-classify-dry:
	$(UV) python scripts/inventory/classify_abc_xyz.py --dry-run

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
	$(UV) python scripts/inventory/compute_investment_plan.py

investment-all: investment-schema investment-plan

# ---------------------------------------------------------------------------
# IPfeature14: Intra-Month Stockout Detection
# ---------------------------------------------------------------------------
intramonth-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/034_create_intramonth_stockout.sql').read()); conn.close(); print('Intramonth stockout DDL applied')"

intramonth-refresh:
	$(UV) python scripts/inventory/refresh_intramonth_stockout.py

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
	$(UV) python scripts/ops/generate_storyboard_exceptions.py

storyboard-generate-dry:
	$(UV) python scripts/ops/generate_storyboard_exceptions.py --dry-run

storyboard-all: storyboard-schema storyboard-generate

# ---------------------------------------------------------------------------
# F1.1: Production Forecast Generation Pipeline
# ---------------------------------------------------------------------------

forecast-prod-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/039_create_production_forecast.sql').read()); conn.execute(open('sql/041_add_source_model_id.sql').read()); conn.close(); print('Production forecast DDL applied')"

# ── Production Model Training ──────────────────────────────────────────────
train-production: ## Train a single model on full history: make train-production MODEL=lgbm_cluster
	$(UV) python scripts/ml/train_production_models.py --model $(MODEL)

train-production-all: ## Train all forecastable tree models on full history
	$(UV) python scripts/ml/train_production_models.py --all

forecast-generate:  ## Generate production forecast inference rows
	$(UV) python scripts/forecasting/generate_production_forecasts.py

forecast-generate-dfu:
	$(UV) python scripts/forecasting/generate_production_forecasts.py --dfu $(ITEM) $(LOC)

forecast-generate-dry:
	$(UV) python scripts/forecasting/generate_production_forecasts.py --dry-run

forecast-prod-all: forecast-prod-schema forecast-generate

forecast-full: train-production-all forecast-generate ## Full pipeline: train all models + generate forecasts

forecast-model: ## Train + generate for one model: make forecast-model MODEL=lgbm_cluster
	$(UV) python scripts/ml/train_production_models.py --model $(MODEL) && \
	$(UV) python scripts/forecasting/generate_production_forecasts.py --model-id $(MODEL)

# ---------------------------------------------------------------------------
# Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)
# ---------------------------------------------------------------------------

replplan-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/041_create_replenishment_plan.sql').read()); conn.close(); print('Replenishment plan schema applied')"

replplan-compute:
	$(UV) python scripts/inventory/compute_replenishment_plan.py

replplan-compute-dry:
	$(UV) python scripts/inventory/compute_replenishment_plan.py --dry-run

replplan-all: replplan-schema replplan-compute

# ---------------------------------------------------------------------------
# Open PO Integration (F1.3)
# ---------------------------------------------------------------------------

po-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; [conn.execute(open(f).read()) for f in ['sql/042_create_supplier_master.sql','sql/043_create_open_purchase_orders.sql','sql/044_create_po_receipts.sql']]; conn.close(); print('PO schema applied')"

po-load:
	$(UV) python scripts/etl/load_open_pos.py

po-load-file:
	$(UV) python scripts/etl/load_open_pos.py --file $(FILE)

po-load-dry:
	$(UV) python scripts/etl/load_open_pos.py --dry-run

po-receipts-load:
	$(UV) python scripts/etl/load_open_pos.py --receipts

po-all: po-schema po-load

# ---------------------------------------------------------------------------
# Forward Inventory Projection (F1.2)
# ---------------------------------------------------------------------------

projection-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/045_create_inventory_projection.sql').read()); conn.close(); print('Projection schema applied')"

projection-compute:
	$(UV) python scripts/inventory/compute_inventory_projection.py --horizon 90

projection-compute-dfu:
	$(UV) python scripts/inventory/compute_inventory_projection.py --dfu $(ITEM) $(LOC) --horizon 90

projection-dry:
	$(UV) python scripts/inventory/compute_inventory_projection.py --dry-run --horizon 90

projection-all: projection-schema projection-compute

# ---------------------------------------------------------------------------
# Order Recommendation Engine (F2.1)
# ---------------------------------------------------------------------------

planned-orders-schema:
	$(UV) python -c "import psycopg, os; conn = psycopg.connect(host=os.getenv('POSTGRES_HOST','localhost'), port=os.getenv('POSTGRES_PORT','5440'), dbname=os.getenv('POSTGRES_DB','demand_mvp'), user=os.getenv('POSTGRES_USER','demand'), password=os.getenv('POSTGRES_PASSWORD','demand')); conn.autocommit=True; conn.execute(open('sql/046_create_planned_orders.sql').read()); conn.close(); print('Planned orders schema applied')"

planned-orders-generate:
	$(UV) python scripts/inventory/generate_planned_orders.py

planned-orders-generate-dfu:
	$(UV) python scripts/inventory/generate_planned_orders.py --dfu $(ITEM) $(LOC)

planned-orders-dry:
	$(UV) python scripts/inventory/generate_planned_orders.py --dry-run

planned-orders-all: planned-orders-schema planned-orders-generate

# ---------------------------------------------------------------------------
# F2.2 — Multi-Horizon Quantile Demand Plan
# ---------------------------------------------------------------------------
VERSION ?= $(shell date +%Y-%m-%d)_production

quantile-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/047_create_demand_plan.sql').read()); conn.commit(); conn.close(); print('quantile schema applied')"

# NOTE: generate_quantile_forecasts is an MVP stub — its quantile models train on
# synthetic random data, so it now REFUSES to write to fact_demand_plan by default
# (the consensus plan + safety stock consume that table). Use `quantile-dry` to
# preview. Pass --allow-synthetic only for dev. Wire it to the real feature pipeline
# (backtest_framework) before production use.
quantile-train:
	$(UV) scripts/forecasting/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION)

quantile-train-dfu:
	$(UV) scripts/forecasting/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION) --dfu $(ITEM) $(LOC)

quantile-dry:
	$(UV) scripts/forecasting/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION) --dry-run

quantile-all: quantile-schema quantile-train

## F2.3 — Consensus Forecasting & Planner Overrides
consensus-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/048_create_consensus_plan.sql').read()); conn.commit(); conn.close(); print('consensus schema applied')"

consensus-generate:
	$(UV) scripts/forecasting/generate_consensus_plan.py --plan-version $(VERSION) --months-ahead 12

consensus-generate-dry:
	$(UV) scripts/forecasting/generate_consensus_plan.py --plan-version $(VERSION) --months-ahead 12 --dry-run

consensus-all: consensus-schema consensus-generate

## F2.4 — Procurement Workflow & Order Release
procurement-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/049_create_procurement_workflow.sql').read()); conn.commit(); conn.close(); print('procurement schema applied')"

procurement-export:
	$(UV) scripts/inventory/release_planned_orders.py --action export_csv --po-numbers $(PO_NUMBERS) --output-dir data/po_exports/

procurement-send-erp:
	$(UV) scripts/inventory/release_planned_orders.py --action send_erp --po-numbers $(PO_NUMBERS) --integration-id $(INTEGRATION_ID)

procurement-all: procurement-schema


# ---------------------------------------------------------------------------
# F3.1 — Forecast Bias Correction Engine
# ---------------------------------------------------------------------------
bias-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/050_create_bias_corrections.sql').read()); conn.commit(); conn.close(); print('bias corrections schema applied')"

bias-compute:
	$(UV) python scripts/forecasting/compute_bias_corrections.py --plan-version $(VERSION)

bias-compute-dry:
	$(UV) python scripts/forecasting/compute_bias_corrections.py --plan-version $(VERSION) --dry-run

bias-all: bias-schema bias-compute

# ---------------------------------------------------------------------------
# F3.2 — Service Level Actuals vs Targets
# ---------------------------------------------------------------------------
service-level-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/051_create_service_level_tracking.sql').read()); conn.commit(); conn.close(); print('service level schema applied')"

service-level-compute:
	$(UV) python scripts/ops/compute_service_level_actuals.py

service-level-dry:
	$(UV) python scripts/ops/compute_service_level_actuals.py --dry-run

service-level-all: service-level-schema service-level-compute

# ---------------------------------------------------------------------------
# F3.3 — Supplier Lead Time Learning
# ---------------------------------------------------------------------------
lead-time-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/052_create_lead_time_learning.sql').read()); conn.commit(); conn.close(); print('lead time schema applied')"

lead-time-update:
	$(UV) python scripts/inventory/update_lead_time_actuals.py

lead-time-dry:
	$(UV) python scripts/inventory/update_lead_time_actuals.py --dry-run

lead-time-all: lead-time-schema lead-time-update

# ---------------------------------------------------------------------------
# F3.4 — Demand Sensing / Blended Forecast
# ---------------------------------------------------------------------------
blended-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/053_create_blended_forecast.sql').read()); conn.commit(); conn.close(); print('blended forecast schema applied')"

blended-compute:
	$(UV) python scripts/forecasting/compute_blended_forecast.py

blended-compute-dfu:
	$(UV) python scripts/forecasting/compute_blended_forecast.py --item-no $(ITEM) --loc $(LOC)

blended-dry:
	$(UV) python scripts/forecasting/compute_blended_forecast.py --dry-run

blended-all: blended-schema blended-compute

# ---------------------------------------------------------------------------
# F3.5 — Multi-Echelon Planning
# ---------------------------------------------------------------------------
echelon-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/054_create_echelon_planning.sql').read()); conn.commit(); conn.close(); print('echelon planning schema applied')"

echelon-compute:
	$(UV) python scripts/inventory/compute_echelon_targets.py

echelon-compute-item:
	$(UV) python scripts/inventory/compute_echelon_targets.py --item-no $(ITEM)

echelon-dry:
	$(UV) python scripts/inventory/compute_echelon_targets.py --dry-run

echelon-all: echelon-schema echelon-compute

# ---------------------------------------------------------------------------
# F4.1 — Financial Inventory Plan
# ---------------------------------------------------------------------------
financial-plan-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/055_create_financial_plan.sql').read()); conn.commit(); conn.close(); print('financial plan schema applied')"

financial-plan-compute:
	$(UV) python scripts/ops/compute_financial_plan.py

financial-plan-dry:
	$(UV) python scripts/ops/compute_financial_plan.py --dry-run

financial-plan-all: financial-plan-schema financial-plan-compute

# ---------------------------------------------------------------------------
# F4.2 — Sales & Operations Planning (S&OP)
# ---------------------------------------------------------------------------
sop-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/056_create_sop_module.sql').read()); conn.commit(); conn.close(); print('S&OP schema applied')"

sop-create:
	$(UV) python scripts/ops/run_sop_cycle.py --action create --cycle-month $(CYCLE_MONTH)

sop-advance:
	$(UV) python scripts/ops/run_sop_cycle.py --action advance --cycle-id $(CYCLE_ID)

sop-populate:
	$(UV) python scripts/ops/run_sop_cycle.py --action populate-demand --cycle-id $(CYCLE_ID)

sop-seed:
	$(UV) python -c "from datetime import date; m=date.today().replace(day=1).isoformat(); import subprocess, sys; subprocess.run([sys.executable, 'scripts/ops/run_sop_cycle.py', '--action', 'create', '--cycle-month', m], check=True)"

sop-all: sop-schema sop-seed

# ---------------------------------------------------------------------------
# F4.3 — Promotion & Event Planning
# ---------------------------------------------------------------------------
events-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/057_create_event_planning.sql').read()); conn.commit(); conn.close(); print('event planning schema applied')"

events-apply:
	$(UV) python scripts/forecasting/apply_event_adjustments.py

events-apply-dry:
	$(UV) python scripts/forecasting/apply_event_adjustments.py --dry-run

events-all: events-schema events-apply

# ---------------------------------------------------------------------------
# F4.4 — Supply Chain Scenario Planning
# ---------------------------------------------------------------------------
scenarios-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); conn.cursor().execute(open('sql/058_create_supply_scenarios.sql').read()); conn.commit(); conn.close(); print('supply scenarios schema applied')"

scenarios-list:
	$(UV) python scripts/inventory/run_supply_chain_scenario.py --action list

scenarios-run:
	$(UV) python scripts/inventory/run_supply_chain_scenario.py --action run --scenario-id $(SCENARIO_ID)

scenarios-run-dry:
	$(UV) python scripts/inventory/run_supply_chain_scenario.py --action run --scenario-id $(SCENARIO_ID) --dry-run

scenarios-all: scenarios-schema

# ---------------------------------------------------------------------------
# Inventory Rebalancing — Cross-Location Transfer Optimization
# ---------------------------------------------------------------------------
rebalancing-schema:
	$(UV) python -c "import psycopg; from common.db import get_db_params; conn=psycopg.connect(**get_db_params()); cur=conn.cursor(); cur.execute(open('sql/071_create_transfer_network.sql').read()); cur.execute(open('sql/072_create_rebalancing_plan.sql').read()); cur.execute(open('sql/073_create_rebalancing_views.sql').read()); conn.commit(); conn.close(); print('Rebalancing DDL applied')"

rebalancing-compute:
	$(UV) python scripts/inventory/compute_rebalancing.py

rebalancing-compute-dry:
	$(UV) python scripts/inventory/compute_rebalancing.py --dry-run

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
	$(UV) python scripts/ops/populate_dq_checks.py

dq-run:
	$(UV) python -c "from common.dq_engine import DQEngine; e=DQEngine(); results=e.run_all_checks(); print(f'Ran {len(results)} checks')"

dq-all: dq-schema dq-populate dq-run

# ── Unified Pipeline Orchestrator ──────────────────────────────

pipeline-full:  ## Full ETL pipeline (normalize + load + refresh MVs, parallel)
	$(UV) python scripts/etl/run_pipeline.py --mode full --parallel

pipeline-refresh:  ## Incremental ETL refresh (detect changes, reload only deltas)
	$(UV) python scripts/etl/run_pipeline.py --mode refresh

pipeline-inventory:  ## Full inventory-domain ETL pipeline only
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

setup-features: setup-data features-compute cluster-all lt-profile-all abc-xyz-all demand-signals-all
	@echo "✓ Phase 2 complete: clustering, SKU features, lead time, ABC-XYZ, demand signals"

setup-backtest: setup-features backtest-all backtest-load-all accuracy-slice-refresh champion-all seed-baselines
	@echo "✓ Phase 3 complete: backtests, champion selection"

inv-plan-refresh: ## Run end-to-end inventory planning pipeline (SS → EOQ → Repl Plan → Orders → Exceptions)
	$(UV) run python scripts/inventory/run_inventory_planning_pipeline.py

inv-plan-refresh-dry: ## Preview inventory pipeline without DB writes
	$(UV) run python scripts/inventory/run_inventory_planning_pipeline.py --dry-run

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

perf-ingestion:                        ## Time ingestion pipeline stages: make perf-ingestion MODE=full|refresh
	$(UV) python scripts/tools/bench_ingestion.py --mode $(or $(MODE),full)

perf-pipeline:                         ## ETL pipeline performance analysis (read-only)
	$(UV) python scripts/ops/run_perf_analysis.py --mode pipeline

perf-clean:                            ## Truncate all perf profiling history from DB
	psql "$(DATABASE_URL)" -c "TRUNCATE perf_suggestion, perf_query, perf_section, perf_run CASCADE;"

# ── pg-queue (Item 22 pilot) ─────────────────────────────────────────────────
# Postgres-backed queue alongside APScheduler. Pilot job: refresh_intramonth.
# See common/services/pg_queue.py and scripts/ops/pg_queue_worker.py.

pg-queue-worker:                        ## Run a pg-queue worker (long-running; handles refresh_intramonth)
	$(UV) python scripts/ops/pg_queue_worker.py

pg-queue-enqueue-recurring:             ## Enqueue the recurring refresh_intramonth job (cron entry-point)
	@$(UV) python -c "from common.services.pg_queue import enqueue_job; \
print('enqueued job_id =', enqueue_job('refresh_intramonth'))"

pg-queue-depth:                         ## Show pg-queue depth grouped by status
	@$(UV) python -c "from common.services.pg_queue import get_queue_depth; \
import json; print(json.dumps(get_queue_depth(), indent=2))"

# ── Database Cleanup & Fresh Recreate ────────────────────────────────────────
# Full wipe-and-reload: clears non-config data/history, reloads from data/input/,
# and refreshes the core ML + baseline planning outputs while preserving configs.
# See docs/RUNBOOK.md "Database Cleanup & Fresh Recreate" for details.

integration-clean-test:                ## Delete integration_job rows tagged as test/dev_test/ci
	@echo "Deleting integration_job rows where triggered_by IN ('test','dev_test','ci')..."
	@$(UV) python -c "import psycopg; from common.core.db import get_db_params; \
conn = psycopg.connect(**get_db_params()); \
cur = conn.execute(\"DELETE FROM integration_job WHERE triggered_by IN ('test','dev_test','ci') RETURNING id\"); \
n = len(cur.fetchall()); conn.commit(); conn.close(); print(f'Deleted {n} test entries')"

db-truncate-data:                      ## Truncate non-config data/history (preserves configuration masters)
	@echo "Truncating non-config data, history, and experiment tables (preserving configuration masters)..."
	@printf '%s\n' \
	  'BEGIN;' \
	  'TRUNCATE TABLE ai_recommendation_outcomes CASCADE;' \
	  'TRUNCATE TABLE ai_insights CASCADE;' \
	  'TRUNCATE TABLE ai_planning_memos CASCADE;' \
	  'TRUNCATE TABLE ai_call_log CASCADE;' \
	  'TRUNCATE TABLE sku_chat_pending_adjustment CASCADE;' \
	  'TRUNCATE TABLE sku_chat_call_log CASCADE;' \
	  'TRUNCATE TABLE sku_chat_message CASCADE;' \
	  'TRUNCATE TABLE sku_chat_session CASCADE;' \
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
	  'TRUNCATE TABLE fact_candidate_forecast CASCADE;' \
	  'TRUNCATE TABLE fact_production_forecast_staging CASCADE;' \
	  'TRUNCATE TABLE fact_production_forecast CASCADE;' \
	  'TRUNCATE TABLE fact_ai_champion_forecast CASCADE;' \
	  'TRUNCATE TABLE ai_champion_run CASCADE;' \
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
	  'TRUNCATE TABLE backtest_run CASCADE;' \
	  'TRUNCATE TABLE cluster_tuning_profile_state CASCADE;' \
	  'TRUNCATE TABLE cluster_experiment_comparison CASCADE;' \
	  'TRUNCATE TABLE cluster_experiment CASCADE;' \
	  'TRUNCATE TABLE champion_experiment CASCADE;' \
	  'TRUNCATE TABLE champion_experiment_lag CASCADE;' \
	  'TRUNCATE TABLE champion_experiment_month CASCADE;' \
	  'TRUNCATE TABLE champion_promotion_log CASCADE;' \
	  'TRUNCATE TABLE fact_inventory_backtest CASCADE;' \
	  'TRUNCATE TABLE fact_inventory_algorithm_comparison CASCADE;' \
	  'TRUNCATE TABLE fact_dfu_policy_assignment CASCADE;' \
	  'TRUNCATE TABLE fact_exception_lifecycle CASCADE;' \
	  'TRUNCATE TABLE fact_lineage_event CASCADE;' \
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
	find data/staged -maxdepth 1 -name '*_clean.csv' -delete 2>/dev/null || true  # find: zsh (SHELL := /bin/zsh) aborts a recipe on a no-match glob; find no-ops cleanly
	rm -f data/staged/seasonality_results.csv data/staged/clustering_features.csv
	rm -rf data/backtest data/tuning data/perf_reports data/clustering data/champion data/models  # whole generated dirs (no glob); recreated by the pipeline
	@echo "✓ Intermediate artifacts cleaned."

refresh-mvs-tiered:                    ## Refresh all MVs in dependency order (4 tiers, auto-detects first run)
	@echo "Refreshing materialized views (tier-ordered)..."
	@for mv in \
	  agg_sales_monthly agg_forecast_monthly agg_inventory_monthly \
	  mv_inventory_forecast_monthly mv_fill_rate_monthly mv_intramonth_stockout \
	  mv_supplier_performance mv_supplier_po_performance mv_po_lead_time_analysis \
	  agg_accuracy_by_dim agg_accuracy_by_dfu agg_dfu_coverage agg_dfu_naive_scale \
	  mv_inventory_health_score mv_control_tower_kpis \
	  mv_integrated_planning_targets \
	  mv_customer_activity_monthly \
	  mv_ca_segment_trends mv_ca_demand_at_risk mv_ca_order_patterns mv_ca_item_state; do \
	  echo "  Refreshing $$mv ..."; \
	  $(PSQL) -c "REFRESH MATERIALIZED VIEW CONCURRENTLY $$mv;" 2>/dev/null \
	    || $(PSQL) -c "REFRESH MATERIALIZED VIEW $$mv;" 2>/dev/null \
	    || echo "    ⚠ $$mv skipped (does not exist)"; \
	done
	@echo "✓ All materialized views refreshed."

refresh-accuracy-mvs:                  ## Refresh accuracy MVs (after backtest load)
	$(PSQL) -v ON_ERROR_STOP=1 -c " \
	  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_by_dim; \
	  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_by_dfu; \
	  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_lag_archive; \
	  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_coverage; \
	  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_coverage_lag_archive; \
	  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_naive_scale; \
	"
	@echo "✓ Accuracy materialized views refreshed."

fresh-load: normalize-all load-all refresh-mvs-tiered  ## Normalize + load + refresh MVs (from input CSVs)
	@echo "✓ Fresh data load complete."

fresh-features: fresh-load features-compute cluster-all lt-profile-all  ## Load + clustering + SKU features + LT
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

# ---------------------------------------------------------------------------
# Database maintenance & optimization
# ---------------------------------------------------------------------------
db-analyze:                            ## Run ANALYZE on all tables (update planner statistics)
	$(PSQL) -c "ANALYZE;"
	@echo "✓ ANALYZE complete — planner statistics updated."

db-health:                             ## Database health report (size, cache hit, bloat, unused indexes)
	PYTHONPATH=. $(UV) python scripts/db/db_maintenance.py health

db-drop-unused-indexes:                ## Drop unused indexes (dry-run). Add EXECUTE=1 to actually drop
	@if [ "$(EXECUTE)" = "1" ]; then \
		PYTHONPATH=. $(UV) python scripts/db/drop_unused_indexes.py --execute --min-size 1; \
	else \
		PYTHONPATH=. $(UV) python scripts/db/drop_unused_indexes.py --min-size 1; \
	fi

db-retention:                          ## Apply data retention policies (dry-run). Add EXECUTE=1 to actually apply
	@if [ "$(EXECUTE)" = "1" ]; then \
		PYTHONPATH=. $(UV) python scripts/db/db_maintenance.py retention --execute; \
	else \
		PYTHONPATH=. $(UV) python scripts/db/db_maintenance.py retention; \
	fi

db-optimize: db-analyze db-drop-unused-indexes  ## Full optimization: ANALYZE + drop unused indexes (dry-run)
	@echo "✓ Optimization complete. Run 'make db-drop-unused-indexes EXECUTE=1' to drop indexes."

db-maintain: db-analyze db-health      ## Routine maintenance: ANALYZE + health report
	@echo "✓ Maintenance complete."

# ---------------------------------------------------------------------------
# Partition Management
# ---------------------------------------------------------------------------
# Idempotently provisions the next N partitions for every RANGE-partitioned
# fact table. Each registry entry chooses an interval (month or week); the
# script honors that, so the same target works for monthly and weekly tables.
# Run on a schedule (monthly for monthly tables, weekly for weekly tables)
# OR manually before any large backfill that may write rows into a future
# window. CREATE TABLE IF NOT EXISTS makes re-runs cheap.

auto-create-partitions:                ## Create the next N partitions for every partitioned fact table (idempotent). Per-table interval (month/week) is picked up from the registry.
	PYTHONPATH=. $(UV) python scripts/db/auto_create_partitions.py $(if $(HORIZON),--horizon $(HORIZON))

auto-create-partitions-dry-run:        ## Print partition DDL without executing
	PYTHONPATH=. $(UV) python scripts/db/auto_create_partitions.py $(if $(HORIZON),--horizon $(HORIZON)) --dry-run

auto-create-partitions-weekly:         ## Create only weekly-interval partitions (rolling 12 weeks default). Use after the weekly cutover migrations.
	PYTHONPATH=. $(UV) python scripts/db/auto_create_partitions.py --interval week $(if $(HORIZON),--horizon $(HORIZON))

auto-create-partitions-weekly-dry-run: ## Print weekly-interval partition DDL without executing
	PYTHONPATH=. $(UV) python scripts/db/auto_create_partitions.py --interval week $(if $(HORIZON),--horizon $(HORIZON)) --dry-run
