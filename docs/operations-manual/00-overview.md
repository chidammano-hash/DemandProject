# Supply Chain Command Center — Operations Manual

End-to-end runbook for populating, operating, and maintaining every module of the platform. Each section is self-contained but assumes prior sections have been executed.

---

## Read in Order (First-Time Setup)

| # | Section | What you accomplish |
|---|---|---|
| 1 | [Setup & Environment](01-setup-environment.md) | Fresh clone → working dev (Docker, Postgres, env vars, schema, sanity) |
| 2 | [Data Ingestion (ETL)](02-data-ingestion.md) | Raw CSVs → 11 normalized domains in Postgres + refreshed MVs |
| 3 | [SKU Features & Clustering](03-sku-features-clustering.md) | `dim_sku` features computed → cluster experiment promoted → assignment table current |
| 4 | [Forecasting Backtests](04-forecasting-backtest.md) | All model families backtested → predictions in `fact_candidate_forecast` + archive |
| 5 | [Tuning & Champion Selection](05-tuning-champion-selection.md) | Bayesian tuning + champion DFU assignments at `data/champion/dfu_assignments.csv` |
| 6 | [Production Forecasting](06-production-forecasting.md) | Generate + promote → `fact_production_forecast` populated, served by `/forecast/*` |
| 7 | [Inventory Planning](07-inventory-planning.md) | Safety stock, EOQ, policies, exceptions, projections, KPIs |
| 8 | [Operations: S&OP, Control Tower, DQ, Jobs](08-operations-sop-control-tower.md) | S&OP cycle, control tower KPIs, events, storyboard, DQ, scheduler |
| 9 | [AI / LLM Intelligence](09-ai-intelligence.md) | AI Planner, SKU Chatbot (Claude Agent SDK), Market Intel, embeddings, tuning chat, admin LLM reset |
| 10 | [Frontend, Testing & Maintenance](10-frontend-testing-maintenance.md) | Vite/React dev, type gen, test suites, perf profiling, AI UX loop (`/ux-loop`), deploy |
| 11 | [Maintenance, Cleanup & Troubleshooting](11-maintenance-troubleshooting.md) | pg-queue, DB maintenance/VACUUM, full wipe-and-reload, data cleanup, read-replica deployment, troubleshooting matrix, phase dependencies |

---

## TL;DR — Cold-Start to Live System

```bash
# 1. Bootstrap
make init                       # venv + uv + sync deps
make ui-init                    # npm install
make up                         # Docker: postgres + mlflow
make db-apply-sql               # Apply DDL (130 files)

# 2. Load all data
make setup-data                 # normalize-all + load-all (all 11 domains)

# 3. Features + clustering
make features-compute           # SKU feature pipeline → dim_sku
make cluster-all                # Cluster experiment → auto-promote → sku_cluster_assignment

# 4. ML — backtest + tune + champion
make customer-features          # Pre-compute customer-derived features
make backtest-all               # All model families (long: ~hours)
make backtest-load-all-bulk     # Load predictions (~4× faster path)
make tune-all                   # Bayesian hyperparameter tuning
make champion-all               # DFU-level champion assignments

# 5. Promote & generate production forecast
curl -X POST -H "X-API-Key: $API_KEY" \
     "$API_BASE/backtest-management/champion/promote"
make forecast-generate

# 6. Inventory planning
make setup-demand-planning      # forecast-driven artifacts
make setup-inv-planning         # SS, EOQ, policies, exceptions, KPIs

# 7. Operations
make setup-ops                  # S&OP, events, storyboard, DQ, financial

# 8. Verify + serve
make health
make check-all
make audit-routers
make dev                        # API :8000 + UI :5173
```

End-to-end runtime on a fresh dataset: **~4–6 hours** (dominated by foundation-model backtests). Use `make setup-all` to chain steps 1–7.

---

## Daily / Weekly / On-Event Cadence

| Trigger | Re-run |
|---|---|
| New inventory snapshot (daily) | `make refresh-agg`, `make intramonth-all`, `make exceptions-generate` |
| New customer-demand month closed | `make pipeline-customer-demand`, `make service-level-all`, `make bias-all` |
| New external forecast file | `make load-forecast-replace` |
| Model backtest cycle (weekly) | `make backtest-all` → `make tune-all` → `make champion-all` → promote → `make forecast-generate` → `make setup-demand-planning` → `make ss-all` |
| Cluster drift detected | `make features-compute`, `make cluster-all`, `make policy-all` |
| LLM key rotated | `POST /admin/llm/reset` |
| New API path prefix added | Add to `frontend/vite.config.ts`, run `make audit-routers` |

---

## Module ↔ Section Cross-Reference

| Module | Code locations | Section |
|---|---|---|
| ETL / domains | `scripts/etl/`, `common/core/domain_specs.py`, `config/etl/etl_config.yaml` | 2 |
| Clustering | `common/ml/clustering/`, `cluster_experiment` table | 3 |
| SKU features | `common/ml/sku_features/`, `dim_sku`, `config/forecasting/sku_features_config.yaml` | 3 |
| Backtest framework | `common/ml/backtest_framework.py`, `common/ml/model_registry.py` | 4 |
| Tuning | `common/ml/tuning.py`, `config/forecasting/tune_strategies.yaml`, `config/forecasting/cluster_tuning_profiles.yaml` | 5 |
| Champion | `common/ml/champion/` (package, 9 modules, 30 strategies), `data/champion/dfu_assignments.csv` | 5 |
| Production forecast | `fact_candidate_forecast` → `fact_production_forecast`, `model_promotion_log` | 6 |
| Inventory planning | `scripts/` (compute_*.py), `api/routers/inventory/`, `frontend/src/tabs/inv-planning/` | 7 |
| Exceptions / DQ | `common/engines/exception_engine.py`, `common/engines/dq_engine.py` | 7, 8 |
| Scheduler | `common/services/job_scheduler.py`, `job_registry.py` | 8 |
| AI / LLM | `api/llm.py`, `api/routers/intelligence/`, `common/ai/` | 9 |
| Frontend | `frontend/src/tabs/`, `frontend/vite.config.ts` | 10 |

---

## Critical Cross-Cutting Rules (See CLAUDE.md for Full List)

- **`get_conn()` not `Depends(_get_pool)`** in all `inv_planning_*.py` routers
- **psycopg3 uses `%s` placeholders** — never `$1`, `$2`
- **`domains.py` mounted last** in `api/main.py` (catch-all `{domain}` shadows other routes)
- **Vite proxy must mirror API prefixes** — run `make audit-routers` after route changes
- **All config in YAML** (no magic numbers in scripts) — `forecast_pipeline_config.yaml` is the ML pipeline source of truth
- **`ml_cluster` is METADATA, not a model feature** (excluded from `feature_cols` to prevent leakage)
- **Forecast promotion**: candidate → production via `POST /backtest-management/{model_id}/promote`
- **Tree-only training endpoint**: `POST /{model_id}/train` rejects foundation/DL models with 400
- **Clustering master switch**: `clustering.enabled=false` → all backtests fall back to global

---

## Known Discrepancies Discovered During Manual Authoring

These are honest gaps where docs/CLAUDE.md and code disagree. Each is flagged in its own section for fix-up.

| Item | Where | Status |
|---|---|---|
| Pool sizing drift (`POOL_MAX_SIZE` code vs docs) | §1 | RECONCILED (P0-1): independent defaults — sync `POOL_MAX_SIZE=12`, async `ASYNC_POOL_MAX_SIZE=20`, read `READ_POOL_MAX_SIZE=12`; `max_connections=200`; `make deploy-check` enforces the multi-pool invariant. |
| `make seasonality-all` / `variability-all` referenced in CLAUDE.md but not in Makefile | §3 | Aliases unimplemented; use `make features-compute` |
| `make tune-cust-enriched-all` referenced in CLAUDE.md but not in Makefile | §5 | Use direct script invocation |
| `scripts/ai/ingest_docs.py` only `--dry-run` mode (TODO) | §9 | Real ingestion not yet wired |
| `deploy-frontend` skips `tsc -b` due to ~30 pre-existing type errors | §10 | Tracked on restructure branch |

---

## Help

- Architecture deep-dive, platform overview & feature catalog: [docs/ARCHITECTURE.md](../ARCHITECTURE.md)
- Cleanup / fresh recreate: [Maintenance, Cleanup & Troubleshooting](11-maintenance-troubleshooting.md)
- Design specs: [docs/specs/README.md](../specs/README.md)
- Project rules: [CLAUDE.md](../../CLAUDE.md)
