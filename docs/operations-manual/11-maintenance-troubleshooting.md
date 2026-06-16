# Maintenance, Cleanup & Troubleshooting

Operational deep-dive reference for keeping the platform healthy: the Postgres-backed job queue, routine database maintenance, full wipe-and-reload procedures, targeted data cleanup, read-replica deployment, a troubleshooting matrix, and the phase dependency map. These procedures are operator-critical — every command, target, and SQL block below is reproduced verbatim. (This material was consolidated from `docs/RUNBOOK.md`; the partition/MV refresh basics are also touched on in `07-inventory-planning.md` and `08-operations-sop-control-tower.md`, but the full procedures live here.)

## pg-queue (Postgres-Backed Job Queue)

A minimal Postgres-backed job queue runs **alongside** APScheduler for jobs that
must survive API restarts, span multiple workers, or run for many hours
without blocking the API thread pool. Item 22 introduces this as a pilot:
exactly one job — `refresh_intramonth` — has been migrated. Remaining
APScheduler jobs are unchanged.

### Architecture

| Component | Purpose | File |
|---|---|---|
| `job_queue` table | Persistent queue rows; status transitions tracked here | `sql/183_create_pg_queue.sql` |
| `common.services.pg_queue` | Public API: `enqueue_job`, `claim_next_job`, `mark_*`, `requeue_failed_with_backoff`, `get_queue_depth` | `common/services/pg_queue.py` |
| `scripts/ops/pg_queue_worker.py` | Long-running worker: claims, executes, transitions state | `scripts/ops/pg_queue_worker.py` |

Claim semantics use `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1` so multiple
workers race safely. Failures auto-retry with exponential backoff
(60s → 120s → 240s → ... capped at 1 h) up to `max_attempts` (default 3).

### Daily Operations

```bash
make pg-queue-worker                   # Run a worker (long-running; one process)
make pg-queue-enqueue-recurring        # Enqueue refresh_intramonth (cron entry-point)
make pg-queue-depth                    # Inspect queue depth by status
```

A typical production setup runs ONE `pg-queue-worker` as a systemd unit (or
docker container) and triggers `pg-queue-enqueue-recurring` from cron / a
lightweight APScheduler "enqueue-only" job once per day.

### When to Migrate a Job From APScheduler to pg-queue

Use pg-queue when the job:

* runs for **>15 min** (long-running jobs hold APScheduler thread pool slots);
* must **survive an API restart** without losing in-flight state;
* is safe to run from a **separate process** (i.e. doesn't depend on in-process
  globals owned by the API);
* benefits from being **horizontally scalable** (multiple workers).

Keep on APScheduler when the job:

* completes in seconds and is fired by a UI action;
* relies on `JobManager`'s per-group concurrency contract or the threading
  primitives in `job_state.py`;
* is interactively cancellable from the Jobs tab.

### Migration Recipe (per job)

1. **Add a handler** in `scripts/ops/pg_queue_worker.py` `HANDLERS` mapping
   the new `job_type` to its callable. Reuse the existing `_run_*` runner
   from `common/services/job_state.py` so the queue path is byte-for-byte
   equivalent.
2. **Drop the APScheduler schedule** (delete the row in `job_schedule` or
   remove the `schedule_recurring` call). Manual triggering of the same
   `job_type` via `JobManager.submit_job` still works.
3. **Wire enqueue** — add a Make target that calls `enqueue_job(...)` and
   trigger it from cron. Or keep a tiny APScheduler job whose only side
   effect is to enqueue.
4. **Run a worker** — `make pg-queue-worker` (one instance per box).
5. **Watch** — `make pg-queue-depth` in CI / dashboards. Set up an alert on
   `failed > 0` after `max_attempts` exhausted (dead-letter).

### Operational Concerns

* **Monitoring queue depth**: poll `get_queue_depth()` every minute. Alert
  on `pending > N` (saturation) or any `failed` row older than 1 h.
* **Dead-letter handling**: jobs that exhaust `max_attempts` stay in
  `status='failed'` forever. Triage manually:
  ```sql
  SELECT id, job_type, attempts, last_error, completed_at
  FROM job_queue WHERE status = 'failed'
  ORDER BY completed_at DESC;
  ```
  Reset for re-run with: `UPDATE job_queue SET status='pending', attempts=0,
  run_at=NOW() WHERE id = <id>;`
* **Worker crash recovery**: if a worker dies between `mark_running` and
  `mark_completed`, the row sits in `status='running'`. A future cleanup job
  can re-enqueue any `running` rows whose `started_at` is older than
  expected. (Not implemented in the pilot; documented as follow-up.)
* **DDL**: `sql/183_create_pg_queue.sql` is idempotent (`CREATE TABLE IF NOT
  EXISTS`). Apply via `make db-apply-sql`.

### Pilot Cutover: refresh_intramonth

`refresh_intramonth` (the `mv_intramonth_stockout` refresh, flagged as
7-20 h at 40× scale) is the proof-point. Its `JobTypeDef` remains in
`JOB_TYPE_REGISTRY` so the manual "Run now" UI button still works, but the
**recurring** path now goes through pg-queue. To complete the cutover in a
deployed environment:

```bash
# 1. Apply schema (one-time)
make db-apply-sql

# 2. If a job_schedule row exists for refresh_intramonth, delete it
psql "$DATABASE_URL" -c "DELETE FROM job_schedule WHERE job_type='refresh_intramonth';"

# 3. Run the worker (systemd / docker)
make pg-queue-worker

# 4. Wire daily enqueue (cron, e.g. 02:00 UTC)
0 2 * * * cd /app && make pg-queue-enqueue-recurring
```

---

## Database Maintenance & Optimization

Routine maintenance procedures to keep PostgreSQL healthy as data grows.

### Quick Reference

```bash
make db-health                          # Full health report (size, cache hit, bloat, seq scans)
make db-analyze                         # Update planner statistics (run after bulk loads)
make db-optimize                        # ANALYZE + identify unused indexes (dry-run)
make db-drop-unused-indexes             # Show unused indexes (dry-run)
make db-drop-unused-indexes EXECUTE=1   # Actually drop unused indexes
make db-retention                       # Show data retention actions (dry-run)
make db-retention EXECUTE=1             # Apply retention policies (drop old partitions, delete old rows)
make db-maintain                        # Routine: ANALYZE + health report
make auto-create-partitions             # Create next 12 months of partitions (idempotent)
make auto-create-partitions HORIZON=24  # Create next 24 months
make auto-create-partitions-dry-run     # Print partition DDL without executing
```

### Recommended Schedule

| Task | Frequency | Command |
|---|---|---|
| ANALYZE (planner stats) | After every bulk load, or weekly | `make db-analyze` |
| Health check | Weekly | `make db-health` |
| Drop unused indexes | Monthly | `make db-drop-unused-indexes EXECUTE=1` |
| Retention policy | Monthly | `make db-retention EXECUTE=1` |
| Full maintenance | Weekly | `make db-maintain` |
| Auto-create future partitions | Monthly (cron) **or** before any backfill | `make auto-create-partitions` |

### Partition Auto-Creation (`auto-create-partitions`)

`scripts/db/auto_create_partitions.py` provisions the next N partitions
(default 12) for every RANGE-partitioned fact table:
`fact_inventory_snapshot`, `fact_customer_demand_monthly`,
`fact_external_signal`. The script uses `CREATE TABLE IF NOT EXISTS ... PARTITION OF`,
so it is fully idempotent — re-running is always safe.

Each registry entry in the script picks an `interval` of `month` (default) or
`week`. Weekly partitions use ISO-8601 numbering (Mon–Sun) and are named
`<prefix>_YYYYwWW`. The base `make auto-create-partitions` target picks up the
correct interval per table; `make auto-create-partitions-weekly` restricts the
run to weekly tables only (useful right after the weekly cutover migrations).

Run it on a schedule (monthly for monthly tables, weekly for weekly tables),
or manually before any large backfill that may write rows beyond the last
hardcoded partition. Without this, future-dated rows fall into the DEFAULT
partition and the planner cannot prune them. Example crontab (1st of every
month at 02:00 + every Monday at 02:30):

```cron
0  2 1 * *  cd /path/to/DemandProject && make auto-create-partitions          >> /var/log/demand_partitions.log 2>&1
30 2 * * 1  cd /path/to/DemandProject && make auto-create-partitions-weekly   >> /var/log/demand_partitions.log 2>&1
```

### Sub-monthly partitioning cutover

The cutover migrations `sql/184_partition_inventory_snapshot_weekly_cutover.sql`
and `sql/185_partition_customer_demand_weekly_cutover.sql` switch the rolling
window of `fact_inventory_snapshot` and `fact_customer_demand_monthly` from
monthly to weekly partitioning. Both files start with a
`!! REVIEW BEFORE RUN !!` banner — they are NOT picked up by `make
db-apply-sql` automatically.

**When to do it:** when single-partition scan cost on the inventory or
customer-demand fact becomes the dominant query latency (for inventory, that's
roughly when monthly partitions exceed ~30M rows; we expect this around the
40× scaling milestone). Weekly partitioning roughly halves the per-partition
scan and improves vacuum/analyze concurrency.

**Strategy:** the migrations KEEP every existing monthly partition (which is
already populated and indexed) and ADD weekly partitions for the next 12
weeks only. Partition pruning still works because monthly and weekly ranges
don't overlap. After cutover, `auto_create_partitions.py`'s registered
`interval` for the affected table is flipped from `"month"` to `"week"` so
the rolling window provisions weekly going forward.

**Estimated downtime:**

| Step | Wall-clock |
|---|---|
| Pause ETL writers | seconds (job graceful stop) |
| `DETACH PARTITION default` (empty default) | < 1 s |
| `DETACH PARTITION default` (~30 M rows) | 2–5 min |
| 12 × `CREATE TABLE ... PARTITION OF` | 5–30 s total |
| `RENAME` + new empty default | < 1 s |
| Resume ETL writers | seconds |
| **Realistic total — `fact_inventory_snapshot`** | **5–30 min** |
| **Realistic total — `fact_customer_demand_monthly`** | **2–10 min** |

The wide range reflects whether the existing default partition is empty.
Drain the default beforehand to land at the low end.

**Pre-flight:**

1. Confirm no long-running transactions are touching the parent table:
   ```sql
   SELECT pid, mode, granted, query_start
     FROM pg_locks l
     JOIN pg_stat_activity a USING (pid)
    WHERE l.relation = 'fact_inventory_snapshot'::regclass;
   ```
2. Verify default partition is empty (or accept the longer DETACH cost):
   ```sql
   SELECT COUNT(*) FROM fact_inventory_snapshot_default;
   ```
3. Verify the most recent monthly partition's `TO` bound is on a Monday — or
   accept the gap days will sit in the default partition until the next
   monthly partition's range covers them.
4. Take a fresh schema-only `pg_dump` of the partitioned table for rollback
   reference.

**Order of operations:**

1. **Stop ETL.** Pause the inventory loader (`load_inventory_postgres.py`)
   and customer-demand loader (`load_customer_demand_postgres.py`). Disable
   APScheduler jobs for these via the Jobs tab.
2. **Edit and apply the migration.** Open
   `sql/184_partition_inventory_snapshot_weekly_cutover.sql`, replace the
   `<YYYYwWW_n>` and `<YYYY-MM-DD_mon_n>` placeholders with the actual ISO
   week numbers and Monday dates for the next 12 weeks, then run via
   `psql … -f sql/184_partition_inventory_snapshot_weekly_cutover.sql`.
   Repeat for `sql/185_…` if cutting over customer-demand at the same time.
3. **Update the auto-create registry.** In
   `scripts/db/auto_create_partitions.py`, change the migrated table's entry
   from `interval="month"` to `interval="week"`.
4. **Extend the rolling window.** Run `make auto-create-partitions-weekly` to
   add any further weeks beyond the 12 already created by the migration.
5. **Restart ETL.** Re-enable the loaders and APScheduler jobs.

**Verification:**

```sql
-- 1. Partition count and shape
SELECT relname, pg_size_pretty(pg_total_relation_size(c.oid)) AS size
  FROM pg_inherits i
  JOIN pg_class c ON c.oid = i.inhrelid
 WHERE i.inhparent = 'fact_inventory_snapshot'::regclass
 ORDER BY relname;

-- 2. New weekly partitions populating as data lands
SELECT relname, COUNT(*) FROM (
    SELECT c.relname, p.snapshot_date
      FROM fact_inventory_snapshot p
      JOIN pg_class c ON c.oid = p.tableoid
     WHERE c.relname LIKE 'fact_inventory_snapshot_2026w%'
) t GROUP BY relname ORDER BY relname;

-- 3. Planner is pruning correctly (only the matching partition appears)
EXPLAIN SELECT COUNT(*) FROM fact_inventory_snapshot
 WHERE snapshot_date BETWEEN DATE '2026-05-04' AND DATE '2026-05-10';
```

**Rollback (if cutover goes wrong):**

1. `ALTER TABLE fact_inventory_snapshot DETACH PARTITION fact_inventory_snapshot_<YYYYwWW>;`
   for every weekly partition just created.
2. `DROP TABLE` each detached weekly partition (they are empty if cutover
   happened with ETL paused).
3. `ALTER TABLE fact_inventory_snapshot DETACH PARTITION fact_inventory_snapshot_default;`
   then re-attach the previous default
   (`fact_inventory_snapshot_default_premigration`) as DEFAULT, and drop the
   new empty one.
4. Revert the registry entry in `scripts/db/auto_create_partitions.py` to
   `interval="month"`.
5. Restart ETL.

The rollback SQL is intentionally NOT shipped as a file — the safe path is to
run it interactively while watching `pg_locks` and partition row counts.

### PostgreSQL Configuration (docker-compose.yml)

Tuned for a 16 GB host with SSD storage:

| Parameter | Value | Why |
|---|---|---|
| `shared_buffers` | 4 GB | 25% of RAM — primary buffer cache |
| `effective_cache_size` | 12 GB | 75% of RAM — planner hint for OS cache |
| `work_mem` | 128 MB | Reduce temp file spills for sorts/joins |
| `maintenance_work_mem` | 1 GB | Faster VACUUM, REINDEX, CREATE INDEX |
| `effective_io_concurrency` | 200 | SSD-optimized parallel I/O |
| `max_parallel_workers_per_gather` | 4 | Parallel query execution |
| `wal_buffers` | 64 MB | Better write throughput |
| `log_min_duration_statement` | 5000 ms | Log slow queries (> 5s) |
| `default_statistics_target` | 200 | More accurate planner stats |

### Index Strategy

- **BRIN indexes** on large partitioned tables (`fact_customer_demand_monthly`, `fact_inventory_snapshot`) — 100-1000x smaller than B-tree for date-ordered data
- **Partial indexes** for hot query paths (open exceptions, urgent signals, pending approvals)
- **Unique indexes on MVs** enable `REFRESH MATERIALIZED VIEW CONCURRENTLY` (zero read downtime)
- **Unused index cleanup**: `scripts/db/drop_unused_indexes.py` identifies indexes with zero planner scans. Preserves primary keys and unique constraints.

### Materialized View Refresh

All MV refreshes now use `CONCURRENTLY` (via unique indexes from migration 119), providing zero-downtime refreshes. The tiered refresh order prevents dependency issues:

1. **Tier 1**: `agg_sales_monthly`, `agg_forecast_monthly`, `agg_inventory_monthly`
2. **Tier 2**: `mv_inventory_forecast_monthly`, `mv_fill_rate_monthly`, `mv_intramonth_stockout`
3. **Tier 3**: `mv_supplier_po_performance`, `agg_accuracy_by_dim`, `agg_dfu_coverage`
4. **Tier 4**: `mv_inventory_health_score`, `mv_control_tower_kpis`

### Data Retention Policies

Configured in `config/platform/db_maintenance_config.yaml`:

| Table | Retention | Method |
|---|---|---|
| `fact_customer_demand_monthly` | 36 months | Drop old partitions |
| `fact_inventory_snapshot` | 24 months | Drop old partitions |
| `job_history` | 3 months | DELETE + VACUUM |
| `ai_call_log` | 2 months | DELETE + VACUUM |
| `fact_audit_log` | 12 months | DELETE + VACUUM |
| `fact_query_performance` | 1 month | DELETE + VACUUM |

### Maintenance Scripts

| Script | Purpose |
|---|---|
| `scripts/db/drop_unused_indexes.py` | Identify and drop indexes with zero planner scans |
| `scripts/db/db_maintenance.py analyze` | Run ANALYZE on all tables |
| `scripts/db/db_maintenance.py health` | Comprehensive health report |
| `scripts/db/db_maintenance.py retention` | Apply retention policies |

---

## Database Cleanup & Fresh Recreate

Full wipe-and-reload procedure: clears non-config data, derived outputs, job/perf history, and experiment history while preserving configuration masters, schedules, and platform settings. Then reloads from `data/input/`, runs the ML pipeline through champion selection, and refreshes baseline planning outputs.

> **When to use:** Starting fresh with new input data, recovering from a corrupted pipeline state, or resetting after major schema changes.

### Quick Start (Make Targets)

**One-command recipes** — pick the level you need:

```bash
make fresh-all              # Full reset: truncate + clean + load + ML + champion + baseline planning (~4-6 hours)
make fresh-champion         # Load + features + backtests + champion (no truncate, ~3-4 hours)
make fresh-backtest         # Load + features + backtests (no champion, ~3 hours)
make fresh-features         # Load + clustering + seasonality + variability + LT (~1 hour)
make fresh-load             # Normalize + load + refresh MVs only (~5 min)
```

**Individual step targets** — run these manually if you need granular control:

| Target | What it does | Time |
|---|---|---|
| `make db-truncate-data` | Truncate non-config data, history, and experiment tables in one transaction while preserving configuration masters | < 1 min |
| `make clean-artifacts` | Remove stale clean CSVs, backtest/tuning outputs, clustering artifacts, champion files, and perf reports | < 1 sec |
| `make normalize-all` | Normalize all 10 input CSVs → `data/staged/*_clean.csv` | ~1 min |
| `make load-all` | Load all 10 domains into Postgres (dimensions first, facts second) | ~1 min |
| `make refresh-mvs-tiered` | Refresh all 13 MVs in 4-tier dependency order | ~30 sec |
| `make cluster-all` | Feature engineering + KMeans training + label + update dim_sku.ml_cluster | ~10 min |
| `make seasonality-all` | Seasonality detection + update dim_sku | ~5 min |
| `make variability-all` | Demand variability computation → dim_sku | ~2 min |
| `make lt-profile-all` | Lead time profiles → dim_item_lead_time_profile | ~2 min |
| `make backtest-lgbm` | LGBM per-cluster backtest (10 timeframes) | ~30 min |
| `make backtest-catboost` | CatBoost per-cluster backtest | ~40 min |
| `make backtest-xgboost` | XGBoost per-cluster backtest | ~20 min |
| `make backtest-chronos` | Chronos T5 foundation model backtest | ~2.5 hours |
| `make backtest-bolt` | Chronos Bolt foundation model backtest | ~12 min |
| `make backtest-chronos2` | Chronos 2 foundation model backtest | ~5.5 hours |
| `make backtest-chronos2e` | Chronos 2 Enriched backtest (31 covariates) | ~6 hours |
| `make backtest-seasonal-naive` | Seasonal Naive baseline backtest | ~5 min |
| `make backtest-rolling-mean` | Rolling Mean baseline backtest | ~5 min |
| `make backtest-mstl` | MSTL decomposition backtest | ~15 min |
| `make backtest-nhits` | N-HiTS deep learning backtest | ~1 hour |
| `make backtest-nbeats` | N-BEATS deep learning backtest | ~1 hour |
| `make backtest-baselines` | Seasonal Naive + Rolling Mean together | ~10 min |
| `make backtest-all` | All tree + foundation backtests sequentially | ~12 hours |
| `make backtest-load-all` | Load all backtest predictions into DB | ~5 min |
| `make backtest-load-all-bulk` | Load all predictions with single index cycle (~4x faster) | ~1.5 min |
| `make backtest-load-bulk` | Load 4 core models (lgbm, catboost, xgboost, chronos) in bulk | ~1 min |
| `make backtest-load-main-only` | Load specific models to main table only (skip archive) | varies |
| `make backtest-load-archive-only` | Load specific models to archive only (skip main table) | varies |
| `make refresh-accuracy-mvs` | Refresh 4 accuracy MVs (after backtest load) | ~10 sec |
| `make champion-all` | Train meta-learner + simulate strategies + select champion | ~15 min |
| `make policy-all` | Refresh policy assignments while preserving manual overrides | ~1 min |
| `make ss-all` | Recompute safety stock targets | ~2 min |
| `make eoq-all` | Recompute EOQ targets | ~1 min |
| `make health-all` | Refresh inventory health score after SS/EOQ refresh | ~1 min |

**Dependency chain:**

```
fresh-all
├── db-truncate-data              (truncate non-config data/history/experiments)
├── clean-artifacts               (remove stale files)
└── fresh-champion
    └── fresh-backtest
        └── fresh-features
            └── fresh-load
                ├── normalize-all     (CSV → clean CSV)
                ├── load-all          (clean CSV → Postgres)
                └── refresh-mvs-tiered (13 MVs, tier-ordered)
            ├── cluster-all           (clustering pipeline)
            ├── seasonality-all       (seasonality detection)
            ├── variability-all       (demand variability)
            └── lt-profile-all        (lead time profiles)
        ├── backtest-all              (LGBM + CatBoost + XGBoost + Chronos + Bolt + Chronos2 + Chronos2e)
        ├── backtest-load-all         (load predictions → DB)
        └── refresh-accuracy-mvs      (accuracy MVs)
    └── champion-all                  (meta-learner + simulate + select)
├── seed-baselines                   (seed baseline forecasts)
├── policy-all                       (refresh DFU policy assignments)
├── ss-all                           (recompute safety stock)
├── eoq-all                          (recompute EOQ)
└── health-all                       (refresh inventory health score)
```

> **Note:** `fresh-all` covers data loading, ML pipeline, and baseline inventory planning only. To fully populate the application (production forecasts, demand planning, operations), run `setup-all` or the remaining phases below.

### Full Application Setup (`setup-all`)

`setup-all` runs all 6 phases end-to-end. Use this for a complete environment build from scratch.

```bash
make setup-all    # Phases 1-6: data → features → backtests → inv planning → demand planning → ops
```

**Phase breakdown:**

| Phase | Target | What it does | Depends on |
|---|---|---|---|
| 1 | `setup-data` | Normalize + load all 10 domains (parallel pipeline) | Input CSVs |
| 2 | `setup-features` | Clustering, seasonality, variability, lead time, ABC-XYZ, demand signals | Phase 1 |
| 3 | `setup-backtest` | All backtests (tree + foundation) + load + accuracy refresh + champion selection + seed baselines | Phase 2 |
| 4 | `setup-inv-planning` | EOQ, policies, safety stock, exceptions, fill rate, health, supplier perf, investment, intramonth, control tower, rebalancing | Phase 1 |
| 5 | `setup-demand-planning` | Production forecasts, projections, POs, quantiles, consensus, planned orders, replenishment plan, bias, blended, service level, lead time, echelon | Phase 3 |
| 6 | `setup-ops` | S&OP, events, financial plan, storyboard, scenarios, DQ | Phase 1 |

**Dependency chain:**

```
setup-all
├── setup-backtest (Phase 3)
│   └── setup-features (Phase 2)
│       └── setup-data (Phase 1)
│           └── run_pipeline.py --mode full --parallel
│       ├── cluster-all
│       ├── seasonality-all
│       ├── variability-all
│       ├── lt-profile-all
│       ├── abc-xyz-all
│       └── demand-signals-all
│   ├── backtest-all           (LGBM + CatBoost + XGBoost + Chronos + Bolt + Chronos2 + Chronos2e)
│   ├── backtest-load-all      (predictions → DB)
│   ├── accuracy-slice-refresh (accuracy MVs)
│   ├── champion-all           (meta-learner + simulate + select)
│   └── seed-baselines         (baseline forecasts)
├── setup-inv-planning (Phase 4)
│   ├── eoq-all                ├── supplier-perf-all
│   ├── policy-all             ├── investment-all
│   ├── ss-all                 ├── intramonth-all
│   ├── exceptions-generate    ├── control-tower-all
│   ├── fill-rate-all          └── rebalancing-all
│   └── health-all
├── setup-demand-planning (Phase 5)
│   ├── forecast-prod-all      ├── planned-orders-all
│   ├── projection-all         ├── replplan-all
│   ├── po-all                 ├── bias-all
│   ├── quantile-all           ├── blended-all
│   ├── consensus-all          ├── service-level-all
│   ├── lead-time-all          └── echelon-all
└── setup-ops (Phase 6)
    ├── sop-all                ├── storyboard-all
    ├── events-all             ├── scenarios-all
    ├── financial-plan-all     └── dq-all
```

**After `fresh-all`, run the remaining phases:**

```bash
make setup-demand-planning    # Phase 5: production forecasts + demand planning
make setup-ops                # Phase 6: S&OP, events, storyboard, DQ
```

### Preserved Tables (Untouched)

These tables are not explicitly truncated by `db-truncate-data` because they hold durable configuration, master data, schedules, or platform settings that should survive a fresh reload:

| Category | Tables |
|---|---|
| Config / Policy | `dim_replenishment_policy`, `fact_dfu_policy_assignment`, `fact_service_level_targets` |
| Master Data / Planning Inputs | `dim_supplier`, `dim_item_supplier`, `dim_item_cost`, `fact_budget_periods`, `dim_echelon_network`, `dim_external_signal_source`, `dim_transfer_lane` |
| Platform | `dim_user`, `dim_dq_check_catalog`, `dim_notification_channel`, `dim_webhook_registration`, `dim_erp_integration`, `dim_report_template`, `fact_report_schedule`, `job_schedule` |
| MLflow | All `experiments`, `runs`, `metrics`, `params`, `tags`, `logged_models`, `model_versions`, `registered_models`, `datasets`, `inputs`, `trace_*` tables |
| System | `alembic_version` |

### Step 1: Truncate Data Tables

Run as a single SQL transaction. Ordered by FK dependency (children before parents). This reset clears transactional facts, derived outputs, job/perf history, and experiment history, while leaving configuration masters in place.

> **Note:** `fact_inventory_snapshot` is monthly RANGE-partitioned (2025-01 through 2026-03 + default). TRUNCATE on the parent cascades to all partitions automatically.

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 <<'EOSQL'
BEGIN;

-- Group 1: AI / Analytics
TRUNCATE TABLE ai_recommendation_outcomes CASCADE;
TRUNCATE TABLE ai_insights CASCADE;
TRUNCATE TABLE ai_planning_memos CASCADE;
TRUNCATE TABLE ai_call_log CASCADE;

-- Group 2: Exceptions / Decisions
TRUNCATE TABLE planner_decisions CASCADE;
TRUNCATE TABLE exception_queue CASCADE;

-- Group 3: S&OP (children → parent)
TRUNCATE TABLE fact_sop_approved_plan CASCADE;
TRUNCATE TABLE fact_sop_gaps CASCADE;
TRUNCATE TABLE fact_sop_supply_constraints CASCADE;
TRUNCATE TABLE fact_sop_demand_review CASCADE;
TRUNCATE TABLE fact_sop_cycles CASCADE;

-- Group 4: Events (children → parent)
TRUNCATE TABLE fact_event_conflicts CASCADE;
TRUNCATE TABLE fact_event_performance CASCADE;
TRUNCATE TABLE fact_event_adjusted_forecast CASCADE;
TRUNCATE TABLE fact_event_calendar CASCADE;

-- Group 5: Scenarios (child → parent)
TRUNCATE TABLE fact_scenario_results CASCADE;
TRUNCATE TABLE fact_supply_scenarios CASCADE;

-- Group 6: Procurement (children → parent)
TRUNCATE TABLE fact_po_approval_log CASCADE;
TRUNCATE TABLE fact_po_receipts CASCADE;
TRUNCATE TABLE fact_open_purchase_orders CASCADE;
TRUNCATE TABLE fact_purchase_orders CASCADE;

-- Group 7: Consensus / Overrides
TRUNCATE TABLE fact_consensus_plan CASCADE;
TRUNCATE TABLE fact_forecast_overrides CASCADE;

-- Group 8: Rebalancing (child → parent)
TRUNCATE TABLE fact_rebalancing_transfer CASCADE;
TRUNCATE TABLE fact_rebalancing_plan CASCADE;

-- Group 9: Forecasting / Backtesting
TRUNCATE TABLE backtest_lag_archive CASCADE;
TRUNCATE TABLE fact_external_forecast_monthly CASCADE;
TRUNCATE TABLE fact_candidate_forecast CASCADE;
TRUNCATE TABLE model_promotion_log CASCADE;
TRUNCATE TABLE fact_production_forecast CASCADE;
TRUNCATE TABLE fact_blended_demand_plan CASCADE;
TRUNCATE TABLE fact_demand_plan CASCADE;
TRUNCATE TABLE fact_demand_plan_weekly CASCADE;
TRUNCATE TABLE fact_bias_corrections CASCADE;
TRUNCATE TABLE fact_bias_correction_history CASCADE;

-- Group 10: Inventory (parent CASCADE → all 15 partitions + default)
TRUNCATE TABLE fact_inventory_snapshot CASCADE;
TRUNCATE TABLE fact_inventory_projection CASCADE;

-- Group 11: Sales
TRUNCATE TABLE fact_sales_monthly CASCADE;
TRUNCATE TABLE fact_sales_monthly_original CASCADE;

-- Group 11b: Customer Demand (parent CASCADE → all partitions + default)
TRUNCATE TABLE fact_customer_demand_monthly CASCADE;

-- Group 12: Inventory Planning
TRUNCATE TABLE fact_ss_simulation_results CASCADE;
TRUNCATE TABLE fact_safety_stock_targets CASCADE;
TRUNCATE TABLE fact_eoq_targets CASCADE;
TRUNCATE TABLE fact_demand_signals CASCADE;
TRUNCATE TABLE fact_replenishment_plan CASCADE;
TRUNCATE TABLE fact_replenishment_exceptions CASCADE;
TRUNCATE TABLE fact_planned_orders CASCADE;
TRUNCATE TABLE fact_plan_versions CASCADE;
TRUNCATE TABLE fact_financial_inventory_plan CASCADE;
TRUNCATE TABLE fact_inventory_investment_plan CASCADE;
TRUNCATE TABLE fact_efficient_frontier CASCADE;

-- Group 13: Echelon
TRUNCATE TABLE fact_echelon_reorder_points CASCADE;
TRUNCATE TABLE fact_echelon_ss_targets CASCADE;

-- Group 14: Service Level / Lead Time
TRUNCATE TABLE fact_service_level_performance CASCADE;
TRUNCATE TABLE fact_lead_time_actuals CASCADE;
TRUNCATE TABLE fact_lt_review_triggers CASCADE;

-- Group 15: External Signals
TRUNCATE TABLE fact_external_signal CASCADE;

-- Group 16: DQ / Collaboration / Audit / Notifications / Reports
TRUNCATE TABLE fact_dq_corrections CASCADE;
TRUNCATE TABLE fact_dq_check_results CASCADE;
TRUNCATE TABLE fact_annotation CASCADE;
TRUNCATE TABLE fact_shared_view CASCADE;
TRUNCATE TABLE fact_intervention_metrics CASCADE;
TRUNCATE TABLE fact_notification_log CASCADE;
TRUNCATE TABLE fact_webhook_delivery CASCADE;
TRUNCATE TABLE fact_report_delivery CASCADE;
TRUNCATE TABLE fact_audit_log CASCADE;
TRUNCATE TABLE fact_query_performance CASCADE;

-- Group 17: Infrastructure
TRUNCATE TABLE audit_load_batch CASCADE;
TRUNCATE TABLE job_history CASCADE;
-- US17: ingestion now writes ONLY to job_history (via JobManager: etl_pipeline /
-- load_domain / chain pipelines). integration_job + integration_chain are
-- read-only ARCHIVES (no new rows). integration_job_unified (sql/188) is a VIEW
-- over integration_job + job_history (ETL job types) — no rows of its own,
-- nothing to TRUNCATE; truncating job_history + the two archives clears the
-- unified read surface too. The archives are kept as permanent history (no
-- migration needed); drop the view with
-- `DROP VIEW IF EXISTS integration_job_unified;` only when rebuilding the schema.
TRUNCATE TABLE perf_suggestion CASCADE;
TRUNCATE TABLE perf_query CASCADE;
TRUNCATE TABLE perf_section CASCADE;
TRUNCATE TABLE perf_run CASCADE;

-- Group 18: Tuning / Model Experiment History
TRUNCATE TABLE tuning_chat_message CASCADE;
TRUNCATE TABLE tuning_chat_session CASCADE;
TRUNCATE TABLE tuning_promotion_log CASCADE;
TRUNCATE TABLE lgbm_tuning_lag_cluster CASCADE;
TRUNCATE TABLE lgbm_tuning_lag CASCADE;
TRUNCATE TABLE lgbm_tuning_comparison CASCADE;
TRUNCATE TABLE lgbm_tuning_month CASCADE;
TRUNCATE TABLE lgbm_tuning_cluster CASCADE;
TRUNCATE TABLE lgbm_tuning_timeframe CASCADE;
TRUNCATE TABLE lgbm_tuning_run CASCADE;

-- Group 19: Cluster / Champion Experiment History
TRUNCATE TABLE cluster_experiment_comparison CASCADE;
TRUNCATE TABLE cluster_experiment CASCADE;
TRUNCATE TABLE champion_experiment CASCADE;

-- Group 20: Dimensions (facts already cleared)
TRUNCATE TABLE dim_sku CASCADE;
TRUNCATE TABLE dim_item CASCADE;
TRUNCATE TABLE dim_location CASCADE;
TRUNCATE TABLE dim_customer CASCADE;
TRUNCATE TABLE dim_time CASCADE;
TRUNCATE TABLE dim_sourcing CASCADE;

-- Group 21: Profiles / Decomposition
TRUNCATE TABLE dim_item_lead_time_profile CASCADE;
TRUNCATE TABLE dim_lead_time_profile CASCADE;
TRUNCATE TABLE mv_demand_decomposition CASCADE;

COMMIT;
EOSQL
```

### Step 2: Clean Intermediate Files

Remove stale artifacts so the pipeline regenerates everything from scratch:

```bash
rm -f data/staged/*_clean.csv data/staged/inventory_clean.csv
rm -rf data/backtest/lgbm_cluster/ data/backtest/catboost_cluster/ data/backtest/xgboost_cluster/
rm -rf data/backtest/chronos/ data/backtest/chronos_bolt/ data/backtest/chronos2/ data/backtest/chronos2_enriched/
rm -rf data/backtest/seasonal_naive/ data/backtest/rolling_mean/ data/backtest/mstl/
rm -rf data/backtest/nhits/ data/backtest/nbeats/
rm -rf data/backtest/logs/ data/backtest/tuning_archive/ data/tuning/ data/perf_reports/
rm -rf data/clustering/ data/champion/ data/models/
rm -f data/staged/seasonality_results.csv data/staged/clustering_features.csv
```

### Step 3: Normalize Input CSVs

```bash
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset item
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset location
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset customer
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset time
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset sku
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset sales
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset forecast
~/.local/bin/uv run python scripts/etl/normalize_inventory_csv.py
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset sourcing
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset purchase_order
```

### Step 4: Load Into Postgres (Dimensions First, Then Facts)

```bash
# Wave 1: Dimensions
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset item
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset location
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset customer
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset time
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset sku
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset sourcing

# Wave 2: Facts
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset sales
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset forecast
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset inventory
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset purchase_order
```

### Step 5: Refresh Materialized Views (Tier-Ordered)

MV dependencies require a specific refresh order. Some MVs (DQ, sensing, projections) depend on data from later pipeline steps — they will populate when those steps run.

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -c "
  -- Tier 1: Base aggregates (no MV dependencies)
  REFRESH MATERIALIZED VIEW agg_sales_monthly;
  REFRESH MATERIALIZED VIEW agg_forecast_monthly;
  REFRESH MATERIALIZED VIEW agg_inventory_monthly;

  -- Tier 2: Depend on tier 1 aggregates
  REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly;
  REFRESH MATERIALIZED VIEW mv_fill_rate_monthly;
  REFRESH MATERIALIZED VIEW mv_intramonth_stockout;
  REFRESH MATERIALIZED VIEW mv_supplier_po_performance;
  REFRESH MATERIALIZED VIEW mv_po_lead_time_analysis;
  REFRESH MATERIALIZED VIEW agg_accuracy_by_dim;
  REFRESH MATERIALIZED VIEW agg_dfu_coverage;

  -- Tier 3: Depend on tier 2 (mv_inventory_forecast_monthly)
  REFRESH MATERIALIZED VIEW mv_inventory_health_score;

  -- Tier 4: Depend on tier 3 (mv_inventory_health_score + mv_fill_rate + mv_intramonth)
  REFRESH MATERIALIZED VIEW mv_control_tower_kpis;
"
```

### Step 6: Clustering & Feature Engineering

```bash
~/.local/bin/uv run python scripts/ml/run_cluster_pipeline.py  # unified: features -> train -> label -> promote
```

### Step 7: SKU Feature Engineering (Seasonality + Variability)

The unified pipeline computes seasonality, variability, and lifecycle features in a single pass.
The legacy scripts (`detect_seasonality.py`, `update_seasonality_profiles.py`, `compute_demand_variability.py`, `generate_clustering_features.py`, `train_clustering_model.py`, `detect_drift.py`) have been removed -- use `make features-compute` (or `scripts/ml/compute_sku_features.py`) and `scripts/ml/run_cluster_pipeline.py` instead.

```bash
~/.local/bin/uv run python scripts/ml/compute_sku_features.py   # or: make features-compute
```

### Step 8: Lead Time Profiles

```bash
~/.local/bin/uv run python scripts/inventory/compute_lead_time_variability.py
```

### Step 9: Run Backtests (Tree + Foundation + Statistical + Deep Learning)

Sequential execution (safe for laptops). For parallel, append `&` to each and `wait` at the end.

```bash
# Tree models
~/.local/bin/uv run python scripts/ml/run_backtest.py
~/.local/bin/uv run python scripts/ml/run_backtest_catboost.py
~/.local/bin/uv run python scripts/ml/run_backtest_xgboost.py

# Foundation models
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos_bolt
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos2
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos2_enriched

# Statistical baselines
~/.local/bin/uv run python scripts/ml/run_backtest.py --model seasonal_naive
~/.local/bin/uv run python scripts/ml/run_backtest.py --model rolling_mean
~/.local/bin/uv run python scripts/ml/run_backtest_mstl.py

# Deep learning models
~/.local/bin/uv run python scripts/ml/run_backtest_dl.py --model nhits
~/.local/bin/uv run python scripts/ml/run_backtest_dl.py --model nbeats
```

### Step 10: Load Backtest Predictions

```bash
# Standard load (per-model index cycle):
~/.local/bin/uv run python scripts/etl/load_backtest_forecasts.py --all --replace

# Bulk load (~4x faster — single index drop/recreate across all models):
~/.local/bin/uv run python scripts/etl/load_backtest_forecasts.py --all --replace --bulk

# Load specific models only:
~/.local/bin/uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster catboost_cluster xgboost_cluster chronos --replace --bulk
```

### Step 11: Refresh Accuracy MVs

These MVs depend on backtest data loaded in Step 10:

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -c "
  REFRESH MATERIALIZED VIEW agg_accuracy_by_dim;
  REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive;
  REFRESH MATERIALIZED VIEW agg_dfu_coverage;
  REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive;
"
```

### Step 12: Champion Model Selection

```bash
~/.local/bin/uv run python scripts/ml/train_meta_learner.py --config config/forecasting/forecast_pipeline_config.yaml
~/.local/bin/uv run python scripts/ml/simulate_champion_strategies.py --config config/forecasting/forecast_pipeline_config.yaml
~/.local/bin/uv run python scripts/ml/run_champion_selection.py --config config/forecasting/forecast_pipeline_config.yaml
```

### Step 13: Baseline Planning Refresh

`fresh-all` finishes by rebuilding the baseline planning outputs that are derived from the refreshed dataset while preserving user-managed configuration tables:

```bash
make policy-all
make ss-all
make eoq-all
make health-all
```

### Validation

After completing the pipeline, verify key table counts:

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -c "
  SELECT 'dim_item' AS tbl, COUNT(*) FROM dim_item
  UNION ALL SELECT 'dim_location', COUNT(*) FROM dim_location
  UNION ALL SELECT 'dim_sku', COUNT(*) FROM dim_sku
  UNION ALL SELECT 'fact_sales_monthly', COUNT(*) FROM fact_sales_monthly
  UNION ALL SELECT 'fact_external_forecast_monthly', COUNT(*) FROM fact_external_forecast_monthly
  UNION ALL SELECT 'fact_inventory_snapshot', COUNT(*) FROM fact_inventory_snapshot
  UNION ALL SELECT 'backtest_lag_archive', COUNT(*) FROM backtest_lag_archive
  UNION ALL SELECT 'fact_safety_stock_targets', COUNT(*) FROM fact_safety_stock_targets
  UNION ALL SELECT 'fact_eoq_targets', COUNT(*) FROM fact_eoq_targets
  UNION ALL SELECT 'champion_rows', COUNT(*) FROM fact_external_forecast_monthly WHERE model_id = 'champion'
  ORDER BY tbl;
"
```

---

## Data Cleanup

### Remove Backtest Model Predictions

```bash
make backtest-list                                 # Row counts per model_id
make backtest-clean MODELS="--dry-run lgbm_cluster" # Preview before deleting
make backtest-clean MODELS="lgbm_cluster catboost_cluster"  # Delete specific tree models
make backtest-clean MODELS="chronos chronos_bolt chronos2 chronos2_enriched"  # Delete all foundation models
make backtest-clean MODELS="seasonal_naive rolling_mean mstl"  # Delete statistical baselines
make backtest-clean MODELS="nhits nbeats"          # Delete deep learning models
make backtest-clean MODELS="--all-backtest"        # Delete all non-external backtest models
```

Removes rows from `fact_external_forecast_monthly` and `backtest_lag_archive`, then refreshes all 5 dependent materialized views. `--all-backtest` never deletes `model_id='external'`.

### Remove Forecasts by Date Range

```bash
make forecast-clean-list                                               # Row counts by model + month

make forecast-clean ARGS="--before 2025-04-01 --dry-run"              # Preview
make forecast-clean ARGS="--before 2025-04-01 --model external"        # Delete external before Apr 2025
make forecast-clean ARGS="--between 2024-01-01 2024-07-01"             # Delete all models Jan–Jun 2024
make forecast-clean ARGS="--months 2024-03 2024-06 2024-09"            # Delete specific months
make forecast-clean ARGS="--months 2025-01 --model external"           # One month, one model
make forecast-clean ARGS="--after 2025-06-01 --forecast-only"          # Main table only
make forecast-clean ARGS="--before 2025-01-01 --date-column fcstdate --archive-only"  # Archive only
```

Accepted date formats: `YYYY-MM-DD`, `YYYY-MM`, `MM/DD/YYYY` (all normalized to month-start).

After any cleanup that affects champion/ceiling rows, re-run `make champion-select`.

---

## Read Replica Deployment (Item 24)

The API supports opt-in routing of read-only analytics queries to a Postgres
read replica. This is a SCAFFOLD: the application picks up the replica when
configured, but provisioning the actual replica is the operator's job.

### When to use it

Useful when the Customer Analytics tab (or other analytics-heavy reads) starts
contending with primary writes — typically once concurrent planner sessions
exceed ~20 or backtest loads run alongside the dashboard. A single read
replica typically doubles available analytics capacity without touching the
write path.

### Configuration

Set `READ_REPLICA_URL` in the API process environment:

```bash
READ_REPLICA_URL=postgres://reader:secret@replica.example.com:5433/demand_mvp
```

Optional sizing knobs (default to the primary pool's values):

- `READ_POOL_MIN_SIZE` (default `POOL_MIN_SIZE`, fallback `5`)
- `READ_POOL_MAX_SIZE` (default `POOL_MAX_SIZE`, fallback `50`)

When `READ_REPLICA_URL` is unset (the default), every endpoint falls through to
the primary pool — i.e. the configured-off code path is bit-for-bit identical
to having no replica code at all.

### Currently routed endpoints

The following endpoints opt in via `get_async_read_only_conn()`:

- `GET /customer-analytics/kpis`
- `GET /customer-analytics/map`
- `GET /customer-analytics/treemap`
- `GET /customer-analytics/heatmap`
- `GET /customer-analytics/channel-mix`
- `GET /customer-analytics/segment-trends`
- `GET /customer-analytics/ranking`

These are all read-only aggregates that tolerate eventual consistency. Other
read-only endpoints can be migrated incrementally as load shifts.

### Provisioning the replica

1. Create a streaming-replication standby on a separate host. See the Postgres
   docs: <https://www.postgresql.org/docs/16/warm-standby.html#STREAMING-REPLICATION>
2. Create a read-only role: `CREATE ROLE reader LOGIN PASSWORD '...' ;`
3. `GRANT CONNECT ON DATABASE demand_mvp TO reader; GRANT USAGE ON SCHEMA public TO reader; GRANT SELECT ON ALL TABLES IN SCHEMA public TO reader;`
4. Set `READ_REPLICA_URL` in the API environment, restart the API.
5. Verify analytics endpoints route via replica: tail API logs for
   `"Async read-replica pool opened on startup"`.

### Monitoring replication lag

```sql
-- On the primary:
SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn
FROM pg_stat_replication;

-- On the replica:
SELECT pg_last_xact_replay_timestamp() AS last_replayed,
       NOW() - pg_last_xact_replay_timestamp() AS lag;
```

Lag under ~30s is acceptable for the analytics endpoints listed above. If the
replica falls behind, set `READ_REPLICA_URL=` (empty) and restart the API to
fail back to the primary.

### When NOT to use the replica

Do NOT migrate endpoints that the user expects to reflect their own writes
immediately after a `POST` / `PUT` (e.g. saving a plan and refreshing the
view). The replica can lag the primary by seconds, so read-after-write flows
will appear to "lose" the update. The customer-analytics endpoints currently
routed are safe because they aggregate over months of history.

### Failure modes

- Replica unreachable on startup: warning logged, all endpoints fall back to
  the primary pool. App still serves traffic.
- `READ_REPLICA_URL` unparseable: warning logged, treated as unset.
- Replica falls behind during traffic: each query that lands on the replica
  returns slightly stale data — acceptable for analytics. To force traffic
  back to the primary, unset the env var and restart.

## Troubleshooting

### Environment / Setup

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` in backend tests | `make init` — installs pytest, httpx, pytest-asyncio |
| Frontend tests fail to start | `make ui-init` — installs vitest, @testing-library/react |
| API tests fail with DB errors | Verify `tests/api/conftest.py` has mock pool fixtures |
| `localStorage.clear is not a function` | jsdom localStorage mock required — see `frontend/src/hooks/__tests__/useTheme.test.ts` |

### Infrastructure

| Problem | Fix |
|---|---|
| API returns DB errors | Verify `.env` DB values and `make up` status |
| pgvector extension not found | Ensure `docker-compose.yml` uses `pgvector/pgvector:pg16`; rebuild: `make down && docker volume rm demand_pg_data && make up` |
| MLflow connection refused (port 5003) | Run `make up`; verify: `docker ps \| grep mlflow`; MLflow only runs when stack is up |

### Data Ingestion

| Problem | Fix |
|---|---|
| Forecast load fails with "missing data for column model_id" | `make normalize-forecast && make load-forecast` |
| Inventory normalize fails | Verify 14 CSVs in `data/input/`: `ls data/input/Inventory_Snapshot_*.csv \| wc -l` (expect 14); files must be UTF-8 CSV with columns `exec_date,item,loc,lead_time,tot_oh,tot_oh_oo,mtd_sls` |
| Inventory load fails | Apply DDL first: `make db-apply-inventory`; verify `ls -lh data/staged/inventory_clean.csv` |
| Inventory tab shows no data | Verify load: `docker compose exec -T postgres psql -U demand -d demand_mvp -c "SELECT COUNT(*) FROM fact_inventory_snapshot"`; refresh view: `make refresh-agg-inventory` |

### ML Pipelines

| Problem | Fix |
|---|---|
| Clustering fails | Load sales first: `make load-sales`; verify MLflow: `docker ps \| grep mlflow`; check: `ls -lh data/staged/clustering_features.csv` |
| Cluster assignments not updating | Preview with `--dry-run`; verify DFU key format matches database; check Postgres connection |
| Backtest fails | Run clustering first: `make cluster-all`; load sales: `make load-sales`; install deps: `uv sync` |
| Champion selection finds no DFUs | Load backtest predictions: `make backtest-load`; lower `min_dfu_rows` in `config/forecasting/forecast_pipeline_config.yaml` champion section; verify models exist: `SELECT DISTINCT model_id FROM fact_external_forecast_monthly` |
| Chat endpoint errors | Set `OPENAI_API_KEY` in `.env`; run `make generate-embeddings`; check API logs for rate limit errors |
| AI Planner errors | Set `ANTHROPIC_API_KEY` in `.env`; verify insight schema exists: `make ai-insights-schema`; check API logs for rate limit or tool dispatch errors |

### Exception Queue

| Problem | Fix |
|---|---|
| Exception queue is empty | Safety stock targets must exist first. Run `make ss-all` to compute safety stock targets, then `make exceptions-generate` to detect exceptions. Dependency chain: safety stock targets -> exception generation |
| `make exceptions-generate` finds no exceptions | Verify `fact_safety_stock_targets` has rows: `SELECT COUNT(*) FROM fact_safety_stock_targets`; check thresholds in `config/operations/exception_config.yaml` |

### Inventory Rebalancing

| Problem | Fix |
|---|---|
| `mv_network_balance` refresh fails | Ensure dependent tables are populated first: `fact_rebalancing_plan`, `dim_transfer_lane`; run `make rebalancing-refresh` after data is loaded |
| Rebalancing compute finds no candidates | Verify safety stock targets exist (`SELECT COUNT(*) FROM fact_safety_stock_targets`); verify inventory snapshots are loaded; check `config/inventory/rebalancing_config.yaml` thresholds are not too restrictive |
| Dry-run shows transfers but compute writes nothing | Confirm you are running `make rebalancing-compute` (not `make rebalancing-compute-dry`); check DB connectivity and table permissions |

### Platform Services (Specs 08-01 through 08-10)

| Problem | Fix |
|---|---|
| JWT auth returns 401 on all requests | Verify `JWT_SECRET` is set in `.env`; check token expiry; re-authenticate to get a fresh token |
| `bcrypt` or `PyJWT` import error | Run `make init` or `uv sync` to install new dependencies |
| Data quality checks return empty results | Run `make db-apply-sql` to ensure DDL 062-070 are applied; verify check catalog has entries via `GET /data-quality/catalog` |
| Cache not working (stale data) | Check `backend:` in `config/platform/cache_config.yaml`; for Redis, verify connectivity: `redis-cli ping` and `docker logs demandproject-redis-1`; InMemory cache clears on restart |
| Redis unreachable / "falling back to in-memory" warning | Cache transparently falls back; check `docker ps \| grep redis`, `docker logs demandproject-redis-1`, and `REDIS_URL` env var. Multi-worker deployments lose cross-worker hit-rate + single-flight protection until Redis returns |
| Notifications not sending | Verify channel is `enabled: true` in `config/platform/notification_config.yaml`; check SMTP credentials / Slack webhook URL |
| Webhook deliveries failing | Check delivery history via `GET /webhooks/deliveries`; verify target URL is reachable; check HMAC signature validation on receiver |
| Rate limiting too aggressive | Adjust limits in `config/api_governance_config.yaml`; increase `default_limit` or add per-endpoint overrides |
| Anonymous admin mode in production | Set `JWT_SECRET` in `.env` — without it, all requests bypass auth |

### Frontend / API

| Problem | Fix |
|---|---|
| Jobs tab returns HTML instead of JSON | Add `/jobs` proxy to `frontend/vite.config.ts`; restart `make ui`; verify backend: `curl http://localhost:8000/jobs/stats` |
| New API route returns HTML in UI | Add the path prefix to Vite proxy config in `frontend/vite.config.ts`; restart `make ui` |

---

## Key Dependencies Between Phases

```
Phase 1  (Schema)
    └─► Phase 1b (Auth/RBAC)  [auto-init on API start]
Phase 2  (Ingest)
    └─► Phase 2b (Data Quality Checks)
    └─► Phase 3  (Inv Planning)
    └─► Phase 4  (Clustering + Seasonality)  ──► Phase 5 (Backtesting)
                                                     └─► Phase 6 (Champion Select)
                                                             └─► Phase 7 (Prod Forecast)
    └─► Phase 8  (AI Insights)  [needs 3 + 6 + 7]
    └─► Phase 9  (Storyboard)   [needs 3]
    └─► Phase 9b (DQ Pipeline — auto)  [needs 2, runs every 4h via APScheduler]
    └─► Phase 9c (FVA Tracking)       [needs user interventions + actuals]
    └─► Phase 9d (S&OP Cycle)         [needs 3 + 7, multi-stage approval]
    └─► Phase 9e (Notifications + Webhooks)  [config-only, no data deps]
    └─► Phase 9f (Report Generation)  [needs loaded data for report queries]
Phase 10 (Start Services)
    └─► Cache layer active (Redis or in-memory fallback)
    └─► Query performance tracking active (fact_query_performance)
    └─► Auth/RBAC middleware enforced on mutation endpoints
```
