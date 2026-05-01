# Section 2 — Data Ingestion

This section covers the end-to-end ingestion path for the Supply Chain Command Center: from raw input CSVs that drop into `data/input/` through normalization, Postgres loading, materialized view (MV) refresh, and verification. It also documents how to add a new data source and how to diagnose the most common failure modes.

The single source of truth for ingestion ordering is `config/etl_config.yaml` (`domain_order:`). The orchestrator script is `scripts/etl/run_pipeline.py`, invoked through the `make pipeline-*` and `make setup-data` targets. Domain schemas live in `common/core/domain_specs.py`.

---

## 2.1 Input CSV Inventory

All raw files land in `data/input/`. The pipeline expects exact filename patterns — file globs (e.g., `Inventory_Snapshot_*.csv`, `*_customer_demand.csv`) discover monthly or yearly partitions automatically. The `time` domain is auto-generated (no input file required).

| # | Domain | Source filename | Delimiter | Target table | Key business fields | Notes / gotchas |
|---|---|---|---|---|---|---|
| 1 | `item` | `itemdata.csv` | `,` | `dim_item` | `item_id` | Source column is `item_no`; renamed to `item_id` during normalize. `class` is a reserved word — exposed in API as `class_`. |
| 2 | `location` | `locationdata.csv` | `,` | `dim_location` | `location_id`, `site_id` | `site_id` is the join key for customer-demand `site -> location_id` resolution. `primary_demand_location='Y'` wins ties when multiple `location_id`s share a `site_id`. |
| 3 | `customer` | `customerdata.csv` | `,` | `dim_customer` | (`site`, `customer_no`) | Composite business key. Premise/channel/chain attributes drive customer analytics. |
| 4 | `time` | _auto-generated_ (`_generated_time_2020_2035`) | n/a | `dim_time` | `date_key` | **No input file.** The normalize stage emits a calendar dimension covering 2020-01-01 through 2035-12-31. Dates outside that window will produce FK violations in fact loads — extend the generator if you need a different range. |
| 5 | `sku` | `dfu.txt` | `\|` (pipe) | `dim_sku` | (`item_id`, `customer_group`, `loc`) | Source columns are `DMDUNIT` / `DMDGROUP` / `U_*`; renamed during normalize. SKU = item + location pair (terminology: "SKU", not "DFU", in clustering contexts). |
| 6 | `sales` | `dfu_lvl2_hist.txt` | `\|` (pipe) | `fact_sales_monthly` | (`item_id`, `customer_group`, `loc`, `startdate`, `type`) | **Only `TYPE=1` rows are loaded** — others are filtered at load time. Forecast-related types are intentionally dropped. |
| 7 | `forecast` | `dfu_stat_fcst.txt` | `\|` (pipe) | `fact_external_forecast_monthly` | (`item_id`, `customer_group`, `loc`, `fcstdate`, `startdate`) | Forecast quantity column is `basefcst_pref` (not `qty`). `model_id` defaults to `'external'` for source-system forecasts; `UNIQUE(forecast_ck, model_id)` prevents duplicates. |
| 8 | `inventory` | `Inventory_Snapshot_YYYY_MM.csv` (one per month) | `,` | `fact_inventory_snapshot` | (`item_id`, `loc`, `snapshot_date`) | **Monthly range-partitioned** by `snapshot_date`. Source column rename: `item -> item_id`, `exec_date -> snapshot_date`, `lead_time -> lead_time_days`, `tot_oh -> qty_on_hand`, `tot_oh_oo -> qty_on_hand_on_order`, `mtd_sls -> mtd_sales`. Drop new files into `data/input/` and rerun `make pipeline-inventory-refresh`. |
| 9 | `customer_demand` | `YYYY_customer_demand.csv` (yearly) or `YYYYMM_customer_demand.csv` (monthly) | `,` | `fact_customer_demand_monthly` | (`item_id`, `customer_no`, `location_id`, `startdate`) | **Monthly range-partitioned** by `startdate`. Source columns: `site, warehouse_no, item_no, customer_no, posting_prd, demand_cases, oos_cases`. Loader resolves `site -> location_id` via `dim_location.site_id` (so `dim_location` MUST be loaded first). `posting_prd` (YYYYMM int) is converted to `startdate` (YYYY-MM-01). |
| 10 | `sourcing` | `sourcing.csv` | `,` | `dim_sourcing` | (`item_id`, `loc`, `source_cd`) | Item-location-supplier sourcing assignments. |
| 11 | `purchase_order` | `purchase_orders.csv` | `,` | `fact_purchase_orders` | (`po_number`, `item_id`, `loc`) | Many source columns are renamed (`purchase_order_no -> po_number`, `delivery_dt -> delivery_date`, etc. — see `PURCHASE_ORDER_SPEC.source_columns`). |

### Customer-demand derived columns

The customer-demand normalizer computes two derived quantities and clips them at zero:

```
demand_qty = MAX(0, demand_cases)
sales_qty  = MAX(0, demand_cases - oos_cases)
oos_qty    = oos_cases
```

Negative source values are silently floored to zero, so audit your inputs upstream if you suspect refund/return signal.

### Null normalization (all CSV-based domains)

The following sentinel values are normalized to SQL `NULL` during the normalize stage:

```
""   "null"   "none"   "na"   "n/a"
```

Matching is case-insensitive and trimmed.

---

## 2.2 Normalize Stage

Normalize reads the raw CSVs from `data/input/`, applies type casts, renames source columns to canonical target columns, normalizes nulls, and emits a `*_clean.csv` into `data/`. Nothing touches the database in this stage.

### One target to rule them all

```bash
make normalize-all
```

This runs every per-domain normalizer in order, including the customer-demand normalizer. Per-domain alternatives:

```bash
make normalize-item
make normalize-location
make normalize-customer
make normalize-time              # generates dim_time CSV (no input file read)
make normalize-dfu               # SKU dimension (dfu.txt -> sku_clean.csv)
make normalize-sales
make normalize-forecast
make normalize-inventory         # globs Inventory_Snapshot_*.csv -> inventory_clean.csv
make normalize-sourcing
make normalize-purchase-order
make normalize-customer-demand   # globs *_customer_demand.csv -> customer_demand_clean.csv
```

All except `inventory` and `customer_demand` are dispatched through `scripts/normalize_dataset_csv.py --dataset <name>`, which reads the matching `DomainSpec` from `common/core/domain_specs.py`. Inventory and customer-demand have dedicated scripts because they fan in across many input files.

### Output location

Clean CSVs land in `data/` (NOT `data/input/`) and follow the `clean_file` field of each `DomainSpec`. Examples:

| Domain | Clean file |
|---|---|
| item | `data/itemdata_clean.csv` |
| location | `data/locationdata_clean.csv` |
| customer | `data/customerdata_clean.csv` |
| time | `data/timedata_clean.csv` |
| sku | `data/sku_clean.csv` |
| sales | `data/sku_lvl2_hist_clean.csv` |
| forecast | `data/sku_stat_fcst_clean.csv` |
| inventory | `data/inventory_clean.csv` |
| customer_demand | `data/customer_demand_clean.csv` |
| sourcing | `data/sourcing_clean.csv` |
| purchase_order | `data/purchase_orders_clean.csv` |

Clean CSVs are gitignored. The `make clean-artifacts` target removes them.

---

## 2.3 Load Stage

The load stage streams clean CSVs into Postgres. Loaders use `psycopg3` with `%s` placeholders, default to UPSERT (`ON CONFLICT <ck> DO UPDATE`) keyed on the composite key field (`<domain>_ck`), and each load registers a row in `audit_load_batch` with the source file's SHA-256 hash for change detection.

### Load everything

```bash
make load-all
```

This is the canonical "load everything from the clean CSVs in dependency order" target. It runs domains in this exact order (matching `etl_config.yaml`):

1. `item` -> `dim_item`
2. `location` -> `dim_location`
3. `customer` -> `dim_customer`
4. `time` -> `dim_time`
5. `sku` -> `dim_sku`
6. `sales` -> `fact_sales_monthly` (only `TYPE=1` rows kept)
7. `forecast` -> `fact_external_forecast_monthly`
8. `inventory` -> `fact_inventory_snapshot` (per-month partitions auto-created)
9. `sourcing` -> `dim_sourcing`
10. `purchase_order` -> `fact_purchase_orders`
11. `customer_demand` -> `fact_customer_demand_monthly` (full replace; per-month partitions auto-created)

After all loads, `make load-all` chains `make refresh-agg` (refreshes `agg_sales_monthly`, `agg_forecast_monthly`, `agg_inventory_monthly`). Non-tier MVs are still stale until you run `make refresh-mvs-tiered` — see Section 2.4.

### Per-domain loads

```bash
make load-item
make load-location
make load-customer
make load-time
make load-dfu                    # loads dim_sku
make load-sales                  # also refreshes agg_sales_monthly
make load-forecast               # UPSERT mode; also refreshes agg_forecast_monthly
make load-forecast-replace       # truncates fact_external_forecast_monthly first
make load-forecast-replace-no-archive   # also skips the backtest_lag_archive write
make load-inventory              # also refreshes agg_inventory_monthly
make load-sourcing
make load-purchase-order
make load-customer-demand                 # full --replace: drop all partitions, reload
make load-customer-demand-month MONTH=YYYY-MM    # drop+reload one partition
```

### Truncate-and-load semantics

| Loader | Default mode | `--replace` flag | Notes |
|---|---|---|---|
| Dimensions (item, location, customer, time, sku, sourcing) | UPSERT on `<ck>` | n/a | Idempotent. Re-running adds new rows and updates changed ones. |
| `sales` | UPSERT on `sales_ck` | n/a | Filters source to `TYPE=1` only. |
| `forecast` | UPSERT on (`forecast_ck`, `model_id`) | yes (`make load-forecast-replace`) | The dual-path archive write copies into `backtest_lag_archive` BEFORE staging mutation — see CLAUDE.md execution-lag rule. Use `--skip-archive` only for fast dev iterations. |
| `inventory` | UPSERT into per-month partitions; partitions auto-created | n/a | Per-file CSV; reloading the same `Inventory_Snapshot_YYYY_MM.csv` is idempotent. |
| `purchase_order` | UPSERT on `po_ck` | n/a | |
| `customer_demand` | UPSERT into per-month partitions | yes (`make load-customer-demand`) | Default Make target uses `--replace` (drops all partitions, recreates them, parallel INSERT). Use `make load-customer-demand-month MONTH=2026-01` to reload one partition without disturbing the rest. |

### Partition lifecycle (inventory & customer_demand)

Both partitioned facts use **monthly range partitioning** keyed on `snapshot_date` and `startdate` respectively. The loaders:

1. Discover the months present in the staged data.
2. Call `_ensure_partition_exists(month_start)` per month — creates `<table>_YYYY_MM` if missing.
3. INSERT each partition in parallel (one connection per worker, up to 6).

If you need to add data for a month outside the existing range, the loader creates the partition automatically. Manual `CREATE TABLE ... PARTITION OF` is never required.

### FK and load-order rules

- Dimensions MUST be loaded before facts that FK into them.
- `customer_demand` MUST run after `location` because it joins on `dim_location.site_id` to resolve `location_id`.
- `forecast` typically runs after `sku` because downstream MVs join on `dim_sku`.
- `time` is auto-generated; if you skip it, every fact load that joins on `dim_time` will fail with FK errors.

The `make load-all` ordering and the parallel two-wave dispatch in `scripts/etl/run_pipeline.py` (dimensions wave 1, facts wave 2) both encode these constraints.

---

## 2.4 Materialized View Refresh

Materialized views are refreshed in **dependency tiers**: base aggregates first, then the derived MVs that join on them. The `etl_config.yaml` `mv_refresh:` block maps each domain to the MVs it directly affects, and `always_refresh:` lists cross-domain MVs (`mv_dq_dashboard`, `mv_control_tower_kpis`) that run after any pipeline.

### Single command: tiered refresh

```bash
make refresh-mvs-tiered
```

This iterates the following list in order, attempting `REFRESH MATERIALIZED VIEW CONCURRENTLY` first and falling back to a non-concurrent refresh on first run (when the MV has no data and no unique index populated yet):

```
Tier 1 (base aggregates)
  agg_sales_monthly
  agg_forecast_monthly
  agg_inventory_monthly

Tier 2 (inventory derivatives)
  mv_inventory_forecast_monthly
  mv_fill_rate_monthly
  mv_intramonth_stockout
  mv_supplier_performance
  mv_supplier_po_performance
  mv_po_lead_time_analysis

Tier 3 (accuracy & coverage)
  agg_accuracy_by_dim
  agg_dfu_coverage

Tier 4 (composite cross-domain)
  mv_inventory_health_score
  mv_control_tower_kpis
  mv_integrated_planning_targets
```

Missing MVs are reported and skipped — the target tolerates partial schemas during incremental rollouts.

### Targeted refreshes

```bash
make refresh-agg               # Tier 1 only (sales + forecast + inventory aggregates)
make refresh-agg-sales
make refresh-agg-forecast
make refresh-agg-inventory
make refresh-accuracy-mvs      # post-backtest accuracy MVs
make refresh-inv-backtest      # mv_inventory_forecast_monthly only
make refresh-customer-mv       # MVs depending on customer_demand
```

After ANY pipeline run that mutates source tables, run `make refresh-mvs-tiered` once at the end. The KPI dashboards and Control Tower read from MVs, not from the base tables — stale MVs are the most common cause of "the data loaded but the UI shows old numbers" reports.

---

## 2.5 Full Pipelines

Wrappers that chain normalize + load + refresh, with change detection or full-reload semantics.

### Pipeline orchestrators

```bash
make pipeline-full           # Full reload of all domains, parallel two-wave (dims then facts)
make pipeline-refresh        # Incremental: hash-compare clean CSVs vs audit_load_batch, reload only changed
make pipeline-inventory      # Full reload of inventory domain only
make pipeline-inventory-refresh   # Incremental inventory refresh (only changed Inventory_Snapshot_*.csv files)
make pipeline-customer-demand     # normalize-customer-demand + load-customer-demand (full replace)
```

Under the hood these all dispatch to `scripts/etl/run_pipeline.py`:

| Make target | Equivalent script call |
|---|---|
| `pipeline-full` | `python scripts/etl/run_pipeline.py --mode full --parallel` |
| `pipeline-refresh` | `python scripts/etl/run_pipeline.py --mode refresh` |
| `pipeline-inventory` | `python scripts/etl/run_pipeline.py --mode full --domains inventory` |
| `pipeline-inventory-refresh` | `python scripts/etl/run_pipeline.py --mode refresh --domains inventory` |

The orchestrator reads `domain_order` from `etl_config.yaml`, computes per-domain MV lists from `mv_refresh`, deduplicates, and refreshes everything (sequentially or in parallel based on `parallel.mv_refresh_workers`).

### Higher-level setup wrappers

```bash
make setup-data              # = pipeline-full (data only, no ML, no inventory planning) — ~30 min
make setup-features          # setup-data + features-compute + cluster-all + LT/ABC-XYZ/demand-signals
make setup-planning          # setup-data + setup-inv-planning (no ML)
make setup-all               # setup-backtest + setup-inv-planning + setup-demand-planning + setup-ops (~4-6 hours)
```

For a clean slate, see `make fresh-load` (truncate config-preserving tables, then `normalize-all + load-all + refresh-mvs-tiered`) and `make fresh-all` (everything including ML and baseline planning).

### Refresh vs. Full — when to use which

| Use | Why |
|---|---|
| `pipeline-refresh` | Daily or hourly cron. Skips domains whose clean-CSV SHA-256 matches the last successful `audit_load_batch` row. |
| `pipeline-full` | After source-system schema change, after `make db-truncate-data`, or whenever you suspect drift between `audit_load_batch` and reality. |
| `pipeline-inventory-refresh` | When you drop new `Inventory_Snapshot_YYYY_MM.csv` files into `data/input/` and only those need to load. |

Change detection compares the SHA-256 of the **clean** CSV (post-normalize) against the most recent `completed` `audit_load_batch` row per domain. Inventory uses per-file hashing (`source_file` column on the audit row) so partial-month adds are cheap.

---

## 2.6 Adding a New Data Source

Per CLAUDE.md "File Placement Rules" and "Data Loading" — every new input file MUST be a first-class domain. Standalone scripts that load data outside `make load-all` are forbidden because they break `make db-truncate-data` and `make fresh-load`.

### Checklist

1. **Define the `DomainSpec`** in `common/core/domain_specs.py`:
   - `name`, `plural`, `table`, `ck_field`, `business_key_field` (or `business_key_fields` for composites).
   - `columns` list (ordered — drives the CSV header).
   - `source_file`, `clean_file` filenames.
   - `source_columns` dict for any source -> target renames.
   - `int_fields`, `float_fields`, `date_fields`, `bool_fields` sets for type casting.
   - `source_delimiter` (`,` for CSV, `|` for pipe-delimited extracts).
   - Register in the `DOMAIN_SPECS` dict at the bottom of the file.

2. **Add to `etl_config.yaml`**:
   - Append to `domain_order:` in the correct dependency position (dims before facts).
   - If the new domain feeds an MV, add the mapping under `mv_refresh:`.

3. **Wire up the Makefile**:
   - Add `normalize-<name>:` and `load-<name>:` targets (boilerplate: `$(UV) python scripts/normalize_dataset_csv.py --dataset <name>` / `... load_dataset_postgres.py --dataset <name>`).
   - Append the new normalize target to the `normalize-all:` dependency line.
   - Add the load command to the `load-all:` recipe in dependency order.

4. **Schema migration**:
   - Create the next-numbered DDL file in `sql/` (e.g., `sql/090_create_<table>.sql`).
   - Apply it via `make db-apply-sql` (or the bespoke `db-apply-*` target).

5. **Truncate & fresh-load coverage**:
   - Add a `TRUNCATE TABLE <name> CASCADE;` line in the correct FK group of `make db-truncate-data` (Makefile lines ~1471-1571).
   - If the new table feeds MVs, ensure `make refresh-mvs-tiered` covers them — add to the appropriate tier in the loop.
   - Verify `make fresh-load` end-to-end on a scratch DB.

6. **Docs**:
   - Update `docs/RUNBOOK.md` "Database Cleanup & Fresh Recreate" sections.
   - Update `docs/ARCHITECTURE.md` data-model section.
   - Add to this manual's Section 2.1 inventory table.

7. **Tests**:
   - Add a unit test in `tests/unit/` covering normalize edge cases (null sentinels, type coercion).
   - If you exposed it via the generic `/domains/{domain}/*` endpoint, no router test is required; if you added a dedicated router, add an `tests/api/` test using the standard `make_pool()` factory.

For domains with their own load scripts (like `inventory` and `customer_demand`), place them under `scripts/etl/` per CLAUDE.md File Placement Rules, and gate them behind a Makefile target invoked from `load-all`.

---

## 2.7 Verification

After every pipeline run, verify with the standard health checks.

### Quick health probe

```bash
make health           # alias for `make check-all`
make check-all        # = check-db + check-api
make check-db         # estimated row counts for all major tables + per-model forecast coverage
make check-api        # curls /health and a sample endpoint per dimension
```

`check-db` prints estimated row counts (from `pg_class.reltuples`) for: `dim_item`, `dim_location`, `dim_customer`, `dim_time`, `dim_sku`, `fact_sales_monthly`, `fact_external_forecast_monthly`, `fact_customer_demand_monthly`, `fact_inventory_snapshot`, `fact_production_forecast`, `fact_purchase_orders`, `backtest_lag_archive`, `champion_experiment`, `cluster_experiment`, `lgbm_tuning_run`, `job_history`. It then breaks down `fact_external_forecast_monthly` by `model_id` so you can confirm `external` rows landed and which backtest models are present.

### Spot-check the audit log

```sql
SELECT domain, source_file, status, rows_loaded, completed_at
FROM audit_load_batch
WHERE completed_at > NOW() - INTERVAL '1 day'
ORDER BY completed_at DESC;
```

Every `make load-*` and every `pipeline-*` writes one row per domain (or per source file for inventory) with status `running` -> `completed` or `failed`.

### Verify MV freshness

```sql
SELECT relname, pg_stat_get_last_vacuum_time(c.oid) AS last_vacuum,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS size
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND relkind = 'm'
ORDER BY relname;
```

If the UI shows stale numbers, run `make refresh-mvs-tiered` first before assuming the load itself failed.

---

## 2.8 Common Errors

### "FK violation on dim_time"

**Symptom**: Sales/forecast/inventory load fails with `insert or update on table "fact_*" violates foreign key constraint "*_date_key_fkey"`.

**Cause**: A source row has a `startdate` / `snapshot_date` / `fcstdate` outside the auto-generated `dim_time` range (2020-01-01 through 2035-12-31).

**Fix**: Either trim the source data, or extend the `dim_time` generator (`scripts/normalize_dataset_csv.py --dataset time` builder) to cover the new range, then `make load-time` and re-run the failed fact load.

### "Partition does not exist" / "no partition of relation found for row"

**Symptom**: `fact_inventory_snapshot` or `fact_customer_demand_monthly` insert fails with a partition-routing error.

**Cause**: Race condition with manual SQL, OR the loader was bypassed and a raw COPY was attempted.

**Fix**: Always use the supplied loaders (`make load-inventory`, `make load-customer-demand`, `make load-customer-demand-month MONTH=YYYY-MM`). They auto-create the per-month partition before the COPY/INSERT.

### "duplicate key value violates unique constraint" on forecast load

**Symptom**: `fact_external_forecast_monthly` load fails on the `(forecast_ck, model_id)` unique constraint.

**Cause**: Source extract contains true duplicates, or you ran `make load-forecast` twice with non-`UPSERT`-friendly data.

**Fix**: Run `make load-forecast-replace` to truncate and reload. If duplicates persist after that, the source extract is the problem — dedup upstream.

### Type cast failures during load

**Symptom**: `invalid input syntax for type integer/float/date` on COPY.

**Cause**: A source value is not in `NULL_SENTINELS` (`""`, `"null"`, `"none"`, `"na"`, `"n/a"`) and not a parseable number/date. Common culprits: localized number formats with commas, dates in `MM/DD/YYYY` instead of ISO.

**Fix**: Either fix the upstream extract or add a transformation to the relevant `normalize_*` script. Do not modify `data/<file>_clean.csv` by hand — it is regenerated on every `make normalize-*`.

### "demand_qty is negative" or zero everywhere in customer_demand

**Symptom**: Customer-demand records show suspicious zeros even when source had values.

**Cause**: `demand_qty = MAX(0, demand_cases)` and `sales_qty = MAX(0, demand_cases - oos_cases)` — negative `demand_cases` or `oos_cases > demand_cases` will produce zeros.

**Fix**: Audit the source extract. The clip-at-zero is intentional (the downstream forecast pipeline cannot consume negative demand) but it hides upstream issues. Verify against `oos_qty` to see if OOS is the cause.

### `site` column resolves to NULL `location_id` in customer_demand

**Symptom**: `fact_customer_demand_monthly` rows have `location_id IS NULL`.

**Cause**: The `site` value in the source CSV is not present in `dim_location.site_id`, or `dim_location` was not loaded before customer demand.

**Fix**: Confirm `make load-location` ran successfully. Then check `SELECT DISTINCT site FROM <staging> EXCEPT SELECT site_id FROM dim_location;`. Add the missing locations to `locationdata.csv` and rerun `make load-location` followed by `make load-customer-demand`.

### "Sales rows missing" after load

**Symptom**: `fact_sales_monthly` row count is much smaller than the source CSV.

**Cause**: The loader filters to `TYPE=1` only. Any other TYPE values (often source-system forecast or budget rows that share the schema) are intentionally dropped.

**Fix**: Confirm with `SELECT type, COUNT(*) FROM <staging> GROUP BY type;` against the source extract. If you genuinely need other TYPEs, that is a schema change — file an issue, do not just edit the loader.

### Stale data in UI after a successful load

**Symptom**: `make load-all` reports success; `check-db` shows expected row counts; UI dashboards still show old data.

**Cause**: MVs were not refreshed. `make load-all` only refreshes Tier 1 aggregates (`refresh-agg`); Tier 2-4 MVs require `make refresh-mvs-tiered`.

**Fix**: `make refresh-mvs-tiered`. For control tower / health score / integrated targets specifically, this is mandatory.

### `audit_load_batch` shows `running` rows from a killed process

**Symptom**: `pipeline-refresh` thinks domains are still loading and skips them.

**Fix**: Manually mark the orphaned rows: `UPDATE audit_load_batch SET status='failed' WHERE status='running' AND started_at < NOW() - INTERVAL '1 hour';` then re-run.

---

## 2.9 Reference

- **Orchestrator**: `scripts/etl/run_pipeline.py`
- **Per-domain normalize**: `scripts/normalize_dataset_csv.py` (and `scripts/normalize_inventory_csv.py`, `scripts/etl/normalize_customer_demand_csv.py`)
- **Per-domain load**: `scripts/load_dataset_postgres.py` (and `scripts/etl/load_customer_demand_postgres.py`)
- **Domain schemas**: `common/core/domain_specs.py` — `DOMAIN_SPECS` dict
- **Pipeline config**: `config/etl_config.yaml`
- **Makefile**: `Makefile` (sections "Normalize / Load", "Pipelines", "Database Cleanup & Fresh Recreate")
- **Cleanup runbook**: `docs/RUNBOOK.md` "Database Cleanup & Fresh Recreate"
- **Critical rules**: `CLAUDE.md` "Data Loading" section
