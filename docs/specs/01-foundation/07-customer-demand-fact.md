# Customer-Level Demand Fact Table

> Adds a monthly range-partitioned fact table at item + customer + location grain so the platform can track unconstrained customer demand, out-of-stock impact, and customer contribution analysis separately from inventory-constrained sales.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (backend only) |
| **Key Files** | `sql/110_create_fact_customer_demand_monthly.sql`, `scripts/etl/normalize_customer_demand_csv.py`, `scripts/etl/load_customer_demand_postgres.py`, `common/core/domain_specs.py`, `config/etl/etl_config.yaml`, `tests/unit/test_customer_demand_load.py` |

---

## 1. Problem Statement

`fact_sales_monthly` stores demand at item + customer_group + location + month grain, where `customer_group` is an aggregate ("ALL"). `fact_customer_demand_monthly` adds granular per-customer demand to support customer-level demand sensing, customer contribution analysis, out-of-stock (OOS) impact analysis, promotion lift measurement, and customer segmentation for inventory allocation.

Source data arrives as CSV files with columns `site, warehouse_no, item_no, customer_no, posting_prd, demand_cases, oos_cases` - volume is high (~100M+ rows across history), currently split by year with future loads monthly incremental.

## 2. Data Contract

### 2.1 Source File

**Current format:** `{YYYY}_customer_demand.csv` (one file per year)
**Future format:** `{YYYYMM}_customer_demand.csv` (one file per month)
**Location:** `data/input/` | **Delimiter:** Comma

| Source Column | Type | Description | Example |
|---|---|---|---|
| `site` | INT | Site identifier (maps to dim_location via locationdata.csv) | `1` |
| `warehouse_no` | INT | Warehouse within site | `14` |
| `item_no` | INT | Item number (FK to dim_item.item_id) | `28789` |
| `customer_no` | INT | Customer number (FK to dim_customer.customer_no, scoped to site) | `116` |
| `posting_prd` | INT | Posting period in YYYYMM format | `202601` |
| `demand_cases` | NUMERIC | Ordered quantity in cases (= true demand) | `1.000` |
| `oos_cases` | NUMERIC | Out-of-stock cases (unfulfilled portion of demand) | `0.000` |

### 2.2 Derived Columns

| Derived Column | Formula | Description |
|---|---|---|
| `sales_qty` | `MAX(0, demand_cases - oos_cases)` | Actual shipped/sold quantity |
| `demand_qty` | `MAX(0, demand_cases)` | Demand quantity (floored at 0) |
| `location_id` | Lookup: `site` -> `dim_location.site_id` -> `location_id` | Resolved location FK |
| `startdate` | `posting_prd` -> `YYYY-MM-01` date | First day of month |

### 2.3 Location Inference

The source `site` column maps directly to `dim_location.site_id` (all 37 values match 1:1). Multiple `location_id`s can share the same `site_id` - the normalize script picks the one with `primary_demand_location = 'Y'`. The `warehouse_no` column is NOT used for location resolution (it's a different numbering system); duplicate keys on `(item_id, customer_no, location_id, startdate)` from multiple warehouses within the same site are aggregated (quantities summed) during the load's per-partition GROUP BY.

## 3. Target Table Design

### 3.1 DDL

```sql
-- Grain: item_id + customer_no + location_id + startdate. Partitioned by
-- startdate (monthly) for efficient drop-and-reload.
CREATE TABLE fact_customer_demand_monthly (
    demand_sk       BIGSERIAL,
    demand_ck       TEXT NOT NULL,               -- composite key: item_id_customer_no_location_id_startdate
    item_id         TEXT NOT NULL,               -- FK -> dim_item.item_id
    customer_no     TEXT NOT NULL,               -- FK -> dim_customer.customer_no (scoped to site)
    site            TEXT NOT NULL,               -- site identifier (for dim_customer FK resolution)
    location_id     TEXT NOT NULL,               -- FK -> dim_location.location_id (resolved from site+warehouse)
    startdate       DATE NOT NULL,               -- first day of month (partition key)
    demand_qty      NUMERIC(18,4) NOT NULL,      -- ordered quantity (floored at 0)
    sales_qty       NUMERIC(18,4) NOT NULL,      -- shipped quantity = demand - oos (floored at 0)
    oos_qty         NUMERIC(18,4) NOT NULL DEFAULT 0, -- out-of-stock quantity
    load_ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_ts     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_customer_demand_month_start
        CHECK (startdate = date_trunc('month', startdate)::date),
    CONSTRAINT chk_customer_demand_qty_nonneg
        CHECK (demand_qty >= 0 AND sales_qty >= 0 AND oos_qty >= 0)
) PARTITION BY RANGE (startdate);

ALTER TABLE fact_customer_demand_monthly
    ADD CONSTRAINT uq_customer_demand_ck UNIQUE (demand_ck, startdate);
```

### 3.2 Indexes

Seven indexes cover the common query patterns: `item_id`, `customer_no`, `location_id`, `startdate`, `(item_id, location_id)`, `(item_id, customer_no)`, `(site, customer_no)`. Full DDL in `sql/110_create_fact_customer_demand_monthly.sql`.

### 3.3 Partitioning Strategy

Monthly range partitions matching the `fact_inventory_snapshot` pattern, auto-created during load (naming: `fact_customer_demand_monthly_YYYY_MM`). Incremental monthly loads drop and recreate the target partition atomically (`DROP TABLE IF EXISTS ...` then `CREATE TABLE ... PARTITION OF ... FOR VALUES FROM (...) TO (...)`). The initial historical load creates all partitions up front and bulk-loads per year file.

### 3.4 Composite Key

```
demand_ck = item_id || '_' || customer_no || '_' || location_id || '_' || startdate
```

Example: `28789_116_1401-BULK_2026-01-01`

## 4. ETL Pipeline

### 4.1 Normalize

Script: `scripts/etl/normalize_customer_demand_csv.py`. Auto-detects yearly (`YYYY_`) vs. monthly (`YYYYMM_`) filenames, joins `site` + `warehouse_no` to `dim_location` for `location_id`, casts `item_no` -> `item_id` and `customer_no` -> `customer_no` (TEXT), converts `posting_prd` (YYYYMM) to `startdate` (YYYY-MM-01), computes `demand_qty = MAX(0, demand_cases)`, `sales_qty = MAX(0, demand_cases - oos_cases)`, and `oos_qty = MAX(0, oos_cases)`, then writes `data/staged/customer_demand_clean.csv`.

### 4.2 Load

Script: `scripts/etl/load_customer_demand_postgres.py`

| Flag | Behavior |
|---|---|
| `--file PATH` | Specific CSV to load (default: `data/staged/customer_demand_clean.csv`) |
| `--month YYYY-MM` | Load only this month (drop + recreate partition) |
| `--replace` | Drop all partitions and reload (initial load) |
| `--dry-run` | Preview without writing |

`--replace` mode: COPY the clean CSV into an UNLOGGED staging table, discover distinct months, drop/recreate partitions, drop indexes and the unique constraint, then run 6 parallel per-month `INSERT ... GROUP BY item_id, customer_no, location_id, startdate` workers (the GROUP BY absorbs the ~0.3% multi-warehouse duplicate rate), and finally rebuild the unique constraint and 7 indexes. Direct partition INSERT bypasses parent-table routing overhead, and the GROUP BY runs per partition (~7M rows) rather than on the full staging table.

`--month` mode drops and recreates a single partition and reloads just that month's rows - no deletes needed since the whole partition is replaced atomically.

## 5. DomainSpec Registration

```python
# common/core/domain_specs.py

CUSTOMER_DEMAND_SPEC = DomainSpec(
    name="customer_demand",
    plural="customer_demand",
    table="fact_customer_demand_monthly",
    ck_field="demand_ck",
    business_key_fields=("item_id", "customer_no", "location_id", "startdate"),
    business_key_separator="_",
    columns=[
        "item_id", "customer_no", "site", "location_id", "startdate",
        "demand_qty", "sales_qty", "oos_qty",
    ],
    source_file="*_customer_demand.csv",
    clean_file="staged/customer_demand_clean.csv",
    search_fields=["item_id", "customer_no", "location_id", "site"],
    int_fields=set(),
    float_fields={"demand_qty", "sales_qty", "oos_qty"},
    date_fields={"startdate"},
    default_sort="startdate",
    source_delimiter=",",
    source_columns={
        "item_id": "item_no",
        "customer_no": "customer_no",
        "site": "site",
    },
)
```

Registered in `DOMAIN_SPECS` alongside the other 10 domains. `domain_order` in `etl_config.yaml` places `customer_demand` after `inventory`, before `sourcing`.

## 6. Makefile Targets

```makefile
normalize-customer-demand:
	$(UV) run python scripts/etl/normalize_customer_demand_csv.py

load-customer-demand:
	$(UV) run python scripts/etl/load_customer_demand_postgres.py --replace

load-customer-demand-month:
	$(UV) run python scripts/etl/load_customer_demand_postgres.py --month $(MONTH)

pipeline-customer-demand: normalize-customer-demand load-customer-demand
```

## 7. Volume and Performance (Actual)

| Metric | Actual |
|---|---|
| Source rows (3.25 years) | 297,449,085 |
| After dedup (GROUP BY) | 296,649,510 (0.27% duplicate rate) |
| Partitions | 39 (2023-01 through 2026-03) |
| Sites | 37 |
| Normalize step (streaming, 4 files, Docker, Apple Silicon) | ~8 min |
| Load step (`--replace`): COPY to staging, parallel INSERT (6 workers), rebuild constraint + 7 indexes | ~21 min |
| Monthly incremental (`--month`) | ~30s |

## 8. Validation & Data Quality

Post-load checks cover: row/item/customer/location counts per month, a negative-qty check (also enforced by the `chk_customer_demand_qty_nonneg` CHECK constraint), OOS rate per month, and orphan-FK checks against `dim_item` and `dim_customer`. A reconciliation query compares `SUM(sales_qty)` per item+location+month against `fact_sales_monthly.qty` - the customer-level table is higher granularity but should aggregate to similar totals; see `sql/110_create_fact_customer_demand_monthly.sql` and the normalize/load scripts for the exact queries.

## 9. API Endpoints (Live)

`CUSTOMER_DEMAND_SPEC` is registered in `DOMAIN_SPECS`, so the generic domain endpoints are live via `domains.py`'s catch-all router:

| Method | Path | Description |
|---|---|---|
| GET | `/domains/customer_demand/rows` | Paginated rows with filters |
| GET | `/domains/customer_demand/search` | Full-text search |
| GET | `/domains/customer_demand/summary` | Aggregated KPIs |

`fact_customer_demand_monthly` is also queried directly by dedicated routers, not just the generic domain endpoints:

- `api/routers/inventory/demand_history.py` - the `/demand-history/*` Demand History Workbench endpoints (reference, decomposition, comparison, workbench, matrix, matrix/drill)
- `api/routers/intelligence/customer_analytics/` - the kpis, geo, lifecycle, ranking, and segments sub-routers join it with `dim_customer` / `dim_item`
- `api/routers/intelligence/ai_planner.py` - a channel-filter subquery for AI insights, since `fact_sales_monthly` only holds the aggregated `customer_group` and cannot join `dim_customer`

## 10. Downstream Usage

| Consumer | How it uses customer demand |
|---|---|
| **Customer contribution analysis** | Top N customers by demand per item-location |
| **OOS impact scoring** | `oos_qty / demand_qty` per customer - identify chronic stockouts |
| **Demand sensing** | Customer-level signals for short-term forecast adjustment |
| **Allocation optimization** | Fair-share allocation proportional to customer demand |
| **Forecast disaggregation** | Split item-location forecast to customer level using historical shares |
