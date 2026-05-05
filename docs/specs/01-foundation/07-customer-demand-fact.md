# Customer-Level Demand Fact Table

> Adds a monthly range-partitioned fact table at item + customer + location grain so the platform can track unconstrained customer demand, out-of-stock impact, and customer contribution analysis separately from inventory-constrained sales.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (backend only) |
| **Key Files** | `sql/110_create_fact_customer_demand_monthly.sql`, `scripts/etl/normalize_customer_demand_csv.py`, `scripts/etl/load_customer_demand_postgres.py`, `common/core/domain_specs.py` |

---

## 1. Problem Statement

The current `fact_sales_monthly` stores demand at the **item + customer_group + location + month** grain, where `customer_group` is an aggregate ("ALL"). We need granular **customer-level** demand to support:

- Customer-level demand sensing and forecasting
- Customer contribution analysis (which customers drive volume for each item)
- Out-of-stock (OOS) impact analysis at the customer level
- Promotion lift measurement per customer
- Customer segmentation for inventory allocation

Source data arrives as CSV files with columns: `site, warehouse_no, item_no, customer_no, posting_prd, demand_cases, oos_cases`. Volume is high (~100M+ rows across history). Files are currently split by year but future loads will be monthly incremental.

---

## 2. Data Contract

### 2.1 Source File

**Current format:** `{YYYY}_customer_demand.csv` (one file per year)
**Future format:** `{YYYYMM}_customer_demand.csv` (one file per month)
**Location:** `data/input/`
**Delimiter:** Comma

**Source columns:**

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
| `location_id` | Lookup: `site` → `dim_location.site_id` → `location_id` | Resolved location FK |
| `startdate` | `posting_prd` → `YYYY-MM-01` date | First day of month |

### 2.3 Location Inference

The source `site` column maps directly to `dim_location.site_id` (all 37 values match 1:1).
Multiple `location_id`s share the same `site_id` — the normalize script picks the one with `primary_demand_location = 'Y'`.

```
Source: site=1 → dim_location.site_id=1 → location_id="1401-BULK" (primary_demand_location=Y)
Source: site=85 → dim_location.site_id=85 → location_id="7801-DENV" (primary_demand_location=Y)
```

The `warehouse_no` column is NOT used for location resolution (it's a different numbering system). However, multiple `warehouse_no` values within the same `site` can generate duplicate keys on `(item_id, customer_no, location_id, startdate)` — these are aggregated (quantities summed) during the load's per-partition GROUP BY.

---

## 3. Target Table Design

### 3.1 DDL

```sql
-- Monthly range-partitioned customer-level demand fact table.
-- Grain: item_id + customer_no + location_id + startdate
-- Partitioned by startdate (monthly) for efficient drop-and-reload.

CREATE TABLE fact_customer_demand_monthly (
    demand_sk       BIGSERIAL,
    demand_ck       TEXT NOT NULL,               -- composite key: item_id_customer_no_location_id_startdate
    item_id         TEXT NOT NULL,               -- FK → dim_item.item_id
    customer_no     TEXT NOT NULL,               -- FK → dim_customer.customer_no (scoped to site)
    site            TEXT NOT NULL,               -- site identifier (for dim_customer FK resolution)
    location_id     TEXT NOT NULL,               -- FK → dim_location.location_id (resolved from site+warehouse)
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

-- Unique constraint per partition (PostgreSQL requirement for partitioned tables)
-- Applied to each partition individually via template or auto-propagation.
ALTER TABLE fact_customer_demand_monthly
    ADD CONSTRAINT uq_customer_demand_ck UNIQUE (demand_ck, startdate);
```

### 3.2 Indexes

```sql
CREATE INDEX idx_cust_demand_item ON fact_customer_demand_monthly (item_id);
CREATE INDEX idx_cust_demand_customer ON fact_customer_demand_monthly (customer_no);
CREATE INDEX idx_cust_demand_location ON fact_customer_demand_monthly (location_id);
CREATE INDEX idx_cust_demand_startdate ON fact_customer_demand_monthly (startdate);
CREATE INDEX idx_cust_demand_item_loc ON fact_customer_demand_monthly (item_id, location_id);
CREATE INDEX idx_cust_demand_item_cust ON fact_customer_demand_monthly (item_id, customer_no);
CREATE INDEX idx_cust_demand_site_cust ON fact_customer_demand_monthly (site, customer_no);
```

### 3.3 Partitioning Strategy

Monthly range partitions matching the `fact_inventory_snapshot` pattern:

```sql
-- Auto-created during load. Naming: fact_customer_demand_monthly_YYYY_MM
CREATE TABLE fact_customer_demand_monthly_2024_01
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE fact_customer_demand_monthly_2024_02
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- ... auto-created per month as data arrives
```

**Drop-and-reload per month:**
- For incremental monthly loads, drop the partition and recreate:
  ```sql
  DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_01;
  CREATE TABLE fact_customer_demand_monthly_2026_01
      PARTITION OF fact_customer_demand_monthly
      FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
  -- Then bulk load January 2026 data
  ```
- For initial historical load, create all partitions and bulk load per year file.

### 3.4 Composite Key

```
demand_ck = item_id || '_' || customer_no || '_' || location_id || '_' || startdate
```

Example: `28789_116_1401-BULK_2026-01-01`

---

## 4. ETL Pipeline

### 4.1 File Discovery

```
Initial load:   data/input/2024_customer_demand.csv
                 data/input/2025_customer_demand.csv
                 data/input/2026_customer_demand.csv

Monthly load:   data/input/202601_customer_demand.csv
                data/input/202602_customer_demand.csv
```

The script auto-detects whether the filename is yearly (`YYYY_`) or monthly (`YYYYMM_`).

### 4.2 Normalize Step

Script: `scripts/etl/normalize_customer_demand_csv.py`

1. Read source CSV(s) from `data/input/`
2. Load `dim_location` lookup (site_id → location_id mapping)
3. For each row:
   - Map `site` + `warehouse_no` → `location_id` via dim_location join
   - Convert `posting_prd` (YYYYMM int) → `startdate` (YYYY-MM-01 date)
   - Compute `demand_qty = MAX(0, demand_cases)`
   - Compute `sales_qty = MAX(0, demand_cases - oos_cases)`
   - Compute `oos_qty = MAX(0, oos_cases)`
   - Map `item_no` → `item_id` (TEXT)
   - Map `customer_no` → TEXT
4. Write normalized CSV to `data/staged/customer_demand_clean.csv`

**Column mapping:**

| Source | Target | Transform |
|---|---|---|
| `item_no` | `item_id` | Cast to TEXT |
| `customer_no` | `customer_no` | Cast to TEXT |
| `site` | `site` | Cast to TEXT |
| `site` + `warehouse_no` | `location_id` | Lookup via dim_location |
| `posting_prd` | `startdate` | `YYYYMM` → `YYYY-MM-01` |
| `demand_cases` | `demand_qty` | `MAX(0, value)` |
| `demand_cases - oos_cases` | `sales_qty` | `MAX(0, value)` |
| `oos_cases` | `oos_qty` | `MAX(0, value)` |

### 4.3 Load Step

Script: `scripts/etl/load_customer_demand_postgres.py`

**Arguments:**
```
--file PATH           # Specific CSV to load (default: data/staged/customer_demand_clean.csv)
--month YYYY-MM       # Load only this month (drop + recreate partition)
--replace             # Drop all partitions and reload (for initial load)
--dry-run             # Preview without writing
```

**Load algorithm (--replace mode):**

```
1. COPY entire clean CSV into UNLOGGED staging table     (~2 min for 297M rows)
2. SELECT distinct months from staging                    (~30s)
3. Drop all existing partitions, recreate per month       (~1s)
4. Drop all indexes + unique constraint                   (~1s)
5. Parallel INSERT — 6 workers, one per month:            (~11 min for 297M rows)
   INSERT INTO partition_YYYY_MM
     SELECT ... GROUP BY item_id, customer_no, location_id, startdate
     FROM staging WHERE startdate IN [month range]
   GROUP BY handles ~0.3% duplicates (same key from different warehouses)
6. Rebuild unique constraint + 7 indexes                  (~8 min)
7. Drop staging table
```

**Key performance decisions:**
- Direct partition INSERT (bypasses parent table routing overhead)
- GROUP BY per partition (~7M rows each) not on full 297M staging table
- No indexes during INSERT — rebuilt after in single pass
- 6 parallel connections for INSERT (I/O-bound, not CPU)

**Actual performance (297M rows, Docker, Apple Silicon):**

| Step | Time |
|---|---|
| COPY to staging | 119s |
| Month discovery | 33s |
| Parallel INSERT (6 workers, 39 months) | 646s |
| Rebuild indexes | ~480s |
| **Total** | **~21 min** |

### 4.4 Monthly Incremental Load

For future monthly files (`202603_customer_demand.csv`):

```bash
# Normalize the month file
make normalize-customer-demand MONTH=202603

# Drop the March 2026 partition and reload
make load-customer-demand MONTH=2026-03
```

This drops `fact_customer_demand_monthly_2026_03` and recreates it with fresh data. No deletes needed — the entire partition is replaced atomically.

---

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

---

## 6. Makefile Targets

```makefile
# Normalize
normalize-customer-demand:
	$(UV) run python scripts/etl/normalize_customer_demand_csv.py

# Load (full replace — initial load)
load-customer-demand:
	$(UV) run python scripts/etl/load_customer_demand_postgres.py --replace

# Load single month (incremental — future monthly loads)
load-customer-demand-month:
	$(UV) run python scripts/etl/load_customer_demand_postgres.py --month $(MONTH)

# Both
pipeline-customer-demand: normalize-customer-demand load-customer-demand
```

Add to `etl_config.yaml` domain_order (after `inventory`, before `sourcing`):
```yaml
domain_order:
  - item
  - location
  - customer
  - time
  - sku
  - sales
  - forecast
  - inventory
  - customer_demand    # NEW
  - sourcing
  - purchase_order
```

---

## 7. Volume and Performance (Actual)

| Metric | Actual |
|---|---|
| Source rows (3.25 years) | 297,449,085 |
| After dedup (GROUP BY) | 296,649,510 |
| Duplicate rate | 0.27% (from multi-warehouse fulfillment) |
| Rows per month | 6-10M |
| Rows per year | ~90M |
| Partitions | 39 (2023-01 through 2026-03) |
| Sites | 37 |

**Load performance (Docker, Apple Silicon):**

| Step | Time |
|---|---|
| Normalize (streaming, 4 files) | ~8 min |
| COPY to UNLOGGED staging | ~2 min |
| Parallel INSERT (6 workers, GROUP BY per partition) | ~11 min |
| Rebuild unique constraint + 7 indexes | ~8 min |
| **Total initial load** | **~21 min** |
| **Monthly incremental** (`--month`) | **~30s** |

**Performance design:**
- Parallel per-partition INSERT bypasses parent routing overhead
- GROUP BY per partition (~7M rows) not on full 297M staging table
- Indexes dropped during INSERT, rebuilt after in single pass
- Monthly `--month` mode: drop+recreate single partition — 30s for ~7M rows

---

## 8. Validation & Data Quality

### 8.1 Post-Load Checks

```sql
-- Row count per month
SELECT date_trunc('month', startdate) AS month,
       COUNT(*) AS rows,
       COUNT(DISTINCT item_id) AS items,
       COUNT(DISTINCT customer_no) AS customers,
       COUNT(DISTINCT location_id) AS locations
FROM fact_customer_demand_monthly
GROUP BY 1 ORDER BY 1;

-- Negative qty check (should be zero — enforced by CHECK constraint)
SELECT COUNT(*) FROM fact_customer_demand_monthly
WHERE demand_qty < 0 OR sales_qty < 0;

-- OOS rate
SELECT date_trunc('month', startdate) AS month,
       ROUND(100.0 * SUM(oos_qty) / NULLIF(SUM(demand_qty), 0), 2) AS oos_pct
FROM fact_customer_demand_monthly
GROUP BY 1 ORDER BY 1;

-- Orphan items (no dim_item match)
SELECT DISTINCT cd.item_id
FROM fact_customer_demand_monthly cd
LEFT JOIN dim_item di ON di.item_id = cd.item_id
WHERE di.item_id IS NULL;

-- Orphan customers (no dim_customer match)
SELECT DISTINCT cd.site, cd.customer_no
FROM fact_customer_demand_monthly cd
LEFT JOIN dim_customer dc ON dc.site = cd.site AND dc.customer_no = cd.customer_no
WHERE dc.customer_no IS NULL;
```

### 8.2 Reconciliation with fact_sales_monthly

```sql
-- Total demand by item+location+month should be close to (but not equal to)
-- fact_sales_monthly totals. The customer-level table will be higher granularity
-- but should aggregate to similar totals.
SELECT
    cd.item_id, cd.location_id AS loc, cd.startdate,
    SUM(cd.sales_qty) AS cust_total,
    s.qty AS agg_total,
    ABS(SUM(cd.sales_qty) - s.qty) AS delta
FROM fact_customer_demand_monthly cd
JOIN fact_sales_monthly s
    ON s.item_id = cd.item_id
    AND s.loc = cd.location_id
    AND s.startdate = cd.startdate
GROUP BY cd.item_id, cd.location_id, cd.startdate, s.qty
HAVING ABS(SUM(cd.sales_qty) - s.qty) > 0.01
ORDER BY delta DESC
LIMIT 20;
```

---

## 9. API Endpoints (Future)

Following the generic domain pattern:

| Method | Path | Description |
|---|---|---|
| GET | `/domains/customer_demand/rows` | Paginated rows with filters |
| GET | `/domains/customer_demand/search` | Full-text search |
| GET | `/domains/customer_demand/summary` | Aggregated KPIs |

These will be served automatically by `domains.py` once the DomainSpec is registered.

---

## 10. Downstream Usage

| Consumer | How it uses customer demand |
|---|---|
| **Customer contribution analysis** | Top N customers by demand per item-location |
| **OOS impact scoring** | `oos_qty / demand_qty` per customer — identify chronic stockouts |
| **Demand sensing** | Customer-level signals for short-term forecast adjustment |
| **Allocation optimization** | Fair-share allocation proportional to customer demand |
| **Forecast disaggregation** | Split item-location forecast to customer level using historical shares |

---

## 11. File Placement

| New File | Location |
|---|---|
| DDL migration | `sql/110_create_fact_customer_demand_monthly.sql` |
| Normalize script | `scripts/etl/normalize_customer_demand_csv.py` |
| Load script | `scripts/etl/load_customer_demand_postgres.py` |
| DomainSpec entry | `common/core/domain_specs.py` (add `CUSTOMER_DEMAND_SPEC`) |
| Config | `config/etl/etl_config.yaml` (add to domain_order) |
| Tests | `tests/unit/test_customer_demand_load.py` |

---

## 12. Implementation Sequence

1. Create DDL: `sql/110_create_fact_customer_demand_monthly.sql`
2. Apply schema: `psql -f sql/110_create_fact_customer_demand_monthly.sql`
3. Write normalize script: `scripts/etl/normalize_customer_demand_csv.py`
4. Write load script: `scripts/etl/load_customer_demand_postgres.py`
5. Register DomainSpec in `common/core/domain_specs.py`
6. Add Makefile targets
7. Add to etl_config.yaml domain_order
8. Run initial load: `make pipeline-customer-demand`
9. Validate with DQ queries (Section 8)
10. Add API router if dedicated endpoints needed
