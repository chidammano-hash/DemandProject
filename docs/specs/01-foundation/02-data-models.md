# Data Models

> Defines every table, key, and materialized view in the platform — the shared data dictionary that all forecasting, inventory, and planning features build on.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Explorer (all domains) |
| **Key Files** | `common/domain_specs.py`, `sql/001-019*.sql`, `scripts/normalize_dataset_csv.py`, `scripts/load_dataset_postgres.py` |

---

## Problem

Planners and supply chain teams need consistent, well-structured data to trust their analytics. Without a clear data model — defined keys, grain, types, and relationships — different teams build conflicting reports, joins break silently, and nobody agrees on the numbers.

## Solution

Supply Chain Command Center uses a domain-driven design where all 8 datasets share a single `DomainSpec` pattern. Each domain defines its columns, types, keys, and search fields in one central config file (`common/domain_specs.py`). Generic ETL scripts normalize any CSV into a clean format and load it into PostgreSQL. Materialized views pre-aggregate data for fast analytics.

## How It Works

1. Each domain (item, location, customer, time, sku, sales, forecast, inventory) is defined as a `DomainSpec` dataclass.
2. Raw CSVs are normalized via `scripts/normalize_dataset_csv.py` (null normalization, type casting, date parsing).
3. Clean CSVs are loaded into PostgreSQL via `scripts/load_dataset_postgres.py`.
4. Materialized views aggregate data for O(1) KPI queries.
5. FastAPI serves all domains through generic endpoints: `GET /domains/{domain}/rows`, `/search`, `/suggest`, `/distinct`.

## Key Conventions

- **Surrogate key** (`_sk`): auto-increment integer primary key
- **Composite key** (`_ck`): natural business key built from domain-specific fields
- **Audit timestamps**: `load_ts` and `modified_ts` on every table
- **Null normalization**: `''`, `'null'`, `'none'`, `'NA'` all become NULL during load
- **Full-text search**: `pg_trgm` GIN trigram indexes on configured text fields
- **Reserved words**: `class` column aliased as `class_` in API responses

## Data Model

### Dimension Tables

| Table | Grain | Composite Key (`_ck`) | Key Business Fields | Row Count |
|---|---|---|---|---|
| `dim_item` | One row per item | `item_id` | item_desc, brand_name, category, class, supplier_no | ~3K |
| `dim_location` | One row per location | `location_id` | site_id, site_desc, state_id, primary_demand_location | ~1K |
| `dim_customer` | One row per site+customer | `site-customer_no` | customer_name, city, state, chain_type_desc, rpt_channel_desc | ~50K |
| `dim_time` | One row per day | `date_key` | day_name, month_bucket, quarter_bucket, year_number | ~5.8K |
| `dim_sku` | One row per item+group+location | `item_id_customer_group_loc` | brand, region, abc_vol, execution_lag, cluster_assignment, ml_cluster, seasonality_profile | ~113K |

### DFU Extended Attributes

The DFU (Demand Forecast Unit — an item+customerGroup+location combination) table carries additional computed columns from downstream pipelines:

| Column | Source | Purpose |
|---|---|---|
| `ml_cluster` | Clustering pipeline | KMeans-assigned cluster label |
| `cluster_assignment` | Cluster labeling | Business-readable cluster name |
| `seasonality_profile` | Seasonality detection | Seasonal pattern label (e.g., "yearly_strong") |
| `seasonality_strength` | Seasonality detection | ACF-based strength metric (0-1) |
| `peak_month`, `trough_month` | Seasonality detection | Month numbers for seasonal peaks/troughs |
| `abc_vol` | Source data | Volume classification (A/B/C) |
| `xyz_class` | ABC-XYZ pipeline | Variability classification (X/Y/Z) |
| `execution_lag` | Source data | Months between forecast creation and target month |

### Fact Tables

| Table | Grain | Key Measures | Notes |
|---|---|---|---|
| `fact_sales_monthly` | item_id + customer_group + loc + startdate + type | qty_shipped, qty_ordered, qty | Only TYPE=1 loaded; startdate must be month-start |
| `fact_external_forecast_monthly` | item_id + customer_group + loc + fcstdate + startdate + model_id | basefcst_pref, tothist_dmd | UNIQUE(forecast_ck, model_id); lags 0-4 |
| `fact_inventory_snapshot` | item_id + loc + snapshot_date | qty_on_hand, qty_on_order, mtd_sales, lead_time_days | ~190M rows from 14 monthly CSVs |
| `backtest_lag_archive` | forecast_ck + model_id + lag | basefcst_pref, tothist_dmd, timeframe | All-lags (0-4) backtest predictions |

### Forecast Loading (Dual-Path)

The forecast loader uses phase ordering to preserve archive integrity:

1. **Phase 3b** (Archive): Load ALL 5 lag rows into `backtest_lag_archive` from untouched staging data
2. **Phase 3c** (Mutation): Update staging `execution_lag` from `dim_sku` values
3. **Phase 5** (Main): Insert only rows where `lag = execution_lag` into the main forecast table

This ensures the archive always has all 5 lags per DFU while the main table keeps only the execution-lag row.

### Materialized Views

| View | Grain | Purpose |
|---|---|---|
| `agg_sales_monthly` | month + item_id + loc | Pre-aggregated sales for KPI queries |
| `agg_forecast_monthly` | month + item_id + loc + model_id | Pre-aggregated forecasts per model |
| `agg_inventory_monthly` | month + item_id + loc | EOM on-hand, monthly sales, avg daily sales, lead time |
| `agg_accuracy_by_dim` | model_id + lag + month + 8 dimensions | Forecast vs actual by every dimension |
| `agg_accuracy_lag_archive` | model_id + lag + month + dimensions | Same as above for archive lags |
| `agg_dfu_coverage` | model_id + lag | DFU count per model per lag |
| `mv_inventory_forecast_monthly` | item_id + loc + month + model_id | Inventory-forecast bridge for backtest attribution |
| `mv_dq_dashboard` | domain + run_date | Data quality pass/fail/warn counts |

### Indexes

Each table has B-tree indexes on key lookup columns and GIN trigram indexes on text search fields. Composite indexes on `(item_id, loc, startdate)` and `(model_id, lag)` serve the most common query patterns.

## Pipeline

| Step | Command | Description |
|---|---|---|
| Normalize all datasets | `make normalize-all` | CSV to clean CSV (null handling, type casting) |
| Load all datasets | `make load-all` | Clean CSV to PostgreSQL + refresh materialized views |
| Load forecast (preserve backtest) | `make load-forecast-replace` | Reload external forecast only |
| Load forecast (fast, skip archive) | `make load-forecast-replace-no-archive` | Skip 45M-row archive INSERT |
| Load inventory | `make inventory-pipeline` | Merge 14 monthly CSVs, normalize, load |
| Verify | `make check-db` | Row counts for all tables |

## Configuration

All domain definitions live in `common/domain_specs.py` as frozen `DomainSpec` dataclasses. Each spec declares:
- `columns`: ordered list of column names
- `key_fields`: columns forming the composite key (`_ck`)
- `search_fields`: columns with GIN trigram indexes
- `int_fields`, `float_fields`, `date_fields`, `bool_fields`: type-aware filtering

## Dependencies

- PostgreSQL 16 with `pg_trgm` extension
- [Infrastructure](01-infrastructure.md) — Docker Compose services

## See Also

- [Data Quality](03-data-quality.md) — automated validation across all 8 domains
- [Planning Date](04-planning-date.md) — configurable date for frozen data environments
