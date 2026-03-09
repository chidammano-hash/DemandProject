<!-- SOURCE: feature2.md (Data Architecture) -->
# Feature 2: Internal Data Architecture & Data Contracts

## Objective
Define the parent architecture for demand forecasting data across master data, transactions, forecasts, and governance, including full entity-relationship diagrams and canonical data contracts.

## Scope
Feature 2 covers:
- shared architecture and design principles
- common conformance and mapping patterns
- supersession policy at item-location level
- forecast storage, archival, KPI, and lineage standards
- weekly and monthly planning-grain support
- table-level contracts (ERD) for all canonical entities

## Core Principles
- `Canonical + Mapping`: all client source data maps into canonical keys and structures.
- `Lakehouse First`: analytical persistence in Iceberg (`bronze/silver/gold`).
- `Metadata Separation`: workflow/config/approvals in Postgres.
- `Lineage by Default`: model and pipeline lineage in MLflow + table-level run IDs.
- `Time Variance`: SCD2 and effective dating for historically correct analytics.

## Platform Roles
- `Iceberg`: analytical source of truth and time-travel storage.
- `Spark`: ingestion, conformance, feature prep, training/scoring, KPI jobs.
- `Trino`: interactive query serving for APIs and dashboards.
- `MLflow`: model experiment, version, and run lineage.
- `Postgres`: operational metadata, overrides, approvals, audit states.

## Cross-Domain Canonical Model
- Core dimensions:
  - `dim_item_scd`
  - `dim_location_scd`
  - `dim_customer_scd`
  - `dim_calendar` / date dimension
- Core mapping tables:
  - `map_item_source_to_canonical`
  - `map_location_source_to_canonical`
  - `map_customer_source_to_canonical`
  - domain-specific source-attribute-to-canonical mappings
- Core facts:
  - historical demand/sales/fulfillment
  - pricing/promo/external drivers
  - forecasts, lag archives, accuracy KPIs, override audit

## Supersession Standard
- Use `bridge_item_location_supersession` for old-item to new-item continuity by location.
- Preserve immutable raw history.
- Serve continuity-adjusted analytics and archives by default.
- Keep approval/version lineage for supersession rules.

## Planning Grain Standard
- All forecast, lag, and KPI facts must support:
  - `planning_grain` (`WEEKLY`, `MONTHLY`)
  - `period_id`, `period_start_date`, `period_end_date`
- Daily actuals remain atomic and are aggregated for weekly/monthly planning layers.

## Governance and Quality
- Mandatory data contract checks at bronze-to-silver and silver-to-gold boundaries.
- Required integrity checks:
  - key mapping completeness
  - SCD validity windows
  - supersession overlap/cycle checks
  - null/volume/drift checks
- Required lineage fields in forecast outputs:
  - `scenario_id`, `algorithm_id`, `model_version`, `run_id`, `planning_grain`

## Implementation Sequence
1. Shared canonical keys, mapping standards, and metadata tables.
2. Domain dimensions (Feature 3).
3. Shared transactional and forecast facts (Feature 4).
4. Supersession bridge and continuity serving layer.
5. KPI marts and dashboard-serving datasets.

---

## ERD: Canonical Forecasting Data Contracts

### Conventions
- SQL types are vendor-neutral and can be adapted per engine.
- `_sk` = surrogate key, `_ck` = canonical business key.
- `planning_grain` enum values: `WEEKLY`, `MONTHLY`.
- Timestamps are UTC.
- Iceberg tables include audit columns: `created_at`, `updated_at`, `pipeline_run_id`

### Iceberg Silver Layer

#### `silver.dim_item_scd`
Purpose: canonical item hierarchy and attributes (SCD2).

| Column | Type | Notes |
|--------|------|-------|
| `item_sk` | BIGINT | PK |
| `item_ck` | VARCHAR(100) | NOT NULL |
| `item_name` | VARCHAR(255) | |
| `brand` | VARCHAR(100) | |
| `category` | VARCHAR(100) | |
| `family` | VARCHAR(100) | |
| `segment` | VARCHAR(100) | |
| `uom_base` | VARCHAR(20) | |
| `pack_size` | DECIMAL(18,6) | |
| `lifecycle_status` | VARCHAR(30) | |
| `effective_from` | DATE | NOT NULL |
| `effective_to` | DATE | NOT NULL |
| `is_current` | BOOLEAN | NOT NULL |

Unique (business): `item_ck, effective_from`. Partition by month on `effective_from`.

#### `silver.dim_location_scd`
Purpose: canonical location hierarchy and attributes (SCD2).

| Column | Type | Notes |
|--------|------|-------|
| `location_sk` | BIGINT | PK |
| `location_ck` | VARCHAR(100) | NOT NULL |
| `location_name` | VARCHAR(255) | |
| `region` | VARCHAR(100) | |
| `country` | VARCHAR(100) | |
| `state` | VARCHAR(100) | |
| `city` | VARCHAR(100) | |
| `location_type` | VARCHAR(30) | store, dc, plant |
| `cluster` | VARCHAR(100) | |
| `effective_from` | DATE | NOT NULL |
| `effective_to` | DATE | NOT NULL |
| `is_current` | BOOLEAN | NOT NULL |

Unique (business): `location_ck, effective_from`. Partition by month on `effective_from`.

#### `silver.dim_customer_scd`
Purpose: canonical customer hierarchy and attributes (SCD2).

| Column | Type | Notes |
|--------|------|-------|
| `customer_sk` | BIGINT | PK |
| `customer_ck` | VARCHAR(100) | NOT NULL |
| `customer_name` | VARCHAR(255) | |
| `channel` | VARCHAR(100) | |
| `account_level_1` | VARCHAR(100) | |
| `account_level_2` | VARCHAR(100) | |
| `segment` | VARCHAR(100) | |
| `fulfillment_terms` | VARCHAR(100) | |
| `effective_from` | DATE | NOT NULL |
| `effective_to` | DATE | NOT NULL |
| `is_current` | BOOLEAN | NOT NULL |

Unique (business): `customer_ck, effective_from`. Partition by month on `effective_from`.

#### `silver.dim_calendar`
Purpose: standard calendar and fiscal attributes.

| Column | Type | Notes |
|--------|------|-------|
| `date_key` | DATE | PK |
| `day_of_week` | SMALLINT | |
| `week_of_year` | SMALLINT | |
| `month_of_year` | SMALLINT | |
| `quarter_of_year` | SMALLINT | |
| `year_num` | SMALLINT | |
| `fiscal_week` | SMALLINT | |
| `fiscal_month` | SMALLINT | |
| `fiscal_quarter` | SMALLINT | |
| `fiscal_year` | SMALLINT | |
| `is_weekend` | BOOLEAN | |
| `is_holiday` | BOOLEAN | |
| `holiday_name` | VARCHAR(100) | |

### Source-to-Canonical Mappings

#### `silver.map_item_source_to_canonical`
Maps source item keys to canonical item keys.

| Column | Type | Notes |
|--------|------|-------|
| `map_id` | BIGINT | PK |
| `source_system` | VARCHAR(100) | NOT NULL |
| `source_item_key` | VARCHAR(255) | NOT NULL |
| `item_ck` | VARCHAR(100) | NOT NULL |
| `rule_version` | VARCHAR(50) | NOT NULL |
| `confidence` | DECIMAL(5,4) | 0.0000 to 1.0000 |
| `effective_from` | DATE | NOT NULL |
| `effective_to` | DATE | NOT NULL |
| `is_active` | BOOLEAN | NOT NULL |

Unique: `source_system, source_item_key, effective_from`. Same pattern for `map_location_source_to_canonical` and `map_customer_source_to_canonical`.

#### `silver.bridge_item_location_supersession`
Effective-dated supersession rules from old item to new item by storage location.

| Column | Type | Notes |
|--------|------|-------|
| `supersession_id` | BIGINT | PK |
| `old_item_ck` | VARCHAR(100) | NOT NULL |
| `new_item_ck` | VARCHAR(100) | NOT NULL |
| `location_ck` | VARCHAR(100) | NOT NULL |
| `supersession_start_date` | DATE | NOT NULL |
| `supersession_end_date` | DATE | NULL |
| `supersession_type` | VARCHAR(30) | one_to_one, many_to_one |
| `conversion_factor` | DECIMAL(18,8) | DEFAULT 1.0 |
| `is_active` | BOOLEAN | NOT NULL |
| `rule_source` | VARCHAR(100) | NOT NULL |
| `approved_by` | VARCHAR(100) | NULL |
| `rule_version` | VARCHAR(50) | NOT NULL |

### Silver Fact Tables

#### `silver.fact_demand_history_daily`
Demand and fulfillment history at daily grain.

Grain: `date_key, item_sk, location_sk, customer_sk`. Measures: `qty_sold`, `qty_demanded`, `qty_delivered`, `qty_returned`, `stockout_flag`, `revenue_net`.

#### `silver.fact_pricing_daily`
Effective daily pricing inputs. Grain: `date_key, item_sk, location_sk, customer_sk`.

#### `silver.fact_promo_daily`
Promotion plan and execution signals. Grain: `date_key, item_sk, location_sk, customer_sk, promo_id`.

#### `silver.fact_external_drivers_daily`
External signals (weather, holidays, events, disruptions). Grain: `date_key, location_sk, event_name`.

### Gold Layer

#### `gold.dim_algorithm`
Algorithm registry. PK: `algorithm_id`. Fields: `algorithm_name`, `algorithm_family`, `objective`, `is_active`.

#### `gold.dim_model_version`
Model version registry with MLflow lineage. PK: `model_version_id`. FK: `algorithm_id`. MLflow fields: `experiment_id`, `run_id`, `model_name`, `model_version`.

#### `gold.fact_forecast`
Forecast outputs for all algorithms/scenarios. Grain: `forecast_date, planning_grain, period_start_date, item_sk, location_sk, customer_sk, algorithm_id, scenario_id, forecast_version`.

#### `gold.fact_forecast_archive_lag`
Immutable lag snapshots for forecast performance. Adds `lag_value`, `lag_uom`, `superseded_flag`, `continuity_item_sk`.

#### `gold.fact_forecast_accuracy`
Model and scenario KPI facts. Metrics: `wmape`, `mape`, `mae`, `rmse`, `bias`, `service_level_impact`.

#### `gold.fact_override_audit`
Human overrides with approval and reason traceability.

### Postgres Metadata Schema

- `meta.source_system` — source system registry
- `meta.ingestion_contract` — ingestion contracts per entity
- `meta.scenario` — scenario types (baseline, constrained, promo, what-if)
- `meta.supersession_workflow` — supersession approval lifecycle
- `meta.workflow_state` — generic entity workflow states
- `meta.dashboard_view` — saved dashboard layouts and filters

### Relationship Summary
- `silver.fact_*` joins to canonical dimensions by `_sk`.
- `silver.map_*_source_to_canonical` resolves source identifiers to canonical `_ck`.
- `silver.bridge_item_location_supersession` governs old->new item continuity by location.
- `gold.fact_forecast` joins to `gold.dim_algorithm`, `gold.dim_model_version`, and canonical dimensions.
- `gold.vw_forecast_archive_continuity` is the default archive-serving layer with supersession replacement.

### Recommended DDL Order
1. `silver.dim_*_scd` and `silver.dim_calendar`
2. `silver.map_*_source_to_canonical`
3. `silver.bridge_item_location_supersession`
4. `silver.fact_*` (demand, pricing, promo, external drivers)
5. `gold.dim_algorithm`, `gold.dim_model_version`
6. `gold.fact_forecast`, `gold.fact_forecast_archive_lag`, `gold.fact_forecast_accuracy`, `gold.fact_override_audit`
7. Postgres `meta.*` tables

---

## Implementation Status (MVP)

### What was implemented:
- Flat dimension tables in Postgres: `dim_item`, `dim_location`, `dim_customer`, `dim_time`, `dim_dfu` (no SCD2, no effective dating)
- Fact tables in Postgres: `fact_sales_monthly`, `fact_external_forecast_monthly`, `fact_inventory_snapshot`
- Archive table: `backtest_lag_archive` (all-lags backtest predictions)
- 9 materialized views: `agg_sales_monthly`, `agg_forecast_monthly`, `agg_inventory_monthly`, `agg_accuracy_by_dim`, `agg_accuracy_lag_archive`, `agg_dfu_coverage`, `agg_dfu_coverage_lag_archive`, `mv_top_movers`, `mv_inventory_forecast_monthly`
- Iceberg mirror via Spark (optional, Postgres is primary)
- `DomainSpec` dataclass in `common/domain_specs.py` as the central schema contract (8 domains: item, location, customer, time, dfu, sales, forecast, inventory)
- Generic ETL pipeline: CSV → normalize → clean CSV → Postgres (+ optional Spark → Iceberg)
- MLflow for clustering and backtest experiment tracking
- Dual-path forecast loading with execution-lag filtering (see illustration below)

---

### Forecast Loading: Dual-Path with Phase Ordering

The source CSV contains 5 rows per DFU per forecast date (lags 0–4). During normalization, each row's `execution_lag` is set equal to its own `lag` (the source has no `execution_lag` column). The loader performs a dual-path insert using **phase ordering** to preserve archive integrity:

```
Phase 3b: Archive load (BEFORE staging mutation)
  → backtest_lag_archive receives ALL 5 lag rows
  → each row's original lag preserved as execution_lag

Phase 3c: Staging mutation
  → UPDATE staging SET execution_lag = dim_dfu.execution_lag
  → all 5 rows now have execution_lag = 2 (the DFU-level value)

Phase 5: Main table insert
  → WHERE lag = execution_lag
  → only lag=2 row enters fact_external_forecast_monthly
```

**Critical design:** The archive loads FIRST (Phase 3b) before the staging mutation (Phase 3c). This ensures the archive reads untouched staging data where each row's `execution_lag` equals its own `lag`. The staging mutation is only needed for the main table's `WHERE lag = execution_lag` filter.

#### Concrete Example

**DFU:** `ITEM-X_GRP-A_LOC-1` with `dim_dfu.execution_lag = 2`

**Source CSV rows (after normalization, before any mutation):**

| row | dmdunit | loc | fcstdate | startdate | lag | execution_lag | basefcst | tothist_dmd |
|-----|---------|-----|----------|-----------|-----|---------------|----------|-------------|
| 1   | ITEM-X  | 1   | 2024-01  | 2024-01   | 0   | 0             | 100      | 90          |
| 2   | ITEM-X  | 1   | 2024-01  | 2024-02   | 1   | 1             | 105      | 95          |
| 3   | ITEM-X  | 1   | 2024-01  | 2024-03   | 2   | 2             | 110      | 88          |
| 4   | ITEM-X  | 1   | 2024-01  | 2024-04   | 3   | 3             | 108      | 92          |
| 5   | ITEM-X  | 1   | 2024-01  | 2024-05   | 4   | 4             | 112      | 87          |

**Phase 3b — Archive INSERT** (reads untouched staging):

| row | lag | execution_lag (preserved) | basefcst | tothist_dmd |
|-----|-----|--------------------------|----------|-------------|
| 1   | 0   | 0                        | 100      | 90          |
| 2   | 1   | 1                        | 105      | 95          |
| 3   | 2   | 2                        | 110      | 88          |
| 4   | 3   | 3                        | 108      | 92          |
| 5   | 4   | 4                        | 112      | 87          |

All 5 lags are preserved in the archive for multi-horizon accuracy analysis.

**Phase 3c — Staging mutation** (overwrites execution_lag from dim_dfu):

| row | lag | execution_lag (after mutation) |
|-----|-----|-------------------------------|
| 1   | 0   | **2** ← was 0                 |
| 2   | 1   | **2** ← was 1                 |
| 3   | 2   | 2 (unchanged)                 |
| 4   | 3   | **2** ← was 3                 |
| 5   | 4   | **2** ← was 4                 |

**Phase 5 — Main table INSERT** with `WHERE lag = execution_lag`:

| row | lag | execution_lag | lag = exec_lag? | Inserted? |
|-----|-----|---------------|-----------------|-----------|
| 1   | 0   | 2             | 0 != 2          | No        |
| 2   | 1   | 2             | 1 != 2          | No        |
| 3   | 2   | 2             | **2 = 2**       | **Yes**   |
| 4   | 3   | 2             | 3 != 2          | No        |
| 5   | 4   | 2             | 4 != 2          | No        |

Only row 3 (lag=2) enters `fact_external_forecast_monthly`.

**Unmatched DFUs** (not found in `dim_dfu`): execution_lag defaults to `0` during staging mutation, so only the lag-0 row enters the main table. The archive still receives all 5 rows with their original lag values (loaded in Phase 3b before mutation).

**`--skip-archive` flag:** When `--skip-archive` is passed (or via `make load-forecast-replace-no-archive`), Phase 3b (archive load) and Phase 8 (archive view refresh) are skipped entirely. Only the execution-lag row is loaded into the main table. This is useful for fast external forecast reloads when the 45M-row archive INSERT is not needed.

### What remains aspirational (not yet implemented):
- SCD2 (slowly changing dimensions) with `effective_from` / `effective_to` / `is_current`
- Source-to-canonical mapping tables (`map_*_source_to_canonical`)
- Supersession bridge (`bridge_item_location_supersession`)
- Silver/gold Iceberg layer separation (current implementation is single-layer)
- Postgres `meta.*` operational metadata tables (source_system, ingestion_contract, scenario, workflow_state, dashboard_view)
- `gold.dim_algorithm` and `gold.dim_model_version` registries
- `gold.fact_override_audit` for human overrides
- Weekly planning grain support (only monthly is implemented)
- Data quality gates at layer boundaries


---

## Examples

### Example: Canonical DFU composite key

```sql
SELECT dfu_ck, dmdunit, loc, execution_lag, cluster_assignment
FROM dim_dfu WHERE dmdunit = '100320' AND loc = '1401-BULK';
-- dfu_ck: '100320_ALL_1401-BULK' | execution_lag: 2 | cluster: high_volume_steady
```

### Example: Dual-path forecast load (archive before main table)

```bash
# Phase 3b: load ALL lags (0-4) into archive FIRST (before execution_lag mutation)
# Phase 3c: mutate staging table execution_lag from dim_dfu
# Phase 5:  insert only WHERE lag = execution_lag into main table

# Fast reload (preserves backtest, skips archive):
make load-forecast-replace-no-archive
# Loaded 1,842,310 rows into fact_external_forecast_monthly (model: external)
```

### Example: Archive vs main table row counts

```sql
SELECT 'archive' AS tbl, COUNT(*) FROM backtest_lag_archive WHERE model_id='lgbm_global'
UNION ALL
SELECT 'main',           COUNT(*) FROM fact_external_forecast_monthly WHERE model_id='lgbm_global';
-- archive | 478,105  (5 lags × 95,621 DFU-months)
-- main    |  95,621  (only execution-lag rows)
```


---

<!-- SOURCE: feature3.md (Dimension Tables) -->
# Feature 3: Dimension Tables

## Objective
Define all five dimension tables for the demand forecasting platform: Item, Location, Customer, Time, and DFU.

---

## 3A. Item Dimension (`dim_item`)

### Purpose
Standard item master data for planning.

### Table
`dim_item`

### Required Fields
- `item_no` (item number)
- `item_desc` (item description)
- `item_status`
- `brand_name`
- `category`
- `class`
- `sub_class`
- `country`
- `scm_rtd_flag` (ready-to-drink flag)
- `size` (item size descriptor)
- `case_weight` (weight per case)
- `cpl` (cases per layer)
- `cpp` (cases per pallet)
- `lpp` (layers per pallet)
- `case_weight_uom`
- `bpc`
- `bottle_pack`
- `pack_case`
- `item_proof`
- `upc`
- `national_service_model`
- `supplier_no`
- `supplier_name`
- `item_is_deleted`
- `producer_name`

### Internal Fields
- `item_sk`
- `item_ck` (same as `item_no`)
- `load_ts`
- `modified_ts`

### Rules
- one row per `item_ck` (`item_no`)
- required fields must be populated
- numeric fields must be valid: `case_weight`, `cpl`, `cpp`, `lpp`, `bpc`, `bottle_pack`, `pack_case`, `item_proof`

---

## 3B. Location Dimension (`dim_location`)

### Purpose
Standard location dimension for planning.

### Table
`dim_location`

### Required Fields
- `location_id` (location identifier)
- `site_id` (site grouping identifier)
- `site_desc` (site description)
- `state_id` (state code)
- `primary_demand_location` (`Y`/`N` flag)

### Internal Fields
- `location_sk`
- `location_ck` (same as `location_id`)
- `load_ts`
- `modified_ts`

### Rules
- one row per `location_ck` (`location_id`)
- required fields must be populated
- `primary_demand_location` constrained to `Y` or `N`

### Source Mapping
Source file: `datafiles/locationdata.csv`

---

## 3C. Customer Dimension (`dim_customer`)

### Purpose
Slim, analytics-ready customer dimension for the unified demand MVP.

### Table
`dim_customer`

### Required Fields
- `site` (operating site / region code)
- `customer_no` (source customer identifier)
- `customer_name` (customer display name)
- `city`
- `state`
- `zip`
- `premise_code`
- `status`
- `license_name`
- `store_type_desc`
- `chain_type_desc`
- `state_chain_name`
- `corp_chain_name`
- `rpt_channel_desc`
- `rpt_sub_channel_desc`
- `rpt_ship_type_desc`
- `customer_acct_type_desc`
- `delivery_freq_code`

### Internal Fields
- `customer_sk`
- `customer_ck` (composite key: `site` + `-` + `customer_no`)
- `load_ts`
- `modified_ts`

### Rules
- one row per `customer_ck` (`site-customer_no`)
- `site` and `customer_no` are required and define identity
- if duplicate keys exist during load, the latest row in file order wins

### Source Mapping
Source file: `datafiles/customerdata.csv`

### MVP Pipeline
1. Normalize `customerdata.csv` into `customerdata_clean.csv`.
2. Load into Postgres `dim_customer` with `customer_ck = site-customer_no`.
3. Publish to Iceberg as `iceberg.silver.dim_customer`.
4. Expose via unified API and UI (`/domains/customer`).

---

## 3D. Time Dimension (`dim_time`)

### Purpose
Reusable calendar/time dimension for all domains. Auto-generated, not sourced from a file.

### Table
`dim_time`

### Internal Keys
- `time_sk` (surrogate key)
- `time_ck` (same as `date_key`)

### Required Fields
- `date_key` (date)
- `day_name`, `day_of_week` (ISO: Monday=1...Sunday=7), `day_of_month`, `day_of_year`
- `iso_week_year`, `iso_week`, `week_start_date`, `week_end_date`
- `month_number`, `month_name`, `month_start_date`, `month_end_date`
- `quarter_number`, `quarter_label`, `quarter_start_date`, `quarter_end_date`
- `year_number`, `year_start_date`, `year_end_date`
- `week_bucket` (`YYYY-Www`), `month_bucket` (`YYYY-MM`), `quarter_bucket` (`YYYY-Qn`), `year_bucket` (`YYYY`)
- `load_ts`, `modified_ts`

### Generation Rules
- Range: `2020-01-01` through `2035-12-31` (inclusive)
- One row per day
- `time_ck = date_key`
- Week values use ISO calendar semantics
- ISO week-year can differ from calendar year near year boundaries

### Pipeline
Generated by: `mvp/demand/scripts/normalize_dataset_csv.py --dataset time`
Output: `mvp/demand/data/timedata_clean.csv`

---

## 3E. DFU Dimension (`dim_dfu`)

### Purpose
DFU attribute dimension at item-customerGroup-location grain.

### Grain
- One row per item-customerGroup-location triple (`dmdunit`, `dmdgroup`, `loc`)
- `dmdgroup` is present in source but currently aggregated at `ALL`

### Table
`dim_dfu`

### Internal Fields
- `dfu_sk`
- `dfu_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc`)
- `load_ts`
- `modified_ts`

### Required Fields
- `dmdunit`, `loc`, `dmdgroup`
- `brand`, `brand_desc`, `region`, `state_plan`, `sales_div`
- `prod_cat_desc`, `prod_class_desc`, `subclass_desc`
- `supplier_desc`, `producer_desc`
- `size`, `brand_size`
- `file_dt`, `histstart`
- `cluster_assignment`

### Additional Attributes Loaded
- `abc_vol`, `ded_div_sw`, `execution_lag`, `otc_status`, `premise`
- `prod_subgrp_desc`, `service_lvl_grp`, `supergroup`, `total_lt`, `vintage`
- `purge_sw`, `alcoh_pct`, `bot_type_desc`, `cnty`, `dom_imp_opt`
- `grape_vrty_desc`, `material`, `proof`, `sop_ref`

### Source Mapping
Source file: `datafiles/dfu.txt` (pipe-delimited, header row)

Key mappings:
- `DMDUNIT` -> `dmdunit`, `DMDGROUP` -> `dmdgroup`, `LOC` -> `loc`
- `BRAND` -> `brand`, `U_ABC_VOL` -> `abc_vol`
- `U_BRAND_DESC` -> `brand_desc`, `U_REGION` -> `region`
- `U_EXECUTION_LAG` -> `execution_lag`, `U_TOTAL_LT` -> `total_lt`
- `U_CLUSTER_ASSIGNMENT` -> `cluster_assignment`
- (40+ total column mappings from source)

### MVP Pipeline
1. Normalize: `make normalize-dfu` -> `data/dfu_clean.csv`
2. Load: `make load-dfu` -> Postgres `dim_dfu`
3. Publish: `make spark-dfu` -> `iceberg.silver.dim_dfu`
4. API: `/domains/dfu`, `/domains/dfu/page`, `/dfus`, `/dfus/page`

---

## Shared Conventions (All Dimensions)
- Surrogate key `_sk`, composite business key `_ck`
- `load_ts` and `modified_ts` audit timestamps
- Full-text search via `pg_trgm` trigram indexes on configured fields
- All domains served via generic API: `GET /domains/{domain}/rows`, `GET /domains/{domain}/search`
- Reserved word workaround: `class` column aliased as `class_` in API responses

---

## Implementation Details

### Indexes (implemented, per table)
- **dim_item**: 3 B-tree (`brand_name`, `category`, `item_status`) + 2 GIN trigram (`item_desc`, `brand_name`)
- **dim_location**: 3 B-tree (`site_id`, `state_id`, `primary_demand_location`) + 1 GIN trigram (`site_desc`)
- **dim_customer**: 5 B-tree (`site`, `customer_no`, `state`, `status`, `channel`) + 1 GIN trigram (`customer_name`). Note: `customer_name` is nullable (DDL drops NOT NULL constraint)
- **dim_time**: 5 B-tree (`date`, `week_bucket`, `month_bucket`, `quarter_bucket`, `year_number`)
- **dim_dfu**: 6 B-tree (`dmdunit`, `loc`, `brand`, `region`, `cluster_assignment`, `ml_cluster`) + 1 GIN trigram (`brand_desc`) + 2 seasonality B-tree (`seasonality_profile`, `is_yearly_seasonal`)

### Additional dim_dfu Columns (not in original spec)
- `ml_cluster TEXT` — KMeans-assigned cluster label (Feature 7)
- `seasonality_profile TEXT` — seasonal pattern label (Feature 30)
- `seasonality_strength NUMERIC(10,4)` — ACF-based strength metric
- `is_yearly_seasonal BOOLEAN` — yearly seasonal flag
- `peak_month INTEGER`, `trough_month INTEGER` — seasonal peak/trough months
- `peak_trough_ratio NUMERIC(10,4)` — peak-to-trough demand ratio

### Additional API Endpoints (generic, all domains)
- `GET /domains/{domain}/suggest` — column-level typeahead suggestions
- `GET /domains/{domain}/sample-pair` — random item+location pair
- `GET /domains/{domain}/meta` — domain metadata (columns, types, sort info)
- `GET /domains/{domain}/analytics` — summary stats, trend, category distribution, KPIs
- `GET /domains/{domain}/distinct` — distinct values for filter dropdowns
- `GET /domains/forecast/models` — distinct model_id values

### DomainSpec Type Classifications
Each domain in `common/domain_specs.py` defines `search_fields` (for full-text search), `int_fields`, `float_fields`, `date_fields`, and `bool_fields` (for type-aware API filtering).


---

## Examples

### Example: Query item dimension

```sql
SELECT item_no, description, brand, category, class_
FROM dim_item WHERE item_no = '100320';
-- 100320 | CABERNET SAUV 750ML | COASTAL RIDGE | WINE | RED
```

### Example: Query location dimension

```sql
SELECT loc, loc_desc, state, region
FROM dim_location WHERE loc = '1401-BULK';
-- 1401-BULK | CALIFORNIA DC BULK | CA | WEST
```

### Example: DFU dimension with all extended attributes

```sql
SELECT dfu_ck, dmdunit, loc, execution_lag, cluster_assignment,
       seasonality_profile, seasonality_strength, peak_month
FROM dim_dfu WHERE dmdunit='100320' AND loc='1401-BULK';
-- 100320_ALL_1401-BULK | 100320 | 1401-BULK | 2 | high_volume_steady | yearly_strong | 0.78 | 11
```

### Example: Load all dimensions

```bash
make normalize-all   # normalize all 8 datasets
make load-all        # load into Postgres + refresh materialized views
make check-db        # verify row counts
```

### Example: Typeahead suggestions for explorer

```bash
curl -s "http://localhost:8000/domains/item/suggest?col=brand&q=coast" | jq .
# {"suggestions": ["COASTAL RIDGE", "COASTAL HIGHWAY", "COAST MOUNTAINS"]}
```


---

<!-- SOURCE: feature4.md (Fact Tables) -->
# Feature 4: Fact Tables

## Objective
Define the two core fact tables for demand analytics: monthly sales history and external forecast archive.

---

## 4A. Sales Fact (`fact_sales_monthly`)

### Purpose
Monthly shipped and ordered quantities from `dfu_lvl2_hist.txt` for analytics and UI exploration.

### Grain
- One row per `dmdunit` + `dmdgroup` + `loc` + `startdate` + `type`
- `startdate` is monthly grain and must be first day of month (`YYYYMM01`)
- Only `TYPE=1` is loaded for MVP

### Table
`fact_sales_monthly`

### Internal Fields
- `sales_sk`
- `sales_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc` + `_` + `startdate` + `_` + `type`)
- `load_ts`
- `modified_ts`

### Required Fields
- `dmdunit`, `dmdgroup` (mostly `ALL`), `loc`, `startdate`, `type`

### Measures
- `qty_shipped` (cases shipped)
- `qty_ordered` (cases ordered / demand)
- `qty` (as provided in source)

### Source Mapping
Source file: `datafiles/dfu_lvl2_hist.txt` (pipe-delimited)

- `DMDUNIT` -> `dmdunit`, `DMDGROUP` -> `dmdgroup`, `LOC` -> `loc`
- `STARTDATE` (`YYYYMMDD`) -> `startdate` (`YYYY-MM-DD`)
- `TYPE` -> `type` (load only value `1`)
- `U_QTY_SHIPPED` -> `qty_shipped`, `U_QTY_ORDERED` -> `qty_ordered`, `QTY` -> `qty`

### MVP Pipeline
1. Normalize: `make normalize-sales` -> `data/dfu_lvl2_hist_clean.csv` (keep TYPE=1 only, parse dates, reject non-month-start)
2. Load: `make load-sales` -> Postgres `fact_sales_monthly`
3. Publish: `make spark-sales` -> `iceberg.silver.fact_sales_monthly`
4. API: `/domains/sales`, `/domains/sales/page`

### Technology
- Ingest/normalize: Python (`csv`), `uv`, Make
- OLTP sink: PostgreSQL + `psycopg` bulk copy
- Lakehouse sink: Spark 3.5 + Apache Iceberg + MinIO
- Query engine: Trino
- Serving/UI: FastAPI + React/Vite

---

## 4B. External Forecast Fact (`fact_external_forecast_monthly`)

### Purpose
Archived external statistical forecasts for lag-based forecast accuracy analysis.

### Grain
- One row per `dmdunit` + `dmdgroup` + `loc` + `fcstdate` + `startdate` + `model_id`
- `fcstdate` is forecast generation month (month-start)
- `startdate` is forecasted demand month (month-start)
- `lag` is computed as month difference between `startdate` and `fcstdate`
- Only lags `0..4` are stored in MVP
- `model_id` identifies the forecasting algorithm (default `'external'`); see Feature 6

### Table
`fact_external_forecast_monthly`

### Internal Fields
- `forecast_sk`
- `forecast_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc` + `_` + `fcstdate` + `_` + `startdate`)
- `load_ts`
- `modified_ts`

### Business Fields
- `dmdunit`, `dmdgroup`, `loc`
- `fcstdate`, `startdate`
- `lag`, `execution_lag`
- `model_id` (forecasting algorithm identifier; default `'external'`)
- `basefcst_pref` (base statistical forecast)
- `tothist_dmd` (actual sales for the month)

### Constraint
`UNIQUE(forecast_ck, model_id)` — each business key appears once per model.

### Source Mapping
Source file: `datafiles/dfu_stat_fcst.txt` (pipe-delimited)

### MVP Pipeline
1. Normalize: `make normalize-forecast` (enforce month-start, compute lag, keep lag 0-4)
2. Load: `make load-forecast` -> Postgres `fact_external_forecast_monthly`
3. Publish: `make spark-forecast` -> `iceberg.silver.fact_external_forecast_monthly`
4. API: `/domains/forecast`, `/domains/forecast/page`

---

## Materialized Views

### `agg_sales_monthly`
Pre-aggregated sales for O(1) KPI queries.

### `agg_forecast_monthly`
Pre-aggregated forecasts including `model_id` in GROUP BY for per-model analytics.

---

## Shared Conventions
- Surrogate key `_sk`, composite business key `_ck`
- `load_ts` and `modified_ts` audit timestamps
- Null normalization: `''`, `'null'`, `'none'`, `'NA'` treated as NULL during load
- Type casting: integer/float/date fields auto-cast with null coercion
- All domains served via generic API with pagination (offset/limit)

---

## Implementation Details

### fact_sales_monthly
- Additional column: `file_dt DATE`
- CHECK constraints: `type = 1` (enforced at DB level), `startdate = date_trunc('month', startdate)` (month-start grain)
- Indexes: 4 B-tree (`dmdunit`, `loc`, `startdate`, `type`) + 2 composite (`(dmdunit, loc, startdate)`, `startdate`) + 3 GIN trigram (`dmdunit`, `loc`, `dmdgroup`)
- `pg_trgm` extension created for trigram-based substring search

### fact_external_forecast_monthly
- 4 CHECK constraints: lag 0-4, fcst month-start alignment, start month-start alignment, lag-matches-dates
- UNIQUE constraint: `(forecast_ck, model_id)`
- Indexes: 6 B-tree + 2 composite from `008` + 4 GIN trigram + 2 composite from `013` (`(dmdunit, dmdgroup, loc)`, `(model_id, lag)`)

### Additional Fact Tables (not in original spec)
- **backtest_lag_archive** (`sql/010`): All-lags (0-4) backtest predictions. Grain = `forecast_ck + model_id + lag`. Includes `timeframe` column. 4 CHECK constraints, `UNIQUE(forecast_ck, model_id, lag)`, 4 B-tree + 2 composite indexes.
- **fact_inventory_snapshot** (`sql/017`): Grain = `item_no + loc + snapshot_date`. Columns: `inventory_sk`, `inventory_ck`, `item_no`, `loc`, `snapshot_date`, `lead_time_days`, `qty_on_hand`, `qty_on_hand_on_order`, `qty_on_order`, `mtd_sales`. 4 B-tree + 2 GIN trigram indexes.

### Materialized Views (full list — 9 views)
| View | Grain | Key Measures | DDL |
|------|-------|-------------|-----|
| `agg_sales_monthly` | month_start, dmdunit, loc | row_count, qty_shipped, qty_ordered, qty | sql/008 |
| `agg_forecast_monthly` | month_start, dmdunit, loc, model_id | row_count, basefcst_pref, tothist_dmd | sql/008 |
| `agg_inventory_monthly` | month_start, item_no, loc | eom_qty_on_hand, monthly_sales, avg_daily_sls, latest_lead_time_days | sql/017 |
| `agg_accuracy_by_dim` | model_id, lag, month_start + 8 dims | sum_forecast, sum_actual, sum_abs_error | sql/011, sql/016 |
| `agg_accuracy_lag_archive` | model_id, lag, month_start + dims | sum_forecast, sum_actual, sum_abs_error | sql/011, sql/016 |
| `agg_dfu_coverage` | model_id, lag | dfu_count | sql/012 |
| `agg_dfu_coverage_lag_archive` | model_id, lag | dfu_count | sql/012 |
| `mv_top_movers` | dmdunit | current/prior period qty, pct_change | sql/018 |
| `mv_inventory_forecast_monthly` | item_no, loc, month_start, model_id | forecast, actual, error, dos, stockout/excess flags | sql/019 |


---

## Examples

### Example: Sales fact — recent months for item 100320

```sql
SELECT startdate, qty_shipped, qty_ordered
FROM fact_sales_monthly
WHERE dmdunit='100320' AND loc='1401-BULK' AND type=1
ORDER BY startdate DESC LIMIT 4;
-- 2026-01-01 | 788 | 801
-- 2025-12-01 | 910 | 928
-- 2025-11-01 | 842 | 860
-- 2025-10-01 | 875 | 891
```

### Example: Forecast fact — 5 lags for one forecast date

```sql
SELECT fcstdate, startdate, lag, model_id, basefcst_pref, tothist_dmd
FROM fact_external_forecast_monthly
WHERE dmdunit='100320' AND loc='1401-BULK'
  AND fcstdate='2025-01-01' AND model_id='external'
ORDER BY lag;
-- 2025-01-01 | 2025-01-01 | 0 | external | 920 | 921
-- 2025-01-01 | 2025-02-01 | 1 | external | 905 | 895
-- 2025-01-01 | 2025-03-01 | 2 | external | 910 | 875  ← execution-lag row
-- 2025-01-01 | 2025-04-01 | 3 | external | 895 | 842
-- 2025-01-01 | 2025-05-01 | 4 | external | 880 | 788
```

### Example: Load sales and forecast

```bash
make load-sales                       # load fact_sales_monthly
make load-forecast-replace            # reload external forecast, preserve backtest
make load-forecast-replace-no-archive # faster: skip 45M-row archive insert
```
