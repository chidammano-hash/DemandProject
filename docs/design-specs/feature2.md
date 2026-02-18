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
