# Feature 1 ERD: Canonical Forecasting Data Contracts

## Scope
This document defines table-level contracts for the `Feature 1` architecture in `docs/design-specs/feature1.md`:
- canonical master data (`item`, `location`, `customer`)
- source-to-canonical mappings
- item supersession at storage-location level
- historical demand and business drivers
- forecasting outputs, lag archives, and KPI facts (weekly + monthly planning grains)
- metadata/workflow tables in Postgres

## Conventions
- SQL types are vendor-neutral and can be adapted per engine.
- `_sk` = surrogate key, `_ck` = canonical business key.
- `planning_grain` enum values: `WEEKLY`, `MONTHLY`.
- Timestamps are UTC.
- Iceberg tables include audit columns:
  - `created_at TIMESTAMP`
  - `updated_at TIMESTAMP`
  - `pipeline_run_id VARCHAR(100)`

## Iceberg Catalog

### `silver.dim_item_scd`
Purpose: canonical item hierarchy and attributes (SCD2).

Columns:
- `item_sk BIGINT` PK
- `item_ck VARCHAR(100)` NOT NULL
- `item_name VARCHAR(255)`
- `brand VARCHAR(100)`
- `category VARCHAR(100)`
- `family VARCHAR(100)`
- `segment VARCHAR(100)`
- `uom_base VARCHAR(20)`
- `pack_size DECIMAL(18,6)`
- `lifecycle_status VARCHAR(30)`
- `effective_from DATE` NOT NULL
- `effective_to DATE` NOT NULL
- `is_current BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `item_sk`
- Unique (business): `item_ck, effective_from`
- Recommended index for serving: `item_ck, is_current`

Partitioning:
- Partition by month on `effective_from`

---

### `silver.dim_location_scd`
Purpose: canonical location hierarchy and attributes (SCD2).

Columns:
- `location_sk BIGINT` PK
- `location_ck VARCHAR(100)` NOT NULL
- `location_name VARCHAR(255)`
- `region VARCHAR(100)`
- `country VARCHAR(100)`
- `state VARCHAR(100)`
- `city VARCHAR(100)`
- `location_type VARCHAR(30)`  -- store, dc, plant, etc.
- `cluster VARCHAR(100)`
- `effective_from DATE` NOT NULL
- `effective_to DATE` NOT NULL
- `is_current BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `location_sk`
- Unique (business): `location_ck, effective_from`
- Recommended index: `location_ck, is_current`

Partitioning:
- Partition by month on `effective_from`

---

### `silver.dim_customer_scd`
Purpose: canonical customer hierarchy and attributes (SCD2).

Columns:
- `customer_sk BIGINT` PK
- `customer_ck VARCHAR(100)` NOT NULL
- `customer_name VARCHAR(255)`
- `channel VARCHAR(100)`
- `account_level_1 VARCHAR(100)`
- `account_level_2 VARCHAR(100)`
- `segment VARCHAR(100)`
- `fulfillment_terms VARCHAR(100)`
- `effective_from DATE` NOT NULL
- `effective_to DATE` NOT NULL
- `is_current BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `customer_sk`
- Unique (business): `customer_ck, effective_from`
- Recommended index: `customer_ck, is_current`

Partitioning:
- Partition by month on `effective_from`

---

### `silver.dim_calendar`
Purpose: standard calendar and fiscal attributes.

Columns:
- `date_key DATE` PK
- `day_of_week SMALLINT`
- `week_of_year SMALLINT`
- `month_of_year SMALLINT`
- `quarter_of_year SMALLINT`
- `year_num SMALLINT`
- `fiscal_week SMALLINT`
- `fiscal_month SMALLINT`
- `fiscal_quarter SMALLINT`
- `fiscal_year SMALLINT`
- `is_weekend BOOLEAN`
- `is_holiday BOOLEAN`
- `holiday_name VARCHAR(100)`
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `date_key`

Partitioning:
- Not required (small dimension)

---

### `silver.map_item_source_to_canonical`
Purpose: map source item keys to canonical item keys.

Columns:
- `map_id BIGINT` PK
- `source_system VARCHAR(100)` NOT NULL
- `source_item_key VARCHAR(255)` NOT NULL
- `item_ck VARCHAR(100)` NOT NULL
- `rule_version VARCHAR(50)` NOT NULL
- `confidence DECIMAL(5,4)`  -- 0.0000 to 1.0000
- `effective_from DATE` NOT NULL
- `effective_to DATE` NOT NULL
- `is_active BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `map_id`
- Unique: `source_system, source_item_key, effective_from`
- FK (logical): `item_ck -> silver.dim_item_scd.item_ck`

Partitioning:
- Partition by `source_system`

---

### `silver.map_location_source_to_canonical`
Purpose: map source location keys to canonical location keys.

Columns:
- `map_id BIGINT` PK
- `source_system VARCHAR(100)` NOT NULL
- `source_location_key VARCHAR(255)` NOT NULL
- `location_ck VARCHAR(100)` NOT NULL
- `rule_version VARCHAR(50)` NOT NULL
- `confidence DECIMAL(5,4)`
- `effective_from DATE` NOT NULL
- `effective_to DATE` NOT NULL
- `is_active BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `map_id`
- Unique: `source_system, source_location_key, effective_from`
- FK (logical): `location_ck -> silver.dim_location_scd.location_ck`

Partitioning:
- Partition by `source_system`

---

### `silver.map_customer_source_to_canonical`
Purpose: map source customer keys to canonical customer keys.

Columns:
- `map_id BIGINT` PK
- `source_system VARCHAR(100)` NOT NULL
- `source_customer_key VARCHAR(255)` NOT NULL
- `customer_ck VARCHAR(100)` NOT NULL
- `rule_version VARCHAR(50)` NOT NULL
- `confidence DECIMAL(5,4)`
- `effective_from DATE` NOT NULL
- `effective_to DATE` NOT NULL
- `is_active BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `map_id`
- Unique: `source_system, source_customer_key, effective_from`
- FK (logical): `customer_ck -> silver.dim_customer_scd.customer_ck`

Partitioning:
- Partition by `source_system`

---

### `silver.bridge_item_location_supersession`
Purpose: effective-dated supersession rules from old item to new item by storage location.

Columns:
- `supersession_id BIGINT` PK
- `old_item_ck VARCHAR(100)` NOT NULL
- `new_item_ck VARCHAR(100)` NOT NULL
- `location_ck VARCHAR(100)` NOT NULL
- `supersession_start_date DATE` NOT NULL
- `supersession_end_date DATE` NULL
- `supersession_type VARCHAR(30)` NOT NULL  -- one_to_one, many_to_one
- `conversion_factor DECIMAL(18,8)` NOT NULL DEFAULT 1.0
- `is_active BOOLEAN` NOT NULL
- `rule_source VARCHAR(100)` NOT NULL
- `approved_by VARCHAR(100)` NULL
- `approved_at TIMESTAMP` NULL
- `rule_version VARCHAR(50)` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `supersession_id`
- Unique: `old_item_ck, new_item_ck, location_ck, supersession_start_date, rule_version`
- FK (logical): `old_item_ck -> silver.dim_item_scd.item_ck`
- FK (logical): `new_item_ck -> silver.dim_item_scd.item_ck`
- FK (logical): `location_ck -> silver.dim_location_scd.location_ck`

Partitioning:
- Partition by month on `supersession_start_date`
- Secondary partition by `location_ck`

---

### `silver.fact_demand_history_daily`
Purpose: demand and fulfillment history at daily grain.

Columns:
- `date_key DATE` NOT NULL
- `item_sk BIGINT` NOT NULL
- `location_sk BIGINT` NOT NULL
- `customer_sk BIGINT` NULL
- `qty_sold DECIMAL(18,4)` NOT NULL
- `qty_demanded DECIMAL(18,4)` NULL
- `qty_delivered DECIMAL(18,4)` NULL
- `qty_returned DECIMAL(18,4)` NULL
- `stockout_flag BOOLEAN`
- `revenue_net DECIMAL(18,4)` NULL
- `currency VARCHAR(10)` NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `date_key, item_sk, location_sk, customer_sk`
- FK (logical): `item_sk -> silver.dim_item_scd.item_sk`
- FK (logical): `location_sk -> silver.dim_location_scd.location_sk`
- FK (logical): `customer_sk -> silver.dim_customer_scd.customer_sk`

Partitioning:
- Partition by month on `date_key`
- Cluster/sort by `location_sk, item_sk`

---

### `silver.fact_pricing_daily`
Purpose: effective daily pricing inputs.

Columns:
- `date_key DATE` NOT NULL
- `item_sk BIGINT` NOT NULL
- `location_sk BIGINT` NOT NULL
- `customer_sk BIGINT` NULL
- `base_price DECIMAL(18,4)` NOT NULL
- `net_price DECIMAL(18,4)` NULL
- `discount_pct DECIMAL(9,6)` NULL
- `price_zone VARCHAR(50)` NULL
- `currency VARCHAR(10)` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `date_key, item_sk, location_sk, customer_sk`

Partitioning:
- Partition by month on `date_key`
- Cluster/sort by `location_sk, item_sk`

---

### `silver.fact_promo_daily`
Purpose: promotion plan and execution signals.

Columns:
- `date_key DATE` NOT NULL
- `item_sk BIGINT` NOT NULL
- `location_sk BIGINT` NOT NULL
- `customer_sk BIGINT` NULL
- `promo_id VARCHAR(100)` NOT NULL
- `promo_type VARCHAR(50)` NULL
- `promo_depth_pct DECIMAL(9,6)` NULL
- `display_support_flag BOOLEAN`
- `promo_start_date DATE` NULL
- `promo_end_date DATE` NULL
- `cannibalization_group VARCHAR(100)` NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `date_key, item_sk, location_sk, customer_sk, promo_id`

Partitioning:
- Partition by month on `date_key`
- Cluster/sort by `location_sk, item_sk`

---

### `silver.fact_external_drivers_daily`
Purpose: external signals such as weather, holidays, events, disruptions.

Columns:
- `date_key DATE` NOT NULL
- `location_sk BIGINT` NOT NULL
- `weather_station_id VARCHAR(100)` NULL
- `avg_temp_c DECIMAL(8,3)` NULL
- `rain_mm DECIMAL(10,3)` NULL
- `snow_mm DECIMAL(10,3)` NULL
- `weather_anomaly_score DECIMAL(8,4)` NULL
- `event_name VARCHAR(200)` NULL
- `event_type VARCHAR(100)` NULL
- `event_impact_score DECIMAL(8,4)` NULL
- `disruption_flag BOOLEAN`
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `date_key, location_sk, event_name`

Partitioning:
- Partition by month on `date_key`

---

### `gold.dim_algorithm`
Purpose: algorithm registry used in forecast outputs.

Columns:
- `algorithm_id VARCHAR(50)` PK
- `algorithm_name VARCHAR(100)` NOT NULL
- `algorithm_family VARCHAR(50)` NOT NULL  -- statistical, ml, ensemble, etc.
- `objective VARCHAR(100)` NULL
- `feature_set_version VARCHAR(50)` NULL
- `is_active BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

---

### `gold.dim_model_version`
Purpose: model version registry with MLflow lineage.

Columns:
- `model_version_id VARCHAR(100)` PK
- `algorithm_id VARCHAR(50)` NOT NULL
- `mlflow_experiment_id VARCHAR(100)` NOT NULL
- `mlflow_run_id VARCHAR(100)` NOT NULL
- `mlflow_model_name VARCHAR(255)` NULL
- `mlflow_model_version VARCHAR(50)` NULL
- `training_start_date DATE` NULL
- `training_end_date DATE` NULL
- `hyperparam_hash VARCHAR(128)` NULL
- `data_version VARCHAR(100)` NULL
- `is_champion BOOLEAN` NOT NULL
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints:
- FK (logical): `algorithm_id -> gold.dim_algorithm.algorithm_id`

---

### `gold.fact_forecast`
Purpose: forecast outputs for all algorithms/scenarios.

Columns:
- `forecast_date DATE` NOT NULL
- `planning_grain VARCHAR(20)` NOT NULL  -- WEEKLY, MONTHLY
- `period_id VARCHAR(30)` NOT NULL
- `period_start_date DATE` NOT NULL
- `period_end_date DATE` NOT NULL
- `item_sk BIGINT` NOT NULL
- `location_sk BIGINT` NOT NULL
- `customer_sk BIGINT` NULL
- `algorithm_id VARCHAR(50)` NOT NULL
- `model_version_id VARCHAR(100)` NOT NULL
- `scenario_id VARCHAR(100)` NOT NULL
- `forecast_version BIGINT` NOT NULL
- `forecast_qty DECIMAL(18,4)` NOT NULL
- `forecast_p10 DECIMAL(18,4)` NULL
- `forecast_p50 DECIMAL(18,4)` NULL
- `forecast_p90 DECIMAL(18,4)` NULL
- `pipeline_version VARCHAR(50)` NULL
- `feature_snapshot_id VARCHAR(100)` NULL
- `data_version VARCHAR(100)` NULL
- `mlflow_run_id VARCHAR(100)` NOT NULL
- `supersession_applied_flag BOOLEAN` NOT NULL DEFAULT FALSE
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `forecast_date, planning_grain, period_start_date, item_sk, location_sk, customer_sk, algorithm_id, scenario_id, forecast_version`
- FK (logical): `algorithm_id -> gold.dim_algorithm.algorithm_id`
- FK (logical): `model_version_id -> gold.dim_model_version.model_version_id`

Partitioning:
- Partition by `planning_grain`
- Partition by month on `period_start_date`
- Secondary partition by `scenario_id`
- Cluster/sort by `location_sk, item_sk, algorithm_id`

---

### `gold.fact_forecast_archive_lag`
Purpose: immutable lag snapshots for forecast performance by generation lag, including supersession status.

Columns:
- `planning_grain VARCHAR(20)` NOT NULL  -- WEEKLY, MONTHLY
- `period_id VARCHAR(30)` NOT NULL
- `period_start_date DATE` NOT NULL
- `period_end_date DATE` NOT NULL
- `lag_value INTEGER` NOT NULL  -- 1,2,4 etc.
- `lag_uom VARCHAR(10)` NOT NULL  -- WEEK, MONTH
- `asof_forecast_date DATE` NOT NULL
- `item_sk BIGINT` NOT NULL
- `location_sk BIGINT` NOT NULL
- `customer_sk BIGINT` NULL
- `algorithm_id VARCHAR(50)` NOT NULL
- `scenario_id VARCHAR(100)` NOT NULL
- `forecast_qty DECIMAL(18,4)` NOT NULL
- `forecast_version BIGINT` NOT NULL
- `mlflow_run_id VARCHAR(100)` NOT NULL
- `supersession_applied_flag BOOLEAN` NOT NULL DEFAULT FALSE
- `superseded_flag BOOLEAN` NOT NULL DEFAULT FALSE
- `continuity_item_sk BIGINT` NULL  -- replacement item when superseded
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `planning_grain, period_start_date, lag_value, lag_uom, item_sk, location_sk, customer_sk, algorithm_id, scenario_id, forecast_version`
- FK (logical): `continuity_item_sk -> silver.dim_item_scd.item_sk`

Partitioning:
- Partition by `planning_grain`
- Partition by month on `period_start_date`
- Cluster/sort by `lag_uom, lag_value, location_sk, item_sk`

---

### `gold.vw_forecast_archive_continuity`
Purpose: default serving layer that replaces superseded old item-location archives with continuity records on replacement items.

Columns:
- same as `gold.fact_forecast_archive_lag`, but `item_sk` resolved to continuity item when `superseded_flag = TRUE`

Serving rule:
- APIs and dashboards should read this view/table by default
- advanced users can query `gold.fact_forecast_archive_lag` directly for original pre-supersession records

---

### `gold.fact_forecast_accuracy`
Purpose: model and scenario KPI facts across hierarchies and lags.

Columns:
- `planning_grain VARCHAR(20)` NOT NULL  -- WEEKLY, MONTHLY
- `target_period_start DATE` NOT NULL
- `target_period_end DATE` NOT NULL
- `horizon_bucket VARCHAR(50)` NOT NULL
- `lag_value INTEGER` NOT NULL
- `lag_uom VARCHAR(10)` NOT NULL  -- WEEK, MONTH
- `aggregation_level VARCHAR(50)` NOT NULL  -- item/location/customer/portfolio
- `aggregation_key VARCHAR(255)` NOT NULL
- `algorithm_id VARCHAR(50)` NOT NULL
- `scenario_id VARCHAR(100)` NOT NULL
- `wmape DECIMAL(10,6)` NULL
- `mape DECIMAL(10,6)` NULL
- `mae DECIMAL(18,6)` NULL
- `rmse DECIMAL(18,6)` NULL
- `bias DECIMAL(18,6)` NULL
- `service_level_impact DECIMAL(10,6)` NULL
- `supersession_applied_flag BOOLEAN` NOT NULL DEFAULT TRUE
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK (composite): `planning_grain, target_period_start, target_period_end, horizon_bucket, lag_value, lag_uom, aggregation_level, aggregation_key, algorithm_id, scenario_id`

Partitioning:
- Partition by `planning_grain`
- Partition by month on `target_period_end`
- Cluster/sort by `aggregation_level, lag_uom, lag_value, algorithm_id, scenario_id`

---

### `gold.fact_override_audit`
Purpose: human overrides with approval and reason traceability.

Columns:
- `override_id BIGINT` PK
- `planning_grain VARCHAR(20)` NOT NULL  -- WEEKLY, MONTHLY
- `period_id VARCHAR(30)` NOT NULL
- `period_start_date DATE` NOT NULL
- `period_end_date DATE` NOT NULL
- `item_sk BIGINT` NOT NULL
- `location_sk BIGINT` NOT NULL
- `customer_sk BIGINT` NULL
- `scenario_id VARCHAR(100)` NOT NULL
- `algorithm_id VARCHAR(50)` NOT NULL
- `original_forecast_qty DECIMAL(18,4)` NOT NULL
- `overridden_forecast_qty DECIMAL(18,4)` NOT NULL
- `override_reason_code VARCHAR(50)` NOT NULL
- `override_comment VARCHAR(1000)` NULL
- `requested_by VARCHAR(100)` NOT NULL
- `approved_by VARCHAR(100)` NULL
- `approval_state VARCHAR(30)` NOT NULL  -- pending, approved, rejected
- `requested_at TIMESTAMP` NOT NULL
- `decided_at TIMESTAMP` NULL
- `supersession_applied_flag BOOLEAN` NOT NULL DEFAULT FALSE
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `pipeline_run_id VARCHAR(100)`

Constraints and indexes:
- PK: `override_id`
- Index: `planning_grain, period_start_date, location_sk, item_sk, approval_state`

Partitioning:
- Partition by `planning_grain`
- Partition by month on `period_start_date`

## Postgres Metadata Schema

### `meta.source_system`
Columns:
- `source_system_id SERIAL` PK
- `source_name VARCHAR(100)` UNIQUE NOT NULL
- `source_type VARCHAR(50)` NOT NULL
- `owner_team VARCHAR(100)` NULL
- `is_active BOOLEAN` NOT NULL DEFAULT TRUE
- `created_at TIMESTAMP NOT NULL`
- `updated_at TIMESTAMP NOT NULL`

### `meta.ingestion_contract`
Columns:
- `contract_id SERIAL` PK
- `source_system_id INTEGER` NOT NULL FK -> `meta.source_system.source_system_id`
- `entity_name VARCHAR(100)` NOT NULL
- `schema_version VARCHAR(50)` NOT NULL
- `expected_frequency VARCHAR(30)` NOT NULL
- `sla_minutes INTEGER` NULL
- `quality_rules_json JSONB` NOT NULL
- `effective_from TIMESTAMP NOT NULL`
- `effective_to TIMESTAMP NULL`
- `created_at TIMESTAMP NOT NULL`
- `updated_at TIMESTAMP NOT NULL`

### `meta.scenario`
Columns:
- `scenario_id VARCHAR(100)` PK
- `scenario_name VARCHAR(200)` NOT NULL
- `scenario_type VARCHAR(50)` NOT NULL  -- baseline, constrained, promo, what-if
- `planning_grain VARCHAR(20)` NOT NULL  -- WEEKLY, MONTHLY, BOTH
- `description TEXT` NULL
- `is_active BOOLEAN` NOT NULL
- `created_by VARCHAR(100)` NOT NULL
- `created_at TIMESTAMP NOT NULL`
- `updated_at TIMESTAMP NOT NULL`

### `meta.supersession_workflow`
Columns:
- `workflow_id BIGSERIAL` PK
- `supersession_id BIGINT` NOT NULL
- `state VARCHAR(50)` NOT NULL  -- draft, pending_approval, approved, rejected, expired
- `state_comment TEXT` NULL
- `changed_by VARCHAR(100)` NOT NULL
- `changed_at TIMESTAMP NOT NULL`
- `created_at TIMESTAMP NOT NULL`
- `updated_at TIMESTAMP NOT NULL`

### `meta.workflow_state`
Columns:
- `workflow_id BIGSERIAL` PK
- `entity_type VARCHAR(50)` NOT NULL  -- forecast, override, publish
- `entity_key VARCHAR(255)` NOT NULL
- `state VARCHAR(50)` NOT NULL
- `state_comment TEXT` NULL
- `changed_by VARCHAR(100)` NOT NULL
- `changed_at TIMESTAMP NOT NULL`

### `meta.dashboard_view`
Columns:
- `view_id BIGSERIAL` PK
- `view_name VARCHAR(200)` NOT NULL
- `owner_user VARCHAR(100)` NOT NULL
- `layout_json JSONB` NOT NULL
- `filters_json JSONB` NOT NULL
- `is_shared BOOLEAN` NOT NULL DEFAULT FALSE
- `created_at TIMESTAMP NOT NULL`
- `updated_at TIMESTAMP NOT NULL`

## Relationship Summary
- `silver.fact_*` joins to canonical dimensions by `_sk`.
- `silver.map_*_source_to_canonical` resolves source identifiers to canonical `_ck`.
- `silver.bridge_item_location_supersession` governs old->new item continuity by location.
- `gold.fact_forecast` joins to `gold.dim_algorithm`, `gold.dim_model_version`, and canonical dimensions.
- `gold.fact_forecast_archive_lag` and `gold.fact_forecast_accuracy` are derived from `gold.fact_forecast` plus actuals.
- `gold.vw_forecast_archive_continuity` is the default archive-serving layer with supersession replacement applied.
- `meta.scenario` keys are referenced by `gold.fact_forecast*` and override workflows.
- `meta.supersession_workflow` governs approval lifecycle for supersession rules.

## Recommended First DDL Order
1. `silver.dim_*_scd` and `silver.dim_calendar`
2. `silver.map_*_source_to_canonical`
3. `silver.bridge_item_location_supersession`
4. `silver.fact_demand_history_daily`, `silver.fact_pricing_daily`, `silver.fact_promo_daily`, `silver.fact_external_drivers_daily`
5. `gold.dim_algorithm`, `gold.dim_model_version`
6. `gold.fact_forecast`, `gold.fact_forecast_archive_lag`, `gold.fact_forecast_accuracy`, `gold.fact_override_audit`
7. Postgres `meta.*` tables
