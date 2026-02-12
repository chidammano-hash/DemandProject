# Feature 1: Internal Data Architecture for Supply Chain Demand Forecasting

## Objective
Design a robust internal data architecture that:
- supports multi-hierarchy master data for `item`, `location`, and `customer`
- supports item supersession at storage-location level (old SKU to replacement SKU continuity)
- stores demand/sales, pricing, promotions, external drivers (weather/events), and forecasts
- enables multi-algorithm forecasting, forecast archival at multiple lags, and KPI tracking
- supports both weekly and monthly forecasting/planning cycles
- maps heterogeneous customer source systems into a canonical model using mapping tables
- integrates cleanly with `Apache Iceberg`, `MLflow`, `Postgres`, and analytics/dashboard layers

## Core Principles
- `Canonical + Mapping`: internal canonical entities with explicit source-to-canonical mapping tables
- `Lakehouse First`: all heavy analytical facts/dimensions in Iceberg (bronze/silver/gold)
- `Metadata Separation`: operational metadata, workflow, and governance in Postgres
- `ML Lineage`: training/scoring lineage and model lifecycle in MLflow + run IDs in Iceberg
- `Time-Variant Data`: SCD2 dimensions and effective-dated mappings for historical correctness
- `Supersession-Aware Continuity`: preserve raw history while creating conformed continuity series for replacements
- `Forecast Governance`: retain all forecast versions, lags, algorithms, overrides, approvals, and audits

## Technology Roles
- `Iceberg`: source of truth for analytical data, history, partition evolution, and time travel
- `Spark`: ingestion, conformance, feature generation, model training/scoring, KPI computation
- `Trino`: interactive SQL for APIs and dashboards on curated Iceberg gold tables
- `MLflow`: experiment tracking, model registry, artifacts, model version governance
- `Postgres`: metadata/configuration, workflow states, approvals, overrides, UI/user context

## Storage and Metadata Layers

### 1) Storage Layer (Iceberg)
Logical zones:
- `bronze_*`: raw landed source extracts with minimal transformation
- `silver_*`: conformed canonical model with keys, standard units, quality checks
- `gold_*`: dashboard-ready tables, KPIs, exceptions, and business-facing aggregates

### 2) Metadata Layer (Postgres)
Stores:
- source system registry, ingestion configs, data contracts
- mapping rules/versioning and entity survivorship policies
- forecast scenario metadata (baseline, promo uplift, constrained plan)
- workflow state (reviewed/approved/published), overrides, comments, audit trails
- dashboard configuration and saved views

## Canonical Data Model

### Core Dimensions (Iceberg Silver)
- `dim_item_scd`
  - keys: `item_sk` (surrogate), `item_ck` (canonical business key)
  - attributes: product hierarchies (`category`, `brand`, `family`, `segment`), pack/UOM, lifecycle status
  - SCD2 fields: `effective_from`, `effective_to`, `is_current`
- `dim_location_scd`
  - keys: `location_sk`, `location_ck`
  - attributes: geo hierarchy (`region`, `country`, `state`, `city`, `store_dc_flag`, cluster)
  - SCD2 fields
- `dim_customer_scd`
  - keys: `customer_sk`, `customer_ck`
  - attributes: channel, account hierarchy, customer segment, fulfillment terms
  - SCD2 fields
- `dim_calendar`
  - day/week/month/quarter/year/fiscal flags, holiday indicators

### Mapping and Conformance Tables (Iceberg Silver + Postgres Metadata)
- `map_item_source_to_canonical`
- `map_location_source_to_canonical`
- `map_customer_source_to_canonical`
- `map_attribute_normalization`
- `map_uom_conversion`
- `bridge_item_location_supersession`

Each mapping table should be effective-dated and include:
- `source_system`, `source_key`, `canonical_key`, `confidence`, `rule_version`, `effective_from`, `effective_to`

`bridge_item_location_supersession` should include:
- `old_item_ck`, `new_item_ck`, `location_ck`, `supersession_start_date`, `supersession_end_date`, `supersession_type`, `conversion_factor`, `is_active`
- business rule: for the specified storage location and effective period, demand/forecast continuity rolls from `old_item_ck` to `new_item_ck`
- lineage fields: `rule_source`, `approved_by`, `approved_at`, `rule_version`

### Supersession Handling (Critical Behavior)
- Keep immutable raw records: never delete or physically overwrite original history for old SKUs.
- Build conformed continuity views/tables that remap old SKU history to the replacement SKU by location and date.
- Support one-to-one and many-to-one supersessions through `conversion_factor` and explicit rule versioning.
- Archive replacement policy:
  - historical forecast archives for superseded old item+location combinations are marked `superseded=true`
  - equivalent continuity archives are generated under the new item for the same location
  - UI and APIs default to continuity view (new item), with toggle to inspect original pre-supersession archives
- KPI calculations run on continuity series by default to avoid artificial error inflation at replacement cutovers.

### Forecast Time Grain Support (Weekly + Monthly)
- Add `planning_grain` (`WEEKLY`, `MONTHLY`) in forecast, archive, and KPI facts.
- Use period keys instead of only day-level keys for planning outputs:
  - `period_start_date`, `period_end_date`, `period_id`, `planning_grain`
- Keep daily actuals as atomic source, then aggregate to weekly/monthly feature and target sets.
- Allow dual-run orchestration:
  - weekly operational forecast (short-to-mid horizon)
  - monthly S&OP/IBP forecast (mid-to-long horizon)

### Core Fact Tables (Iceberg Silver/Gold)
- `fact_demand_history_daily`
  - grain: `date`, `item_sk`, `location_sk`, `customer_sk` (nullable by business scope)
  - measures: `qty_sold`, `qty_demanded`, `qty_delivered`, `qty_returned`, `stockout_flag`
- `fact_pricing_daily`
  - `base_price`, `net_price`, `discount_pct`, `currency`, `price_zone`
- `fact_promo_daily`
  - promo type, promo depth, start/end, display support, cannibalization group
- `fact_external_drivers_daily`
  - weather features (temp/rain/snow anomalies), events/holidays, local disruptions

### Forecast and Accuracy Tables (Iceberg Gold)
- `fact_forecast`
  - grain: `forecast_date`, `planning_grain`, `period_start_date`, `item_sk`, `location_sk`, `customer_sk`, `algorithm_id`, `scenario_id`
  - fields: `forecast_qty`, `forecast_p10`, `forecast_p50`, `forecast_p90`, `forecast_version`, `run_id`, `supersession_applied_flag`
- `fact_forecast_archive_lag`
  - captures snapshots by lag for weekly and monthly cycles (`lag_1w`, `lag_2w`, `lag_4w`, `lag_1m`, `lag_2m`, etc.)
- `fact_forecast_accuracy`
  - `wmape`, `mape`, `mae`, `rmse`, `bias`, `service_level_impact`
  - dimensions: forecast lag, `planning_grain`, algorithm, hierarchy level, horizon bucket
- `fact_override_audit`
  - original vs overridden forecast, reason code, user, approval state, timestamps

## Multi-Algorithm Forecasting Architecture
- `dim_algorithm`
  - algorithm metadata (`statistical`, `ml`, `foundation`), objective, feature set version
- `dim_model_version`
  - link to MLflow model/version, training window, hyperparameter set hash
- Support champion/challenger and segmented model selection:
  - by hierarchy node, demand pattern class, lifecycle stage
- Persist every scored output with strict lineage:
  - `run_id` (MLflow), `pipeline_version`, `feature_snapshot_id`, `data_version`

## Forecast Archival at Multiple Lags
For each `target_date`, store forecasts generated at multiple historical cutoffs:
- example: forecasts generated 1, 2, 4, 8 weeks before target date
- for monthly cycles, also retain 1, 2, and 3 month cutoffs where applicable
- do not overwrite prior runs; append immutable versions
- enable exact "as-was" replay via Iceberg time travel and lag snapshot tables

## KPI and Dashboarding Layer
Curated gold marts for low-latency dashboarding:
- `mart_forecast_accuracy_weekly`
- `mart_forecast_accuracy_monthly`
- `mart_forecast_bias_hierarchy`
- `mart_service_risk_exceptions`
- `mart_promo_lift_vs_baseline`
- `mart_value_at_risk`
- `mart_supersession_impact`

Dashboard capabilities:
- drilldown across item/location/customer hierarchies
- compare algorithms/scenarios side-by-side
- compare weekly vs monthly plans for the same portfolio slice
- inspect lag-based forecast degradation
- inspect supersession impact (pre/post replacement) at item-location level
- trace forecast to model run and source data version

## Performance and Scalability Design
- partition large facts by `target_date` or `date` (monthly/weekly based on volume)
- partition forecast facts by `planning_grain` + `period_start_date`
- cluster/sort by high-cardinality filters: `location_sk`, `item_sk`, `algorithm_id`
- maintain aggregate gold tables for common UI queries
- implement API query guardrails (required time filters, top-N limits, pagination)
- compact small files and tune Iceberg maintenance jobs

## Governance, Quality, and Reliability
- data contracts per source and schema validation at bronze-to-silver boundaries
- conformance checks for hierarchy integrity and key mapping completeness
- supersession quality checks:
  - no overlapping active supersession rules for same old item + location + date
  - no circular supersession chains
  - conversion factor validation and continuity reconciliation checks
- metric tests for null/volume/drift anomalies
- row-level audit fields: `created_at`, `created_by_pipeline`, `updated_at`
- retention policy:
  - keep raw bronze snapshots per compliance policy
  - keep forecast archives long enough for seasonal evaluation and audit

## Suggested Minimal Schema Starter (MVP)
Build first:
1. `dim_item_scd`, `dim_location_scd`, `dim_customer_scd`, `dim_calendar`
2. `map_*_source_to_canonical` for all three master entities
3. `bridge_item_location_supersession` with approval/version workflow
4. `fact_demand_history_daily`, `fact_pricing_daily`, `fact_promo_daily`
5. `fact_forecast`, `fact_forecast_archive_lag`, `fact_forecast_accuracy` with `planning_grain`
6. Postgres metadata tables for scenarios, overrides, approvals, supersession governance, and audit
7. MLflow linkage (`run_id`, model version) integrated into forecast facts

## Final Recommendation
Use a canonical Iceberg lakehouse model with effective-dated mapping tables, location-aware item supersession bridging, dual weekly/monthly planning grain support, plus a Postgres metadata layer and MLflow lineage integration. This architecture is robust, customer-agnostic, and designed to onboard varied enterprise data models quickly while preserving forecast continuity, explainability, governance, and dashboard performance at scale.
