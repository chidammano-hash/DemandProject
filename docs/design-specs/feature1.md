# Feature 1: Internal Data Architecture (Parent Feature)

## Objective
Define the parent architecture for demand forecasting data across master data, transactions, forecasts, and governance, while delegating domain-specific dimension design to sub-features.

## Scope
Feature 1 covers:
- shared architecture and design principles
- common conformance and mapping patterns
- supersession policy at item-location level
- forecast storage, archival, KPI, and lineage standards
- weekly and monthly planning-grain support

Feature 1 does not define full domain-level attribute lists. Those are in:
- `docs/design-specs/feature1a.md` (item dimension)
- `docs/design-specs/feature1b.md` (location dimension)
- `docs/design-specs/feature1c.md` (customer dimension)
- `docs/design-specs/feature1d.md` (date/calendar dimension)

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
2. Domain dimensions via sub-features (`1a`, `1b`, `1c`, `1d`).
3. Shared transactional and forecast facts.
4. Supersession bridge and continuity serving layer.
5. KPI marts and dashboard-serving datasets.

## Final Recommendation
Keep Feature 1 as the architecture contract and use sub-features for dimension-level detail. This keeps governance centralized and domain models independently evolvable.
