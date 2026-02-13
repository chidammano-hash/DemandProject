# Feature 1B: Location Dimension (Extensible Attributes)

## Objective
Design a client-agnostic location dimension that:
- supports core storage/location attributes required by all customers
- supports expandable attributes where names and meanings differ by client
- provides a standard generic attribute structure (`attribute_001` to `attribute_150`)
- enables client-specific mapping from source attributes to canonical attributes

## Scope
This sub-feature is focused only on location master data and location attribute extensibility.

## Design Principles
- `Canonical Core + Extensible Layer`: keep stable core attributes and flexible client-defined attributes.
- `Metadata-Driven`: attribute meaning, datatype, and usage are controlled by metadata tables.
- `Client Mapping First`: source-to-canonical attribute mapping is explicit and versioned.
- `Backward Compatible`: adding new client attributes should not break existing pipelines.

## Location Dimension Structure

### 1) Core Location Dimension (SCD2)
Table: `dim_location_scd`

Required fields:
- `location_sk`, `location_ck`
- `location_name`, `location_type`, `region`, `country`, `state`, `city`
- `postal_code`, `timezone_code`, `latitude`, `longitude`
- `cluster`, `zone`, `sub_zone`, `district`
- `is_storage_location`, `is_dc`, `is_plant`, `is_crossdock`, `is_virtual_location`
- `parent_location_ck`, `node_level`, `network_type`
- `default_customer_ck`, `serving_region_ck`
- `temperature_zone`, `storage_type`, `capacity_uom`, `storage_capacity_qty`
- `dock_count`, `picking_type`, `throughput_class`
- `working_calendar_ck`, `shift_pattern`, `working_days_mask`
- `inbound_lead_time_days`, `outbound_lead_time_days`
- `replenishment_policy`, `safety_stock_policy`, `service_level_target_pct`
- `active_flag`, `status_code`, `status_reason`
- `open_date`, `close_date`
- `effective_from`, `effective_to`, `is_current`

### 2) Generic Attribute Columns
Table: `dim_location_scd` (same table, extensible section)

Generic fields:
- `attribute_001` ... `attribute_150`

Recommended storage type:
- store as `VARCHAR` at ingestion/conformance layer for maximum compatibility
- cast/validate to typed values in derived views as needed

## Metadata Layer for Attribute Semantics

### `meta_location_attribute_definition`
Defines the business meaning of each generic attribute by client.

Fields:
- `client_id`
- `attribute_code` (`attribute_001` ... `attribute_150`)
- `attribute_display_name` (example: `Warehouse Class`, `Cold Storage Flag`, `Dock Count`)
- `attribute_datatype` (`string`, `integer`, `decimal`, `boolean`, `date`)
- `attribute_domain` (optional controlled list/domain)
- `is_required`
- `effective_from`, `effective_to`
- `version`

### `map_location_source_attribute_to_canonical`
Maps source columns to canonical generic attributes.

Fields:
- `client_id`, `source_system`, `source_table`, `source_column`
- `attribute_code` (`attribute_001` ... `attribute_150`)
- `mapping_rule` (1:1, transform expression, lookup)
- `transformation_rule`
- `confidence`, `rule_version`
- `effective_from`, `effective_to`, `is_active`

## Retrieval and Consumption Pattern
- Use canonical table for broad analytics.
- Use client-specific semantic view for readability:
  - `vw_location_client_<client_id>` exposes business-friendly column aliases from metadata.
- UI and APIs read semantic view by default; raw generic attributes remain available for advanced use.

## Validation Rules
- each `client_id + attribute_code` must have at most one active definition at a point in time
- mapped source columns must resolve to valid `attribute_code` values
- no duplicate active mappings for same `client_id + source_table + source_column`
- datatype compatibility checks between source value and declared `attribute_datatype`

## Performance Guidance
- keep `dim_location_scd` partitioned by `effective_from` period
- avoid scanning all 150 attributes when not needed; select only mapped/used columns
- create curated per-client views for commonly queried attributes

## MVP Implementation Order
1. Create `dim_location_scd` core fields.
2. Add `attribute_001` to `attribute_150` columns.
3. Create `meta_location_attribute_definition`.
4. Create `map_location_source_attribute_to_canonical`.
5. Build client semantic views (`vw_location_client_<client_id>`).
6. Add validation checks in ingestion and publish pipelines.

## Final Recommendation
Use a hybrid model: stable core location attributes plus `attribute_001..attribute_150` as a metadata-driven extensible layer. This gives strong standardization while allowing rapid client-specific mapping without schema redesign.
