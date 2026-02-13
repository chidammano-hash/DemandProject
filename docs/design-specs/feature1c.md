# Feature 1C: Customer Dimension (Extensible Attributes)

## Objective
Design a client-agnostic customer dimension that:
- supports core customer attributes required by all customers
- supports expandable attributes where names and meanings differ by client
- provides a standard generic attribute structure (`attribute_001` to `attribute_150`)
- enables client-specific mapping from source attributes to canonical attributes

## Scope
This sub-feature is focused only on customer master data and customer attribute extensibility.

## Design Principles
- `Canonical Core + Extensible Layer`: keep stable core attributes and flexible client-defined attributes.
- `Metadata-Driven`: attribute meaning, datatype, and usage are controlled by metadata tables.
- `Client Mapping First`: source-to-canonical attribute mapping is explicit and versioned.
- `Backward Compatible`: adding new client attributes should not break existing pipelines.

## Customer Dimension Structure

### 1) Core Customer Dimension (SCD2)
Table: `dim_customer_scd`

Required fields:
- `customer_sk`, `customer_ck`
- `customer_name`, `channel`, `sub_channel`
- `account_level_1`, `account_level_2`, `account_level_3`
- `segment`, `customer_group`, `banner`, `route_to_market`
- `ship_to_ck`, `bill_to_ck`, `payer_ck`
- `default_location_ck`, `sales_territory_ck`
- `priority_tier`, `service_tier`, `service_level_target_pct`
- `order_cycle_code`, `delivery_frequency_code`, `min_drop_size`
- `fulfillment_terms`, `incoterms_code`, `payment_terms_code`, `credit_class`, `credit_limit_amount`
- `tax_region_code`, `currency_code`, `price_group_code`, `discount_group_code`
- `on_premise_flag`, `key_account_flag`, `modern_trade_flag`
- `active_flag`, `status_code`, `status_reason`
- `onboard_date`, `offboard_date`
- `effective_from`, `effective_to`, `is_current`

### 2) Generic Attribute Columns
Table: `dim_customer_scd` (same table, extensible section)

Generic fields:
- `attribute_001` ... `attribute_150`

Recommended storage type:
- store as `VARCHAR` at ingestion/conformance layer for maximum compatibility
- cast/validate to typed values in derived views as needed

## Metadata Layer for Attribute Semantics

### `meta_customer_attribute_definition`
Defines the business meaning of each generic attribute by client.

Fields:
- `client_id`
- `attribute_code` (`attribute_001` ... `attribute_150`)
- `attribute_display_name` (example: `On-Premise Flag`, `Priority Tier`, `Route Class`)
- `attribute_datatype` (`string`, `integer`, `decimal`, `boolean`, `date`)
- `attribute_domain` (optional controlled list/domain)
- `is_required`
- `effective_from`, `effective_to`
- `version`

### `map_customer_source_attribute_to_canonical`
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
  - `vw_customer_client_<client_id>` exposes business-friendly column aliases from metadata.
- UI and APIs read semantic view by default; raw generic attributes remain available for advanced use.

## Validation Rules
- each `client_id + attribute_code` must have at most one active definition at a point in time
- mapped source columns must resolve to valid `attribute_code` values
- no duplicate active mappings for same `client_id + source_table + source_column`
- datatype compatibility checks between source value and declared `attribute_datatype`

## Performance Guidance
- keep `dim_customer_scd` partitioned by `effective_from` period
- avoid scanning all 150 attributes when not needed; select only mapped/used columns
- create curated per-client views for commonly queried attributes

## MVP Implementation Order
1. Create `dim_customer_scd` core fields.
2. Add `attribute_001` to `attribute_150` columns.
3. Create `meta_customer_attribute_definition`.
4. Create `map_customer_source_attribute_to_canonical`.
5. Build client semantic views (`vw_customer_client_<client_id>`).
6. Add validation checks in ingestion and publish pipelines.

## Final Recommendation
Use a hybrid model: stable core customer attributes plus `attribute_001..attribute_150` as a metadata-driven extensible layer. This gives strong standardization while allowing rapid client-specific mapping without schema redesign.
