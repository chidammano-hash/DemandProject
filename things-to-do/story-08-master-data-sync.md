# Story 08 — Master Data Synchronization & Data Governance

## Problem
All master data (item, location, customer, DFU) is loaded via manual CSV upload. There is no automated synchronization with source ERP/MDM systems, no change detection, no data stewardship workflow, and no product hierarchy management. Master data staleness causes cascading errors in planning.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `product_hierarchy.csv` | CSV | hierarchy_node | node_id, parent_node_id, level (division|category|subcategory|brand|item), node_name, node_code, sort_order | MDM/ERP |
| **F2** | `location_hierarchy.csv` | CSV | hierarchy_node | node_id, parent_node_id, level (region|state|dc|store), node_name, lat, lng, timezone | MDM/ERP |
| **F3** | `customer_hierarchy.csv` | CSV | hierarchy_node | node_id, parent_node_id, level (national_chain|regional_chain|account|store), node_name, channel | CRM |
| **F4** | `item_lifecycle.csv` | CSV | item × event | item_no, event_type (introduction|phase_out|substitute|discontinue), event_date, substitute_item_no, reason, planned_last_ship_date | Product Mgmt |
| **F5** | `uom_conversion.csv` | CSV | item × uom | item_no, from_uom, to_uom, conversion_factor | ERP |
| **F6** | `master_data_changelog.csv` | CSV | change_event | entity_type, entity_key, field_name, old_value, new_value, changed_by, changed_at, change_reason | CDC/MDM |

## Incremental Implementation

### Phase 1: Product & Location Hierarchies (1 sprint)
- `sql/096_create_hierarchies.sql` — `dim_product_hierarchy`, `dim_location_hierarchy`, `dim_customer_hierarchy`
- Recursive CTE for roll-up aggregation at any hierarchy level
- API: `GET /hierarchy/{type}/tree`, `/hierarchy/{type}/children/{node_id}`
- Extend global filter bar: drill from division → category → subcategory → brand → item
- Panel: Hierarchy browser with drag-to-filter capability

### Phase 2: Item Lifecycle Management (1 sprint)
- `sql/097_create_lifecycle.sql` — `fact_item_lifecycle_events`
- New item introduction: auto-create DFU, assign initial forecast (analog-based)
- Phase-out: wind-down forecast, flag remaining inventory for markdown
- Substitution chain: A → B mapping for demand transfer
- API: `GET /lifecycle/introductions`, `/phase-outs`, `/substitutions`
- Panel: Lifecycle timeline per item, substitution chain visualization

### Phase 3: Automated Master Data Sync (1 sprint)
- `scripts/sync_master_data.py` — CDC-based incremental sync from ERP
- Change detection: compare incoming vs current, log differences
- `sql/098_create_master_data_audit.sql` — `fact_master_data_changelog`
- Data steward workflow: auto-accept minor changes, flag major changes for review
- API: `GET /master-data/changes`, `/pending-review`, `POST /approve/{change_id}`
- Panel: Data steward queue with change impact preview

### Phase 4: UOM & Cross-Reference Management (1 sprint)
- `dim_uom_conversion` — supports planning in cases, units, pallets, weight
- `dim_item_xref` — cross-reference table (ERP item → supplier item → customer item → UPC/EAN)
- All planning calculations respect UOM conversions
- API: `GET /master-data/uom/{item_no}`, `/xref/{item_no}`

## Dependencies
- Requires MDM system or ERP master data exports (SAP MDG, Informatica MDM, Stibo)
- CDC can use database triggers, Debezium, or scheduled delta exports
- Product hierarchy is critical for aggregate-level planning and reporting
- Current `dim_item.category` and `dim_item.class` are flat — hierarchy adds depth

## Business Value
- Eliminates stale master data errors (wrong supplier, discontinued items still ordered)
- Hierarchy-based planning enables top-down/bottom-up reconciliation
- Item lifecycle management prevents ordering discontinued products
- Automated sync reduces manual data entry effort by 80%
