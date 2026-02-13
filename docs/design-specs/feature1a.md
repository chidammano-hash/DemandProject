# Feature 1A: Item Dimension

## Purpose
Define a standard item dimension for planning.

## Table
`dim_item`

## Required Fields
- `item_no` ( item number)
- `item_desc` ( item description)
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

## Internal Fields
- `item_sk`
- `item_ck` (same as `item_no`)
- `load_ts`
- `modified_ts`

## Rules
- one row per `item_ck` (`item_no`)
- required fields must be populated
- numeric fields must be valid: `case_weight`, `cpl`, `cpp`, `lpp`, `bpc`, `bottle_pack`, `pack_case`, `item_proof`

## MVP
1. Create `dim_item` with required fields and internal fields.
