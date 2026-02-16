# Feature E: DFU Item-Location Attributes (MVP)

## Purpose
Define a DFU attribute dimension at item-location grain using `dfu.txt`.

## Grain
- one row per item-customerGroup-location triple (`dmdunit`, `dmdgroup`, `loc`)
- `dmdgroup` is present in source but currently aggregated at `ALL`; customer-level detail is out of scope for this feed

## Table
`dim_dfu`

## Internal Fields
- `dfu_sk`
- `dfu_ck` (`dmdunit` + `_` + `dmdgroup` + `_` + `loc`)
- `load_ts`
- `modified_ts`

## Required Fields
- `dmdunit`
- `loc`
- `dmdgroup`
- `brand`
- `brand_desc`
- `region`
- `state_plan`
- `sales_div`
- `prod_cat_desc`
- `prod_class_desc`
- `subclass_desc`
- `supplier_desc`
- `producer_desc`
- `size`
- `brand_size`
- `file_dt`
- `histstart`
- `cluster_assignment`

## Additional Attributes Loaded (MVP)
- `abc_vol`
- `ded_div_sw`
- `execution_lag`
- `otc_status`
- `premise`
- `prod_subgrp_desc`
- `service_lvl_grp`
- `supergroup`
- `total_lt`
- `vintage`
- `purge_sw`
- `alcoh_pct`
- `bot_type_desc`
- `cnty`
- `dom_imp_opt`
- `grape_vrty_desc`
- `material`
- `proof`
- `sop_ref`

## Source Mapping (MVP)
Source file: `datafiles/dfu.txt` (pipe-delimited, header row)

- `DMDUNIT` -> `dmdunit`
- `DMDGROUP` -> `dmdgroup`
- `LOC` -> `loc`
- `BRAND` -> `brand`
- `U_ABC_VOL` -> `abc_vol`
- `U_BRAND_DESC` -> `brand_desc`
- `U_DED_DIV_SW` -> `ded_div_sw`
- `U_EXECUTION_LAG` -> `execution_lag`
- `U_OTC_STATUS` -> `otc_status`
- `U_PREMISE` -> `premise`
- `U_PROD_SUBGRP_DESC` -> `prod_subgrp_desc`
- `U_REGION` -> `region`
- `U_SERVICE_LVL_GRP` -> `service_lvl_grp`
- `U_SIZE` -> `size`
- `U_STATE_PLAN` -> `state_plan`
- `U_SUPERGROUP` -> `supergroup`
- `U_SUPPLIER_DESC` -> `supplier_desc`
- `U_TOTAL_LT` -> `total_lt`
- `U_VINTAGE` -> `vintage`
- `U_SALES_DIV` -> `sales_div`
- `U_PURGE_SW` -> `purge_sw`
- `U_ALCOH_PCT` -> `alcoh_pct`
- `U_BOT_TYPE_DESC` -> `bot_type_desc`
- `U_BRAND_SIZE` -> `brand_size`
- `U_CNTY` -> `cnty`
- `U_DOM_IMP_OPT` -> `dom_imp_opt`
- `U_GRAPE_VRTY_DESC` -> `grape_vrty_desc`
- `U_MATERIAL` -> `material`
- `U_PROD_CAT_DESC` -> `prod_cat_desc`
- `U_PRODUCER_DESC` -> `producer_desc`
- `U_PROOF` -> `proof`
- `U_SUBCLASS_DESC` -> `subclass_desc`
- `U_PROD_CLASS_DESC` -> `prod_class_desc`
- `FILE_DT` -> `file_dt`
- `HISTSTART` -> `histstart`
- `U_CLUSTER_ASSIGNMENT` -> `cluster_assignment`
- `U_SOP_REF` -> `sop_ref`

## MVP Pipeline
1. Normalize source:
   - `make -C mvp/demand normalize-dfu`
   - output: `mvp/demand/data/dfu_clean.csv`
2. Load to Postgres:
   - `make -C mvp/demand load-dfu`
   - table: `dim_dfu`
3. Publish to Iceberg:
   - `make -C mvp/demand spark-dfu`
   - table: `iceberg.silver.dim_dfu`
4. Query and UI:
   - API: `/domains/dfu`, `/domains/dfu/page`, `/dfus`, `/dfus/page`
   - UI: `http://127.0.0.1:5173/?domain=dfu`
