# Feature 3: Customer Dimension (MVP)

## Purpose
Define a slim, analytics-ready customer dimension for the unified demand MVP.

## Table
`dim_customer`

## Required Fields
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

## Internal Fields
- `customer_sk`
- `customer_ck` (composite key: `site` + `-` + `customer_no`)
- `load_ts`
- `modified_ts`

## Rules
- one row per `customer_ck` (`site-customer_no`)
- `site` and `customer_no` are required and define identity
- non-numeric descriptive attributes remain as text in MVP
- if duplicate keys exist during load, the latest row in file order wins

## Source Mapping (MVP)
Source file: `datafiles/customerdata.csv`

- `site` -> `site`
- `customer_no` -> `customer_no`
- `customer_name` -> `customer_name`
- `city` -> `city`
- `state` -> `state`
- `zip` -> `zip`
- `premise_code` -> `premise_code`
- `status` -> `status`
- `license_name` -> `license_name`
- `store_type_desc` -> `store_type_desc`
- `chain_type_desc` -> `chain_type_desc`
- `state_chain_name` -> `state_chain_name`
- `corp_chain_name` -> `corp_chain_name`
- `rpt_channel_desc` -> `rpt_channel_desc`
- `rpt_sub_channel_desc` -> `rpt_sub_channel_desc`
- `rpt_ship_type_desc` -> `rpt_ship_type_desc`
- `customer_acct_type_desc` -> `customer_acct_type_desc`
- `delivery_freq_code` -> `delivery_freq_code`

## MVP
1. Normalize `customerdata.csv` into `customerdata_clean.csv`.
2. Load into Postgres `dim_customer` with `customer_ck = site-customer_no`.
3. Publish to Iceberg as `iceberg.silver.dim_customer`.
4. Expose via unified API and UI (`/domains/customer` and `http://127.0.0.1:5173/?domain=customer`).
