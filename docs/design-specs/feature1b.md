# Feature 1B: Location Dimension

## Purpose
Define a standard location dimension for planning.

## Table
`dim_location`

## Required Fields
- `location_id` (location identifier)
- `site_id` (site grouping identifier)
- `site_desc` (site description)
- `state_id` (state code)
- `primary_demand_location` (`Y`/`N` flag)

## Internal Fields
- `location_sk`
- `location_ck` (same as `location_id`)
- `load_ts`
- `modified_ts`

## Rules
- one row per `location_ck` (`location_id`)
- required fields must be populated
- `primary_demand_location` should be constrained to `Y` or `N`

## Source Mapping (MVP)
Source file: `datafiles/locationdata.csv`

- `location_id` -> `location_id`
- `site_id` -> `site_id`
- `site_desc` -> `site_desc`
- `state_id` -> `state_id`
- `primary_demand_location` -> `primary_demand_location`

## MVP
1. Create `dim_location` with required fields and internal fields.
