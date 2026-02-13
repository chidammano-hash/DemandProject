# Feature 1D: Date Dimension (Generic and Common)

## Objective
Define a single reusable date/calendar dimension used by all domains (`item`, `location`, `customer`) and all planning grains (`daily`, `weekly`, `monthly`).

## Scope
This sub-feature is a shared/common dimension for:
- transactional facts (daily actuals)
- forecast and KPI facts (weekly/monthly period views)
- fiscal and reporting calendars
- holiday/event-aware analytics

## Design Principles
- `Single Source of Calendar Truth`: one canonical date dimension for all datasets.
- `Multi-Calendar Support`: standard calendar + fiscal calendar in one model.
- `Planning Ready`: explicit weekly/monthly period fields and keys.
- `Stable and Reusable`: no client-specific logic in core date table.

## Date Dimension Structure

### 1) Core Date Table
Table: `dim_date`

Required fields:
- `date_key` (DATE, PK)
- `day_number_in_week`, `day_number_in_month`, `day_number_in_year`
- `day_name`, `day_name_short`
- `week_start_date`, `week_end_date`, `week_of_year`, `iso_week_of_year`
- `month_start_date`, `month_end_date`, `month_number`, `month_name`, `month_name_short`
- `quarter_number`, `quarter_name`
- `year_number`
- `is_weekend`, `is_month_end`, `is_quarter_end`, `is_year_end`
- `fiscal_week`, `fiscal_month`, `fiscal_quarter`, `fiscal_year`
- `fiscal_period_start_date`, `fiscal_period_end_date`
- `holiday_flag`, `holiday_name`, `event_flag`
- `working_day_flag`
- `effective_from`, `effective_to`, `is_current` (optional; include if calendar versioning is required)

### 2) Planning Period View/Table
Table/View: `dim_planning_period`

Required fields:
- `planning_grain` (`WEEKLY`, `MONTHLY`)
- `period_id`
- `period_start_date`
- `period_end_date`
- `period_label`
- `year_number`, `quarter_number`, `month_number`, `week_of_year`
- `fiscal_year`, `fiscal_quarter`, `fiscal_month`, `fiscal_week`

## Mapping and Join Standards
- Daily facts join on `date_key`.
- Weekly/monthly forecast and KPI facts join on:
  - `planning_grain`
  - `period_start_date` (or `period_id` if standardized globally)
- All downstream marts must use this dimension for period labeling and rollups.

## Validation Rules
- no missing dates in range (continuous date coverage)
- unique `date_key`
- consistent week/month boundaries
- valid fiscal mappings for every `date_key`
- one active period definition per `planning_grain + period_id`

## Performance Guidance
- `dim_date` is small; no heavy partitioning needed.
- create lightweight serving views for common rollups:
  - `vw_date_weekly`
  - `vw_date_monthly`
- cache/join-broadcast this dimension in Spark for large fact joins.

## MVP Implementation Order
1. Build `dim_date` with standard + fiscal attributes.
2. Build `dim_planning_period` for weekly/monthly planning.
3. Align forecast/KPI facts to `planning_grain` + period keys.
4. Add holiday/event enrichment and validation checks.

## Final Recommendation
Use one canonical date dimension plus a planning-period layer as shared infrastructure across all forecasting domains. Keep it generic, stable, and client-independent.
