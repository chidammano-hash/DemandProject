# Feature 5: Forecast Accuracy KPI Cards

## Objective
Define and surface forecast-accuracy KPIs for external monthly forecasts in the forecast UI.

## Scope
- Dataset: `fact_external_forecast_monthly`
- Forecast measure: `basefcst_pref`
- Actual measure: `tothist_dmd`
- Grain: monthly (`startdate`) for selected filter context (item, location, lag/date scope)

## KPI Definitions
Let:
- `F` = forecast (`basefcst_pref`)
- `A` = actual (`tothist_dmd`)
- `E` = error = `F - A`

For the selected filtered set of rows:

1. Total Forecast
- Formula: `SUM(F)`

2. Total Actual
- Formula: `SUM(A)`

3. Absolute Error
- Formula: `SUM(ABS(E))`

4. Bias
- Formula: `(SUM(F) / SUM(A)) - 1`
- If `SUM(A) = 0`, return `NULL`.

5. WAPE (%)
- Formula: `100 * SUM(ABS(E)) / ABS(SUM(A))`
- If `SUM(A) = 0`, return `NULL`.

6. MAPE (%)
- Formula: `100 * AVG(ABS(E) / ABS(A))` for rows where `A != 0`
- Rows with `A = 0` are excluded.

7. Forecast Accuracy (%)
- Formula: `100 - WAPE%`
- If `WAPE%` is `NULL`, return `NULL`.

## UI Mapping (Forecast Domain)
Show the following KPI cards in the forecast analytics panel:
- Forecast Accuracy
- WAPE
- MAPE
- Total Forecast
- Total Actual
- Absolute Error
- Bias

KPI window selector:
- Add a dropdown `Accuracy Window (Months)` with values `1..12`.
- Selected value `N` means KPI cards are computed on the latest `N` monthly buckets (by `startdate`) within active filters.
- Accuracy metrics shown on cards (`Forecast Accuracy`, `WAPE`, `MAPE`) are **average monthly** values across this `N`-month window.
- Volume/error cards (`Total Forecast`, `Total Actual`, `Absolute Error`) are summed over the same `N`-month window.
- `Bias` is computed as `(SUM(Forecast) / SUM(History)) - 1` across the `N`-month window.

Trend chart requirement:
- Support monthly `Forecast Accuracy %` series in the trend chart.
- Formula per month bucket:
  - `Forecast Accuracy % = 100 - (100 * SUM(ABS(F - A)) / ABS(SUM(A)))`
  - If monthly `SUM(A) = 0`, value is null for that month.
- This measure should be selectable alongside volume measures (`basefcst_pref`, `tothist_dmd`).

These cards are computed on the backend using the same active filters used for trend/chart queries.

## Dependencies
- Feature 4 (fact tables: `fact_sales_monthly`, `fact_external_forecast_monthly`)
