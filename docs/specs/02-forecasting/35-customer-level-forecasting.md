# 35 — Customer-Level Forecasting

| | |
|---|---|
| **Status** | Implemented |
| **Models** | Chronos 2 Enriched (`chronos2_enriched`) with customer-only Croston/SBA (`croston`) fallback |
| **History window** | Latest 18 fully closed months |
| **Forecast horizon** | 18 monthly periods beginning with the current system month |
| **Forecast grain** | Item + location + customer + forecast month |
| **Related** | [Customer Demand Fact](../01-foundation/07-customer-demand-fact.md), [Chronos 2 Enriched](18-chronos-foundation-models.md) |

## 1. Goal

Generate monthly customer-level demand forecasts from historical customer demand.
Each run produces 18 future months for active item-location-customer series,
beginning with the month containing the system date. Series with no sales in
the latest six fully closed months are ignored and produce no forecast rows.
Full-history active series use Chronos 2E; remaining active series use Croston/SBA.

This feature only generates customer-level forecasts. It does not create a
consensus plan, adjust an item-location forecast, select or replace a champion,
or provide AI/planner overrides.

## 2. Date contract

The run derives its dates at execution time. Dates are not hardcoded in the
model or UI.

```text
forecast_start = first day of the month containing the system date
history_end    = final day of the month before forecast_start
history_start  = first day of the month 18 months before forecast_start
forecast_end   = final day of the 18th forecast month
```

Example for a system date in July 2026:

| Window | Inclusive months |
|---|---|
| Historical input | January 2025 through June 2026 |
| Forecast output | July 2026 through December 2027 |

Only fully closed months are historical input. Actuals from the current partial
month or any future month must never enter the model context.

## 3. Scope

### In scope

- read the latest 18 closed months from the normalized customer-demand fact;
- generate one 18-month Chronos 2E forecast for each full-history customer
  series and one Croston/SBA forecast for each remaining series;
- ignore customer-SKUs with no `sales_qty` in the latest six closed months;
- commit resumable batches and report exact completed customer-SKU counts;
- persist point forecasts and, when supported, confidence intervals with run
  lineage;
- expose run status and generated rows for filtering, viewing, and export; and
- derive item-location totals by summing the generated customer rows for display
  or validation.

### Out of scope

- AI recommendations or AI-assisted adjustments;
- planner edits, previews, approvals, overrides, or audit workflows;
- reconciliation to the item-location champion or any other top-level target;
- customer-level backtesting, tuning, model competition, or champion selection;
- candidate, staging, promotion, or production-release workflows;
- changing `fact_production_forecast` or the active item-location champion;
- customer-level inventory planning; and
- weekly or daily customer forecasts.

An item-location total shown by this feature is only the arithmetic sum of its
generated customer forecasts. It is not a reconciled plan and is not written
back to the item-location forecasting lifecycle.

## 4. Source data and grain

The generator reads `fact_customer_demand_monthly`; it does not read raw source
files directly.

| Field | Use |
|---|---|
| `item_id` | Item key |
| `location_id` | Canonical location key resolved by the existing ETL |
| `customer_no` | Customer key within the source location/site |
| `startdate` | Historical month |
| `demand_qty` | Forecast target |
| `sales_qty` | Six-closed-month activity filter only; not the forecast target |
| `oos_qty` | Historical diagnostic retained by the source fact; not a v1 model input |

The unique output grain is:

```text
run_id + item_id + location_id + customer_no + forecast_month
```

Customer names, addresses, and other direct customer identifiers are not model
features. The stored forecast retains the customer key needed to retrieve the
series.

## 5. Readiness and history preparation

A series enters customer forecasting only when it has positive `sales_qty` in
the latest six fully closed months. A dormant series is counted as ignored and
does not create 18 zero rows. This filter does not change or replace the
separate item-location forecast.

An active series is eligible for Chronos 2E when it has all of the following:

- a valid item, location, and customer key;
- 18 consecutive closed historical months after monthly densification;
- no duplicate rows at item-location-customer-month grain after standard
  aggregation; and
- at least one positive-demand month in the 18-month history window.

Missing months inside the 18-month window are filled with zero demand. Negative
quantities are rejected as a data-quality error rather than silently changed.
Active series with insufficient history are routed to the configured
Croston/SBA fallback. Invalid keys, duplicate grains, negative demand, and
missing source freshness remain run-level data-quality blockers.

The run-level readiness response reports:

- the resolved system month and exact history/forecast windows;
- source freshness through the latest closed month;
- total observed, forecastable, Chronos-routed, Croston-routed, and dormant
  ignored series counts;
- unresolved key and duplicate counts;
- negative-demand row counts; and
- a clear corrective action when generation cannot start.

The all-history series bounds used by readiness and generation come from
`mv_customer_demand_series_profile`. The materialized profile is refreshed by
the standard customer-demand post-load lifecycle and includes the last
positive-sales month. Request-time readiness and manifest queries therefore do
not scan the partitioned fact table. This keeps the readiness API inside the
normal statement timeout without weakening the first-observed-month or recent-
sales eligibility rules.

## 6. Forecast generation

Chronos 2E receives one causal 18-month `demand_qty` sequence per eligible
item-location-customer series. The shared adapter derives its causal calendar
and Fourier features from those timestamps. Series that are not Chronos
eligible receive a configured Syntetos-Boylan bias-adjusted Croston forecast
from the same bounded history. V1 does not pass customer PII, out-of-stock
quantities, future demand, or any other future operational value to either
route.

Generation rules:

1. Resolve the system month and the two date windows once at run start.
2. Build the durable active-series manifest from the refreshable series
   profile, then load only each claimed batch's bounded fact history.
3. Ignore series with no sales in the latest six closed months. Route active
   full-history series to Chronos 2E and active short-history series to Croston/SBA.
4. Claim batches with `FOR UPDATE SKIP LOCKED`. One Chronos worker owns the
   model and uses `device: auto` (MPS/CUDA when available, CPU otherwise); six
   configured CPU workers process Croston batches in parallel.
5. Generate exactly 18 consecutive monthly predictions beginning with the
   resolved forecast start.
6. Clip point forecasts and interval bounds to zero.
7. Commit each batch's forecast rows, checksum, and completed-series count in
   one transaction. A failed batch cannot leave partial output.
8. Resume only unfinished batches after cancellation, failure, retry, or
   service restart; already completed batches are not recomputed.
9. Validate every batch and the exact final row count before atomically marking
   the run completed. Partial output is never exposed as successful.

Either route must fail the affected run on inference errors. It must not replace
a failed non-zero-history prediction with zero, because zero is a valid demand
forecast and would hide the failure.

Croston is scoped only to this customer generation feature. Its parameters live
outside the governed algorithm roster, so it cannot appear in item-location
tuning, backtesting, champion selection, staging, or production.

## 7. Stored output

### `customer_forecast_run`

One immutable record per generation request containing:

- `run_id` and durable-job lineage;
- status and timestamps;
- resolved history start/end and forecast start/end;
- model and configuration version;
- source-data checksum or equivalent lineage marker;
- completed and skipped series counts plus Chronos/Croston route counts;
- total/completed batch and customer-SKU counters for progress and ETA;
- skipped-series reason counts; and
- terminal error summary when the run fails.

### `fact_customer_forecast`

Run-versioned forecast rows containing:

- `run_id`, `item_id`, `location_id`, `customer_no`, and `forecast_month`;
- point forecast quantity;
- lower and upper bounds when available; and
- model, batch, history-end, and generated-at lineage.

### `customer_forecast_batch` and `customer_forecast_batch_series`

The batch manifest records route, expected series count, claim attempts,
status, checksum, row count, and timestamps. The series ledger assigns each
active customer-SKU to exactly one batch. Completed batch rows are durable
recovery checkpoints, not temporary UI progress.

Generated runs are immutable. Read APIs default to the latest successfully
completed run and accept an explicit `run_id` for reproducibility. A failed,
cancelled, or incomplete run never replaces the latest completed result.

## 8. API and durable job

| Method | Path | Purpose |
|---|---|---|
| GET | `/customer-forecast/readiness` | Return the resolved windows, source freshness, eligibility, and blockers |
| POST | `/customer-forecast/generate` | Launch one durable 18-month generation run |
| GET | `/customer-forecast/runs/latest` | Return the latest run for progress and retry guidance; `completed_only=true` preserves access to the last successful result |
| GET | `/customer-forecast/runs/{run_id}` | Return status, counts, dates, and errors for one run |
| POST | `/customer-forecast/runs/{run_id}/cancel` | Request cancellation of an active run |
| POST | `/customer-forecast/runs/{run_id}/retry` | Resume unfinished batches for a failed/cancelled run with the same configuration |
| GET | `/customer-forecast/series` | Return history and generated forecast for one item-location-customer selection |
| GET | `/customer-forecast/export` | Stream generated rows for a completed run and optional filters |

Generation uses the existing durable-job framework for progress, cancellation,
retry, restart recovery, and terminal-status reconciliation. Retrying a
manifested run resumes the same run and preserves completed batches. A
configuration change requires a new generation. Write endpoints use the
project API-key guard.

## 9. UI behavior

Provide one **Customer Forecast** view focused on generation and results.

The view shows:

- the system month, 18-month history window, and 18-month forecast window;
- readiness status with actionable blockers;
- forecastable coverage split between Chronos 2E and Croston;
- a **Generate Customer Forecasts** action;
- exact completed/total customer-SKU and batch progress, throughput ETA,
  cancel, resumable retry, and failure details;
- filters for item, location, and customer;
- an actual-versus-forecast chart with confidence intervals when available;
- a monthly results table and export action; and
- the selected run ID, generation timestamp, and model used for the selected
  customer series.

The view is read-only after generation. It has no adjustment, approval,
promotion, champion, reconciliation, or AI controls. The current Chronos
adapter returns a point forecast; the persisted interval columns remain null
until that adapter exposes calibrated bounds.

## 10. Implementation

- DDL: `sql/210_create_customer_forecast.sql`,
  `sql/211_create_customer_demand_series_profile.sql`,
  `sql/212_add_customer_forecast_model_routes.sql`, and
  `sql/213_add_customer_forecast_batches.sql`; existing installations extend
  the activity profile with `sql/214_add_customer_series_activity.sql`, while
  `sql/215_enforce_customer_batch_lineage.sql` couples every batch to its run
- Croston implementation: `common/ml/croston.py`
- Generation service: `common/services/customer_forecast.py`
- Batch execution: `common/services/customer_forecast_batches.py`
- Durable runner: `scripts/forecasting/generate_customer_forecasts.py`
- API: `api/routers/forecasting/customer_forecast.py`
- UI: **Forecasting → Customer Forecast**
- Tests: `tests/unit/test_customer_forecast.py`,
  `tests/api/test_customer_forecast.py`, and
  `frontend/src/tabs/forecast/__tests__/CustomerForecastPanel.test.tsx`

## 11. Acceptance criteria

- A July 2026 run reads January 2025 through June 2026 and forecasts July 2026
  through December 2027.
- Every active series has exactly 18 consecutive forecast rows; series with no
  sales in the latest six closed months have no output rows.
- Chronos-eligible active series use `chronos2_enriched`; every remaining active
  valid series uses customer-only `croston`.
- Forecasts use `demand_qty`, not `sales_qty`.
- No current-partial-month or future actual enters the model context.
- Output is unique at run-item-location-customer-month grain.
- Point forecasts and interval bounds are non-negative and ordered correctly.
- Every row retains run, model, configuration, source, and history-end lineage.
- Model route counts are retained at run level and each forecast row retains its
  actual generating model.
- Partial, failed, or cancelled runs are not exposed as the latest completed run.
- Completed batches survive retry/restart, and progress shows exact completed
  customer-SKU and batch counts with an estimated time remaining.
- Summing customer rows may provide an item-location total for display, but no
  reconciliation or write to the item-location forecast occurs.
- The UI contains no AI, planner-adjustment, approval, champion, staging, or
  production-promotion workflow for customer forecasts.
