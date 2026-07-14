# 35 — Customer-Level Forecasting and Bottom-Up Blend

| | |
|---|---|
| **Status** | Implemented |
| **Customer model** | Croston/SBA (`croston`) |
| **Customer history window** | Latest 18 fully closed months |
| **Customer forecast horizon** | 18 monthly periods beginning with the planning month |
| **Customer forecast grain** | Item + location + customer + forecast month |
| **Blend model** | Customer bottom-up + source champion (`customer_bottom_up_blend`) |
| **Blend output grain** | Item + location + forecast month on the active production spine |
| **Related** | [Customer Demand Fact](../01-foundation/07-customer-demand-fact.md), [Production Forecast](08-production-forecast.md), [Forecast Release Readiness](34-forecast-release-readiness.md) |

## 1. Goal

Generate monthly customer demand forecasts, aggregate them bottom-up to the
canonical item-location grain, normalize the demand target to the item-location
sales target, and combine that signal with the active production champion. The
result is an immutable, reviewable `customer_bottom_up_blend` release draft.

Customer generation remains its own run-versioned bounded context. It does not
mutate customer forecasts after completion and does not directly rewrite the
active item-location release. A blend can enter the normal staging and
promotion lifecycle only when a matching completed historical backtest meets
the configured common-cohort and no-WAPE-degradation thresholds. Nothing in
this workflow auto-promotes a draft.

## 2. Date contracts

### 2.1 Forward customer generation

The run resolves its dates once from the planning month:

```text
forecast_start = first day of the planning month
history_end    = final day of the preceding month
history_start  = first day of the month 18 months before forecast_start
forecast_end   = final day of the 18th forecast month
```

For a July 2026 planning month, customer generation reads January 2025 through
June 2026 and forecasts July 2026 through December 2027. Current-partial-month
or future actuals never enter the model context.

### 2.2 Historical blend backtest

The backtest evaluates the latest six fully closed forecast months with six
causal one-month-ahead origins. Each origin has at least six closed training
months. Customer forecasts, fulfillment ratios, champion forecasts, and
actuals are all resolved as they were knowable at that origin. The run must not
use later customer demand, sales, champion assignments, or actuals as model
inputs. Historical activity is evaluated independently at every origin from
the all-series demand population; the backtest does not reuse the current
forward run's active-series manifest. Historical champion rows use their
stamped execution lag, not mutable current SKU metadata.

## 3. Scope

### In scope

- generate an 18-month Croston/SBA forecast for every active valid customer
  series;
- ignore customer-SKUs with no `sales_qty` in the latest six closed months;
- persist immutable customer forecasts in resumable 10,000-series batches;
- aggregate customer demand forecasts to item-location-month;
- normalize bottom-up demand with a causal 18-month historical fulfillment
  ratio;
- blend normalized customer quantity and the source champion at configured
  50/50 weights;
- retain every component, weight, coverage decision, interval policy, source
  release, customer run, and backtest run as immutable evidence;
- backtest customer bottom-up, source champion, and the blend on one common
  cohort;
- create a reviewable draft on the full active production spine, using
  champion fallback when customer evidence is unavailable; and
- use the existing governed stage and promotion path after all normal release
  gates and the customer-blend backtest gate pass.

### Out of scope

- adding Croston to the canonical five-model item-location competition;
- customer-level tuning, per-customer champion selection, or planner/AI edits;
- adding customer-only item-locations to the production population;
- changing the source customer forecast run or production release in place;
- bypassing the normal staging, release-readiness, interval, lineage, checksum,
  or transactional promotion gates;
- automatic promotion; and
- weekly or daily customer forecasts.

## 4. Source data, targets, and grains

Customer generation reads `fact_customer_demand_monthly`:

| Field | Use |
|---|---|
| `item_id` | Item key |
| `location_id` | Canonical warehouse/location key resolved by ETL |
| `customer_no` | Customer key within the source site |
| `startdate` | Historical month |
| `demand_qty` | Croston/SBA forecast target |
| `sales_qty` | Activity filter and fulfillment-ratio numerator |
| `oos_qty` | Diagnostic only; not a model input |

The immutable customer output grain is:

```text
run_id + item_id + location_id + customer_no + forecast_month
```

The bottom-up and production grains use canonical `location_id` as `loc`:

```text
run_id + item_id + loc + forecast_month
```

Source `warehouse_no` is not a separate planning key. Customer rows loaded
under the same canonical site/location aggregate to the same item-location.
Customer names, addresses, and other direct identifiers are not model inputs.

The customer model predicts demand. The governed item-location release is a
sales-target forecast. The fulfillment normalization in Section 7 is therefore
mandatory before blending; raw customer demand must never be blended directly
with the champion.

## 5. Customer readiness and generation

A series is active when it has positive `sales_qty` in the latest six fully
closed months. Dormant series are counted as ignored and write no forecast
rows. Active series require valid item, location, and customer keys; duplicate
source grains, negative quantities, missing latest-closed-month freshness, and
unresolved keys remain data-quality blockers.

Missing months in the bounded 18-month history are densified with zero demand.
Every active valid series uses configured Syntetos-Boylan bias-adjusted Croston
with `alpha: 0.1` and `variant: sba`. There is no Chronos customer route or
fallback route. Croston remains customer-scoped and outside the governed
five-model algorithm roster.

Generation rules:

1. Resolve planning, history, and forecast windows once.
2. Build the active-series manifest from
   `mv_customer_demand_series_profile`.
3. Claim 10,000-series Croston batches with `FOR UPDATE SKIP LOCKED`.
4. Load only each claimed batch's bounded fact history.
5. Generate exactly 18 non-negative monthly predictions.
6. Commit each batch's series rows, checksum, counts, and status in one
   transaction.
7. Resume only unfinished batches after cancellation, failure, retry, or
   process restart.
8. Mark the run completed only after all batches and exact row counts pass.

Inference errors fail the affected run. The generator must not silently
substitute zero for a failed non-zero-history series because zero is a valid
demand forecast.

## 6. Bottom-up aggregation and coverage

For completed customer run `c`, item `i`, location `l`, and month `m`:

```text
raw_customer_demand_qty(i,l,m)
  = SUM(fact_customer_forecast.forecast_qty over customer_no)
```

The source population is the exact active verified production release selected
by `source_promotion_id` and `source_production_run_id`. It is a left-side
spine:

- every active champion item-location-month produces one draft row;
- a customer-only item-location is excluded; and
- a missing customer row or unusable normalization produces
  `coverage_status='champion_fallback'`.

The current production horizon is 24 months while customer output is 18
months. Months 19–24 therefore pass through the champion. Any other production
DFU/month without qualified customer evidence also passes through the champion.
This preserves production coverage and prevents customer-level breadth from
silently expanding the governed release population.

## 7. Demand-to-sales normalization and blend formula

For each item-location and forecast origin, use up to the latest 18 causal
closed months from `fact_customer_demand_monthly`:

```text
historical_fulfillment_ratio(i,l)
  = clamp(
      SUM(sales_qty) / NULLIF(SUM(demand_qty), 0),
      0.0,
      1.0
    )
```

The ratio is usable only when summed demand is at least `1.0`. Otherwise the
row uses champion fallback. Backtests recompute the ratio independently at
every forecast origin; they never reuse the present-day ratio.

```text
normalized_customer_qty = raw_customer_demand_qty * fulfillment_ratio

blended_qty
  = 0.50 * normalized_customer_qty
  + 0.50 * champion_qty
```

For a blended row, configured and effective customer weights are `0.5`. For a
fallback row, the effective customer weight is `0.0` and
`blended_qty = champion_qty`. Component rows retain raw demand, ratio,
normalized quantity, source champion, configured weights, effective weight,
customer-series count, and coverage status.

### Confidence intervals

Customer forecasts do not currently supply calibrated intervals. The blend
preserves the source champion's asymmetric interval widths by shifting its
bounds with the point-forecast delta:

```text
delta         = blended_qty - champion_qty
blended_lower = max(0, champion_lower + delta)
blended_upper = max(blended_qty, champion_upper + delta)
```

This is `interval_method='champion_width_shift'`. Champion fallback rows use
`champion_passthrough`. If the source champion has no complete interval pair,
both draft bounds remain null and the method is `none`; the normal release
interval gate still applies before promotion.

## 8. Historical blend backtest and promotion gate

`POST /customer-forecast/backtest/generate` launches a config-driven durable
run with these fixed defaults:

| Setting | Value |
|---|---:|
| Evaluation lookback | 6 closed months |
| Minimum causal training history | 6 months |
| Forecast horizon per origin | 1 month |
| Batch size | 10,000 customer series |

Each component row retains the causal customer bottom-up forecast, fulfillment
ratio, normalized customer quantity, source champion forecast, 50/50 blended
quantity, actual sales quantity, weights, customer-series count, and coverage
status. Accuracy uses only the common cohort where customer bottom-up,
champion, blend, and actual are all available at the same DFU-month.

For each of customer bottom-up, source champion, and blend:

```text
WAPE%     = 100 * SUM(ABS(forecast - actual)) / SUM(actual)
MAE       = AVG(ABS(forecast - actual))
bias%     = 100 * SUM(forecast - actual) / SUM(actual)
accuracy% = max(0, 100 - WAPE%)
```

The blend-versus-champion WAPE delta is quantized to the stored six-decimal
metric precision before applying the zero-degradation gate, so binary floating-
point noise cannot reject mathematically equal aggregate errors.

The persisted degradation measure is a WAPE percentage-point delta:

```text
blend_wape_degradation_pct = blend_wape_pct - champion_wape_pct
```

A backtest gate passes only when all of the following are true:

- the run is completed with a canonical component checksum;
- the common cohort spans at least 6 months;
- the common cohort contains at least 1,000 distinct DFUs; and
- blend WAPE degradation is no greater than `0.0`, so blend WAPE is no worse
  than source champion WAPE.

A promotable forward draft must reference a matching completed passing
backtest. Matching means the same customer run/config checksum, Croston/SBA
parameters, blend weights, normalization policy, source promotion, and source
production lineage. Changed inputs require a new backtest and a new draft.
Failure is closed: missing, stale, mismatched, incomplete, or failed evidence
blocks staging/promotion rather than falling back to an untested release. The
source promotion must be a fresh unblended champion; a previously promoted
customer blend cannot recursively become the champion input to another blend.

## 9. Stored output and lineage

### Customer generation

- `customer_forecast_run` stores immutable dates, model/config/source lineage,
  including the exact latest completed and profile-refreshed `customer_demand`
  audit batch, durable job state, progress, counts, and terminal error details.
  `customer_demand_profile_refresh_state` is a singleton proof that
  `mv_customer_demand_series_profile` represents that batch. The loader creates
  a `running` audit row before fact mutation, refreshes every dependent MV,
  then completes the audit row and stamps the profile marker in one
  transaction. An active load, missing/mismatched marker, refresh failure, or
  newer completed load makes the run stale: generation, batch persistence,
  retry, backtest creation, blend readiness, and promotion all fail closed
  until a successful load and new customer forecast complete. After a hard
  process exit, the next canonical loader obtains the exclusive lineage lock,
  marks abandoned `running` batches failed, and clears the profile marker
  before opening its new batch. Backtest, forward-blend generation, and
  customer-blend promotion hold the matching shared lock through evidence
  publication.
- The queued manifest is committed before managed-job submission. Reads derive
  its job id from `job_history.params.run_id`; a later submission reconciles
  terminal jobs and retires a still-unlinked manifest after a five-minute
  grace period, so a process crash cannot leave the one-active-run guard
  permanently occupied.
- `customer_forecast_batch` and `customer_forecast_batch_series` provide
  resumable work ownership and committed recovery checkpoints.
- `fact_customer_forecast` stores run-versioned customer forecast rows and
  actual generating model lineage. Rows can change only while their parent run
  is generating; failed, cancelled, and completed output is frozen. A resumed
  run must re-enter `generating` before replacing a batch. Completed output
  receives a canonical checksum when bound into blend lineage.

### Historical evidence

- `customer_forecast_backtest_run` stores durable job state, customer run,
  source release, causal windows, model/config settings, a versioned checksum
  of the exact source-series membership, series/batch counts, row count, and
  canonical component checksum. Execution revalidates membership and counts,
  then evaluates every batch and normalization query under one repeatable-read
  database snapshot. Backtest and forward-blend generation hold a shared
  customer-demand session advisory lock for that snapshot; the loader takes
  the conflicting exclusive lock, so it cannot change facts between snapshot
  validation and evidence publication. Backtest blend evidence is stored at
  four-decimal precision to satisfy the component formula; release staging is
  the boundary that rounds publishable quantities to two decimals.
- `customer_bottom_up_backtest_component` accepts inserts only while its run is
  generating and is immutable afterward. It stores DFU-origin-month
  evidence for customer bottom-up, champion, blend, actual, normalization,
  weights, and coverage.
- `customer_bottom_up_backtest_accuracy` follows the same generation-only
  insert and terminal immutability rule. Its common-cohort evidence covers
  WAPE, MAE, bias, accuracy, threshold snapshots, WAPE degradation, and the
  pass/fail gate with reasons.

### Forward draft evidence

`customer_bottom_up_blend_component` accepts inserts only while its generation
run is generating, is immutable afterward, and is keyed to the exact
`forecast_generation_run`, completed customer run, passing backtest run, source
promotion, and source production run. `fact_production_forecast_staging`
contains the corresponding reviewable draft under model ID
`customer_bottom_up_blend`. A generated draft has no effect on
`fact_production_forecast` until the normal explicit promotion transaction
succeeds. Aggregation, normalization, checksum verification, and staging writes
share one repeatable-read snapshot.

## 10. API and durable jobs

| Method | Path | Purpose |
|---|---|---|
| GET | `/customer-forecast/readiness` | Resolve customer source windows, latest completed load-batch lineage, coverage, and blockers |
| POST | `/customer-forecast/generate` | Launch one durable Croston/SBA customer run |
| GET | `/customer-forecast/runs/latest` | Return latest progress or latest completed result |
| GET | `/customer-forecast/runs/{run_id}` | Return one customer run |
| POST | `/customer-forecast/runs/{run_id}/cancel` | Request cancellation |
| POST | `/customer-forecast/runs/{run_id}/retry` | Resume unfinished batches |
| GET | `/customer-forecast/series` | Return one customer history/forecast series |
| GET | `/customer-forecast/export` | Stream completed customer rows |
| POST | `/customer-forecast/backtest/generate` | Launch `generate_customer_forecast_backtest` from config |
| GET | `/customer-forecast/backtest/latest` | Return run, common-cohort metrics, thresholds, and gate |
| GET | `/customer-forecast/blend/readiness` | Return the current customer, source champion, and backtest gate |
| POST | `/customer-forecast/blend/generate` | Create a reviewable full-spine blend draft after evidence passes |
| GET | `/customer-forecast/blend/latest` | Return latest blend draft lineage and coverage summary |
| GET | `/customer-forecast/blend/series` | Compare bottom-up, champion, and blend for one item-location |

Write endpoints use the API-key guard. Customer generation and backtest jobs
use the durable job framework for progress, cancellation, restart recovery,
and terminal reconciliation. Blend output is deterministic and bound to the
exact customer/backtest/source-release/config identity recorded by its run.
Blend generation and both stage/promotion re-compute the customer-output,
historical-component, source-production, and forward-component checksums
instead of trusting stored checksum strings alone. Lightweight UI readiness
uses frozen manifests and defers the large payload scans to those write gates.
Customer-output, historical-backtest, and forward-blend evidence use the
versioned `xor256-v1` multiset checksum: each canonical row is SHA-256 hashed,
the 256-bit row hashes are combined with order-independent XOR, and the
resulting digest is sealed with exact row, series/DFU, and coverage counts.
This detects payload drift without sorting the full forecast fact or depending
on arbitrary chunk borders.

## 11. UI behavior

**Forecasting → Customer Forecast** remains the main customer run and result
view. It shows:

- the planning month, 18-month input/output windows, source readiness, and
  dormant-series exclusions;
- Croston/SBA-only forecastable coverage and durable batch progress;
- customer history versus forecast with run/model lineage and export;
- a **Run Blend Backtest** action with six-month common-cohort WAPE, MAE, bias,
  accuracy, thresholds, and pass/fail reasons;
- a **Generate Customer Blend Draft** action enabled only for matching passing
  evidence; and
- three explicit item-location series: **Customer Bottom-Up**, **Source
  Champion**, and **Customer Blend**, with fallback months visibly identified.

The same three read-only series are surfaced in Item Analysis and Demand
History, while the Forecast release view receives the resulting draft through
the existing staging workflow. The Forecast readiness and stage/promotion
cards identify that draft as **Customer Bottom-Up Blend**, even though it uses
the governed `champion` release slot, and show its common-cohort backtest gate,
blend WAPE, and WAPE delta versus the source champion before an operator acts.

Default overlays revalidate current customer-demand, customer-config,
backtest-config, and source-promotion lineage every 30 seconds and on window
focus; a lineage change clears the mounted overlay. Explicit `run_id` reads
remain available for historical review, while archived candidates are never
presented as current. An active promoted blend remains visible as the
authoritative production release until a later promotion replaces it.
A fallback row must not be drawn or labeled as though a customer signal
contributed. Customer-only item-locations remain excluded and are reported as
coverage diagnostics.

The UI does not directly promote. It hands the exact generated run to the
existing staging/release review, where normal readiness and explicit promotion
controls remain authoritative.

## 12. Implementation

- DDL: `sql/210_create_customer_forecast.sql` through
  `sql/215_enforce_customer_batch_lineage.sql`, plus
  `sql/216_create_customer_bottom_up_blend.sql`
- Croston/SBA: `common/ml/croston.py`
- Customer generation: `common/services/customer_forecast.py` and
  `common/services/customer_forecast_batches.py`
- Blend policy contract: `common/services/customer_forecast_blend_contract.py`
- Blend operational service: `common/services/customer_forecast_blend.py`
- Backtest service: `common/services/customer_forecast_backtest.py`
- Durable customer runner: `scripts/forecasting/generate_customer_forecasts.py`
- Durable blend/backtest runners: `scripts/forecasting/generate_customer_bottom_up_blend.py`
  and `scripts/forecasting/generate_customer_forecast_backtest.py`
- API: `api/routers/forecasting/customer_forecast.py` and
  `api/routers/forecasting/customer_forecast_blend.py`
- UI: **Forecasting → Customer Forecast** and shared forecast comparison
  overlays
- Config: `config/forecasting/forecast_pipeline_config.yaml` under
  `customer_forecast`

## 13. Acceptance criteria

- A July 2026 customer run reads January 2025–June 2026 and forecasts July
  2026–December 2027.
- Every active valid customer series uses `croston` and has exactly 18
  consecutive non-negative rows; dormant series have none.
- Customer generation has no Chronos route or fallback model.
- Raw customer forecasts target `demand_qty`; normalized blend components use
  the causal fulfillment ratio to align to sales semantics.
- A six-origin backtest uses only data knowable at each origin and persists
  immutable component and accuracy checksums; origin-specific activity does
  not inherit the current forward population.
- Promotion evidence covers at least six common months and 1,000 common DFUs,
  and blend WAPE is no worse than champion WAPE.
- The blend output exactly follows the active production DFU-month spine;
  customer-only rows are excluded and missing/unusable customer rows use
  champion fallback.
- The 18 customer months use 50/50 weights when qualified; production months
  outside that horizon pass through the champion.
- Confidence intervals use champion width shift, champion passthrough, or an
  explicit paired-null `none` state.
- Every draft component retains generation run, customer run, passing backtest,
  source promotion, and source production lineage.
- A missing, stale, mismatched, incomplete, or failing backtest blocks the
  draft from staging and promotion.
- A promoted customer blend cannot be used recursively as the source champion
  for another customer blend.
- No draft auto-promotes or changes active production outside the normal
  transactional promotion path.
