# 24 — Candidate Forecast & Model Promotion Workflow

**Status:** Implemented
**Last updated:** 2026-07-13
**Related:** 08-production-forecast.md, 07-champion-selection.md, 12-dual-promotion.md

---

## 1. Overview

This feature implements a **run-scoped staged promotion workflow** for forward
forecasts:

```
Train → Generate immutable draft → Promote exact run to staging → Promote staged run to production
```

Forward predictions land in an immutable, purpose-scoped
`fact_production_forecast_staging` run with `promotion_eligible=FALSE`. The
explicit staging action reviews and marks that exact run eligible without
changing its payload. Only one explicitly selected staged `release_candidate`
`source_run_id` can be copied to
`fact_production_forecast`. Historical backtest predictions use the separate
`fact_external_forecast_monthly`/`backtest_lag_archive` flow; the legacy
`fact_candidate_forecast` table remains inert and is not a release source.

Customer blend generation additionally creates a paired, immutable
`customer_bottom_up` `shadow_candidate` containing only normalized customer
rows. It is visible for comparison but is never stage-approved or promotable.
The full-spine `customer_bottom_up_blend` remains the sole customer-derived
`release_candidate` and uses the governed `champion` release slot.

## 2. Motivation

Previously, the production forecast was generated directly via a job that wrote
to `fact_production_forecast`. There was no staging area to compare models
side-by-side before committing one to production. This workflow adds:

- **Immutable candidate staging** — generations coexist by run and purpose
- **Coherent champion generation** — one candidate run preserves every routed
  source model without combining separately generated rows
- **Two explicit promotions** — the request must name the exact source run when
  moving a generated draft to staging and again when publishing it
- **Transactional release control** — validate, archive the outgoing release,
  publish, and audit as one all-or-nothing operation
- **Exact audit trail** — source/production run ids, gate report, and canonical
  payload checksums make the released values reproducible

## 3. Database Schema

### 3.1 `fact_candidate_forecast` (NEW)

Staging table for all model predictions. Populated by the "Load" action.

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | Auto-increment |
| `item_id` | VARCHAR(50) | DFU item |
| `loc` | VARCHAR(50) | DFU location |
| `model_id` | VARCHAR(100) | Algorithm (e.g. `lgbm_cluster`) |
| `forecast_month` | DATE | First day of forecast month |
| `forecast_qty` | NUMERIC(12,2) | Point forecast |
| `forecast_qty_lower` | NUMERIC(12,2) | P10 confidence interval |
| `forecast_qty_upper` | NUMERIC(12,2) | P90 confidence interval |
| `actual_qty` | NUMERIC(12,2) | Actual demand (backtest only) |
| `accuracy_pct` | NUMERIC(8,4) | Per-row accuracy |
| `wape` | NUMERIC(8,4) | WAPE |
| `bias` | NUMERIC(8,4) | Bias |
| `horizon_months` | SMALLINT | Horizon (T+1, T+2, ...) |
| `cluster_id` | TEXT | ML cluster label |
| `backtest_run_id` | INTEGER FK | Links to `backtest_run.id` |
| `loaded_at` | TIMESTAMPTZ | When loaded |
| `is_promoted` | BOOLEAN | TRUE after promotion |
| `promoted_at` | TIMESTAMPTZ | When promoted |

**Grain:** `(item_id, loc, model_id, forecast_month)` — UNIQUE index.

### 3.2 `model_promotion_log` (NEW)

Immutable audit trail for promotions and demotions.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment |
| `model_id` | VARCHAR(100) | Promoted model (or `'champion'`) |
| `promotion_type` | VARCHAR(20) | `'single'` or `'champion'` |
| `champion_experiment_id` | INTEGER FK | Links to champion experiment (if applicable) |
| `plan_version` | VARCHAR(30) | Version tag (e.g. `v20260408_143022`) |
| `promoted_at` | TIMESTAMPTZ | When promoted |
| `demoted_at` | TIMESTAMPTZ | When replaced by next promotion |
| `is_active` | BOOLEAN | Only one active at a time |
| `dfu_count` | INTEGER | DFUs promoted |
| `total_rows` | INTEGER | Rows copied to production |
| `promoted_by` | TEXT | Who/what triggered it |
| `notes` | TEXT | Free-text reason |
| `config_snapshot` | JSONB | Frozen config at promotion time |

### 3.3 `backtest_run` (ALTERED)

Added columns:
- `is_loaded_to_candidate` BOOLEAN — tracks candidate table loading
- `candidate_loaded_at` TIMESTAMPTZ — when loading completed

### 3.4 Forward release lineage (`sql/203`)

`forecast_generation_run` is the manifest for one immutable forward-generation
payload. Its `generation_purpose` is one of:

- `release_candidate` — generation finishes `ready` and non-eligible; `POST
  /backtest-management/{model_id}/stage` verifies the immutable manifest and
  payload before setting `promotion_eligible=TRUE`;
- `snapshot_contender` — one of the three bounded live-FVA alternatives, never
  promotable;
- `shadow_candidate` — immutable review-only evidence, currently the normalized
  `customer_bottom_up` paired to one exact customer blend run; it always remains
  non-eligible and cannot transition to `promoted`;
- `legacy_invalid` — pre-migration staging retained for inspection/cleanup but
  never promotable.

The staging uniqueness key is
`(run_id, generation_purpose, candidate_model_id, item_id, loc,
forecast_month)`. For champion generation, `candidate_model_id='champion'`
while `model_id` remains the exact routed producing model. This allows one
coherent champion run to carry multiple source models without colliding with
single-model or snapshot-contender runs.

The customer pair deliberately has different identities. The bottom-up shadow
uses its own deterministic run id, `candidate_model_id='customer_bottom_up'`,
and `generation_purpose='shadow_candidate'`; its metadata points back to the
exact blend run. The blend uses `candidate_model_id='champion'`,
`model_id='customer_bottom_up_blend'`, and
`generation_purpose='release_candidate'`. `/forecast/production/staging`
classifies manifests before grouping so both are returned under their true
display identities instead of collapsing the blend under `champion` or a
routed source model.

A ready champion manifest freezes the sales batch, champion and cluster
experiment ids, winners-artifact checksum, exact experiment-stamped historical
champion-results checksum, row/DFU/source-model counts, horizon, and canonical
forward payload checksum. Metadata must include
`generator_contract_version=canonical-five-artifact-lineage-v2`; older or missing
contracts are non-promotable even if the payload predates this guard with a
`ready` status.

Migration 203 also extends `model_promotion_log` with `source_run_id`,
`production_run_id`, `gate_report`, `candidate_checksum`,
`production_checksum`, `archive_checksum`, `archived_at`, and replacement
lineage. A partial unique index enforces exactly zero or one active promotion;
another unique index prevents a source run from being promoted twice.

### 3.5 Canonical model and generator cutover (`sql/205` and `sql/206`)

Migration 205 sets the champion experiment default to exactly LightGBM,
N-HiTS, N-BEATS, MSTL, and Chronos 2E. Its constraint permits a non-empty subset
of those five for a deliberate experiment, rejects duplicates and retired ids
on new or updated rows, and remains `NOT VALID` so historical experiments stay
auditable.

Migration 206 invalidates every still-`ready` `release_candidate` whose manifest
does not carry
`generator_contract_version=canonical-five-real-adapters-v1`. It sets
`promotion_eligible=FALSE` without deleting the immutable manifest or staging
payload. It does not rewrite source lineage and does not convert old forecasts
into canonical output.

Migration 209 advances the required version to
`canonical-five-artifact-lineage-v2` and applies the same fail-closed retirement
to both ready release candidates and ready snapshot contenders. New v2 output
must prove the source-model roster, current artifact lineage, and reconciled
snapshot payload rather than inheriting trust from the v1 status.

## 4. API Endpoints

All under `/backtest-management/` prefix.

### 4.1 Read Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/promotion-status` | Returns currently active promoted model |
| GET | `/candidate-summary` | Per-model candidate forecast **counts/accuracy** (no time-series) |

**Forecast comparison reads** (NOT under the `/backtest-management/` prefix):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/candidate?item_id=&loc=&model_id=` | Per-model backtest (past, out-of-sample) predictions for one DFU, grouped by `model_id` — forecast vs `actual_qty` + per-row accuracy. `model_id` optional. Degrades to empty `models` when the table is absent (clean install). |
| GET | `/forecast/production/staging?item_id=&loc=` | Latest future staging rows grouped by true display identity, including separate `customer_bottom_up` shadow and `customer_bottom_up_blend` release-candidate runs with their exact source run ids. |
| GET | `/customer-forecast/blend/trend?run_id=...` | Exact-lineage historical backtest plus future staged bottom-up, source-champion, and blend comparison used by Portfolio and Item Analysis. |

`/forecast/candidate` is the past-period counterpart to
`/forecast/production/staging`; both are implemented in
`api/routers/forecasting/production_forecast.py`. The customer trend endpoint
is implemented in `api/routers/forecasting/customer_forecast_blend.py`.

> ⚠️ **Currently inert.** `fact_candidate_forecast` is **not populated by any code** (the
> backtest load writes `fact_external_forecast_monthly`, not this table), so
> `/forecast/candidate` always returns empty `models` and the `backtest_<model>` chart
> overlay never renders. The working past-backtest line is `forecast_<model>` (from
> `agg_forecast_monthly`), which auto-load now populates. The dead `/forecast/candidate`
> endpoint + `backtest_<model>` overlay are slated for removal — track this before relying
> on them.

### 4.2 Write Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/{model_id}/generate` | Submit a new `release_candidate`; returns its allocated `source_run_id` immediately |
| POST | `/{model_id}/stage?source_run_id=<uuid>` | Approve exactly one `release_candidate`; a `shadow_candidate` is rejected and never becomes eligible |
| POST | `/{model_id}/promote?source_run_id=<uuid>` | Promote exactly one ready, eligible release-candidate run |

### 4.3 Staging approval and transactional production promotion

`stage_forecast_run()` locks one exact current-month generated run, validates
its model, generator contract, horizon, counts, and checksum, then changes only
`promotion_eligible` from false to true. It is safe to retry and returns
`already_staged` when the run was previously approved. Multiple models may have
staged runs at the same time. This boundary accepts only
`generation_purpose='release_candidate'`; the database also prevents a
`shadow_candidate` from acquiring promoted status.

`promote_model()` does not read `fact_candidate_forecast` and has no bypass
token. It passes the explicit `source_run_id` to
`common/services/forecast_promotion.py`, which performs the following on the
primary database in one `SERIALIZABLE` transaction under a transaction-scoped
advisory lock:

1. Lock the `forecast_generation_run` manifest and require purpose
   `release_candidate`, status `ready`, explicit staging eligibility, requested model
   equality, current planning month, sufficient horizon, and a non-empty
   checksummed payload produced by the current canonical-five generator
   contract.
2. Recompute the staging checksum and cardinalities and require an exact match
   to the immutable manifest. Reject a newer completed sales batch than the one
   stamped at generation.
3. For champion, require the sole results-promoted champion experiment, the
   matching sole promoted cluster experiment, current assignments, no stale
   tuning profiles for cluster labels present in the current assignment
   generation, an exact SHA-256 match for
   `data/champion/experiment_<id>_winners.csv`, and equality with the checksum
   and row count of the exact experiment-stamped historical champion results.
   Re-evaluate WAPE, baseline lift, incumbent delta, bias, common-cohort
   sufficiency, and actual alignment from those stamped historical rows. For a
   single model, require one source model in the payload.
   Seasonal-naive baseline forecasts are derived from the matching DFU's sales
   twelve months earlier, so the release gate does not depend on retaining a
   deprecated algorithm series in the forecast fact. Sales facts are sparse:
   an absent prior-year month represents zero demand, so the baseline
   densifies that gap to zero while preserving a materialized zero sale as
   zero.
4. Evaluate the fixed six-month eligible-DFU forward window. Generation builds
   from full `(item_id, customer_group, loc)` sales grain; the gate independently
   aggregates that source universe to production `(item_id, loc)` grain using
   the same configured minimum history and 12-month active window. Route gaps
   are hard failures; coverage must meet policy; quantities and interval
   ordering must be valid; confidence-interval coverage must meet policy. The
   gate report stores the historical quality checks, but does not stamp a
   "candidate WAPE" on the future release rows because they do not yet have
   actuals.
5. Lock the currently active promotion, if one exists, without restricting its
   model or planning month. Record its exact source and production lineage in
   the incoming gate report as the release being replaced.
6. Demote the outgoing audit row, insert the incoming audit record, replace
   `fact_production_forecast`, and stamp every row with the source run, new
   production run, promotion id, and `lineage_status='verified'`.
7. Recompute the published payload checksum and require it to equal the
   candidate checksum. Mark the generation manifest `promoted` and commit.

Any failure rolls back demotion, delete, insert, and manifest changes.
The database unique indexes also prevent concurrent active releases and reuse of
one candidate run. `model_id='champion'` and single algorithm ids share this
contract; the difference is only their lineage validation and
`promotion_type` audit value. Snapshot preparation and archival are deliberately
outside this transaction in the separate Period Roll workflow, so snapshot state
never blocks an otherwise valid production replacement.

## 5. Frontend UI

### 5.1 Backtest stage table (Forecasting → Backtest)

Models are grouped by family (Tree / Foundation / Statistical / Deep Learning) in compact
tables. All five start checked; **Run selected (N)** runs any chosen subset, each row can run
individually, and **Select all** restores the five-model cycle. The forward release flow is:

```
[Train when required] → [Generate immutable run] → [Promote that exact run]
```

| Step | Action | API Call | Condition |
|------|--------|----------|-----------|
| Run | Run backtest (per model or per group) | `POST /{id}/run` | Trained (tree) or always (foundation/statistical/DL) |
| Auto-load | **Automatic** on run completion — server-side, same job | (none — chained inside the backtest job) | Backtest produced predictions |
| Promote | Publish one exact ready run | `POST /{id}/promote?source_run_id=<uuid>` | The selected run is ready and promotion-eligible |

> **Auto-load.** When a backtest run completes, `_run_backtest`
> (`common/services/job_state.py`) chains the load (`_auto_load_backtest` →
> `scripts/etl/load_backtest_forecasts.py`) **before** marking the run `completed`, so the UI's
> "Loaded" status and `is_loaded_to_db` flag are consistent. The load is best-effort: a
> failure leaves the run `completed` and logs for manual retry. There is **no Load button**
> in the grid; a recovery **Load to DB** appears on the detail panel only for a
> completed-but-unloaded run. `POST /{id}/load` remains for that recovery path.

> **Important — actual load target.** Despite this spec's original framing, the load writes
> to **`fact_external_forecast_monthly`** (+ `backtest_lag_archive`) and refreshes the
> `agg_forecast_monthly` MV — **not** `fact_candidate_forecast`. Nothing in the codebase
> populates `fact_candidate_forecast`; the per-model historical backtest surfaces in Item
> Analysis as the `forecast_<model>` line (from `agg_forecast_monthly`). See §4.1 note.

**State indicators:**
- Gray badge: Not available (prerequisite incomplete)
- Outline button: Ready to execute
- Spinner: Running
- Green check badge: Completed
- Crown badge: Currently promoted to production

### 5.2 Select, stage, and promote one candidate

The Forecast panel lets the planner select `lgbm_cluster`, `chronos2_enriched`,
`mstl`, `nbeats`, `nhits`, or `champion`. **Generate Forecast** creates one
immutable staging run for that selection and returns its `source_run_id`.
Champion generation routes the assigned experiment's per-month winners inside
one coherent run while preserving each row's producing `model_id`; a single
model run contains only that source model. The UI polls for the exact run, not
merely any older staged rows. **Promote _selection_ to Production** is enabled
only when that run is ready and promotion-eligible and the snapshot roster is
current, and sends that exact id to the promotion endpoint.

A generated customer blend appears in this workflow as
**Customer Bottom-Up Blend** while retaining `candidate_model_id='champion'`
for release ownership. Its paired **Customer Bottom-Up** shadow is displayed for
review only: it has a distinct exact run id, never enables Stage or Promote,
and cannot replace production by itself.

Generation preserves the model-training DFU grain
`(item_id, customer_group, loc)` through sales loading, current-cluster lookup,
per-month champion routing, and recursive inference. Training and serving both
use the `production_forecast.lookback_months` closed-month profile window
(currently 36 months) and a calendar-complete
monthly spine; absent sparse sales months are zero demand. Every expected
customer group must produce the complete requested horizon. Only then are the
group point forecasts summed to the production `(item_id, loc, month)` grain.
Mixed source algorithms are labeled `ensemble`, a common source retains its
algorithm lineage, and confidence intervals are calculated once on the
item/location aggregate rather than repeated for each customer group.

The generator does not use the winners artifact itself as the population. It
resolves the same original/current sales table used for training and derives
the full active, non-null type-1-sales population using
`cold_start_min_months` and `forecast_snapshot.active_window_months`. An
item/location is eligible only when every active customer group meets the
history floor. It drops
stale routes outside that population, and adds an explicit LightGBM route for
every uncovered eligible DFU. For each production month, routing resolves the
latest dated single-model or ensemble decision available by the planning
cutoff; future-only evidence is never used. A route that resolves to MSTL requires its
configured 25 months of history. Any shorter MSTL route fails generation with
instructions to rerun the MSTL backtest and champion selection.

Every publish must run and load all five backtests, then complete a governed
champion refresh before production generation. This is especially important after an
adapter, population, data, or clustering change, because an older champion
artifact can be structurally valid but no longer serve the current eligible
population. The named `model-refresh` pipeline loads the five governed runs;
the separate `champion-refresh` pipeline atomically promotes the new
experiment/results only after exact five-run
sales/cluster lineage and winner checksums pass; the
subsequent `forecast-publish` pipeline final-refits the persisted LightGBM,
N-HiTS, and N-BEATS families and generates the release candidate. MSTL and
Chronos 2E remain direct adapters. Period Roll separately prepares and archives
the top-three snapshot contenders.

### 5.3 Candidates Column

New table column showing loaded candidate DFU count per model.

### 5.4 Item Analysis backtest overlay

The **Item Analysis** chart (`UnifiedChartPanel` / `UnifiedChart`) uses the exact
blend manifest to render historical `backtest_customer_bottom_up`, source
champion, and `backtest_customer_bottom_up_blend` series over existing actual
sales months. Those customer lines come from
`/customer-forecast/blend/trend`, not the inert
`fact_candidate_forecast` table.

Future customer lines use the standard `/forecast/production/staging` response:
`staging_customer_bottom_up` comes from the paired shadow run and
`staging_customer_bottom_up_blend` from the release-candidate run. Historical
rows are merged only onto existing history, future staging months are appended
once, and the component overlay is suppressed for an identity already present
in standard staging. The result is one toggle/line per model and month rather
than duplicate bottom-up or blend series.

### 5.5 Portfolio customer comparison

Portfolio Analysis has a dedicated **Customer Blend** mode backed by one
explicit blend `run_id`. It shows the manifest-matched historical common-cohort
actual/bottom-up/source-champion/blend series and WAPE, followed by the paired
future staged series, blend interval, and blended-versus-fallback coverage.
Brand, category, market, item, location, and cluster filters are applied to the
exact component lineage; channel is explicitly reported as not applicable at
warehouse-item grain. This evidence is not inserted into the standard accuracy
materialized views.

## 6. Data Flow

> This diagram reflects the verified `promote_model()` mechanism (§4.3), not the original
> `fact_candidate_forecast` design described in §2-3 - that table is inert (§4.1).

```
┌─────────────┐   Final fit   ┌────────────────────────────┐
│ Closed Sales │────────────► │ Immutable active artifacts │
│ History      │              │ LightGBM + N-HiTS/N-BEATS │
└──────┬──────┘              └─────────────┬──────────────┘
       │ MSTL/Chronos direct               │ validated artifact ids
       └──────────────────────┬─────────────┘
                              │ Generate (production forecast)
                              ▼
                             ┌───────────────────────┐
                             │ forecast_generation_run │ ◄── Immutable run manifest
                             │ + production staging    │     and exact payload hash
                             └────────┬──────────────┘
                                      │ Promote
                                      ▼
                             ┌─────────────────────┐
                             │ fact_production_     │ ◄── Only the promoted model
                             │ forecast             │     (or per-DFU champion mix)
                             └─────────────────────┘
```

The separate historical **backtest** flow (Train (backtest) → Generate (backtest) → Load) writes to
`fact_external_forecast_monthly` + `backtest_lag_archive` (§5.1) for accuracy analysis and the Item
Analysis backtest overlay (§5.4) - it does not feed this staging→promotion pipeline.

## 7. Migration

**DDL files:**

- `sql/121_candidate_forecast_and_promotion.sql` — original candidate/audit
  tables;
- `sql/122_create_production_forecast_staging.sql` — forward staging; and
- `sql/203_create_forecast_generation_run.sql` — run/purpose manifests,
  immutable staging key, release/archive checksums, and production lineage;
- `sql/205_enforce_champion_model_roster.sql` — canonical five-model default and
  no-retired-id/no-duplicate constraint for new champion experiments; and
- `sql/206_invalidate_pre_canonical_generator_runs.sql` — one-way invalidation
  of pre-contract ready release candidates while retaining their evidence; and
- `sql/207_add_forecast_sales_grain_indexes.sql` — full-grain partial indexes
  for the canonical non-null type-1 sales population/history reads.

Apply migrations in numeric order with `ON_ERROR_STOP=1`. After migration 203,
all pre-manifest staging is `legacy_invalid`, and pre-migration champion rows
lack experiment/checksum evidence. Explicitly promote the selected champion
experiment's results again, then generate a fresh release candidate before
calling promote. There is no supported path that blesses an old mixed staging
population as promotable.

Migration 203 is transactional and repairs duplicate historical active audit
rows deterministically before installing the unique active-promotion index.
Apply migrations 205 through 207 in numeric order after the schema has reached 204.
Then rerun the complete five-model backtest roster and champion selection before
creating the first promotable canonical release candidate.
