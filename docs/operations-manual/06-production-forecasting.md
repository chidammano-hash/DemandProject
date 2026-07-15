# 06 — Production Forecasting

This section covers **generating, promoting, and serving production forecasts**. Production forecasts are the forward-looking demand signals consumed by inventory planning, S&OP, and replenishment. They are produced from the canonical five-model competition—LightGBM, N-HiTS, N-BEATS, MSTL, and Chronos 2E—followed by champion selection (covered in Sections 04 and 05).

Repo root for all paths and commands: `/Users/manoharchidambaram/projects/DemandProject`.

---

## 1. Architecture Overview

The production forecast pipeline follows an **immutable run → transactional
promote** pattern. Forward predictions are grouped by an explicit generation
manifest; only one ready, eligible run can reach the consumer-facing production
table.

```
                ┌──────────────────────────────────┐
                │  Canonical champion routing      │
                │  Five adapters + sales history   │
                └─────────────┬────────────────────┘
                              │ make forecast-generate
                              ▼
        ┌─────────────────────────────────────────────────┐
        │  forecast_generation_run +                     │
        │  fact_production_forecast_staging               │
        │  Immutable run/purpose-scoped payload           │
        └─────────────┬───────────────────────────────────┘
                      │ POST /backtest-management/{model_id}/promote?source_run_id=...
                      │ (single mode OR champion mode)
                      ▼
        ┌─────────────────────────────────────────────────┐
        │  fact_production_forecast                       │
        │  (sql/039_create_production_forecast.sql)       │
        │  Single source of truth — what consumers read   │
        └─────────────┬───────────────────────────────────┘
                      │ append-only audit
                      ▼
        ┌─────────────────────────────────────────────────┐
        │  model_promotion_log                            │
        │  Exact source/release/archive lineage           │
        │  gate report + SHA-256 payload checksums        │
        └─────────────────────────────────────────────────┘
```

Historical backtest load writes to `fact_external_forecast_monthly` and
`backtest_lag_archive`; it does not feed forward promotion. The legacy
`fact_candidate_forecast` table has no active writer and must not be treated as
a release source.

| Table | Grain | Lifecycle | DDL |
|---|---|---|---|
| `forecast_generation_run` | one row per generation UUID | Manifest status, purpose, generator-contract metadata, input lineage, counts, checksums | `sql/203_create_forecast_generation_run.sql`; canonical cutover in `sql/206`; customer shadow purpose in `sql/217` |
| `fact_production_forecast_staging` | run + purpose + candidate + item + loc + month | Immutable run-scoped rows; `release_candidate`, `snapshot_contender`, and review-only `shadow_candidate` evidence never mix with retained `legacy_invalid` rows | `sql/122`, extended by `sql/203` and `sql/217` |
| `fact_production_forecast` | plan_version + item + loc + forecast_month | Replaced only inside verified promotion; every new row carries source run, production run, and audit id | `sql/039`, extended by `sql/203` |
| `model_promotion_log` | one row per promote/demote event | Exact source/release/archive lineage; one active row enforced by unique index | `sql/121`, extended by `sql/203` |
| `customer_forecast_backtest_run` + bottom-up evidence | run manifest plus DFU-origin-month components and one common-cohort scorecard | Six causal one-step origins; immutable customer/champion/blend components, checksum, metrics, and promotion gate | `sql/216`; rule-router contract extended by `sql/218` |
| `customer_bottom_up_blend_component` | generation + item + loc + month | Immutable forward components bound to customer, backtest, promotion, and production lineage | `sql/216` |

`model_promotion_log.promotion_type` is constrained to `('single', 'champion')`;
the database enforces at most one `is_active=TRUE` row and at most one promotion
per non-null `source_run_id`.

Customer bottom-up blend generation publishes two separately manifested views
of the same exact component evidence. The normalized, unblended
`customer_bottom_up` signal uses a deterministic companion UUID and
`generation_purpose='shadow_candidate'`; it is review-only and cannot be
staged for release or promoted. The full-spine `customer_bottom_up_blend`
retains the submitted blend UUID and normal `release_candidate` lifecycle.
Keeping separate manifest UUIDs preserves the existing one-purpose-per-run
primary key and staging foreign key.

---

## 2. Generate Forecasts (`make forecast-generate`)

### One-time canonical release cutover (migrations 203, 205, and 206)

Use the normal numeric migration runner for a new database. For an environment
where migration 203 is already present, apply the two canonical cutover
migrations before using the Generate or Promote UI:

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp \
  -v ON_ERROR_STOP=1 < sql/205_enforce_champion_model_roster.sql
docker compose exec -T postgres psql -U demand -d demand_mvp \
  -v ON_ERROR_STOP=1 < sql/206_invalidate_pre_canonical_generator_runs.sql
```

Migration 203 intentionally classifies all pre-manifest staging as
`legacy_invalid`. It does not infer that an old or mixed population was a safe
release candidate. It also leaves pre-migration champion result rows without an
experiment id/checksum. After applying it, run **model-refresh**, then the named
**champion-refresh** pipeline so the exact current five-run lineage is evaluated and the new champion
experiment/results are promoted atomically. The retired manual
`/champion-experiments/{id}/promote` and `/promote-results` routes return 410 and
cannot be used for cutover. Then run a fresh full Generate action and promote
only the returned `source_run_id`. Existing production
remains readable until the new candidate passes every gate. Promotion locks and
audits the active row, then replaces it atomically; a monthly snapshot is not a
precondition. Run Period Roll separately to create the Champion-plus-three FVA
archive.

Migration 205 makes the five canonical model ids the default champion roster
and rejects new or updated rosters with retired or duplicate ids while leaving
historical rows auditable. Migration 206 marks every still-`ready`
`release_candidate` without
`generator_contract_version=canonical-five-real-adapters-v1` as `invalid` and
non-promotable. Its staging rows remain immutable evidence; do not relabel them
or manually restore eligibility. Snapshot contenders are not promoted and are
separately rejected or rebuilt by contender preparation when their contract is
old.
Migration 207 adds partial full-DFU-grain indexes to the current sales fact and
its immutable-original mirror. Production training and generation require the
mirror; the current-table index supports adjacent analytics and recovery checks.
Migration 209 then invalidates any still-ready release candidate or snapshot
contender that predates
`generator_contract_version=canonical-five-artifact-lineage-v2`. The v2
contract binds the exact source-model roster, immutable tree/neural artifacts,
current source lineage, and deeply reconciled snapshot payload.
Migration 217 adds the review-only `shadow_candidate` manifest purpose. Ready
shadow manifests must carry a checksummed, non-empty payload, and database
guards make both the ready manifest and its staging rows immutable. It does not
relax the promotion invariant: only `release_candidate` manifests may
transition to `promoted`.

The API image includes `procps` in addition to the five model runtimes. Durable
JobManager attempts capture `ps` start/command markers before releasing a child
process; removing `procps` makes every managed subprocess fail closed before it
can execute.
Within Docker Compose, the API must use `MLFLOW_TRACKING_URI=http://mlflow:5000`;
the host-facing `localhost:5003` address is valid only for host processes and
causes long tracking retries from inside the API container.

### Required model refresh before publish

Run **model-refresh**, then **champion-refresh**, before every
**forecast-publish**. Model refresh runs and loads all five backtests in
order—LightGBM, N-HiTS, N-BEATS, MSTL, and Chronos 2E. Champion refresh then
creates and atomically promotes a governed champion experiment after proving
all five runs share current sales and cluster lineage. Do not generate from an older
promoted experiment: it can contain routes produced before the canonical
adapters or population rules changed, including MSTL routes for DFUs with too
little history. Only after the five backtests and governed champion refresh complete
should `forecast-publish` final-refit the three persisted model families
(LightGBM, N-HiTS, and N-BEATS) and generate the immutable champion run.
Promotion remains a separate reviewed action. Top-three snapshot preparation,
archival, and cleanup belong only to the separate **Period Roll** workflow.

### What it does

`make forecast-generate` runs
`scripts/forecasting/generate_production_forecasts.py`, which loads the promoted
champion experiment's per-month winners artifact and dispatches each winning
canonical model over the next `horizon_months` months. LightGBM uses a
persisted promoted-cluster artifact bundle; N-HiTS and N-BEATS use immutable
global final-refit artifacts. `common/ml/production_non_tree.py` dispatches the
two fit-at-inference families, MSTL and Chronos 2E, through the same adapters
used by backtesting.

Generation and final-fit training require
`fact_sales_monthly_original` as their canonical source; they never fall back
to `fact_sales_monthly`. The mirror must be non-empty and synchronized to the
latest positive completed sales batch: the batch must not be the legacy
`safe_upsert` path, its `row_count_out` must equal the mirror row count, and the
mirror's maximum `load_ts` must be at or after that batch's `started_at`.
`completed_at` is deliberately not used for the timestamp comparison because
rows are written before the audit batch is marked complete.

If any check fails, run one canonical dual-track reload and then restart the
model lifecycle:

```bash
make normalize-sales
make load-sales
```

The canonical loader replaces `fact_sales_monthly` and
`fact_sales_monthly_original` in one transaction and records one content hash
for the synchronized batch. Generation then derives the complete eligible population at
`(item_id, customer_group, loc)` grain from non-null type-1 sales. An
item/location is eligible only when every active customer group meets the
model's history floor through the latest closed month (the month immediately
before the planning month); partial planning-month actuals are excluded. An
inactive old group does not disqualify it. Routes are then
aligned to that population: stale routes outside it are discarded, the latest
dated single-model or ensemble route on or before the planning cutoff is
carried forward, future-only evidence is discarded, and every otherwise
uncovered DFU receives an explicit LightGBM route.
This population-first alignment prevents a historically narrower champion
artifact from silently omitting eligible DFUs.

Inputs:

- **Persisted production artifacts** — immutable, lineage-validated LightGBM
  promoted-cluster bundles plus global N-HiTS and N-BEATS final fits, produced
  atomically by `make train-production-all`.
- **Champion routing** —
  `data/champion/experiment_<promoted_id>_winners.csv`, whose exact bytes are
  SHA-256 stamped on the generation manifest. `dfu_assignments.csv` is not a
  production routing source. The artifact must come from the current five-model
  backtest and champion cycle.
- **Sales history** — last `lookback_months` (default 36) from the validated
  `fact_sales_monthly_original` mirror, used to seed lag features for T+1.
- **Pipeline config** — `config/forecasting/forecast_pipeline_config.yaml` `production_forecast:` and `forecast_snapshot:` blocks.

#### Persisted artifact layout and recovery boundary

LightGBM is activated as one complete cluster set, never as independent loose
pickles:

```text
data/models/lgbm_cluster/production_tree/
├── active.json
└── versions/<artifact_set_id>/
    ├── metadata.json
    ├── training_metadata.json
    ├── checksums.json
    └── models/*.pkl
```

`active.json` stores the active artifact-set id and the checksum of the version
manifest. `metadata.json` maps promoted cluster labels to opaque model files and
binds the set to the sales batch/hash, latest closed history month, promoted
cluster experiment, model configuration, and generator contract. The loader
validates that exact cluster roster and every checksum before unpickling. When
clustering is explicitly disabled, the only valid roster is `global`.

Final-fit training writes a temporary version, validates it, renames it into
`versions/`, and only then atomically replaces `active.json`. A failed or
partial refit leaves the prior active version unchanged. Do not repair a failed
run by creating `cluster_<id>.pkl`, copying a backtest model into `data/models`,
or editing `active.json`; rerun `make train-production MODEL=lgbm_cluster`.

The two neural families use the corresponding immutable roots
`data/models/nhits/neuralforecast/` and
`data/models/nbeats/neuralforecast/`, each with an active pointer and
checksummed version. The generation manifest records the exact LightGBM
`artifact_set_id` and neural `artifact_id` values it used.

Output:

- One `forecast_generation_run` manifest plus immutable rows in
  **`fact_production_forecast_staging`**, one row per
  `(run_id, candidate_model_id, item_id, loc, forecast_month)`. A normal
  champion generation is one coherent `release_candidate` run; each row's
  `model_id` preserves the true routed source model.
- A completed manifest stamped with the current planning month, latest completed
  sales batch, champion/cluster experiment lineage, row/DFU/source-model counts,
  routing artifact checksum, exact historical champion-results checksum, and
  canonical staging payload checksum. Its metadata also carries
  `generator_contract_version=canonical-five-artifact-lineage-v2`; promotion
  rejects older runs so output produced without the current artifact, roster,
  source, and snapshot guarantees cannot be relabeled or published.

The Forecast UI's Generate endpoint allocates the UUID before the job starts and
returns it as `source_run_id`. The UI then polls for that exact run; it never
mistakes an older completed generation for the newly submitted job.

### Command variants

| Target | Purpose | Source |
|---|---|---|
| `make forecast-generate` | Full DFU population | `Makefile` production-forecast targets |
| `make forecast-generate-dfu ITEM=<id> LOC=<loc>` | One DFU only — debugging | `Makefile` production-forecast targets |
| `make forecast-generate-dry` | Inference + log row count, no DB writes | `Makefile` production-forecast targets |
| `make forecast-prod-all` | `forecast-prod-schema` + `forecast-generate` | `Makefile` production-forecast targets |
| `make forecast-full` | Final-refit LightGBM, N-BEATS, and N-HiTS + generation; does not rerun backtests or champion selection | `Makefile` production-forecast targets |
| `make forecast-model MODEL=<id>` | Generate for a single model id | `Makefile` production-forecast targets |

### LightGBM recursive inference (how it predicts beyond T+1)

For a LightGBM-routed DFU the script loops `h = 1 .. horizon_months`. At step `h+1`, lag and rolling features are reconstructed from predictions emitted at earlier steps. Non-tree routes call `common/ml/mstl.py`, `common/ml/neural_forecast.py`, or `common/ml/chronos2_enriched.py` without loading tree artifacts. Each row records its source model and generation lineage.

### Non-tree runtime and failure contract

MSTL, N-HiTS, N-BEATS, and Chronos 2E must execute their real canonical
adapters. The dispatcher rejects a mismatched algorithm id, duplicate or
missing DFU-month output, non-finite or negative quantities, and non-contiguous
publish months. An absent optional dependency or incomplete adapter result
therefore fails the generation run; it is never replaced by a heuristic carrying
the requested model label.

The production image installs `libgomp1` and runs `uv sync` with the
`foundation`, `dl`, and `statistical` extras. For a local base environment, run:

```bash
uv sync --extra foundation --extra dl --extra statistical
```

---

## 3. Cold-Start Routing

The DFU population is heterogeneous — some items have years of history, some
are brand-new. The pipeline routes each full training DFU
`(item_id, customer_group, loc)` by calendar months since its first sale, using
the configured closed-month lookback window. Sparse missing months after the
first sale are real zero-demand history; pre-introduction padding is not.
Population activity comes from `forecast_snapshot.active_window_months`; the
history thresholds live under `production_forecast:` in
`config/forecasting/forecast_pipeline_config.yaml`.

| Path | Rule | `model_id` used |
|---|---|---|
| **Fail release generation** | `n_months < cold_start_min_months` (default **3**) | No partial item/location aggregate is staged; seed or exclude the whole cohort |
| **Cold start** | `cold_start_min_months <= n_months < min_history_months` | Configured `cold_start_model_id` |
| **Champion** | `n_months >= min_history_months` (default **12**) | Latest-as-of assigned per-month champion source; explicit LightGBM fallback if the DFU has no route |

Implementation: `scripts/forecasting/generate_production_forecasts.py`.

```yaml
# config/forecasting/forecast_pipeline_config.yaml
production_forecast:
  horizon_months: 24
  lookback_months: 36
  min_history_months: 12          # threshold for full champion model
  cold_start_model_id: lgbm_cluster
  cold_start_min_months: 3        # below this, the coherent run fails
  fallback_model_id: lgbm_cluster # used only for a missing/legacy route source
```

> **Operator note:** LightGBM artifacts are mandatory whenever a route needs
> LightGBM. Missing artifacts fail generation rather than substituting a
> differently labeled forecast. Run `make forecast-generate-dry` after changing
> a history threshold to surface affected customer groups before promotion.

MSTL has a stricter configured `min_history` of 25 months. Direct MSTL
generation builds only that eligible population. Champion generation validates
every resolved single or ensemble route before inference and fails when MSTL is
routed to a DFU with fewer than 25 months; the remediation is to rerun the MSTL
backtest and champion selection, not to replace or relabel the MSTL forecast.


---

## 4. Staging and Production Promotion Workflow

Generation creates an immutable draft. Staging approval and production
publication are separate operator actions. Multiple exact runs may be staged;
production publication remains auth-guarded, fail-closed, audited, and limited
to one active release.

### Endpoints

```
POST /backtest-management/{model_id}/stage
Headers: X-API-Key: <key>
Query  : ?source_run_id=<uuid>        # generated draft to approve

POST /backtest-management/{model_id}/promote
Headers: X-API-Key: <key>             # require_api_key dependency
Query  : ?source_run_id=<uuid>        # required; returned by Generate/staging-summary
         &notes=<text>                # optional, recorded on the audit row
         &promoted_by=<user|system>   # optional, default 'api'
```

Defined in `api/routers/forecasting/forecast_promotion.py`.

The staging action rechecks the exact manifest and payload checksum before
setting `promotion_eligible=TRUE`; it does not copy or rewrite forecast rows.
The production action rejects a generated-but-not-staged run with
`candidate_not_staged`.

### Single mode vs Champion mode

| Mode | When to use | What gets copied |
|---|---|---|
| **Single** — `model_id != "champion"` | A reviewed single algorithm is the release candidate | Only rows whose manifest and `candidate_model_id` match the requested model and `source_run_id` |
| **Champion** — `model_id == "champion"` | The promoted champion experiment routes different source models by DFU-month | Only rows in the one coherent champion run; `source_model_id` in production preserves each staged row's producing `model_id` |

Champion mode requires the sole results-promoted champion experiment, its
matching sole promoted cluster experiment, current assignments, fresh tuning,
and a winners CSV whose bytes still match the SHA-256 stamped at generation.

### Step-by-step flow (executed atomically inside one transaction)

1. Begin a `SERIALIZABLE` primary-database transaction and obtain the
   transaction-scoped `forecast_release_promotion` advisory lock.
2. Lock and validate the exact generation manifest: purpose
   `release_candidate`, status `ready`, explicitly staged/eligible, requested model/current month,
   at least six months, non-empty, and checksummed. Pre-migration
   `legacy_invalid` and `snapshot_contender` runs cannot pass.
3. Recompute the payload hash/cardinalities and require current sales lineage.
   For champion, also require current champion, cluster, assignment, tuning, and
   routing-artifact lineage; recompute the exact experiment-stamped historical
   champion checksum/row count and evaluate its common-cohort WAPE, baseline
   lift, incumbent delta, bias, sufficiency, and actual-alignment policy. A
   single-model candidate must contain exactly one source model.
4. Evaluate the fixed six-month forward structural gate: eligible-DFU coverage,
   no partial route gaps, nonnegative and ordered quantities, and configured CI
   coverage. The transaction stores the historical quality checks in the gate
   report; it does not stamp a "candidate WAPE" on future rows that have no
   actuals.
5. Lock the currently active promotion, if one exists. This lock has no model or
   planning-month restriction: Champion or any retained individual algorithm may
   replace production at any time using a newly staged immutable run.
6. Demote the outgoing audit row; insert the incoming audit row with source and
   production run ids, gate report, candidate/production checksums, replacement
   lineage, and an audit description of the replaced release; then replace production.
7. Hash production and require exact equality with the selected candidate,
   mark the source manifest `promoted`, and commit. Any error rolls back the
   demotion, delete, insert, audit, and manifest transition together.

Production replacement never creates, rewrites, or requires a forecast snapshot.
The database partial unique index still enforces exactly one active production
release. Run **Period Roll** independently when the monthly Champion-plus-three
snapshot is due.

Response payload:

```json
{
  "model_id": "lgbm_cluster",
  "promotion_type": "single",
  "plan_version": "2026-07",
  "source_run_id": "6bc73b5a-b51c-4d53-a780-dfe421774270",
  "production_run_id": "4d7805de-2186-447d-9529-dc19bd124da0",
  "candidate_checksum": "<64 lowercase hex characters>",
  "outgoing_archive_checksum": null,
  "rows_promoted": 1234567,
  "dfu_count": 51234
}
```

### Inspecting promotion state (no auth required, read-only)

| Endpoint | Returns |
|---|---|
| `GET /backtest-management/promotion-status` | The single active promotion plus source/production/checksum/archive lineage (or `{"promoted": null}`) |
| `GET /backtest-management/candidate-summary` | Per-model row/DFU/avg-accuracy in `fact_candidate_forecast` |
| `GET /backtest-management/staging-summary` | Latest immutable release-candidate manifest per requested model, including exact `source_run_id`, actual `candidate_model_id`, status, eligibility, row/DFU/source-model counts, horizon dates, and a sanitized customer-blend lineage/backtest gate when applicable |
| `POST /backtest-management/{model_id}/stage` | Validate and approve one exact generated draft for staging; safe to retry |

There is no bypass token. Fix the failed evidence, generate a new candidate when
inputs changed, and retry with its new run id.

---

## 5. Other Production Scripts

These scripts produce **additional** forecast layers that sit alongside (or downstream of) the champion point forecast. They are run on demand or as part of `make setup-demand-planning`.

| Script | Make target | Purpose | Output table |
|---|---|---|---|
| `scripts/forecasting/generate_production_forecasts.py` | `forecast-generate` | Build an immutable champion release candidate; promotion is a separate API action | `forecast_generation_run` + `fact_production_forecast_staging` |
| `scripts/forecasting/compute_blended_forecast.py` | `blended-all` | Blends short-horizon demand-sensing signals with the statistical baseline using a linearly decaying alpha over a 4-week sensing horizon | `fact_blended_forecast` |
| `scripts/forecasting/generate_consensus_plan.py` | `consensus-generate VERSION=<v>` | Applies approved planner overrides to an existing saved P50 demand plan, honoring the override-priority chain (`CAPACITY_LOCK` > `PROMO`/`LAUNCH` > `PHASE_OUT`/`MARKET_EVENT` > `MANUAL`) | `fact_consensus_plan` |
| `scripts/forecasting/generate_customer_forecasts.py` | Forecasting → Customer Forecast | Generates 18 customer-level months through the ordered `customer_rule_router` in resumable parallel route batches for customer-SKUs with sales in the latest six closed months | `customer_forecast_run` + batch ledger + `fact_customer_forecast` |

**When to run each:**

- **Blended** — short horizon (4 weeks). Run weekly once near-real-time demand-sensing signals are refreshed.
- **Consensus** — only when the selected `plan_version` already exists in `fact_demand_plan`; it remains a historical/imported-plan consumer and is not a producer of quantile forecasts.
- **Customer Forecast** — run on demand after customer-demand ETL is current.
  The load is current only when no `customer_demand` audit batch is `running`
  and `customer_demand_profile_refresh_state.source_batch_id` equals the latest
  completed batch. The loader publishes that marker only after every dependent
  materialized view refresh succeeds; a post-write failure deletes it. The run
  records this exact refreshed batch. If a newer load completes during or after
  generation, start a new customer forecast; persistence, resume, backtest
  creation, blend readiness, and promotion intentionally reject the stale run.
  On startup under the exclusive lineage lock, the canonical loader reconciles
  abandoned `running` audit batches as failed and clears the marker before
  creating the next batch.
  Generation does not consume or change production. Run its historical
  bottom-up backtest next; only passing evidence may create a staged blend
  draft against the active production spine.

The production, blended, and consensus workflows share the same planning-date semantics so the cycle "as-of" date is consistent. Production uncertainty bands come from the residual-based CI path in `forecast-generate`; the retired standalone synthetic quantile generator is not part of the production cycle.
The `fact_demand_plan` and `fact_demand_plan_weekly` schemas and read APIs remain available for previously loaded or externally sourced versions.

### 5a. Customer Forecast, Bottom-Up Backtest, and Blend Draft

Use **Forecasting → Customer Forecast** to check source readiness, launch the
durable `generate_customer_forecast` job, inspect one customer series, and
export a completed run. The configured contract is fixed at the latest 18
fully closed months as context and 18 future monthly outputs. For a system date
in July 2026, that means January 2025–June 2026 history and July 2026–December
2027 forecast output.

Readiness separates active and ignored series. A customer-SKU with no
`sales_qty` in January–June 2026 is ignored and writes no forecast rows. The
coverage card splits each eligible series under the ordered
`customer_rule_router` policy:

1. If its earliest positive demand is in January–June 2026, use
   `moving_average_3`. Each future value is the mean of the latest three
   calendar-month states and is appended before calculating the next month.
2. Otherwise, if at least nine of July 2025–June 2026 have positive demand, use
   `seasonal_repeat_12` and repeat those last 12 actual months cyclically.
3. Otherwise, use recursive `croston` with the configured SBA `alpha` and
   `recursive_damping`.

The production route is frozen in the 10,000-series batch manifest and on its
forecast rows. Six CPU workers share the route batches in parallel. These
customer-scoped routes are not added to the canonical five-model competition.

Each 10,000-series batch commits independently. Jobs displays exact completed
customer-SKU and batch counts plus ETA. Cancellation, worker failure, managed
retry, or API restart preserves completed batches. Use **Resume Saved Batches**
for a failed/cancelled manifested run; use a new generation if configuration
or the completed customer-demand source batch changed. Fact rows can be
replaced only while their parent run is `generating`. A run becomes readable
only after every batch, source-lineage, and final row-count check passes.

The ordered thresholds and Croston recursive settings are part of the customer
and backtest checksums. After changing any of them, do not resume or reuse an
older run: generate a new customer forecast, rerun the blend backtest, and
generate a new blend draft only if the no-WAPE-degradation gate passes.
Migration 218 enforces this cutover by marking queued/generating pre-router
customer runs and backtests failed and invalidating a generating blend tied to
legacy customer evidence. Completed legacy `croston` runs remain readable, but
they are not current `customer_rule_router` evidence and cannot satisfy new
blend or backtest readiness. Apply the migration in a maintenance window with
temporary disk for a side-built copy of the customer profile MV.

If readiness reports an active load, wait for the loader to finish. If it
reports a missing or stale profile marker, rerun the standard customer-demand
load; a standalone MV refresh deliberately does not publish source lineage.
Customer-demand loading also waits while a customer backtest, forward blend,
or customer-blend promotion holds its shared snapshot lock; this prevents a
load from crossing an evidence publication boundary.

After customer generation completes:

1. Confirm the active source is a freshly promoted, unblended champion. A
   promoted `customer_bottom_up_blend` cannot recursively feed another blend.
2. Call `POST /customer-forecast/backtest/generate`. The durable
   `generate_customer_forecast_backtest` job evaluates six causal one-month
   origins with at least six training months and 10,000-customer-series
   batches. Series eligibility and the full ordered route policy are
   re-evaluated from data available at every historical origin rather than
   inherited from the current customer run. The job rejects source-membership,
   series-count, or batch-count drift after submission and uses one
   repeatable-read snapshot for the full multi-batch evaluation.
3. Inspect `GET /customer-forecast/backtest/latest`. The common-cohort
   scorecard reports WAPE, MAE, bias, and accuracy for customer bottom-up,
   source champion, and the 50/50 blend. It passes only with at least six
   months, 1,000 DFUs, and blend WAPE no worse than champion WAPE.
4. Confirm `GET /customer-forecast/blend/readiness` is ready, then call
   `POST /customer-forecast/blend/generate`. The generator sums customer
   forecasts to item-location and converts ordered demand to the sales target
   with a causal 18-month fulfillment ratio. One transaction writes two exact
   staged views:

   - `customer_bottom_up` contains only qualified normalized customer rows. It
     has its own deterministic companion run UUID, uses
     `generation_purpose='shadow_candidate'`, has no confidence interval, and
     is permanently non-promotable.
   - `customer_bottom_up_blend` preserves the full active champion spine under
     the submitted blend run UUID. Qualified rows use the configured 50/50
     blend; missing customer evidence and months beyond the customer horizon
     pass through the source champion. It remains a normal
     `release_candidate` in the governed champion release slot.

   `GET /customer-forecast/blend/latest` returns both the blend `run_id` and its
   exact `bottom_up_staging_run_id`, status, and counts; the manifests retain
   the corresponding payload checksum lineage.
5. Review the exact-run outputs before any release action:

   - For one warehouse-item, call
     `GET /forecast/production/staging?item_id=<item>&loc=<location>`. The
     `models` map exposes `customer_bottom_up` from the companion shadow run and
     `customer_bottom_up_blend` from the release-candidate run. Inspect each
     row's `source_run_id`; do not compare against a different vintage.
   - For historical and forward context together, call
     `GET /customer-forecast/blend/trend?run_id=<blend-run-id>&window=12`.
     The required blend run binds the response to its recorded backtest and
     deterministic shadow run. It returns actual sales and customer bottom-up,
     source-champion, and blend backtests before the planning month, followed
     by the three exact staged totals, blend interval, coverage, and
     common-cohort WAPE. Optional item, location, brand, category, market, and
     cluster filters use the same lineage; channel is reported as not
     applicable at warehouse-item grain.
   - `GET /customer-forecast/blend/series` remains the row-level component
     inspection for one exact item/location, including raw demand, normalized
     customer quantity, source champion, blend, fulfillment, weights, and
     coverage.

   In **Portfolio**, switch **Forecast vs Actual** from **Standard** to
   **Customer Blend**. The chart uses the current blend run and the active
   Portfolio filters, shows the backtest-to-staged boundary, the three series,
   WAPE badges, fallback coverage, planning month, and shortened run identity.
   In **Item Analysis**, select an exact item and location; the unified chart
   overlays the same run's historical bottom-up/source-champion/blend backtests
   and future staged series. The customer legend shows the run and vintage so
   operators can verify both screens are reviewing the same evidence.
6. Only stage and promote the exact `customer_bottom_up_blend` release run
   through the normal backtest-management actions. Promotion revalidates the
   matching completed backtest plus every standard release gate and recomputes
   customer, backtest, source, component, and staging checksums. The companion
   `customer_bottom_up` shadow is already present for review, never appears as
   a release action, and must not be submitted to `/stage` or `/promote`.
   In **Forecast**, confirm the readiness row and action card say **Customer
   Bottom-Up Blend** and show the passing common-cohort gate before staging;
   the API route remains `/backtest-management/champion/...` because the blend
   occupies the governed champion release slot.

The deterministic companion UUID makes the pairing reproducible; it does not
make completed shadow evidence mutable. A new blend submission receives a new
blend UUID and therefore a new companion UUID. Neither generation nor review
rewrites a completed shadow, promoted blend, production forecast, or immutable
component evidence.

`customer_forecast` and `customer_forecast_backtest` each have a 24-hour job
ceiling. Full-spine `customer_forecast_blend` generation has an 8-hour ceiling,
configured in `config/forecasting/pipelines.yaml`. Any change to the customer
model parameters, normalization, blend weights, or source release requires a
fresh backtest and draft. There is no bypass or automatic promotion. See spec
35 for formulas, lineage, and failure rules.

The backtest deliberately keeps one repeatable-read snapshot and shared
customer-demand/source-promotion locks for its full run. On a full population,
that can retain an older PostgreSQL MVCC snapshot and make customer-demand
loads or source replacement wait for several hours. Schedule it outside the
customer-demand load window, monitor the durable job from **Jobs**, and cancel
the job normally if the lock must be released; do not start a competing manual
load or bypass the lineage lock.

---

## 5b. Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)

Run **after** the production forecast exists (Section 2 generate + promote). This step computes a forward-looking replenishment plan — forward safety stock, EOQ, and order quantities — driven by the **CI-band** production forecasts. Output lands in `fact_replenishment_plan`.

```bash
make replplan-compute        # Compute 12-month replenishment plan → fact_replenishment_plan
# preview without writing:
make replplan-compute-dry
```

Schema bootstrap and full pipeline:

```bash
make replplan-schema         # apply sql/041_create_replenishment_plan.sql
make replplan-all            # replplan-schema + replplan-compute (full pipeline)
```

Script: `scripts/inventory/compute_replenishment_plan.py` (`Makefile:1159-1165`).

**Dependency chain for `make replplan-compute`:**

1. `fact_production_forecast` must have rows (from Section 2 generate + promote)
2. `fact_safety_stock_targets` must have rows (`make ss-compute`)
3. `fact_eoq_targets` must have rows (`make eoq-compute`)

---

## 6. Persisted Production Model Training Endpoint

```
POST /backtest-management/{model_id}/train
Headers: X-API-Key: <key>
```

LightGBM, N-HiTS, and N-BEATS require separately persisted production-training
artifacts. MSTL and Chronos 2E fit or infer through their canonical direct
adapters and reject this endpoint with HTTP 400.

| Model family (`type` in algorithm roster) | Accepted? | Reason |
|---|---|---|
| `tree` (`lgbm_cluster`) | YES | Produces the promoted-cluster LightGBM artifacts used by production inference |
| `deep_learning` (`nbeats`, `nhits`) | YES | Produces immutable, lineage-validated global neural final fits |
| `statistical` (`mstl`) | NO -> 400 | Fits per DFU through the canonical MSTL adapter |
| `foundation` (`chronos2_enriched`) | NO -> 400 | Uses the canonical Chronos 2E adapter |

`model_id="all"` is a convenience form that submits the production-training
job for the complete persisted roster: LightGBM, N-BEATS, and N-HiTS. Use
`GET /backtest-management/training-status` to poll completion.

CLI equivalents (`Makefile:1075-1078`):

```bash
make train-production MODEL=lgbm_cluster   # one model
make train-production-all                  # LightGBM + N-BEATS + N-HiTS
```

---

## 7. Verification

After promotion, run these checks before declaring the cycle done.

### 7.1 Confirm month coverage in production

```sql
SELECT plan_version,
       MIN(forecast_month) AS first_month,
       MAX(forecast_month) AS last_month,
       COUNT(DISTINCT forecast_month) AS month_count,
       COUNT(DISTINCT (item_id, loc)) AS dfu_count,
       COUNT(*) AS total_rows
FROM fact_production_forecast
GROUP BY plan_version
ORDER BY plan_version DESC;
```

`month_count` should equal `production_forecast.horizon_months` (default 24). `dfu_count` should match the `dfu_count` returned by the promote endpoint.

### 7.2 Confirm the active model

```sql
SELECT id, model_id, promotion_type, plan_version, dfu_count, total_rows,
       source_run_id, production_run_id,
       candidate_checksum, production_checksum,
       archive_checksum, archived_at,
       promoted_by, promoted_at, notes
FROM model_promotion_log
WHERE is_active = TRUE;
```

Or via API:

```bash
curl -s http://localhost:8000/backtest-management/promotion-status | jq
```

Exactly one row should be active. New releases must have non-null source and
production run ids and equal candidate/production checksums. If zero rows,
nothing has been promoted yet. More than one active row is blocked by
`uq_model_promotion_log_one_active`; treat an index/schema error as a failed
migration, not something to repair with an ad-hoc audit update.

### 7.3 Verify served data

The frontend reads through `api/routers/forecasting/production_forecast.py` (prefix-less, all routes under `/forecast/production*` and `/forecast/demand-plan*`):

| Endpoint | What it returns |
|---|---|
| `GET /forecast/production` | Forecast rows for one or more DFUs (paginated) |
| `GET /forecast/production/summary` | Aggregated rollup over the active `plan_version` |
| `GET /forecast/production/versions` | All distinct `plan_version` values (for version diffing) |
| `GET /forecast/production/staging` | Exact item/location staging read-through, including the latest `customer_bottom_up` shadow and `customer_bottom_up_blend` release views with their distinct source run ids |
| `GET /customer-forecast/blend/trend` | Run-bound Portfolio/Item Analysis comparison of historical backtest actuals and three-model forecasts with their exact staged future totals, accuracy, and coverage |
| `GET /forecast/demand-plan` / `…/weekly` / `…/comparison` | Consensus / quantile blended views |

Quick smoke test:

```bash
curl -s 'http://localhost:8000/forecast/production?item_id=100320&loc=1401-BULK&limit=24' | jq '.rows | length'
# Expect: 24 (one row per horizon month)
```

---

## 8. Re-Run Cadence

The production forecast is a **derived** artifact. Anything that invalidates the inputs requires a re-run.

| Trigger | Steps |
|---|---|
| Monthly planning cycle (default) | Run the required backtests (one, a selected subset, or all five), assign Champion when its composition should change, generate the chosen single model or Champion, then promote that exact staged source run |
| New backtest + champion cycle | Use **Run selected** for the desired models (all five for a new Champion comparison); results auto-load. Assign the chosen Champion experiment only when publishing Champion, then generate and promote the selected candidate |
| Source data refresh (`make pipeline-full` / `pipeline-refresh`) | Run `model-refresh`, `champion-refresh`, review, then `forecast-publish` and promote |
| Hyperparameter tuning (`make tune-all`) | Run `model-refresh`, `champion-refresh`, review, then `forecast-publish` and promote |
| Cluster scenario promotion (`cluster-all`) | Run `model-refresh` against the new assignments, then `champion-refresh`, review, `forecast-publish`, and promote |
| Cold-start config change (`min_history_months`, etc.) | Dry-run to preview, then run `model-refresh`, `champion-refresh`, `forecast-publish`, and promote |

A scheduled job (`forecast_pipeline_config.yaml` `production_forecast.scheduler.cron: "0 6 2 * *"`) runs `generate_production_forecast` on the 2nd of each month at 06:00. **The scheduler does NOT auto-promote** — promotion always requires an explicit API call so an operator owns the gate decision.

---

## 9. Troubleshooting

### 9.1 Cold-start DFUs missing forecast

**Symptom:** A DFU shows up in `dim_sku` and has sales history, but `fact_production_forecast` returns no rows for it.

**Diagnose:**

```sql
SELECT item_id, loc, COUNT(*) AS history_months
FROM fact_sales_monthly
WHERE item_id = '<id>' AND loc = '<loc>'
  AND startdate >= (CURRENT_DATE - INTERVAL '36 months')
GROUP BY item_id, loc;
```

Then check thresholds in `config/forecasting/forecast_pipeline_config.yaml`:

- `history_months < cold_start_min_months` (default 3) -> coherent release
  generation fails rather than publishing a partial item/location population.
  Seed or explicitly remove the cohort from eligibility before retrying.
- `12 <= history_months < 25` with an MSTL champion route -> the promoted
  experiment is stale for the canonical MSTL adapter. Rerun the MSTL backtest
  and the complete five-model champion selection; do not bypass the check.

### 9.2 Promotion failed

**Symptom:** `POST /backtest-management/{model_id}/promote` returned non-2xx.

**Common causes:**

| HTTP | Body fragment | Cause / fix |
|---|---|---|
| 422 | missing `source_run_id` | Read the exact UUID from `GET /backtest-management/staging-summary` or the Generate response and resend it |
| 404 | `candidate_run_not_found` | The UUID is unknown; do not substitute a different/older run—generate a fresh candidate |
| 409 | `candidate_run_not_promotable` | The run is a snapshot contender, pre-migration `legacy_invalid`, already promoted, partial/debug generation, or otherwise ineligible; generate a full release candidate |
| 409 | `stale_candidate_evidence` | Planning month or latest completed sales batch changed after generation; generate again from current inputs |
| 409 | `candidate_lineage_mismatch` | Generator contract, manifest hash/cardinality, champion/cluster lineage, assignments, tuning, source-model count, or winners-file hash changed; repair lineage and generate again |
| 409 | `candidate_quality_failed` | The exact experiment-stamped historical champion common cohort fails WAPE lift, required external comparison, bias, sufficiency, or actual-alignment policy. If the external feed is deliberately unavailable, set `champion.release_readiness.require_external_benchmark: false`; this exempts only the external comparison and keeps the champion-versus-naive six-month gate intact. Re-enable it after loading representative external data; no regeneration is required solely for that policy change. |
| 409 | `candidate_gate_failed` | Six-month coverage, route continuity, quantities, horizon, or confidence intervals do not meet policy; inspect the generation log and manifest before retrying |
| 409 | `concurrent_release_conflict` | Another release changed state or the same source run was reused; refresh status and generate a new immutable candidate if needed |
| 409 | `production_checksum_mismatch` | Published values did not reproduce the selected run; the transaction rolled back—investigate database triggers/schema and do not retry blindly |
| 401 | `Missing API key` | Pass `X-API-Key` header |
| 500 | opaque server detail | Inspect API/Postgres logs. The transaction is atomic, so the outgoing active row and production payload should remain unchanged |

There is no force/bypass parameter. Snapshot state is not a promotion gate;
candidate lineage, structural quality, coverage, confidence intervals, and
checksums remain mandatory. A rejection leaves the current release unchanged
and is represented by the stable error code returned to the caller.

### 9.3 Candidate vs Production discrepancy

**Symptom:** A model shows good historical accuracy on the Backtest Management
tab (`fact_external_forecast_monthly`/`backtest_lag_archive`) but consumers
report different forward numbers from `/forecast/production`.

**Likely causes:**

1. **Different model is promoted.** Check `GET /backtest-management/promotion-status` — `model_id` may not match what you're inspecting.
2. **Stale `plan_version`.** Production keeps the most recently promoted `plan_version` only (DELETE-then-INSERT). If a downstream consumer cached an older `plan_version`, refresh.
3. **Generate not run since last train/input change.** Backtest evidence and
   forward inference are separate. If you retrained or loaded newer sales without
   generating a new run, the old run is stale and promotion rejects it.
4. **Champion mode mismatch.** In champion mode, `fact_production_forecast.model_id = 'champion'` but `source_model_id` carries the per-DFU winner. Filter by `source_model_id` when comparing.

Useful diagnostic:

```sql
-- Compare staging vs production for a single DFU
SELECT 'staging' AS src, candidate_model_id AS release_model,
       model_id AS source_model_id, run_id,
       forecast_month, forecast_qty
FROM fact_production_forecast_staging
WHERE item_id = '<id>' AND loc = '<loc>'
  AND run_id = '<source_run_id>'::uuid
UNION ALL
SELECT 'prod' AS src, model_id AS release_model,
       source_model_id, source_run_id AS run_id,
       forecast_month, forecast_qty
FROM fact_production_forecast
WHERE item_id = '<id>' AND loc = '<loc>'
ORDER BY src, forecast_month;
```

---

## 10. Quick Reference

| Action | Command |
|---|---|
| Schema bootstrap | `make forecast-prod-schema` |
| Train persisted production models | `make train-production-all` |
| Refresh five backtests | Run named pipeline `model-refresh` from the Jobs UI/API |
| Select and atomically assign champion | Run named pipeline `champion-refresh` from the Jobs UI/API |
| Train, generate, and prepare top three | Run named pipeline `forecast-publish` from the Jobs UI/API |
| Generate all DFUs | `make forecast-generate` |
| Generate single DFU | `make forecast-generate-dfu ITEM=<id> LOC=<loc>` |
| Dry-run | `make forecast-generate-dry` |
| Retrain persisted models + generate (not a model refresh) | `make forecast-full` |
| Select exact ready run | `RUN_ID=$(curl -s "$BASE/backtest-management/staging-summary" | jq -r '.champion.source_run_id')` |
| Promote champion blend | `curl -X POST -H "X-API-Key: $KEY" "$BASE/backtest-management/champion/promote?source_run_id=$RUN_ID"` |
| Promote single model | Set `RUN_ID` from that model's staging-summary entry, then call the same path with its model id |
| Active promotion | `curl -s "$BASE/backtest-management/promotion-status"` |
| Staging summary | `curl -s "$BASE/backtest-management/staging-summary"` |
| Candidate summary | `curl -s "$BASE/backtest-management/candidate-summary"` |
| Replenishment plan (CI bands) | `make replplan-compute` |
| Replenishment plan dry-run | `make replplan-compute-dry` |
| Blended forecast | `make blended-all` |
| Consensus plan | `make consensus-generate VERSION=$(date +%Y-%m)` |
| Customer-level forecast | Open **Forecasting → Customer Forecast**, confirm readiness, then click **Generate Customer Forecasts** |
| Customer blend backtest | `curl -X POST -H "X-API-Key: $KEY" "$BASE/customer-forecast/backtest/generate"` |
| Customer blend gate | `curl -s "$BASE/customer-forecast/backtest/latest"` |
| Generate customer blend draft | `curl -X POST -H "X-API-Key: $KEY" "$BASE/customer-forecast/blend/generate"` |
| Inspect customer blend | `curl -s "$BASE/customer-forecast/blend/latest"` |
| Inspect both staged customer views for one DFU | `curl -s "$BASE/forecast/production/staging?item_id=$ITEM&loc=$LOC"` |
| Select exact customer blend run | `BLEND_RUN_ID=$(curl -s "$BASE/customer-forecast/blend/latest" | jq -r '.run_id')` |
| Inspect exact-run customer trend | `curl -s "$BASE/customer-forecast/blend/trend?run_id=$BLEND_RUN_ID&window=12"` |

Source-of-truth files referenced in this section:

- `Makefile` (production-forecast targets)
- `config/forecasting/forecast_pipeline_config.yaml` (`production_forecast:`, `customer_forecast:`, and `forecast_snapshot:` blocks)
- `config/forecasting/pipelines.yaml` (customer forecast/backtest/blend job ceilings)
- `scripts/forecasting/generate_production_forecasts.py`
- `scripts/forecasting/generate_customer_forecasts.py`
- `common/ml/customer_forecast_rules.py`
- `common/services/customer_forecast_backtest_rules.py`
- `scripts/ml/train_production_models.py`
- `common/ml/tree_artifacts.py`
- `common/ml/neural_artifacts.py`
- `api/routers/forecasting/backtest_management.py`
- `api/routers/forecasting/forecast_promotion.py`
- `api/routers/forecasting/production_forecast.py`
- `common/services/forecast_promotion.py`
- `common/services/forecast_lineage.py`
- `common/services/customer_forecast_blend.py`
- `sql/039_create_production_forecast.sql`
- `sql/121_candidate_forecast_and_promotion.sql`
- `sql/122_create_production_forecast_staging.sql`
- `sql/203_create_forecast_generation_run.sql`
- `sql/205_enforce_champion_model_roster.sql`
- `sql/206_invalidate_pre_canonical_generator_runs.sql`
- `sql/216_create_customer_bottom_up_blend.sql`
- `sql/217_add_customer_bottom_up_shadow_staging.sql`
- `sql/218_enable_customer_rule_router.sql`
