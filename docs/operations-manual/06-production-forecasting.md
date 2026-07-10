# 06 — Production Forecasting

This section covers **generating, promoting, and serving production forecasts**. Production forecasts are the forward-looking demand signals consumed by inventory planning, S&OP, and replenishment. They are produced from the trained champion models that emerged from the backtest + champion selection cycle (covered in Sections 04 and 05).

Repo root for all paths and commands: `/Users/manoharchidambaram/projects/DemandProject`.

---

## 1. Architecture Overview

The production forecast pipeline follows an **immutable run → transactional
promote** pattern. Forward predictions are grouped by an explicit generation
manifest; only one ready, eligible run can reach the consumer-facing production
table.

```
                ┌──────────────────────────────────┐
                │  Trained Champion Models (.pkl)  │
                │  data/models/<model_id>/         │
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
| `forecast_generation_run` | one row per generation UUID | Manifest status, purpose, input lineage, counts, checksums | `sql/203_create_forecast_generation_run.sql` |
| `fact_production_forecast_staging` | run + purpose + candidate + item + loc + month | Immutable rows; `release_candidate`, `snapshot_contender`, and retained `legacy_invalid` never mix | `sql/122`, extended by `sql/203` |
| `fact_production_forecast` | plan_version + item + loc + forecast_month | Replaced only inside verified promotion; every new row carries source run, production run, and audit id | `sql/039`, extended by `sql/203` |
| `model_promotion_log` | one row per promote/demote event | Exact source/release/archive lineage; one active row enforced by unique index | `sql/121`, extended by `sql/203` |

`model_promotion_log.promotion_type` is constrained to `('single', 'champion')`;
the database enforces at most one `is_active=TRUE` row and at most one promotion
per non-null `source_run_id`.

---

## 2. Generate Forecasts (`make forecast-generate`)

### One-time migration 203 cutover

Apply migrations through `sql/203_create_forecast_generation_run.sql` before
using the Generate or Promote UI:

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp \
  -v ON_ERROR_STOP=1 < sql/203_create_forecast_generation_run.sql
```

Migration 203 intentionally classifies all pre-manifest staging as
`legacy_invalid`. It does not infer that an old or mixed population was a safe
release candidate. It also leaves pre-migration champion result rows without an
experiment id/checksum. After applying it, use the Champion tab (or
`POST /champion-experiments/{id}/promote-results`) to reload and explicitly
promote the selected experiment's cached winners, then run a fresh full Generate
action and promote only the returned `source_run_id`. Existing production
remains readable, but replacement can proceed only when the outgoing release has
verifiable run lineage and its champion-plus-three archive can reconcile.

### What it does

`make forecast-generate` runs
`scripts/forecasting/generate_production_forecasts.py`, which loads the promoted
champion experiment's per-month winners artifact, pulls matching trained `.pkl`
artifacts under `data/models/<model_id>/`, and runs **recursive inference** over
the next `horizon_months` months.

Inputs:

- **Trained model artifacts** — `data/models/<model_id>/cluster_*.pkl`, produced by `make train-production-all` (one fitted model per `ml_cluster`).
- **Champion routing** —
  `data/champion/experiment_<promoted_id>_winners.csv`, whose exact bytes are
  SHA-256 stamped on the generation manifest. `dfu_assignments.csv` is not a
  production routing source.
- **Sales history** — last `lookback_months` (default 36) of `fact_sales_monthly`, used to seed lag features for T+1.
- **Pipeline config** — `config/forecasting/forecast_pipeline_config.yaml` `production_forecast:` block (lines 435-468).

Output:

- One `forecast_generation_run` manifest plus immutable rows in
  **`fact_production_forecast_staging`**, one row per
  `(run_id, candidate_model_id, item_id, loc, forecast_month)`. A normal
  champion generation is one coherent `release_candidate` run; each row's
  `model_id` preserves the true routed source model.
- A completed manifest stamped with the current planning month, latest completed
  sales batch, champion/cluster experiment lineage, row/DFU/source-model counts,
  routing artifact checksum, exact historical champion-results checksum, and
  canonical staging payload checksum.

The Forecast UI's Generate endpoint allocates the UUID before the job starts and
returns it as `source_run_id`. The UI then polls for that exact run; it never
mistakes an older completed generation for the newly submitted job.

### Command variants

| Target | Purpose | Source |
|---|---|---|
| `make forecast-generate` | Full DFU population | `Makefile:1081` |
| `make forecast-generate-dfu ITEM=<id> LOC=<loc>` | One DFU only — debugging | `Makefile:1084` |
| `make forecast-generate-dry` | Inference + log row count, no DB writes | `Makefile:1087` |
| `make forecast-prod-all` | `forecast-prod-schema` + `forecast-generate` | `Makefile:1090` |
| `make forecast-full` | `train-production-all` + `forecast-generate` (full retrain + regenerate) | `Makefile:1092` |
| `make forecast-model MODEL=<id>` | Generate for a single model id | `Makefile:1096` |

### Recursive inference (how it predicts beyond T+1)

For each DFU the script loops `h = 1 .. horizon_months`. At step `h+1`, the lag features (`lag_1 .. lag_n`, `rolling_*`) are reconstructed from the predictions emitted at earlier `h` steps — i.e. predictions feed back into the feature grid. Each row records `lag_source = 'actual'` (T+1) or `'predicted'` (T+2+) so consumers can identify recursive uncertainty.

---

## 3. Cold-Start Routing

The DFU population is heterogeneous — some items have years of history, some are brand-new. The pipeline routes each DFU through one of three paths based on `(item_id, loc)` history depth. All thresholds live in `config/forecasting/forecast_pipeline_config.yaml` under `production_forecast:` (lines 435-441).

| Path | Rule | `model_id` used |
|---|---|---|
| **Skip** (absolute floor) | `n_months < cold_start_min_months` (default **3**) | — DFU produces **no rows** in `fact_production_forecast_staging` |
| **Champion** | `n_months >= min_history_months` (default **12**) | DFU's assigned champion `model_id` (or `fallback_model_id` if the champion artifact is missing) |

Implementation: `scripts/forecasting/generate_production_forecasts.py`.

```yaml
# config/forecasting/forecast_pipeline_config.yaml (lines 435-441)
production_forecast:
  horizon_months: 24
  lookback_months: 36
  min_history_months: 12          # threshold for full champion model
  cold_start_min_months: 3        # absolute floor — below this, DFU is skipped
  fallback_model_id: lgbm_cluster # used when champion artifact is missing
```

> **Operator note:** if you change `cold_start_min_months`, expect the DFU count in `fact_production_forecast_staging` to shift. Run `make forecast-generate-dry` first to preview the impact.


---

## 4. Promotion Workflow

Promotion is the deliberate operator action that publishes one exact immutable
run. The endpoint is auth-guarded, fail-closed, and audited.

### Endpoint

```
POST /backtest-management/{model_id}/promote
Headers: X-API-Key: <key>             # require_api_key dependency
Query  : ?source_run_id=<uuid>        # required; returned by Generate/staging-summary
         &notes=<text>                # optional, recorded on the audit row
         &promoted_by=<user|system>   # optional, default 'api'
```

Defined in `api/routers/forecasting/forecast_promotion.py`.

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
   `release_candidate`, status `ready`, eligible, requested model/current month,
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
5. If an outgoing release exists, archive its exact champion plus the frozen
   top three `snapshot_contender` runs for lags 0-5. All four series must cover
   all six lags, and the archived champion checksum must equal the outgoing
   production checksum. Any failure stops replacement.
6. Demote the outgoing audit row; insert the incoming audit row with source and
   production run ids, gate report, candidate/production checksums, replacement
   lineage, and outgoing archive checksum; then replace production.
7. Hash production and require exact equality with the selected candidate,
   mark the source manifest `promoted`, and commit. Any error rolls back the
   archive, demotion, delete, insert, audit, and manifest transition together.

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
| `GET /backtest-management/staging-summary` | Latest immutable release-candidate manifest per requested model, including exact `source_run_id`, status, eligibility, row/DFU/source-model counts, and horizon dates |

There is no bypass token. Fix the failed evidence, generate a new candidate when
inputs changed, and retry with its new run id.

---

## 5. Other Production Scripts

These scripts produce **additional** forecast layers that sit alongside (or downstream of) the champion point forecast. They are run on demand or as part of `make setup-demand-planning` (`Makefile:1430`).

| Script | Make target | Purpose | Output table |
|---|---|---|---|
| `scripts/forecasting/generate_production_forecasts.py` | `forecast-generate` | Build an immutable champion release candidate; promotion is a separate API action | `forecast_generation_run` + `fact_production_forecast_staging` |
| `scripts/generate_quantile_forecasts.py` | `quantile-train VERSION=<v>` | LightGBM quantile regression for P10/P50/P90; weekly disaggregation | `fact_demand_plan` (⚠ MVP stub — see note) |
| `scripts/compute_blended_forecast.py` | `blended-all` | Blends short-horizon demand-sensing signals with the statistical baseline using a linearly decaying alpha over a 4-week sensing horizon | `fact_blended_forecast` |
| `scripts/generate_consensus_plan.py` | `consensus-generate VERSION=<v>` | Merges P50 baseline with approved planner overrides (`fact_forecast_overrides`) honoring the override-priority chain (`CAPACITY_LOCK` > `PROMO`/`LAUNCH` > `PHASE_OUT`/`MARKET_EVENT` > `MANUAL`) | `fact_consensus_plan` |

**When to run each:**

- **Quantile** — when downstream planning needs uncertainty bands (safety-stock sizing, service-level optimisation). Uses its own quantile models, independent of the champion point forecast. **⚠ Stub as of 2026-06-20:** `generate_quantile_forecasts.py` trains on `rng.uniform` random data with constant dummy features, so its output is statistically meaningless. It now **refuses to write `fact_demand_plan`** unless run with `--dry-run` (preview) or `--allow-synthetic` (dev override) — `make quantile-train` will fail loud with a `NotImplementedError` until the script is wired to the real feature pipeline. **Do not run it in a production cycle.** For production CI bands, use the residual-based bands produced by `forecast-generate` (see [Forecast CI Bands](../specs/02-forecasting/10-forecast-ci-bands.md)).
- **Blended** — short horizon (4 weeks). Run weekly once near-real-time demand-sensing signals are refreshed.
- **Consensus** — after planners post overrides for the cycle. Always run after `forecast-generate` + planner sign-off.

All four scripts share the same date semantics via `common.planning_date.get_planning_date()` so the cycle "as-of" date is consistent.

`scripts/ml/run_expert_system_backtest.py` is evaluation-only. It writes
backtest evidence but has no code path that writes
`fact_production_forecast`; every production release must pass the explicit
generation and promotion contract above.

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

## 6. Tree-Only Production Training Endpoint

```
POST /backtest-management/{model_id}/train
Headers: X-API-Key: <key>
```

Defined in `api/routers/forecasting/backtest_management.py:174`.

**Validation gate** (line 192):

```python
_TRAINABLE_TYPES = {"tree"}
if algo_info.get("type") not in _TRAINABLE_TYPES:
    raise HTTPException(
        status_code=400,
        detail=f"Model '{model_id}' (type={algo_info.get('type')}) does not require training. "
               f"Only tree models need explicit training.",
    )
```

| Model family (`type` in algorithm roster) | Accepted? | Reason |
|---|---|---|
| `deep_learning` (nbeats, nhits) | NO -> 400 | Trained inside their own backtest scripts |

`model_id="all"` is a special form that submits a single job training every forecastable tree model on full history. Use `GET /backtest-management/training-status` to poll completion (returns `trained`, `trained_at`, `n_dfus`, `planning_date` per model).

CLI equivalents (`Makefile:1075-1078`):

```bash
make train-production MODEL=lgbm_cluster   # one model
make train-production-all                  # all forecastable tree models
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
| `GET /forecast/production/staging` | Read-through to staging (compare across models pre-promote) |
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
| Monthly planning cycle (default) | `make forecast-full` (retrain + regenerate) -> promote via API |
| New backtest + champion cycle | `make champion-all` -> review on Champion tab -> `make forecast-generate` -> promote |
| Source data refresh (`make pipeline-full` / `pipeline-refresh`) | `make forecast-generate` (no retrain needed unless coverage shifted) -> promote |
| Hyperparameter tuning (`make tune-all`) | `make train-production-all` -> `make forecast-generate` -> promote |
| Cluster scenario promotion (`cluster-all`) | `make train-production-all` (cluster assignments changed) -> `forecast-generate` -> promote |
| Cold-start config change (`min_history_months`, etc.) | `make forecast-generate-dry` to preview -> `make forecast-generate` -> promote |

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

- `history_months < cold_start_min_months` (default 3) -> DFU is **intentionally skipped**. Lower the floor only if you accept very low-confidence forecasts.

### 9.2 Promotion failed

**Symptom:** `POST /backtest-management/{model_id}/promote` returned non-2xx.

**Common causes:**

| HTTP | Body fragment | Cause / fix |
|---|---|---|
| 422 | missing `source_run_id` | Read the exact UUID from `GET /backtest-management/staging-summary` or the Generate response and resend it |
| 404 | `candidate_run_not_found` | The UUID is unknown; do not substitute a different/older run—generate a fresh candidate |
| 409 | `candidate_run_not_promotable` | The run is a snapshot contender, pre-migration `legacy_invalid`, already promoted, partial/debug generation, or otherwise ineligible; generate a full release candidate |
| 409 | `stale_candidate_evidence` | Planning month or latest completed sales batch changed after generation; generate again from current inputs |
| 409 | `candidate_lineage_mismatch` | Manifest hash/cardinality, champion/cluster lineage, assignments, tuning, source-model count, or winners-file hash changed; repair lineage and generate again |
| 409 | `candidate_quality_failed` | The exact experiment-stamped historical champion common cohort fails WAPE lift, incumbent, bias, sufficiency, or actual-alignment policy; review/promote better champion results before regenerating |
| 409 | `candidate_gate_failed` | Six-month coverage, route continuity, quantities, horizon, or confidence intervals do not meet policy; inspect the generation log and manifest before retrying |
| 409 | `outgoing_archive_incomplete` | The current release cannot be replaced until its champion + top-three lags 0-5 archive is complete and the champion checksum reconciles; prepare the frozen contenders/roster and retry |
| 409 | `concurrent_release_conflict` | Another release changed state, the same source run was reused, or a second same-planning-month release was attempted; refresh status and generate for the next supported record month |
| 409 | `production_checksum_mismatch` | Published values did not reproduce the selected run; the transaction rolled back—investigate database triggers/schema and do not retry blindly |
| 401 | `Missing API key` | Pass `X-API-Key` header |
| 500 | opaque server detail | Inspect API/Postgres logs. The transaction is atomic, so the outgoing active row and production payload should remain unchanged |

There is no force/bypass path. A successful promotion persists its gate report
and checksums in `model_promotion_log`; a rejection leaves the current release
unchanged and is represented by the stable error code returned to the caller.

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
| Train all tree models | `make train-production-all` |
| Generate all DFUs | `make forecast-generate` |
| Generate single DFU | `make forecast-generate-dfu ITEM=<id> LOC=<loc>` |
| Dry-run | `make forecast-generate-dry` |
| Train + generate (full pipeline) | `make forecast-full` |
| Select exact ready run | `RUN_ID=$(curl -s "$BASE/backtest-management/staging-summary" | jq -r '.champion.source_run_id')` |
| Promote champion blend | `curl -X POST -H "X-API-Key: $KEY" "$BASE/backtest-management/champion/promote?source_run_id=$RUN_ID"` |
| Promote single model | Set `RUN_ID` from that model's staging-summary entry, then call the same path with its model id |
| Active promotion | `curl -s "$BASE/backtest-management/promotion-status"` |
| Staging summary | `curl -s "$BASE/backtest-management/staging-summary"` |
| Candidate summary | `curl -s "$BASE/backtest-management/candidate-summary"` |
| Replenishment plan (CI bands) | `make replplan-compute` |
| Replenishment plan dry-run | `make replplan-compute-dry` |
| Quantile forecasts (⚠ stub — refuses DB write, see §5) | `make quantile-train VERSION=$(date +%Y-%m)` |
| Blended forecast | `make blended-all` |
| Consensus plan | `make consensus-generate VERSION=$(date +%Y-%m)` |

Source-of-truth files referenced in this section:

- `Makefile` (lines 1071-1096 production-forecast targets)
- `config/forecasting/forecast_pipeline_config.yaml` (`production_forecast:` block, lines 435-468)
- `scripts/forecasting/generate_production_forecasts.py`
- `api/routers/forecasting/backtest_management.py`
- `api/routers/forecasting/forecast_promotion.py`
- `api/routers/forecasting/production_forecast.py`
- `common/services/forecast_promotion.py`
- `common/services/forecast_lineage.py`
- `sql/039_create_production_forecast.sql`
- `sql/121_candidate_forecast_and_promotion.sql`
- `sql/122_create_production_forecast_staging.sql`
- `sql/203_create_forecast_generation_run.sql`
