# 06 — Production Forecasting

This section covers **generating, promoting, and serving production forecasts**. Production forecasts are the forward-looking demand signals consumed by inventory planning, S&OP, and replenishment. They are produced from the trained champion models that emerged from the backtest + champion selection cycle (covered in Sections 04 and 05).

Repo root for all paths and commands: `/Users/manoharchidambaram/projects/DemandProject`.

---

## 1. Architecture Overview

The production forecast pipeline follows a **stage → promote** pattern. All model predictions land in staging tables; only the actively promoted model's rows reach the consumer-facing production table.

```
                ┌──────────────────────────────────┐
                │  Trained Champion Models (.pkl)  │
                │  data/models/<model_id>/         │
                └─────────────┬────────────────────┘
                              │ make forecast-generate
                              ▼
        ┌─────────────────────────────────────────────────┐
        │  fact_production_forecast_staging               │
        │  (sql/122_create_production_forecast_staging)   │
        │  All model variants coexist, keyed by model_id  │
        └─────────────┬───────────────────────────────────┘
                      │ POST /backtest-management/{model_id}/promote
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
        │  (sql/121_candidate_forecast_and_promotion.sql) │
        │  is_active flag; promotion_type=single|champion │
        └─────────────────────────────────────────────────┘
```

There is also a parallel staging table — **`fact_candidate_forecast`** — fed by the backtest-load step (`backtest-load-all-bulk`). It stores **historical** model predictions (with actuals + accuracy) used for champion selection. Production forecast inference writes the **forward** horizon into `fact_production_forecast_staging`.

| Table | Grain | Lifecycle | DDL |
|---|---|---|---|
| `fact_candidate_forecast` | item + loc + model + month (historical) | Append on backtest-load; `is_promoted` flipped on promote | `sql/121_candidate_forecast_and_promotion.sql` |
| `fact_production_forecast_staging` | model + item + loc + forecast_month | Replace per `model_id` on each `forecast-generate` | `sql/122_create_production_forecast_staging.sql` |
| `fact_production_forecast` | plan_version + item + loc + forecast_month | DELETE-then-INSERT on each promote | `sql/039_create_production_forecast.sql` |
| `model_promotion_log` | one row per promote/demote event | Insert-only audit; `is_active` toggled | `sql/121_candidate_forecast_and_promotion.sql` |

`model_promotion_log.promotion_type` is constrained to `('single', 'champion')`; `is_active=TRUE` for at most one row at a time.

---

## 2. Generate Forecasts (`make forecast-generate`)

### What it does

`make forecast-generate` runs `scripts/generate_production_forecasts.py` (`Makefile:1081-1082`), which loads the configured champion (or per-DFU champion assignments), pulls the matching trained `.pkl` artifacts under `data/models/<model_id>/`, and runs **recursive inference** over the next `horizon_months` months.

Inputs:

- **Trained model artifacts** — `data/models/<model_id>/cluster_*.pkl`, produced by `make train-production-all` (one fitted model per `ml_cluster`).
- **Champion DFU assignments** — `data/champion/dfu_assignments.csv` (or, when missing, the top-accuracy candidate per DFU read from `fact_candidate_forecast`; see `_load_dfu_assignments` in `api/routers/forecasting/backtest_management.py:533`).
- **Sales history** — last `lookback_months` (default 36) of `fact_sales_monthly`, used to seed lag features for T+1.
- **Pipeline config** — `config/forecast_pipeline_config.yaml` `production_forecast:` block (lines 435-468).

Output:

- Rows inserted into **`fact_production_forecast_staging`** (DELETE-before-INSERT per `model_id`), one row per `(item_id, loc, forecast_month)` for `h = 1 .. horizon_months`.
- Confidence-interval bounds (`forecast_qty_lower`, `forecast_qty_upper`) when `production_forecast.confidence_interval.enabled=true`, computed from backtest residuals for the configured `source_model_ids` (default: lgbm/catboost/xgboost cluster).

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

The DFU population is heterogeneous — some items have years of history, some are brand-new. The pipeline routes each DFU through one of three paths based on `(item_id, loc)` history depth. All thresholds live in `config/forecast_pipeline_config.yaml` under `production_forecast:` (lines 435-441).

| Path | Rule | `model_id` used |
|---|---|---|
| **Skip** (absolute floor) | `n_months < cold_start_min_months` (default **3**) | — DFU produces **no rows** in `fact_production_forecast_staging` |
| **Cold-start** | `cold_start_min_months <= n_months < min_history_months` (default 3 ≤ n < **12**) | `cold_start_model_id` (default `rolling_mean`) |
| **Champion** | `n_months >= min_history_months` (default **12**) | DFU's assigned champion `model_id` (or `fallback_model_id` if the champion artifact is missing) |

Implementation: `scripts/generate_production_forecasts.py:1442-1454`.

```yaml
# config/forecast_pipeline_config.yaml (lines 435-441)
production_forecast:
  horizon_months: 24
  lookback_months: 36
  min_history_months: 12          # threshold for full champion model
  cold_start_model_id: rolling_mean
  cold_start_min_months: 3        # absolute floor — below this, DFU is skipped
  fallback_model_id: lgbm_cluster # used when champion artifact is missing
```

> **Operator note:** if you change `cold_start_min_months`, expect the DFU count in `fact_production_forecast_staging` to shift. Run `make forecast-generate-dry` first to preview the impact.

After generation, the run summary log line reports the cold-start count, e.g. `cold-start routed: 1,247 rows -> rolling_mean`.

---

## 4. Promotion Workflow

Generation writes **all** participating models into staging; promotion is the deliberate operator action that moves one model's (or a per-DFU champion blend's) rows into `fact_production_forecast`. The endpoint is auth-guarded and audited.

### Endpoint

```
POST /backtest-management/{model_id}/promote
Headers: X-API-Key: <key>             # require_api_key dependency
Query  : ?notes=<text>                # optional, recorded on the audit row
         &promoted_by=<user|system>   # optional, default 'api'
         &bypass_token=<token>        # optional, only when promote_gate is enabled
```

Defined in `api/routers/forecasting/backtest_management.py:833`.

### Single mode vs Champion mode

| Mode | When to use | What gets copied |
|---|---|---|
| **Single** — `model_id != "champion"` | One model has uniformly best accuracy across the population | All staging rows where `model_id = <id>` |
| **Champion** — `model_id == "champion"` | Per-DFU best model from a champion experiment | For each DFU, the staging row from that DFU's winning model. Routing read from `data/champion/experiment_<id>_winners.csv` |

Champion mode requires a **promoted champion experiment** (`champion_experiment.is_promoted = TRUE`) plus its winners CSV; otherwise the endpoint returns 400.

### Step-by-step flow (executed atomically inside one transaction)

1. **Promotion gate** (single mode only) — `_evaluate_promotion_gate` checks `champion.promote_gate` config: minimum WAPE improvement vs current champion + minimum DFU coverage. A `bypass_token` matching `gate_cfg.bypass_token` skips the check. Pass/reject is recorded in the AI decision ledger via `append_decision` either way.
2. **Validate staging** — reject with HTTP 400 if no rows exist for the requested `model_id` in `fact_production_forecast_staging`.
3. **Demote current** — `UPDATE model_promotion_log SET is_active=FALSE, demoted_at=NOW() WHERE is_active=TRUE`.
4. **Clear production** — `DELETE FROM fact_production_forecast` (single source of truth, replaced wholesale).
5. **Copy staging → production** — INSERT...SELECT, stamping a fresh `plan_version` (`YYYY-MM`) and `run_id` (UUID).
   - Single mode: `WHERE model_id = %s`.
   - Champion mode: JOIN to a temp `_dfu_champion(item_id, loc, winning_model_id)` table populated via `COPY` from the winners CSV.
6. **Insert audit row** into `model_promotion_log` with `promotion_type`, `champion_experiment_id`, `dfu_count`, `total_rows`, `promoted_by`, `notes`.
7. **Emit lineage event** (best-effort) — `emit_lineage_event` writes an OpenLineage `COMPLETE` event linking `fact_production_forecast_staging` → `fact_production_forecast`. Failures are logged but do not block the promotion.
8. **Commit** — single transaction, so any failure rolls back demote + delete + insert together.

Response payload:

```json
{
  "model_id": "lgbm_cluster",
  "promotion_type": "single",
  "plan_version": "2026-04",
  "rows_promoted": 1234567,
  "dfu_count": 51234
}
```

### Inspecting promotion state (no auth required, read-only)

| Endpoint | Returns |
|---|---|
| `GET /backtest-management/promotion-status` | The single active promotion (or `{"promoted": null}`) |
| `GET /backtest-management/candidate-summary` | Per-model row/DFU/avg-accuracy in `fact_candidate_forecast` |
| `GET /backtest-management/staging-summary` | Per-model row/DFU/horizon coverage in `fact_production_forecast_staging` |

---

## 5. Other Production Scripts

These scripts produce **additional** forecast layers that sit alongside (or downstream of) the champion point forecast. They are run on demand or as part of `make setup-demand-planning` (`Makefile:1430`).

| Script | Make target | Purpose | Output table |
|---|---|---|---|
| `scripts/generate_production_forecasts.py` | `forecast-generate` | Champion point forecast — the canonical demand signal | `fact_production_forecast_staging` -> `fact_production_forecast` |
| `scripts/generate_quantile_forecasts.py` | `quantile-train VERSION=<v>` | LightGBM quantile regression for P10/P50/P90; weekly disaggregation | `fact_demand_plan` |
| `scripts/compute_blended_forecast.py` | `blended-all` | Blends short-horizon demand-sensing signals with the statistical baseline using a linearly decaying alpha over a 4-week sensing horizon | `fact_blended_forecast` |
| `scripts/generate_consensus_plan.py` | `consensus-generate VERSION=<v>` | Merges P50 baseline with approved planner overrides (`fact_forecast_overrides`) honoring the override-priority chain (`CAPACITY_LOCK` > `PROMO`/`LAUNCH` > `PHASE_OUT`/`MARKET_EVENT` > `MANUAL`) | `fact_consensus_plan` |

**When to run each:**

- **Quantile** — when downstream planning needs uncertainty bands (safety-stock sizing, service-level optimisation). Uses its own quantile models, independent of the champion point forecast.
- **Blended** — short horizon (4 weeks). Run weekly once near-real-time demand-sensing signals are refreshed.
- **Consensus** — after planners post overrides for the cycle. Always run after `forecast-generate` + planner sign-off.

All four scripts share the same date semantics via `common.planning_date.get_planning_date()` so the cycle "as-of" date is consistent.

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
| `tree` (lgbm_cluster, catboost_cluster, xgboost_cluster + `_cust_enriched` variants) | YES | Need fitted `.pkl` per cluster |
| `foundation` (chronos, chronos_bolt, chronos2, chronos2_enriched, bolt_hierarchical) | NO -> 400 | Zero/few-shot — no training required |
| `deep_learning` (nbeats, nhits) | NO -> 400 | Trained inside their own backtest scripts |
| `statistical` (mstl, seasonal_naive, rolling_mean) | NO -> 400 | No fit step |

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
       promoted_by, promoted_at, notes
FROM model_promotion_log
WHERE is_active = TRUE;
```

Or via API:

```bash
curl -s http://localhost:8000/backtest-management/promotion-status | jq
```

Exactly one row should be active. If zero rows: nothing has been promoted yet. If more than one: trigger investigation — the demote step in the promote transaction should make this impossible.

### 7.3 Verify served data

The frontend reads through `api/routers/production_forecast.py` (prefix-less, all routes under `/forecast/production*` and `/forecast/demand-plan*`):

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

Then check thresholds in `config/forecast_pipeline_config.yaml`:

- `history_months < cold_start_min_months` (default 3) -> DFU is **intentionally skipped**. Lower the floor only if you accept very low-confidence forecasts.
- `cold_start_min_months <= history_months < min_history_months` (default 12) -> DFU should be served by `cold_start_model_id` (`rolling_mean`). Check that staging contains rolling_mean rows for the DFU. If missing, regenerate: `make forecast-generate-dfu ITEM=<id> LOC=<loc>`.

### 9.2 Promotion failed

**Symptom:** `POST /backtest-management/{model_id}/promote` returned non-2xx.

**Common causes:**

| HTTP | Body fragment | Cause / fix |
|---|---|---|
| 400 | `No staged forecasts found` | Run `make forecast-generate` first; verify with `GET /backtest-management/staging-summary` |
| 400 | `No promoted champion experiment found` | Promote a champion experiment on the Champion tab, or pass a non-`champion` `model_id` |
| 400 | `Winners file missing for experiment <id>` | Re-run that champion experiment (writes `data/champion/experiment_<id>_winners.csv`) |
| 409 | `Promotion blocked by policy gate: wape_improvement_too_small` | Candidate did not beat the active champion by `champion.promote_gate.min_wape_improvement_pct`. Improve the model, or pass `?bypass_token=<token>` (audited) |
| 409 | `Promotion blocked by policy gate: coverage_below_min` | Candidate scored fewer DFUs than `min_coverage_frac` of the active champion. Re-run generation; verify staging row count |
| 401 | `Missing API key` | Pass `X-API-Key` header |
| 500 | (server log) | Inspect `model_promotion_log` — the demote/insert transaction is atomic, so on rollback `is_active` should still point at the previous winner. If two rows are active, fire a manual `UPDATE` to fix and open a bug |

The `_log_promotion_to_ledger` call records both rejected and applied gate decisions to the AI decision ledger; query it for the audit trail.

### 9.3 Candidate vs Production discrepancy

**Symptom:** A model shows good accuracy on the Backtest Management tab (data from `fact_candidate_forecast`) but consumers report different numbers from `/forecast/production`.

**Likely causes:**

1. **Different model is promoted.** Check `GET /backtest-management/promotion-status` — `model_id` may not match what you're inspecting.
2. **Stale `plan_version`.** Production keeps the most recently promoted `plan_version` only (DELETE-then-INSERT). If a downstream consumer cached an older `plan_version`, refresh.
3. **Generate not run since last train.** `fact_candidate_forecast` reflects backtest evaluation (historical); `fact_production_forecast` reflects forward inference. If you retrained without re-generating, production is stale. Fix: `make forecast-generate` -> promote.
4. **Champion mode mismatch.** In champion mode, `fact_production_forecast.model_id = 'champion'` but `source_model_id` carries the per-DFU winner. Filter by `source_model_id` when comparing.

Useful diagnostic:

```sql
-- Compare staging vs production for a single DFU
SELECT 'staging' AS src, model_id, forecast_month, forecast_qty
FROM fact_production_forecast_staging
WHERE item_id = '<id>' AND loc = '<loc>'
UNION ALL
SELECT 'prod' AS src, model_id, forecast_month, forecast_qty
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
| Promote single model | `curl -X POST -H "X-API-Key: $KEY" "$BASE/backtest-management/lgbm_cluster/promote"` |
| Promote champion blend | `curl -X POST -H "X-API-Key: $KEY" "$BASE/backtest-management/champion/promote"` |
| Active promotion | `curl -s "$BASE/backtest-management/promotion-status"` |
| Staging summary | `curl -s "$BASE/backtest-management/staging-summary"` |
| Candidate summary | `curl -s "$BASE/backtest-management/candidate-summary"` |
| Quantile forecasts | `make quantile-train VERSION=$(date +%Y-%m)` |
| Blended forecast | `make blended-all` |
| Consensus plan | `make consensus-generate VERSION=$(date +%Y-%m)` |

Source-of-truth files referenced in this section:

- `Makefile` (lines 1071-1096 production-forecast targets)
- `config/forecast_pipeline_config.yaml` (`production_forecast:` block, lines 435-468)
- `scripts/generate_production_forecasts.py`
- `api/routers/forecasting/backtest_management.py`
- `api/routers/production_forecast.py`
- `sql/039_create_production_forecast.sql`
- `sql/121_candidate_forecast_and_promotion.sql`
- `sql/122_create_production_forecast_staging.sql`
