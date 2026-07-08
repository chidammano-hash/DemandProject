# 24 — Candidate Forecast & Model Promotion Workflow

**Status:** Implemented  
**Date:** 2026-04-08  
**Related:** 08-production-forecast.md, 07-champion-selection.md, 12-dual-promotion.md

---

## 1. Overview

This feature introduces a **staged promotion workflow** for forecast models:

```
Train → Generate → Load → Promote
```

All model predictions land in a **candidate forecast table** first. Only the
promoted model's predictions are copied to the **production forecast table**.
This ensures only vetted, explicitly promoted forecasts serve the downstream
planning pipeline.

## 2. Motivation

Previously, the production forecast was generated directly via a job that wrote
to `fact_production_forecast`. There was no staging area to compare models
side-by-side before committing one to production. This workflow adds:

- **Candidate staging** — all models load to `fact_candidate_forecast`
- **Side-by-side comparison** — candidates table holds multiple models at once
- **Explicit promotion** — only one model (or champion selection) is promoted
- **Audit trail** — `model_promotion_log` tracks every promotion/demotion event

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

## 4. API Endpoints

All under `/backtest-management/` prefix.

### 4.1 Read Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/promotion-status` | Returns currently active promoted model |
| GET | `/candidate-summary` | Per-model candidate forecast **counts/accuracy** (no time-series) |

**Per-DFU backtest time-series** (NOT under the `/backtest-management/` prefix —
it lives in the production-forecast router so it sits next to the staging
endpoint):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/candidate?item_id=&loc=&model_id=` | Per-model backtest (past, out-of-sample) predictions for one DFU, grouped by `model_id` — forecast vs `actual_qty` + per-row accuracy. `model_id` optional. Degrades to empty `models` when the table is absent (clean install). |

This is the past-period counterpart to `/forecast/production/staging` (future
forecasts). Implemented in `api/routers/forecasting/production_forecast.py`.

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
| POST | `/{model_id}/promote` | Promote model to production |

### 4.3 Promotion Types

> **Verified against code.** `promote_model()` in
> `api/routers/forecasting/backtest_management.py` does **not** read from
> `fact_candidate_forecast` (that table is inert - see §4.1/§5.1). It copies **staged** forecasts
> from `fact_production_forecast_staging` into `fact_production_forecast`. The `data/champion/
> dfu_assignments.csv` file and its loader, `_load_dfu_assignments()`, are dead code with zero
> callers in the file - champion routing does not use them.

**Single Model** (`model_id = 'lgbm_cluster'`, etc.):
1. Gate check: enforces a minimum WAPE improvement and DFU coverage against the currently active
   champion (bypassable via `bypass_token`); every allow/reject is logged to the AI decision ledger
2. Validates staged rows exist in `fact_production_forecast_staging` for the model
3. Demotes the current active promotion (`model_promotion_log.is_active = FALSE`)
4. Deletes all rows from `fact_production_forecast` (full replace, not append)
5. Copies every staged row for that `model_id` from `fact_production_forecast_staging` →
   `fact_production_forecast`
6. Logs the promotion (row count, `run_id`, etc.)

**Champion** (`model_id = 'champion'`):
1. Validates staged rows exist (any model) in `fact_production_forecast_staging`; demotes the
   current active promotion; deletes all rows from `fact_production_forecast` (champion promotions
   bypass the WAPE gate - gating happens at champion-experiment level instead)
2. Looks up the currently **promoted** `champion_experiment` row (`is_promoted = TRUE`, most
   recently promoted)
3. Reads that experiment's per-run winners file at
   `data/champion/experiment_{champion_experiment_id}_winners.csv` - the source of truth for
   DFU→model routing, written by `run_champion_experiment.py`
4. Builds a **per-month** winners map keyed by `(item_id, loc, startdate)` - a DFU can win a
   *different* model for each forecast month. Ties on `(item_id, loc, startdate)` resolve
   deterministically to the lowest `model_id`, never file order
5. Builds a **per-DFU fallback** map (one model per `(item_id, loc)`) from the same winners file,
   used for any staged month that falls outside the backtest window the winners file covers (the
   winners CSV is backtest-grain; staging spans the full future horizon)
6. Two temp tables - `_dfu_champion` (per-DFU fallback) and `_dfu_champion_month` (per-month
   overrides) - drive a single INSERT: for each staged row, resolve
   `COALESCE(per-month override, per-DFU fallback)` and copy it to `fact_production_forecast` only
   if a staged row exists for that resolved model
7. Coverage check: any staged champion DFU-month whose resolved model has **no** staged row (that
   model's Generate wasn't run, or a `model_id` spelling mismatch) is logged as a warning and
   dropped from production rather than failing the promotion
8. Logs the promotion with `promotion_type = 'champion'` and the resolved `champion_experiment_id`

**Non-champion, non-single-model calls are not supported** - every `model_id` other than
`'champion'` is treated as a single-model promotion.

## 5. Frontend UI

### 5.1 Backtest stage table (Model Tuning → Backtest)

Models are grouped by family (Tree / Foundation / Statistical / Deep Learning) in compact
tables, each with a **Run all** action. The flow is:

```
[Run backtest] → (auto-load) → [Promote]
```

| Step | Action | API Call | Condition |
|------|--------|----------|-----------|
| Run | Run backtest (per model or per group) | `POST /{id}/run` | Trained (tree) or always (foundation/statistical/DL) |
| Auto-load | **Automatic** on run completion — server-side, same job | (none — chained inside the backtest job) | Backtest produced predictions |
| Promote | Copy staged forecast → production | `POST /{id}/promote` | A staged forecast exists (see §08 / Forecast stage) |

> **Auto-load.** When a backtest run completes, `_run_backtest`
> (`common/services/job_state.py`) chains the load (`_auto_load_backtest` →
> `load_backtest_forecasts.py`) **before** marking the run `completed`, so the UI's
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

### 5.2 Promote Champion Button

Appears in the header when 2+ models have loaded candidates. Routes each DFU (per forecast month) to
its winning model from the promoted champion experiment's winners CSV - see §4.3.

### 5.3 Candidates Column

New table column showing loaded candidate DFU count per model.

### 5.4 Item Analysis backtest overlay

The **Item Analysis** chart (`UnifiedChartPanel` / `UnifiedChart`) consumes
`/forecast/candidate` and renders a `backtest_<model>` line per model over the
**historical** window, alongside the existing `staging_<model>` (future) lines.
A **"Backtest"** pill row (mirroring the "Staging" row) toggles the lines per
model and all-at-once. Backtest lines are dotted (`strokeDasharray="1 3"`) and
share each model's color, so a model's past fit and forward forecast read as one
series across the timeline. The merge happens in `ItemAnalysisTab.tsx`
(`mergedFilteredSeries`): candidate months are out-of-sample/historical, so they
land only on existing past points (never the synthesized future points).

This closes the gap where Item Analysis could show a model's **future** staging
forecast but not its **past** backtest predictions — `/candidate-summary` only
exposed counts, with no per-DFU time-series to chart.

## 6. Data Flow

> This diagram reflects the verified `promote_model()` mechanism (§4.3), not the original
> `fact_candidate_forecast` design described in §2-3 - that table is inert (§4.1).

```
┌─────────────┐    Train     ┌─────────────────┐
│ Raw Sales    │───────────►  │ Model Artifacts  │
│ History      │              │ (*.pkl files)    │
└─────────────┘              └────────┬────────┘
                                      │ Generate (production forecast)
                                      ▼
                             ┌───────────────────────┐
                             │ fact_production_        │ ◄── Every model's staged
                             │ forecast_staging         │     forecast lands here
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

**DDL file:** `sql/121_candidate_forecast_and_promotion.sql`

Apply: `docker exec -i <container> psql -U demand -d demand_mvp < sql/121_candidate_forecast_and_promotion.sql`

The migration is idempotent (`IF NOT EXISTS` on all objects).
