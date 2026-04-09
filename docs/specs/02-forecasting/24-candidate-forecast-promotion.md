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
| GET | `/candidate-summary` | Per-model candidate forecast statistics |

### 4.2 Write Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/{model_id}/promote` | Promote model to production |

### 4.3 Promotion Types

**Single Model** (`model_id = 'lgbm_cluster'`, etc.):
1. Validates candidate rows exist for the model
2. Demotes current active promotion
3. Copies all candidate rows for this model → `fact_production_forecast`
4. Marks candidates as promoted
5. Logs in `model_promotion_log`

**Champion** (`model_id = 'champion'`):
1. Loads DFU-level assignments from `data/champion/dfu_assignments.csv`
2. Falls back to best-accuracy model per DFU from candidates
3. For each DFU, picks the assigned model's forecast from candidates
4. Copies matched rows → `fact_production_forecast`
5. Logs with `promotion_type = 'champion'`

## 5. Frontend UI

### 5.1 Model Readiness Table (ForecastPanel Step 1)

Each model row displays a 4-step action pipeline:

```
[Train] → [Generate] → [Load] → [Promote]
```

| Step | Action | API Call | Condition |
|------|--------|----------|-----------|
| Train | Train on full history | `POST /{id}/train` | Tree models only |
| Generate | Run backtest | `POST /{id}/run` | Trained (tree) or always (foundation) |
| Load | Load to candidates | `POST /{id}/load` | Predictions exist on disk |
| Promote | Copy to production | `POST /{id}/promote` | Candidates loaded in DB |

**State indicators:**
- Gray badge: Not available (prerequisite incomplete)
- Outline button: Ready to execute
- Spinner: Running
- Green check badge: Completed
- Crown badge: Currently promoted to production

### 5.2 Promote Champion Button

Appears in the header when 2+ models have loaded candidates. Uses DFU
assignments to select the best model per DFU.

### 5.3 Candidates Column

New table column showing loaded candidate DFU count per model.

## 6. Data Flow

```
┌─────────────┐    Train     ┌─────────────────┐
│ Raw Sales    │───────────►  │ Model Artifacts  │
│ History      │              │ (*.pkl files)    │
└─────────────┘              └────────┬────────┘
                                      │ Generate (backtest)
                                      ▼
                             ┌─────────────────┐
                             │ Predictions CSV  │
                             │ (on disk)        │
                             └────────┬────────┘
                                      │ Load
                                      ▼
                             ┌─────────────────────┐
                             │ fact_candidate_      │ ◄── All models land here
                             │ forecast             │
                             └────────┬────────────┘
                                      │ Promote
                                      ▼
                             ┌─────────────────────┐
                             │ fact_production_     │ ◄── Only promoted model
                             │ forecast             │
                             └─────────────────────┘
```

## 7. Migration

**DDL file:** `sql/121_candidate_forecast_and_promotion.sql`

Apply: `docker exec -i <container> psql -U demand -d demand_mvp < sql/121_candidate_forecast_and_promotion.sql`

The migration is idempotent (`IF NOT EXISTS` on all objects).
