# Champion Experimentation Studio

> A full experiment lifecycle for champion selection strategies — create, run, compare, promote config, and load results — integrated as a sub-tab within the Model Tuning tab.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Model Tuning > Champion sub-tab |
| **Key Files** | `api/routers/forecasting/champion_experiments.py`, `frontend/src/tabs/champion/*.tsx`, `sql/102_champion_experiments.sql`, `common/ml/champion_strategies.py` |

| | |
|---|---|
| **API Prefix** | `/champion-experiments` |
| **Router** | `api/routers/forecasting/champion_experiments.py` |
| **Frontend** | `frontend/src/api/queries/champion-experiments.ts`, `frontend/src/tabs/champion/*.tsx` |
| **Tests** | `tests/api/test_champion_experiments.py` (31 tests) |
| **Depends On** | Feature 46 (Unified Model Tuning), Feature 44 (Resilient Jobs), `common/ml/champion_strategies.py` |

---

## Table of Contents

1. [Problem](#problem)
2. [Solution Overview](#solution-overview)
3. [Database Schema](#database-schema)
4. [API Design](#api-design)
5. [Frontend Components](#frontend-components)
6. [Champion Strategies](#champion-strategies)
7. [2-Stage Promotion](#2-stage-promotion)
8. [Configuration Templates](#configuration-templates)
9. [Job Pipeline](#job-pipeline)
10. [Test Strategy](#test-strategy)
11. [Implementation Sequence](#implementation-sequence)

---

## Problem

The existing champion selection pipeline (`scripts/run_champion_selection.py`) is a one-shot fire-and-forget operation. Users cannot:

1. **Experiment with strategies** — No way to test expanding vs rolling vs decay vs ensemble vs meta_learner side-by-side
2. **Compare results** — No infrastructure for comparing champion accuracy, ceiling accuracy, gap, or model distribution between configurations
3. **Track history** — No persistent record of past champion selection runs or their parameters
4. **Stage promotion** — The current workflow either writes config OR loads results, with no staged approach and no audit trail

---

## Solution Overview

| Component | Purpose |
|-----------|---------|
| `champion_experiment` (5 tables) | Persistent experiment lifecycle with per-lag and per-month breakdowns |
| `/champion-experiments` (15 endpoints) | Full CRUD + comparison + 2-stage promotion |
| Champion sub-tab (4 components) | KPI dashboard, experiment table, builder, comparison panel, promote modal |
| 9 templates | Pre-built strategy configurations |

The Champion sub-tab is **not gated by the model selector** (unlike algorithm tuning) because champion selection operates across all 3 models simultaneously.

---

## Database Schema

**5 tables** defined in `sql/102_champion_experiments.sql`:

### champion_experiment (28 columns)

| Group | Columns |
|-------|---------|
| Lifecycle | `experiment_id` (PK SERIAL), `label`, `notes`, `template_id`, `status`, `created_at`, `started_at`, `completed_at`, `runtime_seconds`, `job_id` |
| Input Config | `strategy` (text), `strategy_params` (JSONB), `meta_learner_params` (JSONB), `models` (JSONB), `metric`, `lag_mode`, `min_sku_rows` |
| Results | `champion_accuracy`, `ceiling_accuracy`, `gap_bps`, `n_champions`, `n_dfu_months`, `model_distribution` (JSONB) |
| Promotion | `is_promoted`, `promoted_at`, `is_results_promoted`, `results_promoted_at`, `results_promote_job_id` |

### champion_experiment_lag
Per-execution-lag breakdown: `experiment_id` (FK), `exec_lag`, `champion_accuracy`, `ceiling_accuracy`, `gap_bps`, `n_dfu_months`, `model_distribution` (JSONB).

### champion_experiment_month
Per-month breakdown: `experiment_id` (FK), `month_start`, `champion_accuracy`, `ceiling_accuracy`, `gap_bps`, `n_champions`, `model_distribution` (JSONB).

### champion_experiment_comparison
Cached pairwise comparisons: `experiment_a_id` (FK), `experiment_b_id` (FK), `overall_comparison` (JSONB), `per_lag_comparison` (JSONB), `per_month_comparison` (JSONB), `model_dist_comparison` (JSONB), `config_diffs` (JSONB).

### champion_promotion_log
Audit trail: `experiment_id` (FK), `promoted_at`, `promoted_by`, `previous_experiment_id`, `strategy`, `champion_accuracy`, `config_snapshot` (JSONB).

---

## API Design

15 endpoints on prefix `/champion-experiments`:

| # | Method | Path | Status | Notes |
|---|--------|------|--------|-------|
| 1 | GET | `/` | 200 | List experiments (paginated, status filter) |
| 2 | GET | `/templates` | 200 | Load from YAML |
| 3 | GET | `/promoted` | 200 | Current promoted experiment |
| 4 | GET | `/promotions` | 200 | Audit trail |
| 5 | GET | `/compare?a_id=&b_id=` | 200 | Compare two experiments (cached) |
| 6 | GET | `/{id}` | 200 | Single experiment detail |
| 7 | GET | `/{id}/lags` | 200 | Per-execution-lag breakdown |
| 8 | GET | `/{id}/months` | 200 | Per-month breakdown |
| 9 | GET | `/{id}/logs` | 200 | Incremental log streaming |
| 10 | POST | `/` | 202 | Create + launch async job |
| 11 | POST | `/{id}/promote` | 200 | Stage 1: write config |
| 12 | POST | `/{id}/promote-results` | 201 | Stage 2: load results |
| 13 | GET | `/{id}/promote-results/status` | 200 | Poll Stage 2 |
| 14 | POST | `/{id}/cancel` | 200 | Cancel running/queued |
| 15 | DELETE | `/{id}` | 200 | Delete (refuse if running) |

Static paths (`/templates`, `/promoted`, `/promotions`, `/compare`) are defined before `/{id}` to avoid path shadowing.

---

## Frontend Components

### ChampionExperimentsPanel
Main panel with:
- **KPI cards**: Best Champion Accuracy, Production Strategy, Gap to Ceiling (bps), Active Runs
- **Status filter** dropdown + "New Experiment" button
- **Experiment table**: ID, Label, Strategy, Accuracy%, Ceiling%, Gap, Status, Duration, Actions
- **Row click** → baseline/candidate selection for comparison
- **Log viewer** slide-over panel
- **Comparison panel** (right side when two selected)

### ChampionExperimentBuilder
Full-screen modal:
- Label + Notes inputs
- Template radio buttons (9 + custom)
- Strategy dropdown with dynamic params form per strategy
- Models checkboxes (min 2 required)
- Metric / Lag mode / min_sku_rows selectors

### ChampionComparisonPanel
Side-by-side comparison:
- Verdict badge (A Better / B Better / Mixed)
- Overall metrics delta (champion accuracy, ceiling accuracy, gap)
- Per-lag accuracy table
- Model distribution comparison table
- Config differences table

### ChampionPromoteModal
2-stage promotion dialog:
- Stage 1: Promote config to `forecast_pipeline_config.yaml` champion section (with backup)
- Stage 2: Run champion selection → load results to `fact_external_forecast_monthly`
- Progress polling for Stage 2

---

## Champion Strategies

8 strategies from `common/ml/champion_strategies.py` `STRATEGY_REGISTRY`:

| Strategy | Key Params | Description |
|----------|-----------|-------------|
| **expanding** | `min_prior_months` | All historical data, growing window |
| **rolling** | `window_months`, `min_prior_months` | Fixed-width sliding window |
| **decay** | `decay_factor`, `min_prior_months` | Exponential time-decay weighting |
| **ensemble** | `top_k`, `weight_method` | Blend top-K models |
| **meta_learner** | `model_type`, `n_estimators`, `max_depth`, `test_months` | ML model to predict best algorithm |
| **hybrid_warmup** | `warmup_strategy`, `warmup_window`, `warmup_min_prior`, `primary_strategy`, `primary_top_k` | Fast-adapting strategy for warm-up months, then switches to ensemble/expanding once enough history accumulates |
| **adaptive_ensemble** | `min_k`, `max_k`, `spread_threshold`, `weight_method` | Varies top-K per DFU-month based on model WAPE spread |
| **ensemble_rolling** | `top_k`, `window_months`, `weight_method` | Blend top-K models using rolling-window WAPE instead of expanding |

All strategies use `shift(exec_lag+1)` to prevent data leakage.

---

## 2-Stage Promotion

### Stage 1: Promote Config
1. Verify experiment is completed
2. Backup `config/forecast_pipeline_config.yaml` → `.bak.<experiment_id>`
3. Write new champion strategy, params, metric, lag_mode, min_sku_rows to `champion` section
4. Clear previous `is_promoted` flags
5. Set `is_promoted=TRUE`, insert into `champion_promotion_log`

### Stage 2: Load Results
1. Submit `champion_results_load` job via `JobManager`
2. Job runs `scripts/run_champion_selection.py` (reads updated `forecast_pipeline_config.yaml` champion section)
3. On completion: sets `is_results_promoted=TRUE`
4. Frontend polls via `/promote-results/status`

---

## Configuration Templates

9 templates in `config/champion_experiment_templates.yaml`:

| Template | Strategy | Key Params |
|----------|----------|-----------|
| production_baseline | (from live config) | source: model_competition_config |
| expanding_conservative | expanding | min_prior_months=5 |
| rolling_6m | rolling | window_months=6 |
| rolling_3m | rolling | window_months=3 |
| decay_090 | decay | decay_factor=0.90 |
| decay_095 | decay | decay_factor=0.95 |
| ensemble_top3_inverse | ensemble | top_k=3, weight_method=inverse_wape |
| ensemble_top2_equal | ensemble | top_k=2, weight_method=equal |
| meta_learner_rf | meta_learner | n_estimators=200, max_depth=15 |

---

## Job Pipeline

Two job types registered in `common/services/job_registry.py`:

| Job Type | Group | Callable | Script |
|----------|-------|----------|--------|
| `champion_experiment` | champion | `_run_champion_experiment` | `scripts/run_champion_experiment.py` |
| `champion_results_load` | champion | `_run_champion_results_load` | `scripts/run_champion_selection.py` |

The experiment runner (`scripts/run_champion_experiment.py`) reuses:
- `STRATEGY_REGISTRY`, `compute_strategy_accuracy()`, `compute_ceiling()` from `common/ml/champion_strategies.py`
- `load_monthly_errors_df()`, `load_dfu_features()` from `scripts/run_champion_selection.py`

---

## Test Strategy

31 backend tests in `tests/api/test_champion_experiments.py`:

| Group | Tests | What's Covered |
|-------|-------|---------------|
| List | 2 | Basic list + status filter |
| Detail | 2 | Found + not found |
| Create | 3 | Success + invalid strategy + insufficient models |
| Lags | 1 | Per-lag breakdown |
| Months | 1 | Per-month breakdown |
| Logs | 2 | With job + no job |
| Compare | 3 | Cached + same ID + not found |
| Promote | 3 | Success + not completed + not found |
| Promote Results | 3 | Success + not promoted + already loaded |
| Results Status | 1 | Poll status |
| Cancel | 2 | Success + completed fails |
| Delete | 3 | Success + running fails + promoted fails |
| Templates | 2 | Loaded + missing file |
| Promoted | 2 | Found + none |
| Promotions | 1 | Audit trail |

---

## Implementation Sequence

1. `sql/102_champion_experiments.sql` — DB schema
2. `config/champion_experiment_templates.yaml` — Templates
3. `scripts/run_champion_experiment.py` — Async runner
4. `common/services/job_state.py` + `job_registry.py` — Job registration
5. `api/routers/forecasting/champion_experiments.py` — API router
6. `api/main.py` + `frontend/vite.config.ts` + `index.ts` — Integration
7. `frontend/src/api/queries/champion-experiments.ts` — Query layer
8. `frontend/src/tabs/champion/*.tsx` — 4 UI components
9. `frontend/src/tabs/ModelTuningTab.tsx` — Sub-tab integration
10. `tests/api/test_champion_experiments.py` — Backend tests
11. Documentation updates
