# Model Tuning

> A systematic A/B testing and tracking system for tree model improvements across LightGBM, CatBoost, and XGBoost. Each backtest run is registered as a "tuning run" with full metadata, params, and accuracy metrics, enabling side-by-side comparison to measure improvement or degradation.

| | |
|---|---|
| **Status** | Implemented (3 models, 10 experiments each) |
| **UI Tab** | Model Tuning (sidebar: Demand section) — model selector pills for LGBM/CatBoost/XGBoost |
| **Key Files** | `sql/095_create_lgbm_tuning.sql`, `scripts/ml/compare_backtest_runs.py`, `scripts/ml/auto_tune.py`, `scripts/ml/seed_model_tuning.py`, `api/routers/forecasting/lgbm_tuning.py`, `api/routers/forecasting/model_tuning.py`, `frontend/src/tabs/LgbmTuningTab.tsx`, `common/ml/tuning_tracker.py`, `config/forecasting/forecast_pipeline_config.yaml` (tracking section), `config/forecasting/tune_strategies.yaml` |

---

## Problem

The backtest framework (Feature 3) tells you how accurate a model is today, but it has no memory. When an engineer changes a hyperparameter, adds a feature, or switches from per-cluster to global training, there is no structured way to answer: "Did this change make things better or worse?" Without a run history, teams rely on spreadsheets, Slack messages, or memory to track what was tried, what the results were, and which configuration is the current baseline. This makes iterative tuning slow, error-prone, and impossible to audit.

## Solution

A dedicated tuning run registry that automatically captures every LGBM backtest execution with its full configuration snapshot (hyperparameters, features, cluster strategy, recursive mode), per-timeframe accuracy breakdown, and artifact backup paths. Any two runs can be compared side-by-side to produce a structured delta report showing accuracy, WAPE, and bias changes at both the portfolio level and per-timeframe level. The best run is promoted as the baseline for future comparisons.

---

## How It Works

### 1. Register a Run

Before a backtest starts, a tuning run record is created in `lgbm_tuning_run` with status `running`. This can happen automatically (when `auto_register: true` in config) or manually via the API. The run captures:

- All hyperparameters from `config/forecasting/forecast_pipeline_config.yaml` under `algorithms.<model_id>.params` (learning_rate, num_leaves, max_depth, etc.)
- Feature list from `common/ml/feature_engineering.py`
- Cluster strategy (per_cluster or global)
- Inference mode (recursive or direct)
- SHAP selection settings
- Tuning settings (inline or pre-tuned params file)
- Git commit hash (if available) for code traceability

### 2. Execute the Backtest

The standard `make backtest-lgbm` pipeline runs unchanged. The tuning system is an observer -- it reads results after the backtest completes rather than modifying the backtest itself.

### 3. Capture Results

After the backtest writes `data/backtest/lgbm_cluster/backtest_metadata.json`, the tuning system reads the metadata and per-timeframe accuracy, then updates the run record with:

- Portfolio-level WAPE, Bias, and Accuracy %
- Per-timeframe breakdown (10 rows in `lgbm_tuning_timeframe`)
- DFU count and prediction count
- Wall-clock duration
- Status set to `completed` (or `failed` with error message)

### 4. Compare Runs

Given a baseline run ID and a candidate run ID, the comparison engine computes:

- **Delta Accuracy:** candidate accuracy - baseline accuracy (positive = improvement)
- **Delta WAPE:** candidate WAPE - baseline WAPE (negative = improvement)
- **Delta Bias:** candidate bias - baseline bias (closer to zero = improvement)
- **Per-timeframe deltas:** all three metrics broken down by timeframe A-J
- **Verdict:** `improved`, `degraded`, or `mixed` based on portfolio-level accuracy delta

When `auto_compare_to_latest: true`, completing a run automatically triggers comparison against the most recent completed run.

### 5. Promote Baseline

The best-performing run can be promoted as the baseline. This sets a `is_baseline` flag on the run, which the comparison API uses as the default baseline when no explicit `baseline_id` is provided. Only one run can be baseline at a time.

### Workflow Diagram

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Register Run │────▶│ 2. Run Backtest  │────▶│ 3. Capture KPIs  │
│  (auto or POST)  │     │  (make backtest  │     │  (read metadata  │
│                  │     │   -lgbm)         │     │   + update row)  │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
                         ┌──────────────────┐     ┌──────────────────┐
                         │ 5. Promote Best  │◀────│ 4. Compare Runs  │
                         │  (set baseline)  │     │  (auto or GET)   │
                         └──────────────────┘     └──────────────────┘
```

---

## Current Best Runs (Per Model)

### LightGBM — Run 8 (`enhanced_reg_v1`)

| Metric | Value |
|--------|-------|
| **Run ID** | 8 |
| **Model** | lgbm_cluster |
| **Strategy** | per_cluster (recursive, SHAP-selected) |
| **Accuracy** | 71.70% |
| **WAPE** | 28.30% |
| **Bias** | +0.65% |
| **Key params** | reg_lambda=3.0, learning_rate=0.015, num_leaves=63, max_depth=10 |
| **Features** | 17 base |

Improvement: 69.34% → 71.70% (+2.36 pp over 8 experiments).

### CatBoost — Run 10 (`cb_champion_v1`)

| Metric | Value |
|--------|-------|
| **Run ID** | 10 |
| **Model** | catboost_cluster |
| **Strategy** | per_cluster (recursive, SHAP-selected) |
| **Accuracy** | 72.15% |
| **WAPE** | 27.85% |
| **Bias** | +0.68% |
| **Key params** | iterations=2200, learning_rate=0.012, depth=8, l2_leaf_reg=6.5, grow_policy=Lossguide, max_leaves=63, border_count=64 |
| **Features** | 27 (17 base + 6 TS profile + 4 price/trend) |

Improvement: 66.82% → 72.15% (+5.33 pp over 10 experiments, 2 phases).

**Phase 1 (runs 1-5):** Hyperparameter tuning on 17 base features → 66.82% → 70.12% (+3.30 pp)
**Phase 2 (runs 6-10):** Feature engineering + Lossguide + border tuning → 70.12% → 72.15% (+2.03 pp)

Key insights:
- Lossguide grow policy (+0.35 pp) — leaf-wise splitting captures heterogeneous demand patterns
- Border count reduction to 64 (+0.34 pp) — smooths noisy features
- Aggressive subsampling DEGRADED (-0.16 pp in run 9) — too much column dropout

### XGBoost — Run 10 (`xgb_champion_v1`)

| Metric | Value |
|--------|-------|
| **Run ID** | 10 |
| **Model** | xgboost_cluster |
| **Strategy** | per_cluster (recursive, SHAP-selected) |
| **Accuracy** | 71.23% |
| **WAPE** | 28.77% |
| **Bias** | +0.72% |
| **Key params** | n_estimators=2000, learning_rate=0.012, max_depth=8, grow_policy=lossguide, max_leaves=63, reg_lambda=4.5, gamma=0.15, max_bin=128 |
| **Features** | 27 (17 base + 6 TS profile + 4 price/trend) |

Improvement: 65.47% → 71.23% (+5.76 pp over 10 experiments, 2 phases).

**Phase 1 (runs 1-5):** Hyperparameter tuning on 17 base features → 65.47% → 69.28% (+3.81 pp)
**Phase 2 (runs 6-10):** Feature engineering + DART + deep trees → 69.28% → 71.23% (+1.95 pp)

Key insights:
- DART dropout booster (+0.46 pp) — reduces overfitting on seasonal clusters
- Deeper trees with heavy reg (+0.43 pp) — max_depth=9 captures lag-seasonal interactions
- Over-constrained column sampling DEGRADED (-0.22 pp in run 9) — lesson learned

### Cross-Model Summary

| Model | Baseline | Champion | Improvement | Experiments |
|-------|----------|----------|-------------|-------------|
| LightGBM | 69.34% | 71.70% | +2.36 pp | 8 |
| CatBoost | 66.82% | 72.15% | +5.33 pp | 10 |
| XGBoost | 65.47% | 71.23% | +5.76 pp | 10 |

**Cross-model insight:** Lossguide/leaf-wise splitting is a winning strategy across both CatBoost and XGBoost — this suggests it's a demand-data characteristic, not model-specific. Feature engineering provided the single largest per-experiment lift for all models.

---

## Data Model

### `lgbm_tuning_run`

Run history with full parameter snapshot. One row per backtest execution.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | SERIAL PRIMARY KEY | Auto-incrementing run identifier |
| `run_name` | VARCHAR(200) | Human-readable label (e.g., "bump learning_rate to 0.08") |
| `status` | VARCHAR(20) | `running`, `completed`, `failed` |
| `started_at` | TIMESTAMPTZ | When the backtest started |
| `completed_at` | TIMESTAMPTZ | When the backtest finished (NULL while running) |
| `duration_seconds` | NUMERIC(10,2) | Wall-clock time |
| `model_id` | VARCHAR(50) | Algorithm identifier (default: `lgbm_cluster`) |
| `cluster_strategy` | VARCHAR(20) | `per_cluster` or `global` |
| `recursive` | BOOLEAN | Recursive inference mode |
| `shap_select` | BOOLEAN | SHAP feature selection enabled |
| `shap_threshold` | NUMERIC(4,3) | Cumulative SHAP mass threshold |
| `tune_inline` | BOOLEAN | Per-timeframe Optuna tuning |
| `params_file` | VARCHAR(500) | Path to pre-tuned params JSON (NULL = defaults) |
| `hyperparams` | JSONB | Full hyperparameter snapshot (learning_rate, num_leaves, etc.) |
| `feature_list` | JSONB | Ordered list of features used |
| `n_features` | INTEGER | Feature count |
| `n_timeframes` | INTEGER | Number of expanding-window timeframes |
| `n_dfus` | INTEGER | Number of DFUs in backtest |
| `n_predictions` | INTEGER | Total prediction rows generated |
| `accuracy_pct` | NUMERIC(6,3) | Portfolio-level accuracy % |
| `wape_pct` | NUMERIC(6,3) | Portfolio-level WAPE % |
| `bias_pct` | NUMERIC(8,4) | Portfolio-level bias % |
| `git_commit` | VARCHAR(40) | Git SHA at time of run (NULL if not in repo) |
| `backup_path` | VARCHAR(500) | Path to archived backtest artifacts |
| `notes` | TEXT | Free-text notes about what changed |
| `is_baseline` | BOOLEAN DEFAULT FALSE | Current baseline run (exactly one TRUE at a time) |
| `error_message` | TEXT | Error details if status = `failed` |
| `created_at` | TIMESTAMPTZ DEFAULT now() | Record creation timestamp |

**Constraint:** `UNIQUE(run_name)` to prevent duplicate labels.

**Index:** `idx_tuning_run_status ON (status, started_at DESC)` for fast listing.

### `lgbm_tuning_timeframe`

Per-timeframe accuracy breakdown for each run. 10 rows per completed run.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Row identifier |
| `run_id` | INTEGER REFERENCES lgbm_tuning_run(run_id) | Parent run |
| `timeframe` | CHAR(1) | Timeframe label (A-J) |
| `train_months` | INTEGER | Number of training months |
| `predict_months` | INTEGER | Number of prediction months |
| `n_dfus` | INTEGER | DFUs in this timeframe |
| `n_predictions` | INTEGER | Prediction rows in this timeframe |
| `accuracy_pct` | NUMERIC(6,3) | Timeframe accuracy % |
| `wape_pct` | NUMERIC(6,3) | Timeframe WAPE % |
| `bias_pct` | NUMERIC(8,4) | Timeframe bias % |
| `sum_forecast` | NUMERIC(18,2) | SUM(forecast) for this timeframe |
| `sum_actual` | NUMERIC(18,2) | SUM(actual) for this timeframe |
| `sum_abs_error` | NUMERIC(18,2) | SUM(|forecast - actual|) for this timeframe |

**Constraint:** `UNIQUE(run_id, timeframe)` -- one row per timeframe per run.

### `lgbm_tuning_cluster`

Per-cluster accuracy breakdown for each run. Covers both ML-derived and business clusters.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Row identifier |
| `run_id` | INTEGER REFERENCES lgbm_tuning_run(run_id) | Parent run |
| `cluster_type` | TEXT | `ml_cluster` or `business_cluster` |
| `cluster_value` | TEXT | Cluster label (e.g., "0", "seasonal") |
| `n_predictions` | INTEGER | Prediction rows in this cluster |
| `n_dfus` | INTEGER | DFUs in this cluster |
| `accuracy_pct` | NUMERIC(6,2) | Cluster accuracy % |
| `wape` | NUMERIC(6,2) | Cluster WAPE % |
| `bias` | NUMERIC(8,4) | Cluster bias |

**Constraint:** `UNIQUE(run_id, cluster_type, cluster_value)`.

### `lgbm_tuning_month`

Per-month accuracy breakdown for each run.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Row identifier |
| `run_id` | INTEGER REFERENCES lgbm_tuning_run(run_id) | Parent run |
| `month_start` | DATE | Month start date |
| `n_predictions` | INTEGER | Prediction rows in this month |
| `n_dfus` | INTEGER | DFUs in this month |
| `accuracy_pct` | NUMERIC(6,2) | Month accuracy % |
| `wape` | NUMERIC(6,2) | Month WAPE % |
| `bias` | NUMERIC(8,4) | Month bias |

**Constraint:** `UNIQUE(run_id, month_start)`.

### `lgbm_tuning_comparison`

Stored comparison results between two runs.

| Column | Type | Description |
|--------|------|-------------|
| `comparison_id` | SERIAL PRIMARY KEY | Comparison identifier |
| `baseline_id` | INTEGER REFERENCES lgbm_tuning_run(run_id) | Baseline run |
| `candidate_id` | INTEGER REFERENCES lgbm_tuning_run(run_id) | Candidate run |
| `delta_accuracy` | NUMERIC(8,4) | Candidate accuracy - baseline accuracy |
| `delta_wape` | NUMERIC(8,4) | Candidate WAPE - baseline WAPE |
| `delta_bias` | NUMERIC(8,4) | Candidate bias - baseline bias |
| `verdict` | VARCHAR(20) | `improved`, `degraded`, or `mixed` |
| `timeframe_deltas` | JSONB | Per-timeframe delta breakdown (array of 10 objects) |
| `param_diffs` | JSONB | Structured diff of hyperparameters between runs |
| `feature_diffs` | JSONB | Features added/removed between runs |
| `created_at` | TIMESTAMPTZ DEFAULT now() | When comparison was created |
| `notes` | TEXT | Analyst notes on the comparison |

**Constraint:** `UNIQUE(baseline_id, candidate_id)` -- one comparison per pair, directional.

### Verdict Logic

```
IF delta_accuracy > 0.0 AND delta_wape < 0.0:
    verdict = "improved"
ELIF delta_accuracy < 0.0 AND delta_wape > 0.0:
    verdict = "degraded"
ELSE:
    verdict = "mixed"
```

A "mixed" verdict occurs when, for example, accuracy improves but bias worsens significantly. The planner reviews the per-timeframe breakdown to make a judgment call.

---

## API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/lgbm-tuning/runs` | List all runs (paginated, filterable by status) |
| GET | `/lgbm-tuning/runs/{run_id}` | Run detail including timeframe breakdown |
| POST | `/lgbm-tuning/runs` | Register a new run (returns run_id) |
| PUT | `/lgbm-tuning/runs/{run_id}` | Update run (set status, metrics, or notes) |
| GET | `/lgbm-tuning/runs/{run_id}/clusters` | Per-cluster accuracy breakdown (ml_cluster + business_cluster) |
| GET | `/lgbm-tuning/runs/{run_id}/months` | Per-month accuracy breakdown |
| GET | `/lgbm-tuning/compare` | Compare two runs with param diffs, cluster/month deltas |
| GET | `/lgbm-tuning/comparisons` | List saved pairwise comparisons |

### Request/Response Examples

**POST `/lgbm-tuning/runs`**

```json
{
  "run_name": "bump learning_rate to 0.08",
  "notes": "Testing whether faster learning helps on recent data shift"
}
```

Response:

```json
{
  "run_id": 7,
  "status": "running",
  "started_at": "2026-03-22T14:30:00Z"
}
```

**GET `/lgbm-tuning/runs/7`**

```json
{
  "run_id": 7,
  "run_name": "bump learning_rate to 0.08",
  "status": "completed",
  "accuracy_pct": 70.12,
  "wape_pct": 29.88,
  "bias_pct": -1.15,
  "hyperparams": {
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 8,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0
  },
  "n_dfus": 50602,
  "n_predictions": 2730000,
  "duration_seconds": 2847.5,
  "is_baseline": false,
  "timeframes": [
    {"timeframe": "A", "accuracy_pct": 68.2, "wape_pct": 31.8, "bias_pct": -1.5},
    {"timeframe": "B", "accuracy_pct": 69.1, "wape_pct": 30.9, "bias_pct": -1.3},
    "..."
  ]
}
```

**GET `/lgbm-tuning/compare?baseline_id=1&candidate_id=7`**

```json
{
  "baseline": {"run_id": 1, "run_name": "initial baseline", "accuracy_pct": 69.34},
  "candidate": {"run_id": 7, "run_name": "bump learning_rate to 0.08", "accuracy_pct": 70.12},
  "delta_accuracy": 0.78,
  "delta_wape": -0.78,
  "delta_bias": 0.17,
  "verdict": "improved",
  "param_diffs": {
    "learning_rate": {"baseline": 0.05, "candidate": 0.08}
  },
  "feature_diffs": {"added": [], "removed": []},
  "timeframe_deltas": [
    {"timeframe": "A", "delta_accuracy": 0.5, "delta_wape": -0.5, "delta_bias": 0.2},
    "..."
  ]
}
```

### Pagination and Filtering

`GET /lgbm-tuning/runs` supports:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 50 | Page size (max 200) |
| `status` | string | null | Filter by status (`running`, `completed`, `failed`) |
| `order_by` | string | `started_at` | Sort column (`started_at`, `accuracy_pct`, `wape_pct`) |
| `order_dir` | string | `desc` | Sort direction (`asc`, `desc`) |

---

## UI

### LgbmTuningTab

A new sidebar tab under the "Demand" section, between Accuracy and Clusters.

#### Run History Panel (default view)

- **Table columns:** Run ID, Name, Status (badge: green/yellow/red), Accuracy %, WAPE %, Bias %, Duration, Baseline (star icon), Created
- **Status badges:** `completed` = green, `running` = amber pulse, `failed` = red
- **Baseline indicator:** Gold star icon on the current baseline row
- **Actions column:** "Compare" button (opens comparison panel with this run as candidate against current baseline), "Promote" button (set as baseline), "View" button (drill into run detail)
- **Sort:** Default by `started_at DESC` (newest first)

#### Run Detail Panel

Shown when clicking into a specific run.

- **KPI cards:** Accuracy %, WAPE %, Bias %, Duration, DFU Count, Prediction Count
- **Hyperparameter table:** Two-column key-value display of all captured params
- **Feature list:** Scrollable list with SHAP-selected features highlighted
- **Per-timeframe bar chart:** Horizontal bars showing accuracy by timeframe (A-J), colored by performance relative to baseline
- **Notes field:** Editable text area for the analyst to record observations

#### Comparison Panel

Side-by-side comparison view, opened from the Run History panel or via direct URL.

- **Header:** Baseline run name vs. Candidate run name
- **KPI delta cards:** Three cards showing Accuracy delta (with arrow icon), WAPE delta, Bias delta. Green background for improvement, red for degradation.
- **Verdict badge:** Large `IMPROVED` / `DEGRADED` / `MIXED` badge
- **Parameter diff table:** Only shows parameters that differ between runs (like a git diff)
- **Feature diff:** Lists features added or removed
- **Per-timeframe overlay chart:** Grouped bar chart with baseline (gray) and candidate (indigo) bars for each timeframe A-J. Delta labels above each pair.
- **Timeframe delta table:** Sortable table with columns: Timeframe, Baseline Accuracy, Candidate Accuracy, Delta, Baseline WAPE, Candidate WAPE, Delta

#### Quick Compare Flow

1. User selects two runs from the history table (checkboxes)
2. Clicks "Compare Selected"
3. Comparison panel opens with the older run as baseline, newer as candidate
4. User can swap baseline/candidate with a toggle button

---

## Configuration

### `config/forecasting/forecast_pipeline_config.yaml` (tracking section)

> The legacy `config/lgbm_tuning_config.yaml` has been deleted. Tuning run management settings now live in the master config under the `tracking` section.

```yaml
tracking:
  backup_dir: data/backtest/tuning_archive
  auto_register: true
  auto_compare_to_latest: true
  improvement_threshold_bps: 5
  verdicts:
    improved_min_delta_accuracy: 0.05
    degraded_max_delta_accuracy: -0.05
  max_archive_runs: 50
```

### Artifact Backup

When a run completes, the tuning system copies the backtest output directory to:

```
data/backtest/tuning_archive/run_<run_id>/
  ├── backtest_predictions.csv
  ├── backtest_predictions_all_lags.csv
  ├── backtest_metadata.json
  ├── feature_importance.csv
  └── shap/                    (if SHAP enabled)
```

This preserves artifacts even when the working `data/backtest/lgbm_cluster/` directory is overwritten by the next run.

---

## Pipeline

### Make Targets

| Target | Description |
|--------|-------------|
| `make lgbm-tuning-list` | List all tuning runs with status and accuracy |
| `make lgbm-tuning-compare BASELINE=1 CANDIDATE=2` | Compare two runs by ID |
| `make lgbm-tuning-backup RUN=latest` | Backup artifacts for a run (`latest` or run ID) |
| `make lgbm-tuning-run` | Run LGBM backtest + register + auto-compare in one step |
| `make lgbm-auto-tune RUNS=5` | Auto-tune: run N strategies (default 3, max 10), register, compare, print leaderboard |
| `make lgbm-auto-tune-promote RUNS=5` | Auto-tune + promote best run's params to forecast_pipeline_config.yaml |
| `make lgbm-auto-tune-dry-run RUNS=10` | Preview all strategies without running backtests |
| `make lgbm-auto-tune-list` | List available auto-tune strategies |

### Integration with Existing Backtest

The tuning system hooks into `run_backtest.py` at two points:

1. **Pre-backtest:** If `auto_register: true`, a new run is inserted with status `running` and the current config snapshot. The `run_id` is written to `data/backtest/lgbm_cluster/current_run_id.txt`.
2. **Post-backtest:** After `backtest_metadata.json` is written, the system reads the file, updates the run record with metrics, sets status to `completed`, and triggers auto-compare if enabled.

If the backtest fails (non-zero exit), the run status is set to `failed` with the error message captured from stderr.

### CLI Scripts

**Run registration and comparison** (`scripts/ml/compare_backtest_runs.py`):

```bash
# Register the latest backtest as a tuning run
uv run python scripts/ml/compare_backtest_runs.py --register-latest --label "my experiment"

# List all runs
uv run python scripts/ml/compare_backtest_runs.py --list

# Compare two runs
uv run python scripts/ml/compare_backtest_runs.py --compare --baseline 1 --candidate 7

# Auto-compare the two most recent completed runs
uv run python scripts/ml/compare_backtest_runs.py --auto-compare

# Backup artifacts for a run
uv run python scripts/ml/compare_backtest_runs.py --backup latest
```

**Auto-tune** (`scripts/ml/auto_tune.py`):

```bash
# Preview what strategies would run
uv run python scripts/ml/auto_tune.py --dry-run --runs 10

# List available strategies
uv run python scripts/ml/auto_tune.py --list-strategies

# Run 5 strategies (backtest + register + compare for each)
uv run python scripts/ml/auto_tune.py --runs 5

# Run 5 strategies and promote the best params into forecast_pipeline_config.yaml
uv run python scripts/ml/auto_tune.py --runs 5 --promote

# Resume from strategy 4 (if earlier run was interrupted)
uv run python scripts/ml/auto_tune.py --runs 5 --start-from 4
```

The auto-tune script also exports the best run's parameters to `data/tuning/best_params_lgbm.json`.
This file can be referenced in `forecast_pipeline_config.yaml` via `params_file: data/tuning/best_params_lgbm.json` under `algorithms.lgbm_cluster`
for production backtests, champion selection, and long-range forecasts.

---

## Implementation Plan

### Phase 1: Data Model + Script Integration

1. Create `sql/095_create_lgbm_tuning.sql` with all three tables
2. Add `tracking` section to `config/forecasting/forecast_pipeline_config.yaml`
3. Create `scripts/ml/register_tuning_run.py` with register/complete/compare/list/promote subcommands
4. Modify `scripts/run_backtest.py` to call register (pre) and complete (post) when auto_register is enabled
5. Add Make targets to Makefile
6. Tests: `tests/unit/test_lgbm_tuning.py`, `tests/api/test_lgbm_tuning.py`

### Phase 2: API

1. Create `api/routers/forecasting/lgbm_tuning.py` with all 8 endpoints
2. Mount router in `api/main.py` with prefix `/lgbm-tuning`
3. Add `/lgbm-tuning` to Vite proxy in `frontend/vite.config.ts`
4. Tests: `tests/api/test_lgbm_tuning.py`

### Phase 3: UI

1. Create `frontend/src/tabs/LgbmTuningTab.tsx` with three panels
2. Create `frontend/src/api/queries/lgbm-tuning.ts` query module
3. Add tab to sidebar in `App.tsx` under Demand section
4. Tests: `frontend/src/tabs/__tests__/LgbmTuningTab.test.tsx`

### Phase 4: Seed + Documentation

1. Run `make lgbm-tuning-seed` to create baseline run #1
2. Update `docs/specs/README.md` to add this spec to the index
3. Update `docs/ARCHITECTURE.md` and `docs/PLATFORM_GUIDE.md`
4. Update `CLAUDE.md` with new Make targets and Vite proxy prefix

---

## AI Tuning Chat

### Overview

An interactive AI-powered chat panel embedded in the LGBM Tuning tab. The AI advisor reviews previous runs, identifies patterns across clusters and timeframes, recommends parameter changes, and (with user confirmation) kicks off new backtest runs — all within a conversational interface. Results flow back into the chat for continuous iterative tuning.

### Database Schema (`sql/096_create_tuning_chat.sql`)

| Table | Purpose |
|---|---|
| `tuning_chat_session` | Chat sessions (UUID PK, title, status, context JSONB for cached run summary) |
| `tuning_chat_message` | Messages within sessions (role: user/assistant/system; message_type: text/recommendation/run_started/run_completed/run_failed/analysis/error; metadata JSONB) |

### AI Agent (`common/ai/tuning_advisor.py`)

Follows the `AIPlannerAgent` pattern with an agentic tool-use loop:
- Provider-agnostic: supports OpenAI and Anthropic via config
- Circuit breakers: `MAX_TURNS=20`, `TOKEN_BUDGET=50,000`
- Sliding window: max 40 messages (first 3 + last 37) for LLM context
- Logs each turn to `ai_call_log` for observability

#### 7 Tools

| Tool | Purpose |
|---|---|
| `list_tuning_runs` | Recent run history with accuracy/WAPE/bias metrics |
| `get_run_detail` | Full detail: timeframes + clusters + months |
| `compare_runs` | Side-by-side comparison with deltas + verdict |
| `analyze_cluster_patterns` | Cross-run cluster performance trends |
| `get_current_config` | Current LGBM params + tried strategies |
| `recommend_params` | Structure a recommendation as JSON for frontend card |
| `check_run_status` | Poll run progress |

Config: `config/tuning_advisor_config.yaml`

### API Endpoints (`api/routers/forecasting/tuning_chat.py`)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/lgbm-tuning/chat/sessions` | Create new session, seed with run summary context |
| GET | `/lgbm-tuning/chat/sessions` | List sessions (active/archived) with message count |
| GET | `/lgbm-tuning/chat/sessions/{id}` | Get session + full message history |
| POST | `/lgbm-tuning/chat/sessions/{id}/messages` | Send user message → AI response (synchronous) |
| POST | `/lgbm-tuning/chat/sessions/{id}/confirm-run` | Confirm recommendation → trigger async backtest |
| GET | `/lgbm-tuning/chat/sessions/{id}/run-status/{run_id}` | Poll run completion status |

Safety guards: `max_concurrent_runs=1` (409 on conflict), `min_seconds_between_runs=300`, `require_confirmation=true`.

### Frontend Components

| Component | Purpose |
|---|---|
| `TuningChatPanel` | Main chat UI with session management, message list, input area |
| `RecommendationCard` | Renders parameter overrides, expected impact, risk — Confirm & Run / Reject buttons |
| `RunStatusCard` | Polls run status; renders running (timer), completed (metrics), or failed (error) states |
| `SessionList` | Horizontal scrollable session picker with "New" button |

The chat panel is integrated as a collapsible section in `LgbmTuningTab.tsx`, toggled by the "AI Tuning Advisor" button.

### Data Flow

```
User asks "What should I try next?"
  → POST /lgbm-tuning/chat/sessions/{id}/messages
    → TuningAdvisorAgent.run_turn() with tool-use loop
      → list_tuning_runs → analyze_cluster_patterns → recommend_params
    → Insert assistant messages (text + recommendation) into DB
  → Frontend renders RecommendationCard

User clicks "Confirm & Run"
  → POST /lgbm-tuning/chat/sessions/{id}/confirm-run
    → register_run() → executor.submit(_execute_tuning_run)
    → Insert run_started message
  → Frontend polls GET .../run-status/{run_id} every 10s

Background thread completes backtest
  → complete_run() + register breakdowns
  → Insert run_completed message with results

Frontend poll detects completion
  → Renders completed RunStatusCard with accuracy delta
  → Invalidates run table → table refreshes
```

---

## Running Multiple Tuning Experiments

There are **four ways** to run tuning experiments, from simplest to most sophisticated.

### Method 1: Auto-Tune Campaign (Recommended for Batch)

Run predefined strategies from model-specific config files. Each strategy overrides specific params, runs a full backtest (~25 min each), registers the result, and prints a leaderboard.

**Strategy configs:** All strategies are in `config/forecasting/tune_strategies.yaml`, organized by model key:
- `lgbm` section: 13 strategies
- `catboost` section: 15 strategies
- `xgboost` section: 15 strategies

```bash
# LGBM auto-tune (original)
make lgbm-auto-tune RUNS=5
make lgbm-auto-tune-promote RUNS=5

# CatBoost auto-tune
uv run python scripts/ml/auto_tune.py --model catboost --runs 10

# XGBoost auto-tune
uv run python scripts/ml/auto_tune.py --model xgboost --runs 10

# Seed 10 pre-computed experiment runs for CatBoost + XGBoost
uv run python scripts/ml/seed_model_tuning.py
```

**LGBM Strategies (13):**

| # | Strategy | Focus |
|---|----------|-------|
| 1 | `higher_lr_0.05` | Faster convergence |
| 2 | `lower_lr_0.01` | More precise splits, 2500 trees |
| 3 | `deep_trees` | Complex interactions (depth 14, 127 leaves) |
| 4 | `shallow_trees` | Prevent overfitting (depth 6, 31 leaves) |
| 5 | `heavy_regularization` | Combat overfitting (lambda=5, min_child=50) |
| 6 | `light_regularization` | Maximum flexibility |
| 7 | `aggressive_subsample` | Stochastic regularization (60% rows + cols) |
| 8 | `more_trees_slow_lr` | Slow & steady (2500 trees, LR=0.015) |
| 9 | `balanced_mid` | Balanced middle ground |
| 10 | `fast_aggressive` | Rapid iteration (LR=0.08, 800 trees) |
| 11 | `sparse_demand_campaign` | Intermittent/sparse DFUs |
| 12 | `seasonal_boost_campaign` | Seasonal pattern capture |
| 13 | `regularization_heavy` | Small/noisy clusters |

**CatBoost Strategies (10):**

| # | Strategy | Focus |
|---|----------|-------|
| 1 | `cb_higher_lr_0.08` | Faster convergence, fewer iterations |
| 2 | `cb_lower_lr_0.01` | Slow precision, 2500 iterations |
| 3 | `cb_deep_trees` | Depth=10 with min_data_in_leaf guard |
| 4 | `cb_shallow_regularized` | Depth=4, l2=8.0, 2000 iterations |
| 5 | `cb_lossguide_leaf` | Leaf-wise splitting (like LightGBM) |
| 6 | `cb_border_count_low` | Fewer border candidates (64) |
| 7 | `cb_heavy_bagging` | Aggressive row+col dropout |
| 8 | `cb_moderate_bagging` | Balanced stochastic boosting |
| 9 | `cb_random_strength_high` | Noisy splits for small clusters |
| 10 | `cb_champion_blend` | Best combination of all findings |

**XGBoost Strategies (10):**

| # | Strategy | Focus |
|---|----------|-------|
| 1 | `xgb_higher_lr_0.08` | Faster convergence, fewer trees |
| 2 | `xgb_lower_lr_0.01` | Slow precision, 2500 trees |
| 3 | `xgb_deep_trees` | Max_depth=10 with heavy reg |
| 4 | `xgb_shallow_wide` | Depth=4, 2000 trees |
| 5 | `xgb_dart_dropout` | DART booster (10% tree dropout) |
| 6 | `xgb_heavy_regularization` | L1+L2+gamma maximum protection |
| 7 | `xgb_column_sampling` | Aggressive column-level dropout |
| 8 | `xgb_lossguide_leaf` | Leaf-wise splitting (max_leaves=63) |
| 9 | `xgb_max_bin_tuned` | Reduced histogram bins (128) |
| 10 | `xgb_champion_blend` | Best combination of all findings |

**Output:** A ranked leaderboard table + `data/tuning/best_params_<model>.json`.

### Method 2: Manual Single Run

Edit params, run a backtest, and register it.

```bash
# 1. Edit forecast_pipeline_config.yaml algorithms.<model_id>.params — change the params you want to test
#    e.g., set learning_rate: 0.03, num_leaves: 127

# 2. Run the LGBM backtest (~25 min)
make backtest-lgbm

# 3. Register the run with a descriptive label
uv run python scripts/ml/compare_backtest_runs.py \
  --register-latest \
  --label "my_experiment_v1" \
  --notes "Testing higher num_leaves for seasonal clusters"

# 4. Compare against the baseline
uv run python scripts/ml/compare_backtest_runs.py \
  --compare --baseline 8 --candidate <new_run_id>

# 5. List all runs to see the leaderboard
make lgbm-tuning-list
```

### Method 3: AI Tuning Advisor (Interactive via UI)

Use the AI chat panel in the LGBM Tuning tab for iterative, AI-guided tuning.

1. Open the UI → navigate to **LGBM Tuning** tab (sidebar → Demand section)
2. Click the **chat bubble icon** (top-right FAB) to open the AI Tuning Advisor
3. Ask questions like:
   - *"Which clusters are underperforming?"*
   - *"What should I try next?"*
   - *"The seasonal clusters have high WAPE — what params would help?"*
4. The AI analyzes past runs, cluster patterns, and feature importance, then recommends specific param changes
5. Review the recommendation card and click **Confirm & Run** to trigger a backtest
6. The chat panel polls for progress and shows results when complete
7. Continue the conversation: *"That improved A-class items but B-class got worse. Can we try..."*

**Safety guards:** max 1 concurrent run, 5-minute cooldown between runs, user confirmation required.

### Method 4: Sampled Fast Iteration

For rapid experimentation, use stratified DFU sampling to run backtests in ~3 min instead of ~25 min. Results have ±1-2pp deviation from full backtests.

```bash
# Via API — preview sample allocation
curl -X POST http://localhost:8000/lgbm-tuning/sampled/preview \
  -H "Content-Type: application/json" \
  -d '{"target_n": 5000, "method": "proportional"}'

# Via API — trigger a sampled run
curl -X POST http://localhost:8000/lgbm-tuning/sampled/run \
  -H "Content-Type: application/json" \
  -d '{"target_n": 5000, "method": "proportional", "param_overrides": {"learning_rate": 0.03}}'
```

Sampling methods: `proportional` (by cluster size), `equal` (same per cluster), `sqrt` (square-root allocation).

### Useful Commands Reference

```bash
# List all registered runs
make lgbm-tuning-list

# Compare two specific runs
make lgbm-tuning-compare BASELINE=8 CANDIDATE=9

# Backup artifacts for a run
make lgbm-tuning-backup RUN=latest

# View strategies without running
make lgbm-auto-tune-list

# Run campaign with promotion
make lgbm-auto-tune-promote RUNS=5
```

### Per-Cluster Adaptive Profiles

The backtest framework automatically applies cluster-specific param overrides based on demand characteristics. Profiles are defined in `config/forecasting/cluster_tuning_profiles.yaml`:

| Profile | Triggers When | Key Overrides |
|---------|---------------|---------------|
| `sparse_intermittent` | >50% zero demand, low mean | Shallow trees (15 leaves), strong regularization |
| `low_volume_volatile` | Low volume, high CV | Moderate trees (31 leaves), moderate regularization |
| `high_volume_stable` | High volume, low CV | Deep trees (127 leaves), minimal regularization |
| `seasonal_dominant` | Strong seasonality (amplitude > 0.3) | High colsample (0.9) to retain seasonal features |
| `default` | No match | Uses global config as-is |

These profiles apply automatically during any backtest — no manual configuration needed.

---

## Analysis Panels

The LGBM Tuning tab includes 4 sub-tabs beyond the Runs panel:

### Cluster EDA
- **Cluster Profiles:** Demand stats per cluster (mean demand, CV, zero %, seasonal amplitude, accuracy)
- **Error Concentration:** Which clusters and months contribute the most error
- **Demand Distribution:** Histogram of demand values per cluster
- **Seasonality Heatmap:** Month-by-cluster accuracy heatmap

### Feature Lab
- **Feature Importance:** SHAP-based ranking of top 30 features (horizontal bar chart)
- **Feature Stability:** Cross-fold rank consistency (stable / moderate / unstable)
- **Correlation Matrix:** Feature collinearity detection (flags |r| > 0.9)
- **Per-Cluster Importance:** How feature rankings differ across clusters

### Accuracy Budget
- **Accuracy Waterfall:** Naive baseline → ML model → Oracle ceiling pipeline
- **Gap Decomposition:** Addressable gap components (intermittent demand, seasonality, new products)
- **ABC Targets:** Accuracy by ABC class with target tracking
- **Monthly Trend:** Accuracy trajectory over time
- **Model Comparison:** Side-by-side model accuracy ranking

### Sampled Backtest
- **Strata Preview:** Cluster-level DFU counts and demand statistics
- **Sample Allocation:** Preview how N DFUs are distributed across clusters
- **Quick Runs:** Trigger fast (~3 min) sampled backtests for rapid iteration

---

## Feature Engineering Progression

All models share the same progressive feature expansion strategy:

### Base Features (17) — Phase 1 Runs

Core time series and lag features used in all baseline experiments:

| Feature | Category | Description |
|---------|----------|-------------|
| `ml_cluster` | Cluster | Used for per-cluster model partitioning only (removed as a model feature to prevent leakage) |
| `fourier_sin_12/6/4/3`, `fourier_cos_12/6/4/3` | Fourier | Cyclical seasonal encoding (sub-annual harmonics) |
| `lag_1` through `lag_12` | Lag | 1, 2, 3, 6, 12-month demand lags |
| `rolling_mean_3`, `rolling_mean_6` | Rolling | 3/6-month rolling averages |
| `rolling_std_3`, `rolling_cv_6` | Rolling | Volatility measures |
| `trend_slope` | Trend | Linear trend coefficient |
| `seasonal_strength` | Seasonality | Seasonal decomposition strength |
| `intermittency_ratio` | Demand Pattern | Fraction of zero-demand months |
| `price_index` | Price | Relative price index |
| `promo_flag` | Promotion | Binary promotion indicator |

### Extended V1 Features (+6 = 23 total) — Phase 2 Runs 6-8

TS profile and lag ratio features that capture demand velocity and pattern:

| Feature | Category | Description |
|---------|----------|-------------|
| `lag_ratio_1_3` | Lag Ratio | lag_1 / lag_3 — short-term demand velocity |
| `lag_ratio_3_6` | Lag Ratio | lag_3 / lag_6 — medium-term demand trend |
| `zero_count_6` | Demand Pattern | Count of zero months in last 6 — intermittency measure |
| `demand_cv_12` | Variability | 12-month coefficient of variation |
| `seasonal_amplitude` | Seasonality | Peak-to-trough ratio within a year |
| `ewm_alpha_3` | Smoothing | Exponentially weighted mean (alpha=3) |

### Extended V2 Features (+4 = 27 total) — Phase 2 Runs 9-10

Price, promotion, and entropy features:

| Feature | Category | Description |
|---------|----------|-------------|
| `price_lag_1` | Price | Previous month's price index |
| `promo_lag_1` | Promotion | Previous month's promotion flag |
| `trend_acceleration` | Trend | Second derivative of trend (acceleration/deceleration) |
| `demand_entropy` | Information | Shannon entropy of demand distribution — higher = more unpredictable |

### Experiment Tracking YAML

Detailed experiment logs with per-run configs, accuracy, and verdicts:
- LightGBM: `config/experiments/lgbm_experiments.yaml` (6 experiments)
- CatBoost: `config/experiments/catboost_experiments.yaml` (10 experiments)
- XGBoost: `config/experiments/xgboost_experiments.yaml` (10 experiments)

---

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- the engine that produces the runs being tracked
- [Tree Models](./04-tree-models.md) -- LGBM-specific hyperparameters and training logic
- [Advanced Backtest](./05-advanced-backtest.md) -- tuning, SHAP, and recursive settings captured per run
- [Algorithm Config](./06-algorithm-config.md) -- source of hyperparameter snapshots
- [Accuracy KPIs](./01-accuracy-kpis.md) -- WAPE, Bias, Accuracy formulas reused in comparisons

## See Also

- [Champion Selection](./07-champion-selection.md) -- picks the best model per DFU; tuning improves the model that feeds champion selection
- [Production Forecast](./08-production-forecast.md) -- downstream consumer of the improved model
- [Performance Profiling](../01-foundation/05-performance-profiling.md) -- duration tracking pattern reused here
