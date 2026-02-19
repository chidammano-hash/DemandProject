# Feature 15 — Champion Model Selection (Best-of-Models)

## Overview

Automatically identify the best-performing forecasting model **per DFU** and create a composite "champion" forecast. This implements the industry-standard **Forecast Value Added (FVA)** approach used in demand planning — the champion is not a single model but a per-DFU best pick based on accuracy metrics.

## Problem

With 10 forecasting models (3 frameworks × 3 strategies + 1 baseline), users need an automated way to:
1. Compare model performance at the DFU level (not just aggregate)
2. Select the best model for each DFU based on a configurable metric
3. Create a composite "champion" forecast from the per-DFU winners
4. Track which models win the most DFUs (forecast value added analysis)

## Architecture

### Champion Selection Flow

```
config/model_competition.yaml
          ↓
  run_champion_selection.py
          ↓
  ┌───────────────────────┐
  │ For each DFU:         │
  │   Compute WAPE per    │
  │   competing model     │
  │   → Pick lowest WAPE  │
  └───────────┬───────────┘
              ↓
  fact_external_forecast_monthly
    (model_id = 'champion')
              ↓
  Refresh materialized views
              ↓
  Champion auto-appears in all
  accuracy comparison views
```

### Key Design Decision: model_id = 'champion'

The champion forecast is stored in the same `fact_external_forecast_monthly` table using `model_id = 'champion'`. Because all existing accuracy views, lag curves, and KPI computations are model_id-aware, the champion automatically appears in every comparison with **zero downstream changes**.

## Configuration

### `config/model_competition.yaml`

```yaml
competition:
  name: "default"
  metric: "wape"              # wape (lowest wins) or accuracy_pct (highest wins)
  lag: "execution"            # "execution" (per-DFU) or 0, 1, 2, 3, 4
  min_dfu_rows: 3             # min forecast rows per DFU to qualify
  champion_model_id: "champion"
  models:
    - external
    - lgbm_global
    - lgbm_cluster
    - lgbm_transfer
    - catboost_global
    - catboost_cluster
    - catboost_transfer
    - xgboost_global
    - xgboost_cluster
    - xgboost_transfer
```

| Field | Description |
|-------|-------------|
| `metric` | `wape` (lowest wins) or `accuracy_pct` (highest wins). WAPE is the industry default. |
| `lag` | `execution` uses each DFU's own execution lag; integers 0-4 for fixed horizon |
| `min_dfu_rows` | Minimum forecast rows per DFU to qualify — prevents winning on 1 data point |
| `champion_model_id` | The model_id used for champion rows (default: `champion`) |
| `models` | List of model_ids to compete; configurable from the UI |

## Selection Algorithm

DFU-level selection (industry standard, avoids per-month overfitting):

1. **Group** all forecast rows by DFU (`dmdunit + dmdgroup + loc`) and `model_id`
2. **Compute WAPE** per DFU per model: `SUM(ABS(F-A)) / ABS(SUM(A))`
3. **Filter** DFUs with fewer than `min_dfu_rows` rows
4. **Rank** models within each DFU by WAPE ascending
5. **Select** the winner (rank = 1) per DFU
6. **Copy** all forecast rows from the winning model for each DFU, inserting with `model_id = 'champion'`

The result: DFU A might use `lgbm_cluster`, DFU B might use `catboost_global`, etc.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/competition/config` | GET | Return current config + available model_ids from DB |
| `/competition/config` | PUT | Update config (writes YAML to disk) |
| `/competition/run` | POST | Execute champion selection, return summary |
| `/competition/summary` | GET | Return last run summary from disk |

### POST /competition/run Response

```json
{
  "config": { "metric": "wape", "lag": "execution", ... },
  "total_dfus": 5432,
  "total_champion_rows": 54320,
  "model_wins": {
    "lgbm_cluster": 1234,
    "catboost_global": 1100,
    "lgbm_global": 987,
    "external": 876,
    ...
  },
  "overall_champion_wape": 28.5,
  "overall_champion_accuracy_pct": 71.5,
  "run_ts": "2026-02-19T10:30:00+00:00"
}
```

## Frontend UI

### Champion Selection Panel (Accuracy Tab)

Located below the Accuracy Comparison table in the Accuracy tab:

1. **Model Checkboxes**: Toggle which models compete (excludes `champion` itself)
2. **Metric Selector**: WAPE (Lowest Wins) or Accuracy % (Highest Wins)
3. **Lag Selector**: Execution Lag or fixed lag 0-4
4. **Save Config**: Persists checkbox/metric/lag changes to YAML
5. **Run Competition**: Executes champion selection (auto-saves config first)
6. **Results Card**: Shows DFUs evaluated, champion accuracy/WAPE, and model wins bar chart

### Model Wins Visualization

Horizontal bar chart showing how many DFUs each model won, sorted by win count descending. Each bar displays model name, DFU count, and percentage.

## CLI

```bash
make champion-select  # Run champion selection from config
```

Equivalent to:
```bash
uv run python scripts/run_champion_selection.py --config config/model_competition.yaml
```

## Key Files

| File | Purpose |
|------|---------|
| `config/model_competition.yaml` | Competition configuration |
| `scripts/run_champion_selection.py` | Standalone champion selection script |
| `api/main.py` | API endpoints for config, run, summary |
| `frontend/src/App.tsx` | Champion Selection UI panel |
| `data/champion/champion_summary.json` | Last run summary (generated) |

## Design Rationale

| Decision | Why |
|----------|-----|
| DFU-level (not per-month) | Per-month overfits to noise; planners choose one method per product-location |
| WAPE as default metric | Volume-weighted, handles zero-demand months; industry standard |
| Store as model_id='champion' | Reuses existing fact table + views; zero downstream changes |
| YAML config (not DB table) | Matches clustering config pattern; simple and git-trackable |
| Synchronous API | SQL + bulk INSERT completes in <10s; no async complexity needed |
| DELETE + INSERT | Idempotent full replace; consistent with backtest loading |

## Dependencies

- `pyyaml>=6.0.0` (already in pyproject.toml)
- `psycopg` (already in pyproject.toml)
- Existing `fact_external_forecast_monthly` table with `UNIQUE(forecast_ck, model_id)` constraint
- Existing materialized views (`agg_accuracy_by_dim`, `agg_forecast_monthly`)
