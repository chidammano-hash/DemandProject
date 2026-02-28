# Feature 15 — Champion Model Selection (Best-of-Models)

## Overview

Automatically identify the best-performing forecasting model **per DFU per month** using an expanding window of prior performance, and create a composite "champion" forecast. This implements the industry-standard **Forecast Value Added (FVA)** approach used in demand planning — the champion simulates what a planner would do: at each month, pick the model that has performed best historically (before-the-fact).

The ceiling (oracle) picks the best model per DFU per month using that month's actual error — the theoretical upper bound with perfect foresight. The gap between champion and ceiling quantifies how close the rolling selection is to optimal.

## Problem

With 13+ forecasting models, users need an automated way to:
1. Compare model performance at the DFU level (not just aggregate)
2. Select the best model for each DFU at each month based on prior performance
3. Create a composite "champion" forecast from the per-DFU per-month winners
4. Track which models win the most DFU-months (forecast value added analysis)
5. Benchmark against a theoretical ceiling (oracle) to quantify improvement opportunity

## Architecture

### Champion Selection Flow

```
config/model_competition.yaml
          ↓
  run_champion_selection.py
          ↓
  ┌────────────────────────────────┐
  │ For each DFU × month:          │
  │   Compute cumulative WAPE per  │
  │   model from PRIOR months only │
  │   → Pick model with lowest     │
  │     prior WAPE (before-the-    │
  │     fact expanding window)     │
  └───────────┬────────────────────┘
              ↓
  fact_external_forecast_monthly
    (model_id = 'champion')
              ↓
  ┌────────────────────────────────┐
  │ For each DFU × month:          │
  │   Pick model with lowest       │
  │   absolute error FOR that      │
  │   month (after-the-fact        │
  │   oracle / perfect foresight)  │
  └───────────┬────────────────────┘
              ↓
  fact_external_forecast_monthly
    (model_id = 'ceiling')
              ↓
  Refresh materialized views
              ↓
  Champion + Ceiling auto-appear
  in all accuracy comparison views
```

### Key Design Decision: model_id = 'champion' / 'ceiling'

Both forecasts are stored in the same `fact_external_forecast_monthly` table using `model_id = 'champion'` and `model_id = 'ceiling'`. Because all existing accuracy views, lag curves, and KPI computations are model_id-aware, they automatically appear in every comparison with **zero downstream changes**.

## Configuration

### `config/model_competition.yaml`

```yaml
competition:
  name: "default"
  metric: "wape"              # wape (lowest wins) or accuracy_pct (highest wins)
  lag: "execution"            # "execution" (per-DFU) or 0, 1, 2, 3, 4
  min_dfu_rows: 3             # min prior months required before champion can be selected
  champion_model_id: "champion"
  ceiling_model_id: "ceiling" # oracle ceiling (best model per DFU per month)
  models:
    - lgbm_global
    - lgbm_cluster
    - lgbm_transfer
    - catboost_global
    - catboost_cluster
    - xgboost_global
    - xgboost_cluster
```

| Field | Description |
|-------|-------------|
| `metric` | `wape` (lowest wins) or `accuracy_pct` (highest wins). WAPE is the industry default. |
| `lag` | `execution` uses each DFU's own execution lag; integers 0-4 for fixed horizon |
| `min_dfu_rows` | Minimum prior months of history required before a model qualifies for champion selection at a given month |
| `champion_model_id` | The model_id used for champion rows (default: `champion`) |
| `ceiling_model_id` | The model_id used for ceiling/oracle rows (default: `ceiling`) |
| `models` | List of model_ids to compete; configurable from the UI |

## Selection Algorithm

### Champion (Rolling/Expanding Window — Before-the-Fact)

At each month, for each DFU, pick the model with the best cumulative performance from prior months only. This is a **before-the-fact** selection — it only uses information available at decision time:

1. **Compute** per-model per-DFU per-month absolute error and actual values at the configured lag
2. **Accumulate** cumulative absolute error and cumulative actual for each model using an expanding window over all strictly prior months (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`)
3. **Compute prior WAPE** per model: `cum_abs_err / ABS(cum_actual)`
4. **Filter** models with fewer than `min_dfu_rows` prior months of history
5. **Rank** models within each DFU-month by prior WAPE ascending
6. **Select** the winner (rank = 1) per DFU-month
7. **Copy** forecast rows from the per-month winner, inserting with `model_id = 'champion'`

The result: at month 4, if `lgbm_cluster` had the lowest WAPE over months 1-3, its forecast is used. At month 5, if `catboost_global` now has the lowest WAPE over months 1-4, it takes over. Different DFUs and different months can use different models.

**Note:** The first `min_dfu_rows` months per DFU have no champion selections (insufficient history). This is correct — you cannot pick a "best" model without enough track record.

### Ceiling / Oracle (After-the-Fact — Perfect Foresight)

The ceiling model picks the best forecast **per DFU per month** — the theoretical upper bound of accuracy achievable with perfect foresight (if you always knew which model would be most accurate for every single month):

1. **Compute absolute error** per row: `ABS(basefcst_pref - tothist_dmd)`
2. **Rank** models within each DFU-month by absolute error ascending
3. **Select** the winner (rank = 1) per DFU-month
4. **Copy** forecast rows from the per-month winner, inserting with `model_id = 'ceiling'`

The ceiling is **not a deployable model** — it uses actuals retroactively (oracle/perfect foresight). It serves as a benchmark: the gap between champion and ceiling quantifies how much room exists to improve the rolling selection algorithm.

### Champion vs Ceiling

| Aspect | Champion | Ceiling |
|--------|----------|---------|
| Decision basis | Prior months only (before-the-fact) | Current month actuals (after-the-fact) |
| Granularity | Per DFU per month | Per DFU per month |
| Oracle? | No — deployable strategy | Yes — theoretical upper bound |
| WAPE formula | `SUM(\|F-A\|) / \|SUM(A)\|` | `SUM(\|F-A\|) / \|SUM(A)\|` |
| Use case | Operational selection benchmark | Perfect foresight benchmark |

### Data Leak Note

The **champion** uses a before-the-fact expanding window — at each month, it only sees prior months' performance. There is no data leak in the champion selection. The **ceiling** uses actuals retroactively (oracle). Both are diagnostic tools for evaluating the Forecast Value Added pipeline.

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
  "total_dfu_months": 43000,
  "total_champion_rows": 43000,
  "model_wins": {
    "lgbm_cluster": 15000,
    "catboost_global": 12000,
    "lgbm_global": 9870,
    ...
  },
  "overall_champion_wape": 28.5,
  "overall_champion_accuracy_pct": 71.5,
  "total_ceiling_rows": 54000,
  "ceiling_model_wins": {
    "lgbm_cluster": 18000,
    "catboost_global": 15000,
    "xgboost_global": 11500,
    ...
  },
  "overall_ceiling_wape": 18.2,
  "overall_ceiling_accuracy_pct": 81.8,
  "run_ts": "2026-02-25T10:30:00+00:00"
}
```

## Frontend UI

### Champion Selection Panel (Accuracy Tab)

Located below the Accuracy Comparison table in the Accuracy tab:

1. **Model Checkboxes**: Toggle which models compete (excludes `champion` and `ceiling`)
2. **Metric Selector**: WAPE (Lowest Wins) or Accuracy % (Highest Wins)
3. **Lag Selector**: Execution Lag or fixed lag 0-4
4. **Save Config**: Persists checkbox/metric/lag changes to YAML
5. **Run Competition**: Executes champion selection + ceiling computation (auto-saves config first)
6. **Champion Results**: DFUs evaluated, DFU-months, champion accuracy/WAPE, champion rows count, and champion model wins bar chart (indigo)
7. **Ceiling Results**: Ceiling accuracy/WAPE (emerald green), ceiling rows count, gap-to-ceiling indicator (amber, in percentage points), and ceiling model wins bar chart (emerald)

### Model Wins Visualization

Two horizontal bar charts at the same DFU-month granularity:
- **Champion Model Wins** (indigo, before-the-fact): How many DFU-months each model won via expanding window selection. Sorted by win count descending.
- **Ceiling Model Wins — Oracle** (emerald, after-the-fact): How many DFU-months each model won with perfect foresight. Shows which models are most frequently the best.

### Gap to Ceiling

The "Gap to Ceiling" KPI card shows how many percentage points the champion accuracy is below the ceiling accuracy. A small gap means the rolling champion selection is near-optimal; a large gap suggests room for improvement in the selection algorithm.

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
| `frontend/src/tabs/AccuracyTab.tsx` | Champion Selection UI panel |
| `data/champion/champion_summary.json` | Last run summary (generated) |
| `tests/unit/test_champion_selection.py` | Unit tests for summary + config logic |

## Design Rationale

| Decision | Why |
|----------|-----|
| Rolling/expanding window champion (before-the-fact) | Simulates what a planner would do: pick best model based on available history. No data leak. |
| Ceiling as separate oracle model | Provides theoretical upper bound; quantifies gap to perfect foresight |
| Both at DFU-month granularity | Makes gap-to-ceiling directly comparable (same denominator) |
| WAPE as selection metric | Volume-weighted, handles zero-demand months; industry standard |
| `SUM(\|F-A\|) / \|SUM(A)\|` for both | Consistent formula makes gap calculation meaningful |
| Store as model_id='champion'/'ceiling' | Reuses existing fact table + views; zero downstream changes |
| YAML config (not DB table) | Matches clustering config pattern; simple and git-trackable |
| Synchronous API | SQL + bulk INSERT completes in <30s; no async complexity needed |
| DELETE + INSERT | Idempotent full replace; consistent with backtest loading |
| `min_dfu_rows` as minimum prior months | Prevents champion selection on insufficient history |

## Dependencies

- `pyyaml>=6.0.0` (already in pyproject.toml)
- `psycopg` (already in pyproject.toml)
- Existing `fact_external_forecast_monthly` table with `UNIQUE(forecast_ck, model_id)` constraint
- Existing materialized views (`agg_accuracy_by_dim`, `agg_forecast_monthly`, `agg_dfu_coverage`)

---

## Implementation Details

### Config YAML
- `ceiling_model_id` field exists in actual `config/model_competition.yaml` (missing from spec's example block)

### Router Module
- `api/routers/competition.py` (354 lines) implements same 4 endpoints
- Not mounted via `include_router` (inline routes in `main.py` take precedence)

### API Authentication
- `PUT /competition/config` and `POST /competition/run` require `X-API-Key` header (`require_api_key` dependency)

### Request Body
- `CompetitionConfigUpdate` Pydantic model: `metric`, `lag`, `min_dfu_rows`, `champion_model_id`, `models`

### View Refresh
- `refresh_views()` refreshes 3 views: `agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_dfu_coverage`

### Test Files
- `tests/unit/test_champion_selection.py` — unit tests for `generate_summary()` and `load_config()`
- `tests/api/test_competition.py` — API tests for `/competition/config` and `/competition/summary`
