# Spec 13 — Production Baseline Seeding

> Seed production backtest runs as promoted experiments in Model Tuning, Champion, and Clustering tabs so users always have a baseline to compare against.

**Status:** Draft
**Depends on:** Spec 11 (Unified Model Tuning), Spec 12 (Dual Promotion), Spec 04-cluster-experimentation-studio, Spec 05-champion-experimentation-studio

---

## 1. Problem

After running the production ML pipeline (`make backtest-all`, `make champion-all`, `make cluster-all`), results exist only as files on disk (`data/backtest/`, `data/champion/`, `data/clustering/`) and in fact tables. The Model Tuning, Clustering Experiments, and Champion Experiments tabs show **zero experiments** — there is no promoted baseline to compare new experiments against.

Users must:
- Manually create a "production run" experiment to establish a baseline
- Guess what parameters were used for the current production models
- Have no visibility into production performance metrics within the experimentation UI

This defeats the purpose of the comparison panels, which are designed to show deltas against a promoted baseline.

---

## 2. Goal

After any production pipeline run completes, the system **automatically registers** the results as an experiment row in the corresponding DB table with `is_promoted = TRUE`. This gives the UI:

- A promoted baseline row with a crown icon in the experiment list
- Full metrics (accuracy, WAPE, bias) and per-lag/per-cluster/per-month breakdowns
- Production hyperparameters visible in the comparison panel's "Params" tab
- A reference point for every future experiment's delta calculations

---

## 3. Scope

### 3.1 Model Tuning (LGBM, CatBoost, XGBoost)

**Source artifacts per model** (`data/backtest/{model_id}/`):
- `backtest_metadata.json` — portfolio accuracy, WAPE, bias, n_predictions, n_dfus, per-timeframe breakdowns
- `backtest_predictions.csv` — raw predictions for per-cluster and per-month aggregation

**Source config** (`config/algorithm_config.yaml`):
- `algorithms.{model}.params` — production hyperparameters
- `algorithms.{model}.training` — cluster_strategy, recursive, shap_select, shap_threshold, etc.

**Target tables:**

| Table | What to insert |
|---|---|
| `lgbm_tuning_run` | One row per model with `run_label='Production Baseline'`, `model_id`, `status='completed'`, `params` (from YAML), `accuracy_pct`, `wape`, `bias`, `n_predictions`, `n_dfus`, `metadata` (full JSON blob), `is_promoted=TRUE`, `promoted_at=NOW()`, `template_id='production_baseline'` |
| `lgbm_tuning_timeframe` | 10 rows per model (timeframes A-J) with accuracy/WAPE/bias per window |
| `lgbm_tuning_cluster` | N rows per model — per-`ml_cluster` and per-`cluster_assignment` accuracy, from predictions CSV joined to `dim_sku` |
| `lgbm_tuning_month` | N rows per model — per-month accuracy from predictions CSV |
| `lgbm_tuning_lag` | 5 rows per model (exec_lag 0-4) — per-lag accuracy from predictions CSV or all-lags archive |

**Features list:** Read from model artifacts at `data/models/{model_id}/cluster_*/` — each `.pkl` file contains `feature_cols`. Collect the union across clusters and store in `features` JSONB column.

### 3.2 Champion Selection

**Source artifacts** (`data/champion/`):
- `champion_summary.json` — strategy, champion_accuracy, ceiling_accuracy, gap_bps, model_distribution, per-lag and per-month breakdowns

**Source config** (`config/model_competition.yaml`):
- `strategy`, `metric`, `lag_mode`, `min_sku_rows`, `models`

**Target tables:**

| Table | What to insert |
|---|---|
| `champion_experiment` | One row: `label='Production Baseline'`, `strategy` (from config), `champion_accuracy`, `ceiling_accuracy`, `gap_bps`, `n_champions`, `n_dfu_months`, `model_distribution`, `is_promoted=TRUE`, `promoted_at=NOW()`, `template_id='production_baseline'` |
| `champion_experiment_lag` | 5 rows (exec_lag 0-4) — per-lag champion vs ceiling accuracy |
| `champion_experiment_month` | N rows — per-month champion vs ceiling accuracy |

### 3.3 Clustering

**Source artifacts** (`data/clustering/`):
- `cluster_metadata.json` — optimal_k, silhouette_score, inertia, total_dfus, n_clusters, cluster_sizes, k_selection_results, features used
- `cluster_profiles.json` — cluster demand profiles
- `cluster_assignments.csv` — the promoted label mapping

**Source config** (`config/clustering_config.yaml`):
- `time_window_months`, `min_months_history`, `k_range`, `min_cluster_size_pct`, feature flags

**Target tables:**

| Table | What to insert |
|---|---|
| `cluster_experiment` | One row: `label='Production Baseline'`, `scenario_id` (generate as `sc_production_YYYYMMDD`), `optimal_k`, `silhouette_score`, `inertia`, `total_dfus` (renamed from `total_dfus` to `total_skus` per naming convention), `n_clusters`, `cluster_sizes`, `profiles`, `k_selection_results`, `feature_params`, `model_params`, `label_params`, `is_promoted=TRUE`, `promoted_at=NOW()`, `template_id='production_baseline'`, `artifacts_path='data/clustering'` |

---

## 4. Implementation

### 4.1 New Script: `scripts/ml/seed_production_baselines.py`

Single entry point that reads production artifacts and seeds all experiment tables. Idempotent — if a `Production Baseline` row already exists for a given model/table, it updates rather than duplicates.

```
Usage:
  python scripts/ml/seed_production_baselines.py                    # Seed all (model tuning + champion + clustering)
  python scripts/ml/seed_production_baselines.py --scope tuning     # Model tuning only (lgbm, catboost, xgboost)
  python scripts/ml/seed_production_baselines.py --scope champion   # Champion only
  python scripts/ml/seed_production_baselines.py --scope clustering # Clustering only
  python scripts/ml/seed_production_baselines.py --model lgbm       # Single model tuning only
```

**Logic per model tuning seed:**

```python
def seed_tuning_baseline(model_id: str) -> None:
    """Read production artifacts and upsert a promoted baseline experiment."""
    meta_path = f"data/backtest/{model_id}/backtest_metadata.json"
    pred_path = f"data/backtest/{model_id}/backtest_predictions.csv"
    config = load_config("algorithm_config")

    # 1. Load metadata
    metadata = json.loads(Path(meta_path).read_text())
    acc = metadata["accuracy_at_execution_lag"]

    # 2. Load params from config YAML
    model_key = model_id.replace("_cluster", "")  # lgbm, catboost, xgboost
    params = config["algorithms"][model_key]["params"]
    training = config["algorithms"][model_key]["training"]

    # 3. Collect features from trained model artifacts
    features = collect_model_features(model_id)

    # 4. Upsert into lgbm_tuning_run
    #    - DELETE existing WHERE run_label = 'Production Baseline' AND model_id = %s
    #    - INSERT new row with is_promoted=TRUE, promoted_at=NOW()
    #    - Clear any other is_promoted=TRUE for this model_id

    # 5. Insert timeframe breakdowns from metadata["timeframes"]

    # 6. Read predictions CSV, join dim_sku for clusters,
    #    aggregate per-cluster, per-month, per-lag breakdowns
    #    Insert into lgbm_tuning_cluster, lgbm_tuning_month, lgbm_tuning_lag
```

**Logic for champion seed:**

```python
def seed_champion_baseline() -> None:
    summary = json.loads(Path("data/champion/champion_summary.json").read_text())
    config = load_config("model_competition")

    # 1. Upsert champion_experiment row with is_promoted=TRUE
    # 2. Insert per-lag rows into champion_experiment_lag
    # 3. Insert per-month rows into champion_experiment_month
```

**Logic for clustering seed:**

```python
def seed_clustering_baseline() -> None:
    meta = json.loads(Path("data/clustering/cluster_metadata.json").read_text())
    profiles = json.loads(Path("data/clustering/cluster_profiles.json").read_text())
    config = load_config("clustering_config")

    # 1. Upsert cluster_experiment row with is_promoted=TRUE
    # 2. Store feature_params, model_params, label_params from config
    # 3. Store results: optimal_k, silhouette_score, cluster_sizes, profiles
```

### 4.2 Idempotency Strategy

Each seed function uses a transaction:

```sql
BEGIN;
-- Remove any existing production baseline for this scope
DELETE FROM lgbm_tuning_lag WHERE run_id IN (
    SELECT run_id FROM lgbm_tuning_run
    WHERE run_label = 'Production Baseline' AND model_id = %s
);
-- (same for _timeframe, _cluster, _month, _lag_cluster)
DELETE FROM lgbm_tuning_run
    WHERE run_label = 'Production Baseline' AND model_id = %s;

-- Clear any other promoted flag for this model
UPDATE lgbm_tuning_run SET is_promoted = FALSE, promoted_at = NULL
    WHERE model_id = %s AND is_promoted = TRUE;

-- Insert the new baseline
INSERT INTO lgbm_tuning_run (...) VALUES (...);
-- Insert child rows...
COMMIT;
```

This ensures re-running after a new production backtest replaces the old baseline cleanly.

### 4.3 Makefile Integration

```makefile
seed-baselines:          ## Seed production baselines into experiment tables
	$(UV) run python scripts/ml/seed_production_baselines.py

seed-baselines-tuning:
	$(UV) run python scripts/ml/seed_production_baselines.py --scope tuning

seed-baselines-champion:
	$(UV) run python scripts/ml/seed_production_baselines.py --scope champion

seed-baselines-clustering:
	$(UV) run python scripts/ml/seed_production_baselines.py --scope clustering
```

**Pipeline integration** — append to existing pipeline targets:

```makefile
backtest-all: backtest-lgbm backtest-catboost backtest-xgboost seed-baselines-tuning
champion-all: champion-select champion-simulate seed-baselines-champion
cluster-all: cluster-train cluster-label seed-baselines-clustering
setup-backtest: ... seed-baselines
```

### 4.4 Config: `config/baseline_seeding.yaml`

```yaml
# Production baseline seeding configuration
baseline_label: "Production Baseline"
template_id: "production_baseline"

tuning:
  models:
    - lgbm_cluster
    - catboost_cluster
    - xgboost_cluster
  artifact_base: "data/backtest"
  model_artifact_base: "data/models"

champion:
  artifact_path: "data/champion/champion_summary.json"
  config_source: "model_competition"

clustering:
  metadata_path: "data/clustering/cluster_metadata.json"
  profiles_path: "data/clustering/cluster_profiles.json"
  assignments_path: "data/clustering/cluster_assignments.csv"
  config_source: "clustering_config"
```

---

## 5. UI Behavior

No frontend changes required. The existing UI already handles promoted experiments:

| UI Element | Current Behavior | After Seeding |
|---|---|---|
| Experiment list (Model Tuning) | Empty or shows only user-created experiments | Shows "Production Baseline" with crown icon per model |
| Comparison panel | No baseline to compare against | New experiments auto-compare against "Production Baseline" |
| Promoted badge | Shows on manually promoted runs | Shows on seeded baseline |
| `GET /{model}/promoted` | Returns 404/null | Returns production params + metrics |
| Cluster experiments list | Empty | Shows "Production Baseline" with optimal_k, silhouette |
| Champion experiments list | Empty | Shows "Production Baseline" with champion accuracy, gap |

The comparison panel will show full deltas: param diffs, accuracy changes per timeframe/cluster/month/lag — all populated from the seeded child table rows.

---

## 6. Data Mapping Reference

### 6.1 `backtest_metadata.json` -> `lgbm_tuning_run`

| JSON field | DB column |
|---|---|
| `accuracy_at_execution_lag.accuracy_pct` | `accuracy_pct` |
| `accuracy_at_execution_lag.wape` | `wape` |
| `accuracy_at_execution_lag.bias` | `bias` |
| `n_predictions` | `n_predictions` |
| `n_dfus` | `n_dfus` |
| (entire JSON blob) | `metadata` |

### 6.2 `backtest_metadata.json` timeframes -> `lgbm_tuning_timeframe`

| JSON field | DB column |
|---|---|
| `timeframes[].label` | `timeframe` |
| `timeframes[].train_end` | `train_end` |
| `timeframes[].predict_start` | `predict_start` |
| `timeframes[].predict_end` | `predict_end` |
| `timeframes[].n_predictions` | `n_predictions` |
| `timeframes[].accuracy_pct` | `accuracy_pct` |
| `timeframes[].wape` | `wape` |
| `timeframes[].bias` | `bias` |

### 6.3 `champion_summary.json` -> `champion_experiment`

| JSON field | DB column |
|---|---|
| `champion_accuracy` or `100 - overall_champion_wape` | `champion_accuracy` |
| `ceiling_accuracy` or `100 - overall_ceiling_wape` | `ceiling_accuracy` |
| `gap_bps` | `gap_bps` |
| `total_dfus` | `n_champions` |
| `total_dfu_months` | `n_dfu_months` |
| `model_wins` (dict) | `model_distribution` |

### 6.4 `cluster_metadata.json` -> `cluster_experiment`

| JSON field | DB column |
|---|---|
| `optimal_k` | `optimal_k` |
| `silhouette_score` | `silhouette_score` |
| `inertia` | `inertia` |
| `total_dfus` | `total_dfus` |
| `n_clusters` | `n_clusters` |
| `cluster_sizes` | `cluster_sizes` |
| `k_scores` (array) | `k_selection_results` |

---

## 7. Edge Cases

| Scenario | Handling |
|---|---|
| Artifacts don't exist yet (fresh install) | Script logs warning and skips that scope. No error. |
| Partial artifacts (e.g., LGBM done, CatBoost not) | Seeds only models with complete artifacts |
| User already promoted a different experiment | Production baseline replaces it (clears old `is_promoted`) |
| User re-runs production backtest | Re-running `seed-baselines` replaces the old baseline with fresh data |
| predictions CSV missing but metadata exists | Seed run-level metrics only; skip cluster/month/lag breakdowns. Log warning. |
| `n_dfus` in DB column name (cluster_experiment) | Use existing column name in DB; map to `total_skus` only in API response |

---

## 8. Testing

### Backend tests (`tests/unit/test_seed_production_baselines.py`):
1. Seed tuning baseline — verify row inserted with correct params, metrics, is_promoted=TRUE
2. Seed tuning baseline idempotency — run twice, verify only one row exists
3. Seed tuning baseline clears previous promoted — pre-insert a promoted row, seed, verify old one cleared
4. Seed champion baseline — verify champion_experiment row + lag/month child rows
5. Seed clustering baseline — verify cluster_experiment row with correct metadata
6. Missing artifacts — verify graceful skip with warning log
7. Partial artifacts — verify only available models are seeded

### Integration verification:
- After seeding, `GET /model-tuning/lgbm/promoted` returns the baseline
- After seeding, `GET /model-tuning/lgbm/compare?baseline_id=<seed>&candidate_id=<new>` returns valid deltas
- After seeding, the cluster and champion experiment list endpoints return promoted rows

---

## 9. Rollout

1. Merge script + config + Makefile targets
2. Run `make seed-baselines` on existing environment
3. Verify all three Model Tuning tabs, Champion tab, and Clustering tab show the production baseline
4. Update `make setup-backtest` and `make setup-all` to include seeding at the end
