# Known Gaps

> Technical debt and known limitations tracked for future resolution.

| | |
|---|---|
| **Status** | Living document |
| **UI Tab** | N/A (engineering reference) |
| **Key Files** | `common/core/constants.py`, `common/ml/feature_engineering.py`, `config/forecasting/forecast_pipeline_config.yaml` |

---

## 1. Clustering Leakage in Backtesting

**Status**: Resolved (2026-04-06)
**Severity**: Was Medium — inflated backtest accuracy metrics

`ml_cluster` was computed from all available sales history and stored as a static label in `dim_sku`. During backtesting, this label was read once at load time and used across all timeframes — including early ones where the training cutoff predated the data that determined the cluster assignment.

**Leakage surfaces (now removed):**

- `ml_cluster` was a direct categorical feature fed to tree models (LightGBM, CatBoost, XGBoost)
- Cross-DFU cluster aggregates (`cluster_mean_lag1`, `cluster_demand_trend`, etc.) used `ml_cluster` as the grouping key

**Fix applied — Option C (drop ml_cluster as a model feature):**

- Removed `ml_cluster` from `CAT_FEATURES` in `common/core/constants.py`
- Added `ml_cluster` to `METADATA_COLS` so `get_feature_columns()` excludes it from model features
- Added `ml_cluster` to `dfu_feat_cols` in `build_feature_matrix()` so it's merged into the grid for per-cluster partitioning
- Removed `ml_cluster` from `PROTECTED_FEATURES`
- Emptied `CROSS_DFU_FEATURES` (was `cluster_mean_lag1`, `cluster_total_lag1`, `cluster_demand_trend`, `cluster_zero_pct`)
- Removed `_compute_cross_dfu_features()` calls from `feature_engineering.py`

**Note:** `ml_cluster` is still used for per-cluster model training (splitting DFUs into cluster partitions for separate models). This is a legitimate use — the leakage was only in using it as a *feature* within each model. Clustering also remains used for the SKU Features tab, cluster experimentation studio, and inventory planning (ABC-XYZ segmentation).

**Remaining consideration:** Per-cluster training still uses cluster assignments derived from full history. Options A or B (re-cluster per timeframe, or cluster on earliest-cutoff data only) could further improve causal correctness, but the accuracy impact is expected to be small since the training partition boundary is the more significant factor.

---

## 2. Feature Redundancy in Backtest Feature Matrix

**Status**: Resolved (2026-04-06)

The backtest feature matrix contained highly correlated features (34 pairs with r > 0.9) including 5 exact duplicates (backward-compat aliases). This destabilized SHAP-based feature selection — SHAP distributes importance randomly across correlated features, making the selection set non-deterministic across runs.

**Fix applied — Multi-stage pre-SHAP feature selection pipeline:**

- Stage 0: Static duplicate alias removal (`DUPLICATE_FEATURE_ALIASES` in `constants.py`)
- Stage 1: Near-zero variance filter (per-timeframe, configurable threshold)
- Stage 2: Correlation pre-filter (per-timeframe, keeps higher-variance member of each pair)
- Stage 3: SHAP cumulative selection (existing, now operates on reduced set)

All stages are per-timeframe (causal) and configurable via `forecast_pipeline_config.yaml`.
