# 02 — Forecasting & Models

This folder contains design specs for all forecasting model implementations, backtesting infrastructure, model selection, and algorithm optimization within Demand Studio.

## Files

| File | Feature | Summary |
|---|---|---|
| `feature5.md` | Forecast Accuracy KPIs | Defines WAPE, bias, and accuracy % formulas; KPI cards and rolling window selector in the Accuracy tab |
| `feature6.md` | Multi-Model Forecast Support | Extends the data model to store multiple `model_id` forecasts per DFU-month alongside the external baseline |
| `feature8.md` | Backtesting Framework (Expanding Window) | Shared backtest orchestrator: expanding-window timeframes, data loading, `run_tree_backtest()`, per-cluster strategy |
| `feature9.md` | LGBM Backtesting | LightGBM per-cluster backtest implementation using the shared backtest framework |
| `feature12.md` | CatBoost Backtesting | CatBoost per-cluster backtest implementation using the shared backtest framework |
| `feature13.md` | XGBoost Backtesting | XGBoost per-cluster backtest implementation using the shared backtest framework |
| `feature14.md` | Transfer Learning Backtest Strategy | (Archived) Transfer learning strategy that trains on all clusters and fine-tunes per cluster; removed in feature44 |
| `feature15.md` | Champion Model Selection | Per-DFU per-month best-of-models selection via 5 configurable strategies (expanding, rolling, decay, ensemble, meta_learner) with exec-lag-aware causality guards |
| `feature19.md` | PatchTST Backtesting | (Archived) Transformer-based time-series backtesting on Apple MPS GPU; removed in feature44 |
| `feature20.md` | DeepAR Backtesting | (Archived) LSTM probabilistic forecasting with DeepAR; removed in feature44 |
| `feature21.md` | Prophet Backtesting | (Archived) Per-DFU Prophet time-series backtest; removed in feature44 |
| `feature23.md` | Backtest Model Cleanup Utility | `clean_backtest_models.py` — selective removal of model predictions from Postgres by `model_id` with materialized view refresh |
| `feature24.md` | StatsForecast Backtesting | (Archived) Vectorized AutoARIMA + AutoETS via StatsForecast; removed in feature44 |
| `feature25.md` | NeuralProphet Backtesting | (Archived) PyTorch-based Prophet with GPU support; removed in feature44 |
| `feature41.md` | Hyperparameter Tuning | Bayesian Optuna tuning for LGBM, CatBoost, and XGBoost cluster models with walk-forward CV and early stopping |
| `feature42.md` | SHAP-Based Feature Selection | Per-timeframe automatic feature selection using SHAP cumulative importance; 4 REST endpoints and Accuracy tab UI panel |
| `feature43.md` | Recursive Multi-Step Forecasting | Recursive inference mode: model predictions written back into the feature grid via `update_grid_with_predictions()` for richer lag signals |
| `feature44.md` | Algorithm Configuration & Simplification | Centralizes all backtest algorithm options in `config/algorithm_config.yaml`; removes deprecated model types (Prophet, StatsForecast, NeuralProphet, PatchTST, DeepAR) |
