# 32 — Lag-Decomposed Accuracy Leaderboard

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Portfolio (Lag Curve section); Model Tuning (Backtest stage) |
| **Key Files** | `api/routers/forecasting/accuracy.py`, `frontend/src/tabs/aggregate-analysis/LagLeaderboardPanel.tsx`, `frontend/src/tabs/model-tuning/BacktestStagePanel.tsx` |

## Problem

Planners need to see model accuracy at each fixed forecast lag (0–4 months), not only portfolio-level execution-lag accuracy. `backtest_lag_archive` already stores all-lag predictions; aggregating per lag manually is slow.

## Solution

`GET /forecast/accuracy/lag-leaderboard` ranks models by accuracy at each lag using `agg_accuracy_lag_archive`. Optional `month_from` / `month_to` window; `limit` caps models per lag (default 10).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/forecast/accuracy/lag-leaderboard` | Per-lag ranked models with `accuracy_pct`, `wape`, `bias`, `n_rows` |


## UI

`LagLeaderboardPanel` renders under the Lag Curve section on the Portfolio tab when the lag curve panel is visible.

The Model Tuning Backtest stage also consumes this endpoint once for the model roster. Each model
row displays `Exec` (each DFU evaluated at its own production-relevant execution lag) followed by
`L0` through `L4` (all DFUs evaluated at each fixed forecast lag). Missing lag observations render
as an em dash instead of being confused with zero accuracy.

## Dependencies

- `agg_accuracy_lag_archive` MV (`sql/011_create_accuracy_slice_views.sql`)
- `backtest_lag_archive` populated by backtest load
