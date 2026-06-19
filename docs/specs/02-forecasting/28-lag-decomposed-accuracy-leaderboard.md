# 28 — Lag-Decomposed Accuracy Leaderboard

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Portfolio (Lag Curve section) |
| **Key Files** | `api/routers/forecasting/accuracy.py`, `frontend/src/tabs/aggregate-analysis/LagLeaderboardPanel.tsx` |

## Problem

Planners need to see which model wins at each execution lag (0–4 months), not only portfolio-level accuracy. `backtest_lag_archive` already stores all-lag predictions; aggregating per lag manually is slow.

## Solution

`GET /forecast/accuracy/lag-leaderboard` ranks models by accuracy at each lag using `agg_accuracy_lag_archive`. Optional `month_from` / `month_to` window; `limit` caps models per lag (default 10).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/forecast/accuracy/lag-leaderboard` | Per-lag ranked models with `accuracy_pct`, `wape`, `bias`, `n_rows` |

**Note:** Pinball loss requires quantile forecast rows; not computed until quantile heads ship in production backtests.

## UI

`LagLeaderboardPanel` renders under the Lag Curve section on the Portfolio tab when the lag curve panel is visible.

## Dependencies

- `agg_accuracy_lag_archive` MV (`sql/011_create_accuracy_slice_views.sql`)
- `backtest_lag_archive` populated by backtest load
