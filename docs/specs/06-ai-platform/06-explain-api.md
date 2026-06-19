# 06 — Forecast Explain API

| | |
|---|---|
| **Status** | Implemented (scaffold) |
| **UI Tab** | Item Analysis (DFU drill) |
| **Key Files** | `api/routers/intelligence/explain.py` |

## Problem

Planners ask why a forecast is high or low for a specific DFU. SHAP attributions exist but need a structured API for the UI.

## Solution

`GET /forecast/explain/{item_id}/{loc}` returns top SHAP features and a simple counterfactual shock (+/- 1 std dev) for the promoted model forecast. Appends an advisory ledger row on each read (audit trail).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/forecast/explain/{item_id}/{loc}` | Top features + counterfactual delta |

## Future

Production counterfactuals require partial-dependence or model re-run; current scaffold uses linear SHAP approximation.
