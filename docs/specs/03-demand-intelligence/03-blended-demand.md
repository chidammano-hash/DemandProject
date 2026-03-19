# 03-03 Blended Demand

> **Status:** Implemented | **Feature:** F3.4

## Why This Moved Here

Blended demand combines multiple demand signals into a single consensus view. It is about *interpreting* and *weighting* demand inputs -- a demand-intelligence activity -- rather than generating a new statistical forecast.

---

## Problem

Planners receive demand signals from multiple sources: statistical forecasts (ML models), sensing signals (short-horizon sales velocity, POS data), and manual overrides. Without a systematic blending mechanism, planners either pick one source or mentally average them, leading to inconsistent planning inputs across the portfolio.

---

## Solution

An alpha-weighted blending engine that combines statistical forecast output with demand sensing signals into a single blended demand quantity per DFU per month. The alpha parameter controls the weight given to each source, and can be configured globally or overridden per segment.

---

## How It Works

### Blending Formula

For each DFU-month:

```
blended_qty = alpha * sensing_qty + (1 - alpha) * statistical_qty
```

Where:
- `statistical_qty` = champion model forecast (from `fact_external_forecast_monthly` where `model_id = 'champion'`)
- `sensing_qty` = short-horizon demand signal (from `fact_demand_signals`, extrapolated to monthly granularity)
- `alpha` = blending weight (0.0 = pure statistical, 1.0 = pure sensing)

### Alpha Selection

| Segment | Default Alpha | Rationale |
|---|---|---|
| Global default | 0.3 | Statistical models dominate for most items |
| High-velocity (ABC-A) | 0.4 | Recent signals more informative for fast movers |
| Intermittent | 0.1 | Sensing signals too noisy for sporadic demand |
| Promotional periods | 0.5 | Equal weight during events with demand shifts |

Alpha overrides can be set per ABC class or per individual DFU in the configuration file.

### Near-Horizon Emphasis

Sensing signals are most valuable for the immediate planning horizon (1-3 months). Beyond that window, the blend reverts toward the statistical forecast:

| Horizon | Effective Alpha |
|---|---|
| Month +1 | Configured alpha |
| Month +2 | alpha * 0.7 |
| Month +3 | alpha * 0.4 |
| Month +4 onward | 0.0 (pure statistical) |

---

## Data Model

Output is written to the blended demand forecast table, consumed by downstream replenishment and financial planning.

| Column | Type | Purpose |
|---|---|---|
| `item_no` | TEXT | Item identifier |
| `loc` | TEXT | Location identifier |
| `month_start` | DATE | Planning month |
| `statistical_qty` | NUMERIC | Champion model forecast |
| `sensing_qty` | NUMERIC | Demand signal extrapolation |
| `alpha` | NUMERIC | Weight applied |
| `blended_qty` | NUMERIC | Final blended output |
| `computed_at` | TIMESTAMPTZ | Computation timestamp |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/blended-demand/summary` | Portfolio-level blended vs statistical comparison |
| GET | `/inv-planning/blended-demand/detail` | Per-DFU blended demand with alpha values |
| PUT | `/inv-planning/blended-demand/alpha` | Override alpha for segment or DFU |

Router: `api/routers/blended_forecast.py`

---

## Pipeline

```
make blended-demand    # Compute blended demand from champion + sensing signals
```

| Step | Script | Output |
|---|---|---|
| Compute blend | `scripts/compute_blended_forecast.py` | Blended demand rows |

Requires champion selection and demand signals to have run first.

---

## Configuration

File: `config/blended_demand_config.yaml` (referenced as F3.4 in algorithm config)

```yaml
default_alpha: 0.3
horizon_decay: [1.0, 0.7, 0.4, 0.0]
segment_overrides:
  ABC_A: 0.4
  intermittent: 0.1
  promotional: 0.5
```

---

## Dependencies

- **Upstream:** `fact_external_forecast_monthly` (champion model), `fact_demand_signals` (sensing), `dim_dfu` (ABC class)
- **Downstream:** Replenishment planning, financial planning, S&OP demand review
- **Libraries:** pandas, numpy

---

## See Also

- [../04-inventory/06-analytics](../04-inventory/06-analytics.md) -- Demand signals (sensing input)
- [../02-forecasting/02-05-champion-selection](../02-forecasting/02-05-champion-selection.md) -- Statistical forecast source
- [01-dfu-clustering](01-dfu-clustering.md) -- Cluster-level alpha defaults possible via segment mapping

## Frontend

`BlendedDemandPanel.tsx` in the Sensing group of the Inventory Planning tab. Displays alpha distribution, blended vs statistical comparison, and per-DFU drill-down.
