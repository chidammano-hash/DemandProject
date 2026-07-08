# Blended Demand

> Combines statistical forecast output with demand sensing signals into a single alpha-weighted blended demand quantity per DFU per week, with a linear alpha decay over the sensing horizon and a velocity-spike outlier cap.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning (Sensing group) |
| **Key Files** | `scripts/forecasting/compute_blended_forecast.py`, `api/routers/forecasting/blended_forecast.py`, `config/forecasting/forecast_domain_config.yaml` (`sensing` section) |

---

## Why This Moved Here

Blended demand combines multiple demand signals into a single consensus view. It is about *interpreting* and *weighting* demand inputs -- a demand-intelligence activity -- rather than generating a new statistical forecast.

---

## Problem

Planners receive demand signals from multiple sources: statistical forecasts (ML models), sensing signals (short-horizon sales velocity, POS data), and manual overrides. Without a systematic blending mechanism, planners either pick one source or mentally average them, leading to inconsistent planning inputs across the portfolio.

---

## Solution

An alpha-weighted blending engine that combines statistical forecast output with demand sensing signals into a single blended demand quantity per DFU per week. Alpha decays linearly from 1.0 (pure sensing) at the current week to 0.0 (pure statistical) at the edge of the sensing horizon, and a velocity-spike cap prevents a single large in-month order from blowing up the near-term blended plan.

---

## How It Works

### Blending Formula

For each DFU-week:

```
blended_qty = alpha_weight * sensing_signal_qty + (1 - alpha_weight) * statistical_forecast_qty
```

Where:
- `statistical_forecast_qty` = champion model forecast (from `fact_external_forecast_monthly` where `model_id = 'champion'`, monthly figure converted to a weekly rate via `monthly_to_weekly()`, divisor 4.33 weeks/month)
- `sensing_signal_qty` = MTD-velocity-projected monthly demand (see Velocity Projection below), converted to a weekly rate the same way
- `alpha_weight` = blending weight for that week, 0.0 (pure statistical) to 1.0 (pure sensing) - see Alpha Decay below
- `blended_qty` is floored at 0.0

### Alpha Decay

`compute_alpha(week_offset, sensing_horizon_weeks)` in `scripts/forecasting/compute_blended_forecast.py` decays linearly from 1.0 at the current week (`week_offset = 0`) to 0.0 at `week_offset >= sensing_horizon_weeks`:

```
alpha = max(0.0, min(1.0, 1.0 - week_offset / sensing_horizon_weeks))
```

Examples with the default 4-week horizon: `week_offset=0` -> 1.0 (pure sensing), `week_offset=3` -> 0.25 (mostly statistical), `week_offset=4` and beyond -> 0.0 (pure statistical). There is no per-ABC-class or per-DFU alpha override - the decay curve is uniform across the portfolio, driven only by `sensing_horizon_weeks`.

Each run writes `sensing_horizon_weeks + 4` weeks of rows per DFU (8 weeks with the default horizon of 4), so the tail weeks beyond the horizon are persisted as pure-statistical rows rather than dropped.

### Velocity Projection & Outlier Cap

`compute_velocity_signal()` projects the sensing signal from month-to-date sales velocity: `daily_run_rate = mtd_sales / days_elapsed`, compared against the historical daily average to get `spike_ratio = daily_run_rate / historical_daily_avg`. If `spike_ratio` exceeds `outlier_threshold` (default 3.0), the run rate is capped at `historical_daily_avg * outlier_threshold` before projecting to a monthly (then weekly) quantity - this is what `velocity_spike_ratio` and `is_outlier_capped` record.

---

## Data Model

Output is written to `fact_blended_demand_plan`, grain **item_id + loc + week_start + plan_version** (unique constraint `uq_blended_plan`).

| Column | Type | Purpose |
|---|---|---|
| `item_id` | VARCHAR(50) | Item identifier |
| `loc` | VARCHAR(50) | Location identifier |
| `week_start` | DATE | Week start (Monday-anchored) |
| `plan_version` | VARCHAR(50) | Plan version tag, default `latest` |
| `alpha_weight` | NUMERIC(4,3) | Sensing weight for this week, 0.0-1.0 |
| `sensing_signal_qty` | NUMERIC(12,2) | Weekly demand-sensing projection |
| `statistical_forecast_qty` | NUMERIC(12,2) | Weekly champion-model forecast |
| `blended_qty` | NUMERIC(12,2) | Final blended output (floored at 0) |
| `velocity_spike_ratio` | NUMERIC(6,3) | MTD daily run-rate vs. historical daily average |
| `is_outlier_capped` | BOOLEAN | Whether the velocity spike was capped at `outlier_threshold` |
| `computed_at` | TIMESTAMPTZ | Computation timestamp |

`mv_sensing_overrides_active` (materialized view, `sql/053_create_blended_forecast.sql`) narrows this to the nearest future `week_start` where `alpha_weight > 0.5`, and backs `GET /forecast/sensing-active`.

---

## API

| Method | Path | Params | Purpose |
|---|---|---|---|
| GET | `/forecast/blended` | `item_id`, `loc` (required), `weeks` (default 8, capped 1-52), `plan_version` | Weekly blended forecast for one DFU, plus the summed `monthly_total_blended` |
| GET | `/forecast/blended/summary` | none | Portfolio sensing status as of the planning date: total DFUs, count with `alpha_weight > 0.3`, average spike ratio, capped-row count |
| GET | `/forecast/sensing-active` | `page` (default 1), `page_size` (default 50, capped 1-200) | Paginated list of DFUs where sensing dominates (`alpha_weight > 0.5`), from `mv_sensing_overrides_active` |

Router: `api/routers/forecasting/blended_forecast.py`

---

## Pipeline

```
make blended-compute        # Compute blended demand from champion + sensing signals
make blended-compute-dfu ITEM=<item_no> LOC=<loc>   # Single DFU
make blended-dry             # Dry run, no DB writes
```

| Step | Script | Output |
|---|---|---|
| Compute blend | `scripts/forecasting/compute_blended_forecast.py` | `fact_blended_demand_plan` rows |

Requires champion selection and demand signals to have run first.

---

## Configuration

File: `config/forecasting/forecast_domain_config.yaml`, `sensing` section - read via `load_config().get("sensing", {})`.

```yaml
sensing:
  sensing_horizon_weeks: 4   # weeks over which alpha decays to 0
  outlier_threshold: 3.0     # cap velocity spike_ratio at this multiple of the historical average
```

The `sensing` key is not currently present in the checked-in config file, so both values fall back to the hardcoded defaults in `compute_blended_forecast.py` (`SENSING_HORIZON_WEEKS = 4`, `OUTLIER_THRESHOLD = 3.0`).

---

## Dependencies

- **Upstream:** `fact_external_forecast_monthly` (champion model, `model_id = 'champion'`), `fact_demand_signals` (MTD sensing signal)
- **Downstream:** Replenishment planning, financial planning, S&OP demand review
- **Libraries:** psycopg, PyYAML

---

## See Also

- [../04-inventory/06-analytics](../04-inventory/06-analytics.md) -- Demand signals (sensing input)
- [../02-forecasting/07-champion-selection](../02-forecasting/07-champion-selection.md) -- Statistical forecast source
- [01-sku-clustering](01-sku-clustering.md) -- DFU clustering (not currently wired into alpha selection)

## Frontend

`BlendedDemandPanel.tsx` in the Sensing group of the Inventory Planning tab. Displays alpha distribution, blended vs statistical comparison, and per-DFU drill-down.
