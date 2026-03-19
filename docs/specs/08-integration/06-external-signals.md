# External Signal Ingestion

> Brings outside data (weather, economic indicators, promotions) into the forecasting process and measures how much each signal explains demand variation.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | Demand Signals panel (within Inventory Planning tab) |
| **Key Files** | `api/routers/external_signals.py` |

---

## Problem

Statistical forecasts use only historical sales data. They cannot account for weather shifts, economic downturns, or upcoming promotions that a planner knows will affect demand. Without a structured way to ingest and measure external signals, planners rely on gut feel to adjust forecasts, with no way to validate whether those adjustments actually improve accuracy.

## Solution

A signal ingestion framework stores external data (weather, economic indices, promotion calendars) aligned to the monthly planning calendar. A decomposition engine computes how much each signal correlates with demand variation for each DFU, using rolling Pearson correlation with configurable lag windows. Planners see ranked signal contributions in the Demand Signals panel and can use signal-adjusted weights in the Blended Demand panel.

## How It Works

1. Signal sources are registered with a type (weather, economic, promotion, market, custom) and refresh method
2. Signals are pulled via API, file upload, or manual entry and stored with monthly alignment
3. Each signal value is linked to a specific date and optionally to a specific item + location (NULL = global signal)
4. The decomposition engine retrieves demand history and aligned signal values for a DFU
5. It computes rolling Pearson correlation between each signal and demand (lag window 0-3 months)
6. Variance decomposition (R-squared attribution) ranks signals by how much demand variation they explain
7. Results show correlation direction, magnitude, and optimal lag per signal

## Data Model

### `dim_signal_source`

| Column | Type | Description |
|---|---|---|
| `source_id` | `SERIAL PK` | Auto-increment ID |
| `source_name` | `TEXT UNIQUE` | e.g., "weather_noaa", "econ_fred", "promo_internal" |
| `source_type` | `TEXT` | weather, economic, promotion, market, custom |
| `refresh_method` | `TEXT` | api_pull, file_upload, manual |
| `refresh_url` | `TEXT` | API endpoint for automated pulls |
| `refresh_schedule` | `TEXT` | Cron expression for scheduled refreshes |
| `is_active` | `BOOLEAN` | Whether the source is enabled |

### `fact_external_signal`

| Column | Type | Description |
|---|---|---|
| `signal_id` | `BIGSERIAL PK` | Auto-increment ID |
| `source_id` | `INTEGER FK` | References dim_signal_source |
| `signal_name` | `TEXT` | e.g., "temperature_avg", "cpi_index", "promo_discount_pct" |
| `signal_date` | `DATE` | Aligned to month-start for monthly signals |
| `item_no` | `TEXT` | Item (nullable -- NULL means global signal) |
| `loc` | `TEXT` | Location (nullable -- NULL means global signal) |
| `value` | `NUMERIC` | Signal value |
| `unit` | `TEXT` | e.g., "fahrenheit", "index", "percent" |
| `metadata` | `JSONB` | Additional signal context |

Unique constraint on `(source_id, signal_name, signal_date, item_no, loc)`.

### `fact_signal_decomposition`

| Column | Type | Description |
|---|---|---|
| `decomp_id` | `BIGSERIAL PK` | Auto-increment ID |
| `item_no` | `TEXT` | DFU item |
| `loc` | `TEXT` | DFU location |
| `month_start` | `DATE` | Month of the decomposition |
| `signal_name` | `TEXT` | Which signal was analyzed |
| `contribution_pct` | `NUMERIC(7,4)` | Percent of forecast variance explained |
| `correlation` | `NUMERIC(7,4)` | Pearson correlation with demand |
| `lag_months` | `INTEGER` | Optimal leading indicator offset (0-3) |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/external-signals` | List signals with filters (source, date range, item/loc) |
| GET | `/external-signals/sources` | List registered signal sources |
| POST | `/external-signals/refresh` | Trigger signal pull from configured sources (returns 202) |
| GET | `/external-signals/decomposition` | Signal impact decomposition for a DFU or aggregate |

## Configuration

Signal sources are managed via the API (database-driven, not YAML). Refresh schedules integrate with the job scheduler for automated pulls.

## Dependencies

- `scipy.stats.pearsonr` for correlation computation (already in deps)
- Optional: `httpx` for API-based signal pulls
- Signal refresh integrates with job scheduler (`common/job_registry.py`)

## See Also

- [Blended Demand](../04-operations/12-blended-demand.md) -- incorporates signal-adjusted weights
- [Notifications](./04-notifications.md) -- signal refresh completion triggers notifications
- [Integration Architecture](./01-integration-architecture.md) -- external signals as a data source vector
