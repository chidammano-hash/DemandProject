# External Signal Ingestion

> Brings outside data (weather, economic indicators, promotions) into the forecasting process. A demand-decomposition engine that would measure how much each signal explains demand variation is scaffolded but not yet built.

| | |
|---|---|
| **Status** | Partial - signal listing, sources, and registry endpoints are functional; the demand-decomposition table (`mv_demand_decomposition`) is an unpopulated placeholder with no compute engine behind it; the source-refresh endpoint is a stub |
| **UI Tab** | None currently wired - query helpers exist in `frontend/src/api/queries/platform.ts` but no tab component calls them |
| **Key Files** | `api/routers/intelligence/external_signals.py` |

---

## Problem

Statistical forecasts use only historical sales data. They cannot account for weather shifts, economic downturns, or upcoming promotions that a planner knows will affect demand. Without a structured way to ingest and measure external signals, planners rely on gut feel to adjust forecasts, with no way to validate whether those adjustments actually improve accuracy.

## Solution

A signal registry (`dim_external_signal_source`) and fact table (`fact_external_signal`) store external data (weather, economic indices, promotion calendars, market signals) tagged to a source, date, and optional item/location. A `mv_demand_decomposition` table is reserved to hold a per-DFU-month decomposition (base, trend, seasonal, promotional, external-signal, and residual components), but its own DDL comment marks it a placeholder - "will be converted to a materialized view once the decomposition query is finalized" - and no job currently populates it, so the decomposition endpoint 404s in practice. Manually triggering a source refresh is likewise a stub today: it looks up the source row and returns a queued-style response without invoking any real ingestion.

## How It Works

1. Signal sources are registered in `dim_external_signal_source` with a type (weather, economic, promotion, market, custom) and an `api_config` payload
2. `fact_external_signal` rows are inserted per source/date/item/loc with a `signal_type`, `signal_value`, and `confidence` score (the ingestion path that would write these rows is not part of this router)
3. `GET /demand-signals/external` lists signals with source name, date, item/loc, and confidence, filterable by item, loc, and a trailing-days window
4. `GET /demand-signals/external/sources` lists registered sources and their `enabled` / `last_refresh_at` state
5. `POST /demand-signals/external/sources/{source_id}/refresh` looks up the source and returns `{"status": "refresh_queued"}` - the handler's own comment reads "In full implementation, this would trigger the signal ingestion script"; nothing is actually triggered
6. `GET /demand-signals/external/decomposition` reads `mv_demand_decomposition` for a DFU and would return its trend/seasonal/promotional/external/residual columns - but since nothing writes to that table, the endpoint returns 404 ("No decomposition data found") today

## Data Model

### `dim_external_signal_source`

| Column | Type | Description |
|---|---|---|
| `source_id` | `SERIAL PK` | Auto-increment ID |
| `name` | `TEXT UNIQUE` | e.g., "weather_noaa", "econ_fred", "promo_internal" |
| `source_type` | `TEXT NOT NULL` | weather, economic, promotion, market, custom |
| `api_config` | `JSONB` | Connection/auth config for automated pulls |
| `refresh_interval_hours` | `INT` | Default 24 |
| `enabled` | `BOOLEAN` | Whether the source is active (default true) |
| `last_refresh_at` | `TIMESTAMPTZ` | Most recent refresh timestamp |
| `created_at` | `TIMESTAMPTZ` | Row creation time |

### `fact_external_signal`

| Column | Type | Description |
|---|---|---|
| `signal_id` | `BIGSERIAL PK` | Auto-increment ID |
| `source_id` | `INTEGER FK` | References `dim_external_signal_source` |
| `signal_date` | `DATE` | Date the signal applies to |
| `item_id` | `TEXT` | Item (nullable -- NULL means global signal) |
| `loc` | `TEXT` | Location (nullable -- NULL means global signal) |
| `signal_type` | `TEXT` | e.g., "temperature_avg", "cpi_index", "promo_discount_pct" |
| `signal_value` | `NUMERIC` | Signal value |
| `confidence` | `NUMERIC` | Confidence score for the signal |
| `raw_payload` | `JSONB` | Original payload from the source |
| `created_at` | `TIMESTAMPTZ` | Row creation time |

### `mv_demand_decomposition` (placeholder table)

Despite the `mv_` prefix this is a plain table, not a materialized view. Its DDL comment in `sql/067_create_external_signals.sql` states it is a "placeholder" that will be converted to a materialized view "once the decomposition query is finalized." No job in the codebase currently writes to it.

| Column | Type | Description |
|---|---|---|
| `item_id` | `TEXT` | DFU item |
| `loc` | `TEXT` | DFU location |
| `month` | `DATE` | Month of the decomposition |
| `base_demand` | `NUMERIC` | Baseline demand component |
| `trend_component` | `NUMERIC` | Trend contribution |
| `seasonal_component` | `NUMERIC` | Seasonal contribution |
| `promotional_uplift` | `NUMERIC` | Promotion-driven uplift |
| `external_signal_effect` | `NUMERIC` | Attributed effect of external signals |
| `residual` | `NUMERIC` | Unexplained remainder |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/demand-signals/external` | List signals with filters (item, loc, trailing-days window) |
| GET | `/demand-signals/external/sources` | List registered signal sources |
| POST | `/demand-signals/external/sources/{source_id}/refresh` | Stub - validates the source exists and returns `{"status": "refresh_queued"}`; does not trigger ingestion |
| GET | `/demand-signals/external/decomposition` | Reads `mv_demand_decomposition` for a DFU; returns 404 today because nothing populates the table |

## Configuration

Signal sources are managed via the API (database-driven, not YAML). There is no scheduler wiring a source's `refresh_interval_hours` to an automated pull job today - refresh is manual-only, and the manual refresh endpoint is itself a stub.

## Dependencies

- No decomposition computation exists yet - no correlation or time-series decomposition library is wired into this router
- Optional: `httpx` for a future API-based signal-pull implementation (not currently imported by this router)

## See Also

- [Blended Demand](../03-demand-intelligence/03-blended-demand.md) - would incorporate signal-adjusted weights once decomposition is built
- [Notifications](./04-notifications.md) - natural delivery channel for refresh-completion alerts once ingestion is implemented
- [Integration Architecture](./01-integration-architecture.md) -- external signals as a data source vector
