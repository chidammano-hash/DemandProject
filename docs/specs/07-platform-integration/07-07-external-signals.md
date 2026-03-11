# 07-07 External Signals Ingestion

## Overview

Framework for ingesting external demand signals (weather, economic indicators, promotions, market indices) and decomposing their impact on forecast accuracy. Signals are stored with temporal alignment to the planning calendar and surfaced in the Demand Signals panel.

## Components

| Component | Path | Purpose |
|---|---|---|
| Router | `api/routers/external_signals.py` | 4 REST endpoints for signal management and decomposition |
| DDL | `sql/066_create_external_signals.sql` | `dim_signal_source`, `fact_external_signal`, `fact_signal_decomposition` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/external-signals` | List signals with filters (source, date range, item/loc) |
| GET | `/external-signals/sources` | List registered signal sources with metadata |
| POST | `/external-signals/refresh` | Trigger signal pull from configured sources (returns 202) |
| GET | `/external-signals/decomposition` | Signal impact decomposition for a DFU or aggregate |

## Database Schema

### `dim_signal_source`
- `source_id SERIAL PRIMARY KEY`
- `source_name TEXT UNIQUE NOT NULL` (e.g., "weather_noaa", "econ_fred", "promo_internal")
- `source_type TEXT NOT NULL` (weather, economic, promotion, market, custom)
- `refresh_method TEXT` (api_pull, file_upload, manual)
- `refresh_url TEXT`, `refresh_schedule TEXT` (cron expression)
- `is_active BOOLEAN DEFAULT true`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

### `fact_external_signal`
- `signal_id BIGSERIAL PRIMARY KEY`
- `source_id INTEGER REFERENCES dim_signal_source(source_id)`
- `signal_name TEXT NOT NULL` (e.g., "temperature_avg", "cpi_index", "promo_discount_pct")
- `signal_date DATE NOT NULL` (aligned to month-start for monthly signals)
- `item_no TEXT`, `loc TEXT` (nullable; NULL = global signal)
- `value NUMERIC NOT NULL`
- `unit TEXT` (e.g., "fahrenheit", "index", "percent")
- `metadata JSONB`
- `created_at TIMESTAMPTZ DEFAULT NOW()`
- Indexes: `(source_id, signal_date)`, `(item_no, loc, signal_date)`, `(signal_name)`
- Unique: `(source_id, signal_name, signal_date, item_no, loc)`

### `fact_signal_decomposition`
- `decomp_id BIGSERIAL PRIMARY KEY`
- `item_no TEXT NOT NULL`, `loc TEXT NOT NULL`
- `month_start DATE NOT NULL`
- `signal_name TEXT NOT NULL`
- `contribution_pct NUMERIC(7,4)` (percent of forecast variance explained)
- `correlation NUMERIC(7,4)` (Pearson correlation with demand)
- `lag_months INTEGER DEFAULT 0` (leading indicator offset)
- `computed_at TIMESTAMPTZ DEFAULT NOW()`
- Index: `(item_no, loc, month_start)`

## Signal Decomposition

The decomposition endpoint computes how much each external signal contributes to demand variation for a given DFU or aggregate:

1. Retrieve demand history and aligned signal values for the DFU
2. Compute rolling Pearson correlation between each signal and demand (with configurable lag window 0-3 months)
3. Run variance decomposition (R-squared attribution) across all signals
4. Return ranked signal contributions with correlation direction and optimal lag

## Integration Points

- Demand Signals panel (`DemandSignalsPanel.tsx`) extended to show external signals alongside internal velocity signals
- Blended Demand panel (`BlendedDemandPanel.tsx`) can incorporate signal-adjusted weights
- AI Planner agent tool `get_dfu_full_context` extended to include top external signal correlations
- Signal refresh integrated with job scheduler (`common/job_registry.py`)

## Make Targets

```bash
make ext-signals-schema     # Apply DDL (one-time)
make ext-signals-refresh    # Pull latest signals from configured sources
make ext-signals-decompose  # Compute signal decomposition for all DFUs
make ext-signals-all        # schema + refresh + decompose
```

## Dependencies

- `scipy.stats.pearsonr` for correlation computation (already in deps)
- Optional: `requests` or `httpx` for API-based signal pulls
