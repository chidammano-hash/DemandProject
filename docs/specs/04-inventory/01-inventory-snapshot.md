# Inventory Snapshot & Backtest

> Two-layer architecture that loads raw daily inventory snapshots into a partitioned fact table and aggregates them into a monthly materialized view, with a bridge view attributing stockout and excess events to specific forecast models.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory |
| **Key Files** | `scripts/normalize_inventory_csv.py`, `scripts/load_dataset_postgres.py`, `sql/017_create_fact_inventory_snapshot.sql`, `sql/019_inventory_forecast_view.sql` |

---

## Problem

Inventory planning requires a reliable time-series of stock positions across all locations. Raw daily snapshots (~190M rows across 14 monthly CSVs) are too granular for planning and too large for direct querying. Without aggregation, planners cannot compute Days of Supply (DOS), stock-out frequency, or attribute forecast errors to inventory outcomes.

---

## Solution

A two-layer architecture: (1) raw daily snapshots loaded into `fact_inventory_snapshot`, (2) a monthly aggregation materialized view (`agg_inventory_monthly`) that derives daily sales from cumulative MTD via LAG(), computes EOM positions, and enables O(1) KPI queries. A bridge view (`mv_inventory_forecast_monthly`) joins inventory with forecast data to attribute stockout/excess events to specific forecast models.

---

## How It Works

### Ingestion Pipeline

14 monthly CSV files are merged by `normalize_inventory_csv.py` into a single clean CSV. `qty_on_order` is derived as `qty_on_hand_on_order - qty_on_hand` during normalization.

### Monthly Aggregation (agg_inventory_monthly)

| Metric | Derivation |
|---|---|
| EOM on-hand | Last snapshot of the month |
| EOM on-hand + on-order | Last snapshot of the month |
| Monthly sales | MAX of cumulative `mtd_sales` (not SUM -- MTD is cumulative) |
| Avg daily sales | Derived via LAG() window function on cumulative MTD |
| Snapshot days | COUNT of distinct snapshot dates |
| Latest lead time | Last non-null `lead_time_days` in the month |

### Inventory-Forecast Bridge (Feature 37)

`mv_inventory_forecast_monthly` joins `agg_inventory_monthly` + `fact_external_forecast_monthly` + `dim_sku` at grain: item_id + loc + month_start + model_id. Computed columns:

| Column | Formula | Purpose |
|---|---|---|
| `forecast_error` | forecast - actual | Signed error |
| `abs_error` | ABS(forecast - actual) | Absolute error |
| `dos` | on_hand / avg_daily_sales | Days of supply |
| `is_stockout` | on_hand <= 0 | Stockout flag |
| `is_excess` | dos > 90 | Excess flag |
| `bias_direction` | SIGN(forecast_error) | Over/under attribution |

### KPI Computation

`/inventory/kpis` uses a two-query pattern: point-in-time totals from the latest snapshot + trailing-month aggregates for supply chain KPIs (DOS, Weeks of Cover, Inventory Turns, Lead Time Coverage). KPI cards use severity color-coding (green/yellow/red thresholds).

---

## Data Model

| Table / View | Grain | Row Count | Notes |
|---|---|---|---|
| `fact_inventory_snapshot` | item_id + loc + snapshot_date | ~198M | Monthly range partitioned by `snapshot_date` |
| `agg_inventory_monthly` | item_id + loc + month | MV | Aggregates from partitioned parent |
| `mv_inventory_forecast_monthly` | item_id + loc + month + model_id | MV | Bridge view |

### Partitioning

`fact_inventory_snapshot` uses PostgreSQL declarative range partitioning on `snapshot_date`:
- **Partition granularity:** 1 calendar month per partition (~13M rows each)
- **Benefits:** Instant TRUNCATE per partition (no index rebuild), partition pruning on date queries, parallel partition scans
- **Auto-creation:** The loader creates new partitions automatically via `_ensure_partition_exists()` if data arrives for a month without a pre-existing partition
- **Default partition:** Catches out-of-range dates (should remain empty in normal operation)
- **No surrogate key:** `inventory_sk BIGSERIAL` was removed; uniqueness enforced by `UNIQUE(inventory_ck, snapshot_date)`

DDL: `sql/017_create_fact_inventory_snapshot.sql`, `sql/019_inventory_forecast_view.sql`

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inventory/position` | Paginated inventory positions |
| GET | `/inventory/kpis` | Portfolio KPIs (DOS, WOC, Turns, LT Coverage) |
| GET | `/inventory/trend` | Monthly trend (5 lines: on-hand, on-order, sales, LT, DOS) |
| GET | `/inventory/item-detail` | Single item drill-down |

Inventory backtest endpoints (Feature 37):

| Method | Path | Purpose |
|---|---|---|
| GET | `/inventory-backtest/summary` | Per-model outcome metrics: stockout rate, excess rate, cycle service level, WAPE |
| GET | `/inventory-backtest/trend` | Monthly stockout/excess/WAPE trend by model |
| GET | `/inventory-backtest/root-cause` | Bias-direction correlation with stockout/excess events for a single model (correlational, not causal) |
| GET | `/inventory-backtest/detail` | Paginated DFU-level stockout/excess events with forecast error detail |

---

## Pipeline

```
make inventory-pipeline    # normalize + load + refresh (all-in-one)
make refresh-inv-backtest  # Refresh inventory-forecast bridge view
```

| Step | Script | Output |
|---|---|---|
| Normalize | `scripts/normalize_inventory_csv.py` | Single merged CSV |
| Load | `scripts/load_dataset_postgres.py --dataset inventory` | `fact_inventory_snapshot` |
| Refresh agg | (auto on load) | `agg_inventory_monthly` refreshed |
| Refresh bridge | `sql/019_inventory_forecast_view.sql` | `mv_inventory_forecast_monthly` |

---

## Dependencies

- **Upstream:** 14 monthly CSV files in `data/input/`
- **Downstream:** Safety stock, EOQ, health scores, fill rate, demand signals, intramonth stockouts, rebalancing
- **Libraries:** pandas, psycopg

---

## See Also

- [02-demand-variability](02-demand-variability.md) -- Uses inventory aggregates for variability profiling
- [03-safety-stock](03-safety-stock.md) -- Consumes DOS and daily sales from agg view
- [06-analytics](06-analytics.md) -- Fill rate and intramonth stockouts from snapshot data
