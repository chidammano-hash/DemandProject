# Feature 33: Inventory Snapshot Ingestion & Analytics

## Objective

Ingest daily inventory snapshot data (`inventory_snapshot.csv`) into a new fact table, surface inventory trends alongside sales and forecasts in the DFU Analysis tab, enrich DFU attributes with latest inventory position, and enable world-class supply chain inventory analytics — including days of supply, lead time coverage, inventory turns, and demand-supply gap analysis.

## Motivation

- **Demand-supply alignment:** Demand planners today see forecasts and history but have no visibility into whether inventory exists to fulfill demand. Adding inventory data closes the planning loop — planners can see if a forecast uplift is actionable given current stock levels.
- **Stock-out prevention:** Combining inventory position with demand forecasts identifies items at risk *before* they go out of stock, rather than reacting after the fact.
- **Safety stock optimization:** Days of Supply (DOS) and Weeks of Cover (WOC) metrics, computed from on-hand inventory and demand run rate, reveal where safety stock is insufficient or excessive.
- **Lead time coverage:** Comparing on-hand + on-order (`tot_oh_oo`) against `lead_time` × daily demand shows whether the supply pipeline can cover the replenishment cycle.
- **Inventory turns:** Annual demand divided by average inventory quantifies capital efficiency — a core supply chain KPI.
- **Forecast value added (FVA) extension:** Champion model selection (Feature 15) can incorporate inventory feasibility — the best forecast means nothing if there's no stock to ship.

## Scope

**In scope:**
- New fact table: `fact_inventory_snapshot` — daily grain inventory positions per item-location
- New domain spec: `INVENTORY_SPEC` in `common/domain_specs.py` (8th domain)
- Normalization rules for `inventory_snapshot.csv` in `normalize_dataset_csv.py`
- SQL DDL: `sql/016_create_fact_inventory_snapshot.sql`
- Materialized view: `agg_inventory_monthly` for monthly rollups aligned with sales/forecast grain
- Makefile targets: `normalize-inventory`, `load-inventory`, `refresh-agg-inventory`, `inventory-all`
- API: extend `/dfu/analysis` response with `inventory_series` and `inventory_kpis`
- DFU Analysis tab: new inventory chart section below existing sales/forecast overlay
- DFU attributes enrichment: latest inventory snapshot values joined to DFU attributes panel
- Update `feature2.md` ERD with `fact_inventory_snapshot`
- Comprehensive world-class supply chain analytics catalog (Section: World-Class Analytics)

**Out of scope:**
- Real-time or intra-day inventory feeds (daily snapshot is sufficient for planning)
- Allocation fields: `site_allocation`, `national_allocation` are excluded from ingestion
- DC fields: `dc_oh`, `dc_oh_oo` are excluded from ingestion
- SSC fields: `ssc_oh`, `ssc_oh_oo` are excluded from ingestion
- OOS field: `national_oos` is excluded from ingestion
- Lot-level or serial-level inventory tracking
- Inventory optimization engine (automated reorder point / safety stock calculation)
- Warehouse management system (WMS) integration
- Purchase order or replenishment order tracking
- Inventory valuation (cost, FIFO/LIFO)

## Architecture Overview

```
inventory_snapshot.csv (daily, ~July 2026+)
        ↓
normalize_dataset_csv.py --dataset inventory
  1. Parse exec_date (YYYYMMDD → YYYY-MM-DD)
  2. Validate item and loc are non-empty
  3. Drop excluded columns: site_allocation, national_allocation, dc_oh, dc_oh_oo, ssc_oh, ssc_oh_oo, national_oos
  4. Null-normalize blanks → NULL
  5. Cast numeric fields (lead_time, on-hand, on-order, mtd_sls)
  6. Output: data/inventory_snapshot_clean.csv
        ↓
load_dataset_postgres.py --dataset inventory
  1. Stage → type-cast → dedup by composite key
  2. Insert into fact_inventory_snapshot
  3. Refresh agg_inventory_monthly
        ↓
fact_inventory_snapshot (PostgreSQL)
        ↓
┌───────────────────────────────────────────────┐
│  agg_inventory_monthly (materialized view)     │
│  Monthly avg/end-of-month rollups for charts   │
└───────────────────────────────────────────────┘
        ↓
FastAPI /dfu/analysis (extended response)
        ↓
React DFU Analysis Tab
  - Inventory trend chart (OH, OH+OO, daily sales, DOS)
  - DOS / WOC computed KPIs
  - DFU attributes panel with latest inventory position
```

## Source File Specification

**File:** `datafiles/inventory_snapshot.csv` (comma-delimited)

**Frequency:** Daily snapshots, expected start ~July 2026

### Source Fields

| Source Column | Description | Load? |
|---------------|-------------|-------|
| `exec_date` | Snapshot date (daily grain) | **Yes** |
| `item` | Item code (maps to `dmdunit` in dim_item/dim_dfu) | **Yes** |
| `loc` | Location code (maps to `loc` in dim_location/dim_dfu) | **Yes** |
| `lead_time` | Replenishment lead time in days | **Yes** |
| `site_allocation` | Quantity allocated at the location/site level | **No** (excluded) |
| `national_allocation` | Quantity allocated at the national/supplier level | **No** (excluded) |
| `tot_oh` | Total on-hand inventory (all locations) on that day | **Yes** |
| `tot_oh_oo` | Total on-hand + on-order inventory on that day | **Yes** |
| `dc_oh` | On-hand inventory at distribution center(s) | **No** (excluded) |
| `dc_oh_oo` | On-hand + on-order at distribution center(s) | **No** (excluded) |
| `mtd_sls` | Month-to-date sales (cumulative qty sold this month) | **Yes** |
| `ssc_oh` | Secondary supply chain on-hand | **No** (excluded) |
| `ssc_oh_oo` | Secondary supply chain on-hand + on-order | **No** (excluded) |
| `national_oos` | National out-of-stock indicator | **No** (excluded) |

### Loaded Columns (Clean CSV)

| Clean Column | Type | Description |
|-------------|------|-------------|
| `item` | TEXT | Item code (NOT NULL) |
| `loc` | TEXT | Location code (NOT NULL) |
| `exec_date` | DATE | Snapshot date (NOT NULL, daily grain) |
| `lead_time` | INTEGER | Replenishment lead time in days |
| `tot_oh` | NUMERIC(18,4) | Total on-hand inventory at this location |
| `tot_oh_oo` | NUMERIC(18,4) | Total on-hand + on-order at this location |
| `mtd_sls` | NUMERIC(18,4) | Month-to-date cumulative sales at this location |

### Derived Fields (computed in SQL views or API, not stored in fact table)

| Field | Formula | Description |
|-------|---------|-------------|
| `daily_sls` | See daily sales derivation algorithm below | Actual single-day sales quantity |
| `on_order` | `tot_oh_oo - tot_oh` | Quantity currently on order (in transit) |

### Daily Sales Derivation from `mtd_sls`

The `mtd_sls` field is a **cumulative month-to-date** counter that resets on the 1st of each month. To get actual daily sales, apply this logic:

```
For each (item, loc, exec_date) ordered by exec_date:
  IF exec_date is the 1st day of the month:
    daily_sls = mtd_sls                           -- first day: MTD IS the daily sales
  ELSE:
    daily_sls = mtd_sls - previous_day_mtd_sls    -- subsequent days: delta from prior day
```

**Edge cases to handle:**
- **First day of month:** `daily_sls = mtd_sls` (the MTD value IS that day's sales since the counter just reset)
- **Missing previous day:** If a snapshot day is missing (gap in data), use the most recent prior day's `mtd_sls` for the delta. If no prior day exists in the month, treat `daily_sls = mtd_sls`.
- **Negative delta:** If `mtd_sls` decreases day-over-day (data correction or return), keep the negative value — it represents a reversal/return.
- **Month boundary:** Never subtract across month boundaries. Each month's 1st day always uses `mtd_sls` directly.

**SQL window function implementation:**

```sql
-- Derived daily sales via window function (partitioned by item + loc + month)
daily_sls = CASE
    WHEN EXTRACT(day FROM exec_date) = 1 THEN mtd_sls
    ELSE mtd_sls - LAG(mtd_sls) OVER (
        PARTITION BY item, loc, date_trunc('month', exec_date)
        ORDER BY exec_date
    )
END
```

This is computed in the materialized view and API queries — **not stored** in the fact table (since it's derivable from `mtd_sls`).

## Data Model

### Fact Table: `fact_inventory_snapshot`

**Grain:** One row per `item` + `loc` + `exec_date` (daily snapshot per item-location)

**Composite key:** `inventory_ck = item_loc_exec_date` (e.g., `100320_6601-KAPO_2026-07-15`)

```
┌─────────────────────────────────────────────────┐
│           fact_inventory_snapshot                 │
├─────────────────────────────────────────────────┤
│ inventory_sk       BIGSERIAL PRIMARY KEY         │
│ inventory_ck       TEXT UNIQUE NOT NULL           │
│ ─────────────────── grain ───────────────────    │
│ item               TEXT NOT NULL                  │
│ loc                TEXT NOT NULL                  │
│ exec_date          DATE NOT NULL                  │
│ ─────────────────── measures ────────────────    │
│ lead_time          INTEGER                        │
│ tot_oh             NUMERIC(18,4)                  │
│ tot_oh_oo          NUMERIC(18,4)                  │
│ mtd_sls            NUMERIC(18,4)                  │
│ ─────────────────── audit ───────────────────    │
│ load_ts            TIMESTAMPTZ DEFAULT NOW()      │
│ modified_ts        TIMESTAMPTZ DEFAULT NOW()      │
└─────────────────────────────────────────────────┘

Indexes:
  idx_fact_inv_item           ON (item)
  idx_fact_inv_loc            ON (loc)
  idx_fact_inv_exec_date      ON (exec_date)
  idx_fact_inv_item_loc_date  ON (item, loc, exec_date)  — covering index for time series

Constraints:
  UNIQUE (inventory_ck)
```

### Materialized View: `agg_inventory_monthly`

Roll up daily snapshots to monthly grain for alignment with `agg_sales_monthly` and `agg_forecast_monthly`.

```sql
CREATE MATERIALIZED VIEW agg_inventory_monthly AS
WITH daily AS (
    -- Derive daily sales from cumulative MTD sales
    SELECT *,
        CASE
            WHEN EXTRACT(day FROM exec_date) = 1 THEN mtd_sls
            ELSE mtd_sls - LAG(mtd_sls) OVER (
                PARTITION BY item, loc, date_trunc('month', exec_date)
                ORDER BY exec_date
            )
        END AS daily_sls
    FROM fact_inventory_snapshot
)
SELECT
    item AS dmdunit,
    loc,
    date_trunc('month', exec_date)::date AS month_start,
    -- Averages over the month (smoothed position)
    AVG(tot_oh)::numeric(18,4)              AS avg_tot_oh,
    AVG(tot_oh_oo)::numeric(18,4)           AS avg_tot_oh_oo,
    -- End-of-month snapshot (point-in-time position)
    (ARRAY_AGG(tot_oh ORDER BY exec_date DESC))[1]::numeric(18,4)    AS eom_tot_oh,
    (ARRAY_AGG(tot_oh_oo ORDER BY exec_date DESC))[1]::numeric(18,4) AS eom_tot_oh_oo,
    -- Sales: total monthly (= last day MTD) and avg daily
    MAX(mtd_sls)::numeric(18,4)              AS monthly_sls,
    AVG(daily_sls)::numeric(18,4)            AS avg_daily_sls,
    -- Snapshot count (for partial month handling)
    COUNT(*)::integer                                                  AS snapshot_days,
    -- Lead time (use last known value per month)
    (ARRAY_AGG(lead_time ORDER BY exec_date DESC))[1]::integer       AS lead_time
FROM daily
GROUP BY 1, 2, 3;

CREATE UNIQUE INDEX idx_agg_inv_monthly_pk ON agg_inventory_monthly (dmdunit, loc, month_start);
```

**Key design decisions:**
- **Item-location grain:** The inventory data is at the item+location level, matching the DFU grain in sales and forecast tables. This enables per-location inventory analytics and direct joins with `agg_sales_monthly` and `agg_forecast_monthly` on `(dmdunit, loc, month_start)`.
- **Daily sales derivation:** The CTE computes `daily_sls` from cumulative `mtd_sls` using `LAG()` partitioned by item, loc, and month. Day 1 of each month uses `mtd_sls` directly (counter just reset). Subsequent days use the delta from the prior day.
- **Monthly sales:** `MAX(mtd_sls)` gives the total monthly sales (last cumulative value). `AVG(daily_sls)` gives the average daily sales rate for the month — used directly in DOS/WOC calculations without needing to join `agg_sales_monthly`.
- **Avg and EOM:** Both average (smoothed) and end-of-month (point-in-time) measures are materialized. Average is better for DOS/WOC computation; EOM is better for "where do we stand today" reporting.
- **Snapshot days:** Total observation days per month (handles partial months correctly in DOS calculations).
- **Join key:** `(dmdunit, loc, month_start)` aligns directly with `agg_sales_monthly` and `agg_forecast_monthly` for demand-supply joins.
- **Self-contained DOS:** With `avg_daily_sls` available in the same view, DOS can be computed without joining to `agg_sales_monthly`: `DOS = avg_tot_oh / avg_daily_sls`. This gives a more accurate daily-resolution DOS vs. the monthly approximation.

## Domain Spec

Add `INVENTORY_SPEC` to `common/domain_specs.py`:

```python
INVENTORY_SPEC = DomainSpec(
    name="inventory",
    plural="inventory",
    table="fact_inventory_snapshot",
    ck_field="inventory_ck",
    business_key_field="item",
    business_key_fields=("item", "loc", "exec_date"),
    business_key_separator="_",
    columns=[
        "item",
        "loc",
        "exec_date",
        "lead_time",
        "tot_oh",
        "tot_oh_oo",
        "mtd_sls",
    ],
    source_file="inventory_snapshot.csv",
    clean_file="inventory_snapshot_clean.csv",
    search_fields=["item", "loc"],
    int_fields={"lead_time"},
    float_fields={"tot_oh", "tot_oh_oo", "mtd_sls"},
    date_fields={"exec_date"},
    default_sort="exec_date",
)
```

Register in `ALL_SPECS` / `get_spec()` alongside the existing 7 domains. This makes the inventory domain immediately queryable via the generic API:
- `GET /domains/inventory/rows` — paginated daily snapshots
- `GET /domains/inventory/search?q=100320` — search by item or location
- `GET /domains/inventory/rows?sort=exec_date&order=desc` — latest snapshots first

## Normalization Rules

Add inventory-specific handling in `normalize_dataset_csv.py`:

```python
if spec.name == "inventory":
    # 1. Parse exec_date
    out["exec_date"] = to_iso_date_yyyymmdd(out.get("exec_date", ""))
    if not out["exec_date"]:
        continue  # skip rows with invalid dates

    # 2. Require non-empty item and loc
    if not out.get("item", "").strip():
        continue
    if not out.get("loc", "").strip():
        continue

    # 3. Drop excluded columns (not loaded)
    out.pop("site_allocation", None)
    out.pop("national_allocation", None)
    out.pop("dc_oh", None)
    out.pop("dc_oh_oo", None)
    out.pop("ssc_oh", None)
    out.pop("ssc_oh_oo", None)
    out.pop("national_oos", None)
```

## SQL DDL

**File:** `mvp/demand/sql/016_create_fact_inventory_snapshot.sql`

```sql
-- Feature 33: Inventory Snapshot Fact Table
CREATE TABLE IF NOT EXISTS fact_inventory_snapshot (
    inventory_sk        BIGSERIAL PRIMARY KEY,
    inventory_ck        TEXT UNIQUE NOT NULL,
    item                TEXT NOT NULL,
    loc                 TEXT NOT NULL,
    exec_date           DATE NOT NULL,
    lead_time           INTEGER,
    tot_oh              NUMERIC(18,4),
    tot_oh_oo           NUMERIC(18,4),
    mtd_sls             NUMERIC(18,4),
    load_ts             TIMESTAMPTZ DEFAULT NOW(),
    modified_ts         TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_fact_inv_item
    ON fact_inventory_snapshot (item);
CREATE INDEX IF NOT EXISTS idx_fact_inv_loc
    ON fact_inventory_snapshot (loc);
CREATE INDEX IF NOT EXISTS idx_fact_inv_exec_date
    ON fact_inventory_snapshot (exec_date);
CREATE INDEX IF NOT EXISTS idx_fact_inv_item_loc_date
    ON fact_inventory_snapshot (item, loc, exec_date);

-- Column documentation
COMMENT ON TABLE  fact_inventory_snapshot IS 'Daily inventory position snapshots per item-location';
COMMENT ON COLUMN fact_inventory_snapshot.exec_date IS 'Snapshot date (daily grain)';
COMMENT ON COLUMN fact_inventory_snapshot.item IS 'Item code (maps to dmdunit in dim_item/dim_dfu)';
COMMENT ON COLUMN fact_inventory_snapshot.loc IS 'Location code (maps to loc in dim_location/dim_dfu)';
COMMENT ON COLUMN fact_inventory_snapshot.lead_time IS 'Replenishment lead time in days';
COMMENT ON COLUMN fact_inventory_snapshot.tot_oh IS 'On-hand inventory at this item-location';
COMMENT ON COLUMN fact_inventory_snapshot.tot_oh_oo IS 'On-hand + on-order inventory at this item-location';
COMMENT ON COLUMN fact_inventory_snapshot.mtd_sls IS 'Month-to-date cumulative sales; daily_sls = delta from prior day (day 1 = mtd_sls itself)';

-- Materialized view: monthly rollup with daily sales derivation
-- Daily sales derived from cumulative mtd_sls via LAG() window function
CREATE MATERIALIZED VIEW IF NOT EXISTS agg_inventory_monthly AS
WITH daily AS (
    SELECT *,
        CASE
            WHEN EXTRACT(day FROM exec_date) = 1 THEN mtd_sls
            ELSE mtd_sls - LAG(mtd_sls) OVER (
                PARTITION BY item, loc, date_trunc('month', exec_date)
                ORDER BY exec_date
            )
        END AS daily_sls
    FROM fact_inventory_snapshot
)
SELECT
    item AS dmdunit,
    loc,
    date_trunc('month', exec_date)::date AS month_start,
    AVG(tot_oh)::numeric(18,4)              AS avg_tot_oh,
    AVG(tot_oh_oo)::numeric(18,4)           AS avg_tot_oh_oo,
    (ARRAY_AGG(tot_oh ORDER BY exec_date DESC))[1]::numeric(18,4)    AS eom_tot_oh,
    (ARRAY_AGG(tot_oh_oo ORDER BY exec_date DESC))[1]::numeric(18,4) AS eom_tot_oh_oo,
    MAX(mtd_sls)::numeric(18,4)              AS monthly_sls,
    AVG(daily_sls)::numeric(18,4)            AS avg_daily_sls,
    COUNT(*)::integer                                                  AS snapshot_days,
    (ARRAY_AGG(lead_time ORDER BY exec_date DESC))[1]::integer       AS lead_time
FROM daily
GROUP BY 1, 2, 3;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_inv_monthly_pk
    ON agg_inventory_monthly (dmdunit, loc, month_start);
```

## Makefile Targets

```makefile
# Inventory snapshot pipeline
normalize-inventory:
	$(UV) python scripts/normalize_dataset_csv.py --dataset inventory

load-inventory:
	$(UV) python scripts/load_dataset_postgres.py --dataset inventory
	$(MAKE) refresh-agg-inventory

refresh-agg-inventory:
	docker exec -i demand-mvp-postgres psql -U demand -d demand_mvp \
	    -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW agg_inventory_monthly;"

inventory-all: normalize-inventory load-inventory
```

Add `016_create_fact_inventory_snapshot.sql` to `db-apply-sql` target. Add `normalize-inventory` to `normalize-all`. Add `load-inventory` to `load-all`. Add `refresh-agg-inventory` to `refresh-agg`.

## API Integration

### Extend `/dfu/analysis` Response (Recommended)

Add `inventory_series` and `inventory_kpis` to the existing DFU analysis response. This keeps all DFU-level analytics in one API call and one TanStack Query cache entry.

**New query parameter:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_inventory` | bool | `true` | Whether to include inventory data (allows opt-out for performance) |

**Extended response shape:**

```json
{
  "mode": "item_location",
  "item": "100320",
  "location": "1401-BULK",
  "points": 36,
  "models": ["external", "lgbm_global", "champion"],
  "series": [ ... ],
  "model_monthly": { ... },
  "dfu_attributes": [ ... ],
  "inventory_series": [
    {
      "month": "2026-07-01",
      "avg_tot_oh": 1500.0,
      "avg_tot_oh_oo": 2200.0,
      "eom_tot_oh": 1450.0,
      "eom_tot_oh_oo": 2150.0,
      "monthly_sls": 600.0,
      "avg_daily_sls": 20.0,
      "snapshot_days": 31,
      "lead_time": 14,
      "dos": 45.2,
      "woc": 6.5
    }
  ],
  "inventory_kpis": {
    "current_dos": 45.2,
    "current_woc": 6.5,
    "avg_inventory_turns": 8.3,
    "lead_time_coverage_ratio": 2.1
  }
}
```

**SQL for inventory series (add to `api/routers/analysis.py`):**

The inventory query uses the same `WHERE` clause logic as the existing sales/forecast queries, since the inventory data is at the same item+location grain:

```python
# Fetch monthly inventory from materialized view
# Reuses the same where_parts / params logic from the existing analysis endpoint
inventory_where_parts: list[str] = []
inventory_params: list[Any] = []
if mode == "item_location":
    inventory_where_parts.append("dmdunit = %s")
    inventory_params.append(item_val)
    inventory_where_parts.append("loc = %s")
    inventory_params.append(loc_val)
elif mode == "all_items_at_location":
    inventory_where_parts.append("loc = %s")
    inventory_params.append(loc_val)
elif mode == "item_at_all_locations":
    inventory_where_parts.append("dmdunit = %s")
    inventory_params.append(item_val)

inv_where_sql = "WHERE " + " AND ".join(inventory_where_parts) if inventory_where_parts else ""

inventory_sql = f"""
    SELECT month_start,
           SUM(avg_tot_oh)::double precision       AS avg_tot_oh,
           SUM(avg_tot_oh_oo)::double precision    AS avg_tot_oh_oo,
           SUM(eom_tot_oh)::double precision       AS eom_tot_oh,
           SUM(eom_tot_oh_oo)::double precision    AS eom_tot_oh_oo,
           SUM(monthly_sls)::double precision       AS monthly_sls,
           AVG(avg_daily_sls)::double precision     AS avg_daily_sls,
           SUM(snapshot_days)::integer              AS snapshot_days,
           AVG(lead_time)::integer                  AS lead_time
    FROM agg_inventory_monthly
    {inv_where_sql}
    GROUP BY 1 ORDER BY 1 ASC
"""
cur.execute(inventory_sql, inventory_params)
```

Note: `SUM` aggregation handles the `all_items_at_location` mode (summing across items) and `item_at_all_locations` mode (summing across locations). For `item_location` mode with a single DFU, the SUM is a no-op.

**DOS/WOC computation (server-side):**

```python
# Compute DOS/WOC using avg_daily_sls derived from mtd_sls (self-contained, no sales join needed)
for inv in inventory_rows:
    daily_demand = inv.get("avg_daily_sls", 0) or 0
    inv["dos"] = round(inv["avg_tot_oh"] / daily_demand, 1) if daily_demand > 0 else None
    inv["woc"] = round(inv["dos"] / 7, 1) if inv["dos"] else None
```

**Inventory KPIs computation:**

```python
# Current DOS/WOC (latest month with data)
current_dos = inventory_rows[-1]["dos"] if inventory_rows else None
current_woc = inventory_rows[-1]["woc"] if inventory_rows else None

# Inventory turns (last 12 months) — using monthly_sls from inventory view
total_demand_12m = sum(r.get("monthly_sls", 0) or 0 for r in inventory_rows[-12:])
avg_inventory_12m = mean([r["avg_tot_oh"] for r in inventory_rows[-12:]]) if inventory_rows else 0
turns = round(total_demand_12m / avg_inventory_12m, 1) if avg_inventory_12m > 0 else None

# Lead time coverage ratio
latest = inventory_rows[-1] if inventory_rows else {}
lt = latest.get("lead_time", 0) or 0
daily_d = latest.get("avg_daily_sls", 0) or 0
lt_coverage = round(latest.get("eom_tot_oh_oo", 0) / (lt * daily_d), 2) if (lt * daily_d) > 0 else None
```

### Mode Behavior

Since inventory data is at the item+location grain (matching sales and forecast), all three DFU analysis modes work naturally:

| Mode | Inventory Behavior |
|------|-------------------|
| `item_location` | Inventory filtered by `dmdunit = item` AND `loc = location`. Shows the exact item-location inventory position. |
| `all_items_at_location` | Inventory filtered by `loc = location`, aggregated (SUM) across all items. Shows total inventory at the location. |
| `item_at_all_locations` | Inventory filtered by `dmdunit = item`, aggregated (SUM) across all locations. Shows total network inventory for the item. |

## DFU Analysis Tab: Inventory Chart

### New Chart Section

Add a second chart below the existing sales/forecast overlay chart. The inventory chart uses the same time axis (month) for visual alignment.

```
┌─────────────────────────────────────────────────────────┐
│  Sales & Forecast Overlay (existing)                     │
│  ────────────────────────────────────────                │
│  Lines: qty_shipped, qty_ordered, tothist_dmd,           │
│         forecast_external, forecast_lgbm, ...            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Inventory Position (NEW)                                │
│  ────────────────────────────────────────                │
│  Left Y-axis: Quantity (cases)                           │
│  Right Y-axis: Days of Supply                            │
│                                                          │
│  Area: Total OH+OO (background, translucent green)       │
│  Area: Total OH (foreground, solid green fill)            │
│  Line: Daily Sales (orange)                              │
│  Line: DOS (right axis, bold pink)                       │
│                                                          │
│  Toggles: [OH] [OH+OO] [Daily Sales] [DOS] [On Order]   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Inventory KPI Cards (NEW)                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ DOS: 45d │ │ WOC: 6.5 │ │ Turns: 8 │ │LT Cvr: 2.1│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Inventory Measure Toggles

| Measure | Color | Style | Y-Axis | Default On |
|---------|-------|-------|--------|------------|
| Total OH | `#4CAF50` (green) | Area (solid fill) | Left | Yes |
| Total OH+OO | `#81C784` (light green) | Area (translucent) | Left | Yes |
| Daily Sales | `#FF9800` (orange) | Line | Left | Yes |
| On Order | `#2196F3` (blue) | Dashed line | Left | No |
| DOS | `#E91E63` (pink) | Bold line | Right | Yes |

### Inventory KPI Card Definitions

| KPI | Formula | Unit | Good | Warning | Critical |
|-----|---------|------|------|---------|----------|
| **Days of Supply** | `tot_oh / avg_daily_sls` | days | 14–60 | 7–14 or 60–90 | <7 or >90 |
| **Weeks of Cover** | `DOS / 7` | weeks | 2–8 | 1–2 or 8–12 | <1 or >12 |
| **Inventory Turns** | `annual_demand / avg_inventory` | turns/yr | >8 | 4–8 | <4 |
| **LT Coverage** | `tot_oh_oo / (lead_time × avg_daily_sls)` | ratio | >1.5 | 1.0–1.5 | <1.0 |

### Synchronized Time Axis

The inventory chart shares the same `From`/`To` month range selectors and `Points` control as the sales/forecast chart. When the user adjusts the time window, both charts update together.

## DFU Attributes Enrichment

Extend the DFU attributes panel in the DFU Analysis tab with latest inventory position. Join the most recent inventory snapshot to the DFU attributes response:

```sql
-- Latest inventory snapshot for the item-location
SELECT
    exec_date AS latest_inventory_date,
    lead_time,
    tot_oh AS current_tot_oh,
    tot_oh_oo AS current_tot_oh_oo,
    mtd_sls AS current_mtd_sls
FROM fact_inventory_snapshot
WHERE item = %s AND loc = %s
ORDER BY exec_date DESC
LIMIT 1
```

For `item_at_all_locations` mode (no specific location), aggregate across locations:

```sql
-- Latest inventory snapshot aggregated across all locations for the item
SELECT
    exec_date AS latest_inventory_date,
    AVG(lead_time)::integer AS lead_time,
    SUM(tot_oh) AS current_tot_oh,
    SUM(tot_oh_oo) AS current_tot_oh_oo,
    SUM(mtd_sls) AS current_mtd_sls
FROM fact_inventory_snapshot
WHERE item = %s AND exec_date = (
    SELECT MAX(exec_date) FROM fact_inventory_snapshot WHERE item = %s
)
GROUP BY exec_date
```

These fields appear in the collapsible DFU Attributes panel alongside existing fields (brand, category, cluster, seasonality, etc.) under a new "Inventory Position" subsection.

## feature2.md ERD Update

Add to the **Silver Fact Tables** section of `docs/design-specs/feature2.md`:

```markdown
#### `silver.fact_inventory_snapshot`
Daily inventory position snapshots per item-location. Grain: `item, loc, exec_date`.
Measures: `lead_time`, `tot_oh`, `tot_oh_oo`, `mtd_sls` (cumulative month-to-date sales, daily sales derived via LAG window function).
Materialized view: `agg_inventory_monthly` rolls up to monthly grain with avg/EOM measures and snapshot day counts.
Joins to `agg_sales_monthly` and `agg_forecast_monthly` on `(dmdunit, loc, month_start)`.
```

---

## World-Class Supply Chain Analytics

The following analytics capabilities become possible with inventory snapshot data combined with existing sales, forecasts, and DFU attributes. These represent the standard analytics toolkit of enterprise supply chain platforms (Blue Yonder, Kinaxis, o9 Solutions, SAP IBP).

### Tier 1: Core Inventory Health Metrics

#### 1. Days of Supply (DOS)
```
DOS = On-Hand Inventory / Average Daily Sales
    = tot_oh / avg_daily_sls
```
- The single most important inventory metric in demand planning
- **< 7 DOS:** critically low, stock-out imminent
- **7–30 DOS:** healthy range (varies by category)
- **> 90 DOS:** excess inventory, capital at risk
- **Trend visualization:** DOS over time reveals whether inventory is being drawn down or building
- **Forward-looking DOS:** Use champion forecast instead of historical demand: `tot_oh / (next_month_champion_forecast / days_in_month)` — more actionable for planning

#### 2. Weeks of Cover (WOC)
```
WOC = DOS / 7
```
- Preferred metric in FMCG/CPG for communicating with commercial teams
- Can be computed against forecast (forward-looking) or history (backward-looking)
- **Forward WOC** = `tot_oh / (next_month_forecast / days_in_next_month) / 7`

#### 3. Inventory Turns
```
Annual Turns = Annual Demand / Average Inventory
             = SUM(last_12m_qty_shipped) / AVG(last_12m_tot_oh)
```
- Higher is better (more capital-efficient)
- Industry benchmarks: Grocery 12–20, Beverage 8–15, Spirits 4–8
- Track monthly for trend analysis

#### 4. Fill Rate Proxy
```
Fill Rate ≈ 1 - (Unfilled Demand / Total Demand)
```
- When true order-level fill rate data isn't available, approximate using inventory position vs. demand
- Items where `tot_oh` < `daily_demand × lead_time` are at risk of partial fills

### Tier 2: Supply Pipeline Analytics

#### 5. Lead Time Coverage Ratio
```
LT Coverage = tot_oh_oo / (lead_time × avg_daily_sls)
```
- **Ratio > 1.5:** supply pipeline safely covers the lead time
- **Ratio 1.0–1.5:** tight, any demand spike or supply delay causes stock-out
- **Ratio < 1.0:** will run out before replenishment arrives — urgent
- **Heatmap visualization:** Items × months colored by LT coverage ratio

#### 6. On-Order Visibility
```
On Order = tot_oh_oo - tot_oh
```
- Reveals the supply pipeline: how much is in transit or on PO
- Sudden drops in on-order signal supplier disruptions
- Ratio `on_order / tot_oh` shows pipeline dependency
- Trend over time reveals supply pipeline consistency

### Tier 3: Demand-Supply Integration

#### 7. Demand-Supply Gap Analysis
```
Gap = Forecast - Available Inventory (tot_oh_oo)
    = next_month_champion_forecast - current_tot_oh_oo
```
- **Positive gap:** demand exceeds available supply — action needed
- **Negative gap:** sufficient supply to cover demand
- Time series of gaps reveals persistent supply shortages vs. one-time blips
- **Integration with Feature 15 (Champion Selection):** Use champion forecast for the most accurate gap estimate

#### 8. Forecast Bias Impact on Inventory
```
Excess Inventory from Bias = Σ(Forecast - Actual) over trailing 12 months
```
- Combine forecast bias (Feature 5) with inventory data to quantify the practical cost of systematic over-forecasting
- High positive bias + high inventory = concrete evidence that over-forecasting is causing excess stock
- High negative bias + low inventory = under-forecasting causing stock-outs

#### 9. Safety Stock Adequacy
```
Safety Stock = tot_oh - (avg_daily_demand × lead_time)
SS Days = Safety Stock / avg_daily_demand
```
- Negative safety stock: will stock out before replenishment without above-average demand
- Compare SS Days across items to prioritize replenishment actions
- Combine with `seasonality_profile` (Feature 30): seasonal items need higher SS before peak months

### Tier 5: Strategic & Advanced Analytics

#### 10. Seasonal Pre-Build Tracking
```
Pre-Build Index = current_month_tot_oh / avg_tot_oh_same_month_prior_years
```
- For items with `seasonality_profile = 'high'`, track whether inventory is building ahead of peak season
- Compare current inventory trajectory against the pre-build trajectory from prior years
- Alert if pre-build is behind schedule vs. prior year

#### 11. Bullwhip Effect Detection
```
Bullwhip Ratio = Variance(Inventory Changes) / Variance(Demand Changes)
```
- Measures amplification of demand variability through the supply chain
- **Ratio > 1.0:** inventory swings exceed demand swings — classic bullwhip
- Computed over rolling 6–12 month windows
- High bullwhip items benefit from demand sensing / smoothing algorithms

#### 12. ABC-Inventory Cross Analysis
```
Matrix: ABC Volume Class × Inventory Health (DOS)
  A-items with high DOS → over-stock risk on high-revenue items (capital trap)
  A-items with low DOS  → stock-out risk on high-revenue items (revenue risk)
  C-items with high DOS → candidate for delisting or markdown
  C-items with low DOS  → low priority (acceptable stock-out risk)
```
- Classic supply chain prioritization matrix
- Uses `abc_vol` from `dim_dfu` (Feature 3) + DOS from inventory

#### 13. Inventory Velocity Segmentation
```
Velocity = Monthly Demand / Average Inventory
Segments: Fast (>2.0), Medium (0.5–2.0), Slow (<0.5), Dead (0 demand, >0 OH)
```
- Complements ABC classification with a velocity dimension
- **Dead stock detection:** items with zero demand for 3+ months but positive on-hand
- Velocity × seasonality matrix reveals items that are seasonally dead vs. permanently dead

#### 14. Cash-to-Serve Optimization (requires cost data — future enhancement)
```
Inventory Carrying Cost = avg_tot_oh × unit_cost × carrying_rate
Revenue at Risk from Stockout = stockout_days × daily_demand × unit_price
Net Inventory Value = Revenue_at_Risk - Carrying_Cost
```
- Quantifies the trade-off: cost of holding vs. cost of not holding
- Optimization target: minimize total cost of carrying + stockouts

#### 15. Supplier Performance Proxy
```
Lead Time Stability = STDDEV(lead_time) over rolling 30-day window
On-Order Fulfillment = (tot_oh_oo - tot_oh) trend over time
```
- Increasing lead time variability signals supply chain instability
- Declining on-order levels may suggest supplier capacity constraints
- Can be correlated with external events (Feature 18: Market Intelligence)

#### 16. Service Level Simulation
```
Simulated Fill Rate = P(demand ≤ available_inventory) over historical distribution
```
- Using historical demand distribution + current inventory position
- "What-if" analysis: what fill rate would we achieve at 10% more / less inventory?
- Connects inventory investment decisions to customer service outcomes

#### 17. Cluster-Level Inventory Benchmarking
```
DOS by Cluster = GROUP BY cluster_assignment, AVG(dos)
Turns by Cluster = GROUP BY cluster_assignment, AVG(turns)
```
- Uses `cluster_assignment` from Feature 7 (DFU Clustering)
- "High-volume steady" cluster should have different inventory targets than "seasonal spiky"
- Identifies clusters that are systematically over- or under-inventoried

#### 18. Projected Inventory (future enhancement — requires PO data)
```
Projected Inventory(month+N) = Current OH + Incoming Orders - Σ(Forecasted Demand for months 1..N)
```
- Project forward N months using champion forecast + current on-order
- Identify future stock-outs before they happen
- Requires integration with purchase order/replenishment data

#### 19. Inventory Health Score (composite)
```
Health Score = weighted combination of:
  - DOS within target range (weight: 0.35)
  - LT Coverage > 1.0 (weight: 0.25)
  - No bullwhip (ratio < 1.5) (weight: 0.20)
  - Turns above industry benchmark (weight: 0.20)
```
- Single 0–100 score per item summarizing overall inventory health
- Enables ranking and exception-based management: "show me the 50 worst-scoring items"
- Color-coded: green (80+), yellow (50–80), red (<50)

---

## Relationship to Existing Features

| Feature | Relationship |
|---------|-------------|
| Feature 2 (Data Architecture) | Add `fact_inventory_snapshot` to the ERD as a new silver-layer fact table. Add `agg_inventory_monthly` as a gold-layer materialized view. |
| Feature 3 (Dimensions) | `item` maps to `dmdunit` in `dim_dfu` / `dim_item`; `loc` maps to `loc` in `dim_location` / `dim_dfu`. Same item-location grain enables direct joins for attribute enrichment (brand, category, ABC, region, state). |
| Feature 4 (Fact Tables) | Third fact table alongside `fact_sales_monthly` and `fact_external_forecast_monthly`. Follows same conventions: `_sk`, `_ck`, `load_ts`, `modified_ts`, null normalization. |
| Feature 5 (Forecast Accuracy) | DOS and WOC require demand data from sales facts. Forecast bias × inventory reveals dollar cost of forecast error. |
| Feature 7 (Clustering) | Cluster-level inventory benchmarking. Different clusters should have different DOS/WOC targets. |
| Feature 15 (Champion Selection) | Champion forecast provides the most accurate demand signal for forward-looking DOS/WOC. Gap analysis uses champion forecast. |
| Feature 17 (DFU Analysis) | Inventory chart added below the existing sales/forecast overlay. Inventory KPI cards alongside forecast accuracy KPIs. DFU attributes enriched with latest inventory. |
| Feature 30 (Seasonality) | Seasonal pre-build tracking. Seasonal items need higher safety stock before peak months. DOS targets should vary by `seasonality_profile`. |

## Dependencies

- **Required:** Feature 3 (`dim_dfu`, `dim_item` for attribute joins and item mapping)
- **Required:** Feature 4 (`fact_sales_monthly` for demand data in DOS/WOC calculations)
- **Required:** Feature 17 (DFU Analysis tab for chart integration)
- **Optional:** Feature 15 (Champion forecast for forward-looking DOS)
- **Optional:** Feature 30 (Seasonality profile for seasonal pre-build analytics)
- **Python packages (all in stack):** `pandas`, `psycopg`, `csv`
- **No new dependencies required**

## Testing & Validation

1. **Load test:** Ingest a sample `inventory_snapshot.csv` with 30 days × 100 item-locations = 3,000 rows. Verify `fact_inventory_snapshot` row count matches.
2. **Dedup test:** Load the same file twice. Verify no duplicate `inventory_ck` values (UNIQUE constraint enforced).
3. **Materialized view:** After load, verify `agg_inventory_monthly` contains one row per item-location per month. Verify `avg_tot_oh` is between `MIN(tot_oh)` and `MAX(tot_oh)` for the month.
4. **EOM snapshot:** Verify `eom_tot_oh` matches the `tot_oh` from the last `exec_date` in each item-location-month group.
5. **DOS computation:** For an item-location with `avg_tot_oh = 1000` and `avg_daily_sls = 20`, verify DOS ≈ 50.
6. **API response:** Hit `/dfu/analysis?mode=item_location&item=100320&location=6601-KAPO` and verify `inventory_series` array is present and non-empty (when inventory data exists).
7. **Frontend chart:** Verify inventory chart renders with toggleable measures and syncs with the time range selector.
8. **DFU attributes:** Verify latest inventory position appears in the DFU attributes panel with correct item+location filtering.
9. **Missing inventory:** For item-locations without inventory data, verify the chart section gracefully shows "No inventory data available" instead of errors.
10. **Partial months:** Verify DOS/WOC calculations handle months with fewer than 28 snapshot days correctly (e.g., first month of data).
11. **Excluded columns:** Verify `site_allocation`, `national_allocation`, `dc_oh`, `dc_oh_oo`, `ssc_oh`, `ssc_oh_oo`, and `national_oos` are not present in `fact_inventory_snapshot` or the clean CSV.
12. **Daily sales derivation (day 1):** Load snapshot for item+loc on July 1 with `mtd_sls = 50`. Verify `daily_sls = 50` (first day of month = MTD value itself).
13. **Daily sales derivation (subsequent days):** Load item+loc July 1 (`mtd_sls = 50`) and July 2 (`mtd_sls = 120`). Verify July 2 `daily_sls = 70` (delta from prior day).
14. **Daily sales derivation (month boundary):** Load item+loc June 30 (`mtd_sls = 1500`) and July 1 (`mtd_sls = 40`). Verify July 1 `daily_sls = 40` (never subtracts across months).
15. **Monthly sales in view:** Verify `monthly_sls = MAX(mtd_sls)` for the item-location-month matches the last day's cumulative value.
16. **Location filtering:** Verify `loc` column is present and non-null in all loaded rows. Verify inventory query respects all three DFU analysis modes (`item_location`, `all_items_at_location`, `item_at_all_locations`).
17. **Cross-location aggregation:** In `item_at_all_locations` mode, verify inventory is SUM'd across all locations for the given item.

---

## Implementation Status

**Status:** Partially Implemented

### Phase 1 — Data Pipeline & Basic UI (Feature 34, completed earlier)
- `fact_inventory_snapshot` table with 190M+ rows loaded from 14 monthly CSVs
- Dedicated normalize script (`scripts/normalize_inventory_csv.py`)
- Basic API endpoints: `/inventory/position`, `/inventory/kpis`, `/inventory/trend`, `/inventory/item-detail`
- Basic InventoryTab with position table, trend chart, item detail drill-down

### Phase 2 — Supply Chain KPIs & View Rebuild (completed)
- Rebuilt `agg_inventory_monthly` materialized view with:
  - `daily_sls` CTE via `LAG()` partitioned by item_no, loc, month (daily sales from cumulative mtd_sales)
  - `eom_qty_on_hand` / `eom_qty_on_hand_on_order` via `ARRAY_AGG(ORDER BY snapshot_date DESC)[1]`
  - `monthly_sales = MAX(mtd_sales)` (fixes SUM bug on cumulative data)
  - `avg_daily_sls = AVG(NULLIF(daily_sls, 0))` (demand rate for DOS/WOC)
  - `latest_lead_time_days` via `ARRAY_AGG` (last known, not average)
  - `snapshot_days = COUNT(*)` for partial month handling
- Fixed `/inventory/kpis`: two-query pattern (latest snapshot PIT totals + trailing-month agg for DOS/WOC/Turns/LT Coverage)
- Fixed `/inventory/trend`: correct field names, added monthly_sales and DOS lines
- 7 severity-coded KPI cards (On-Hand, On-Order, Lead Time, DOS, WOC, Turns, LT Coverage)
- 5-line trend chart (On Hand, On Order, Monthly Sales, Lead Time, Days of Supply)

### Not Yet Implemented
- Inventory overlay in DFU Analysis tab (inventory chart below sales/forecast chart)
- Inventory KPI cards on DFU Analysis tab
- DFU attributes enrichment with latest inventory position
- Forward-looking DOS using champion forecast
- Demand-supply gap analysis
- Safety stock computation
- ABC-inventory cross analysis
- Inventory health score
- Bullwhip effect detection
- Seasonal pre-build tracking


---

## Examples

### Example: Planned inventory overlay API contract

```bash
# When implemented, inventory data will be served alongside forecast in DFU Analysis:
curl -s "http://localhost:8000/dfu/analysis?item=100320&loc=1401-BULK&include_inventory=true" \
  | jq '.inventory_overlay[] | {month: .startdate, dos, woc, qty_on_hand, stockout_risk}'
# {"month": "2025-11-01", "dos": 52.3, "woc": 7.5, "qty_on_hand": 5020, "stockout_risk": false}
# {"month": "2025-12-01", "dos": 45.2, "woc": 6.4, "qty_on_hand": 4320, "stockout_risk": false}
# {"month": "2026-01-01", "dos": 12.4, "woc": 1.8, "qty_on_hand":  980, "stockout_risk": true}
```

### Example: Planned dual-axis chart design

```
Forecast vs Sales chart (existing):
  Left Y-axis:  Volume (cases)  — sales line + model forecast lines
  Right Y-axis: Accuracy %      — WAPE trend per model

Inventory Overlay (planned addition):
  Second chart below (same X-axis, linked zoom):
  Left Y-axis:  Inventory (cases on hand, on order)
  Right Y-axis: Days of Supply  — dos threshold line at 30 days
  Markers:      Red dot when DOS < 14 (stockout risk)
```

### Example: Data source for overlay

```sql
-- agg_inventory_monthly already has DOS, WOC, avg_on_hand
SELECT item_no, loc, month_start, eom_on_hand, dos, woc
FROM agg_inventory_monthly
WHERE item_no='100320' AND loc='1401-BULK'
  AND month_start >= '2025-08-01'
ORDER BY month_start;
-- 2025-08-01 | 100320 | 1401-BULK | 4891 | 50.8 | 7.3
-- 2025-09-01 | 100320 | 1401-BULK | 4210 | 43.9 | 6.3
```


#### Example — Planned Make Pipeline Commands

```bash
# One-time setup: apply DDL for fact_inventory_snapshot and agg_inventory_monthly
make db-apply-sql   # includes 016_create_fact_inventory_snapshot.sql

# Full inventory pipeline (normalize → load → refresh view)
make inventory-pipeline

# Individual steps:
make normalize-inventory
# → reads datafiles/inventory_snapshot.csv
# → drops excluded columns (site_allocation, dc_oh, ssc_oh, national_oos, etc.)
# → parses exec_date from YYYYMMDD to YYYY-MM-DD
# → derives qty_on_order = qty_on_hand_on_order - qty_on_hand
# → writes data/inventory_snapshot_clean.csv

make load-inventory
# → stages clean CSV into Postgres staging table
# → deduplicates by (item_no, loc, snapshot_date) composite key
# → inserts ~190M rows into fact_inventory_snapshot
# → refreshes agg_inventory_monthly materialized view

# Verify load:
make check-db
# fact_inventory_snapshot  | 190,000,000 rows
# agg_inventory_monthly    |      18,432 rows (one per DFU-month)

# Query DOS for a specific DFU after loading:
psql -h localhost -p 5440 -U demand -d demand -c "
  SELECT month_start, eom_on_hand, dos, woc, avg_daily_sales
  FROM agg_inventory_monthly
  WHERE item_no='100320' AND loc='1401-BULK'
  ORDER BY month_start DESC LIMIT 6;
"
# month_start | eom_on_hand |  dos  | woc  | avg_daily_sales
# 2025-12-01  |    4320     | 45.2  | 6.4  |      95.6
# 2025-11-01  |    5020     | 52.3  | 7.5  |      95.9
# 2025-10-01  |    4891     | 50.8  | 7.3  |      96.3
# 2025-09-01  |    4210     | 43.9  | 6.3  |      95.9
# 2025-08-01  |    3980     | 41.5  | 5.9  |      95.9
# 2025-07-01  |    4100     | 42.7  | 6.1  |      96.0
```

#### Example — Planned Inventory KPI API Response

```bash
# When the DFU Analysis endpoint is extended with inventory overlay:
curl -s "http://localhost:8000/dfu/analysis?item=100320&loc=1401-BULK&mode=overlay&include_inventory=true"   | jq '{
      dfu_attributes: .dfu_attributes | {qty_on_hand, qty_on_order, lead_time_days, dos, woc},
      inventory_kpis: .inventory_kpis,
      inventory_series_count: (.inventory_series | length)
    }'
# {
#   "dfu_attributes": {
#     "qty_on_hand": 4320,
#     "qty_on_order": 1200,
#     "lead_time_days": 14,
#     "dos": 45.2,
#     "woc": 6.4
#   },
#   "inventory_kpis": {
#     "avg_dos_12m": 47.8,
#     "stockout_months": 1,
#     "excess_months": 2,
#     "inv_turns_annual": 8.1,
#     "lt_coverage_ratio": 3.2
#   },
#   "inventory_series_count": 12
# }
```
