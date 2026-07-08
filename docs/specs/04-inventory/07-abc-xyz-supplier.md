# ABC-XYZ Classification & Supplier Performance

> Two complementary segmentation systems: an ABC-XYZ 3x3 matrix combining volume ranking with demand variability to drive differentiated replenishment policies, and supplier delivery KPIs quantifying reliability for lead time assumptions.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/classify_abc_xyz.py`, `api/routers/inventory/inv_planning_abc_xyz.py`, `api/routers/inventory/inv_planning_supplier.py`, `sql/143_add_otif_drop_old_supplier_mv.sql` |

---

## Problem

Not all items deserve the same planning effort. Without a systematic segmentation, planners apply uniform policies across high-revenue stable items and low-volume erratic items alike. Additionally, supplier delivery reliability directly impacts safety stock sizing but is rarely quantified.

---

## Solution

Two complementary classification systems: (1) ABC-XYZ matrix (a 3x3 segmentation combining volume ranking with demand variability) that drives differentiated policies; (2) supplier performance analytics that quantify delivery reliability and feed into lead time variability assumptions.

---

## How It Works

### ABC Classification (Volume)

Items ranked by annual revenue contribution:

| Class | Cumulative Revenue | Typical % of SKUs |
|---|---|---|
| A | Top 80% | ~20% |
| B | Next 15% | ~30% |
| C | Remaining 5% | ~50% |

### XYZ Classification (Variability)

Based on demand CV (coefficient of variation):

| Class | CV Range | Demand Pattern |
|---|---|---|
| X | CV < 0.3 | Stable, predictable |
| Y | 0.3 <= CV < 0.6 | Some variability |
| Z | CV >= 0.6 | Erratic, hard to forecast |

### 9-Cell Policy Matrix

| | X (stable) | Y (variable) | Z (erratic) |
|---|---|---|---|
| **A** (high value) | Fixed Qty, tight SS | Min/Max, moderate SS | Min/Max, high SS |
| **B** (medium value) | Fixed Qty, moderate SS | Periodic Review | Periodic Review, buffer |
| **C** (low value) | Periodic Review | Periodic Review, longer interval | MTO or minimal stock |

The combined segment (e.g., `AX`, `BZ`) is written to `dim_sku` and drives auto-assignment of replenishment policies, service level targets, and review frequencies.

### Supplier Performance (IPfeature12)

`mv_supplier_po_performance` aggregates delivery KPIs from `fact_purchase_orders` (the legacy `mv_supplier_performance` proxy view, based on `dim_item_lead_time_profile`, was retired in `sql/143`):

| Metric | Formula | Purpose |
|---|---|---|
| OTD % (on-time delivery) | Closed PO lines delivered by promise date / delivery-evaluated lines | Reliability measure |
| OTIF % (on-time in-full) | Closed PO lines on-time AND received_qty >= ordered_qty / delivery-evaluated lines | Fulfillment measure |
| In-full % | Closed PO lines with received_qty >= ordered_qty / delivery-evaluated lines | Quantity accuracy |
| Avg lead time | Mean days from order to receipt (closed POs) | Planning parameter |
| LT variability (LT CV) | Std dev of observed lead times / avg lead time | SS input |
| Reliability score | `40% * otif_pct + 20% * otd_pct + 40% * (1 - lt_cv)`, clamped to 0-100 | Composite 0-100 score |
| Distinct items | Distinct items supplied by the supplier | Coverage breadth |

---

## Data Model

ABC-XYZ -- columns on `dim_sku`:

| Column | Type | Example |
|---|---|---|
| `abc_vol` | TEXT | `A` |
| `xyz_class` | TEXT | `X` |
| `abc_xyz_segment` | TEXT | `AX` |

DDL: `sql/005_create_dim_dfu.sql`

Supplier performance:

| View | Grain | Key Columns |
|---|---|---|
| `mv_supplier_po_performance` | supplier_id | otd_pct, otif_pct, in_full_pct, avg_lead_time_days, stddev_lead_time_days, reliability_score, distinct_items |

DDL: `sql/092_mv_supplier_po_performance.sql` (base view); OTIF columns added and legacy `mv_supplier_performance` dropped in `sql/143_add_otif_drop_old_supplier_mv.sql`

---

## API

ABC-XYZ:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/abc-xyz/matrix` | 9-cell matrix: DFU counts, avg service level, avg DOS min/max per cell |
| GET | `/inv-planning/abc-xyz/summary` | Portfolio-level summary: total/classified DFUs, X/Y/Z counts, avg demand CV, avg intermittency ratio |
| GET | `/inv-planning/abc-xyz/detail` | Paginated per-DFU classification; filters `abc_vol`, `xyz_class`, `segment`, `item`, `location`, `brand`, `category`, `market`; `limit`/`offset` |

Supplier performance:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/supplier-performance/summary` | Portfolio-level supplier reliability summary; filters `brand`, `category`, `market` |
| GET | `/inv-planning/supplier-performance/detail` | Paginated per-supplier metrics; filters `supplier`, `min_score`, `max_score`, `brand`, `category`, `market`; `sort_by`/`sort_dir`, `limit`/`offset` |
| GET | `/inv-planning/supplier-performance/items` | Items supplied by a given `supplier_no` (required) with lead-time profile data; `limit`/`offset` |

Routers: `inv_planning_abc_xyz.py`, `inv_planning_supplier.py`

---

## Pipeline

```
make abc-xyz-all           # abc-xyz-schema + abc-xyz-classify
make supplier-perf-all     # supplier-perf-schema + supplier-perf-refresh
```

| Step | Script | Output |
|---|---|---|
| Classify | `scripts/classify_abc_xyz.py` | `dim_sku` columns updated |
| Refresh | MV refresh | `mv_supplier_po_performance` |

---

## Configuration

ABC thresholds are embedded in the classification script (80/95 percentile cutoffs). XYZ thresholds align with the CV bands in `config/forecasting/sku_features_config.yaml` (`variability` section) - `config/variability_config.yaml` does not exist and should not be recreated. Supplier reliability formula weights are in the materialized view SQL.

---

## Dependencies

- **Upstream:** `fact_sales_monthly` (revenue for ABC), `dim_sku` (demand_cv for XYZ), `fact_purchase_orders` (supplier PO delivery and lead-time data)
- **Downstream:** Policy auto-assignment, safety stock service levels, investment optimization prioritization

---

## See Also

- [02-demand-variability](02-demand-variability.md) -- CV feeds XYZ classification
- [04-replenishment](04-replenishment.md) -- 9-cell matrix drives policy auto-assignment
- [03-safety-stock](03-safety-stock.md) -- Service levels differentiated by ABC class
