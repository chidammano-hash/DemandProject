# 04-07 ABC-XYZ Classification & Supplier Performance

> **Status:** Implemented | **Features:** IPfeature11, IPfeature12

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

`mv_supplier_performance` aggregates delivery KPIs from inventory receipt patterns:

| Metric | Formula | Purpose |
|---|---|---|
| On-time delivery % | Orders within +/- 2 days of promise / total | Reliability measure |
| Avg lead time | Mean days from order to receipt | Planning parameter |
| LT variability | Std dev of observed lead times | SS input |
| Reliability score | `0.5 * on_time_pct + 0.3 * (1 - lt_cv) + 0.2 * fill_rate` | Composite 0-100 score |
| SKU-location count | Distinct items supplied | Coverage breadth |

Supports trending over time to detect supplier deterioration.

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
| `mv_supplier_performance` | supplier + item grouping | on_time_pct, avg_lt, lt_cv, reliability_score, sku_loc_count |

DDL: `sql/032_create_supplier_performance.sql`

---

## API

ABC-XYZ:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/abc-xyz/matrix` | 9-cell distribution counts |
| GET | `/inv-planning/abc-xyz/detail` | Per-DFU classification |
| GET | `/inv-planning/abc-xyz/migration` | Period-over-period class changes |

Supplier performance:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/supplier/summary` | Top/bottom supplier ranking |
| GET | `/inv-planning/supplier/detail` | Per-supplier metrics |
| GET | `/inv-planning/supplier/trend` | Supplier performance over time |

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
| Refresh | MV refresh | `mv_supplier_performance` |

---

## Configuration

ABC thresholds are embedded in the classification script (80/95 percentile cutoffs). XYZ thresholds align with `config/variability_config.yaml` CV bands. Supplier reliability formula weights are in the materialized view SQL.

---

## Dependencies

- **Upstream:** `fact_sales_monthly` (revenue for ABC), `dim_sku` (demand_cv for XYZ), `fact_inventory_snapshot` (supplier receipts)
- **Downstream:** Policy auto-assignment, safety stock service levels, investment optimization prioritization

---

## See Also

- [02-demand-variability](02-demand-variability.md) -- CV feeds XYZ classification
- [04-replenishment](04-replenishment.md) -- 9-cell matrix drives policy auto-assignment
- [03-safety-stock](03-safety-stock.md) -- Service levels differentiated by ABC class
