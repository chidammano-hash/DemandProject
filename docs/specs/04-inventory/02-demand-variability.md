# Demand & Lead Time Variability

> Two profiling pipelines that quantify demand variability (CV, MAD, skewness) and lead time variability (LT CV, reliability) per DFU, classifying each into a volatility or reliability profile for use in safety stock formulas.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/ml/compute_sku_features.py` (unified pipeline, `make features-compute`), `scripts/compute_lead_time_variability.py`, `config/forecasting/sku_features_config.yaml`, `sql/022_add_demand_variability_columns.sql`. (The legacy `scripts/compute_demand_variability.py` was deleted; use `make features-compute`.) |

---

## Problem

Safety stock formulas require quantified demand variability (sigma-D) and lead time variability (sigma-LT) as inputs. Without profiling these, planners either use rules of thumb or a single safety factor across the entire portfolio, resulting in misallocated buffer stock.

---

## Solution

Two complementary profiling pipelines: (1) demand variability computes CV (coefficient of variation), MAD (mean absolute deviation), dispersion, and skewness per DFU from monthly sales history, then classifies each into a volatility profile; (2) lead time variability computes LT CV, mean, and reliability metrics per item-location from daily inventory snapshots, detecting change-points in supplier behavior.

---

## How It Works

### Demand Variability (IPfeature1)

| Metric | Formula | Purpose |
|---|---|---|
| CV | std_dev / mean | Scale-independent volatility |
| MAD | mean(abs(x - median)) | Robust central dispersion |
| Skewness | scipy skew | Tail asymmetry |
| IQR | Q3 - Q1 | Spread of middle 50% |
| Zero-month % | months with qty=0 / total | Intermittency indicator |

Classification into `variability_class`:

| Class | Criteria |
|---|---|
| `low` | CV < 0.3 |
| `moderate` | 0.3 <= CV < 0.6 |
| `high` | 0.6 <= CV < 1.0 |
| `erratic` | CV >= 1.0 |

13 columns written to `dim_sku` including: `demand_cv`, `demand_mad`, `demand_skewness`, `demand_iqr`, `zero_month_pct`, `variability_class`, and additional stats.

### Lead Time Variability (IPfeature2)

Computed from daily `fact_inventory_snapshot` by detecting order-receipt patterns:

| Metric | Formula | Purpose |
|---|---|---|
| LT mean | Average observed lead time (days) | Central estimate |
| LT CV | std_dev / mean of observed LTs | Variability for SS formula |
| LT reliability | % of orders within +/- 2 days of mean | Supplier consistency |
| Change-point flag | Significant shift in recent vs historical LT | Alerts to supplier behavior change |

Written to `dim_item_lead_time_profile` table.

| Reliability Band | Criteria |
|---|---|
| `reliable` | reliability >= 0.85 |
| `moderate` | 0.65 <= reliability < 0.85 |
| `unreliable` | reliability < 0.65 |

---

## Data Model

Demand variability -- 13 columns on `dim_sku`:

| Column | Type | Example |
|---|---|---|
| `demand_cv` | NUMERIC | 0.45 |
| `demand_mad` | NUMERIC | 120.5 |
| `variability_class` | TEXT | `moderate` |

DDL: `sql/022_add_demand_variability_columns.sql`

Lead time variability:

| Table | Grain | Key Columns |
|---|---|---|
| `dim_item_lead_time_profile` | item_id + loc | lt_mean, lt_cv, lt_reliability, reliability_band |

DDL: `sql/023_create_lead_time_profile.sql`

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/variability/summary` | Portfolio variability distribution |
| GET | `/inv-planning/variability/detail` | Per-DFU variability metrics |
| GET | `/inv-planning/lead-time/summary` | LT reliability distribution |
| GET | `/inv-planning/lead-time/profile` | Per-item LT profile |

Routers: `inv_planning_variability.py`, `inv_planning_lead_time.py`

---

## Pipeline

| Step | Script | Output |
|---|---|---|
| SKU features (incl. demand variability) | `scripts/ml/compute_sku_features.py` (`make features-compute`) | `dim_sku` seasonality + variability columns updated |
| Lead time variability | `scripts/compute_lead_time_variability.py` | `dim_item_lead_time_profile` rows |

> **Note:** The legacy `scripts/compute_demand_variability.py` has been deleted. Use `make features-compute` for all new work.

---

## Configuration

`config/variability_config.yaml` does not exist - it predates the unified pipeline and was never recreated (see CLAUDE.md "Do not recreate deleted configs"). Variability thresholds now live in `config/forecasting/sku_features_config.yaml`, `variability` section:

```yaml
variability:
  cv_thresholds:
    smooth: 0.3
    erratic: 0.8
  intermittency_threshold: 0.15
```

Note: `compute_sku_features.py` classifies into `smooth`/`erratic`/`intermittent`/`lumpy` (see [../03-demand-intelligence/02-sku-feature-engineering](../03-demand-intelligence/02-sku-feature-engineering.md)), not the `low`/`moderate`/`high`/`erratic` taxonomy shown above - that table describes the legacy scheme and has not been reconciled with the current classifier.

File: `config/inventory/inventory_planning_config.yaml` (lead_time section)

```yaml
lt_cv_thresholds:
  low: 0.15
  moderate: 0.30
reliability_bands:
  reliable: 0.85
  moderate: 0.65
```

---

## Dependencies

- **Upstream:** `fact_sales_monthly` (demand), `fact_inventory_snapshot` (lead time)
- **Downstream:** Safety stock formula (sigma-D, sigma-LT inputs), ABC-XYZ classification (CV feeds XYZ)
- **Libraries:** pandas, scipy, numpy

---

## See Also

- [03-safety-stock](03-safety-stock.md) -- Primary consumer of demand CV and LT CV
- [07-abc-xyz-supplier](07-abc-xyz-supplier.md) -- XYZ classification derived from demand CV
- [01-inventory-snapshot](01-inventory-snapshot.md) -- Source data for lead time detection
