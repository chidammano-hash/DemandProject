# Seasonality Detection

> Classifies demand patterns across calendar cycles into named seasonality profiles per DFU, feeding forecasting models, accuracy views, and inventory planning as filter dimensions.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | DFU Analysis (filter dimension) |
| **Key Files** | `scripts/detect_seasonality.py`, `scripts/update_seasonality_profiles.py`, `config/seasonality_config.yaml`, `sql/005_create_dim_dfu.sql` |

---

## Why This Moved Here

Seasonality detection classifies *how* demand behaves over calendar cycles -- it is descriptive pattern recognition, not a forecast. The resulting profiles feed into forecasting models, accuracy views, and inventory planning as filter dimensions.

---

## Problem

Planners cannot visually inspect thousands of DFUs for seasonal patterns. Without automated detection, seasonal items get the same safety stock and replenishment logic as steady-demand items, leading to over-stock in troughs and stock-outs at peaks.

---

## Solution

A three-metric detection pipeline computes seasonality strength, yearly periodicity, and peak/trough timing per DFU, then classifies each into a named profile. Six columns are written to `dim_sku` and joined into accuracy materialized views for cross-dimensional slicing.

---

## How It Works

### Detection Metrics

| Metric | Formula | Purpose |
|---|---|---|
| Seasonality strength | CV (coefficient of variation) of monthly means across years | Higher CV = stronger seasonal signal |
| YoY correlation | Pearson correlation between consecutive years' monthly volumes | Positive = repeatable yearly pattern |
| ACF lag 12 | Autocorrelation at 12-month lag | Confirms yearly periodicity |
| Peak month | Month with highest average volume | Timing for pre-build |
| Trough month | Month with lowest average volume | Timing for inventory draw-down |
| Peak-to-trough ratio | Peak avg / trough avg | Amplitude of seasonal swing |

### Profile Classification

| Profile | Criteria |
|---|---|
| `strong_seasonal` | strength >= 0.5 AND yoy_corr >= 0.6 |
| `moderate_seasonal` | strength >= 0.3 AND yoy_corr >= 0.3 |
| `weak_seasonal` | strength >= 0.15 OR yoy_corr >= 0.15 |
| `non_seasonal` | All metrics below thresholds |

`is_yearly_seasonal` flag is set when `yoy_corr >= 0.5 AND acf_lag12 >= 0.3`.

### Accuracy View Integration (Feature 32)

The accuracy materialized views (`agg_accuracy_by_dim`) join `dim_sku` seasonality columns, enabling planners to slice forecast accuracy by seasonality profile (e.g., "How accurate are our forecasts for strong-seasonal items?").

---

## Data Model

Six columns added to `dim_sku`:

| Column | Type | Example |
|---|---|---|
| `seasonality_profile` | TEXT | `strong_seasonal` |
| `seasonality_strength` | NUMERIC | 0.72 |
| `is_yearly_seasonal` | BOOLEAN | true |
| `peak_month` | INTEGER | 7 (July) |
| `trough_month` | INTEGER | 2 (February) |
| `peak_trough_ratio` | NUMERIC | 3.4 |

DDL: `sql/005_create_dim_dfu.sql (seasonality columns), sql/011_create_accuracy_slice_views.sql (seasonality in accuracy MVs)

---

## API

Seasonality columns are exposed through the generic Data Explorer endpoints (`GET /domains/sku/rows`) and as filter dimensions in accuracy endpoints. No dedicated seasonality API router.

---

## Pipeline

```
make seasonality-all    # schema + detect + update
```

| Step | Script | Output |
|---|---|---|
| Apply DDL | `sql/005_create_dim_dfu.sql` | 6 columns on `dim_sku` |
| Detect patterns | `scripts/detect_seasonality.py` | CSV with per-DFU metrics |
| Write profiles | `scripts/update_seasonality_profiles.py` | `dim_sku` updated |

---

## Configuration

File: `config/seasonality_config.yaml`

```yaml
history_months: 36
min_months_required: 24
strength_thresholds:
  strong: 0.5
  moderate: 0.3
  weak: 0.15
yoy_corr_thresholds:
  strong: 0.6
  moderate: 0.3
  weak: 0.15
yearly_seasonal_criteria:
  yoy_corr_min: 0.5
  acf_lag12_min: 0.3
```

---

## Dependencies

- **Upstream:** `fact_sales_monthly`, `dim_sku`
- **Downstream:** Accuracy views (filter dimension), DFU Analysis tab (profile filter), safety stock (seasonal pre-build)
- **Libraries:** pandas, scipy (ACF), numpy

---

## See Also

- [01-sku-clustering](01-sku-clustering.md) -- Clustering uses `cv_monthly` and `yoy_corr` as features
- [../02-forecasting/02-01-accuracy-kpis](../02-forecasting/02-01-accuracy-kpis.md) -- Accuracy slicing by seasonality profile
- [../04-inventory/03-safety-stock](../04-inventory/03-safety-stock.md) -- Seasonal items may need different service levels
