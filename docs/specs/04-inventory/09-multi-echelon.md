# 04-09 Multi-Echelon Safety Stock

> **Status:** Implemented | **Feature:** F3.5

## Problem

Single-echelon safety stock treats each location independently. In a multi-tier network (distribution center feeds regional warehouses), this ignores risk pooling (the statistical benefit of aggregating demand variability at higher levels). The result is excess total inventory because each node carries its own buffer without coordinating with upstream/downstream partners.

---

## Solution

A 2-echelon optimization that computes safety stock across DC (distribution center) and downstream locations jointly, exploiting risk pooling to reduce total network inventory while maintaining end-customer service levels. Cascade risk badges surface locations where upstream shortages propagate downstream.

---

## How It Works

### Network Structure

| Echelon | Role | SS Logic |
|---|---|---|
| DC (upstream) | Supplies regional locations | Carries pooled SS for aggregate demand |
| Regional (downstream) | Serves end customers | Carries location-specific SS minus risk-pooling benefit |

### Risk Pooling Principle

Aggregate demand variability at the DC level is less than the sum of individual location variabilities (due to demand correlation < 1.0). The DC holds SS sized to aggregate sigma, and each downstream location reduces its SS by its share of the pooling benefit.

**Pooled sigma at DC:**
```
sigma_DC = sqrt(sum(sigma_i^2) + 2 * sum(rho_ij * sigma_i * sigma_j))
```

Where `rho_ij` is the demand correlation between locations i and j.

**Downstream SS reduction:**
```
SS_i_echelon = SS_i_single - pooling_benefit_i
pooling_benefit_i = (sigma_i / sigma_DC) * (SS_DC_single - SS_DC_echelon)
```

### Cascade Risk Scoring

Each downstream location gets a cascade risk severity based on upstream coverage:

| Severity | Condition | Badge |
|---|---|---|
| `low` | DC DOS > 2x avg downstream LT | Green |
| `medium` | DC DOS between 1-2x avg downstream LT | Yellow |
| `high` | DC DOS < 1x avg downstream LT | Orange |
| `critical` | DC projected stockout within LT | Red |

### Worked Example

DC-EAST supplies 5 regional warehouses. Single-echelon total SS = 15,000 units. After multi-echelon optimization with demand correlation = 0.4 across locations, total network SS = 11,200 units (25% reduction). DC holds 4,500 pooled units; each downstream location reduces by ~760 units.

---

## Data Model

| Table | Grain | Key Columns |
|---|---|---|
| `dim_supply_network` | source_loc + dest_loc | echelon_level, transit_days, is_active |
| `fact_echelon_targets` | item_no + loc + echelon | ss_single, ss_echelon, pooling_benefit, cascade_risk |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/echelon/summary` | Network-wide SS comparison (single vs multi) |
| GET | `/inv-planning/echelon/detail` | Per-location echelon targets with cascade risk |
| GET | `/inv-planning/echelon/network` | Network topology with flow volumes |

Router: `echelon_planning.py`

---

## Pipeline

```
make echelon-compute    # Compute multi-echelon SS targets
```

| Step | Script | Output |
|---|---|---|
| Compute | `scripts/compute_echelon_targets.py` | `fact_echelon_targets` rows |

---

## Configuration

File: `config/echelon_config.yaml`

```yaml
echelon_levels: 2
correlation_method: pearson    # demand correlation estimation
min_pooling_benefit_pct: 5.0   # don't pool if benefit < 5%
cascade_risk_thresholds:
  low: 2.0        # DC DOS / avg downstream LT
  medium: 1.0
  high: 0.5
```

---

## Dependencies

- **Upstream:** `fact_safety_stock_targets` (single-echelon baseline), `dim_supply_network` (topology), `fact_sales_monthly` (demand correlations)
- **Downstream:** Rebalancing (network-aware transfers), investment optimization (echelon-adjusted SS)

---

## See Also

- [03-safety-stock](03-safety-stock.md) -- Single-echelon baseline that multi-echelon improves upon
- [11-rebalancing](11-rebalancing.md) -- Transfers consider echelon position
- [01-inventory-snapshot](01-inventory-snapshot.md) -- Source position data per location
