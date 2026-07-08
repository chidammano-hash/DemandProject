# Multi-Echelon Safety Stock

> 2-echelon optimization that computes safety stock across DC and downstream locations jointly using risk pooling, reducing total network inventory while maintaining end-customer service levels, with a cascade risk flag for upstream shortage propagation.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/inventory/compute_echelon_targets.py`, `api/routers/operations/echelon_planning.py`, `config/inventory/echelon_config.yaml` |

---

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

### Cascade Risk Flag

Each reorder-point row carries `cascade_risk_flag` (boolean) on `fact_echelon_reorder_points`, surfaced by `GET /supply/echelon/reorder-points` and used to sort at-risk locations to the top of the response. It is not a multi-tier severity - the API exposes a single flag, not graded low/medium/high/critical badges.

### Worked Example

DC-EAST supplies 5 regional warehouses. Single-echelon total SS = 15,000 units. After multi-echelon optimization with demand correlation = 0.4 across locations, total network SS = 11,200 units (25% reduction). DC holds 4,500 pooled units; each downstream location reduces by ~760 units.

---

## Data Model

Three tables (`sql/054_create_echelon_planning.sql`):

| Table | Grain | Key Columns |
|---|---|---|
| `dim_echelon_network` | parent_loc + child_loc (unique) | `echelon_level` (1=DC, 2=regional, 3=store), `link_type`, `replenishment_lead_time_days`, `is_active` |
| `fact_echelon_ss_targets` | item_id + loc + echelon_level (unique) | `echelon_ss_qty`, `standalone_ss_qty`, `pooling_benefit_pct`, `service_level_target`, `computed_at` |
| `fact_echelon_reorder_points` | item_id + loc + echelon_level (unique) | `reorder_point_qty`, `echelon_ss_qty`, `demand_during_lt_qty`, `cascade_risk_flag`, `computed_at` |

---

## API

| Method | Path | Params | Purpose |
|---|---|---|---|
| GET | `/supply/echelon/network` | none | Location hierarchy (parent to child links) from `dim_echelon_network` where `is_active = TRUE` |
| GET | `/supply/echelon/targets` | `item_id`, `loc`, `echelon_level`, `page`, `page_size` (default 50, capped 200) | Paginated network-optimized SS targets per echelon from `fact_echelon_ss_targets` |
| GET | `/supply/echelon/summary` | none | Portfolio pooling-benefit and coverage KPIs (total SKU-locs, avg pooling benefit %, total units saved, echelon depth, last computed) |
| GET | `/supply/echelon/reorder-points` | `item_id`, `loc`, `page`, `page_size` (default 50, capped 200) | Paginated echelon ROPs with cascade risk flag from `fact_echelon_reorder_points`, sorted cascade-risk-first |

Router: `api/routers/operations/echelon_planning.py`

---

## Pipeline

```
make echelon-compute    # Compute multi-echelon SS targets
```

| Step | Script | Output |
|---|---|---|
| Compute | `scripts/inventory/compute_echelon_targets.py` | `fact_echelon_ss_targets` rows |

---

## Configuration

File: `config/inventory/echelon_config.yaml`, `echelon` section:

```yaml
echelon:
  z_score_default: 1.645              # Z-score for echelon SS (1.645 = 95%, 2.326 = 99%)
  default_service_level: 0.95         # Target fill rate for echelon nodes
  min_downstream_nodes: 2             # Minimum downstream nodes to apply pooling
  cascade_risk_multiplier: 1.0        # Scales risk propagation from downstream to upstream
  coverage_alert_threshold_days: 7    # DC coverage below this triggers a control-tower alert
  default_lt_days: 10                 # DC-to-store lead time fallback
  default_lt_std_days: 2.0            # Lead time std-dev fallback
```

A `scheduler` section registers the weekly recompute job (`compute_echelon_targets`, Monday 04:00 cron) with APScheduler.

---

## Dependencies

- **Upstream:** `dim_echelon_network` (topology), `dim_sku` (demand mean/std for pooling), `fact_inventory_snapshot` (DC on-hand)
- **Downstream:** Rebalancing (network-aware transfers), investment optimization (echelon-adjusted SS)

---

## See Also

- [03-safety-stock](03-safety-stock.md) -- Single-echelon baseline that multi-echelon improves upon
- [11-rebalancing](11-rebalancing.md) -- Transfers consider echelon position
- [01-inventory-snapshot](01-inventory-snapshot.md) -- Source position data per location
