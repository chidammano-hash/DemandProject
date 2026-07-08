# Safety Stock & Monte Carlo Simulation

> Analytical safety stock calculator using the combined variability formula with ABC-differentiated Z-scores, validated by a Monte Carlo simulator that runs 10,000 demand/lead-time scenarios to produce empirical service-level probability curves.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/compute_safety_stock.py`, `scripts/run_ss_simulation.py`, `config/inventory/safety_stock_config.yaml`, `sql/037_create_safety_stock_targets.sql` |

---

## Problem

Setting safety stock (buffer inventory to absorb demand and supply variability) by gut feel leads to either excess inventory (wasted capital) or frequent stockouts (lost sales). Planners need a mathematically grounded approach that accounts for both demand uncertainty and lead time uncertainty, with the ability to simulate alternative service levels.

---

## Solution

Two complementary engines: (1) an analytical safety stock calculator using the combined variability formula with service-level-driven Z-scores, differentiated by ABC classification; (2) a Monte Carlo simulator that runs thousands of demand/lead-time scenarios to produce empirical service-level probability curves, validating and stress-testing the analytical targets.

---

## How It Works

### Analytical Safety Stock (IPfeature3)

**Combined variability formula:**

```
SS = Z * sqrt(LT * sigma_D^2 + D_bar^2 * sigma_LT^2)
```

Where:
- Z = service level Z-score (from lookup table, differentiated by ABC class)
- LT = average lead time (days)
- sigma_D = demand standard deviation (daily)
- D_bar = average daily demand
- sigma_LT = lead time standard deviation (days)

**Reorder Point (ROP):** `ROP = D_bar * LT + SS`

| ABC Class | Target Service Level | Z-Score |
|---|---|---|
| A | 98% | 2.054 |
| B | 95% | 1.645 |
| C | 90% | 1.282 |

Guard rails prevent extreme values: minimum SS of 1 unit, maximum capped at configurable months of supply.

### Monte Carlo Simulation (IPfeature10)

1. Sample demand from empirical distribution (historical monthly sales, not assumed normal)
2. Sample lead time from observed LT distribution
3. Simulate `n_simulations` (default 10,000) inventory cycles
4. For each SS level tested, compute achieved service level (fill rate)
5. Output: SS quantity vs service-level probability curve per DFU

This validates whether the analytical SS actually achieves the target service level under real demand patterns (which may be skewed, intermittent, or bimodal).

---

## Data Model

| Table | Grain | Key Columns |
|---|---|---|
| `fact_safety_stock_targets` | item_id + loc | ss_combined, rop, target_service_level, z_score, demand_std, lt_std, abc_class, computed_at |
| `fact_ss_simulation_results` | simulation_id + item_id + loc | n_simulations, ss_levels_tested, achieved_service_levels, optimal_ss |

DDL: `sql/037_create_safety_stock_targets.sql`, `sql/030_create_ss_simulation_results.sql`

---

## API

Safety stock endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/safety-stock/summary` | By-class summary + top gaps |
| GET | `/inv-planning/safety-stock/detail` | Per-DFU SS targets, paginated |
| GET | `/inv-planning/safety-stock/waterfall` | SS waterfall decomposition (demand component vs. LT component) for a single item+location |
| GET | `/inv-planning/safety-stock/explain` | Formula-substituted SS decomposition + what-if sensitivity analysis for a single DFU |
| POST | `/inv-planning/safety-stock/override` | Manually override `ss_combined` for an item+location, recalculating gap/coverage (auth required) |
| POST | `/inv-planning/safety-stock/what-if` | Simulate SS under modified demand/lead-time/service-level inputs; read-only, no DB write (auth required) |
| GET | `/inv-planning/safety-stock/config` | Return `safety_stock_config.yaml` as JSON |

Simulation endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/inv-planning/simulation/run` | Trigger simulation for item set (201) |
| GET | `/inv-planning/simulation/results` | Simulation output with curves |
| GET | `/inv-planning/simulation/compare` | Compare analytical vs simulated SS |
| GET | `/inv-planning/simulation/{sim_run_id}/status` | Poll status of a running/completed simulation |

Router: `inv_planning_safety_stock.py`, `inv_planning_simulation.py`

---

## Pipeline

```
make ss-all     # ss-schema + ss-compute (analytical)
make sim-run    # Monte Carlo simulation
```

| Step | Script | Output |
|---|---|---|
| Compute SS | `scripts/compute_safety_stock.py` | `fact_safety_stock_targets` |
| Simulate | `scripts/run_ss_simulation.py` | `fact_ss_simulation_results` |

---

## Configuration

File: `config/inventory/safety_stock_config.yaml`

```yaml
service_levels:
  A: 0.98
  B: 0.95
  C: 0.90
z_table:
  0.90: 1.282
  0.95: 1.645
  0.98: 2.054
guard_rails:
  min_ss_units: 1
  max_ss_months_supply: 6
```

File: `config/inventory/inventory_planning_config.yaml` (simulation section)

```yaml
n_simulations: 10000
random_seed: 42
```

---

## Dependencies

- **Upstream:** `dim_sku` (demand_cv, variability_class), `dim_item_lead_time_profile` (lt_mean, lt_cv), `fact_sales_monthly`, ABC classification
- **Downstream:** Health scores (SS coverage component), exception queue (below-SS alerts), rebalancing (excess/shortage detection), investment optimization
- **Libraries:** pandas, numpy, scipy

---

## See Also

- [02-demand-variability](02-demand-variability.md) -- Provides sigma-D and sigma-LT inputs
- [04-replenishment](04-replenishment.md) -- Health score uses SS coverage as a component
- [05-exception-queue](05-exception-queue.md) -- Triggers exceptions when stock < SS
- [08-investment](08-investment.md) -- Budget allocation uses SS targets as the reference
