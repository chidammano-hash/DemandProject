# Inventory Rebalancing

> Detects cross-location inventory imbalances using DOS CV, builds transfer candidates between excess and shortage locations, solves for optimal transfers via greedy or LP solver, and manages execution through a 4-state approval workflow.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/compute_rebalancing.py`, `api/routers/inventory/inv_planning_rebalancing.py`, `config/rebalancing_config.yaml`, `sql/071_create_transfer_network.sql` |

---

## Problem

Inventory imbalances across the network -- one location has 120 days of supply while another has 5 -- result from uneven demand, forecast errors, and batch ordering. New procurement takes weeks, but lateral transfers between locations can resolve imbalances in days. Without a systematic rebalancing engine, planners identify transfer opportunities manually (if at all).

---

## Solution

An automated rebalancing engine that detects cross-location imbalances using DOS CV (coefficient of variation of Days of Supply across locations for the same item), builds transfer candidates between excess and shortage locations, solves for optimal transfers via greedy or LP (linear programming) solver, and presents plans through an approval workflow.

---

## How It Works

### Imbalance Detection

`mv_network_balance` computes per-item network balance metrics from `agg_inventory_monthly` + `fact_safety_stock_targets`:

| Metric | Formula | Purpose |
|---|---|---|
| DOS CV | std(DOS across locations) / mean(DOS) | Imbalance severity indicator |
| Excess location count | Locations with DOS > threshold | Transfer sources |
| Shortage location count | Locations with DOS < lead time | Transfer destinations |

Items with DOS CV above the configured threshold (default 0.5) and 2+ stocking locations are flagged as imbalanced.

### Transfer Network

`dim_transfer_lane` defines valid source-to-destination pairs:

| Column | Type | Purpose |
|---|---|---|
| `source_loc` | TEXT | Sending location |
| `dest_loc` | TEXT | Receiving location |
| `transit_days` | INTEGER | Transport time |
| `cost_per_unit` | NUMERIC | Transfer cost |
| `is_active` | BOOLEAN | Lane availability |

Only active lanes are considered for transfers.

### Solver Options

| Solver | Method | Best For |
|---|---|---|
| Greedy | Sort candidates by urgency, allocate top-down | Fast, good-enough solutions |
| LP | Minimize total transfer cost subject to constraints | Optimal when lane costs vary significantly |

Constraints: minimum transfer quantity, maximum % of source stock transferable, respect lane capacity.

### Urgency Assignment

| Urgency | Criteria |
|---|---|
| `critical` | Destination DOS < 0.5 * lead time |
| `high` | Destination DOS < lead time |
| `medium` | Destination DOS < 2 * lead time |
| `low` | Destination DOS below target but > 2 * lead time |

### Financial Model

Each transfer has a computed financial case:

| Metric | Calculation |
|---|---|
| Transfer cost | qty * cost_per_unit |
| Avoided stockout value | qty * unit_price * stockout probability |
| Net benefit | Avoided stockout - transfer cost |
| ROI | Net benefit / transfer cost |

Only transfers with positive net benefit are included in the plan.

### Approval Workflow

```
draft -> approved -> in_transit -> completed
```

Plans start as drafts. Planners review, approve specific transfers, and track execution status.

---

## Data Model

| Table / View | Grain | Purpose |
|---|---|---|
| `dim_transfer_lane` | source_loc + dest_loc | Network topology |
| `fact_rebalancing_plan` | plan_id | Plan header (status, solver, total cost) |
| `fact_rebalancing_transfer` | transfer_id | Individual transfers within a plan |
| `mv_network_balance` | item_id | Per-item network balance metrics |

DDL: `sql/071_create_transfer_network.sql`, `sql/072_create_rebalancing_plan.sql`, `sql/073_create_rebalancing_views.sql`

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/rebalancing/kpis` | Network balance KPIs |
| GET | `/inv-planning/rebalancing/network` | Transfer lane topology |
| GET | `/inv-planning/rebalancing/imbalances` | Imbalanced items list |
| POST | `/inv-planning/rebalancing/compute` | Trigger rebalancing computation |
| GET | `/inv-planning/rebalancing/plans` | List rebalancing plans |
| GET | `/inv-planning/rebalancing/plans/{id}` | Plan detail with transfers |
| PUT | `/inv-planning/rebalancing/plans/{id}/approve` | Approve a plan |
| GET | `/inv-planning/rebalancing/transfers` | All transfers across plans |
| PUT | `/inv-planning/rebalancing/transfers/{id}/status` | Update transfer status |

Plus 3 additional endpoints for transfer history, plan deletion, and bulk approval. Router: `inv_planning_rebalancing.py` (12 endpoints total).

---

## Pipeline

```
make rebalancing-all          # schema + compute
make rebalancing-compute-dry  # Preview without writing
make rebalancing-refresh      # Refresh mv_network_balance
```

| Step | Script | Output |
|---|---|---|
| Detect imbalances | `scripts/compute_rebalancing.py` | Imbalanced items identified |
| Build candidates | (same script) | Source-destination pairs ranked |
| Solve | (same script) | Optimal transfer plan |
| Write | (same script) | `fact_rebalancing_plan` + `fact_rebalancing_transfer` |

Supports `--dry-run` for previewing without writing.

---

## Configuration

File: `config/rebalancing_config.yaml`

```yaml
solver: greedy                # greedy or lp
dos_cv_threshold: 0.5         # minimum imbalance to act on
min_transfer_qty: 10
max_source_depletion_pct: 30  # never take > 30% of source stock
urgency_thresholds:
  critical: 0.5               # DOS / LT ratio
  high: 1.0
  medium: 2.0
```

---

## Dependencies

- **Upstream:** `agg_inventory_monthly` (DOS per location), `fact_safety_stock_targets` (target levels), `dim_transfer_lane` (network topology)
- **Downstream:** Control tower (rebalancing activity), financial planning (transfer costs)

---

## See Also

- [01-inventory-snapshot](01-inventory-snapshot.md) -- DOS data source
- [03-safety-stock](03-safety-stock.md) -- Targets define excess vs shortage
- [09-multi-echelon](09-multi-echelon.md) -- Network-aware inventory positioning
- [05-exception-queue](05-exception-queue.md) -- Excess inventory exceptions may trigger rebalancing
