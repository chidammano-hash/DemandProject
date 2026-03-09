# Feature F3.5 — Network / Multi-Echelon Planning

**Phase:** 3 — Operational Feedback Loop
**Feature Number:** F3.5 (file: feature_06_12)
**Status:** Design / Not Started
**Depends On:** IPfeature3 (Safety Stock), IPfeature4 (EOQ), IPfeature7 (Exception Queue), feature_06_09 (Service Level), feature_06_10 (Lead Time Learning)

---

## 1. Problem Statement

The current system plans each item-location (DFU) independently. Safety stock, EOQ, reorder points, and exceptions are all computed for a single item_no + loc pair without regard for where that location sits in the replenishment network.

In reality, a distribution center (DC) holds inventory to replenish multiple downstream stores. When the DC runs low, ALL downstream stores face a supply disruption simultaneously — a cascade failure. Planning the DC as if it were a single store with single-store demand ignores:

1. **Demand pooling benefits:** A DC serving N stores faces less volatility than N individual stores, because demand fluctuations partially cancel across locations. Over-estimating DC safety stock is expensive.
2. **Cascade risk:** A DC stockout is categorically worse than a store stockout — it can deny supply to 5, 10, or 50 stores at once.
3. **Echelon safety stock:** The correct formula for DC safety stock requires knowledge of downstream σ values and the DC's replenishment lead time, not just the DC's historical sales.

### What Fails Today — Concrete Example

**Item 100320, DC-EAST serving 3 downstream stores:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TODAY'S APPROACH (independent planning)                                 │
│                                                                          │
│  DC-EAST:   planned as if it is its own store                            │
│             SS_DC  = Z × σ_DC_sales × sqrt(LT_DC)                       │
│             Actual DC demand = sum of store orders (not direct sales)    │
│             This overstates variability by ignoring pooling effects      │
│                                                                          │
│  STORE-A:   planned independently, SS=45 units                          │
│  STORE-B:   planned independently, SS=36 units                          │
│  STORE-C:   planned independently, SS=54 units                          │
│                                                                          │
│  PROBLEM: DC holds 500 units. 3 stores are each placing orders.         │
│  When DC drops to 200 units (threshold not known to planner), all 3    │
│  store replenishments fail simultaneously. A single DC stockout          │
│  creates 3 simultaneous store stockouts.                                 │
│                                                                          │
│  CORRECT APPROACH (multi-echelon):                                       │
│  DC daily demand = sum of store orders = 45 units/day                   │
│  DC σ_demand     = sqrt(σ_A² + σ_B² + σ_C²) = 26.3 units/day (pooled) │
│  DC SS = Z × σ_DC × sqrt(DC_supplier_LT) = 137 units                   │
│  DC ROP = 45 × 10 (LT) + 137 = 587 units                               │
│  ALERT: DC-EAST at 500 units is BELOW ROP of 587 units                 │
└─────────────────────────────────────────────────────────────────────────┘
```

The alert cannot be generated today because: (A) the DC→Store relationship is not modeled, (B) pooled σ is not computed, (C) the cascade impact (3 stores affected by 1 DC stockout) is not quantified.

---

## 2. Multi-Echelon Inventory Theory

### Risk Pooling Principle

When N independent stores each have demand variance σ², a DC serving all N stores faces aggregate demand variance:

```
σ²_DC = σ²_1 + σ²_2 + ... + σ²_N   (independent stores, no correlation)

If all stores have identical σ:
  σ_DC = σ × sqrt(N)

  N=3 stores, σ_each=15:
  σ_DC = 15 × sqrt(3) = 26.0 units/day   (NOT 45 = 3×15)

  Safety stock savings = (3 × 15 - 26.0) / (3 × 15) = 42% less SS needed at DC
  vs naive sum of individual store SS budgets
```

### Two-Echelon Network Structure

```
         ┌─────────────────────────────────────────────────────┐
         │  SUPPLIER (Echelon 0 — external)                    │
         │  LT to DC: 10 days                                  │
         └──────────────────┬──────────────────────────────────┘
                            │ replenishment orders + lead time
                            ▼
         ┌──────────────────────────────────────────────────────┐
         │  DC-EAST (Echelon 2 — distribution center)          │
         │  Current on-hand: 500 units                         │
         │  Echelon SS (pooled): 137 units                     │
         │  Echelon ROP: 587 units  ← BELOW ROP TODAY          │
         │  Downstream coverage: 11.1 days (500/45 per day)    │
         └──────┬────────────────┬────────────────┬────────────┘
                │                │                │
    LT=1 day    │    LT=1 day    │    LT=2 days   │
                ▼                ▼                ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │  STORE-A        │  │  STORE-B        │  │  STORE-C        │
  │  Echelon 1      │  │  Echelon 1      │  │  Echelon 1      │
  │  σ=15 units/day │  │  σ=12 units/day │  │  σ=18 units/day │
  │  SS=45 units    │  │  SS=36 units    │  │  SS=54 units    │
  │  ROP=85 units   │  │  ROP=68 units   │  │  ROP=122 units  │
  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### N-Echelon Extension (Factory → Regional DC → Local DC → Store)

```
Factory (Echelon 3) → Regional DC (Echelon 2) → Local DC (Echelon 1) → Store (Echelon 0)

Computation order: bottom-up (stores → local DCs → regional DCs → factory)
Cascade risk propagation: top-down (factory stockout → regional → local → stores)
```

---

## 3. Supply Network Configuration Requirements

### Currently Missing Data

| Data Element | Source | Gap |
|---|---|---|
| DC → Store mapping | ERP location hierarchy or manual config | Not modeled |
| Transfer lead time (DC→Store) | WMS transfer orders | Not tracked |
| Transfer frequency (daily/weekly/on-demand) | Replenishment policy | Not modeled |
| Transfer method (push/pull) | Planning policy | Not modeled |
| Factory → DC mapping | ERP vendor/warehouse hierarchy | Not modeled |
| Transfer orders (DC→Store) | WMS inter-warehouse transfer | Not tracked |
| Min transfer quantities | Replenishment policy | Not configured |

**For the initial implementation:** Supply network topology is entered manually via the UI (from/to location table with lead times). ERP integration is a future enhancement (see Out of Scope).

---

## 4. Data Model

### 4.1 `dim_supply_network` — Location-to-Location Replenishment Relationships

**Grain:** from_loc + to_loc + item_category (nullable = applies to all items)

```sql
CREATE TABLE dim_supply_network (
    network_id              SERIAL           PRIMARY KEY,
    from_loc                VARCHAR(50)      NOT NULL,  -- upstream location (DC, Factory)
    to_loc                  VARCHAR(50)      NOT NULL,  -- downstream location (Store, DC)
    relationship_type       VARCHAR(30)      NOT NULL,  -- dc_to_store / factory_to_dc /
                                                        -- regional_dc_to_local_dc / cross_dock
    item_category           VARCHAR(50),               -- NULL = all items
    replenishment_lt_days   INTEGER          NOT NULL,  -- lead time from from_loc to to_loc
    min_transfer_qty        NUMERIC(12,2)    NOT NULL DEFAULT 1,
    max_transfer_qty        NUMERIC(12,2),             -- NULL = no maximum
    transfer_frequency      VARCHAR(20)      NOT NULL DEFAULT 'daily',
                                                        -- daily / weekly / biweekly / on_demand
    transfer_method         VARCHAR(10)      NOT NULL DEFAULT 'pull',
                                                        -- push / pull / hybrid
    priority_rank           INTEGER          NOT NULL DEFAULT 1,
                                                        -- if from_loc has multiple upstreams
    effective_from          DATE             NOT NULL DEFAULT CURRENT_DATE,
    effective_to            DATE,                       -- NULL = currently active
    active                  BOOLEAN          NOT NULL DEFAULT TRUE,
    notes                   TEXT,
    created_by              VARCHAR(100),
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_supply_network_arc
    ON dim_supply_network(from_loc, to_loc, COALESCE(item_category, ''));

CREATE INDEX idx_supply_network_to_loc
    ON dim_supply_network(to_loc)
    WHERE active = TRUE;

CREATE INDEX idx_supply_network_from_loc
    ON dim_supply_network(from_loc)
    WHERE active = TRUE;
```

### 4.2 `fact_echelon_targets` — Computed Multi-Echelon SS and ROP per Location per Item

**Grain:** item_no + loc + computation_date (one row per DFU per location per computation run)

```sql
CREATE TABLE fact_echelon_targets (
    echelon_id              BIGSERIAL        PRIMARY KEY,
    item_no                 VARCHAR(50)      NOT NULL,
    loc                     VARCHAR(50)      NOT NULL,   -- this location (can be DC or store)
    echelon_level           INTEGER          NOT NULL,   -- 1=store, 2=DC, 3=regional_DC, 4=factory
    parent_loc              VARCHAR(50),                 -- NULL for root node (factory or top-DC)
    computation_date        DATE             NOT NULL,

    -- Demand aggregation at this echelon
    downstream_dfu_count    INTEGER          NOT NULL DEFAULT 0,
    downstream_loc_count    INTEGER          NOT NULL DEFAULT 0,
    mean_demand_daily       NUMERIC(12,4)    NOT NULL,  -- aggregated from downstream
    pooled_sigma_demand     NUMERIC(12,4)    NOT NULL,  -- sqrt(sum of downstream σ²)
    naive_sigma_demand      NUMERIC(12,4),              -- sum of downstream σ (without pooling)
    pooling_benefit_pct     NUMERIC(6,2),               -- 1 - (pooled/naive) as percentage

    -- Lead time for this echelon's supplier (upstream location or external supplier)
    upstream_lt_days        NUMERIC(8,2)     NOT NULL,
    upstream_sigma_lt_days  NUMERIC(8,2)     NOT NULL DEFAULT 0,

    -- Safety stock (echelon-aware formula)
    z_score                 NUMERIC(6,4)     NOT NULL,
    echelon_ss_qty          NUMERIC(12,2)    NOT NULL,
    echelon_rop_qty         NUMERIC(12,2)    NOT NULL,

    -- Coverage and risk metrics
    current_on_hand_qty     NUMERIC(12,2),
    dc_coverage_days        NUMERIC(8,2),               -- current_on_hand / mean_demand_daily
    cascade_risk_score      NUMERIC(5,2),               -- 0-100, 100 = highest risk
                                                        -- based on: coverage below ROP × downstream count
    is_below_rop            BOOLEAN          GENERATED ALWAYS AS
                                (current_on_hand_qty < echelon_rop_qty) STORED,
    downstream_locations_at_risk INTEGER,               -- # downstream locs affected if this DC stockouts

    -- Previous period for change tracking
    prior_echelon_ss_qty    NUMERIC(12,2),
    prior_cascade_risk_score NUMERIC(5,2),

    computed_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_echelon_targets_item_loc_date
    ON fact_echelon_targets(item_no, loc, computation_date);

CREATE INDEX idx_echelon_targets_item_loc
    ON fact_echelon_targets(item_no, loc);

CREATE INDEX idx_echelon_targets_cascade_risk
    ON fact_echelon_targets(cascade_risk_score DESC, computation_date DESC);

CREATE INDEX idx_echelon_targets_below_rop
    ON fact_echelon_targets(is_below_rop, computation_date DESC)
    WHERE is_below_rop = TRUE;

CREATE INDEX idx_echelon_targets_level
    ON fact_echelon_targets(echelon_level, computation_date DESC);
```

### 4.3 `fact_network_alerts` — Cascade Risk and Structural Supply Alerts

**Grain:** item_no + from_loc + to_loc + alert_date + alert_type (event-based, one row per alert event)

```sql
CREATE TABLE fact_network_alerts (
    alert_id                BIGSERIAL        PRIMARY KEY,
    item_no                 VARCHAR(50)      NOT NULL,
    from_loc                VARCHAR(50)      NOT NULL,
    to_loc                  VARCHAR(50),                 -- NULL if alert is at the from_loc level
    alert_date              DATE             NOT NULL,
    alert_type              VARCHAR(60)      NOT NULL,   -- cascade_stockout_risk /
                                                        -- pooling_benefit_lost /
                                                        -- transfer_frequency_mismatch /
                                                        -- dc_coverage_critical /
                                                        -- downstream_loc_added /
                                                        -- echelon_ss_increased
    severity                VARCHAR(20)      NOT NULL,  -- critical / high / medium / low
    affected_downstream_locations INTEGER   NOT NULL DEFAULT 0,
    dc_coverage_days        NUMERIC(8,2),
    current_on_hand_qty     NUMERIC(12,2),
    echelon_rop_qty         NUMERIC(12,2),
    shortfall_qty           NUMERIC(12,2),
    cascade_risk_score      NUMERIC(5,2),
    alert_message           TEXT             NOT NULL,
    acknowledged            BOOLEAN          NOT NULL DEFAULT FALSE,
    acknowledged_by         VARCHAR(100),
    acknowledged_at         TIMESTAMPTZ,
    resolved                BOOLEAN          NOT NULL DEFAULT FALSE,
    resolved_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_network_alerts_item_loc
    ON fact_network_alerts(item_no, from_loc, alert_date DESC);

CREATE INDEX idx_network_alerts_open
    ON fact_network_alerts(acknowledged, resolved, severity)
    WHERE acknowledged = FALSE AND resolved = FALSE;

CREATE INDEX idx_network_alerts_cascade
    ON fact_network_alerts(alert_type)
    WHERE alert_type = 'cascade_stockout_risk';
```

---

## 5. Python Script

### `scripts/compute_echelon_targets.py`

```python
#!/usr/bin/env python3
"""
compute_echelon_targets.py

Bottom-up multi-echelon inventory target computation.

Algorithm:
  1. Build the supply network DAG from dim_supply_network.
  2. Identify leaf nodes (stores at echelon level 1).
  3. Compute leaf safety stock (same as IPfeature3 single-echelon).
  4. For each parent node (DC, level 2+):
     a. Aggregate downstream demand (mean and pooled σ).
     b. Apply echelon SS formula with upstream lead time.
     c. Compute cascade risk score.
  5. Write results to fact_echelon_targets.
  6. Generate fact_network_alerts where DCs are below ROP.

Usage:
    uv run scripts/compute_echelon_targets.py
    uv run scripts/compute_echelon_targets.py --item-no 100320 --dry-run
"""

import argparse
import logging
import math
import yaml
from collections import defaultdict, deque
from datetime import date
from typing import Optional
import psycopg
from common.db import get_db_params

CONFIG_PATH = "config/network_planning_config.yaml"
log = logging.getLogger(__name__)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_echelon_ss(
    z_score: float,
    pooled_sigma_demand: float,
    upstream_lt_days: float,
    mean_demand_daily: float,
    upstream_sigma_lt_days: float,
) -> float:
    """
    Multi-echelon safety stock formula (full variance formula).

    SS_echelon = Z × sqrt(LT × σ_demand_pooled² + μ_demand² × σ_LT²)

    Args:
        z_score: Service level Z-score (e.g., 2.054 for 98%).
        pooled_sigma_demand: Pooled demand σ across downstream locations
                             = sqrt(sum(σ_i²)) under independence assumption.
        upstream_lt_days: Mean lead time from this node's upstream supplier.
        mean_demand_daily: Aggregated mean daily demand from all downstream locations.
        upstream_sigma_lt_days: Standard deviation of upstream lead time (from dim_lead_time_profile).

    Returns:
        Safety stock quantity (units).

    Example (DC-EAST, item 100320):
        z=1.65, σ_pooled=26.3, LT=10, μ_demand=45, σ_LT=0
        SS = 1.65 × sqrt(10 × 26.3² + 45² × 0²)
           = 1.65 × sqrt(6916.9)
           = 1.65 × 83.2 = 137 units
    """
    demand_variance_term = upstream_lt_days * (pooled_sigma_demand ** 2)
    lt_variance_term     = (mean_demand_daily ** 2) * (upstream_sigma_lt_days ** 2)
    ss = z_score * math.sqrt(demand_variance_term + lt_variance_term)
    return round(ss, 2)


def compute_cascade_risk_score(
    current_on_hand: float,
    echelon_rop: float,
    downstream_loc_count: int,
    echelon_level: int,
) -> float:
    """
    Cascade risk score: 0-100 scale.
    Higher = more urgent. Considers:
    - Proximity to ROP (coverage ratio)
    - Downstream location count (amplifies impact)
    - Echelon level (DC failures are worse than store failures)

    Formula:
        base_risk = max(0, (ROP - on_hand) / ROP) × 100  [0 if above ROP]
        amplifier = log(1 + downstream_loc_count) × echelon_level
        risk_score = min(100, base_risk × amplifier_factor)

    Args:
        current_on_hand: Current on-hand inventory at this location.
        echelon_rop: Reorder point for this echelon.
        downstream_loc_count: Number of downstream locations (stores) served.
        echelon_level: Echelon level (1=store, 2=DC, 3=regional DC).

    Returns:
        Risk score between 0.0 and 100.0.
    """
    if echelon_rop <= 0:
        return 0.0
    shortfall_ratio = max(0.0, (echelon_rop - current_on_hand) / echelon_rop)
    if shortfall_ratio == 0:
        # Still above ROP — assign low-but-nonzero risk for monitoring
        coverage_ratio = current_on_hand / echelon_rop
        shortfall_ratio = max(0.0, 1.0 - coverage_ratio) * 0.3  # muted risk

    amplifier = math.log(1 + downstream_loc_count) * echelon_level
    raw_score  = shortfall_ratio * 100 * min(amplifier, 5.0)  # cap amplifier at 5×
    return round(min(100.0, raw_score), 2)


def build_network_graph(conn) -> dict[str, list[str]]:
    """
    Returns {parent_loc: [child_loc, ...]} adjacency list from dim_supply_network.
    Only includes active network arcs effective today.
    """
    cur = conn.execute("""
        SELECT from_loc, to_loc
        FROM dim_supply_network
        WHERE active = TRUE
          AND effective_from <= CURRENT_DATE
          AND (effective_to IS NULL OR effective_to >= CURRENT_DATE)
        ORDER BY from_loc, to_loc
    """)
    graph   = defaultdict(list)
    reverse = defaultdict(list)  # child → parents
    for from_loc, to_loc in cur.fetchall():
        graph[from_loc].append(to_loc)
        reverse[to_loc].append(from_loc)
    return dict(graph), dict(reverse)


def topological_sort_bottom_up(graph: dict, reverse: dict) -> list[str]:
    """
    Returns locations in bottom-up order: leaf nodes (stores) first,
    root nodes (DCs, factories) last.
    Uses Kahn's algorithm on the reverse graph.
    """
    # Leaf nodes have no outgoing edges in reverse (no children)
    in_degree = {loc: len(children) for loc, children in graph.items()}
    all_locs  = set(graph.keys()) | set(
        child for children in graph.values() for child in children
    )
    for loc in all_locs:
        if loc not in in_degree:
            in_degree[loc] = 0

    queue = deque([loc for loc, deg in in_degree.items() if deg == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for parent in reverse.get(node, []):
            in_degree[parent] -= 1
            if in_degree[parent] == 0:
                queue.append(parent)
    return order


def fetch_leaf_node_stats(conn, item_no: str, loc: str) -> Optional[dict]:
    """
    For a leaf node (store), fetch its demand stats from single-echelon safety stock computation.
    Returns None if no data available.
    """
    row = conn.execute("""
        SELECT ss_combined, service_level_target, z_score,
               mean_demand_daily, sigma_demand_daily, lead_time_days
        FROM fact_safety_stock_targets
        WHERE item_no = %s AND loc = %s
        ORDER BY computed_at DESC
        LIMIT 1
    """, (item_no, loc)).fetchone()
    if row is None:
        return None
    return {
        "ss_qty": float(row[0] or 0),
        "service_level_target": float(row[1] or 95),
        "z_score": float(row[2] or 1.65),
        "mean_demand_daily": float(row[3] or 0),
        "sigma_demand_daily": float(row[4] or 0),
        "lead_time_days": float(row[5] or 1),
    }


def fetch_upstream_lt(conn, from_loc: str, to_loc: str) -> tuple[float, float]:
    """Returns (mean_lt_days, sigma_lt_days) for the from_loc→to_loc arc."""
    row = conn.execute("""
        SELECT replenishment_lt_days, 0  -- sigma_lt from dim_lead_time_profile when available
        FROM dim_supply_network
        WHERE from_loc = %s AND to_loc = %s AND active = TRUE
        LIMIT 1
    """, (from_loc, to_loc)).fetchone()
    if row:
        return float(row[0]), float(row[1])
    return 10.0, 0.0  # fallback


def fetch_current_on_hand(conn, item_no: str, loc: str) -> float:
    """Fetch latest on-hand quantity from inventory snapshot."""
    row = conn.execute("""
        SELECT qty_on_hand
        FROM fact_inventory_snapshot
        WHERE item_no = %s AND loc = %s
        ORDER BY snapshot_date DESC
        LIMIT 1
    """, (item_no, loc)).fetchone()
    return float(row[0]) if row else 0.0


def generate_network_alert(
    item_no: str, from_loc: str, echelon_rop: float,
    current_on_hand: float, downstream_count: int,
    coverage_days: float, cascade_risk: float,
) -> Optional[dict]:
    """
    Return alert dict if DC is below ROP, else None.
    Severity based on cascade_risk_score and coverage_days.
    """
    if current_on_hand >= echelon_rop:
        return None
    shortfall = echelon_rop - current_on_hand
    severity  = "critical" if cascade_risk >= 70 else ("high" if cascade_risk >= 40 else "medium")
    return {
        "item_no": item_no,
        "from_loc": from_loc,
        "to_loc": None,
        "alert_date": date.today(),
        "alert_type": "cascade_stockout_risk",
        "severity": severity,
        "affected_downstream_locations": downstream_count,
        "dc_coverage_days": round(coverage_days, 1),
        "current_on_hand_qty": current_on_hand,
        "echelon_rop_qty": echelon_rop,
        "shortfall_qty": round(shortfall, 2),
        "cascade_risk_score": cascade_risk,
        "alert_message": (
            f"{from_loc} on-hand ({current_on_hand:.0f} units) is below echelon ROP "
            f"({echelon_rop:.0f} units). Shortfall: {shortfall:.0f} units. "
            f"Coverage: {coverage_days:.1f} days. "
            f"Cascade risk to {downstream_count} downstream locations."
        ),
    }


def run(item_filter: Optional[str] = None, dry_run: bool = False) -> None:
    cfg = load_config()
    default_z = cfg.get("default_z_score", 1.65)
    today     = date.today()

    log.info("Computing echelon targets (dry_run=%s)", dry_run)

    with psycopg.connect(**get_db_params()) as conn:
        graph, reverse = build_network_graph(conn)
        if not graph and not reverse:
            log.warning("No active supply network arcs found in dim_supply_network. "
                        "Configure the network topology first.")
            return

        # Bottom-up traversal order
        processing_order = topological_sort_bottom_up(graph, reverse)

        # Cache: loc → {item_no → stats_dict}
        loc_stats: dict[str, dict[str, dict]] = {}

        # Fetch all distinct items to process
        items_cur = conn.execute("""
            SELECT DISTINCT item_no FROM fact_safety_stock_targets
            WHERE (%s IS NULL OR item_no = %s)
        """, (item_filter, item_filter))
        all_items = [r[0] for r in items_cur.fetchall()]

        written_targets = 0
        written_alerts  = 0

        for loc in processing_order:
            children = graph.get(loc, [])
            is_leaf  = len(children) == 0
            parents  = reverse.get(loc, [])
            echelon_level = 1 if is_leaf else (2 if parents else 3)

            for item_no in all_items:
                if is_leaf:
                    # Store node: use single-echelon SS from fact_safety_stock_targets
                    stats = fetch_leaf_node_stats(conn, item_no, loc)
                    if stats is None:
                        continue
                    loc_stats.setdefault(loc, {})[item_no] = {
                        "mean_demand_daily": stats["mean_demand_daily"],
                        "sigma_demand_daily": stats["sigma_demand_daily"],
                        "z_score": stats["z_score"],
                        "ss_qty": stats["ss_qty"],
                        "lead_time_days": stats["lead_time_days"],
                    }
                    # Write leaf node record to fact_echelon_targets (echelon_level=1)
                    on_hand    = fetch_current_on_hand(conn, item_no, loc)
                    single_rop = stats["mean_demand_daily"] * stats["lead_time_days"] + stats["ss_qty"]
                    coverage   = on_hand / max(stats["mean_demand_daily"], 0.001)
                    cascade    = 0.0  # stores do not cascade to other stores

                    row = {
                        "item_no": item_no, "loc": loc,
                        "echelon_level": 1, "parent_loc": parents[0] if parents else None,
                        "computation_date": today,
                        "downstream_dfu_count": 0, "downstream_loc_count": 0,
                        "mean_demand_daily": stats["mean_demand_daily"],
                        "pooled_sigma_demand": stats["sigma_demand_daily"],
                        "naive_sigma_demand": stats["sigma_demand_daily"],
                        "pooling_benefit_pct": 0.0,
                        "upstream_lt_days": stats["lead_time_days"],
                        "upstream_sigma_lt_days": 0.0,
                        "z_score": stats["z_score"],
                        "echelon_ss_qty": stats["ss_qty"],
                        "echelon_rop_qty": round(single_rop, 2),
                        "current_on_hand_qty": on_hand,
                        "dc_coverage_days": round(coverage, 2),
                        "cascade_risk_score": cascade,
                        "downstream_locations_at_risk": 0,
                        "prior_echelon_ss_qty": None,
                    }
                    if not dry_run:
                        _upsert_echelon_target(conn, row)
                        written_targets += 1

                else:
                    # DC / Parent node: aggregate downstream demand
                    downstream_sigmas_sq = []
                    downstream_means     = []
                    downstream_count     = 0

                    for child_loc in children:
                        child_stats = loc_stats.get(child_loc, {}).get(item_no)
                        if child_stats is None:
                            continue
                        downstream_sigmas_sq.append(child_stats["sigma_demand_daily"] ** 2)
                        downstream_means.append(child_stats["mean_demand_daily"])
                        downstream_count += 1

                    if not downstream_means:
                        continue

                    mean_demand_daily   = sum(downstream_means)
                    pooled_sigma        = math.sqrt(sum(downstream_sigmas_sq))
                    naive_sigma         = sum(math.sqrt(sq) for sq in downstream_sigmas_sq)
                    pooling_benefit_pct = (1 - pooled_sigma / max(naive_sigma, 0.001)) * 100

                    # Upstream LT for this DC (from its parent or from external supplier)
                    parent_loc  = parents[0] if parents else None
                    up_lt, up_σ = fetch_upstream_lt(conn, parent_loc or "SUPPLIER", loc)

                    z = default_z  # Use configured Z score for DC level
                    ss = compute_echelon_ss(
                        z_score=z,
                        pooled_sigma_demand=pooled_sigma,
                        upstream_lt_days=up_lt,
                        mean_demand_daily=mean_demand_daily,
                        upstream_sigma_lt_days=up_σ,
                    )
                    rop         = round(mean_demand_daily * up_lt + ss, 2)
                    on_hand     = fetch_current_on_hand(conn, item_no, loc)
                    coverage    = on_hand / max(mean_demand_daily, 0.001)
                    cascade_risk = compute_cascade_risk_score(
                        on_hand, rop, downstream_count, echelon_level
                    )

                    # Store stats for parent of this DC
                    loc_stats.setdefault(loc, {})[item_no] = {
                        "mean_demand_daily": mean_demand_daily,
                        "sigma_demand_daily": pooled_sigma,
                        "z_score": z,
                        "ss_qty": ss,
                        "lead_time_days": up_lt,
                    }

                    row = {
                        "item_no": item_no, "loc": loc,
                        "echelon_level": echelon_level,
                        "parent_loc": parent_loc,
                        "computation_date": today,
                        "downstream_dfu_count": downstream_count,
                        "downstream_loc_count": downstream_count,
                        "mean_demand_daily": round(mean_demand_daily, 4),
                        "pooled_sigma_demand": round(pooled_sigma, 4),
                        "naive_sigma_demand": round(naive_sigma, 4),
                        "pooling_benefit_pct": round(pooling_benefit_pct, 2),
                        "upstream_lt_days": up_lt,
                        "upstream_sigma_lt_days": up_σ,
                        "z_score": z,
                        "echelon_ss_qty": ss,
                        "echelon_rop_qty": rop,
                        "current_on_hand_qty": on_hand,
                        "dc_coverage_days": round(coverage, 2),
                        "cascade_risk_score": cascade_risk,
                        "downstream_locations_at_risk": downstream_count if on_hand < rop else 0,
                        "prior_echelon_ss_qty": None,
                    }

                    if dry_run:
                        log.info("[DRY-RUN] %s / %s | echelon=%d | σ_pooled=%.1f | SS=%.0f | ROP=%.0f | "
                                 "on_hand=%.0f | below_rop=%s | cascade_risk=%.1f",
                                 item_no, loc, echelon_level, pooled_sigma, ss, rop,
                                 on_hand, on_hand < rop, cascade_risk)
                    else:
                        _upsert_echelon_target(conn, row)
                        written_targets += 1
                        alert = generate_network_alert(
                            item_no, loc, rop, on_hand, downstream_count, coverage, cascade_risk
                        )
                        if alert:
                            conn.execute("""
                                INSERT INTO fact_network_alerts
                                    (item_no, from_loc, to_loc, alert_date, alert_type, severity,
                                     affected_downstream_locations, dc_coverage_days,
                                     current_on_hand_qty, echelon_rop_qty, shortfall_qty,
                                     cascade_risk_score, alert_message)
                                VALUES (%(item_no)s,%(from_loc)s,%(to_loc)s,%(alert_date)s,
                                        %(alert_type)s,%(severity)s,
                                        %(affected_downstream_locations)s,%(dc_coverage_days)s,
                                        %(current_on_hand_qty)s,%(echelon_rop_qty)s,
                                        %(shortfall_qty)s,%(cascade_risk_score)s,%(alert_message)s)
                                ON CONFLICT DO NOTHING
                            """, alert)
                            written_alerts += 1

        if not dry_run:
            conn.commit()
        log.info("Done. Echelon targets written: %d. Cascade alerts: %d",
                 written_targets, written_alerts)


def _upsert_echelon_target(conn, row: dict) -> None:
    sql = """
        INSERT INTO fact_echelon_targets (
            item_no, loc, echelon_level, parent_loc, computation_date,
            downstream_dfu_count, downstream_loc_count,
            mean_demand_daily, pooled_sigma_demand, naive_sigma_demand, pooling_benefit_pct,
            upstream_lt_days, upstream_sigma_lt_days, z_score,
            echelon_ss_qty, echelon_rop_qty,
            current_on_hand_qty, dc_coverage_days, cascade_risk_score,
            downstream_locations_at_risk, computed_at
        ) VALUES (
            %(item_no)s,%(loc)s,%(echelon_level)s,%(parent_loc)s,%(computation_date)s,
            %(downstream_dfu_count)s,%(downstream_loc_count)s,
            %(mean_demand_daily)s,%(pooled_sigma_demand)s,%(naive_sigma_demand)s,%(pooling_benefit_pct)s,
            %(upstream_lt_days)s,%(upstream_sigma_lt_days)s,%(z_score)s,
            %(echelon_ss_qty)s,%(echelon_rop_qty)s,
            %(current_on_hand_qty)s,%(dc_coverage_days)s,%(cascade_risk_score)s,
            %(downstream_locations_at_risk)s,NOW()
        )
        ON CONFLICT (item_no, loc, computation_date)
        DO UPDATE SET
            echelon_ss_qty              = EXCLUDED.echelon_ss_qty,
            echelon_rop_qty             = EXCLUDED.echelon_rop_qty,
            cascade_risk_score          = EXCLUDED.cascade_risk_score,
            current_on_hand_qty         = EXCLUDED.current_on_hand_qty,
            downstream_locations_at_risk = EXCLUDED.downstream_locations_at_risk,
            computed_at                 = NOW()
    """
    conn.execute(sql, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-no", help="Filter to single item")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(item_filter=args.item_no, dry_run=args.dry_run)
```

---

## 6. Config File

### `config/network_planning_config.yaml`

```yaml
# Network / Multi-Echelon Planning Configuration — Feature F3.5

# Default Z-score for DC-level safety stock (95% service level = 1.645)
# DC service level may differ from store service level
default_z_score: 1.645

# Cascade risk thresholds for alert severity
cascade_risk_critical_threshold: 70.0   # score >= 70 → critical
cascade_risk_high_threshold: 40.0       # score >= 40 → high

# Demand independence assumption for pooling calculation
# "independent" = σ_DC = sqrt(sum(σ_i²))
# "correlated"  = requires correlation matrix (future enhancement)
demand_correlation_assumption: independent

# Minimum downstream locations for pooling benefit to apply
# (1 store = no pooling, needs at least 2)
min_downstream_for_pooling: 2

# DC coverage warning thresholds (days of supply)
dc_coverage_critical_days: 7
dc_coverage_warning_days: 14

# Alert deduplication: do not re-raise the same alert within N days
alert_cooldown_days: 1
```

---

## 7. API Endpoints

### `GET /supply/network/topology`

Returns the supply network structure.

**Response:**
```json
{
  "nodes": [
    { "loc": "DC-EAST", "echelon_level": 2, "type": "distribution_center", "children": ["STORE-A","STORE-B","STORE-C"] },
    { "loc": "STORE-A", "echelon_level": 1, "type": "store", "children": [] }
  ],
  "arcs": [
    { "from_loc": "DC-EAST", "to_loc": "STORE-A", "replenishment_lt_days": 1, "transfer_method": "pull" },
    { "from_loc": "DC-EAST", "to_loc": "STORE-B", "replenishment_lt_days": 1, "transfer_method": "pull" },
    { "from_loc": "DC-EAST", "to_loc": "STORE-C", "replenishment_lt_days": 2, "transfer_method": "pull" }
  ]
}
```

### `GET /supply/network/echelon-targets?item_no=100320&loc=DC-EAST`

Returns echelon-aware SS and ROP for a specific DC or store.

**Response:**
```json
{
  "item_no": "100320",
  "loc": "DC-EAST",
  "echelon_level": 2,
  "computation_date": "2026-03-06",
  "mean_demand_daily": 45.0,
  "pooled_sigma_demand": 26.3,
  "naive_sigma_demand": 45.0,
  "pooling_benefit_pct": 41.6,
  "upstream_lt_days": 10.0,
  "echelon_ss_qty": 137.0,
  "echelon_rop_qty": 587.0,
  "current_on_hand_qty": 500.0,
  "dc_coverage_days": 11.1,
  "cascade_risk_score": 82.4,
  "is_below_rop": true,
  "downstream_locations_at_risk": 3,
  "alert": "DC-EAST on-hand (500) is below echelon ROP (587). 3 downstream stores at risk."
}
```

### `GET /supply/network/cascade-risks?min_risk_score=40`

Returns all DC locations currently below their echelon ROP.

**Response:**
```json
{
  "total_at_risk": 7,
  "items": [
    {
      "item_no": "100320",
      "loc": "DC-EAST",
      "cascade_risk_score": 82.4,
      "shortfall_qty": 87.0,
      "dc_coverage_days": 11.1,
      "downstream_locations_at_risk": 3,
      "severity": "critical"
    }
  ]
}
```

### `POST /supply/network/topology`

Add or update a supply network arc (requires auth).

**Request body:**
```json
{
  "from_loc": "DC-EAST",
  "to_loc": "STORE-D",
  "relationship_type": "dc_to_store",
  "replenishment_lt_days": 1,
  "transfer_method": "pull",
  "transfer_frequency": "daily",
  "effective_from": "2026-04-01"
}
```

---

## 8. Frontend Components

### Location: "Network Planning" panel in InvPlanningTab

```
┌───────────────────────────────────────────────────────────────────────────┐
│  NETWORK PLANNING — MULTI-ECHELON                    [Configure Network]  │
│                                                                             │
│  CRITICAL: 7 DCs are below echelon ROP — affecting 23 downstream stores  │
│                                                                             │
│  Supply Network Topology:                                                   │
│                                                                             │
│       [SUPPLIER]                                                            │
│           │ 10 days                                                         │
│           ▼                                                                 │
│      [DC-EAST] ← 500 units ON HAND | ROP=587 ⚠ BELOW | 11.1 days cover   │
│       ╱  │   ╲                                                              │
│    1d ╱  │1d  ╲ 2d                                                          │
│      ╱   │     ╲                                                            │
│ [STR-A] [STR-B] [STR-C]                                                    │
│  ✓ OK   ✓ OK    ✓ OK                                                       │
│                                                                             │
│  Item: [100320 ▼]                                                           │
│                                                                             │
│  DC-EAST — Item 100320 — Echelon Metrics:                                  │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │  Pooled σ demand:    26.3 units/day                  │                  │
│  │  Naive σ demand:     45.0 units/day (3 stores × σ)  │                  │
│  │  Pooling benefit:    41.6% less SS needed            │                  │
│  │  Echelon SS:        137 units  (vs 225 units naive)  │                  │
│  │  Echelon ROP:       587 units                        │                  │
│  │  Current on-hand:   500 units  ← BELOW ROP           │                  │
│  │  Coverage:          11.1 days                        │                  │
│  │  Cascade risk:      82.4 / 100  [CRITICAL]           │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                                                                             │
│  [Flag for Reorder]  [View Cascade Detail]                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

**Network Configuration UI** (accessed via "Configure Network" button):

```
┌──────────────────────────────────────────────────────────────────────┐
│  SUPPLY NETWORK CONFIGURATION                            [Save]      │
│                                                                       │
│  From Location  →  To Location    LT (days)  Frequency  Method      │
│  ─────────────     ────────────   ─────────  ─────────  ──────      │
│  DC-EAST        →  STORE-A        1          Daily      Pull   [✕]  │
│  DC-EAST        →  STORE-B        1          Daily      Pull   [✕]  │
│  DC-EAST        →  STORE-C        2          Daily      Pull   [✕]  │
│  SUPPLIER-01    →  DC-EAST        10         Weekly     Pull   [✕]  │
│                                                                       │
│  [+ Add Network Arc]                                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Worked Example — Complete Numbers

**Item 100320. DC-EAST serving 3 downstream stores.**

### Step 1: Leaf node (store) stats

| Store   | Mean Demand/day | σ Demand/day | Single-echelon SS | LT (DC→Store) |
|---------|----------------|--------------|-------------------|---------------|
| STORE-A | 15.0           | 15.0         | 45 units          | 1 day         |
| STORE-B | 12.0           | 12.0         | 36 units          | 1 day         |
| STORE-C | 18.0           | 18.0         | 54 units          | 2 days        |

### Step 2: DC-EAST aggregation

```
Mean daily demand at DC = 15.0 + 12.0 + 18.0 = 45.0 units/day

Pooled σ (independence assumption):
  σ_DC = sqrt(15² + 12² + 18²)
       = sqrt(225 + 144 + 324)
       = sqrt(693)
       = 26.32 units/day

Naive σ (if planned like 3 separate stores):
  σ_naive = 15 + 12 + 18 = 45 units/day

Pooling benefit = (1 - 26.32/45) × 100 = 41.5%
(41.5% less safety stock needed at DC vs naive approach)
```

### Step 3: Echelon SS for DC-EAST

```
Z = 1.645 (95% service level for DC)
σ_LT (DC's supplier) = 0 (assumes stable supplier lead time)
LT (supplier → DC) = 10 days

SS_DC = Z × sqrt(LT × σ_demand_pooled² + μ_demand² × σ_LT²)
      = 1.645 × sqrt(10 × 26.32² + 45² × 0²)
      = 1.645 × sqrt(10 × 692.74)
      = 1.645 × sqrt(6927.4)
      = 1.645 × 83.23
      = 136.9 ≈ 137 units

ROP_DC = mean_demand × upstream_LT + SS
       = 45.0 × 10 + 137
       = 450 + 137
       = 587 units

Naive ROP (without pooling, using sum of individual store SS):
  SS_naive = 45 + 36 + 54 = 135 units ... actually same here due to σ coincidence
  ROP_naive_demand = 45 × 10 = 450
  ROP_naive = 450 + 135 = 585 ... (close in this case)

Key insight: For 10 stores σ=15 each:
  σ_pooled = 15 × sqrt(10) = 47.4 vs naive = 150 units/day
  SS savings = (1 - 47.4/150) × 100 = 68% less SS
  Effect grows significantly with more downstream locations.
```

### Step 4: Current on-hand = 500 units

```
DC-EAST on-hand: 500 units
Echelon ROP:     587 units

BELOW ROP: YES
Shortfall: 587 - 500 = 87 units

Coverage: 500 / 45 = 11.1 days

cascade_risk_score calculation:
  shortfall_ratio = (587 - 500) / 587 = 0.148
  amplifier       = log(1 + 3) × 2    = 1.386 × 2 = 2.772
  raw_score       = 0.148 × 100 × min(2.772, 5.0) = 14.8 × 2.772 = 41.0
  cascade_risk    = 41.0 → HIGH severity

ALERT generated:
  "DC-EAST on-hand (500 units) is below echelon ROP (587 units).
   Shortfall: 87 units. Coverage: 11.1 days.
   Cascade risk to 3 downstream locations."
```

### Step 5: What the planner sees and does
```
UI: DC-EAST | BELOW ROP ⚠ | 3 stores at risk | Cascade risk: 41.0 (HIGH)
Action options:
  1. "Flag for Reorder" → creates exception in fact_replenishment_exceptions
     recommended_order_qty = (587 - 500) + SS_target = 87 + 137 = 224 units (bring back to ROP + 1×SS)
  2. View cascade detail → shows which 3 stores and their current on-hand vs daily demand
```

---

## 10. Dependencies

| Dependency | Required For | Status |
|---|---|---|
| IPfeature3 — Safety Stock | Leaf node (store) SS and Z scores | Implemented |
| IPfeature4 — EOQ | Transfer quantity sizing | Implemented |
| IPfeature7 — Exception Queue | Cascade alerts → exception records | Implemented |
| feature_06_10 — Lead Time Learning | `sigma_lt_days` for upstream arc | Planned |
| ERP Location Hierarchy (F1.3) | Automatic network topology import | Future |

---

## 11. Out of Scope

- Automated transfer order generation — system recommends reorder, planner approves
- Stochastic network optimization (chance-constrained programming) — beyond MVP scope
- 4+ echelon networks (factory → regional DC → local DC → store → vending) — requires N-echelon solver
- Correlated demand across stores (e.g., regional weather events affecting all stores in a city)
- Returns / reverse logistics flows
- Cross-docking optimization
- Multi-item batch transfer scheduling (truckload optimization)

---

## 12. Test Requirements

### Backend Unit Tests — `tests/unit/test_echelon_planning.py`

```
test_compute_echelon_ss_no_lt_variance()        — σ_LT=0 reduces to Z×σ_d×sqrt(LT)
test_compute_echelon_ss_with_lt_variance()      — full formula with both variance terms
test_compute_echelon_ss_three_stores_example()  — DC-EAST worked example: SS≈137
test_pooled_sigma_three_stores()                — sqrt(15²+12²+18²) = 26.3
test_pooling_benefit_pct()                      — 41.5% savings for 3-store DC
test_cascade_risk_above_rop_zero()              — on_hand >= rop → risk still low/nonzero
test_cascade_risk_below_rop_high()              — shortfall × downstream count drives score
test_cascade_risk_critical_threshold()          — score >= 70 when deep below ROP with many stores
test_topological_sort_simple_chain()            — store → DC → factory order
test_topological_sort_two_stores_one_dc()       — both stores before DC
test_build_network_graph_returns_adjacency()    — correct parent-child mapping
test_generate_alert_below_rop_returns_dict()    — on_hand < rop → alert generated
test_generate_alert_above_rop_returns_none()    — on_hand >= rop → no alert
test_echelon_rop_formula()                      — ROP = mean × LT + SS
```

### Backend API Tests — `tests/api/test_network_planning.py`

```
test_topology_endpoint_returns_nodes_and_arcs()
test_topology_empty_when_no_network_configured()
test_echelon_targets_dc_below_rop()
test_echelon_targets_store_no_cascade()
test_cascade_risks_filters_by_min_score()
test_cascade_risks_empty_when_all_above_rop()
test_post_topology_creates_arc()
test_post_topology_requires_auth()
test_post_topology_rejects_duplicate_arc()
test_echelon_targets_unknown_item_returns_404()
```

### Make Targets to Add

```makefile
network-schema:       # Apply DDL (dim_supply_network, fact_echelon_targets, fact_network_alerts)
network-configure:    # Import network topology from CSV: make network-configure INPUT=data/network.csv
network-compute:      # Run compute_echelon_targets.py
network-compute-dry:  # --dry-run preview
network-all:          # network-schema + network-compute
```
