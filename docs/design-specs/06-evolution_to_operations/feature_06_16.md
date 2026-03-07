# Feature F4.4 — What-If Scenario Planning for Supply Chain Disruptions

**Phase:** 4 — Evolution to Operations
**Feature Number:** F4.4 (Internal: feature_06_16)
**Status:** Specification
**Author:** Supply Chain Systems Architecture
**Date:** 2026-03-06

---

## 1. Problem Statement

The system can model what happens if you change K-means clustering parameters (Feature 38/39). It cannot model what happens when the real world changes unexpectedly — and supply chain reality changes constantly. Suppliers miss commitments. Demand collapses. Distribution centers flood. Trade policies shift overnight.

Today, when a supply chain disruption occurs, the response is ad hoc: planners manually export data to Excel, create scenarios by hand, evaluate options based on intuition, and present the results in a PowerPoint 3 days later. By then the disruption has either been absorbed inefficiently or caused irreversible service level damage.

### What Fails Today

**Scenario A — Lead time shock:**
ABC Trading Co. informs the procurement team that their lead time is increasing from 12 to 24 days due to port congestion — effective immediately, for 90 days. The current system has no way to answer: "Which of our 847 Beverages SKUs at 3 DCs will stock out in the next 30 days if we don't act? What is the additional safety stock investment required to maintain 97% service level across all of them?" A planner must manually check each of the 847 SKU-locations. This takes 2 days. By day 2, 23 SKUs are already below reorder point.

**Scenario B — Demand collapse:**
Sales drops 25% across all categories in response to a macroeconomic shock. The system continues generating replenishment recommendations at historical demand rates. The planner cannot quickly compute: "If we freeze all planned orders for C-class items, how much working capital do we protect? Which A-class items still need ordering?" Without a scenario model, the default response is to "wait and see" — costing $600K in unnecessary inventory build.

**Scenario C — Investment trade-off:**
The CFO asks: "If we reduce the inventory budget from $650K to $500K, what service level will we be able to achieve?" The current efficient frontier (IPfeature13) shows a static snapshot. There is no interactive model that lets the CFO explore the trade-off and understand which specific items would need safety stock cuts and what the stockout risk consequence would be.

**Scenario D — DC offline:**
DC East goes offline for 30 days due to a building fire. Demand served by DC East (approximately 4,200 SKU-locations) must be redistributed to DC Central and DC West. The current system has no model of inter-DC demand redistribution, and no way to compute the secondary stockout risk at the receiving DCs from the additional demand.

---

## 2. Objectives

1. Define and run **supply chain disruption scenarios** in a sandbox environment (no overwrite of production data).
2. Support **6 scenario types**: demand shock, lead time shock, supplier disruption, DC disruption, investment scenario, and policy scenario.
3. Compute **item-level projected inventory** under the scenario — quantity, DOS, and stockout risk.
4. Generate **scenario-specific order recommendations** distinct from the production planned orders.
5. Produce **scenario impact reports** comparing scenario vs. baseline across inventory value, service level, order volume, and stockout count.
6. **Compare up to 3 scenarios** side by side in a summary view.
7. Reuse the **APScheduler background job pattern** from Features 38/39 for long-running scenarios.

---

## 3. Scenario Types

```
SCENARIO TYPES
═══════════════════════════════════════════════════════════════════════════

  DEMAND_SHOCK          LEAD_TIME_SHOCK        SUPPLIER_DISRUPTION
  ─────────────         ───────────────        ───────────────────
  Demand changes ±X%    LT increases/           Supplier goes offline
  for defined period    decreases by X days     or reduces capacity
  and item/category     for defined period      to X% for a period
  scope                 and supplier scope      with alt-source option

  DC_DISRUPTION         INVESTMENT_SCENARIO     POLICY_SCENARIO
  ─────────────         ───────────────────     ───────────────
  DC goes offline,      "What service level     "What if we change
  redirects demand to   at $X budget?"          SS coverage from
  alternate DCs         Extends efficient       98%→95% for C-class?"
  for a defined period  frontier model          Shows investment
                                                & SL impact

═══════════════════════════════════════════════════════════════════════════
```

---

## 4. Data Model

### 4.1 New Table: `fact_scenarios`

**Grain:** scenario_id (one row per scenario definition)
**Purpose:** Master registry of all scenario definitions, parameters, and execution status.

```sql
CREATE TABLE fact_scenarios (
    scenario_id         SERIAL PRIMARY KEY,
    scenario_type       VARCHAR(30)    NOT NULL,
    -- Allowed: DEMAND_SHOCK / LEAD_TIME_SHOCK / SUPPLIER_DISRUPTION /
    --          DC_DISRUPTION / INVESTMENT_SCENARIO / POLICY_SCENARIO
    scenario_name       VARCHAR(200)   NOT NULL,
    description         TEXT,
    parameters          JSONB          NOT NULL,
    -- Type-specific params. See section 4.5 for structure per type.
    scope_items         JSONB,          -- ["100320","100321"] or null for all
    scope_locations     JSONB,          -- ["1401-BULK"] or null for all
    scope_suppliers     JSONB,          -- ["ABC_TRADING"] or null for all
    scope_categories    JSONB,          -- ["Beverages"] or null for all
    period_start        DATE           NOT NULL,
    period_end          DATE           NOT NULL,
    horizon_days        INTEGER        NOT NULL DEFAULT 90,
    status              VARCHAR(20)    NOT NULL DEFAULT 'draft',
    -- Allowed: draft / queued / running / completed / failed / archived
    progress_pct        NUMERIC(5,2),   -- 0.0 – 100.0 during run
    job_id              VARCHAR(100),   -- APScheduler job ID if background run
    error_message       TEXT,
    created_by          VARCHAR(100),
    created_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    run_duration_seconds INTEGER,
    CONSTRAINT scenario_name_unique UNIQUE (scenario_name, created_by)
);

CREATE INDEX idx_scenario_type    ON fact_scenarios (scenario_type);
CREATE INDEX idx_scenario_status  ON fact_scenarios (status);
CREATE INDEX idx_scenario_created ON fact_scenarios (created_at DESC);
```

### 4.2 Scenario Parameter JSONB Schemas

```json
// DEMAND_SHOCK parameters
{
  "demand_change_pct": -25.0,
  "apply_to_classes": ["A", "B", "C"],
  "ramp_weeks": 2
}

// LEAD_TIME_SHOCK parameters
{
  "lt_change_days": 12,
  "affected_suppliers": ["ABC_TRADING"],
  "reorder_adjustment": "auto"
}

// SUPPLIER_DISRUPTION parameters
{
  "supplier_id": "ABC_TRADING",
  "capacity_pct": 40.0,
  "alt_supplier_id": "XYZ_SOURCING",
  "alt_supplier_cost_premium_pct": 15.0
}

// DC_DISRUPTION parameters
{
  "offline_dc": "1401-BULK",
  "redirect_to_dcs": ["2201-DC", "3301-DC"],
  "redirect_rule": "nearest_first"
}

// INVESTMENT_SCENARIO parameters
{
  "budget_amount": 500000.00,
  "current_budget": 650000.00,
  "priority_classes": ["A"],
  "service_levels_to_test": [0.90, 0.92, 0.95, 0.97, 0.98]
}

// POLICY_SCENARIO parameters
{
  "target_classes": ["C"],
  "current_service_level": 0.95,
  "new_service_level": 0.85,
  "apply_to_xyz": ["X", "Y"]
}
```

---

### 4.3 New Table: `fact_scenario_results`

**Grain:** scenario_id + result_type + scope + scope_key + period_start
**Purpose:** High-level impact metrics comparing scenario vs. baseline.

```sql
CREATE TABLE fact_scenario_results (
    id              BIGSERIAL PRIMARY KEY,
    scenario_id     INTEGER       NOT NULL REFERENCES fact_scenarios(scenario_id) ON DELETE CASCADE,
    result_type     VARCHAR(50)   NOT NULL,
    -- Allowed: inventory_value / service_level / order_volume / stockout_count /
    --          excess_value / carrying_cost / ss_investment / on_time_delivery
    scope           VARCHAR(30)   NOT NULL,
    -- Allowed: global / category / abc_class / item_loc
    scope_key       VARCHAR(200),  -- NULL for global; category name, class, or "item_no|loc"
    baseline_value  NUMERIC(14,2) NOT NULL,
    scenario_value  NUMERIC(14,2) NOT NULL,
    delta_value     NUMERIC(14,2) GENERATED ALWAYS AS (scenario_value - baseline_value) STORED,
    delta_pct       NUMERIC(8,2)  GENERATED ALWAYS AS (
                        CASE WHEN baseline_value = 0 THEN NULL
                             ELSE (scenario_value - baseline_value) / ABS(baseline_value) * 100
                        END
                    ) STORED,
    period_start    DATE          NOT NULL,
    period_end      DATE          NOT NULL,
    computed_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_scen_results_id    ON fact_scenario_results (scenario_id);
CREATE INDEX idx_scen_results_type  ON fact_scenario_results (scenario_id, result_type);
CREATE INDEX idx_scen_results_scope ON fact_scenario_results (scenario_id, scope, scope_key);
```

---

### 4.4 New Table: `fact_scenario_projections`

**Grain:** scenario_id + item_no + loc + projection_date
**Purpose:** Item-level projected inventory trajectory under the scenario.

```sql
CREATE TABLE fact_scenario_projections (
    id                    BIGSERIAL PRIMARY KEY,
    scenario_id           INTEGER       NOT NULL REFERENCES fact_scenarios(scenario_id) ON DELETE CASCADE,
    item_no               VARCHAR(50)   NOT NULL,
    loc                   VARCHAR(50)   NOT NULL,
    projection_date       DATE          NOT NULL,  -- Daily or weekly grain
    baseline_qty          NUMERIC(12,2) NOT NULL,
    scenario_qty          NUMERIC(12,2) NOT NULL,
    baseline_dos          NUMERIC(8,2),
    scenario_dos          NUMERIC(8,2),
    scenario_stockout_risk BOOLEAN      NOT NULL DEFAULT FALSE,
    stockout_probability   NUMERIC(5,2),  -- 0–100%
    scenario_order_needed  NUMERIC(12,2) DEFAULT 0,  -- Additional units required vs. baseline
    computed_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_scen_proj_id     ON fact_scenario_projections (scenario_id);
CREATE INDEX idx_scen_proj_item   ON fact_scenario_projections (scenario_id, item_no, loc);
CREATE INDEX idx_scen_proj_risk   ON fact_scenario_projections (scenario_id, projection_date)
    WHERE scenario_stockout_risk = TRUE;
```

---

### 4.5 New Table: `fact_scenario_orders`

**Grain:** scenario_id + item_no + loc + order_by_date
**Purpose:** Order recommendations the system would generate under the scenario.

```sql
CREATE TABLE fact_scenario_orders (
    id                    BIGSERIAL PRIMARY KEY,
    scenario_id           INTEGER       NOT NULL REFERENCES fact_scenarios(scenario_id) ON DELETE CASCADE,
    item_no               VARCHAR(50)   NOT NULL,
    loc                   VARCHAR(50)   NOT NULL,
    supplier_id           VARCHAR(50),
    recommended_qty       NUMERIC(12,2) NOT NULL,
    order_by_date         DATE          NOT NULL,
    expected_receipt_date DATE          NOT NULL,
    order_value           NUMERIC(14,2),
    unit_cost             NUMERIC(12,4),
    priority              VARCHAR(10),   -- critical / high / normal / deferrable
    abc_class             CHAR(1),
    scenario_trigger      VARCHAR(50),   -- What caused this order recommendation (e.g. "lt_shock", "demand_surge")
    computed_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_scen_orders_id   ON fact_scenario_orders (scenario_id);
CREATE INDEX idx_scen_orders_item ON fact_scenario_orders (scenario_id, item_no, loc);
CREATE INDEX idx_scen_orders_date ON fact_scenario_orders (scenario_id, order_by_date);
CREATE INDEX idx_scen_orders_crit ON fact_scenario_orders (scenario_id, priority)
    WHERE priority = 'critical';
```

---

## 5. Python Scripts

### 5.1 `scripts/run_supply_chain_scenario.py`

**Purpose:** Master scenario runner. Reads scenario definition → applies disruption adjustments to in-memory copies of demand plan, lead time profiles, and supplier capacity → re-runs inventory projection → generates scenario orders → writes impact results.

```python
# scripts/run_supply_chain_scenario.py

import yaml
import psycopg
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Any
from common.db import get_db_params

CONFIG_PATH = "config/scenario_config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_scenario(conn, scenario_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT scenario_id, scenario_type, parameters, scope_items, scope_locations,
                   scope_suppliers, scope_categories, period_start, period_end, horizon_days
            FROM fact_scenarios WHERE scenario_id = %s
        """, (scenario_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Scenario {scenario_id} not found")
        cols = ["scenario_id","scenario_type","parameters","scope_items","scope_locations",
                "scope_suppliers","scope_categories","period_start","period_end","horizon_days"]
        return dict(zip(cols, row))

def load_baseline_inventory(conn, scope: dict) -> pd.DataFrame:
    """Load current on-hand inventory for all in-scope item-locations."""
    sql = """
        SELECT
            s.item_no, s.loc, s.qty_on_hand, s.qty_on_order,
            d.avg_daily_sales, d.abc_class, d.lead_time_days,
            d.safety_stock_qty, d.reorder_point, d.max_stock_target,
            i.category, i.supplier_id
        FROM (
            SELECT item_no, loc,
                   MAX(qty_on_hand)        AS qty_on_hand,
                   MAX(qty_on_order)       AS qty_on_order,
                   MAX(snapshot_date)      AS snapshot_date
            FROM fact_inventory_snapshot
            WHERE snapshot_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY item_no, loc
        ) s
        JOIN dim_dfu d ON d.item_no = s.item_no AND d.loc = s.loc
        JOIN dim_item i ON i.item_no = s.item_no
    """
    df = pd.read_sql(sql, conn)

    if scope.get("scope_items"):
        df = df[df["item_no"].isin(scope["scope_items"])]
    if scope.get("scope_locations"):
        df = df[df["loc"].isin(scope["scope_locations"])]
    if scope.get("scope_categories"):
        df = df[df["category"].isin(scope["scope_categories"])]
    if scope.get("scope_suppliers"):
        df = df[df["supplier_id"].isin(scope["scope_suppliers"])]

    return df.reset_index(drop=True)

def apply_demand_shock(
    inv_df: pd.DataFrame,
    params: dict,
    period_start: date,
    period_end: date,
) -> pd.DataFrame:
    """
    Adjust avg_daily_sales by demand_change_pct for items of specified classes.

    params = {"demand_change_pct": -25.0, "apply_to_classes": ["A","B","C"]}
    """
    change_factor = 1 + params["demand_change_pct"] / 100
    target_classes = params.get("apply_to_classes", ["A","B","C"])
    mask = inv_df["abc_class"].isin(target_classes)
    result = inv_df.copy()
    result.loc[mask, "avg_daily_sales"] = (
        result.loc[mask, "avg_daily_sales"] * change_factor
    ).clip(lower=0)
    return result

def apply_lead_time_shock(
    inv_df: pd.DataFrame,
    params: dict,
    scope_suppliers: list[str] | None,
) -> pd.DataFrame:
    """
    Increase lead_time_days by lt_change_days for items sourced from affected suppliers.

    params = {"lt_change_days": 12, "affected_suppliers": ["ABC_TRADING"]}
    """
    result = inv_df.copy()
    affected = params.get("affected_suppliers") or scope_suppliers or []
    if affected:
        mask = result["supplier_id"].isin(affected)
        result.loc[mask, "lead_time_days"] = (
            result.loc[mask, "lead_time_days"] + params["lt_change_days"]
        ).clip(lower=0, upper=365)
    return result

def apply_supplier_disruption(
    inv_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Reduce available supply capacity for a supplier to capacity_pct.
    Marks affected items and adjusts reorder point upward to compensate.

    params = {"supplier_id": "ABC_TRADING", "capacity_pct": 40.0, "alt_supplier_cost_premium_pct": 15.0}
    """
    result = inv_df.copy()
    mask = result["supplier_id"] == params["supplier_id"]
    capacity_factor = params["capacity_pct"] / 100
    # Reduce the effective "supply rate": increase reorder point inversely
    result.loc[mask, "max_order_qty_scenario"] = (
        result.loc[mask, "qty_on_hand"] * capacity_factor
    )
    result.loc[mask, "alt_cost_premium_pct"] = params.get("alt_supplier_cost_premium_pct", 0)
    return result

def apply_policy_scenario(
    inv_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Reduce safety stock for target classes from current_service_level to new_service_level.

    params = {"target_classes": ["C"], "current_service_level": 0.95, "new_service_level": 0.85}
    """
    from scipy.stats import norm
    z_old = norm.ppf(params["current_service_level"])
    z_new = norm.ppf(params["new_service_level"])
    reduction_factor = z_new / z_old  # Ratio of Z-scores

    result = inv_df.copy()
    mask = result["abc_class"].isin(params.get("target_classes", ["C"]))
    if "apply_to_xyz" in params:
        mask &= result.get("xyz_class", pd.Series(["X"] * len(result))).isin(params["apply_to_xyz"])

    result.loc[mask, "safety_stock_qty"] = (
        result.loc[mask, "safety_stock_qty"] * reduction_factor
    ).clip(lower=0)
    result.loc[mask, "reorder_point"] = (
        result.loc[mask, "safety_stock_qty"] +
        result.loc[mask, "avg_daily_sales"] * result.loc[mask, "lead_time_days"]
    )
    return result

def project_inventory(
    inv_df: pd.DataFrame,
    horizon_days: int,
    period_start: date,
) -> pd.DataFrame:
    """
    Forward-project inventory for each item-location daily over horizon_days.

    Assumptions:
    - avg_daily_sales is constant (demand shock already applied to this field)
    - Orders arrive at lead_time_days from today if on_hand drops below reorder_point
    - Simple deterministic projection (stochastic extension deferred to future feature)

    Returns long-form DataFrame: columns [item_no, loc, projection_date, projected_qty, dos, stockout_risk]
    """
    records = []
    for _, row in inv_df.iterrows():
        qty          = float(row["qty_on_hand"])
        on_order     = float(row.get("qty_on_order", 0))
        daily_sales  = float(row.get("avg_daily_sales", 0))
        lead_time    = int(row.get("lead_time_days", 14))
        reorder_pt   = float(row.get("reorder_point", 0))
        max_stock    = float(row.get("max_stock_target", qty * 2))
        pending_order_arrival = None

        for d in range(horizon_days):
            proj_date = period_start + timedelta(days=d)

            # Receive pending order if it arrives today
            if pending_order_arrival is not None and d == pending_order_arrival:
                order_qty = min(max_stock - qty, max_stock)
                qty += max(0, order_qty)
                on_order = 0
                pending_order_arrival = None

            # Consume daily sales
            consumed = min(daily_sales, qty)
            qty = max(0, qty - consumed)

            # Check reorder trigger (only if no pending order)
            if qty <= reorder_pt and pending_order_arrival is None:
                order_qty = max_stock - qty
                if order_qty > 0:
                    pending_order_arrival = d + lead_time
                    on_order = order_qty

            dos = qty / max(daily_sales, 0.001)
            stockout_risk = qty <= 0

            records.append({
                "item_no":           row["item_no"],
                "loc":               row["loc"],
                "projection_date":   proj_date,
                "scenario_qty":      round(qty, 2),
                "scenario_dos":      round(dos, 2),
                "scenario_stockout_risk": stockout_risk,
                "abc_class":         row.get("abc_class"),
                "category":          row.get("category"),
            })

    return pd.DataFrame(records)

def compute_baseline_projection(inv_df: pd.DataFrame, horizon_days: int, period_start: date) -> pd.DataFrame:
    """Run projection with unmodified inputs to get baseline comparison."""
    return project_inventory(inv_df, horizon_days, period_start)

def compute_impact_results(
    baseline_proj: pd.DataFrame,
    scenario_proj: pd.DataFrame,
    scenario_id: int,
) -> list[dict]:
    """
    Aggregate scenario vs. baseline projections into impact result records.

    Returns list of dicts for insertion into fact_scenario_results.
    """
    results = []

    # Global: total stockout-days
    b_stockouts = baseline_proj["scenario_stockout_risk"].sum()
    s_stockouts = scenario_proj["scenario_stockout_risk"].sum()
    results.append({
        "scenario_id":    scenario_id,
        "result_type":    "stockout_count",
        "scope":          "global",
        "scope_key":      None,
        "baseline_value": float(b_stockouts),
        "scenario_value": float(s_stockouts),
        "period_start":   baseline_proj["projection_date"].min(),
        "period_end":     baseline_proj["projection_date"].max(),
    })

    # By category: avg DOS
    for category in scenario_proj["category"].dropna().unique():
        b_dos = baseline_proj.loc[baseline_proj["category"] == category, "scenario_dos"].mean()
        s_dos = scenario_proj.loc[scenario_proj["category"] == category, "scenario_dos"].mean()
        results.append({
            "scenario_id":    scenario_id,
            "result_type":    "service_level",
            "scope":          "category",
            "scope_key":      category,
            "baseline_value": round(float(b_dos), 2),
            "scenario_value": round(float(s_dos), 2),
            "period_start":   scenario_proj["projection_date"].min(),
            "period_end":     scenario_proj["projection_date"].max(),
        })

    return results

def write_results(conn, scenario_id: int, results: list[dict], projections: pd.DataFrame) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM fact_scenario_results    WHERE scenario_id = %s", (scenario_id,))
        cur.execute("DELETE FROM fact_scenario_projections WHERE scenario_id = %s", (scenario_id,))

        for r in results:
            cur.execute("""
                INSERT INTO fact_scenario_results
                    (scenario_id, result_type, scope, scope_key, baseline_value, scenario_value,
                     period_start, period_end)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                r["scenario_id"], r["result_type"], r["scope"], r["scope_key"],
                r["baseline_value"], r["scenario_value"], r["period_start"], r["period_end"],
            ))

        proj_rows = [
            (
                scenario_id, r.item_no, r.loc, r.projection_date,
                r.get("baseline_qty", r.scenario_qty),  # Simplified: first run is baseline
                r.scenario_qty, None, r.scenario_dos,
                bool(r.scenario_stockout_risk),
            )
            for _, r in projections.iterrows()
        ]
        cur.executemany("""
            INSERT INTO fact_scenario_projections
                (scenario_id, item_no, loc, projection_date, baseline_qty, scenario_qty,
                 baseline_dos, scenario_dos, scenario_stockout_risk)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, proj_rows)
        conn.commit()

def update_scenario_status(conn, scenario_id: int, status: str, error: str | None = None) -> None:
    with conn.cursor() as cur:
        if status == "completed":
            cur.execute("""
                UPDATE fact_scenarios
                SET status = %s, completed_at = NOW(),
                    run_duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER
                WHERE scenario_id = %s
            """, (status, scenario_id))
        elif status == "failed":
            cur.execute("""
                UPDATE fact_scenarios
                SET status = %s, completed_at = NOW(), error_message = %s
                WHERE scenario_id = %s
            """, (status, error, scenario_id))
        else:
            cur.execute(
                "UPDATE fact_scenarios SET status = %s, started_at = NOW() WHERE scenario_id = %s",
                (status, scenario_id)
            )
        conn.commit()

def run(scenario_id: int) -> None:
    """
    Main entry point. Called by the APScheduler job thread.
    """
    cfg = load_config()

    with psycopg.connect(**get_db_params()) as conn:
        update_scenario_status(conn, scenario_id, "running")
        try:
            scenario = load_scenario(conn, scenario_id)
            params   = scenario["parameters"]
            scope    = {k: scenario[k] for k in
                        ["scope_items","scope_locations","scope_suppliers","scope_categories"]}

            # Load baseline inventory
            inv_df = load_baseline_inventory(conn, scope)
            period_start = scenario["period_start"]
            horizon_days = scenario["horizon_days"]

            # Compute baseline projection (unmodified)
            baseline_proj = compute_baseline_projection(inv_df.copy(), horizon_days, period_start)

            # Apply scenario-specific disruption to inventory inputs
            stype = scenario["scenario_type"]
            if stype == "DEMAND_SHOCK":
                modified_inv = apply_demand_shock(inv_df, params, period_start, scenario["period_end"])
            elif stype == "LEAD_TIME_SHOCK":
                modified_inv = apply_lead_time_shock(inv_df, params, scenario.get("scope_suppliers"))
            elif stype == "SUPPLIER_DISRUPTION":
                modified_inv = apply_supplier_disruption(inv_df, params)
            elif stype == "POLICY_SCENARIO":
                modified_inv = apply_policy_scenario(inv_df, params)
            else:
                modified_inv = inv_df.copy()  # Other types use same inventory inputs

            # Compute scenario projection
            scenario_proj = project_inventory(modified_inv, horizon_days, period_start)

            # Merge baseline_qty into scenario projections for comparison
            baseline_qty_map = (
                baseline_proj.groupby(["item_no","loc","projection_date"])["scenario_qty"]
                .first().rename("baseline_qty").reset_index()
            )
            scenario_proj = scenario_proj.merge(
                baseline_qty_map, on=["item_no","loc","projection_date"], how="left"
            )

            # Compute impact results
            impact_results = compute_impact_results(baseline_proj, scenario_proj, scenario_id)

            # Write to DB
            write_results(conn, scenario_id, impact_results, scenario_proj)
            update_scenario_status(conn, scenario_id, "completed")
            print(f"[scenario] Scenario {scenario_id} completed. "
                  f"{len(scenario_proj):,} projection rows written.")

        except Exception as e:
            update_scenario_status(conn, scenario_id, "failed", str(e))
            raise

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-id", type=int, required=True)
    args = p.parse_args()
    run(args.scenario_id)
```

### 5.2 Config: `config/scenario_config.yaml`

```yaml
# Supply chain scenario planning configuration
max_concurrent_scenarios: 2          # Max scenarios running simultaneously
max_portfolio_size: 15000            # Max DFU count per scenario run
default_horizon_days: 90             # Default projection horizon

# Investment scenario
investment_scenario_service_levels:
  - 0.90
  - 0.92
  - 0.95
  - 0.97
  - 0.98

# Projection parameters
daily_projection_grain: true         # true = daily grain; false = weekly (faster, less precise)
projection_grain_days: 1             # 1 = daily, 7 = weekly

# Performance thresholds for runtime estimation
estimated_seconds_per_1000_dfus: 8   # ~8s per 1,000 DFUs with daily projection

# Background job settings
job_group: "scenarios"
job_timeout_seconds: 1800            # 30-minute timeout for large portfolios
```

---

## 6. API Endpoints

### `POST /scenarios`

Create a new scenario and kick off background execution. Returns HTTP 202.

**Request body:**
```json
{
  "scenario_type": "LEAD_TIME_SHOCK",
  "scenario_name": "ABC Trading LT Double - 90 days",
  "parameters": { "lt_change_days": 12, "affected_suppliers": ["ABC_TRADING"] },
  "scope_suppliers": ["ABC_TRADING"],
  "period_start": "2026-04-01",
  "period_end": "2026-06-30",
  "horizon_days": 90,
  "created_by": "planner_alice"
}
```

**Response (202):**
```json
{
  "scenario_id": 7,
  "status": "queued",
  "estimated_runtime_seconds": 120,
  "message": "Scenario queued for background execution. Poll GET /scenarios/7/status for progress."
}
```

---

### `GET /scenarios/{scenario_id}/status`

**Response:**
```json
{
  "scenario_id": 7,
  "status": "completed",
  "progress_pct": 100.0,
  "started_at": "2026-04-06T09:02:14Z",
  "completed_at": "2026-04-06T09:04:22Z",
  "run_duration_seconds": 128,
  "dfu_count": 847
}
```

---

### `GET /scenarios/{scenario_id}/results`

Returns high-level impact summary.

**Response:**
```json
{
  "scenario_id": 7,
  "scenario_name": "ABC Trading LT Double - 90 days",
  "scenario_type": "LEAD_TIME_SHOCK",
  "summary": {
    "global_stockout_count_baseline": 12,
    "global_stockout_count_scenario": 89,
    "stockout_count_delta": 77,
    "avg_dos_baseline": 28.4,
    "avg_dos_scenario": 19.1,
    "additional_inventory_investment_required": 890000.00,
    "critical_items_count": 23
  },
  "by_category": [
    {
      "category": "Beverages",
      "result_type": "service_level",
      "baseline_value": 28.4,
      "scenario_value": 17.2,
      "delta_value": -11.2,
      "delta_pct": -39.4
    }
  ]
}
```

---

### `GET /scenarios/{scenario_id}/projections`

Returns item-level projection for a specific DFU.

**Query params:** `item_no=100320`, `loc=1401-BULK`

**Response:**
```json
{
  "scenario_id": 7,
  "item_no": "100320",
  "loc": "1401-BULK",
  "projections": [
    { "projection_date": "2026-04-01", "baseline_qty": 120, "scenario_qty": 120, "baseline_dos": 38.7, "scenario_dos": 38.7, "scenario_stockout_risk": false },
    { "projection_date": "2026-04-15", "baseline_qty": 74,  "scenario_qty": 74,  "baseline_dos": 23.9, "scenario_dos": 23.9, "scenario_stockout_risk": false },
    { "projection_date": "2026-04-22", "baseline_qty": 52,  "scenario_qty": 52,  "baseline_dos": 16.8, "scenario_dos": 16.8, "scenario_stockout_risk": false },
    { "projection_date": "2026-04-29", "baseline_qty": 30,  "scenario_qty": 30,  "baseline_dos":  9.7, "scenario_dos":  9.7, "scenario_stockout_risk": true  }
  ]
}
```

---

### `GET /scenarios/{scenario_id}/comparison`

Compare this scenario vs. up to 2 others side by side.

**Query params:** `compare_with=3,5`

**Response:**
```json
{
  "scenarios": [
    { "scenario_id": 7, "scenario_name": "ABC Trading LT Double - 90 days", "stockout_count": 89, "avg_dos": 19.1, "additional_investment": 890000 },
    { "scenario_id": 3, "scenario_name": "Demand Collapse -25%",              "stockout_count":  4, "avg_dos": 42.3, "additional_investment": 0 },
    { "scenario_id": 5, "scenario_name": "Budget $500K",                      "stockout_count": 34, "avg_dos": 22.1, "additional_investment": 0 }
  ]
}
```

---

### `GET /scenarios`

List all scenarios.

**Query params:** `status=completed`, `scenario_type=LEAD_TIME_SHOCK`, `limit=20`

---

### `DELETE /scenarios/{scenario_id}`

Soft-delete (archive) a scenario. Sets status=archived. Clears projection rows.

---

## 7. Frontend Components

### 7.1 "Scenario Planning" Panel in Inv. Planning Tab

```
┌─────────────────────────────────────────────────────────────────────┐
│  SCENARIO PLANNING                        [+ New Scenario]          │
├─────────────────────┬───────────────────────────────────────────────┤
│  SCENARIO LIST      │  SCENARIO DETAIL: "ABC Trading LT Double"     │
│                     │                                               │
│  ● ABC Trading LT   │  Type: LEAD_TIME_SHOCK   Status: COMPLETED   │
│    COMPLETED Apr 6  │  Period: Apr 1 – Jun 30   Duration: 128s      │
│                     │                                               │
│  ● Demand -25%      │  PARAMETERS                                   │
│    COMPLETED Apr 5  │  LT change: +12 days   Suppliers: ABC_TRADING │
│                     │  Affected DFUs: 847                           │
│  ○ Budget $500K     │                                               │
│    RUNNING 42%      │  IMPACT SUMMARY KPI CARDS                     │
│                     │  ┌───────────┬───────────┬───────────────┐    │
│  ○ DC East Offline  │  │Stockout   │ Avg DOS   │ Additional    │    │
│    QUEUED           │  │Count      │           │ Investment    │    │
│                     │  │           │           │ Required      │    │
│  ○ SS Policy C→85%  │  │  77 more  │ -11.2 days│  $890,000    │    │
│    DRAFT            │  │ (↑ 642%)  │ (-39.4%)  │               │    │
│                     │  └───────────┴───────────┴───────────────┘    │
│  [Compare 3 Scen.]  │                                               │
│                     │  CATEGORY IMPACT TABLE                        │
│                     │  Category   Baseline DOS  Scenario DOS  Delta │
│                     │  Beverages    28.4          17.2         -39% │
│                     │  Snacks       31.2          22.8         -27% │
│                     │                                               │
│                     │  ITEM-LEVEL PROJECTION  [Select Item...]      │
│                     │                                               │
│                     │  160 ┤ ─ ─ Baseline                          │
│                     │  120 ┤ ─ ─ ─ ─ ─ ─ ─ ─╮                     │
│                     │   80 ┤         ━━━━━━━━┤ Scenario             │
│                     │   40 ┤                 ╰─╮ STOCKOUT           │
│                     │    0 ┤─────────────────── ▼ Apr 29            │
│                     │      Apr 1  Apr 15  Apr 29  May 13           │
│                     │                                               │
│                     │  CRITICAL ITEMS (stockout in <14 days): 23   │
│                     │  [View Critical Items]  [Export Orders]       │
└─────────────────────┴───────────────────────────────────────────────┘
```

### 7.2 New Scenario Modal

Scenario type selector drives a dynamic form:

| Type | Parameters shown |
|---|---|
| DEMAND_SHOCK | Demand change % slider, apply to classes, ramp weeks |
| LEAD_TIME_SHOCK | LT change days, supplier multi-select |
| SUPPLIER_DISRUPTION | Supplier selector, capacity %, alt supplier, cost premium % |
| DC_DISRUPTION | Offline DC selector, redirect DCs multi-select |
| INVESTMENT_SCENARIO | Budget amount slider (vs. current budget shown), service level targets |
| POLICY_SCENARIO | Target ABC class, current and new service level, XYZ class filter |

After form submission:
1. System creates scenario record (status=draft)
2. Validates parameters
3. Kicks off background run (status → running)
4. Shows progress bar with polling

### 7.3 Scenario Comparison View

Side-by-side table of up to 3 scenarios:

```
┌──────────────────────────────────────────────────────────────────────┐
│  SCENARIO COMPARISON                                                 │
│                      Baseline   LT Shock   Demand -25%  Budget $500K│
│  Stockout Count         12         89           4            34      │
│  Avg DOS (days)         28.4       19.1         42.3         22.1    │
│  Additional Invest.     —        $890K          $0            $0     │
│  Order Volume           $312K    $1.2M         $42K          $195K   │
│  Service Level Est.     97.2%    81.4%         99.1%         94.1%   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. Worked Examples

### Example 1 — LEAD_TIME_SHOCK: ABC Trading doubles LT (12 → 24 days)

**Setup:**
- Supplier ABC Trading: 847 item-locations in Beverages category
- Current lead time: 12 days
- Shocked lead time: 24 days (+12 days)
- Scenario period: April 1 – June 30

**Computation:**

For item 100320 @ 1401-BULK:
```
Current on-hand:      120 units
Avg daily sales:      3.1 units/day
Current SS:           50 units (12 days × 3.1 + σ)
Current ROP:          50 + (12 × 3.1) = 50 + 37.2 = 87.2 units
                      → ROP already triggered (on-hand = 120 > ROP = 87.2 by only 32.8 units buffer)

Under shocked LT (24 days):
New ROP:              50 + (24 × 3.1) = 50 + 74.4 = 124.4 units
Current on-hand 120 < New ROP 124.4 → IMMEDIATE REORDER triggered
Order required:       max_stock - on_hand = 200 - 120 = 80 units
Order value:          80 × $24.50 = $1,960
Order arrives:        April 1 + 24 days = April 25
Stockout risk period: April 1–25 if sales > 120 units = 38.7 days of supply (not yet at risk)
```

**Portfolio impact:**
```
DFUs analyzed:                     847
DFUs requiring immediate reorder:   23  (on-hand < new ROP AND order lead time > current stock)
DFUs with stockout risk in 30 days: 23
DFUs with stockout risk in 60 days: 67
Additional inventory investment needed: $890,000
  (= sum of additional SS required to maintain 97% service level at 24-day LT vs 12-day LT)
Additional carrying cost/month:     $890,000 × 25% / 12 = $18,542/month
```

---

### Example 2 — DEMAND_SHOCK: COVID-style -25% demand collapse

**Setup:**
- Demand drops 25% across all categories for April–June 2026
- All 15,000 DFUs in scope

**Computation:**

For item 100320:
```
Base avg_daily_sales:     3.1 units/day
Shocked avg_daily_sales:  3.1 × (1 - 0.25) = 2.325 units/day
Current on-hand:          120 units
Current DOS:              120 / 3.1 = 38.7 days
Scenario DOS:             120 / 2.325 = 51.6 days (+12.9 days)
→ Item now OVER-STOCKED relative to demand
→ Planned order: freeze (on-hand > reorder_point under reduced demand)
```

**Portfolio impact:**
```
DFUs with reduced DOS:             15,000 (all improve — less demand means more DOS)
Orders that can be deferred:       3,847 planned orders totaling $1.4M
Orders still needed (A-class):       612 totaling $800K
Net working capital preservation:  $600K
Service level improvement:         97.2% → 99.1% (less demand = less stockout risk)
Excess inventory increase:         $1.8M (items now over-stocked)
Recommended action:                Freeze 3,847 planned orders; monitor A-class weekly
```

---

### Example 3 — INVESTMENT_SCENARIO: $500K budget (down from $650K)

**Computation using existing efficient frontier model (IPfeature13):**

```
Current portfolio:
  Inventory investment:    $650,000
  Portfolio service level: 97.2%
  Stockout count (30d):    12

Target budget: $500,000 (-$150K, -23% reduction)

Efficient frontier point at $500K:
  Achievable service level: 94.1% (interpolated from frontier)
  Stockout count (30d):     34

Items to cut (ranked by impact per dollar saved):
  C-class, XZ segment (low variability, predictable): 312 items
  Reduce SS coverage: 95% → 85% (Z = 1.645 → 1.036)
  SS reduction per item (avg): $481
  Total SS reduction: 312 × $481 = $150,072 (meets target)

Residual stockout risk:
  C-class XZ items stockout probability increases from 5% → 15%
  Expected annual stockout events: 312 × 0.15 × 12 months = 562 events
  At avg $85 lost margin per event: $47,770 annual cost of service level reduction
  vs. $150K annual carrying cost saved: Net saving = $102,230/year
```

**Efficient frontier chart data points:**

| Budget ($) | Service Level (%) | Stockout Count |
|---|---|---|
| 350,000 | 88.4 | 112 |
| 450,000 | 91.8 | 67 |
| 500,000 | 94.1 | 34 |
| 575,000 | 96.0 | 18 |
| 650,000 | 97.2 | 12 |
| 750,000 | 98.1 | 6 |
| 900,000 | 98.9 | 2 |

---

## 9. Dependencies

| Dependency | Feature | Status |
|---|---|---|
| Inventory on-hand (baseline) | fact_inventory_snapshot | Implemented |
| Lead time profiles | dim_dfu / fact_lead_time_profile | Partially implemented |
| Safety stock targets | IPfeature3 | Implemented |
| EOQ and order quantities | IPfeature4 | Implemented |
| Efficient frontier model | IPfeature13 | Implemented |
| ABC-XYZ classification | IPfeature11 | Implemented |
| Background job execution | Feature 39 — APScheduler | Implemented |
| Item costs (for investment scenario) | F4.1 — Financial Inventory Plan | Specified |
| Scenario notification context | Feature 38 — ScenarioNotificationContext | Implemented |

---

## 10. Out of Scope

- Multi-echelon DC redistribution modeling (DC_DISRUPTION type creates a simplified flag; full network re-optimization is a separate OR feature)
- Stochastic (Monte Carlo) projection (this feature uses deterministic daily projection; stochastic extension deferred)
- Automated scenario execution triggered by ERP exception flags
- Scenario promotion to production plan (only INVESTMENT_SCENARIO has a "Promote to Plan" option; other scenario types are read-only analysis tools)
- Real-time supplier capacity data feed (manual entry only in this phase)
- Carbon footprint / sustainability scenarios (separate sustainability planning module)
- Cross-scenario constraint optimization (solving for the optimal parameter set across multiple constraints simultaneously — linear programming extension, future phase)

---

## 11. Test Requirements

### Backend Unit Tests (`tests/unit/test_supply_chain_scenario.py`)

- `test_apply_demand_shock_correct_factor()` — -25% shock reduces avg_daily_sales by 25%
- `test_apply_demand_shock_class_filter()` — only specified classes affected
- `test_apply_demand_shock_clip_zero()` — demand cannot go negative
- `test_apply_lead_time_shock_supplier_filter()` — only affected supplier LT changes
- `test_apply_lead_time_shock_max_cap()` — LT capped at 365 days
- `test_apply_policy_scenario_z_ratio()` — SS reduction proportional to Z-score ratio
- `test_project_inventory_stockout_detected()` — DOS reaches 0 → stockout_risk=True
- `test_project_inventory_reorder_trigger()` — order placed when qty <= reorder_point
- `test_project_inventory_order_arrival()` — qty increases at lead_time_days from order
- `test_compute_impact_results_structure()` — output has required fields for each result type
- `test_weeks_overlap_no_overlap()` — non-overlapping periods return 0

### Backend API Tests (`tests/api/test_scenarios.py`)

- `test_post_scenario_202()` — returns 202 with scenario_id and status=queued
- `test_get_status_running()` — returns progress_pct between 0 and 100
- `test_get_status_completed()` — returns completed_at and run_duration_seconds
- `test_get_status_404()` — returns 404 for non-existent scenario_id
- `test_get_results_200()` — returns summary dict with stockout_count and avg_dos
- `test_get_projections_item_filter()` — returns daily rows for specific item_no + loc
- `test_get_comparison_200()` — returns list of up to 3 scenarios with consistent metrics
- `test_list_scenarios_status_filter()` — status=completed filter returns only completed
- `test_delete_scenario_archives()` — status set to 'archived', projection rows removed

### Frontend Tests (`src/tabs/__tests__/ScenarioPlanningPanel.test.tsx`)

- Scenario list renders with status badges (completed/running/queued)
- Impact summary KPI cards render with delta values and direction arrows
- Category impact table renders with delta_pct in red for negative values
- Item projection chart renders with baseline and scenario lines
- Critical items count badge renders correctly
- New scenario modal opens on button click
- Scenario type selector changes form fields dynamically
- Comparison view renders side-by-side table for up to 3 scenarios
- Running scenario shows progress bar with percentage

---

## 12. Makefile Targets

```makefile
scenario-schema:
	@echo "Applying scenario planning DDL..."
	# Apply: fact_scenarios, fact_scenario_results, fact_scenario_projections, fact_scenario_orders

scenario-run:
	uv run python scripts/run_supply_chain_scenario.py --scenario-id $(SCENARIO_ID)

scenario-list:
	@echo "Listing scenario status..."
	# Query fact_scenarios for status counts
```
