# Story 01 — Bill of Materials & Production Planning

## Problem
The system has no concept of BOM (Bill of Materials), production scheduling, or manufacturing constraints. It cannot answer "Can we produce enough to meet demand?" or plan component procurement for assembled/manufactured products.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `bom_master.csv` | CSV | parent_item × component_item | parent_item_no, component_item_no, qty_per_assembly, uom, bom_level, scrap_pct, effective_from, effective_to | ERP/PLM |
| **F2** | `work_centers.csv` | CSV | work_center | work_center_id, name, capacity_hrs_per_day, efficiency_pct, cost_per_hr, shift_pattern, location_id | ERP |
| **F3** | `routing_master.csv` | CSV | item × operation | item_no, operation_seq, work_center_id, setup_time_hrs, run_time_per_unit_hrs, move_time_hrs, queue_time_hrs | ERP |
| **F4** | `production_orders.csv` | CSV | production_order | order_id, item_no, loc, order_qty, completed_qty, scrapped_qty, status, planned_start, planned_end, actual_start, actual_end | MES/ERP |
| **F5** | `production_calendar.csv` | CSV | work_center × date | work_center_id, calendar_date, available_hrs, is_holiday, shift_code, maintenance_window | ERP |

## Incremental Implementation

### Phase 1: BOM Data Model (1 sprint)
- `sql/074_create_bom_tables.sql` — `dim_bom_header`, `dim_bom_component`, `dim_work_center`, `dim_routing`
- `scripts/normalize_bom_csv.py` + `scripts/load_bom_postgres.py`
- `api/routers/bom.py` — `GET /bom/{item_no}/tree`, `/bom/{item_no}/where-used`, `/bom/explosion`
- Vite proxy: `/bom`

### Phase 2: MRP Explosion (1 sprint)
- `scripts/run_mrp_explosion.py` — time-phased component requirements from demand plan
- `sql/075_create_mrp_results.sql` — `fact_mrp_requirements` (item, loc, period, gross_req, scheduled_receipts, projected_available, net_req, planned_order_release)
- `api/routers/mrp.py` — `GET /mrp/requirements`, `/mrp/pegging`, `/mrp/action-messages`

### Phase 3: Capacity Planning (1 sprint)
- `scripts/compute_capacity_plan.py` — load work centers against routing hours
- `sql/076_create_capacity_plan.sql` — `fact_capacity_load` (work_center, period, required_hrs, available_hrs, utilization_pct)
- `frontend/src/tabs/inv-planning/CapacityPanel.tsx` — utilization heatmap, bottleneck identification

### Phase 4: Production Scheduling (1 sprint)
- `scripts/generate_production_schedule.py` — finite capacity scheduling with sequencing rules
- `fact_production_schedule` — start/end times per order per work center
- Frontend: Gantt chart panel for production timeline

## Dependencies
- Requires BOM data from ERP (SAP PP, Oracle Manufacturing, or similar)
- `dim_item` must be extended with `item_type` (purchased | manufactured | phantom)
- Safety stock engine must differentiate manufactured vs purchased items

## Business Value
- Enables integrated demand-to-production planning
- Component shortage visibility before stockout occurs
- Capacity bottleneck identification weeks in advance
- Make-vs-buy decision support
