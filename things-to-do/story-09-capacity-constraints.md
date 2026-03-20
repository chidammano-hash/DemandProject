# Story 09 — Warehouse & Distribution Capacity Constraints

## Problem
The system optimizes inventory levels and generates replenishment plans but does not check if the receiving location can actually store or handle the incoming volume. Planned orders may exceed warehouse capacity, dock door throughput, or labor availability.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `warehouse_master.csv` | CSV | location | location_id, warehouse_type (dc|regional|store|3pl), total_pallet_positions, total_sqft, temperature_zone (ambient|refrigerated|frozen), max_inbound_units_per_day, max_outbound_units_per_day, dock_doors_inbound, dock_doors_outbound, operating_hours | WMS |
| **F2** | `warehouse_utilization.csv` | CSV | location × week | location_id, week_start, pallet_positions_used, sqft_used, inbound_units, outbound_units, labor_hours_used, labor_hours_available | WMS |
| **F3** | `labor_plan.csv` | CSV | location × week | location_id, week_start, role (picker|packer|receiver|loader), headcount_planned, headcount_actual, hours_planned, hours_actual, overtime_hours, cost | WMS/HR |
| **F4** | `appointment_schedule.csv` | CSV | appointment | appointment_id, location_id, date, time_slot, type (inbound|outbound), carrier_id, po_number, order_id, pallet_count, status | WMS/TMS |

## Incremental Implementation

### Phase 1: Capacity Master & Utilization Tracking (1 sprint)
- `sql/099_create_warehouse_capacity.sql` — `dim_warehouse`, `fact_warehouse_utilization`
- `scripts/load_warehouse_data.py` — weekly utilization feed
- API: `GET /capacity/by-location`, `/capacity/utilization-trend`, `/capacity/at-risk`
- Panel: Capacity utilization heatmap (location × week), threshold alerts

### Phase 2: Capacity-Constrained Replenishment (1 sprint)
- Extend `scripts/generate_planned_orders.py` — check receiving capacity before generating order
- If projected receipts > max inbound: defer order or split across days
- If projected inventory > pallet positions: flag capacity constraint exception
- New exception types: `capacity_breach`, `throughput_exceeded`
- API: `GET /capacity/planned-receipt-calendar`, `/capacity/constraint-violations`
- Panel: Receipt calendar with capacity overlay, constraint alerts

### Phase 3: Labor Planning Integration (1 sprint)
- `sql/100_create_labor_plan.sql` — `fact_labor_plan`, `fact_labor_actual`
- Receipt volume → labor hours required (units/hr by activity type)
- Alert when planned receipts exceed labor capacity
- API: `GET /capacity/labor-forecast`, `/labor-gap`
- Panel: Labor requirement forecast vs plan, overtime projections

### Phase 4: Dock Scheduling (1 sprint)
- `sql/101_create_dock_schedule.sql` — `fact_dock_appointments`
- Appointment slot management for inbound receipts
- Smooth receipt flow: spread PO receipts across available slots
- API: `GET /capacity/dock-schedule/{location}`, `POST /dock-schedule/appointments`
- Panel: Dock door Gantt chart by location

## Dependencies
- Requires WMS integration for utilization data (Manhattan, Blue Yonder, SAP EWM)
- `dim_location` exists — extend with capacity attributes or create `dim_warehouse`
- Replenishment plan must become capacity-aware
- `fact_planned_orders` needs `capacity_feasible` flag

## Business Value
- Prevents warehouse overflow and receiving delays
- Labor planning accuracy improves 20-30% with receipt-driven forecasting
- Dock scheduling reduces driver wait times and detention charges
- Capacity constraints become visible in S&OP process
