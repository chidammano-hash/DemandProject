# Story 03 — Transportation & Logistics

## Problem
The system models transfer lanes (cost, lead time) for rebalancing but has no shipment tracking, carrier management, freight cost optimization, or inbound/outbound logistics visibility. Cannot answer "Where is my shipment?" or "What's my true landed cost?"

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `shipments.csv` | CSV | shipment | shipment_id, po_number, order_id, carrier_id, origin_loc, dest_loc, ship_date, estimated_arrival, actual_arrival, status (in_transit|delivered|delayed|exception), tracking_number, freight_cost, weight_kg, volume_cbm | TMS/Carrier |
| **F2** | `carrier_master.csv` | CSV | carrier | carrier_id, carrier_name, carrier_type (ltl|ftl|parcel|ocean|air|rail), service_level, transit_time_days, on_time_pct, cost_per_kg, cost_per_cbm, min_charge, fuel_surcharge_pct, is_active | TMS |
| **F3** | `freight_rates.csv` | CSV | lane × carrier × mode | origin_loc, dest_loc, carrier_id, transport_mode, rate_per_unit, rate_uom, min_charge, effective_from, effective_to, contract_id | TMS/Procurement |
| **F4** | `warehouse_capacity.csv` | CSV | location × period | location_id, period_start, max_pallet_positions, current_utilization_pct, max_throughput_units_per_day, dock_doors, receiving_capacity_per_day, shipping_capacity_per_day | WMS |
| **F5** | `inbound_asn.csv` | CSV | ASN line | asn_id, po_number, item_no, loc, shipped_qty, ship_date, eta, carrier_id, container_id, customs_status | EDI/Supplier Portal |

## Incremental Implementation

### Phase 1: Shipment Tracking (1 sprint)
- `sql/080_create_shipments.sql` — `fact_shipments`, `dim_carrier`
- `scripts/load_shipments.py` — daily feed from TMS
- `api/routers/shipments.py` — `GET /shipments/in-transit`, `/delayed`, `/by-po/{po_number}`, `/carrier-performance`
- `frontend/src/tabs/inv-planning/ShipmentTrackerPanel.tsx` — in-transit map, delay alerts, ETA timeline

### Phase 2: Freight Cost Analytics (1 sprint)
- `sql/081_create_freight.sql` — `dim_freight_rate`, `mv_freight_cost_summary`
- Landed cost = unit cost + freight per unit + customs duty
- `api/routers/freight.py` — `GET /freight/cost-by-lane`, `/carrier-comparison`, `/budget-vs-actual`
- Panel: Freight spend dashboard, carrier comparison, cost-per-unit trends

### Phase 3: Warehouse Capacity (1 sprint)
- `sql/082_create_warehouse_capacity.sql` — `fact_warehouse_capacity`, `mv_capacity_utilization`
- Alert when projected receipts + current inventory > capacity
- Panel: Warehouse utilization heatmap by location, capacity constraint warnings

### Phase 4: Inbound Visibility (1 sprint)
- ASN (Advance Ship Notice) tracking from suppliers
- Match ASN → PO → receipt for full inbound pipeline
- Panel: Inbound waterfall (on order → shipped → in transit → at dock → received)
- Integration with `fact_open_purchase_orders` and `fact_inventory_projection`

## Dependencies
- Requires TMS integration (SAP TM, Oracle TMS, project44, FourKites, or flat-file from 3PL)
- `dim_transfer_lane` exists — extend with carrier assignments
- Inventory projection should incorporate in-transit inventory as "available pipeline stock"

## Business Value
- Real-time inbound/outbound visibility reduces expediting costs
- Freight cost optimization (carrier selection, mode consolidation)
- Warehouse capacity planning prevents receiving bottlenecks
- Landed cost accuracy improves margin analysis
