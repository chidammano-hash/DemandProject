# Story 02 — Customer Orders & Order Management

## Problem
The system uses monthly aggregated sales history (`fact_sales_monthly`) but has no visibility into individual customer orders, backorders, or order fulfillment status. Cannot track order-level service or allocate during shortages.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `customer_orders.csv` | CSV | order_line | order_id, line_no, customer_no, item_no, loc, order_date, requested_date, promised_date, shipped_date, order_qty, allocated_qty, shipped_qty, backorder_qty, unit_price, status, priority | OMS/ERP |
| **F2** | `customer_sla.csv` | CSV | customer × item_group | customer_no, item_category, target_fill_rate_pct, target_otif_pct, max_lead_time_days, penalty_per_miss, effective_from | CRM/Contract |
| **F3** | `allocation_rules.csv` | CSV | rule | rule_id, priority_rank, customer_tier, item_category, allocation_pct, min_allocation_qty, fairness_method (proportional|priority|round_robin) | Planning |
| **F4** | `returns_rma.csv` | CSV | return_line | rma_id, order_id, line_no, item_no, loc, return_qty, return_reason, disposition (restock|scrap|rework), credit_amount, received_date, status | OMS/WMS |

## Incremental Implementation

### Phase 1: Order Visibility (1 sprint)
- `sql/077_create_customer_orders.sql` — `fact_customer_orders`, `fact_order_lines`
- `scripts/load_customer_orders.py` — daily CSV import with dedup
- `api/routers/orders.py` — `GET /orders/open`, `/orders/backlog`, `/orders/{id}`, `/orders/aging`
- `frontend/src/tabs/inv-planning/OrderBacklogPanel.tsx` — open orders, backorder aging, priority queue

### Phase 2: Service Level by Customer (1 sprint)
- `sql/078_create_customer_sla.sql` — `dim_customer_sla`, `mv_customer_otif`
- OTIF (On-Time In-Full) calculation from order lines vs promised dates
- `api/routers/customer_service.py` — `GET /customer-service/otif`, `/by-customer`, `/by-tier`
- Panel: Customer service level dashboard with SLA breach alerts

### Phase 3: Allocation Engine (1 sprint)
- `scripts/run_allocation.py` — shortage allocation using fairness rules
- `fact_allocation_plan` — proposed allocation per customer per item
- API: `POST /allocation/run`, `GET /allocation/plan`, `PUT /allocation/override`
- Panel: Allocation preview with what-if scenarios

### Phase 4: Returns & Reverse Logistics (1 sprint)
- `sql/079_create_returns.sql` — `fact_returns`, `mv_return_rate_by_item`
- Load RMA data, disposition tracking
- Net demand adjustment: gross demand - expected returns = net demand
- Panel: Return rate trends, disposition breakdown, restocking impact on inventory

## Dependencies
- Requires order-level data from OMS (Salesforce, SAP SD, Oracle OM)
- `dim_customer` already exists — extend with `customer_tier`, `credit_limit`
- Fill rate calculation should shift from proxy (sales/forecast) to actual (shipped/ordered)

## Business Value
- True OTIF measurement (not proxy-based fill rate)
- Backorder visibility drives expediting decisions
- Fair allocation during shortages protects strategic accounts
- Return rate feeds back into demand planning (net demand)
