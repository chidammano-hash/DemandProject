# Things To Do — E2E Supply Chain Gap Analysis

Incremental stories to evolve the Supply Chain Command Center from a demand planning + inventory optimization platform into a full end-to-end supply chain planning and execution system.

## Current State

**What we have (strong):**
- 8 data domains: item, location, customer, time, DFU, sales, forecast, inventory
- 86 SQL DDL files, 18+ materialized views, 58 router files
- Demand forecasting (3 ML models, champion selection, SHAP, production inference)
- Inventory optimization (34 panels: SS, EOQ, policies, rebalancing, multi-echelon)
- S&OP cycle management (6-stage workflow)
- Procurement basics (open POs, planned orders, approval workflow)
- AI planning agent (Claude-powered exception scanning)

**What's missing for true E2E:**

## Story Index

| # | Story | Missing Input Files | Priority | Sprints |
|---|-------|-------------------|----------|---------|
| **01** | [BOM & Production Planning](story-01-bom-production.md) | bom_master, work_centers, routing_master, production_orders, production_calendar | High | 4 |
| **02** | [Customer Orders & Order Management](story-02-customer-orders.md) | customer_orders, customer_sla, allocation_rules, returns_rma | High | 4 |
| **03** | [Transportation & Logistics](story-03-transportation-logistics.md) | shipments, carrier_master, freight_rates, warehouse_capacity, inbound_asn | Medium | 4 |
| **04** | [Supplier Risk & Contracts](story-04-supplier-risk-contracts.md) | supplier_contracts, supplier_financial, supplier_quality, risk_events, sourcing_strategy | High | 4 |
| **05** | [Demand Sensing & External Signals](story-05-demand-sensing-signals.md) | pos_daily_sales, weather_actuals, economic_indicators, promotional_calendar, social_sentiment, competitor_pricing | Medium | 4 |
| **06** | [Financial Integration & Margin](story-06-financial-margin.md) | item_pricing, item_cost_detail, customer_pricing, markdown_history, budget_plan | High | 4 |
| **07** | [Regulatory Compliance & Traceability](story-07-compliance-traceability.md) | lot_master, lot_inventory, regulatory_requirements, customs_tariff, quality_hold_log | Medium | 4 |
| **08** | [Master Data Sync & Governance](story-08-master-data-sync.md) | product_hierarchy, location_hierarchy, customer_hierarchy, item_lifecycle, uom_conversion, changelog | High | 4 |
| **09** | [Warehouse & Distribution Capacity](story-09-capacity-constraints.md) | warehouse_master, warehouse_utilization, labor_plan, appointment_schedule | Medium | 4 |
| **10** | [Sustainability & ESG Metrics](story-10-sustainability-metrics.md) | emission_factors, supplier_emissions, waste_tracking, packaging_master, water_energy_usage | Low | 4 |

## Recommended Implementation Order

**Wave 1 (Foundation):** Stories 08, 06, 02 — Master data hierarchies, financial enrichment, customer orders
**Wave 2 (Supply Side):** Stories 01, 04 — BOM/production, supplier risk & contracts
**Wave 3 (Visibility):** Stories 03, 05 — Transportation tracking, demand sensing
**Wave 4 (Compliance):** Stories 07, 09 — Lot traceability, capacity constraints
**Wave 5 (ESG):** Story 10 — Sustainability metrics

## Summary: 50 Missing Input Files

Total new CSV/data feeds needed: **50 files** across 10 stories
Total new SQL tables: ~60 dimension + fact tables
Total new API endpoints: ~80 endpoints across 10 new router modules
Total estimated effort: ~40 sprints (phased over 10 quarters)

## Existing Backlog

- [Medallion Pipeline Refactoring](medallion-refactoring.md) — bug fixes + code dedup in medallion layer
