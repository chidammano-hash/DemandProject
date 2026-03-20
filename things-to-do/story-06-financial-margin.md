# Story 06 — Financial Integration & Margin Analytics

## Problem
The system tracks inventory value and carrying cost but lacks gross margin modeling, customer profitability, markdown optimization, and P&L impact analysis. Planning decisions are made on units, not dollars. S&OP gap analysis cannot quantify revenue/margin impact.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `item_pricing.csv` | CSV | item × price_list | item_no, price_list_id, list_price, net_price, currency, effective_from, effective_to | ERP/Pricing |
| **F2** | `item_cost_detail.csv` | CSV | item × cost_element | item_no, cost_element (material|labor|overhead|freight|duty), amount, currency, cost_type (standard|actual), period_month | ERP/Cost Accounting |
| **F3** | `customer_pricing.csv` | CSV | customer × item | customer_no, item_no, negotiated_price, discount_pct, rebate_pct, payment_terms, effective_from | CRM/ERP |
| **F4** | `markdown_history.csv` | CSV | item × location × event | item_no, loc, markdown_date, original_price, markdown_price, markdown_pct, qty_sold_at_markdown, reason (aging|seasonal|damage|clearance) | POS/Merch |
| **F5** | `budget_plan.csv` | CSV | category × month | category, loc, budget_month, revenue_target, cogs_target, gross_margin_target, inventory_budget, opex_budget | Finance |

## Incremental Implementation

### Phase 1: Margin-Enriched Inventory (1 sprint)
- Extend `dim_item_cost` with detailed cost elements (material, labor, overhead, freight, duty)
- `sql/089_create_margin_analytics.sql` — `fact_item_margin` (item × month: revenue, COGS, gross_margin, margin_pct)
- Enrich inventory health score with margin impact (high-margin stockout = critical)
- API: `GET /finance/margin-by-item`, `/margin-by-category`, `/margin-at-risk`

### Phase 2: Customer Profitability (1 sprint)
- `sql/090_create_customer_profitability.sql` — `mv_customer_profitability`
- Revenue - COGS - rebates - freight - service cost = customer contribution
- API: `GET /finance/customer-profitability`, `/customer-ranking`, `/unprofitable-accounts`
- Panel: Customer profitability matrix (revenue vs margin), whale chart

### Phase 3: P&L Impact on Planning Decisions (1 sprint)
- Every exception, planned order, and rebalancing transfer gets a margin impact field
- S&OP gap analysis shows revenue/margin impact alongside unit gaps
- `scripts/enrich_financial_impact.py` — batch enrichment of all planning objects
- Extend `fact_sop_gaps` with `revenue_impact`, `margin_impact`
- Panel: Financial waterfall in S&OP tab showing gap → margin impact cascade

### Phase 4: Markdown Optimization (1 sprint)
- `sql/091_create_markdown.sql` — `fact_markdown_events`, `fact_markdown_recommendation`
- `scripts/recommend_markdowns.py` — aging-based markdown curve optimization
- Objective: maximize recovery value from excess/obsolete inventory
- API: `GET /finance/markdown-candidates`, `/markdown-simulation`
- Panel: Markdown recommendation queue with expected recovery value

## Dependencies
- Requires cost accounting data from ERP (SAP CO, Oracle Cost Management)
- Customer pricing from CRM or trade management system
- `dim_item_cost` already exists — extend with cost breakdown
- `fact_budget_periods` already exists — extend with P&L budget lines

## Business Value
- Planning decisions driven by margin, not just units
- Customer profitability visibility drives allocation priorities
- Markdown optimization recovers 10-20% more from excess inventory
- S&OP conversations grounded in financial impact
