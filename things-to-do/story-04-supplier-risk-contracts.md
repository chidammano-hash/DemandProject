# Story 04 — Supplier Risk Scoring & Contract Management

## Problem
The system tracks supplier lead time and on-time delivery (`mv_supplier_performance`, `dim_supplier`) but has no financial risk scoring, contract terms management, dual-sourcing strategy, or geopolitical risk assessment. A critical supplier failure would be invisible until deliveries stop.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `supplier_contracts.csv` | CSV | contract_line | contract_id, supplier_id, item_no, contract_type (fixed_price|index_based|blanket), price_per_unit, volume_commitment_qty, min_order_qty, payment_terms_days, penalty_clause, effective_from, effective_to, auto_renew | Procurement |
| **F2** | `supplier_financial.csv` | CSV | supplier × quarter | supplier_id, assessment_date, revenue_usd, credit_rating, duns_score, days_payable_outstanding, bankruptcy_risk_score, last_audit_date | D&B/Finance |
| **F3** | `supplier_quality.csv` | CSV | supplier × item × month | supplier_id, item_no, inspection_date, lot_qty, accepted_qty, rejected_qty, defect_type, defect_rate_pct, corrective_action_id | QA/Inspection |
| **F4** | `supplier_risk_events.csv` | CSV | event | event_id, supplier_id, event_date, event_type (natural_disaster|labor_strike|bankruptcy|sanctions|port_closure), severity, impact_description, resolution_date, estimated_recovery_weeks | Risk Intelligence |
| **F5** | `sourcing_strategy.csv` | CSV | item × supplier | item_no, supplier_id, sourcing_role (primary|secondary|emergency), allocation_pct, qualification_status, last_qualified_date, switching_cost | Procurement |

## Incremental Implementation

### Phase 1: Supplier Scorecards (1 sprint)
- `sql/083_create_supplier_scorecard.sql` — `fact_supplier_scorecard` (delivery, quality, cost, responsiveness composite)
- Weighted scoring: on-time 30%, quality 25%, cost 25%, responsiveness 20%
- `api/routers/supplier_scorecard.py` — `GET /suppliers/scorecard`, `/ranking`, `/trends`
- Panel: Supplier ranking table with drill-down, radar chart per supplier

### Phase 2: Contract Management (1 sprint)
- `sql/084_create_contracts.sql` — `dim_supplier_contract`, `fact_contract_utilization`
- Track volume commitment vs actuals, expiring contracts alert
- API: `GET /contracts/active`, `/expiring-soon`, `/utilization`, `/price-comparison`
- Panel: Contract compliance dashboard, renewal alerts, price benchmarking

### Phase 3: Risk Scoring (1 sprint)
- `sql/085_create_supplier_risk.sql` — `fact_supplier_risk_score`, `fact_risk_events`
- Composite risk = financial (30%) + quality (25%) + delivery (25%) + geopolitical (20%)
- `scripts/compute_supplier_risk.py` — quarterly recalculation
- API: `GET /suppliers/risk-map`, `/risk-alerts`, `/mitigation-plans`
- Panel: Supplier risk heatmap (x=spend, y=risk), early warning alerts

### Phase 4: Dual-Sourcing Strategy (1 sprint)
- `fact_sourcing_allocation` — planned vs actual split across primary/secondary suppliers
- Auto-recommend secondary sourcing when primary risk exceeds threshold
- Integration with planned orders: split POs across qualified suppliers
- Panel: Sourcing mix visualization, qualification status tracker

## Dependencies
- Requires procurement data (SAP MM, Coupa, Ariba, or manual entry)
- `dim_supplier` already exists — extend with `risk_tier`, `strategic_importance`
- Financial risk data from Dun & Bradstreet, Moody's, or manual assessment
- Geopolitical risk feeds from external APIs (optional)

## Business Value
- Early warning on supplier failures prevents stockouts
- Contract compliance tracking avoids penalty charges
- Dual-sourcing reduces single-point-of-failure risk
- Quality tracking drives supplier development programs
