# Story 10 — Sustainability & ESG Metrics

## Problem
Supply chain sustainability is increasingly required by regulation (EU CSRD, SEC climate rules) and customer demand. The system has no carbon footprint tracking, waste measurement, circular economy metrics, or ESG reporting capability.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `emission_factors.csv` | CSV | activity_type | activity_type (trucking_per_km|air_per_km|ocean_per_teu|warehouse_per_sqft|manufacturing_per_unit), co2e_kg, scope (1|2|3), source_reference, effective_year | EPA/GHG Protocol |
| **F2** | `supplier_emissions.csv` | CSV | supplier × year | supplier_id, reporting_year, scope1_co2e_tonnes, scope2_co2e_tonnes, scope3_co2e_tonnes, energy_source_mix, renewable_pct, cdp_score | Supplier ESG Reports |
| **F3** | `waste_tracking.csv` | CSV | location × month | location_id, period_month, waste_type (packaging|product_expired|damaged|production_scrap), waste_qty_kg, disposed_method (landfill|recycle|compost|incinerate), disposal_cost | WMS/Sustainability |
| **F4** | `packaging_master.csv` | CSV | item | item_no, primary_packaging_material, secondary_packaging_material, packaging_weight_g, recyclable_pct, post_consumer_recycled_pct | Product Design |
| **F5** | `water_energy_usage.csv` | CSV | location × month | location_id, period_month, electricity_kwh, natural_gas_therms, water_gallons, renewable_energy_kwh | Facilities |

## Incremental Implementation

### Phase 1: Carbon Footprint Estimation (1 sprint)
- `sql/102_create_sustainability.sql` — `dim_emission_factor`, `fact_supply_chain_emissions`
- `scripts/compute_carbon_footprint.py` — estimate Scope 3 emissions from:
  - Inbound transportation (PO origin → DC, using freight data or distance estimates)
  - Outbound distribution (DC → store/customer)
  - Warehousing (energy per sqft × inventory duration)
- API: `GET /sustainability/carbon-footprint`, `/by-category`, `/by-supplier`, `/trend`
- Panel: Carbon dashboard with Scope 1/2/3 breakdown, supplier carbon ranking

### Phase 2: Waste & Expiration Reduction (1 sprint)
- `sql/103_create_waste_tracking.sql` — `fact_waste_log`, `mv_waste_summary`
- Link to lot tracking (Story 07): expired lot → waste event
- Waste reduction metrics: waste rate, diversion rate, cost of waste
- API: `GET /sustainability/waste-summary`, `/waste-by-category`, `/expiration-risk`
- Panel: Waste waterfall, expiration risk calendar, diversion rate trends

### Phase 3: Sustainable Sourcing Scorecard (1 sprint)
- Extend supplier scorecard (Story 04) with ESG component
- Weighted score: delivery 25%, quality 20%, cost 20%, ESG 20%, responsiveness 15%
- CDP score integration, renewable energy percentage
- API: `GET /sustainability/supplier-esg`, `/sourcing-risk-esg`
- Panel: ESG-integrated supplier ranking, sustainable sourcing progress tracker

### Phase 4: ESG Reporting & Targets (1 sprint)
- `sql/104_create_esg_targets.sql` — `dim_esg_targets`, `fact_esg_progress`
- Science-Based Targets alignment (SBTi)
- Automated reporting: quarterly ESG metrics roll-up
- API: `GET /sustainability/targets`, `/progress`, `/report`
- Panel: ESG target vs actual dashboard, regulatory compliance checklist

## Dependencies
- Emission factors from EPA, GHG Protocol, or Ecoinvent
- Supplier ESG data from CDP, EcoVadis, or direct supplier reporting
- Transportation data (Story 03) needed for accurate Scope 3 calculations
- Lot tracking (Story 07) needed for waste-from-expiration metrics

## Business Value
- Regulatory compliance (EU CSRD, SEC climate disclosure rules)
- Customer/retailer requirements (Walmart Project Gigaton, Target sustainability)
- Cost savings from waste reduction and energy efficiency
- Competitive advantage in ESG-conscious market segments
