# Story 07 — Regulatory Compliance & Lot Traceability

## Problem
The system handles a beverage/spirits supply chain (item_proof, alcoh_pct, bot_type_desc columns exist) but has no lot/batch tracking, expiration date management, regulatory compliance, or product recall capability. For regulated products (alcohol, food, pharma), this is a critical gap.

## Missing Input Files

| # | File | Format | Grain | Key Columns | Source |
|---|------|--------|-------|-------------|--------|
| **F1** | `lot_master.csv` | CSV | lot | lot_id, item_no, lot_qty, production_date, expiration_date, shelf_life_days, supplier_lot_ref, status (available|hold|quarantine|expired|recalled) | WMS/ERP |
| **F2** | `lot_inventory.csv` | CSV | lot × location | lot_id, loc, qty_on_hand, receipt_date, fifo_rank, days_to_expiry | WMS |
| **F3** | `regulatory_requirements.csv` | CSV | item × regulation | item_no, regulation_code (ttb|fda|state_abc), license_requirement, labeling_requirement, tax_class, excise_rate, reporting_frequency | Compliance |
| **F4** | `customs_tariff.csv` | CSV | item × country | item_no, origin_country, dest_country, hs_code, tariff_rate_pct, quota_qty, anti_dumping_duty_pct, preferential_rate_pct, effective_from | Trade Compliance |
| **F5** | `quality_hold_log.csv` | CSV | hold_event | hold_id, lot_id, item_no, loc, hold_date, hold_reason, hold_qty, release_date, disposition, inspector_id | QA |

## Incremental Implementation

### Phase 1: Lot Tracking (1 sprint)
- `sql/092_create_lot_tracking.sql` — `dim_lot`, `fact_lot_inventory`
- `scripts/load_lot_data.py` — daily WMS feed
- FEFO (First Expired, First Out) allocation logic
- API: `GET /lots/expiring-soon`, `/lots/by-item/{item_no}`, `/lots/aging-report`
- Panel: Expiration calendar, aging buckets (0-30, 31-60, 61-90, 90+ days)

### Phase 2: Recall & Traceability (1 sprint)
- `sql/093_create_traceability.sql` — `fact_lot_movement` (lot genealogy: receipt → storage → pick → ship)
- Forward trace: lot → all shipments/customers impacted
- Backward trace: customer complaint → lot → supplier → production batch
- API: `GET /traceability/forward/{lot_id}`, `/backward/{lot_id}`, `/recall-impact`
- Panel: Traceability tree visualization, recall impact assessment

### Phase 3: Regulatory Compliance (1 sprint)
- `sql/094_create_compliance.sql` — `dim_regulation`, `fact_compliance_status`
- License verification per location per item category
- Excise tax calculation integrated into financial planning
- API: `GET /compliance/status`, `/licenses/expiring`, `/tax-liability`
- Panel: Compliance dashboard with red/green status per location

### Phase 4: Trade Compliance & Tariffs (1 sprint)
- `sql/095_create_tariffs.sql` — `dim_tariff_schedule`, `fact_landed_cost_detail`
- Landed cost = unit cost + freight + duty + tariff + insurance
- Impact analysis: tariff changes → cost impact → margin impact
- API: `GET /trade/landed-cost/{item_no}`, `/tariff-impact`, `/origin-analysis`
- Panel: Landed cost breakdown, tariff impact simulator

## Dependencies
- Requires WMS integration for lot-level data (SAP EWM, Manhattan, Blue Yonder WMS)
- Regulatory data is often manual or from compliance databases
- Tariff data from customs brokers or trade compliance platforms (Amber Road, Thomson Reuters)
- Current `fact_inventory_snapshot` is lot-agnostic — lot tracking adds a new grain level

## Business Value
- Regulatory compliance (TTB, FDA, state ABC) — mandatory for alcohol
- Expiration-driven FEFO allocation reduces waste 15-25%
- Recall readiness: trace impacted lots in minutes, not days
- Tariff optimization can save 3-8% on international procurement
