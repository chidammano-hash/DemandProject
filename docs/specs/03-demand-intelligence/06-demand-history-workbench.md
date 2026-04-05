# Demand History Workbench

> Provides 5 API endpoint groups for customer-level demand analysis, enabling planners to understand demand decomposition, compare hierarchical forecasts, and explore cross-dimensional relationships.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Demand History |
| **Key Files** | `api/routers/inventory/demand_history.py`, `config/demand_history_config.yaml`, `tests/api/test_demand_history.py` |

---

## Problem

Planners need visibility into customer-level demand decomposition, hierarchical forecast comparison (bottom-up vs top-down vs reconciled), and cross-dimensional demand relationships — capabilities that are not available through standard item-level inventory or forecast views.

## Solution

The Demand History Workbench provides 5 API endpoint groups for customer-level demand analysis, enabling planners to understand demand decomposition, compare hierarchical forecasts, and explore cross-dimensional relationships.

## How It Works

### Industry Context

- **SAP IBP** -- Demand Sensing drill-down with customer-level decomposition
- **Kinaxis RapidResponse** -- Multi-level demand waterfall and reconciliation views
- **o9 Solutions** -- Knowledge graph-driven demand decomposition
- **Blue Yonder Luminate** -- Proportional disaggregation and cross-reference matrices

---

## Data Sources

| Table | Role |
|---|---|
| fact_customer_demand_monthly | Primary: customer-level demand history |
| dim_customer | Customer names and attributes for labeling |
| dim_item | Item descriptions for labeling |
| dim_location | Location city/state for labeling |
| agg_inventory_monthly | Latest inventory position |
| backtest_predictions | Hierarchical model predictions (optional) |

---

## API Endpoints

All endpoints prefixed with /demand-history. All SQL uses %s placeholders.

### Feature 1: Reference Panel

GET /demand-history/reference?item_id=X&loc=Y&months=24

Returns: 24-month demand history, top 5 customers with % share, MoM trend, inventory snapshot.

### Feature 2: Proportional Decomposition

GET /demand-history/decomposition?item_id=X&loc=Y&months=24

Returns: Monthly time-series per customer with pct_share. Pareto summary with cumulative_pct.

### Feature 3: Demand Comparison

GET /demand-history/comparison?item_id=X&loc=Y&months=24

Returns: Monthly series with actual_qty, bottom_up_qty, top_down_qty, reconciled_qty. Graceful fallback to nulls if no hierarchical predictions exist.

### Feature 4: Demand Workbench

GET /demand-history/workbench?grain=item&item_id=X&loc=Y&limit=50&offset=0

Returns: Hierarchical drill-down at 3 grain levels (item, item_loc, item_loc_customer). Each series includes total_demand and monthly detail.

### Feature 5: Cross-Reference Matrix

GET /demand-history/matrix?row_dim=item&col_dim=location&metric=demand_qty
GET /demand-history/matrix/drill?item_id=X&loc=Y

Returns: Pivot grid (rows x cols) with cells matrix and label dictionaries. Drill returns monthly history for a single cell.

---

## Configuration

All parameters in config/demand_history_config.yaml:

| Key | Default | Description |
|---|---|---|
| default_months | 24 | Default trailing months for queries |
| max_months | 60 | Maximum allowed trailing months |
| pareto_top_n | 5 | Number of top customers in reference panel |
| matrix_max_rows | 100 | Maximum matrix rows |
| matrix_max_cols | 50 | Maximum matrix columns |
| cache_ttl_seconds | 120 | Cache-Control max-age |
| hierarchical_model_ids | [bolt_hierarchical] | Model IDs for bottom-up predictions |
| top_down_model_ids | [chronos_bolt] | Model IDs for top-down predictions |

---

## Testing

12 tests in tests/api/test_demand_history.py covering all 5 endpoints:
- Reference: full response, empty data, missing params (422)
- Decomposition: series + Pareto with pct calculations, empty data
- Comparison: actuals only (no predictions), with predictions + reconciled
- Workbench: item grain with monthly detail, empty data
- Matrix: 2x2 grid with labels, same dim rejected (422), drill history
