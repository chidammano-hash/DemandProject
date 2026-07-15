# Data Explorer

> A paginated, filterable data grid for browsing all eight domain tables (items, locations, customers, time, DFUs, sales, forecasts, inventory) plus a unified Item Analysis tab that overlays multi-model and governed customer forecasts against actuals for any DFU with interactive SHAP explanations.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | ExplorerTab, ItemAnalysisTab |
| **Key Files** | `ExplorerTab.tsx` (orchestrator, ~245 lines composing 11 sub-files in `tabs/explorer/`), `ItemAnalysisTab.tsx`, `tabs/item-analysis/CustomerBlendUnifiedChartPanel.tsx`, `hooks/useItemForecastOverlays.ts`, `api/routers/domains.py`, `api/routers/forecasting/analysis.py`, `api/routers/forecasting/production_forecast.py` |

---

## Problem

Planners need to browse raw data to validate pipeline outputs, spot anomalies, and answer ad-hoc questions. Without a structured explorer, they export CSVs or write SQL. For item-level analysis, they need to see how multiple forecast models compare against actuals for the same DFU -- and understand which features drive each model's predictions -- without switching between separate forecast and inventory views.

---

## Solution

**Data Explorer:** A generic grid component renders any domain table with type-aware column filtering, server-side pagination, and CSV export. Text columns use GIN trigram indexes (PostgreSQL `pg_trgm` extension) for fast substring search; exact-match queries use B-tree indexes via an `=` prefix syntax.

**Item Analysis:** A unified tab merges the former DFU Analysis and Inventory tabs into a single view with a checkbox toggle toolbar. Seven panels (grouped as Demand and Supply) can be individually shown or hidden. Panel visibility persists in localStorage via the `usePanelToggles` hook.

---

## How It Works

### Data Explorer Features

| Feature | How It Works |
|---|---|
| Type-aware filtering | Text columns: GIN trigram substring search. Numeric/date columns: B-tree range queries. Prefix `=` forces exact B-tree match on text. |
| Typeahead suggestions | Text columns show native HTML datalist populated from `DISTINCT` values (capped at 50 suggestions). |
| Pagination | Server-side offset/limit (50-1000 rows per page). Approximate total count from `pg_class.reltuples` with `100,000+` badge for large results. |
| Column sorting | Click column header to toggle ascending/descending. Server-side ORDER BY. |
| CSV export | Client-side export of current page via papaparse. |
| Virtualized rendering | TanStack Virtual for smooth scrolling on large result sets without DOM bloat. |

### Item Analysis Panels

| Group | Panel | What It Shows |
|---|---|---|
| Demand | Forecast Chart | Recharts overlay of actuals, standard models, production, staged candidates, and exact-lineage customer bottom-up/source-champion/blend series. Click a model line to select it (thicker line, others fade to 30% opacity). |
| Demand | SHAP | Per-DFU signed SHAP (SHapley Additive exPlanations) feature contributions as stacked bar chart. Historical months at 90% opacity, future months at 45%. Falls back to cluster-level summary on 404. |
| Demand | Model KPIs | Per-model accuracy cards: WAPE, bias, accuracy % for the selected DFU. |
| Supply | Inv KPIs | Inventory KPI cards: DOS, WOC, turns, LT coverage with severity color coding. |
| Supply | Position Table | Paginated monthly inventory position: on-hand, on-order, monthly sales, DOS. |
| Supply | Variability | Demand variability profile: CV, MAD, skewness for the selected DFU. |
| Supply | Lead Time | Lead time profile: average LT, LT CV, reliability band. |

### Exact DFU scope

Item Analysis operates at `item_location` scope: one item and one canonical
location. Customer blend reads are disabled until both keys are present, which
prevents a warehouse-item forecast from being presented as a customer, item, or
location-only aggregate.

For customer forecasts, historical lines come from the exact backtest run
stamped on the selected blend manifest. Future `customer_bottom_up` and
`customer_bottom_up_blend` lines use the standard staging endpoint and retain
their distinct shadow/blend run ids. The component overlay fills a future line
only when that display identity is absent from standard staging, so each
model/month is rendered once.

---

## Data Model

The explorer itself adds no tables and reads the eight domain tables through
`/domains/{domain}/rows`. Item Analysis also reads governed forecast evidence:

| Source | Used By |
|---|---|
| `fact_external_forecast_monthly` | Forecast overlay chart |
| `fact_sales_monthly` | Actuals on chart |
| `fact_production_forecast` | Future forecast + CI bands |
| `forecast_generation_run` + `fact_production_forecast_staging` | True display identity and exact-run future staging, including the review-only `customer_bottom_up` shadow and promotable blend |
| `customer_bottom_up_backtest_component` | Exact historical customer bottom-up, source champion, blend, and actual comparison |
| `customer_bottom_up_blend_component` | Exact blend-run component and fallback evidence |
| `agg_inventory_monthly` | Inventory KPIs and position table |
| Backtest SHAP CSVs | SHAP panel (cluster-level fallback) |
| Backtest pkl models | Per-DFU SHAP (on-demand computation) |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/domains/{domain}/rows` | Paginated, filtered data for any domain |
| GET | `/domains/{domain}/search` | Full-text trigram search across configured columns |
| GET | `/domains/{domain}/columns` | Column metadata (name, type, filterable) |
| GET | `/sku/analysis` | Multi-model forecast + actuals for a DFU |
| GET | `/forecast/production/staging` | Future staged models grouped by true display identity and source run |
| GET | `/customer-forecast/blend/latest` | Resolve the current viewable blend manifest |
| GET | `/customer-forecast/blend/series` | Exact item-location forward blend components and fallback status |
| GET | `/customer-forecast/blend/trend` | Exact historical backtest and paired future staging lineage |
| GET | `/forecast/shap/{model_id}/sku` | Per-DFU signed SHAP values (on-demand computation) |
| GET | `/inventory/position` | Monthly inventory position for an item-location |
| GET | `/inventory/kpis` | Point-in-time inventory KPIs |

The `domains.py` router is mounted last in `main.py` because its `{domain}` path parameter is a catch-all.

---

## Dependencies

| Dependency | Reason |
|---|---|
| `pg_trgm` PostgreSQL extension | GIN trigram indexes for substring search |
| TanStack Table + TanStack Virtual | Virtualized data grid rendering |
| papaparse | Client-side CSV export |
| Recharts | Item Analysis unified forecast and customer comparison chart |
| `common/core/domain_specs.py` | Central schema config for all 8 domains |

---

## See Also

- `02-forecasting/05-advanced-backtest.md` -- SHAP feature selection that produces the CSV data shown in the SHAP panel
- `03-inventory-planning/01-inventory-snapshot.md` -- inventory data powering the Supply panels
- `07-user-experience/02-ui-architecture.md` -- component architecture and virtualization strategy
