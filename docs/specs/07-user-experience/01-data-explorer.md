# Data Explorer

> A paginated, filterable data grid for browsing all eight domain tables (items, locations, customers, time, DFUs, sales, forecasts, inventory) plus a unified Item Analysis tab that overlays multi-model forecasts against actuals for any DFU with interactive SHAP explanations.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | ExplorerTab, ItemAnalysisTab |
| **Key Files** | `ExplorerTab.tsx` (orchestrator, ~245 lines composing 11 sub-files in `tabs/explorer/`: `ExplorerHeader.tsx`, `ExplorerTable.tsx`, `ExplorerPagination.tsx`, `ExplorerErrorBanner.tsx`, `DomainFiltersPanel.tsx`, `FieldVisibilityPanel.tsx`, `useExplorerState.ts`, `useExplorerQueries.ts`, `useColumnSuggestions.ts`, `_helpers.ts`, `types.ts`), `ItemAnalysisTab.tsx`, `api/routers/domains.py`, `api/routers/forecasting/analysis.py`, `api/routers/inventory/inventory_main.py` |

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
| Demand | Forecast Chart | ECharts overlay of actuals vs. all forecast models. Click a model line to select it (thicker line, others fade to 30% opacity). |
| Demand | SHAP | Per-DFU signed SHAP (SHapley Additive exPlanations) feature contributions as stacked bar chart. Historical months at 90% opacity, future months at 45%. Falls back to cluster-level summary on 404. |
| Demand | Model KPIs | Per-model accuracy cards: WAPE, bias, accuracy % for the selected DFU. |
| Supply | Inv KPIs | Inventory KPI cards: DOS, WOC, turns, LT coverage with severity color coding. |
| Supply | Position Table | Paginated monthly inventory position: on-hand, on-order, monthly sales, DOS. |
| Supply | Variability | Demand variability profile: CV, MAD, skewness for the selected DFU. |
| Supply | Lead Time | Lead time profile: average LT, LT CV, reliability band. |

### Three Scope Modes

The Item Analysis tab supports three scope modes via a selector:

| Mode | Scope | Use Case |
|---|---|---|
| `item_location` | Single DFU (item + location) | Deep dive into one DFU with all panels |
| `item` | All locations for one item | Compare performance across locations |
| `location` | All items at one location | Assess location-level patterns |

SHAP panel and inventory panels are only available in `item_location` mode.

---

## Data Model

No new tables. The explorer reads from all eight domain tables via the generic `/domains/{domain}/rows` endpoint. Item Analysis reads from:

| Source | Used By |
|---|---|
| `fact_external_forecast_monthly` | Forecast overlay chart |
| `fact_sales_monthly` | Actuals on chart |
| `fact_production_forecast` | Future forecast + CI bands |
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
| ECharts (`echarts-for-react`) | Forecast overlay chart |
| `common/core/domain_specs.py` | Central schema config for all 8 domains |

---

## See Also

- `02-forecasting/05-advanced-backtest.md` -- SHAP feature selection that produces the CSV data shown in the SHAP panel
- `03-inventory-planning/01-inventory-snapshot.md` -- inventory data powering the Supply panels
- `07-user-experience/02-ui-architecture.md` -- component architecture and virtualization strategy
