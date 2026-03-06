# 04 — Inventory Planning Design Specs

This folder contains all design specifications for the Inventory Planning epic in Demand Studio. It covers the full lifecycle of inventory optimization — from statistical profiling and safety stock computation through replenishment automation, exception management, analytics, simulation, and AI-assisted planning.

---

## Files

| File | Summary |
|---|---|
| `feature33.md` | Inventory snapshot ingestion and analytics — ingest daily inventory data into a fact table and surface inventory trends (DOS, turns, LT coverage, demand-supply gap) alongside sales and forecasts. |
| `feature34.md` | Inventory Planning Module Phase 1 — world-class design covering safety stock, replenishment automation, ABC-XYZ classification, what-if simulation, supplier performance, and seasonal buildup planning. |
| `feature37.md` | Inventory Planning Backtesting — bridges forecast accuracy to inventory outcomes by connecting forecast model predictions with stockout and excess inventory events. |
| `IPfeature1.md` | Demand Variability & Statistical Profiling Engine — compute demand CV, MAD, and statistical profiles per DFU to drive downstream safety stock and policy decisions. |
| `IPfeature2.md` | Lead Time Variability Profiling — measure and track supplier lead time mean, standard deviation, and reliability per item-location pair. |
| `IPfeature3.md` | Safety Stock Engine — compute safety stock targets using statistical service-level formulas driven by demand and lead time variability profiles. |
| `IPfeature4.md` | EOQ & Cycle Stock Calculator — implement Wilson EOQ formula with effective EOQ (MOQ + cap), cycle stock metrics, sensitivity curve, and `fact_eoq_targets` table. |
| `IPfeature5.md` | Replenishment Policy Management — define and auto-assign 4 default replenishment policies (ROP, periodic, min-max, JIT) to DFUs by segment via YAML config and API. |
| `IPfeature6.md` | Inventory Health Score Dashboard — 4-component × 25pt scoring model materialized into `mv_inventory_health_score`; Portfolio Health panel with tier KPI cards, donut chart, and heatmap. |
| `IPfeature7.md` | Exception Queue & Replenishment Recommendations — detect 6 exception types (stockout risk, excess, late PO, etc.), generate recommendations, and surface in a priority-ordered Exception Queue panel. |
| `IPfeature8.md` | Fill Rate & Demand Fulfillment Analytics — track line fill rate, order fill rate, and OTIF KPIs at item-location level with monthly trend and root cause attribution. |
| `IPfeature9.md` | Demand Sensing & Short-Horizon Signal Integration — integrate high-frequency (daily/weekly) demand signals to improve near-term replenishment decisions and exception alerting. |
| `IPfeature10.md` | Safety Stock Monte Carlo Simulation — run probabilistic simulations over demand and lead time distributions to compute safety stock at configurable service levels. |
| `IPfeature11.md` | ABC-XYZ Policy Matrix & Portfolio Segmentation — classify inventory into ABC (value) × XYZ (variability) segments and apply differentiated policies and service-level targets per cell. |
| `IPfeature12.md` | Supplier Performance Intelligence — track on-time delivery, lead time reliability, fill rates, and defect rates per supplier and surface actionable supplier scorecards. |
| `IPfeature13.md` | Capital & Space Investment Optimization — compute optimal capital allocation across the portfolio by balancing cycle stock, safety stock, and holding cost against service-level targets. |
| `IPfeature14.md` | Intra-Month Stockout Detection (Daily Granularity) — detect mid-month stockout events from daily snapshot data and attribute them to forecast error or replenishment failure. |
| `IPfeature15.md` | Unified Inventory Control Tower — single-pane executive dashboard aggregating all inventory KPIs, exceptions, health scores, and fulfillment metrics with drill-down capability. |
| `IPAIfeature1.md` | AI Planning Agent — LLM-powered planning agent that interprets natural language queries, surfaces recommendations from the inventory planning data model, and explains exception root causes. |
