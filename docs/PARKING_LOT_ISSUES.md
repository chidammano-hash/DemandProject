# Parking Lot Issues

> 135 improvements ranked by a panel of 3 expert judges: (A) Simplicity & Inventory Planning, (B) Backtesting & Validation, (C) UI/UX. Score = A + B + C (max 30). Rank #1 = highest priority.

---

## Critical Priority (Score 22-24)

| # | ID | Improvement | A | B | C | Score |
|---|-----|------------|---|---|---|-------|
| 1 | 2 | ABC-XYZ matrix service levels — 9-cell differentiated SL instead of ABC-only | 9 | 8 | 7 | 24 |
| 2 | 114 | Reduce to 5-7 daily essential panels — focused daily workflow instead of 33 panels | 9 | 5 | 10 | 24 |
| 3 | 106 | Business outcome KPIs — replace Z-score/coverage ratio with buffer days/carrying cost/service level | 9 | 5 | 9 | 23 |
| 4 | 115 | Consolidate redundant demand panels — merge Demand Forecast + Demand Plan + Blended into one | 9 | 5 | 9 | 23 |
| 5 | 10 | Integrated planning targets — combine SS + EOQ + ROP into cost-benefit-optimized targets | 8 | 8 | 7 | 23 |
| 6 | 45 | Multi-factor stockout risk score (0-100) — combines depletion rate, LT coverage, demand volatility | 8 | 7 | 8 | 23 |
| 7 | 67 | Quantified financial impact per exception — loss-of-sales, carrying cost, opportunity cost in dollars | 8 | 7 | 8 | 23 |
| 8 | 8 | Explainability UI — waterfall decomposition, formula rendering with values, sensitivity sliders | 7 | 6 | 10 | 23 |
| 9 | 7 | Intelligent guard rails — ABC-specific bounds, zero-demand floors, outlier detection via MAD | 9 | 7 | 6 | 22 |
| 10 | 3 | Seasonal SS adaptation — monthly demand profiles to adjust SS by peak/trough season | 8 | 9 | 5 | 22 |
| 11 | 52 | Dynamic ABC-class service level policies — variability + seasonality + regional differentiation | 8 | 8 | 6 | 22 |
| 12 | 101 | "Why This Matters" context on every KPI — business impact interpretation, risk language | 8 | 4 | 10 | 22 |
| 13 | 102 | Unified Action Inbox — rank ALL critical actions across panels by financial impact | 8 | 5 | 9 | 22 |
| 14 | 111 | "Today's Plan" landing dashboard — priority ribbon with exception/order/stockout counts | 8 | 5 | 9 | 22 |
| 15 | 53 | Gap waterfall decomposition — quantified root cause: SS shortfall, LT delay, demand spike, forecast error | 7 | 7 | 8 | 22 |
| 16 | 55 | Target vs actual waterfall bridge — executive view of positive/negative deltas by ABC/region | 7 | 7 | 8 | 22 |
| 17 | 104 | Change deltas and trend sparklines — day-over-day arrows, 7-day sparklines on every KPI | 7 | 6 | 9 | 22 |
| 18 | 107 | Inline drill-down expansion — click row to expand item context, chart, root cause, actions in-place | 7 | 6 | 9 | 22 |
| 19 | 108 | Data quality banners — freshness indicator, stale data warnings, forecast accuracy caveats per panel | 7 | 7 | 8 | 22 |
| 20 | 48 | What-if scenario builder — parameterized overrides (LT change, extra PO, demand shock) with delta comparison | 6 | 8 | 8 | 22 |
| 21 | 109 | What-if scenario buttons — "+20% SS", "LT +15 days" quick simulations with cost/impact preview | 6 | 8 | 8 | 22 |

## High Priority (Score 20-21)

| # | ID | Improvement | A | B | C | Score |
|---|-----|------------|---|---|---|-------|
| 22 | 46 | Economic excess detection — holding cost integration, shelf-life awareness, target inventory turns | 8 | 7 | 6 | 21 |
| 23 | 110 | Daily Planner Summary Report — auto-generated briefing: urgent actions, weekly review, portfolio health | 8 | 5 | 8 | 21 |
| 24 | 103 | Recommended Next Step cards — explicit action recommendations per panel with click-through links | 7 | 5 | 9 | 21 |
| 25 | 112 | Step-by-step guided workflow mode — progress bar: Triage → Projections → Orders → Monitor | 7 | 5 | 9 | 21 |
| 26 | 43 | 95%/80% confidence band envelopes — stochastic demand paths, percentile curves | 6 | 8 | 7 | 21 |
| 27 | 12 | Variable horizon by ABC class — A=6mo weekly, B=12mo monthly, C=18mo, seasonal=24mo | 8 | 7 | 5 | 20 |
| 28 | 63 | Smart deduplication with escalation — update severity if worsening, auto-resolve if improving | 8 | 5 | 7 | 20 |
| 29 | 118 | Planner-friendly group names — Morning Triage, Order Planning, Performance Insights | 8 | 3 | 9 | 20 |
| 30 | 130 | One-click data export (CSV/PDF) — export current table + KPIs with filters and timestamp | 8 | 4 | 8 | 20 |
| 31 | 13 | Forecast quality weighting — model confidence factor from CI width; conservative penalty for bad forecasts | 7 | 9 | 4 | 20 |
| 32 | 18 | Seasonal demand modulation in forward plans — seasonal_factor × SS, stricter SL during peak | 7 | 8 | 5 | 20 |
| 33 | 66 | Dynamic priority ranking — time decay, dependency boost, volume pressure, LT urgency | 7 | 6 | 7 | 20 |
| 34 | 69 | Ranked remediation suggestions — multiple actions per exception type with confidence and trade-offs | 7 | 5 | 8 | 20 |
| 35 | 17 | Dual-scenario SS (optimistic/conservative ROP) — urgent vs normal order urgency levels | 6 | 8 | 6 | 20 |
| 36 | 54 | 3-month fill rate forecast — exponential smoothing with leading indicators | 6 | 8 | 6 | 20 |
| 37 | 105 | Visible workflow connections — breadcrumb path: Action → Exception → Order → Projection | 6 | 5 | 9 | 20 |
| 38 | 113 | Contextual cross-panel navigation — "View Projection" on exceptions, "Create Order" on projections | 6 | 5 | 9 | 20 |

## Medium-High Priority (Score 19)

| # | ID | Improvement | A | B | C | Score |
|---|-----|------------|---|---|---|-------|
| 39 | 15 | Data-driven review cycle tuning — shorten for high-CV, lengthen for stable; seasonal adjustment | 7 | 8 | 4 | 19 |
| 40 | 33 | Price-break MOQ optimization — evaluate all supplier price tiers, select minimum total landed cost | 7 | 7 | 5 | 19 |
| 41 | 37 | Budget-constrained ordering — monthly budget cap, cost-per-day-of-stockout-prevention ranking | 7 | 6 | 6 | 19 |
| 42 | 51 | Weighted fill rate — order-size weighting, partial shipment penalties, lead time impact | 7 | 7 | 5 | 19 |
| 43 | 82 | Dynamic reclassification triggers — auto-recompute on >15% CV change, 90-day max age | 7 | 7 | 5 | 19 |
| 44 | 83 | Lifecycle stage detection — introduction/growth/mature/decline/obsolete with policy overrides | 7 | 7 | 5 | 19 |
| 45 | 85 | Sunset/EOL detection — declining trend, zero-velocity flagging, disposal-oriented policy | 7 | 7 | 5 | 19 |
| 46 | 88 | Policy-segment alignment validation — reconciliation checks, auto-heal, compliance reporting | 7 | 7 | 5 | 19 |
| 47 | 119 | Role-based default landing — auto-open most-used panel per role on login | 7 | 4 | 8 | 19 |
| 48 | 129 | Bulk action bar — multi-select rows, sticky bar with Approve/Archive/Export/Alert | 7 | 4 | 8 | 19 |
| 49 | 11 | Dynamic policy fitness scoring — continuous 0-1 score per DFU, reassign when fitness < 0.7 | 6 | 8 | 5 | 19 |
| 50 | 60 | Intramonth real-time SL tracking — daily cumulative fill rate, trend-to-target pace alerts | 6 | 6 | 7 | 19 |

## Medium Priority (Score 17-18)

| # | ID | Improvement | A | B | C | Score |
|---|-----|------------|---|---|---|-------|
| 51 | 78 | PO delay root-cause attribution — separate forecast miss from planning/supplier/logistics delay | 6 | 7 | 6 | 19 |
| 52 | 87 | Segment migration tracking — audit trail of changes, drift score, thrashing detection | 6 | 7 | 6 | 19 |
| 53 | 126 | KPI executive dashboard with sparklines — micro-charts and color badges with click-to-jump | 6 | 5 | 8 | 19 |
| 54 | 14 | Adaptive order quantity — net requirement considering EOQ, demand clustering, supplier MOQ | 7 | 7 | 4 | 18 |
| 55 | 32 | Dynamic ordering/holding costs — supplier-specific and location-specific cost multipliers | 7 | 7 | 4 | 18 |
| 56 | 35 | Lead time variance in order timing — percentile-based triggers, dynamic safety buffers | 7 | 7 | 4 | 18 |
| 57 | 84 | New product bootstrap policies — synthetic CV from category, default high-service, auto-graduation | 7 | 6 | 5 | 18 |
| 58 | 92 | Data freshness validation gates — SLA per input table, warning/error if stale | 7 | 7 | 4 | 18 |
| 59 | 27 | Forecast lineage tracking — trace each adjustment stage (raw → CI → reconciliation → override) | 6 | 7 | 5 | 18 |
| 60 | 38 | Order approval workflow with SLAs — 4-state machine, escalation rules, audit trail | 6 | 5 | 7 | 18 |
| 61 | 64 | Intelligent alert routing — planner assignment rules, workload balancing, snooze capability | 6 | 5 | 7 | 18 |
| 62 | 65 | SLA-driven resolution workflow — 6-state machine, mandatory transition fields, time tracking | 6 | 5 | 7 | 18 |
| 63 | 116 | Notification badges on group pills — "Daily Ops (3)", "Supply (2)" pending action counts | 6 | 4 | 8 | 18 |
| 64 | 117 | Quick Actions floating menu — persistent FAB: + Create PO, + Approve Order, + Acknowledge | 6 | 4 | 8 | 18 |
| 65 | 124 | Drill-down breadcrumb trail — click history sidebar, "Back" button, "Save this view" | 6 | 4 | 8 | 18 |
| 66 | 125 | Save workflow presets — named filter combinations stored for quick access | 6 | 4 | 8 | 18 |
| 67 | 133 | Interactive onboarding tour — guided walkthrough on first visit with contextual help buttons | 6 | 3 | 9 | 18 |
| 68 | 4 | Forecast CI calibration validation — verify CI coverage matches claimed percentiles | 5 | 9 | 4 | 18 |
| 69 | 19 | 3-tier confidence scoring — separate forecast/LT/data confidence; geometric mean drives routing | 5 | 7 | 6 | 18 |
| 70 | 24 | Empirical CI quantiles — pool top-5 models, compute quantiles from residuals not Z-scores | 5 | 9 | 4 | 18 |
| 71 | 79 | Interactive supplier KPI dashboard — monthly trends, peer ranking, item drill-down, forecast | 5 | 5 | 8 | 18 |
| 72 | 93 | Incremental refresh mode — detect changed DFUs only, bitemporal tracking | 7 | 6 | 4 | 17 |
| 73 | 34 | Multi-item order consolidation — group by (supplier, ±2 day window), compute consolidation ROI | 6 | 6 | 5 | 17 |
| 74 | 70 | Planner workload balancing — capacity model, skill-based routing, cognitive load distribution | 6 | 4 | 7 | 17 |
| 75 | 90 | Segment-driven automation rules — AX=auto-PO, CZ=manual, NEW=high-touch, SUNSET=liquidation | 6 | 5 | 6 | 17 |

## Lower Priority (Score 15-17)

| # | ID | Improvement | A | B | C | Score |
|---|-----|------------|---|---|---|-------|
| 76 | 121 | Breadcrumb navigation — persistent breadcrumb above group pills | 6 | 3 | 8 | 17 |
| 77 | 23 | Multi-horizon champion selection — separate winners for h=1-6, h=7-12, h=13-24 | 5 | 9 | 3 | 17 |
| 78 | 40 | Continuous cost-benefit feedback loop — actual vs planned cost tracking, auto-adjust config | 5 | 8 | 4 | 17 |
| 79 | 44 | Probabilistic delivery windows — supplier reliability distributions, PO confidence levels | 5 | 7 | 5 | 17 |
| 80 | 68 | Exception trend and anomaly clustering — cohort analysis, cascade detection, seasonal patterns | 5 | 7 | 5 | 17 |
| 81 | 72 | Multi-dimensional supplier reliability — OTD, quality, price stability → composite score | 5 | 7 | 5 | 17 |
| 82 | 81 | Probabilistic classification with confidence — weighted scoring from CV, skewness, kurtosis | 5 | 7 | 5 | 17 |
| 83 | 120 | Global search across all panels — "Find item X" with cross-panel results | 5 | 4 | 8 | 17 |
| 84 | 1 | Demand-LT correlation coefficient — add configurable rho to SS formula | 6 | 7 | 3 | 16 |
| 85 | 36 | Supplier capacity constraints — monthly limits, ABC-priority allocation | 6 | 6 | 4 | 16 |
| 86 | 94 | Cross-module data contracts — explicit YAML schemas with column-level validation | 6 | 7 | 3 | 16 |
| 87 | 99 | Snapshot-based rollback — pre-step backup, 7-day retention, one-click undo | 6 | 7 | 3 | 16 |
| 88 | 132 | WCAG 2.1 AA accessibility — focus indicators, screen reader labels, semantic HTML | 6 | 3 | 7 | 16 |
| 89 | 56 | Penalty-weighted fill rate KPI — ABC multipliers, miss-streak escalation | 5 | 6 | 5 | 16 |
| 90 | 74 | Seasonal LT pattern detection — STL decomposition, ARIMA forecasting for volatile suppliers | 5 | 8 | 3 | 16 |
| 91 | 76 | Real-time supplier risk monitoring — daily KPI updates, early warning on OTD degradation | 5 | 6 | 5 | 16 |
| 92 | 89 | Seasonal ABC adjustment — peak-month DOS uplift, trough-month reduction | 5 | 7 | 4 | 16 |
| 93 | 98 | Data lineage tracking — source_run_id + lineage JSON on every fact row | 5 | 7 | 4 | 16 |
| 94 | 135 | Real-time sync status badge — green/spinner/red indicator, manual refresh button | 5 | 4 | 7 | 16 |
| 95 | 5 | Monte Carlo as validation layer — run simulation for high-CV items, reconcile vs analytical | 4 | 9 | 3 | 16 |
| 96 | 42 | 8+ parameterized scenarios — demand shock, LT delay, PO cancellation, partial receives | 4 | 7 | 5 | 16 |
| 97 | 21 | Integrate statsmodels ETS/ARIMA — replace naive fallback with proper statistical methods | 6 | 7 | 2 | 15 |
| 98 | 61 | 12+ exception types — demand_cliff, supplier_risk, policy_fit_gap, seasonal_anomaly | 6 | 5 | 4 | 15 |
| 99 | 16 | Multi-supplier sourcing optimization — score 3 suppliers per item by total landed cost | 5 | 6 | 4 | 15 |
| 100 | 39 | Sourcing intelligence for PO conversion — multi-supplier scoring, recommended vs preferred | 5 | 5 | 5 | 15 |

## Future / Nice-to-Have (Score 10-15)

| # | ID | Improvement | A | B | C | Score |
|---|-----|------------|---|---|---|-------|
| 101 | 73 | Multi-source LT blending — reliability-weighted average across approved suppliers | 5 | 7 | 3 | 15 |
| 102 | 77 | Supplier consolidation analysis — NPV scenarios with volume discounts vs resilience | 5 | 6 | 4 | 15 |
| 103 | 122 | Lazy-loaded panels with skeleton — React.lazy + Suspense for code-splitting | 5 | 3 | 7 | 15 |
| 104 | 123 | Cross-panel query cache — shared React Query context to deduplicate requests | 5 | 4 | 6 | 15 |
| 105 | 128 | Keyboard shortcuts — Alt+1-8 for groups, Alt+N/P for next/prev panel | 5 | 3 | 7 | 15 |
| 106 | 134 | Favorite panels + drag-reorder — star favorites, drag-to-reorder, persist per user | 5 | 3 | 7 | 15 |
| 107 | 28 | Staged promotion with canary rollout — auto-approve on WAPE delta, 10% canary | 4 | 8 | 3 | 15 |
| 108 | 50 | Simulation-projection feedback loop — projection errors feed into SS simulation | 4 | 8 | 3 | 15 |
| 109 | 59 | Closed-loop SL optimization — quarterly policy effectiveness testing, A/B flags | 4 | 8 | 3 | 15 |
| 110 | 31 | Multi-item joint EOQ — supplier-based consolidation groups with volume tiers | 5 | 6 | 3 | 14 |
| 111 | 49 | Event-driven incremental refresh — trigger on inventory/PO changes, 90s latency | 5 | 5 | 4 | 14 |
| 112 | 91 | Pipeline DAG with dependency validation — topological sort, fail-fast on missing upstream | 5 | 6 | 3 | 14 |
| 113 | 96 | Pipeline SLA monitoring — per-step timing, failure rate thresholds, health dashboard | 5 | 5 | 4 | 14 |
| 114 | 97 | Versioned config registry — PostgreSQL-backed with audit trail, rollback | 5 | 6 | 3 | 14 |
| 115 | 131 | Print-friendly view — clean A4 layout, no chrome, charts scaled, watermark | 5 | 3 | 6 | 14 |
| 116 | 26 | Bayesian cold-start bootstrapping — cluster priors and category regression for new SKUs | 4 | 7 | 3 | 14 |
| 117 | 47 | Multi-location network transfers — minimum-cost flow to balance excess/shortage | 4 | 6 | 4 | 14 |
| 118 | 57 | Risk-adjusted fill rate metric — weight shortages by margin, criticality, customer impact | 4 | 6 | 4 | 14 |
| 119 | 71 | CUSUM change-point detection — statistical rigor for LT regime shift identification | 4 | 8 | 2 | 14 |
| 120 | 80 | Demand-driven LT forecasting — elasticity model: demand growth → induced LT increase | 4 | 7 | 3 | 14 |
| 121 | 75 | SS variability breakdown — Sobol sensitivity indices showing which suppliers drive highest SS | 3 | 8 | 3 | 14 |
| 122 | 95 | Event-driven triggers — auto-trigger repl plan on forecast publish via webhooks | 5 | 5 | 3 | 13 |
| 123 | 6 | Multi-echelon SS — hub aggregation reduces CV; network-aware SS for hub vs retail | 4 | 6 | 3 | 13 |
| 124 | 22 | Real foundation model inference — load actual Chronos/NBeats weights, not fallbacks | 4 | 7 | 2 | 13 |
| 125 | 25 | MinTrace forecast reconciliation — enforce hierarchy constraints (item sum = location total) | 4 | 7 | 2 | 13 |
| 126 | 41 | Daily demand disaggregation — day-of-week multipliers, promotional calendar | 4 | 6 | 3 | 13 |
| 127 | 86 | 3D segmentation (ABC × XYZ × Margin × Criticality) — 27-cell matrix | 4 | 5 | 4 | 13 |
| 128 | 29 | Horizon-stratified ensemble — separate 1-6/7-12/13-24 month ensembles | 3 | 7 | 2 | 12 |
| 129 | 30 | Meta-learner ensemble blending — RF to predict optimal blend weights per DFU | 3 | 7 | 2 | 12 |
| 130 | 58 | ML fill rate prediction — gradient boosting on lagged fill rate, demand velocity, SS adequacy | 3 | 6 | 3 | 12 |
| 131 | 62 | Bayesian severity scoring with uncertainty — multi-factor posterior, severity ± confidence | 3 | 6 | 3 | 12 |
| 132 | 100 | Pipeline orchestration API with WebSocket — multi-step execution, pause/resume, streaming | 3 | 4 | 5 | 12 |
| 133 | 9 | Parallel batch computation — ThreadPoolExecutor for DFU processing, checkpoint/resume | 5 | 4 | 2 | 11 |
| 134 | 127 | Mobile responsive bottom nav — collapse pills into bottom bar on mobile | 3 | 2 | 6 | 11 |
| 135 | 20 | DAG-based pipeline orchestration — Airflow/dbt-style with SLA enforcement | 3 | 5 | 2 | 10 |

---

*Generated 2026-04-09 by 12 specialist agents + 3 expert judges. Scores: A = Simplicity & Inventory Planning, B = Backtesting & Validation, C = UI/UX. Each scored 1-10; total = A + B + C.*
