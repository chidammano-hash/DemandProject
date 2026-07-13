import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

// Mock evolution query module (for new F3.x / F4.x panels)
vi.mock("@/api/queries/evolution", () => ({
  blendedKeys: { summary: () => ["blended-summary"], list: () => ["blended-list"] },
  echelonKeys: { summary: () => ["echelon-summary"], targets: () => ["echelon-targets"] },
  financialPlanKeys: { budget: () => ["budget"], workingCapital: () => ["wc"] },
  eventKeys: { calendar: () => ["event-calendar"] },
  scenarioKeys: { list: () => ["scenarios-list"], results: () => ["scenarios-results"] },
  sopKeys: { cycles: () => ["sop-cycles"], gaps: () => ["sop-gaps"], approvedPlan: () => ["sop-plan"] },
  biasKeys: { summary: () => ["bias-summary"], flagged: () => ["bias-flagged"] },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchBiasCorrectionSummary: vi.fn().mockResolvedValue({ sku_count: 0, avg_correction_factor: null, flagged_count: 0, clipped_count: 0, avg_rolling_bias: null, last_computed_at: null }),
  fetchFlaggedBiasCorrections: vi.fn().mockResolvedValue({ total: 0, flagged: [] }),
  fetchBlendedForecast: vi.fn().mockResolvedValue({ total: 0, page: 1, rows: [] }),
  fetchBlendedSummary: vi.fn().mockResolvedValue({ total_skus: 0, total_weeks: 0, avg_alpha: null, capped_count: 0 }),
  fetchEchelonTargets: vi.fn().mockResolvedValue({ total: 0, page: 1, rows: [] }),
  fetchEchelonSummary: vi.fn().mockResolvedValue({ total_nodes: 0, critical_count: 0, high_count: 0, avg_coverage_days: null }),
  fetchBudgetStatus: vi.fn().mockResolvedValue({ total: 0, budgets: [] }),
  fetchWorkingCapitalTrend: vi.fn().mockResolvedValue({ months: [] }),
  fetchEventCalendar: vi.fn().mockResolvedValue({ total: 0, events: [] }),
  fetchSupplyScenarios: vi.fn().mockResolvedValue({ total: 0, scenarios: [] }),
  fetchScenarioResults: vi.fn().mockResolvedValue({ scenario_id: "", items: [], total_impact: 0, total_stockout_days: 0 }),
}));

// Mock cross-domain planning insights query module
vi.mock("@/api/queries/inv-planning-insights", () => ({
  insightKeys: {
    actionFeed: () => ["inv-planning", "action-feed"],
    dailyBriefing: () => ["inv-planning", "daily-briefing"],
    rootCause: (item: string, loc: string) => ["inv-planning", "root-cause", item, loc],
    segmentDashboard: (segment: string) => ["inv-planning", "segment-dashboard", segment],
    ssCostBenefit: (params: Record<string, unknown>) => ["inv-planning", "ss-cost-benefit", params],
    serviceLevelWaterfall: () => ["inv-planning", "service-level-waterfall"],
    serviceLevelBridge: () => ["inv-planning", "service-level-bridge"],
    networkHeatmap: () => ["inv-planning", "network-heatmap"],
    planningScorecard: () => ["inv-planning", "planning-scorecard"],
    cashFlow: () => ["inv-planning", "cash-flow-timeline"],
    constrainedOpt: (budget: number) => ["inv-planning", "constrained-opt", budget],
    proactiveRebalancing: () => ["inv-planning", "proactive-rebalancing"],
  },
  STALE_INSIGHTS: { ONE_MIN: 60000, FIVE_MIN: 300000 },
  fetchActionFeed: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  fetchDailyBriefing: vi.fn().mockResolvedValue({ date: "2026-04-08", urgent: { label: "Act within 24 hours", items: [] }, this_week: { label: "Review this week", items: [] }, portfolio: { label: "Portfolio Health", items: [] }, actions: { label: "Top 3 Recommended Actions", items: [] }, stats: { total_skus: 0, below_ss_count: 0, excess_count: 0, total_excess_value: 0, total_stockout_risk_value: 0, avg_health_score: null } }),
  fetchRootCause: vi.fn().mockResolvedValue({ causes: [] }),
  fetchSegmentDashboard: vi.fn().mockResolvedValue({ segment: "AX", sku_count: 0, kpis: {}, exceptions: [], policy_distribution: {} }),
  fetchSsCostBenefit: vi.fn().mockResolvedValue({ items: [], total: 0, summary: { total_holding_cost: 0, total_stockout_risk: 0, over_stocked_count: 0, under_stocked_count: 0 } }),
  fetchServiceLevelWaterfall: vi.fn().mockResolvedValue({ steps: [], achieved_csl: 0 }),
  fetchServiceLevelBridge: vi.fn().mockResolvedValue({ target: null, actual: null, steps: [], by_class: [], month: null }),
  fetchNetworkHeatmap: vi.fn().mockResolvedValue({ locations: [], categories: [], cells: [] }),
  fetchPlanningScorecard: vi.fn().mockResolvedValue({ metrics: [] }),
  fetchCashFlowTimeline: vi.fn().mockResolvedValue({ months: [] }),
  fetchConstrainedOpt: vi.fn().mockResolvedValue({ budget: 0, allocated: 0, items_improved: 0, avg_csl_before: 0, avg_csl_after: 0, allocations: [] }),
  fetchProactiveRebalancing: vi.fn().mockResolvedValue({ opportunities: [], total_opportunities: 0 }),
}));

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  queryKeys: {
    eoqSummary: (p: Record<string, unknown>) => ["eoq-summary", p],
    eoqDetail: (p: Record<string, unknown>) => ["eoq-detail", p],
    eoqSensitivity: (p: Record<string, unknown>) => ["eoq-sensitivity", p],
    policyList: () => ["policy-list"],
    policyCompliance: () => ["policy-compliance"],
    variabilitySummary: (p?: Record<string, unknown>) => ["variability-summary", p ?? {}],
    variabilityDetail: (p?: Record<string, unknown>) => ["variability-detail", p ?? {}],
    ltSummary: (p?: Record<string, unknown>) => ["lt-summary", p ?? {}],
    ltProfile: (p?: Record<string, unknown>) => ["lt-profile", p ?? {}],
    productionForecast: (p?: Record<string, unknown>) => ["production-forecast", p ?? {}],
    productionForecastSummary: (p?: Record<string, unknown>) => ["production-forecast-summary", p ?? {}],
    productionForecastVersions: () => ["production-forecast-versions"],
    openPOs: (p: Record<string, unknown>) => ["open-pos", p],
    openPOSummary: () => ["open-po-summary"],
    pastDuePOs: () => ["past-due-pos"],
    plannedOrders: (p: Record<string, unknown>) => ["planned-orders", p],
    plannedOrdersSummary: () => ["planned-orders-summary"],
  },
  healthKeys: {
    summary: (f?: Record<string, unknown>) => ["health-summary", f ?? {}],
    detail: (p?: Record<string, unknown>) => ["health-detail", p ?? {}],
    heatmap: (x?: string, y?: string) => ["health-heatmap", x ?? "abc_vol", y ?? "variability_class"],
  },
  exceptionKeys: {
    list:    (p?: Record<string, unknown>) => ["exception-list",    p ?? {}],
    summary: (f?: Record<string, unknown>) => ["exception-summary", f ?? {}],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchEoqSummary: vi.fn().mockResolvedValue({
    total_skus: 150,
    avg_effective_eoq: 219.09,
    total_cycle_stock: 15000,
    avg_order_frequency: 5.5,
    total_annual_cost: 75000,
    by_abc: [
      { abc_vol: "A", count: 30, avg_eoq: 310.0, total_cycle_stock: 5000, total_annual_cost: 25000, avg_order_frequency: 3.9 },
      { abc_vol: "B", count: 80, avg_eoq: 220.0, total_cycle_stock: 8000, total_annual_cost: 35000, avg_order_frequency: 5.5 },
    ],
  }),
  fetchEoqDetail: vi.fn().mockResolvedValue({
    total: 2,
    limit: 50,
    offset: 0,
    rows: [
      {
        item_id: "ITEM001", loc: "LOC1", abc_vol: "A",
        demand_mean_monthly: 100, annual_demand: 1200,
        ordering_cost: 50, holding_cost_pct: 0.25, unit_cost: 10, moq: 1,
        eoq: 219.09, effective_eoq: 219.09, eoq_cycle_stock: 109.54,
        order_frequency: 5.48, annual_holding_cost: 273.86,
        annual_order_cost: 273.86, total_annual_cost: 547.72,
        computed_at: "2025-06-15T00:00:00Z",
      },
    ],
  }),
  fetchEoqSensitivity: vi.fn().mockResolvedValue({
    item_id: null,
    loc: null,
    avg_demand_monthly: 100.0,
    curve: [
      { ordering_cost: 5.0, eoq: 69.28, effective_eoq: 69.28, total_annual_cost: 547.72 },
      { ordering_cost: 200.0, eoq: 438.18, effective_eoq: 438.18, total_annual_cost: 2738.6 },
    ],
  }),
  fetchPolicies: vi.fn().mockResolvedValue({
    policies: [
      {
        policy_id: "A_continuous_v1",
        policy_name: "A-Class Continuous Review (ROP/EOQ)",
        policy_type: "continuous_rop",
        segment: "A",
        review_cycle_days: null,
        service_level: 0.98,
        use_eoq: true,
        use_safety_stock: true,
        active: true,
        sku_count: 150,
      },
      {
        policy_id: "lumpy_manual_v1",
        policy_name: "Lumpy/Intermittent — Manual Review",
        policy_type: "manual",
        segment: "lumpy",
        review_cycle_days: null,
        service_level: 0.85,
        use_eoq: false,
        use_safety_stock: false,
        active: true,
        sku_count: 30,
      },
    ],
  }),
  fetchPolicyCompliance: vi.fn().mockResolvedValue({
    total_skus: 500,
    assigned_count: 420,
    unassigned_count: 80,
    assignment_pct: 84.0,
    by_policy: {
      A_continuous_v1: {
        policy_name: "A-Class Continuous Review (ROP/EOQ)",
        policy_type: "continuous_rop",
        sku_count: 150,
        below_ss_pct: null,
        avg_ss_coverage: null,
        avg_dos: 32.5,
      },
    },
  }),
  assignPolicy: vi.fn().mockResolvedValue({ assigned_count: 5, failed_count: 0, already_assigned_count: 0 }),
  updatePolicy: vi.fn().mockResolvedValue({
    policy_id: "A_continuous_v1",
    policy_name: "A-Class Continuous Review (ROP/EOQ)",
    policy_type: "continuous_rop",
    segment: "A",
    review_cycle_days: null,
    service_level: 0.99,
    use_eoq: true,
    use_safety_stock: true,
    active: true,
    sku_count: 150,
  }),
  fetchHealthSummary: vi.fn().mockResolvedValue({
    total_skus: 500,
    by_tier: { healthy: 200, monitor: 150, at_risk: 100, critical: 50 },
    avg_health_score: 68.5,
    component_avgs: { ss_coverage: 17.0, dos_target: 18.5, stockout_risk: 21.0, forecast_accuracy: 16.5 },
    score_histogram: [
      { bucket: "40-59", count: 100 },
      { bucket: "60-79", count: 150 },
      { bucket: "80-100", count: 200 },
    ],
  }),
  fetchHealthDetail: vi.fn().mockResolvedValue({
    total: 2,
    rows: [
      {
        item_id: "ITEM001", loc: "LOC1", abc_vol: "A", variability_class: "low", cluster_assignment: "c1",
        health_score: 82, health_tier: "healthy",
        score_ss_coverage: 25, score_dos_target: 25, score_stockout_risk: 25, score_forecast_accuracy: 20,
        ss_coverage: 1.8, current_dos: 22.5, target_dos_min: 15.0, target_dos_max: 30.0,
        is_below_ss: false, recent_wape: 0.12, stockout_count_3m: 0,
      },
      {
        item_id: "ITEM002", loc: "LOC2", abc_vol: "C", variability_class: "high", cluster_assignment: "c3",
        health_score: 35, health_tier: "critical",
        score_ss_coverage: 0, score_dos_target: 5, score_stockout_risk: 8, score_forecast_accuracy: 8,
        ss_coverage: 0.2, current_dos: 5.0, target_dos_min: 15.0, target_dos_max: 30.0,
        is_below_ss: true, recent_wape: 0.58, stockout_count_3m: 2,
      },
    ],
  }),
  fetchHealthHeatmap: vi.fn().mockResolvedValue({
    x_labels: ["A", "B", "C"],
    y_labels: ["high", "low", "medium"],
    cells: [
      { x: "A", y: "low", avg_health_score: 82.5, count: 50, critical_count: 2 },
      { x: "C", y: "high", avg_health_score: 45.0, count: 20, critical_count: 8 },
    ],
  }),
  fetchExceptions: vi.fn().mockResolvedValue({
    total: 2,
    limit: 50,
    offset: 0,
    rows: [
      {
        exception_id: "exc-001", item_id: "ITEM001", loc: "LOC1",
        exception_date: "2026-03-04", exception_type: "below_rop", severity: "high",
        current_qty_on_hand: 150, current_dos: 30, ss_combined: 200, reorder_point: 180,
        recommended_order_qty: 100, recommended_order_by: "2026-03-11",
        expected_receipt_date: "2026-03-16", estimated_order_value: 1500,
        policy_id: "A_continuous_v1", status: "open",
        acknowledged_by: null, notes: null,
      },
      {
        exception_id: "exc-002", item_id: "ITEM002", loc: "LOC2",
        exception_date: "2026-03-04", exception_type: "stockout", severity: "critical",
        current_qty_on_hand: 0, current_dos: 0, ss_combined: 100, reorder_point: 120,
        recommended_order_qty: 200, recommended_order_by: "2026-03-04",
        expected_receipt_date: "2026-03-09", estimated_order_value: 3000,
        policy_id: "A_continuous_v1", status: "open",
        acknowledged_by: null, notes: null,
      },
    ],
  }),
  fetchExceptionSummary: vi.fn().mockResolvedValue({
    open_count: 25,
    by_type: { below_rop: 10, below_rop_critical: 0, below_ss: 8, stockout: 2, excess: 3, zero_velocity: 2 },
    by_severity: { critical: 10, high: 8, medium: 5, low: 2 },
    total_recommended_order_value: 15000,
    oldest_open_days: 14,
  }),
  acknowledgeException: vi.fn().mockResolvedValue({}),
  updateExceptionStatus: vi.fn().mockResolvedValue({}),
  generateExceptions: vi.fn().mockResolvedValue({ generated_count: 10, skipped_dedup: 2, by_type: {} }),
  fillRateKeys: {
    summary: () => ["fill-rate-summary"],
    trend: (p?: Record<string, unknown>) => ["fill-rate-trend", p ?? {}],
    detail: (p?: Record<string, unknown>) => ["fill-rate-detail", p ?? {}],
  },
  fetchFillRateSummary: vi.fn().mockResolvedValue({
    total_skus: 500, avg_fill_rate_3m: 0.93, fill_rate_ytd: 0.91,
    below_threshold_count: 25, critical_fill_rate_count: 10,
    by_abc: [{ abc_vol: "A", count: 100, avg_fill_rate: 0.97, shortage_qty: 500 }],
  }),
  fetchFillRateTrend: vi.fn().mockResolvedValue({
    trend: [
      { month_start: "2026-01-01", fill_rate: 0.92, shortage_qty: 1200, total_ordered: 50000 },
      { month_start: "2026-02-01", fill_rate: 0.94, shortage_qty: 900, total_ordered: 52000 },
    ],
  }),
  abcXyzKeys: {
    matrix: () => ["abc-xyz-matrix"],
    summary: () => ["abc-xyz-summary"],
    detail: (p?: Record<string, unknown>) => ["abc-xyz-detail", p ?? {}],
  },
  fetchAbcXyzMatrix: vi.fn().mockResolvedValue({
    matrix: [
      { abc_vol: "A", xyz_class: "X", segment: "AX", count: 50, avg_dos: 25.0, service_level: 0.98 },
    ],
    total_classified: 420,
    total_unclassified: 80,
  }),
  fetchAbcXyzSummary: vi.fn().mockResolvedValue({
    classified_count: 420, unclassified_count: 80, total: 500,
    by_xyz: { X: 150, Y: 200, Z: 70 },
  }),
  fetchAbcXyzDetail: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  supplierKeys: {
    summary: () => ["supplier-summary"],
    detail: (p?: Record<string, unknown>) => ["supplier-detail", p ?? {}],
  },
  fetchSupplierSummary: vi.fn().mockResolvedValue({
    total_suppliers: 25, avg_reliability_score: 72.5,
    reliable_count: 15, unreliable_count: 10,
  }),
  fetchSupplierDetail: vi.fn().mockResolvedValue({
    total: 1,
    rows: [
      { supplier_no: "SUP01", supplier_name: "Supplier One",
        sku_loc_count: 25, distinct_items: 20,
        avg_lt_mean_days: 14.0, avg_lt_cv: 0.15, avg_lt_std_days: 2.1,
        pct_stable_lt: 0.88, pct_volatile_lt: 0.12,
        total_safety_stock_units: 500, total_ss_value: 5000,
        supplier_reliability_score: 78.5 },
    ],
  }),
  intramonthKeys: {
    summary: () => ["intramonth-summary"],
    detail: (p?: Record<string, unknown>) => ["intramonth-detail", p ?? {}],
  },
  fetchIntramonthSummary: vi.fn().mockResolvedValue({
    total_items: 500, items_with_stockout: 45, extended_stockout_count: 12,
    avg_stockout_day_rate: 0.08, total_est_lost_sales: 25000,
  }),
  fetchIntramonthDetail: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  safetyStockKeys: {
    summary: (f?: Record<string, unknown>) => ["safety-stock", "summary", f ?? {}],
    detail: (p?: Record<string, unknown>) => ["safety-stock", "detail", p ?? {}],
    waterfall: (itemNo: string, loc: string) => ["safety-stock", "waterfall", itemNo, loc],
    config: () => ["safety-stock", "config"],
  },
  fetchSafetyStockSummary: vi.fn().mockResolvedValue({
    total_skus: 300, below_ss_count: 45, avg_ss_coverage: 1.8, avg_ss_days: 21,
    by_abc: [{ abc_vol: "A", count: 100, below_ss_count: 10, avg_coverage: 2.1 }],
  }),
  fetchSafetyStockDetail: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchSafetyStockWaterfall: vi.fn().mockResolvedValue(null),
  fetchSafetyStockConfig: vi.fn().mockResolvedValue({}),
  fetchVariabilitySummary: vi.fn().mockResolvedValue({
    total_skus: 400, avg_cv: 0.35,
    by_class: { low: 150, medium: 180, high: 50, lumpy: 20 },
    cv_percentiles: { p25: 0.15, p50: 0.30, p75: 0.55, p95: 0.90 },
    avg_intermittency_ratio: 0.12, top_volatile: [],
  }),
  fetchVariabilityDetail: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchLtSummary: vi.fn().mockResolvedValue({
    total_profiles: 200, avg_lt_cv: 0.18, avg_lt_mean_days: 14.0,
    by_class: { stable: 120, moderate: 60, volatile: 20 },
    lt_cv_p50: 0.12, lt_cv_p95: 0.42,
    top_volatile: [],
  }),
  fetchLtProfile: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  demandSignalsKeys: {
    summary: (date?: string) => ["demand-signals", "summary", date],
    list: (p?: Record<string, unknown>) => ["demand-signals", "list", p ?? {}],
    item: (itemNo: string, loc: string) => ["demand-signals", "item", itemNo, loc],
  },
  fetchDemandSignalsSummary: vi.fn().mockResolvedValue({
    signal_date: "2026-03-05", above_plan_count: 30, below_plan_count: 20,
    on_plan_count: 50, urgent_count: 5, watch_count: 15, projected_stockouts: 3,
  }),
  fetchDemandSignals: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  simulationKeys: {
    results: (p?: Record<string, unknown>) => ["simulation", "results", p ?? {}],
    compare: (itemNo: string, loc: string) => ["simulation", "compare", itemNo, loc],
    status: (id: string) => ["simulation", "status", id],
  },
  fetchSimulationResults: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  runSimulation: vi.fn().mockResolvedValue(null),
  investmentKeys: {
    summary: (planId?: string) => ["investment", "summary", planId],
    detail: (p?: Record<string, unknown>) => ["investment", "detail", p ?? {}],
    frontier: (planId?: string) => ["investment", "frontier", planId],
  },
  fetchInvestmentSummary: vi.fn().mockResolvedValue(null),
  fetchInvestmentDetail: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchInvestmentFrontier: vi.fn().mockResolvedValue([]),
  runInvestmentPlan: vi.fn().mockResolvedValue(null),
  // Production Forecast (F1.1)
  fetchProductionForecastVersions: vi.fn().mockResolvedValue({
    versions: [
      {
        plan_version: "2026-03",
        dfu_count: 1000,
        total_rows: 12000,
        generated_at: "2026-03-01T06:00:00Z",
      },
    ],
  }),
  fetchProductionForecastSummary: vi.fn().mockResolvedValue({
    plan_version: "2026-03",
    horizon_months: 3,
    total_dfu_count: 1000,
    total_forecast_qty: 55000.0,
    generated_at: "2026-03-01T06:00:00Z",
    by_abc_class: [
      { abc_class: "A", dfu_count: 200, forecast_qty: 20000.0 },
      { abc_class: "B", dfu_count: 500, forecast_qty: 25000.0 },
    ],
    ci_coverage_pct: 100,
    avg_ci_width: 60,
  }),
  fetchProductionForecast: vi.fn().mockResolvedValue({
    item_id: "ITEM001", loc: "LOC1",
    plan_version: "2026-03", model_id: "lgbm_cluster",
    generated_at: "2026-03-01T06:00:00Z", horizon_months: 3, is_recursive: true,
    forecasts: [
      { forecast_month: "2026-04-01", forecast_qty: 150.0, forecast_qty_lower: 120.0, forecast_qty_upper: 180.0,
        model_id: "lgbm_cluster", cluster_id: 2, horizon_months: 1, is_recursive: true, lag_source: "actual" },
    ],
  }),
  // Open PO Integration (F1.3) + Projection (F1.2)
  projectionKeys: {
    sku: (p: Record<string, unknown>) => ["projection", "sku", p],
    atRisk: (h: number) => ["projection", "at-risk", h],
  },
  fetchOpenPOSummary: vi.fn().mockResolvedValue({
    total_open_lines: 8,
    total_open_value_usd: 125000.0,
    total_open_qty_by_status: { open: 900.0, partially_received: 100.0 },
    past_due_lines: 2,
    past_due_value_usd: 8750.0,
    avg_days_past_due: 14.5,
    suppliers_with_open_pos: 3,
    last_loaded_at: "2026-03-07T06:00:00Z",
  }),
  fetchOpenPOs: vi.fn().mockResolvedValue({
    total: 2,
    open_po_data_available: true,
    last_loaded_at: "2026-03-07T06:00:00Z",
    page: 1,
    page_size: 50,
    items: [
      {
        po_number: "PO-4521", po_line_number: 1, item_id: "100320", loc: "1401-BULK",
        supplier_id: "VENDOR-0042", supplier_name: "Acme Supply Co.",
        po_date: "2026-02-15", ordered_qty: 150, confirmed_qty: 150, received_qty: 0,
        open_qty: 150, unit_cost: 12.50, line_value: 1875.0,
        promised_delivery_date: "2026-03-14", confirmed_delivery_date: "2026-03-14",
        revised_delivery_date: null, effective_delivery_date: "2026-03-14",
        days_past_due: 0, line_status: "open",
      },
    ],
  }),
  fetchPastDuePOs: vi.fn().mockResolvedValue({ total: 0, items: [] }),
  fetchProjection: vi.fn().mockRejectedValue(new Error("No projection")),
  fetchProjectionAtRisk: vi.fn().mockResolvedValue({ total: 0, horizon_days: 30, page: 1, page_size: 50, items: [] }),
  refreshProjection: vi.fn().mockResolvedValue({ status: "ok", rows_written: 270, run_id: "test-run" }),
  // Planned Orders (F2.1)
  fetchPlannedOrdersSummary: vi.fn().mockResolvedValue({
    status_counts: { proposed: 5, approved: 2, released: 1, rejected: 0 },
    total_proposed_value_usd: 18750.0,
    total_approved_value_usd: 7500.0,
    past_due_proposed_count: 1,
    past_due_proposed_value_usd: 3750.0,
    avg_confidence_score: 0.873,
    low_confidence_count: 1,
    generated_at: "2026-03-02T08:04:22Z",
  }),
  fetchPlannedOrders: vi.fn().mockResolvedValue({
    total: 1,
    total_order_value_usd: 3750.0,
    past_due_count: 0,
    page: 1,
    page_size: 50,
    items: [
      {
        id: 1001, item_id: "100320", loc: "1401-BULK",
        supplier_id: "VENDOR-0042", supplier_name: "Acme Supply Co.",
        net_requirement_qty: 233.4, recommended_qty: 300.0, moq: 100.0,
        unit_cost: 12.5, order_value: 3750.0, currency: "USD",
        trigger_date: "2026-03-10", trigger_reason: "projected_below_ss",
        order_by_date: "2026-03-10", expected_receipt_date: "2026-03-24",
        lead_time_days: 14, current_qty_on_hand: 120.0, safety_stock: 60.0,
        reorder_point: 60.0, confirmed_inbound_qty: 200.0, lt_forecast_demand: 228.2,
        plan_version: "2026-03", confidence_score: 0.95,
        confidence_reason: "all data sources available",
        is_past_due: false, status: "proposed",
        created_at: "2026-03-02T08:04:22Z", approved_by: null, approved_at: null,
      },
    ],
  }),
  approvePlannedOrder: vi.fn().mockResolvedValue({ id: 1001, status: "approved", approved_by: "planner", approved_at: "2026-03-06T09:00:00Z" }),
  rejectPlannedOrder: vi.fn().mockResolvedValue({ id: 1001, status: "rejected" }),
  generatePlannedOrders: vi.fn().mockResolvedValue({ status: "accepted", job_id: "test-job-id" }),
  // Demand Plan (F2.2)
  fetchDemandPlanVersions: vi.fn().mockResolvedValue({ versions: [] }),
  fetchDemandPlan: vi.fn().mockResolvedValue({ item_id: "", loc: "", plan_version: "", generated_at: null, horizon_months: 12, rows: [] }),
  fetchDemandPlanWeekly: vi.fn().mockResolvedValue({ item_id: "", loc: "", plan_version: "", weeks: [] }),
  fetchDemandPlanComparison: vi.fn().mockResolvedValue({ item_id: "", loc: "", v1: "", v2: "", months: [] }),
  // Procurement Workflow (F2.4)
  fetchPurchaseOrders: vi.fn().mockResolvedValue({ total: 0, total_value: 0, page: 1, orders: [] }),
  approvePurchaseOrder: vi.fn().mockResolvedValue({ po_number: "DS-2026-04-001", status: "planner_approved", approved_by: "planner1" }),
  releasePurchaseOrder: vi.fn().mockResolvedValue({ po_number: "DS-2026-04-001", status: "buyer_released", released_by: "buyer1" }),
  exportPOsCSV: vi.fn().mockResolvedValue({ filename: "PO_export_test.csv", line_count: 1, total_value: 7584, csv_content: "" }),
  fetchPOTimeline: vi.fn().mockResolvedValue({ po_number: "DS-2026-04-001", current_status: "proposed", timeline: [] }),
  createPOFromException: vi.fn().mockResolvedValue({ po_number: "DS-2026-04-001", status: "proposed", total_value: null, requested_delivery_date: null }),
  // Cross-domain planning insights
  insightKeys: {
    actionFeed: () => ["inv-planning", "action-feed"],
    dailyBriefing: () => ["inv-planning", "daily-briefing"],
    rootCause: (item: string, loc: string) => ["inv-planning", "root-cause", item, loc],
    segmentDashboard: (segment: string) => ["inv-planning", "segment-dashboard", segment],
    ssCostBenefit: (params: Record<string, unknown>) => ["inv-planning", "ss-cost-benefit", params],
    serviceLevelWaterfall: () => ["inv-planning", "service-level-waterfall"],
    serviceLevelBridge: () => ["inv-planning", "service-level-bridge"],
    networkHeatmap: () => ["inv-planning", "network-heatmap"],
    planningScorecard: () => ["inv-planning", "planning-scorecard"],
    cashFlow: () => ["inv-planning", "cash-flow-timeline"],
    constrainedOpt: (budget: number) => ["inv-planning", "constrained-opt", budget],
    proactiveRebalancing: () => ["inv-planning", "proactive-rebalancing"],
  },
  STALE_INSIGHTS: { ONE_MIN: 60000, FIVE_MIN: 300000 },
  fetchActionFeed: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  fetchDailyBriefing: vi.fn().mockResolvedValue({ date: "2026-04-08", urgent: { label: "Act within 24 hours", items: [] }, this_week: { label: "Review this week", items: [] }, portfolio: { label: "Portfolio Health", items: [] }, actions: { label: "Top 3 Recommended Actions", items: [] }, stats: { total_skus: 0, below_ss_count: 0, excess_count: 0, total_excess_value: 0, total_stockout_risk_value: 0, avg_health_score: null } }),
  fetchRootCause: vi.fn().mockResolvedValue({ causes: [] }),
  fetchSegmentDashboard: vi.fn().mockResolvedValue({ segment: "AX", sku_count: 0, kpis: {}, exceptions: [], policy_distribution: {} }),
  fetchSsCostBenefit: vi.fn().mockResolvedValue({ items: [], total: 0, summary: { total_holding_cost: 0, total_stockout_risk: 0, over_stocked_count: 0, under_stocked_count: 0 } }),
  fetchServiceLevelWaterfall: vi.fn().mockResolvedValue({ steps: [], achieved_csl: 0 }),
  fetchServiceLevelBridge: vi.fn().mockResolvedValue({ target: null, actual: null, steps: [], by_class: [], month: null }),
  fetchNetworkHeatmap: vi.fn().mockResolvedValue({ locations: [], categories: [], cells: [] }),
  fetchPlanningScorecard: vi.fn().mockResolvedValue({ metrics: [] }),
  fetchCashFlowTimeline: vi.fn().mockResolvedValue({ months: [] }),
  fetchConstrainedOpt: vi.fn().mockResolvedValue({ budget: 0, allocated: 0, items_improved: 0, avg_csl_before: 0, avg_csl_after: 0, allocations: [] }),
  fetchProactiveRebalancing: vi.fn().mockResolvedValue({ opportunities: [], total_opportunities: 0 }),
  // Override Queue (F2.3)
  fetchOverrideSummary: vi.fn().mockResolvedValue({
    by_status: { pending_approval: 2, approved: 5, rejected: 1, expired: 0, superseded: 0 },
    sku_count_overridden: 3,
    total_uplift_units: 1200,
    total_uplift_value: 6000,
    by_type: { PROMO: 4, MANUAL: 4 },
  }),
  fetchOverrides: vi.fn().mockResolvedValue({ total: 0, page: 1, overrides: [] }),
  submitOverride: vi.fn().mockResolvedValue({ override_id: 1, status: "pending_approval", requires_approval: true, message: "ok" }),
  approveOverride: vi.fn().mockResolvedValue({ override_id: 1, status: "approved", approved_by: "manager", approved_at: "2026-03-07T00:00:00Z" }),
  rejectOverride: vi.fn().mockResolvedValue({ override_id: 1, status: "rejected" }),
  fetchConsensusPlan: vi.fn().mockResolvedValue({ plan_version: "", item_id: "", loc: "", months: [] }),
  // Sourcing
  fetchSourcingRows: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchSourcingSearch: vi.fn().mockResolvedValue({ rows: [] }),
  fetchSourcingByItem: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchSourcingBySupplier: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchSourcingNetwork: vi.fn().mockResolvedValue({
    total_rows: 0, supplier_count: 0, item_location_count: 0,
    single_source_count: 0, multi_source_count: 0, transit_modes: [],
  }),
  // Purchase Orders (full domain)
  fetchPORows: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchPOSearch: vi.fn().mockResolvedValue({ rows: [] }),
  fetchPOByNumber: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchPOSummary: vi.fn().mockResolvedValue({
    total_lines: 0, closed_lines: 0, open_lines: 0,
    distinct_pos: 0, distinct_suppliers: 0, distinct_items: 0,
    total_value: 0, open_value: 0, closed_value: 0,
  }),
  fetchPOAging: vi.fn().mockResolvedValue({ buckets: [] }),
  fetchPOOnTimeDelivery: vi.fn().mockResolvedValue({ suppliers: [] }),
}));

const { InvPlanningTab } = await import("@/tabs/InvPlanningTab");

function makeFilterContext(): GlobalFilterContextValue {
  const filters: GlobalFilters = {
    brand: [],
    category: [],
    market: [],
    channel: [],
    item: [],
    location: [],
      cluster: [],
    timeGrain: "month",
  };
  return {
    filters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
    planningDate: null,
  };
}

/** Navigate to a panel by clicking the group pill then the sub-tab.
 *  Switches to "All Panels" view first so all groups are visible
 *  (default view is "Daily Essentials" which shows only 6 panels). */
function navigateTo(groupShortLabel: string, tabLabel: string) {
  // Switch to All Panels view to expose all group pills
  fireEvent.click(screen.getByRole("button", { name: /All Panels/i }));
  fireEvent.click(screen.getByRole("tab", { name: groupShortLabel }));
  fireEvent.click(screen.getByRole("tab", { name: tabLabel }));
}

describe("InvPlanningTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "EOQ");
    await waitFor(() => {
      expect(screen.getByText("Total Cycle Stock")).toBeDefined();
      expect(screen.getByText("Avg EOQ Size")).toBeDefined();
      expect(screen.getByText("Avg Order Frequency")).toBeDefined();
      expect(screen.getByText("Total Annual Cost")).toBeDefined();
    });
  });

  it("renders sensitivity section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "EOQ");
    await waitFor(() => {
      expect(screen.getByText("EOQ Sensitivity")).toBeDefined();
    });
  });

  it("renders EOQ detail table header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "EOQ");
    await waitFor(() => {
      expect(screen.getByText("EOQ Detail")).toBeDefined();
    });
  });

  it("renders item from detail rows", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "EOQ");
    await waitFor(() => {
      const items = screen.getAllByText("ITEM001");
      expect(items.length).toBeGreaterThan(0);
    });
  });

  it("renders filter controls", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "EOQ");
    await waitFor(() => {
      // Multiple item/location filter inputs may exist across sections
      const itemInputs = screen.getAllByPlaceholderText("Filter by item…");
      expect(itemInputs.length).toBeGreaterThan(0);
      const locInputs = screen.getAllByPlaceholderText("Filter by location…");
      expect(locInputs.length).toBeGreaterThan(0);
    });
  });

  it("renders Policy Management section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "Policy");
    await waitFor(() => {
      expect(screen.getAllByText("Policy Management").length).toBeGreaterThan(0);
    });
  });

  it("renders Auto-assign All button", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "Policy");
    await waitFor(() => {
      expect(screen.getByText("Auto-assign All")).toBeDefined();
    });
  });

  it("renders policy cards from mocked data", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "Policy");
    await waitFor(() => {
      // Policy name appears in both the card and the compliance table — use getAllByText
      const els = screen.getAllByText("A-Class Continuous Review (ROP/EOQ)");
      expect(els.length).toBeGreaterThan(0);
    });
  });

  it("renders compliance section with DFU coverage", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "Policy");
    await waitFor(() => {
      expect(screen.getByText("DFU Coverage")).toBeDefined();
    });
  });

  it("renders Policy Compliance table", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Optimize", "Policy");
    await waitFor(() => {
      expect(screen.getByText("Policy Compliance")).toBeDefined();
    });
  });

  it("renders Portfolio Health Score section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Health");
    await waitFor(() => {
      expect(screen.getAllByText("Portfolio Health Score").length).toBeGreaterThan(0);
    });
  });

  it("renders health tier KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Health");
    await waitFor(() => {
      // Health tier labels may appear in KPI cards AND detail table rows — use getAllByText
      expect(screen.getAllByText("Healthy").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Monitor").length).toBeGreaterThan(0);
      // "At Risk" also appears in TodaysPlanBanner priority badge
      expect(screen.getAllByText("At Risk").length).toBeGreaterThan(0);
      // "Critical" appears in health cards and exception severity pills
      expect(screen.getAllByText("Critical").length).toBeGreaterThan(0);
    });
  });

  it("renders health detail section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Health");
    await waitFor(() => {
      expect(screen.getByText("Health Detail")).toBeDefined();
    });
  });

  it("renders health detail rows from mocked data", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Health");
    await waitFor(() => {
      // item_id appears in both EOQ and health detail tables
      const items = screen.getAllByText("ITEM001");
      expect(items.length).toBeGreaterThan(0);
    });
  });

  it("renders health distribution section", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Health");
    await waitFor(() => {
      expect(screen.getByText("Health Distribution")).toBeDefined();
    });
  });

  it("renders score components section", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Health");
    await waitFor(() => {
      expect(screen.getByText(/Risk Factor Breakdown/i)).toBeDefined();
    });
  });

  it("renders Exception Queue section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Exceptions");
    await waitFor(() => {
      expect(screen.getAllByText("Exception Queue").length).toBeGreaterThan(0);
    });
  });

  it("renders exception KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Exceptions");
    await waitFor(() => {
      expect(screen.getByText("Open Issues")).toBeDefined();
      // "Urgent" appears as KPI label; "Critical" still in severity filter pills
      const urgentEls = screen.getAllByText("Urgent");
      expect(urgentEls.length).toBeGreaterThan(0);
    });
  });

  it("renders exception table rows from mocked data", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Exceptions");
    await waitFor(() => {
      // exception_type label should appear (now business-friendly)
      const els = screen.getAllByText(/Needs Reorder|Urgent Reorder|below_rop/i);
      expect(els.length).toBeGreaterThan(0);
    });
  });

  it("renders Generate Exceptions button", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Daily Ops", "Exceptions");
    await waitFor(() => {
      expect(screen.getByText("Generate Exceptions")).toBeDefined();
    });
  });

  it("renders Safety Stock section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Safety Stock");
    await waitFor(() => {
      expect(screen.getAllByText("Safety Stock").length).toBeGreaterThan(0);
    });
  });

  it("renders Demand Variability section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Variability");
    await waitFor(() => {
      expect(screen.getByText("Demand Variability")).toBeDefined();
    });
  });

  it("renders Lead Time Analysis section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Lead Time");
    await waitFor(() => {
      expect(screen.getByText("Lead Time Analysis")).toBeDefined();
    });
  });

  it("renders Demand Signals section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Signals");
    await waitFor(() => {
      expect(screen.getAllByText("Demand Signals").length).toBeGreaterThan(0);
    });
  });

  it("renders Safety Stock Simulation section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Simulation");
    await waitFor(() => {
      expect(screen.getByText("Safety Stock Simulation")).toBeDefined();
    });
  });

  it("renders Investment Plan section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Investment");
    await waitFor(() => {
      expect(screen.getByText("Investment Plan")).toBeDefined();
    });
  });

  it("renders Demand Intelligence section header", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Demand Intelligence");
    await waitFor(() => {
      // Multiple elements contain "Demand Intelligence" (tab label, breadcrumb, panel title)
      expect(screen.getAllByText("Demand Intelligence").length).toBeGreaterThan(0);
    });
  });

  it("renders demand intelligence sub-tabs", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("Planning", "Demand Intelligence");
    await waitFor(() => {
      expect(screen.getByText("Production Forecast")).toBeDefined();
      expect(screen.getByText("Demand Plan")).toBeDefined();
      expect(screen.getByText("Blended Demand")).toBeDefined();
    });
  });

  it("renders Planned Orders panel with KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("OTC", "Planned Orders");
    await waitFor(() => {
      expect(screen.getAllByText("Planned Orders").length).toBeGreaterThan(0);
      expect(screen.getByText("Proposed")).toBeDefined();
      expect(screen.getByText("Past Due")).toBeDefined();
    });
  });

  it("renders planned order row from mocked data", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvPlanningTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    navigateTo("OTC", "Planned Orders");
    await waitFor(() => {
      const items = screen.getAllByText("100320");
      expect(items.length).toBeGreaterThan(0);
    });
  });
});
