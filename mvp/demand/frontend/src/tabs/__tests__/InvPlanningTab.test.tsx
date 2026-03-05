import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

// Mock recharts
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

vi.mock("@/api/queries", () => ({
  queryKeys: {
    eoqSummary: (p: Record<string, unknown>) => ["eoq-summary", p],
    eoqDetail: (p: Record<string, unknown>) => ["eoq-detail", p],
    eoqSensitivity: (p: Record<string, unknown>) => ["eoq-sensitivity", p],
    policyList: () => ["policy-list"],
    policyCompliance: () => ["policy-compliance"],
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
    total_dfus: 150,
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
        item_no: "ITEM001", loc: "LOC1", abc_vol: "A",
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
    item_no: null,
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
        dfu_count: 150,
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
        dfu_count: 30,
      },
    ],
  }),
  fetchPolicyCompliance: vi.fn().mockResolvedValue({
    total_dfus: 500,
    assigned_count: 420,
    unassigned_count: 80,
    assignment_pct: 84.0,
    by_policy: {
      A_continuous_v1: {
        policy_name: "A-Class Continuous Review (ROP/EOQ)",
        policy_type: "continuous_rop",
        dfu_count: 150,
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
    dfu_count: 150,
  }),
  fetchHealthSummary: vi.fn().mockResolvedValue({
    total_dfus: 500,
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
        item_no: "ITEM001", loc: "LOC1", abc_vol: "A", variability_class: "low", cluster_assignment: "c1",
        health_score: 82, health_tier: "healthy",
        score_ss_coverage: 25, score_dos_target: 25, score_stockout_risk: 25, score_forecast_accuracy: 20,
        ss_coverage: 1.8, current_dos: 22.5, target_dos_min: 15.0, target_dos_max: 30.0,
        is_below_ss: false, recent_wape: 0.12, stockout_count_3m: 0,
      },
      {
        item_no: "ITEM002", loc: "LOC2", abc_vol: "C", variability_class: "high", cluster_assignment: "c3",
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
        exception_id: "exc-001", item_no: "ITEM001", loc: "LOC1",
        exception_date: "2026-03-04", exception_type: "below_rop", severity: "high",
        current_qty_on_hand: 150, current_dos: 30, ss_combined: 200, reorder_point: 180,
        recommended_order_qty: 100, recommended_order_by: "2026-03-11",
        expected_receipt_date: "2026-03-16", estimated_order_value: 1500,
        policy_id: "A_continuous_v1", status: "open",
        acknowledged_by: null, notes: null,
      },
      {
        exception_id: "exc-002", item_no: "ITEM002", loc: "LOC2",
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
    total_dfus: 500, avg_fill_rate_3m: 0.93, fill_rate_ytd: 0.91,
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
    timeGrain: "month",
  };
  return {
    filters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
  };
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
    await waitFor(() => {
      expect(screen.getByText("Policy Management")).toBeDefined();
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
    await waitFor(() => {
      expect(screen.getByText("Portfolio Health Score")).toBeDefined();
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
    await waitFor(() => {
      expect(screen.getByText("Healthy")).toBeDefined();
      expect(screen.getByText("Monitor")).toBeDefined();
      expect(screen.getByText("At Risk")).toBeDefined();
      // "Critical" appears in health cards and exception severity pills — use getAllByText
      const criticalEls = screen.getAllByText("Critical");
      expect(criticalEls.length).toBeGreaterThan(0);
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
    await waitFor(() => {
      // item_no appears in both EOQ and health detail tables
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
    await waitFor(() => {
      expect(screen.getByText(/Score Components/i)).toBeDefined();
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
    await waitFor(() => {
      expect(screen.getByText("Exception Queue")).toBeDefined();
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
    await waitFor(() => {
      expect(screen.getByText("Total Open")).toBeDefined();
      // "Critical" appears in multiple places — use getAllByText
      const criticalEls = screen.getAllByText("Critical");
      expect(criticalEls.length).toBeGreaterThan(0);
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
    await waitFor(() => {
      // exception_type label should appear
      const els = screen.getAllByText(/below_rop|Below ROP/i);
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
    await waitFor(() => {
      expect(screen.getByText("Generate Exceptions")).toBeDefined();
    });
  });
});
