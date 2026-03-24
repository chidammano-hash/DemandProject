import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    fetchRebalancingKpis: vi.fn(),
    fetchRebalancingPlans: vi.fn(),
    fetchPlanTransfers: vi.fn(),
    computeRebalancingPlan: vi.fn(),
    approveTransfer: vi.fn(),
    rejectTransfer: vi.fn(),
    approveAllTransfers: vi.fn(),
  };
});

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  BarChart: ({ children }: any) => (
    <div data-testid="bar-chart">{children}</div>
  ),
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ScatterChart: ({ children }: any) => (
    <div data-testid="scatter-chart">{children}</div>
  ),
  Scatter: () => null,
  ZAxis: () => null,
  ReferenceLine: () => null,
}));

vi.mock("@/context/GlobalFilterContext", async () => {
  const actual = await vi.importActual("@/context/GlobalFilterContext");
  return {
    ...actual,
    useGlobalFilterContext: () => ({
      filters: {
        brand: [],
        category: [],
        market: [],
        channel: [],
        item: [],
        location: [],
      cluster: [],
        timeGrain: "month" as const,
      },
      setFilters: vi.fn(),
      resetFilters: vi.fn(),
      hasActiveFilters: false,
      planningDate: null,
    }),
  };
});

import {
  fetchRebalancingKpis,
  fetchRebalancingPlans,
  fetchPlanTransfers,
} from "@/api/queries";
import { RebalancingPanel } from "../inv-planning/RebalancingPanel";

const mockKpis = {
  total_multi_loc_items: 500,
  avg_dos_cv: 0.35,
  network_balance_score: 65.0,
  imbalanced_items: 42,
  total_excess_locs: 120,
  total_shortage_locs: 85,
  latest_plan: null,
};

const mockPlan = {
  plan_id: "plan-001",
  computation_date: "2026-02-24",
  solver_method: "greedy",
  objective: "min_cost",
  total_transfer_qty: 5000,
  total_transfer_cost: 2500,
  total_avoided_stockout_value: 15000,
  net_roi: 5.0,
  items_rebalanced: 42,
  lanes_used: 15,
  status: "draft",
  solver_runtime_ms: 1200,
  created_ts: "2026-02-24T05:00:00Z",
};

const mockTransfer = {
  transfer_id: "t-001",
  item_id: "100320",
  source_loc: "1401-BULK",
  dest_loc: "1501-PICK",
  transfer_mode: "truck",
  recommended_qty: 500,
  approved_qty: null,
  source_on_hand: 2000,
  source_dos: 45,
  source_ss_target: 800,
  source_excess_qty: 1200,
  dest_on_hand: 100,
  dest_dos: 3,
  dest_ss_target: 500,
  dest_shortage_qty: 400,
  transfer_cost: 250,
  carrying_cost_saved: 100,
  stockout_cost_avoided: 2000,
  net_benefit: 1850,
  roi: 7.4,
  planned_ship_date: "2026-03-01",
  expected_arrival_date: "2026-03-04",
  transfer_lt_days: 3,
  priority_score: 12.5,
  abc_class: "A",
  urgency: "critical",
  status: "recommended",
  approved_by: null,
  rejection_reason: null,
  notes: null,
};

const mockTransfer2 = {
  ...mockTransfer,
  transfer_id: "t-002",
  item_id: "200450",
  source_loc: "2001-BULK",
  dest_loc: "2501-PICK",
  urgency: "low",
  recommended_qty: 200,
  transfer_cost: 80,
  net_benefit: 600,
  roi: 6.5,
};

beforeEach(() => {
  vi.clearAllMocks();
  (fetchRebalancingKpis as any).mockResolvedValue(mockKpis);
  (fetchRebalancingPlans as any).mockResolvedValue({ total: 0, rows: [] });
  (fetchPlanTransfers as any).mockResolvedValue({ total: 0, rows: [] });
});

describe("RebalancingPanel", () => {
  it("renders 4 KPI cards", async () => {
    (fetchRebalancingPlans as any).mockResolvedValue({
      total: 0,
      rows: [],
    });

    render(
      <TestQueryWrapper>
        <RebalancingPanel />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Transfer Opportunities")).toBeInTheDocument();
    });
    expect(screen.getByText("Est. Cost Savings")).toBeInTheDocument();
    expect(screen.getByText("Urgent Transfers")).toBeInTheDocument();
    expect(screen.getByText("Network Balance")).toBeInTheDocument();
  });

  it("renders compute button", async () => {
    render(
      <TestQueryWrapper>
        <RebalancingPanel />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Compute Plan" }),
      ).toBeInTheDocument();
    });
  });

  it("renders empty state when no plans", async () => {
    (fetchRebalancingPlans as any).mockResolvedValue({
      total: 0,
      rows: [],
    });

    render(
      <TestQueryWrapper>
        <RebalancingPanel />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(
        screen.getByText("No rebalancing plan computed"),
      ).toBeInTheDocument();
    });
  });

  it("renders transfers table when data available", async () => {
    (fetchRebalancingPlans as any).mockResolvedValue({
      total: 1,
      rows: [mockPlan],
    });
    (fetchPlanTransfers as any).mockResolvedValue({
      total: 2,
      rows: [mockTransfer, mockTransfer2],
    });

    render(
      <TestQueryWrapper>
        <RebalancingPanel />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("100320")).toBeInTheDocument();
    });
    expect(screen.getByText("200450")).toBeInTheDocument();
    expect(screen.getByText("1401-BULK")).toBeInTheDocument();
    expect(screen.getByText("1501-PICK")).toBeInTheDocument();
  });

  it("renders urgency badges", async () => {
    (fetchRebalancingPlans as any).mockResolvedValue({
      total: 1,
      rows: [mockPlan],
    });
    (fetchPlanTransfers as any).mockResolvedValue({
      total: 1,
      rows: [mockTransfer],
    });

    render(
      <TestQueryWrapper>
        <RebalancingPanel />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("critical")).toBeInTheDocument();
    });
  });
});
