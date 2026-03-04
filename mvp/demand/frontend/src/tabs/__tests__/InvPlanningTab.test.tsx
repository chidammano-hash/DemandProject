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
      expect(screen.getByText("ITEM001")).toBeDefined();
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
      expect(screen.getByPlaceholderText("Filter by item…")).toBeDefined();
      expect(screen.getByPlaceholderText("Filter by location…")).toBeDefined();
    });
  });
});
