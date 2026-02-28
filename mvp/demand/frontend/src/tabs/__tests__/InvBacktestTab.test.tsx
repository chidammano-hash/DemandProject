import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

// Mock recharts
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ComposedChart: ({ children }: { children: React.ReactNode }) => <div data-testid="composed-chart">{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Bar: () => null,
  Line: () => null,
  Cell: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

vi.mock("@/api/queries", () => ({
  queryKeys: {
    forecastModels: () => ["forecast-models"],
    invBacktestSummary: (p: Record<string, unknown>) => ["inv-backtest-summary", p],
    invBacktestTrend: (p: Record<string, unknown>) => ["inv-backtest-trend", p],
    invBacktestRootCause: (p: Record<string, unknown>) => ["inv-backtest-root-cause", p],
    invBacktestDetail: (p: Record<string, unknown>) => ["inv-backtest-detail", p],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchForecastModels: vi.fn().mockResolvedValue(["external", "lgbm_cluster"]),
  fetchInvBacktestSummary: vi.fn().mockResolvedValue({
    models: ["external", "lgbm_cluster"],
    excess_dos_threshold: 90,
    by_model: {
      external: {
        dfu_months: 5000,
        stockout_count: 150,
        stockout_rate: 3.0,
        excess_count: 400,
        excess_rate: 8.0,
        service_level: 97.0,
        avg_dos: 42.0,
        wape: 28.5,
        bias: 3.2,
      },
      lgbm_cluster: {
        dfu_months: 5000,
        stockout_count: 100,
        stockout_rate: 2.0,
        excess_count: 350,
        excess_rate: 7.0,
        service_level: 98.0,
        avg_dos: 38.5,
        wape: 22.1,
        bias: -1.5,
      },
    },
  }),
  fetchInvBacktestTrend: vi.fn().mockResolvedValue({
    trend: [
      {
        month: "2025-03-01",
        by_model: {
          external: { stockout_rate: 3.5, excess_rate: 8.0, avg_dos: 41, wape: 29 },
          lgbm_cluster: { stockout_rate: 2.5, excess_rate: 7.0, avg_dos: 38, wape: 22 },
        },
      },
    ],
  }),
  fetchInvBacktestRootCause: vi.fn().mockResolvedValue({
    model_id: "external",
    stockout_total: 450,
    stockout_under_forecast: 320,
    stockout_over_forecast: 80,
    stockout_exact: 50,
    excess_total: 1200,
    excess_over_forecast: 950,
    excess_under_forecast: 150,
    excess_exact: 100,
  }),
  fetchInvBacktestDetail: vi.fn().mockResolvedValue({
    total: 1,
    limit: 50,
    offset: 0,
    rows: [
      {
        item_no: "100320",
        loc: "1401-BULK",
        month: "2025-06-01",
        model_id: "lgbm_cluster",
        forecast: 120.5,
        actual_demand: 150.0,
        eom_qty_on_hand: 0,
        dos: null,
        event_type: "stockout",
        forecast_error: -29.5,
        pct_error: -19.7,
        bias_direction: "under",
      },
    ],
  }),
}));

const InvBacktestTab = (await import("@/tabs/InvBacktestTab")).default;

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

describe("InvBacktestTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvBacktestTab />
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
          <InvBacktestTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("Best Service Level")).toBeDefined();
      expect(screen.getByText("Lowest Stockout Rate")).toBeDefined();
      expect(screen.getByText("Lowest Excess Rate")).toBeDefined();
    });
  });

  it("renders model comparison chart", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvBacktestTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByTestId("composed-chart")).toBeDefined();
    });
  });

  it("renders filter controls", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvBacktestTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Filter item...")).toBeDefined();
      expect(screen.getByPlaceholderText("Filter location...")).toBeDefined();
      expect(screen.getByPlaceholderText("Filter cluster...")).toBeDefined();
    });
  });

  it("renders detail table headers", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvBacktestTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("100320")).toBeDefined();
      expect(screen.getByText("1401-BULK")).toBeDefined();
    });
  });

  it("renders root cause section", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InvBacktestTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/Root Cause/)).toBeDefined();
    });
  });
});
