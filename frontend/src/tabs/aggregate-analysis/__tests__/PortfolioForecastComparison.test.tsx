import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const apiMocks = vi.hoisted(() => ({
  fetchLatestCustomerBlend: vi.fn(),
  fetchCustomerBlendTrend: vi.fn(),
}));

vi.mock("@/api/queries", () => ({
  customerForecastKeys: {
    latestBlend: ["customer-forecast", "blend", "latest"],
    blendTrend: (filters: Record<string, unknown>) => [
      "customer-forecast",
      "blend",
      "trend",
      filters,
    ],
  },
  STALE: { THIRTY_SEC: 30_000 },
  fetchLatestCustomerBlend: apiMocks.fetchLatestCustomerBlend,
  fetchCustomerBlendTrend: apiMocks.fetchCustomerBlendTrend,
}));

vi.mock("@/components/ForecastTrendChart", () => ({
  ForecastTrendChart: ({ data }: { data: unknown[] }) => <div>Standard trend: {data.length}</div>,
}));

vi.mock("@/components/CustomerForecastTrendChart", () => ({
  CustomerForecastTrendChart: ({ trend }: { trend: { run_id: string } }) => (
    <div>Customer trend: {trend.run_id}</div>
  ),
}));

const { PortfolioForecastComparison } = await import("../PortfolioForecastComparison");

const trend = {
  run_id: "blend-run-1",
  status: "ready",
  planning_month: "2026-07-01",
  completed_at: "2026-07-14T12:00:00Z",
  backtest_run_id: "backtest-run-1",
  bottom_up_staging_run_id: "shadow-run-1",
  backtest_gate: { passed: true },
  filters_applied: {},
  filter_notes: [],
  accuracy: {
    common_actual_qty: 100,
    customer_bottom_up_wape_pct: 10,
    source_champion_wape_pct: 5,
    customer_blend_wape_pct: 2.5,
  },
  coverage: {
    blended_rows: 1,
    champion_fallback_rows: 0,
    global_customer_only_excluded_count: 0,
  },
  months: [],
};

function renderComparison(onTrendWindowChange = vi.fn()) {
  render(
    <TestQueryWrapper>
      <PortfolioForecastComparison
        kpiModel="champion"
        trendWindow={12}
        onTrendWindowChange={onTrendWindowChange}
        dashboardFilters={{
          brand: ["Brand A"],
          category: [],
          market: [],
          channel: [],
          item: ["ITEM-1"],
          location: ["LOC-1"],
          cluster: ["seasonal"],
          time_grain: "month",
        }}
        standardMonths={[{ month: "2026-06", forecast: 90, actual: 100 }]}
        standardLoading={false}
      />
    </TestQueryWrapper>
  );
}

describe("PortfolioForecastComparison", () => {
  beforeEach(() => {
    apiMocks.fetchLatestCustomerBlend.mockReset().mockResolvedValue({
      run_id: "blend-run-1",
      status: "ready",
      planning_month: "2026-07-01",
    });
    apiMocks.fetchCustomerBlendTrend.mockReset().mockResolvedValue(trend);
  });

  it("switches from the standard timeline to the exact customer blend run", async () => {
    renderComparison();
    expect(screen.getByText("Standard trend: 1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Customer Blend" }));

    expect(await screen.findByText("Customer trend: blend-run-1")).toBeInTheDocument();
    expect(apiMocks.fetchCustomerBlendTrend).toHaveBeenCalledWith(
      expect.objectContaining({
        run_id: "blend-run-1",
        item_id: ["ITEM-1"],
        location_id: ["LOC-1"],
        cluster: ["seasonal"],
      })
    );
  });

  it("changes the shared history window from the accessible selector", () => {
    const onTrendWindowChange = vi.fn();
    renderComparison(onTrendWindowChange);

    fireEvent.click(screen.getByRole("button", { name: "24mo" }));

    expect(onTrendWindowChange).toHaveBeenCalledWith(24);
  });

  it("shows a stable empty state when no current blend exists", async () => {
    apiMocks.fetchLatestCustomerBlend.mockResolvedValue(null);
    renderComparison();

    fireEvent.click(screen.getByRole("button", { name: "Customer Blend" }));

    expect(await screen.findByText("No current customer blend is available.")).toBeInTheDocument();
    await waitFor(() => expect(apiMocks.fetchCustomerBlendTrend).not.toHaveBeenCalled());
  });
});
