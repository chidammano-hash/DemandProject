import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// Hoist mock functions before vi.mock() hoisting
const { mockFetchSummary, mockFetchDetail, mockFetchComparison, mockFetchDfu } = vi.hoisted(() => ({
  mockFetchSummary: vi.fn(),
  mockFetchDetail: vi.fn(),
  mockFetchComparison: vi.fn(),
  mockFetchDfu: vi.fn(),
}));

vi.mock("recharts");

const mockSummary = {
  plan_version: "2026-03-01",
  total_skus: 1200,
  below_ss_count: 45,
  below_ss_pct: 3.75,
  avg_ss: 220.5,
  avg_eoq: 310.0,
  avg_ss_delta_pct: 8.2,
  by_policy_type: [
    { policy_type: "continuous_rop", sku_count: 800, avg_ss: 250.0, avg_eoq: 320.0, total_order_qty: 180000 },
  ],
};

const mockDetail = {
  total: 1200,
  limit: 50,
  offset: 0,
  rows: [
    {
      item_id: "100320",
      loc: "1401-BULK",
      plan_month: "2026-03-01",
      abc_vol: "A",
      policy_type: "continuous_rop",
      forecast_qty: 500,
      ss_combined: 220,
      historical_ss: 200,
      ss_delta: 20,
      ss_delta_pct: 10.0,
      eoq: 310,
      cycle_stock: 155,
      reorder_point: 375,
      order_qty: 310,
      order_up_to_level: 530,
      is_below_ss: false,
    },
    {
      item_id: "200450",
      loc: "2002-DC",
      plan_month: "2026-03-01",
      abc_vol: "C",
      policy_type: "min_max",
      forecast_qty: 80,
      ss_combined: 30,
      historical_ss: 35,
      ss_delta: -5,
      ss_delta_pct: -14.3,
      eoq: 60,
      cycle_stock: 30,
      reorder_point: 65,
      order_qty: 60,
      order_up_to_level: 110,
      is_below_ss: true,
    },
  ],
};

const mockComparison = {
  plan_version: "2026-03-01",
  total_increased: 350,
  total_decreased: 180,
  by_abc: [
    {
      abc_vol: "A",
      sku_count: 300,
      avg_forecast_ss: 250.0,
      avg_historical_ss: 230.0,
      avg_ss_delta: 20.0,
      avg_ss_delta_pct: 8.7,
      count_increased: 200,
      count_decreased: 80,
      count_unchanged: 20,
    },
    {
      abc_vol: "B",
      sku_count: 600,
      avg_forecast_ss: 210.0,
      avg_historical_ss: 195.0,
      avg_ss_delta: 15.0,
      avg_ss_delta_pct: 7.7,
      count_increased: 150,
      count_decreased: 100,
      count_unchanged: 350,
    },
  ],
};

const mockDfu = {
  item_id: "100320",
  loc: "1401-BULK",
  plan_version: "2026-03-01",
  series: [
    {
      plan_month: "2026-03-01",
      horizon_months: 1,
      forecast_qty: 500,
      forecast_qty_lower: 420,
      forecast_qty_upper: 580,
      ss_combined: 220,
      historical_ss: 200,
      ss_delta: 20,
      eoq: 310,
      cycle_stock: 155,
      reorder_point: 375,
      order_qty: 310,
      order_up_to_level: 530,
      avg_daily_demand: 16.7,
      is_below_ss: false,
      sigma_method: "z_score",
    },
  ],
};

vi.mock("@/api/queries", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/queries")>();
  return {
    ...actual,
    replenishmentKeys: {
      summary: (pv?: string, pt?: string, av?: string) =>
        ["replenishment", "summary", pv, pt, av] as const,
      detail: (params: object) => ["replenishment", "detail", params] as const,
      comparison: (pv?: string, av?: string, pt?: string) =>
        ["replenishment", "comparison", pv, av, pt] as const,
      sku: (itemNo: string, loc: string, pv?: string) =>
        ["replenishment", "sku", itemNo, loc, pv] as const,
    },
    fetchReplenishmentSummary: mockFetchSummary,
    fetchReplenishmentDetail: mockFetchDetail,
    fetchReplenishmentComparison: mockFetchComparison,
    fetchReplenishmentSku: mockFetchDfu,
  };
});

import { ReplenishmentPlanPanel } from "@/tabs/inv-planning/ReplenishmentPlanPanel";

describe("ReplenishmentPlanPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchSummary.mockResolvedValue(mockSummary);
    mockFetchDetail.mockResolvedValue(mockDetail);
    mockFetchComparison.mockResolvedValue(mockComparison);
    mockFetchDfu.mockResolvedValue(mockDfu);
  });

  it("renders KPI cards with summary data", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Total SKUs")).toBeDefined();
    expect(await screen.findByText("Avg Forward SS")).toBeDefined();
    expect(await screen.findByText("Avg EOQ")).toBeDefined();
    // "Below SS" appears in both KPI card and table header
    expect((await screen.findAllByText("Below SS")).length).toBeGreaterThan(0);
    // Check that the total DFU count renders
    expect(await screen.findByText("1,200")).toBeDefined();
  });

  it("renders loading state before data arrives", () => {
    mockFetchSummary.mockReturnValue(new Promise(() => {}));
    mockFetchDetail.mockReturnValue(new Promise(() => {}));
    mockFetchComparison.mockReturnValue(new Promise(() => {}));

    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    // KPI cards show loading placeholder
    const loadingTexts = screen.getAllByText("...");
    expect(loadingTexts.length).toBeGreaterThan(0);
  });

  it("renders comparison chart when data is available", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    expect(
      await screen.findByText("SS Comparison: Forecast vs Historical by ABC Class")
    ).toBeDefined();
    expect(screen.getByTestId("bar-chart")).toBeDefined();
  });

  it("renders detail table rows", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    // Item numbers appear in table
    expect(await screen.findByText("100320")).toBeDefined();
    expect(await screen.findByText("200450")).toBeDefined();
    // Below SS badge
    expect(await screen.findByText("YES")).toBeDefined();
  });

  it("renders pagination when total exceeds page size", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    // total=1200, PAGE_SIZE=50 → multiple pages
    expect(await screen.findByText("Prev")).toBeDefined();
    expect(await screen.findByText("Next")).toBeDefined();
    expect(await screen.findByText(/Page 1/)).toBeDefined();
  });

  it("renders SKU drill-down when a row is clicked", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    // Wait for rows to render
    const itemCell = await screen.findByText("100320");
    // Click the row
    fireEvent.click(itemCell.closest("tr")!);
    // Drill-down header should appear
    expect(
      await screen.findByText(/SKU Drill-Down: 100320 @ 1401-BULK/)
    ).toBeDefined();
  });

  it("closes SKU drill-down on close button click", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    const itemCell = await screen.findByText("100320");
    fireEvent.click(itemCell.closest("tr")!);
    // Drill-down is open
    const closeBtn = await screen.findByText("✕ close");
    fireEvent.click(closeBtn);
    // Drill-down header should be gone
    expect(
      screen.queryByText(/SKU Drill-Down: 100320 @ 1401-BULK/)
    ).toBeNull();
  });

  it("renders empty state when no detail rows", async () => {
    mockFetchDetail.mockResolvedValue({ total: 0, limit: 50, offset: 0, rows: [] });
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("No forward replenishment plan")).toBeDefined();
  });

  it("renders policy type filter dropdown", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Policy Type")).toBeDefined();
    expect(screen.getByText("Continuous ROP")).toBeDefined();
    expect(screen.getByText("Min/Max")).toBeDefined();
    expect(screen.getByText("Periodic Review")).toBeDefined();
  });

  it("renders ABC class filter dropdown", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("ABC Class")).toBeDefined();
    // There should be A, B, C options in the ABC select
    const abcSelect = screen.getAllByRole("combobox")[1]; // second select is ABC
    expect(abcSelect).toBeDefined();
  });

  it("renders replenishment detail section heading", async () => {
    render(
      <TestQueryWrapper>
        <ReplenishmentPlanPanel />
      </TestQueryWrapper>
    );
    expect(
      await screen.findByText("Replenishment Plan Detail")
    ).toBeDefined();
  });
});
