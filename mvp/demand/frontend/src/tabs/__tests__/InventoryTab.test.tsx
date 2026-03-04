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
    inventoryPosition: (p: Record<string, unknown>) => ["inventory-position", p],
    inventoryKpis: (p: Record<string, unknown>) => ["inventory-kpis", p],
    inventoryTrend: (p: Record<string, unknown>) => ["inventory-trend", p],
    inventoryItemDetail: (p: Record<string, unknown>) => ["inventory-item-detail", p],
    variabilitySummary: (p: Record<string, unknown>) => ["variability-summary", p],
    variabilityDetail: (p: Record<string, unknown>) => ["variability-detail", p],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchInventoryPosition: vi.fn().mockResolvedValue({
    total: 2,
    limit: 50,
    offset: 0,
    positions: [
      { item_no: "12345", loc: "LOC1", snapshot_date: "2025-06-15", lead_time_days: 30, qty_on_hand: 100, qty_on_hand_on_order: 150, qty_on_order: 50, mtd_sales: 25 },
      { item_no: "67890", loc: "LOC2", snapshot_date: "2025-06-15", lead_time_days: 45, qty_on_hand: 200, qty_on_hand_on_order: 250, qty_on_order: 50, mtd_sales: 40 },
    ],
  }),
  fetchInventoryKpis: vi.fn().mockResolvedValue({
    total_on_hand: 50000,
    total_on_order: 15000,
    avg_lead_time_days: 35.5,
    dos: 45.2,
    woc: 6.5,
    inventory_turns: 8.3,
    lt_coverage: 2.1,
    distinct_items: 500,
    distinct_locations: 50,
    months_covered: 3,
  }),
  fetchInventoryTrend: vi.fn().mockResolvedValue({
    trend: [
      { month: "2025-04-01", total_on_hand: 90000, total_on_order: 40000, monthly_sales: 200000, avg_lead_time: 30, dos: 42.5 },
      { month: "2025-05-01", total_on_hand: 95000, total_on_order: 45000, monthly_sales: 220000, avg_lead_time: 31, dos: 44.0 },
      { month: "2025-06-01", total_on_hand: 100000, total_on_order: 50000, monthly_sales: 250000, avg_lead_time: 32, dos: 45.2 },
    ],
  }),
  fetchInventoryItemDetail: vi.fn().mockResolvedValue({
    item: "12345",
    location: "LOC1",
    snapshots: [
      { item_no: "12345", loc: "LOC1", snapshot_date: "2025-06-15", lead_time_days: 30, qty_on_hand: 100, qty_on_hand_on_order: 150, qty_on_order: 50, mtd_sales: 25 },
    ],
  }),
  fetchVariabilitySummary: vi.fn().mockResolvedValue({
    total_dfus: 100,
    by_class: { low: 40, medium: 30, high: 20, lumpy: 10 },
    cv_percentiles: { p25: 0.15, p50: 0.35, p75: 0.70, p95: 1.20 },
    avg_cv: 0.45,
    avg_intermittency_ratio: 0.08,
    top_volatile: [],
  }),
  fetchVariabilityDetail: vi.fn().mockResolvedValue({
    total: 0,
    rows: [],
  }),
}));

const { InventoryTab } = await import("@/tabs/InventoryTab");

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

describe("InventoryTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InventoryTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders KPI cards with supply chain metrics", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InventoryTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("Total On-Hand")).toBeDefined();
      expect(screen.getByText("Total On-Order")).toBeDefined();
      expect(screen.getByText("Avg Lead Time")).toBeDefined();
      expect(screen.getByText("Days of Supply")).toBeDefined();
      expect(screen.getByText("Weeks of Cover")).toBeDefined();
      expect(screen.getByText("Inventory Turns")).toBeDefined();
      expect(screen.getByText("LT Coverage")).toBeDefined();
    });
  });

  it("renders inventory position table", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InventoryTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("12345")).toBeDefined();
      expect(screen.getByText("LOC1")).toBeDefined();
    });
  });

  it("renders trend chart", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InventoryTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("Monthly Inventory Trend")).toBeDefined();
    });
  });

  it("renders filter controls", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InventoryTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Filter by item...")).toBeDefined();
      expect(screen.getByPlaceholderText("Filter by location...")).toBeDefined();
    });
  });
});
