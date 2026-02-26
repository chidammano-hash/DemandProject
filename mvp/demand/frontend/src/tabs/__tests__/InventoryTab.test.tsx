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
    total_inventory_value: null,
    avg_lead_time_days: 35.5,
    distinct_items: 500,
    distinct_locations: 50,
    snapshot_count: 10000,
    months_covered: 3,
  }),
  fetchInventoryTrend: vi.fn().mockResolvedValue({
    trend: [
      { month: "2025-04-01", avg_on_hand: 90, avg_on_order: 40, avg_lead_time: 30, total_mtd_sales: 200 },
      { month: "2025-05-01", avg_on_hand: 95, avg_on_order: 45, avg_lead_time: 31, total_mtd_sales: 220 },
      { month: "2025-06-01", avg_on_hand: 100, avg_on_order: 50, avg_lead_time: 32, total_mtd_sales: 250 },
    ],
  }),
  fetchInventoryItemDetail: vi.fn().mockResolvedValue({
    item: "12345",
    location: "LOC1",
    snapshots: [
      { item_no: "12345", loc: "LOC1", snapshot_date: "2025-06-15", lead_time_days: 30, qty_on_hand: 100, qty_on_hand_on_order: 150, qty_on_order: 50, mtd_sales: 25 },
    ],
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
          <InventoryTab theme="light" />
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
          <InventoryTab theme="light" />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("Total On-Hand")).toBeDefined();
      expect(screen.getByText("Total On-Order")).toBeDefined();
      expect(screen.getByText("Avg Lead Time")).toBeDefined();
    });
  });

  it("renders inventory position table", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <InventoryTab theme="light" />
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
          <InventoryTab theme="light" />
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
          <InventoryTab theme="light" />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Filter by item...")).toBeDefined();
      expect(screen.getByPlaceholderText("Filter by location...")).toBeDefined();
    });
  });
});
