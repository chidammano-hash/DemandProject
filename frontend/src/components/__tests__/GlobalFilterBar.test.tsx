import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    distinctValues: (domain: string, column: string) => ["distinct-values", domain, column],
    planningDate: () => ["planning-date"],
  },
  filterMetaKeys: {
    skuCount: () => ["sku-count", {}],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDistinctValues: vi.fn().mockResolvedValue({ column: "brand", values: ["BrandA", "BrandB"], total: 2 }),
  fetchPlanningDate: vi.fn().mockResolvedValue({ planning_date: "2026-02-24", use_system_date: false }),
  fetchDfuCount: vi.fn().mockResolvedValue({ count: 0 }),
}));

const { GlobalFilterBar } = await import("@/components/GlobalFilterBar");

function makeContextValue(overrides: Partial<GlobalFilterContextValue> = {}): GlobalFilterContextValue {
  const defaultFilters: GlobalFilters = {
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
    filters: defaultFilters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
    ...overrides,
  };
}

function renderWithContext(contextValue: GlobalFilterContextValue) {
  return render(
    <TestQueryWrapper>
      <GlobalFilterProvider value={contextValue}>
        <GlobalFilterBar />
      </GlobalFilterProvider>
    </TestQueryWrapper>
  );
}

describe("GlobalFilterBar", () => {
  it("renders filter dropdowns and time grain toggle", () => {
    renderWithContext(makeContextValue());
    // 7 filter dropdown buttons (Brand, Category, Item, Location, Market, Channel, Cluster) + 2 time grain buttons (Mo, Qtr)
    expect(screen.getByText("Brand")).toBeInTheDocument();
    expect(screen.getByText("Category")).toBeInTheDocument();
    expect(screen.getByText("Item")).toBeInTheDocument();
    expect(screen.getByText("Location")).toBeInTheDocument();
    expect(screen.getByText("Market")).toBeInTheDocument();
    expect(screen.getByText("Channel")).toBeInTheDocument();
    expect(screen.getByText("Cluster")).toBeInTheDocument();
    expect(screen.getByText("Mo")).toBeInTheDocument();
    expect(screen.getByText("Qtr")).toBeInTheDocument();
  });

  it("renders toolbar with accessible label", () => {
    renderWithContext(makeContextValue());
    expect(screen.getByRole("toolbar", { name: "Global filters" })).toBeInTheDocument();
  });

  it("reset button shows when filters are active", () => {
    const ctx = makeContextValue({ hasActiveFilters: true });
    renderWithContext(ctx);
    expect(screen.getByText("Reset")).toBeInTheDocument();
  });

  it("reset button does NOT show when no filters are active", () => {
    const ctx = makeContextValue({ hasActiveFilters: false });
    renderWithContext(ctx);
    expect(screen.queryByText("Reset")).not.toBeInTheDocument();
  });

  it("clicking reset calls resetFilters", () => {
    const resetFilters = vi.fn();
    const ctx = makeContextValue({ hasActiveFilters: true, resetFilters });
    renderWithContext(ctx);
    fireEvent.click(screen.getByText("Reset"));
    expect(resetFilters).toHaveBeenCalledTimes(1);
  });

  it("uses GlobalFilterContext - displays selected filter count", () => {
    const ctx = makeContextValue({
      filters: {
        brand: ["BrandA", "BrandB"],
        category: [],
        market: [],
        channel: [],
        item: [],
        location: [],
        cluster: [],
        timeGrain: "month",
      },
    });
    renderWithContext(ctx);
    // When 2 brands are selected, label becomes "Brand (2)"
    expect(screen.getByText("Brand (2)")).toBeInTheDocument();
  });

  it("clicking time grain toggle calls setFilters with quarter", () => {
    const setFilters = vi.fn();
    const ctx = makeContextValue({ setFilters });
    renderWithContext(ctx);
    fireEvent.click(screen.getByText("Qtr"));
    expect(setFilters).toHaveBeenCalledWith({ timeGrain: "quarter" });
  });
});
