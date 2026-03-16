import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
  ReferenceLine: () => null,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
}));

vi.mock("@/api/queries/core", () => ({
  fetchDfuShap: vi.fn().mockResolvedValue(null),
  fetchShapSummary: vi.fn().mockResolvedValue({ model_id: "lgbm_cluster", total_features: 0, features: [] }),
}));

vi.mock("@/api/queries/production-forecast", () => ({
  fetchProductionForecast: vi.fn().mockResolvedValue(null),
}));

vi.mock("@/api/queries", () => ({
  queryKeys: {
    samplePair: (d: string) => ["sample-pair", d],
    dfuAnalysis: (p: unknown) => ["dfu-analysis", p],
    forecastModels: () => ["forecast-models"],
    inventoryPosition: (p: Record<string, unknown>) => ["inventory-position", p],
    inventoryKpis: (p: Record<string, unknown>) => ["inventory-kpis", p],
    inventoryTrend: (p: Record<string, unknown>) => ["inventory-trend", p],
    inventoryItemDetail: (p: Record<string, unknown>) => ["inventory-item-detail", p],
    variabilitySummary: (p: Record<string, unknown>) => ["variability-summary", p],
    variabilityDetail: (p: Record<string, unknown>) => ["variability-detail", p],
    ltSummary: (p: Record<string, unknown>) => ["lt-summary", p],
    ltProfile: (p: Record<string, unknown>) => ["lt-profile", p],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchSamplePair: vi.fn().mockResolvedValue({ item: "100320", location: "1401-BULK" }),
  fetchDfuAnalysis: vi.fn().mockResolvedValue({
    mode: "item_location",
    item: "100320",
    location: "1401-BULK",
    points: 0,
    models: [],
    series: [],
    model_monthly: {},
    dfu_attributes: [],
  }),
  fetchForecastModels: vi.fn().mockResolvedValue(["external"]),
  fetchSeasonalityProfileNames: vi.fn().mockResolvedValue([]),
  fetchInventoryPosition: vi.fn().mockResolvedValue({
    total: 1,
    limit: 50,
    offset: 0,
    positions: [
      { item_no: "100320", loc: "1401-BULK", snapshot_date: "2025-06-15", lead_time_days: 30, qty_on_hand: 100, qty_on_hand_on_order: 150, qty_on_order: 50, mtd_sales: 25 },
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
      { month: "2025-06-01", total_on_hand: 100000, total_on_order: 50000, monthly_sales: 250000, avg_lead_time: 32, dos: 45.2 },
    ],
  }),
  fetchInventoryItemDetail: vi.fn().mockResolvedValue({
    item: "100320",
    location: "1401-BULK",
    snapshots: [],
  }),
  fetchVariabilitySummary: vi.fn().mockResolvedValue({ total_dfus: 0, by_class: {}, cv_percentiles: {}, avg_cv: 0, avg_intermittency_ratio: 0, top_volatile: [] }),
  fetchVariabilityDetail: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
  fetchLtSummary: vi.fn().mockResolvedValue({ total_profiles: 0, by_class: {}, avg_lt_cv: 0, avg_lt_mean_days: 0, lt_cv_p50: 0, lt_cv_p95: 0, top_volatile: [] }),
  fetchLtProfile: vi.fn().mockResolvedValue({ total: 0, rows: [] }),
}));

vi.mock("@/components/EChartContainer", () => ({
  EChartContainer: () => <div data-testid="chart-mock" />,
}));

const { ItemAnalysisTab } = await import("@/tabs/ItemAnalysisTab");

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
    planningDate: null,
  };
}

function renderTab() {
  return render(
    <TestQueryWrapper>
      <GlobalFilterProvider value={makeFilterContext()}>
        <ItemAnalysisTab />
      </GlobalFilterProvider>
    </TestQueryWrapper>,
  );
}

// Provide localStorage mock if not present in jsdom
const _store: Record<string, string> = {};
if (typeof globalThis.localStorage === "undefined" || typeof globalThis.localStorage.getItem !== "function") {
  Object.defineProperty(globalThis, "localStorage", {
    value: {
      getItem: (key: string) => _store[key] ?? null,
      setItem: (key: string, value: string) => { _store[key] = value; },
      removeItem: (key: string) => { delete _store[key]; },
      clear: () => { for (const k of Object.keys(_store)) delete _store[k]; },
    },
    writable: true,
  });
}

describe("ItemAnalysisTab", () => {
  beforeEach(() => {
    try { localStorage.removeItem("ds:itemAnalysis:panels"); } catch { /* no-op */ }
  });

  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders toggle toolbar with Demand and Supply sections", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Demand")).toBeDefined();
      expect(screen.getByText("Supply")).toBeDefined();
    });
  });

  it("renders toggle checkboxes for all panels", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByLabelText("Toggle Chart")).toBeDefined();
      expect(screen.getByLabelText("Toggle SHAP")).toBeDefined();
      expect(screen.getByLabelText("Toggle Model KPIs")).toBeDefined();
      expect(screen.getByLabelText("Toggle Inv KPIs")).toBeDefined();
      expect(screen.getByLabelText("Toggle Position Table")).toBeDefined();
      expect(screen.getByLabelText("Toggle Variability")).toBeDefined();
      expect(screen.getByLabelText("Toggle Lead Time")).toBeDefined();
    });
  });

  it("renders inventory KPI cards when panel is enabled", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getAllByText("Total On-Hand").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Days of Supply").length).toBeGreaterThan(0);
    });
  });

  it("hides inventory KPIs when toggled off", async () => {
    renderTab();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getAllByText("Total On-Hand").length).toBeGreaterThan(0);
    });

    const checkbox = screen.getByLabelText("Toggle Inv KPIs");
    await user.click(checkbox);

    await waitFor(() => {
      expect(screen.queryByText("Total On-Hand")).toBeNull();
    });
  });

  it("renders scope mode selector from DFU analysis", async () => {
    renderTab();
    await waitFor(() => {
      // SelectorPanel renders the scope mode dropdown
      expect(document.body.textContent).toContain("Scope");
    });
  });
});
