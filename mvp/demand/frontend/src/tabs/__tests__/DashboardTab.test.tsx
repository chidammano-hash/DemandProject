import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import { ScenarioNotificationProvider } from "@/context/ScenarioNotificationContext";
import { JobNotificationProvider } from "@/context/JobNotificationContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

// Provide localStorage mock (useTheme depends on it)
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
    get length() { return Object.keys(store).length; },
    key: (i: number) => Object.keys(store)[i] ?? null,
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

// Mock ECharts
vi.mock("echarts-for-react/lib/core", () => ({
  default: ({ style, className }: any) => (
    <div data-testid="echarts-mock" style={style} className={className} />
  ),
}));

vi.mock("echarts/core", () => ({
  use: vi.fn(),
}));

vi.mock("echarts/charts", () => ({
  LineChart: {},
  BarChart: {},
}));

vi.mock("echarts/components", () => ({
  GridComponent: {},
  TooltipComponent: {},
  LegendComponent: {},
  DataZoomComponent: {},
}));

vi.mock("echarts/renderers", () => ({
  CanvasRenderer: {},
}));

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dashboardKpis: (p: Record<string, unknown>) => ["dashboard-kpis", p],
    dashboardAlerts: (p: Record<string, unknown>) => ["dashboard-alerts", p],
    dashboardTopMovers: (p: Record<string, unknown>) => ["dashboard-top-movers", p],
    dashboardHeatmap: (p: Record<string, unknown>) => ["dashboard-heatmap", p],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDashboardKpis: vi.fn().mockResolvedValue({
    accuracy_pct: 88.5,
    wape_pct: 11.2,
    bias_pct: -3.4,
    total_forecast: 500000,
    total_actual: 480000,
    weeks_of_supply: 4.2,
    window_months: 3,
    deltas: { accuracy_pct: 1.2, wape_pct: -0.5, bias_pct: 0.8 },
  }),
  fetchDashboardAlerts: vi.fn().mockResolvedValue({
    alerts: [
      { id: "a1", type: "oos_risk", severity: "critical", title: "OOS Risk", detail: "5 items at risk", count: 5 },
    ],
  }),
  fetchDashboardTopMovers: vi.fn().mockResolvedValue({
    movers: [
      { item_description: "Top Item", delta: 12000, pct_change: 15.0, direction: "up" },
    ],
  }),
  fetchDashboardHeatmap: vi.fn().mockResolvedValue({
    rows: [{ label: "Beverages", values: [90, 85, 78] }],
    period_labels: ["Dec", "Jan", "Feb"],
    metric: "accuracy_pct",
  }),
}));

const DashboardTab = (await import("@/tabs/DashboardTab")).default;

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

function renderDashboard() {
  return render(
    <TestQueryWrapper>
      <GlobalFilterProvider value={makeFilterContext()}>
        <ScenarioNotificationProvider>
          <JobNotificationProvider>
            <DashboardTab />
          </JobNotificationProvider>
        </ScenarioNotificationProvider>
      </GlobalFilterProvider>
    </TestQueryWrapper>
  );
}

describe("DashboardTab", () => {
  beforeEach(() => {
    localStorageMock.clear();
    document.documentElement.classList.remove("light", "dark");
    document.documentElement.removeAttribute("data-transitioning");
    document.documentElement.removeAttribute("data-theme");
  });

  it("renders without crashing when wrapped with providers", async () => {
    renderDashboard();
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("shows loading skeletons initially", () => {
    const { container } = renderDashboard();
    // Skeleton elements have animate-pulse class
    const skeletons = container.querySelectorAll(".animate-pulse");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("renders KPI cards after data loads", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("Accuracy")).toBeDefined();
      expect(getByText("WAPE")).toBeDefined();
      expect(getByText("Bias")).toBeDefined();
      expect(getByText("Total Forecast")).toBeDefined();
      expect(getByText("Total Actual")).toBeDefined();
    });
  });

  it("renders widget cards with titles after data loads", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("Alerts")).toBeDefined();
      expect(getByText("Performance Heatmap")).toBeDefined();
      expect(getByText("Top Movers")).toBeDefined();
      expect(getByText("Forecast vs Actual Trend")).toBeDefined();
    });
  });
});
