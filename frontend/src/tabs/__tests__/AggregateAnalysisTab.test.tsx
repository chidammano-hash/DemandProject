import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// localStorage mock
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

// Mock recharts — uses shared mock at frontend/__mocks__/recharts.tsx
vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dashboardKpis: (p: Record<string, unknown>) => ["dashboard-kpis", p],
    dashboardTrend: (p: Record<string, unknown>) => ["dashboard-trend", p],
    dashboardHeatmap: (p: Record<string, unknown>) => ["dashboard-heatmap", p],
    accuracySlice: (p: Record<string, unknown>) => ["accuracy-slice", p],
    lagCurve: (p: Record<string, unknown>) => ["lag-curve", p],
    competitionConfig: () => ["competition-config"],
    competitionSummary: () => ["competition-summary"],
    shapModels: () => ["shap-models"],
    shapSummary: (m: string, n: number) => ["shap-summary", m, n],
    shapTimeframes: (m: string) => ["shap-timeframes", m],
    shapTimeframeDetail: (m: string, idx: number, n: number, c: string) => ["shap-detail", m, idx, n, c],
    shapClusters: (m: string) => ["shap-clusters", m],
    distinctValues: (d: string, c: string) => ["distinct-values", d, c],
    planningDate: () => ["planning-date"],
    forecastModels: () => ["forecast-models"],
  },
  filterMetaKeys: {
    skuCount: (f: unknown) => ["sku-count", f],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDashboardKpis: vi.fn().mockResolvedValue({
    accuracy_pct: 88.5, wape_pct: 11.2, bias_pct: -3.4,
    accuracy_delta: 1.2, wape_delta: -0.5, bias_delta: 0.8,
    total_forecast: 500000, total_actual: 480000, window_months: 3,
  }),
  fetchDashboardTrend: vi.fn().mockResolvedValue({
    months: [
      { month: "2025-10", forecast: 120000, actual: 115000 },
      { month: "2025-11", forecast: 125000, actual: 120000 },
    ],
  }),
  fetchDashboardHeatmap: vi.fn().mockResolvedValue({
    rows: [{ label: "Category A", values: [92, 85, 78] }],
    period_labels: ["2025-09", "2025-10", "2025-11"],
    metric: "accuracy_pct",
  }),
  fetchAccuracySlice: vi.fn().mockResolvedValue({ rows: [], common_sku_count: null, sku_counts: null }),
  fetchLagCurve: vi.fn().mockResolvedValue({ by_lag: [] }),
  fetchCompetitionConfig: vi.fn().mockResolvedValue({ config: null, available_models: [] }),
  fetchCompetitionSummary: vi.fn().mockResolvedValue({ summary: null }),
  fetchShapModels: vi.fn().mockResolvedValue({ models: [] }),
  fetchShapSummary: vi.fn().mockResolvedValue({ model_id: "", total_features: 0, features: [] }),
  fetchShapTimeframes: vi.fn().mockResolvedValue({ timeframes: [] }),
  fetchShapTimeframeDetail: vi.fn().mockResolvedValue({ features: [] }),
  fetchShapClusters: vi.fn().mockResolvedValue({ clusters: [] }),
  fetchSeasonalityProfileNames: vi.fn().mockResolvedValue([]),
  saveCompetitionConfig: vi.fn().mockResolvedValue({}),
  runCompetition: vi.fn().mockResolvedValue({}),
  fetchDistinctValues: vi.fn().mockResolvedValue({ values: [] }),
  fetchPlanningDate: vi.fn().mockResolvedValue({ planning_date: "2026-03-16", is_frozen: false, system_date: "2026-03-16", days_behind: 0 }),
  fetchDfuCount: vi.fn().mockResolvedValue({ count: 0 }),
  fetchForecastModels: vi.fn().mockResolvedValue(["external", "lgbm_cluster", "champion"]),
}));

vi.mock("@/api/queries/core", () => ({
  fetchSkuShap: vi.fn().mockResolvedValue(null),
  fetchShapSummary: vi.fn().mockResolvedValue({ model_id: "", total_features: 0, features: [] }),
}));

vi.mock("@/components/EChartContainer", () => ({
  EChartContainer: () => <div data-testid="chart-mock" />,
}));

// Mock evolution queries used by BiasCorrectionsPanel
vi.mock("@/api/queries/evolution", () => ({
  biasKeys: { corrections: (p: unknown) => ["bias-corr", p] },
  STALE_EVO: { FIVE_MIN: 300000 },
  fetchBiasCorrections: vi.fn().mockResolvedValue({ corrections: [], total: 0 }),
}));

const { AggregateAnalysisTab } = await import("@/tabs/AggregateAnalysisTab");

function renderTab() {
  return render(
    <TestQueryWrapper>
      <AggregateAnalysisTab />
    </TestQueryWrapper>,
  );
}

describe("AggregateAnalysisTab", () => {
  beforeEach(() => {
    try { localStorage.removeItem("ds:aggregateAnalysis:panels"); } catch { /* no-op */ }
  });

  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders Portfolio Analysis heading", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Portfolio Analysis")).toBeDefined();
    });
  });

  it("renders local filter bar with Brand, Category, Item, Location buttons", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "Brand" }).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByRole("button", { name: "Category" }).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders toggle toolbar with panel checkboxes", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getAllByText("KPIs").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Forecast vs Actual").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Heatmap")).toBeDefined();
      expect(screen.getAllByText("Accuracy").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders KPI cards with forecast performance metrics", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getAllByText("Accuracy %").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("WAPE %").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Bias %").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Forecast Vol")).toBeDefined();
      expect(screen.getByText("Actual Vol")).toBeDefined();
    });
  });

  it("renders Forecast vs Actual chart panel", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getAllByText("Forecast vs Actual").length).toBeGreaterThanOrEqual(1);
    });
    // Window selector buttons
    expect(screen.getAllByText("6mo").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("12mo").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("24mo").length).toBeGreaterThanOrEqual(1);
  });

  it("renders Accuracy Heatmap panel with grain selector", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Accuracy Heatmap")).toBeDefined();
    });
    expect(screen.getAllByText("Category").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Brand").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Location").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Class").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Sub-class").length).toBeGreaterThanOrEqual(1);
    // Verify both axis selectors exist (Rows + Columns labels)
    expect(screen.getByText("Rows")).toBeDefined();
    expect(screen.getByText("Columns")).toBeDefined();
  });

  it("renders Accuracy Comparison card when panel is on", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Accuracy Comparison")).toBeDefined();
    });
  });

  it("renders Mo/Qtr time grain toggle", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Mo")).toBeDefined();
      expect(screen.getByText("Qtr")).toBeDefined();
    });
  });

  it("renders a labeled, friendly-formatted planning date badge (U2.5)", async () => {
    renderTab();
    await waitFor(() => {
      // Must match the app-wide "Plan as of Mon D, YYYY" convention, not a bare
      // ISO "2026-03-16" that reads ambiguously as a filter or last-refresh.
      expect(screen.getByText("Plan as of Mar 16, 2026")).toBeDefined();
    });
    expect(screen.queryByText("2026-03-16")).toBeNull();
  });
});
