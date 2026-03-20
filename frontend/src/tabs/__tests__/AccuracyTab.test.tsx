import { describe, it, expect, vi } from "vitest";
import { render, waitFor, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    forecastModels: () => ["forecast-models"],
    accuracySlice: (p: unknown) => ["accuracy-slice", p],
    lagCurve: (p: unknown) => ["lag-curve", p],
    competitionConfig: () => ["competition-config"],
    competitionSummary: () => ["competition-summary"],
    shapModels: () => ["shap-models"],
    shapSummary: (m: string, n: number) => ["shap-summary", m, n],
    shapTimeframes: (m: string) => ["shap-timeframes", m],
    shapTimeframeDetail: (m: string, i: number, n: number, c?: string) => ["shap-timeframe-detail", m, i, n, c ?? "all"],
    shapClusters: (m: string) => ["shap-clusters", m],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchForecastModels: vi.fn().mockResolvedValue(["external", "lgbm_global"]),
  fetchAccuracySlice: vi.fn().mockResolvedValue({
    group_by: "cluster_assignment",
    rows: [],
  }),
  fetchLagCurve: vi.fn().mockResolvedValue({ by_lag: [] }),
  fetchCompetitionConfig: vi.fn().mockResolvedValue(null),
  fetchCompetitionSummary: vi.fn().mockResolvedValue(null),
  fetchShapModels: vi.fn().mockResolvedValue({ models: [] }),
  fetchShapSummary: vi.fn().mockResolvedValue({ model_id: "lgbm_cluster", total_features: 0, features: [] }),
  fetchShapTimeframes: vi.fn().mockResolvedValue({ model_id: "lgbm_cluster", timeframes: [] }),
  fetchShapTimeframeDetail: vi.fn().mockResolvedValue({ model_id: "lgbm_cluster", timeframe_idx: 0, label: "A", cutoff_date: "2024-01-01", total_features: 0, features: [], cluster: "all", available_clusters: ["all"] }),
  fetchShapClusters: vi.fn().mockResolvedValue({ model_id: "lgbm_cluster", clusters: ["all"] }),
  saveCompetitionConfig: vi.fn(),
  runCompetition: vi.fn(),
  seasonalityProfileKeys: {
    list: () => ["seasonality-profiles"],
  },
  fetchSeasonalityProfileNames: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/api/queries/evolution", () => ({
  biasKeys: {
    summary: (p?: unknown) => ["bias-summary", p],
    flagged: (p?: unknown) => ["bias-flagged", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchBiasCorrectionSummary: vi.fn().mockResolvedValue({
    dfu_count: 0,
    avg_correction_factor: null,
    flagged_count: 0,
    clipped_count: 0,
    avg_rolling_bias: null,
    last_computed_at: null,
  }),
  fetchFlaggedBiasCorrections: vi.fn().mockResolvedValue({ total: 0, flagged: [] }),
}));

vi.mock("@/components/EChartContainer", () => ({
  EChartContainer: () => <div data-testid="chart-mock" />,
}));

const { AccuracyTab } = await import("@/tabs/AccuracyTab");

function makeFilterContext(): GlobalFilterContextValue {
  const filters: GlobalFilters = {
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
    filters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
    planningDate: null,
  };
}

describe("AccuracyTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <AccuracyTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders champion summary with DFU-months when data available", async () => {
    const { fetchCompetitionSummary } = await import("@/api/queries");
    (fetchCompetitionSummary as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      summary: {
        total_dfus: 100,
        total_dfu_months: 500,
        total_champion_rows: 5000,
        model_wins: { lgbm_global: 300, catboost_global: 200 },
        overall_champion_wape: 30.0,
        overall_champion_accuracy_pct: 70.0,
        run_ts: "2026-02-25T10:00:00Z",
        total_ceiling_rows: 6000,
        ceiling_model_wins: { lgbm_global: 350, catboost_global: 150 },
        overall_ceiling_wape: 20.0,
        overall_ceiling_accuracy_pct: 80.0,
      },
    });

    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <AccuracyTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/500/)).toBeDefined();
    });
  });

  it("renders before-the-fact label for champion model wins", async () => {
    const { fetchCompetitionSummary } = await import("@/api/queries");
    (fetchCompetitionSummary as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      summary: {
        total_dfus: 50,
        total_dfu_months: 200,
        total_champion_rows: 2000,
        model_wins: { lgbm_global: 200 },
        overall_champion_wape: 25.0,
        overall_champion_accuracy_pct: 75.0,
        run_ts: "2026-02-25T10:00:00Z",
      },
    });

    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <AccuracyTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/before-the-fact/)).toBeDefined();
    });
  });

  it("renders SHAP panel collapsed by default", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <AccuracyTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Feature Importance \(SHAP\)/)).toBeDefined();
    });

    // Panel body should not be visible by default (no shap model selector)
    expect(screen.queryByText(/All timeframes \(average\)/)).toBeNull();
  });

  it("includes brand/category/market in slice query when global filters are set", async () => {
    const { fetchAccuracySlice } = await import("@/api/queries");
    const mockSlice = fetchAccuracySlice as ReturnType<typeof vi.fn>;
    mockSlice.mockClear();

    const filtersWithBrand: GlobalFilters = {
      brand: ["BrandA", "BrandB"],
      category: ["CAT1"],
      market: ["NY"],
      channel: [],
      item: [],
      location: [],
      cluster: [],
      timeGrain: "month",
    };
    const ctx: GlobalFilterContextValue = {
      filters: filtersWithBrand,
      setFilters: vi.fn(),
      resetFilters: vi.fn(),
      hasActiveFilters: true,
      planningDate: null,
    };

    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={ctx}>
          <AccuracyTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(mockSlice).toHaveBeenCalled();
      const callArg = mockSlice.mock.calls[0][0] as Record<string, unknown>;
      expect(callArg.brand).toBe("BrandA,BrandB");
      expect(callArg.category).toBe("CAT1");
      expect(callArg.market).toBe("NY");
    });
  });

  it("renders SHAP panel with 'no outputs' message when models list is empty", async () => {
    const { fireEvent } = await import("@testing-library/react");

    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <AccuracyTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    // Click panel header to open it
    const panelHeader = await screen.findByText(/Feature Importance \(SHAP\)/);
    fireEvent.click(panelHeader.closest("[class]")!);

    await waitFor(() => {
      expect(screen.getByText(/No SHAP outputs found/)).toBeDefined();
    });
  });
});
