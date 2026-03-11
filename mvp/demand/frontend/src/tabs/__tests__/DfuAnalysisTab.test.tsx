import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

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
}));

vi.mock("@/components/EChartContainer", () => ({
  EChartContainer: () => <div data-testid="chart-mock" />,
}));

const { DfuAnalysisTab } = await import("@/tabs/DfuAnalysisTab");

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

describe("DfuAnalysisTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <DfuAnalysisTab />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });
});
