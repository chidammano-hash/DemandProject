import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

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
}));

vi.mock("@/components/EChartContainer", () => ({
  EChartContainer: () => <div data-testid="chart-mock" />,
}));

const { DfuAnalysisTab } = await import("@/tabs/DfuAnalysisTab");

describe("DfuAnalysisTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <DfuAnalysisTab theme="light" />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });
});
