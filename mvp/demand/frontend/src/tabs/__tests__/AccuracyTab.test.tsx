import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    forecastModels: () => ["forecast-models"],
    accuracySlice: (p: unknown) => ["accuracy-slice", p],
    lagCurve: (p: unknown) => ["lag-curve", p],
    competitionConfig: () => ["competition-config"],
    competitionSummary: () => ["competition-summary"],
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
  saveCompetitionConfig: vi.fn(),
  runCompetition: vi.fn(),
}));

vi.mock("@/components/EChartContainer", () => ({
  EChartContainer: () => <div data-testid="chart-mock" />,
}));

const { AccuracyTab } = await import("@/tabs/AccuracyTab");

describe("AccuracyTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyTab theme="light" />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });
});
