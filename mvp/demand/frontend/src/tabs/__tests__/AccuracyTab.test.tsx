import { describe, it, expect, vi } from "vitest";
import { render, waitFor, screen } from "@testing-library/react";
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
        <AccuracyTab theme="light" />
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
        <AccuracyTab theme="light" />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/before-the-fact/)).toBeDefined();
    });
  });
});
