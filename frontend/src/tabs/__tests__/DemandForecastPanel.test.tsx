import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual<typeof import("@/api/queries")>("@/api/queries");
  return {
    ...actual,
    queryKeys: {
      ...actual.queryKeys,
      productionForecastVersions: () => ["pf-versions"],
      productionForecastSummary: (p?: unknown) => ["pf-summary", p],
      productionForecast: (p?: unknown) => ["pf-detail", p],
    },
    STALE: { FIVE_MIN: 300000, TEN_MIN: 600000, ONE_MIN: 60000, TWO_MIN: 120000 },
    fetchProductionForecastVersions: vi.fn().mockResolvedValue({
      versions: [
        {
          plan_version: "v2026-03",
          dfu_count: 500,
          total_rows: 9000,
          generated_at: "2026-03-01T00:00:00Z",
        },
      ],
    }),
    fetchProductionForecastSummary: vi.fn().mockResolvedValue({
      plan_version: "v2026-03",
      horizon_months: 18,
      generated_at: "2026-03-01T00:00:00Z",
      total_dfu_count: 500,
      total_forecast_qty: 125000,
      by_abc_class: [
        { abc_class: "A", forecast_qty: 80000, dfu_count: 100 },
        { abc_class: "B", forecast_qty: 35000, dfu_count: 200 },
      ],
      ci_coverage_pct: 100,
      avg_ci_width: 25,
    }),
    fetchProductionForecast: vi.fn().mockResolvedValue({
      model_id: "lgbm_cluster",
      plan_version: "v2026-03",
      generated_at: "2026-03-01T00:00:00Z",
      is_recursive: false,
      forecasts: [],
    }),
  };
});

import { DemandForecastPanel } from "@/tabs/inv-planning/DemandForecastPanel";

describe("DemandForecastPanel", () => {
  it("renders KPI cards when summary loads", async () => {
    render(
      <TestQueryWrapper>
        <DemandForecastPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Forecast Version")).toBeDefined();
    expect(await screen.findByText("DFUs Planned")).toBeDefined();
    expect(await screen.findByText(/500 DFUs/)).toBeDefined();
  });

  it("renders DFU Forecast Series section", async () => {
    render(
      <TestQueryWrapper>
        <DemandForecastPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("DFU Forecast Series")).toBeDefined();
  });

  it("renders Forecast by ABC Class chart when data available", async () => {
    render(
      <TestQueryWrapper>
        <DemandForecastPanel />
      </TestQueryWrapper>
    );
    // "Forecast by ABC Class" appears as both heading and tooltip — use getAllByText
    expect((await screen.findAllByText("Forecast by ABC Class")).length).toBeGreaterThanOrEqual(1);
  });
});
