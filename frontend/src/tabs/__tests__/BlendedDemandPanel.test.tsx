import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/evolution", () => ({
  blendedKeys: {
    summary: (p?: unknown) => ["blended-summary", p],
    list: (p?: unknown) => ["blended-list", p],
    sensingActive: (p?: unknown) => ["sensing-active", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchBlendedForecast: vi.fn().mockResolvedValue({
    total: 1,
    page: 1,
    rows: [
      {
        item_id: "100320",
        loc: "1401-BULK",
        week_start: "2026-03-09",
        plan_version: "latest",
        alpha_weight: 0.75,
        sensing_signal_qty: 100.5,
        statistical_forecast_qty: 90.0,
        blended_qty: 97.9,
        velocity_spike_ratio: 1.1,
        is_outlier_capped: false,
      },
    ],
  }),
  fetchBlendedSummary: vi.fn().mockResolvedValue({
    total_skus: 200,
    total_weeks: 800,
    avg_alpha: 0.62,
    capped_count: 5,
    plan_version: "latest",
    latest_week: "2026-03-09",
  }),
}));

import { BlendedDemandPanel } from "@/tabs/inv-planning/BlendedDemandPanel";

describe("BlendedDemandPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <BlendedDemandPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("SKUs Active")).toBeDefined();
    expect(await screen.findByText("Weeks Computed")).toBeDefined();
    expect(await screen.findByText("Avg Alpha (Current)")).toBeDefined();
    expect(await screen.findByText("Capped Outliers")).toBeDefined();
  });

  it("renders blended forecast table row", async () => {
    render(
      <TestQueryWrapper>
        <BlendedDemandPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("100320")).toBeDefined();
    expect(await screen.findByText("Blended Weekly Demand Plan")).toBeDefined();
  });
});
