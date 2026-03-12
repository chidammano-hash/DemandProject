import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    demandSignalsKeys: {
      summary: () => ["ds-summary"],
      list: (p?: unknown) => ["ds-list", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchDemandSignalsSummary: vi.fn().mockResolvedValue({
      above_plan_count: 42,
      below_plan_count: 18,
      urgent_count: 5,
      projected_stockouts: 3,
    }),
    fetchDemandSignals: vi.fn().mockResolvedValue({
      total: 1,
      rows: [
        {
          item_no: "100320",
          loc: "1401-BULK",
          signal_type: "above_plan",
          demand_vs_forecast_pct: 15.2,
          alert_priority: "urgent",
          current_on_hand: 500,
          is_below_ss: false,
        },
      ],
    }),
  };
});

import { DemandSignalsPanel } from "@/tabs/inv-planning/DemandSignalsPanel";

describe("DemandSignalsPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <DemandSignalsPanel />
      </TestQueryWrapper>,
    );
    // "Above Plan" and "Below Plan" appear as both KPI labels and filter buttons
    expect((await screen.findAllByText("Above Plan")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("Below Plan")).length).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText("Urgent Alerts")).toBeDefined();
    expect(await screen.findByText("Projected Stockouts")).toBeDefined();
  });

  it("renders signal rows in table", async () => {
    render(
      <TestQueryWrapper>
        <DemandSignalsPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("100320")).toBeDefined();
    expect(await screen.findByText("urgent")).toBeDefined();
  });

  it("renders info banner", async () => {
    render(
      <TestQueryWrapper>
        <DemandSignalsPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Demand Signals", { exact: false })).toBeDefined();
  });
});
