import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/evolution", () => ({
  scenarioKeys: {
    list: (p?: unknown) => ["scenarios-list", p],
    scenario: (id?: string) => ["scenario", id],
    results: (id?: string) => ["scenario-results", id],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchSupplyScenarios: vi.fn().mockResolvedValue({
    total: 2,
    scenarios: [
      {
        scenario_id: "sc-1",
        scenario_name: "Supplier Delay Q1",
        disruption_type: "supplier_delay",
        item_no: "100320",
        loc: "1401-BULK",
        impact_pct: 30,
        duration_weeks: 3,
        status: "completed",
        created_at: "2026-03-01T00:00:00Z",
      },
      {
        scenario_id: "sc-2",
        scenario_name: "Capacity Crunch",
        disruption_type: "capacity_constraint",
        item_no: null,
        loc: null,
        impact_pct: 20,
        duration_weeks: 4,
        status: "draft",
        created_at: "2026-03-02T00:00:00Z",
      },
    ],
  }),
  fetchScenarioResults: vi.fn().mockResolvedValue({
    scenario_id: "sc-1",
    items: [],
    total_impact: 0,
    total_stockout_days: 0,
  }),
}));

import { ScenarioPlanningPanel } from "@/tabs/inv-planning/ScenarioPlanningPanel";

describe("ScenarioPlanningPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioPlanningPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Total Scenarios")).toBeDefined();
    const completedEls = await screen.findAllByText("Completed");
    expect(completedEls.length).toBeGreaterThan(0);
  });

  it("renders scenario list", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioPlanningPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Supplier Delay Q1")).toBeDefined();
    expect(await screen.findByText("Capacity Crunch")).toBeDefined();
  });

  it("renders scenario status badges", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioPlanningPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("completed")).toBeDefined();
    expect(await screen.findByText("draft")).toBeDefined();
  });

  it("renders empty results prompt when no scenario selected", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioPlanningPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Select a Scenario")).toBeDefined();
    expect(await screen.findByText(/Click a scenario to view results/)).toBeDefined();
  });
});
