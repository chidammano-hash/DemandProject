import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    simulationKeys: {
      results: (p?: unknown) => ["sim-results", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchSimulationResults: vi.fn(),
    runSimulation: vi.fn(),
  };
});

import { fetchSimulationResults } from "@/api/queries";
import { SimulationPanel } from "@/tabs/inv-planning/SimulationPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchSimulationResults as any).mockResolvedValue({
    rows: [
      {
        sim_run_id: "sim-001",
        item_id: "100320",
        loc: "1401-BULK",
        simulation_date: "2026-03-01T00:00:00Z",
        recommended_ss: 250,
        analytical_ss: 200,
        sim_vs_analytical_pct: 25.0,
        run_duration_secs: 18.5,
        results_by_ss_level: [
          { ss_qty: 100, csl: 75 },
          { ss_qty: 250, csl: 95 },
        ],
      },
    ],
  });
});

describe("SimulationPanel", () => {
  it("renders Run Simulation button", async () => {
    render(
      <TestQueryWrapper>
        <SimulationPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Run Simulation")).toBeInTheDocument();
  });

  it("renders KPI cards from active result", async () => {
    render(
      <TestQueryWrapper>
        <SimulationPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Safety Stock Target (units)")).toBeInTheDocument();
    });
    expect(screen.getByText("Z-Score Formula (units)")).toBeInTheDocument();
    expect(screen.getByText("Assessment")).toBeInTheDocument();
  });

  it("renders recent simulation runs table", async () => {
    render(
      <TestQueryWrapper>
        <SimulationPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Recent Simulation Runs")).toBeInTheDocument();
    expect(await screen.findByText("100320")).toBeInTheDocument();
  });
});
