import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/evolution", () => ({
  echelonKeys: {
    summary: (p?: unknown) => ["echelon-summary", p],
    targets: (p?: unknown) => ["echelon-targets", p],
    network: () => ["echelon-network"],
    ropList: (p?: unknown) => ["echelon-rop", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchEchelonTargets: vi.fn().mockResolvedValue({
    total: 1,
    page: 1,
    rows: [
      {
        item_id: "100320",
        loc: "DC-MAIN",
        node_type: "dc",
        pooled_sigma: 45.2,
        echelon_ss: 137.0,
        echelon_rop: 587.0,
        cascade_risk_score: 0.0,
        cascade_risk_severity: "ok",
        downstream_coverage_days: 22.5,
        computed_at: "2026-03-01T00:00:00Z",
      },
    ],
  }),
  fetchEchelonSummary: vi.fn().mockResolvedValue({
    total_nodes: 50,
    critical_count: 3,
    high_count: 8,
    avg_risk_score: 12.5,
    avg_coverage_days: 18.3,
  }),
}));

import { EchelonPanel } from "@/tabs/inv-planning/EchelonPanel";

describe("EchelonPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <EchelonPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Network Nodes")).toBeDefined();
    expect(await screen.findByText("Critical Cascade Risk")).toBeDefined();
    expect(await screen.findByText("High Risk")).toBeDefined();
    expect(await screen.findByText("Avg Echelon Coverage (days)")).toBeDefined();
  });

  it("renders echelon target table row", async () => {
    render(
      <TestQueryWrapper>
        <EchelonPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("100320")).toBeDefined();
    expect(await screen.findByText("Echelon Safety Stock Targets")).toBeDefined();
  });

  it("renders OK severity badge", async () => {
    render(
      <TestQueryWrapper>
        <EchelonPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("OK")).toBeDefined();
  });
});
