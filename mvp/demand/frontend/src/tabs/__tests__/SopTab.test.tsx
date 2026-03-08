import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/evolution", () => ({
  sopKeys: {
    cycles: (p?: unknown) => ["sop-cycles", p],
    cycle: (id?: string) => ["sop-cycle", id],
    gaps: (id?: string) => ["sop-gaps", id],
    approvedPlan: (p?: unknown) => ["sop-approved-plan", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchSopCycles: vi.fn().mockResolvedValue({
    total: 2,
    cycles: [
      {
        cycle_id: "cycle-2026-04",
        cycle_month: "2026-04-01",
        current_stage: "demand_review",
        created_at: "2026-03-01T00:00:00Z",
        updated_at: "2026-03-01T00:00:00Z",
        demand_review_date: "2026-03-05",
        supply_review_date: "2026-03-10",
        pre_sop_date: "2026-03-15",
        executive_sop_date: "2026-03-20",
      },
      {
        cycle_id: "cycle-2026-05",
        cycle_month: "2026-05-01",
        current_stage: "approved",
        created_at: "2026-04-01T00:00:00Z",
        updated_at: "2026-04-21T00:00:00Z",
        demand_review_date: "2026-04-05",
        supply_review_date: "2026-04-10",
        pre_sop_date: "2026-04-15",
        executive_sop_date: "2026-04-20",
      },
    ],
  }),
  fetchSopGaps: vi.fn().mockResolvedValue({
    total: 0,
    gaps: [],
  }),
  fetchApprovedPlan: vi.fn().mockResolvedValue({
    total: 0,
    page: 1,
    rows: [],
  }),
}));

import SopTab from "@/tabs/SopTab";

describe("SopTab", () => {
  it("renders S&OP page heading", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("S&OP Cycle Management")).toBeDefined();
  });

  it("renders cycle cards", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("2026-04")).toBeDefined();
    expect(await screen.findByText("2026-05")).toBeDefined();
  });

  it("renders stage labels for current cycle", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    // StageTimeline renders labels inline
    const demandReviews = await screen.findAllByText("Demand Review");
    expect(demandReviews.length).toBeGreaterThan(0);
  });

  it("shows approved plan section heading", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Approved Plan")).toBeDefined();
  });

  it("shows select prompt when no cycle selected", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    expect(await screen.findByText(/Select a cycle to view details/)).toBeDefined();
  });
});
