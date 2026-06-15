import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

const { createSopCycleMock, fetchSopCyclesMock } = vi.hoisted(() => ({
  createSopCycleMock: vi.fn().mockResolvedValue({
    cycle_id: "cycle-new",
    cycle_month: "2026-06-01",
    current_stage: "demand_review",
    created_at: "2026-06-14T00:00:00Z",
    updated_at: "2026-06-14T00:00:00Z",
  }),
  fetchSopCyclesMock: vi.fn(),
}));

vi.mock("@/api/queries/evolution", () => ({
  sopKeys: {
    cycles: (p?: unknown) => ["sop-cycles", p],
    cycle: (id?: string) => ["sop-cycle", id],
    gaps: (id?: string) => ["sop-gaps", id],
    approvedPlan: (p?: unknown) => ["sop-approved-plan", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  advanceSopCycle: vi.fn(),
  approveSopCycle: vi.fn(),
  createSopCycle: createSopCycleMock,
  fetchSopCycles: fetchSopCyclesMock.mockResolvedValue({
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

  // U6.11 — the Approved-Plan month + item-filter inputs were orphan (no
  // label / aria-label), so a screen reader announced nothing and a sighted
  // planner couldn't tell what the inputs filtered.
  it("labels the Approved-Plan month and item filter inputs (U6.11)", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    await screen.findByText("Approved Plan");
    expect(screen.getByLabelText(/plan month/i)).toBeDefined();
    expect(screen.getByLabelText(/item filter/i)).toBeDefined();
  });

  it("shows select prompt when no cycle selected", async () => {
    render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    expect(await screen.findByText(/Select a cycle to view details/)).toBeDefined();
  });

  describe("empty state (U2.21)", () => {
    it("offers an in-app create action instead of a CLI instruction", async () => {
      createSopCycleMock.mockClear();
      fetchSopCyclesMock.mockResolvedValueOnce({ total: 0, cycles: [] });
      render(
        <TestQueryWrapper>
          <SopTab />
        </TestQueryWrapper>
      );

      // No CLI dead-end copy.
      await screen.findByText("S&OP Cycle Management");
      expect(screen.queryByText(/Create one via the API or CLI/i)).toBeNull();

      // A working "Start new cycle" button that calls the create mutation.
      const btn = await screen.findByRole("button", { name: /start new s&op cycle/i });
      fireEvent.click(btn);
      await waitFor(() => expect(createSopCycleMock).toHaveBeenCalledTimes(1));
    });

    // U5.4 — with 0 cycles the detail pane said "Select a cycle to view details"
    // while the cycles card said "Start one to kick off…". The "Select a cycle"
    // instruction is unactionable when there is nothing to select; the two cards
    // contradict each other. In the zero-cycle state the detail pane must not
    // tell the planner to select a cycle.
    it("does not say 'Select a cycle' in the zero-cycle state (U5.4)", async () => {
      fetchSopCyclesMock.mockResolvedValueOnce({ total: 0, cycles: [] });
      render(
        <TestQueryWrapper>
          <SopTab />
        </TestQueryWrapper>
      );
      await screen.findByText("S&OP Cycle Management");
      // Confirm we are in the empty state.
      await screen.findByRole("button", { name: /start new s&op cycle/i });
      expect(screen.queryByText(/Select a cycle to view details/i)).toBeNull();
    });
  });

  // U5.1 — S&OP severity / cycle-stage chips hand-rolled `bg-red-100 text-red-700`
  // with no `dark:` companion, rendering as pale-pastel-on-near-black in Dark.
  // They must now route through the shared themed helper that carries a dark: tint.
  it("cycle-stage chips carry a dark: theme variant (U5.1)", async () => {
    const { container } = render(
      <TestQueryWrapper>
        <SopTab />
      </TestQueryWrapper>
    );
    // approved cycle (2026-05) renders a green stage chip via severityBadgeClass.
    // The hand-rolled chip was Light-only (`bg-green-100 text-green-700`); the
    // migrated chip carries a dark: companion tint.
    await screen.findByText("2026-05");
    expect(container.innerHTML).toMatch(/dark:bg-green-/);
  });
});
