/**
 * F2.3 — OverrideQueuePanel smoke tests
 */

import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import { OverrideQueuePanel } from "../inv-planning/OverrideQueuePanel";
import { TestQueryWrapper } from "./test-utils";

vi.mock("../../api/queries", () => ({
  fetchOverrideSummary: vi.fn().mockResolvedValue({
    by_status: { pending_approval: 3, approved: 5, rejected: 1, expired: 0, superseded: 0 },
    dfu_count_overridden: 4,
    total_uplift_units: 2400,
    total_uplift_value: 12000,
    by_type: { PROMO: 5, MANUAL: 4 },
  }),
  fetchOverrides: vi.fn().mockResolvedValue({
    total: 1,
    page: 1,
    overrides: [
      {
        override_id: 42,
        item_no: "100320",
        loc: "1401-BULK",
        override_month: "2026-05-01",
        override_type: "PROMO",
        override_multiplier: 1.25,
        override_qty: null,
        estimated_impact_units: 300,
        estimated_impact_value: 1500,
        override_reason: "Summer promo campaign",
        created_by: "planner1",
        status: "pending_approval",
        requires_approval: true,
      },
    ],
  }),
  approveOverride: vi.fn().mockResolvedValue({ override_id: 42, status: "approved", approved_by: "manager", approved_at: "2026-03-07T00:00:00Z" }),
  rejectOverride: vi.fn().mockResolvedValue({ override_id: 42, status: "rejected" }),
  STALE: { TWO_MIN: 120000, ONE_MIN: 60000, FIVE_MIN: 300000 },
}));

describe("OverrideQueuePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing", () => {
    render(
      <TestQueryWrapper>
        <OverrideQueuePanel />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Override Queue")).toBeDefined();
  });

  it("shows summary KPI cards after data loads", async () => {
    render(
      <TestQueryWrapper>
        <OverrideQueuePanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("Pending")).toBeDefined();
      // "Approved" and "Rejected" appear in both KPI cards and filter buttons
      expect(screen.getAllByText("Approved").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Rejected").length).toBeGreaterThan(0);
      expect(screen.getByText("Total Uplift")).toBeDefined();
      // "Impact Value" appears in KPI card and table header
      expect(screen.getAllByText("Impact Value").length).toBeGreaterThan(0);
    });
  });

  it("shows status filter buttons", () => {
    render(
      <TestQueryWrapper>
        <OverrideQueuePanel />
      </TestQueryWrapper>
    );
    expect(screen.getByRole("button", { name: "Pending Approval" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Approved" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Rejected" })).toBeDefined();
    expect(screen.getByRole("button", { name: "All" })).toBeDefined();
  });

  it("shows override row after data loads", async () => {
    render(
      <TestQueryWrapper>
        <OverrideQueuePanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("100320")).toBeDefined();
      expect(screen.getByText("PROMO")).toBeDefined();
      expect(screen.getByText("planner1")).toBeDefined();
    });
  });

  it("shows approve and reject buttons for pending overrides", async () => {
    render(
      <TestQueryWrapper>
        <OverrideQueuePanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();
      expect(screen.getByRole("button", { name: "Reject" })).toBeDefined();
    });
  });
});
