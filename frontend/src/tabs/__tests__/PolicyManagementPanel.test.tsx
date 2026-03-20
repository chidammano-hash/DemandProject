import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    queryKeys: {
      ...(actual as any).queryKeys,
      policyList: (p?: unknown) => ["policy-list", p],
      policyCompliance: (p?: unknown) => ["policy-compliance", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchPolicies: vi.fn(),
    fetchPolicyCompliance: vi.fn(),
    assignPolicy: vi.fn(),
    updatePolicy: vi.fn(),
  };
});

import { fetchPolicies, fetchPolicyCompliance } from "@/api/queries";
import { PolicyManagementPanel } from "@/tabs/inv-planning/PolicyManagementPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchPolicies as any).mockResolvedValue({
    policies: [
      {
        policy_id: "p1",
        policy_name: "Continuous ROP",
        policy_type: "continuous_rop",
        segment: "A",
        service_level: 0.98,
        review_cycle_days: null,
        dfu_count: 500,
      },
    ],
  });
  (fetchPolicyCompliance as any).mockResolvedValue({
    assignment_pct: 82,
    assigned_count: 820,
    total_dfus: 1000,
    unassigned_count: 180,
    by_policy: {
      p1: {
        policy_name: "Continuous ROP",
        policy_type: "continuous_rop",
        dfu_count: 500,
        below_ss_pct: 12.5,
        avg_ss_coverage: 85.0,
        avg_dos: 22.0,
      },
    },
  });
});

describe("PolicyManagementPanel", () => {
  it("renders Policy Management heading", async () => {
    render(
      <TestQueryWrapper>
        <PolicyManagementPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Policy Management")).toBeInTheDocument();
    });
  });

  it("renders policy cards", async () => {
    render(
      <TestQueryWrapper>
        <PolicyManagementPanel />
      </TestQueryWrapper>,
    );
    // "Continuous ROP" appears in card title and compliance table
    expect((await screen.findAllByText("Continuous ROP")).length).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText("Edit")).toBeInTheDocument();
  });

  it("renders compliance gauge", async () => {
    render(
      <TestQueryWrapper>
        <PolicyManagementPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("DFU Coverage")).toBeInTheDocument();
    expect(await screen.findByText("82%")).toBeInTheDocument();
  });

  it("renders Auto-assign All button", async () => {
    render(
      <TestQueryWrapper>
        <PolicyManagementPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Auto-assign All")).toBeInTheDocument();
  });
});
