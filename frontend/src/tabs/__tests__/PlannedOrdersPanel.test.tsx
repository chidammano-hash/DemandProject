import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    queryKeys: {
      ...(actual as any).queryKeys,
      plannedOrdersSummary: () => ["planned-orders-summary"],
      plannedOrders: (p?: unknown) => ["planned-orders", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchPlannedOrders: vi.fn(),
    fetchPlannedOrdersSummary: vi.fn(),
    approvePlannedOrder: vi.fn(),
    rejectPlannedOrder: vi.fn(),
    generatePlannedOrders: vi.fn(),
  };
});

import {
  fetchPlannedOrders,
  fetchPlannedOrdersSummary,
} from "@/api/queries";
import { PlannedOrdersPanel } from "@/tabs/inv-planning/PlannedOrdersPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchPlannedOrdersSummary as any).mockResolvedValue({
    status_counts: { proposed: 15, approved: 8 },
    past_due_proposed_count: 3,
    total_proposed_value_usd: 75000,
    total_approved_value_usd: 40000,
    low_confidence_count: 4,
    avg_confidence_score: 0.72,
    generated_at: "2026-03-10T10:00:00Z",
  });
  (fetchPlannedOrders as any).mockResolvedValue({
    total: 1,
    items: [
      {
        id: 1,
        item_id: "100320",
        loc: "1401-BULK",
        supplier_name: "Acme Corp",
        supplier_id: "S001",
        recommended_qty: 500,
        order_value: 5000,
        order_by_date: "2026-03-20",
        lead_time_days: 14,
        confidence_score: 0.85,
        is_past_due: false,
        status: "proposed",
      },
    ],
  });
});

describe("PlannedOrdersPanel", () => {
  it("renders heading and Generate button", async () => {
    render(
      <TestQueryWrapper>
        <PlannedOrdersPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Planned Orders")).toBeInTheDocument();
    });
    expect(screen.getByText("Generate")).toBeInTheDocument();
  });

  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <PlannedOrdersPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Proposed")).toBeInTheDocument();
    expect(await screen.findByText("Past Due")).toBeInTheDocument();
    expect(await screen.findByText("Approved")).toBeInTheDocument();
    expect(await screen.findByText("Low Confidence")).toBeInTheDocument();
  });

  it("renders order table rows with Approve/Reject buttons", async () => {
    render(
      <TestQueryWrapper>
        <PlannedOrdersPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("100320")).toBeInTheDocument();
    expect(await screen.findByText("Approve")).toBeInTheDocument();
    expect(await screen.findByText("Reject")).toBeInTheDocument();
  });

  it("renders confidence score legend", async () => {
    render(
      <TestQueryWrapper>
        <PlannedOrdersPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Confidence Score:", { exact: false })).toBeInTheDocument();
  });
});
