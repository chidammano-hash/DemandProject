/**
 * F2.4 — ProcurementPanel smoke tests
 */

import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import { ProcurementPanel } from "../inv-planning/ProcurementPanel";
import { TestQueryWrapper } from "./test-utils";

vi.mock("../../api/queries", () => ({
  fetchPurchaseOrders: vi.fn().mockResolvedValue({
    total: 2,
    total_value: 15168.0,
    page: 1,
    orders: [
      {
        po_number: "DS-2026-04-001",
        line_number: 1,
        item_id: "100320",
        item_description: "Bulk Cleaning Solution",
        loc: "1401-BULK",
        supplier_id: "SUP-4821",
        supplier_name: "ABC Trading Co",
        ordered_qty: 316.0,
        unit_cost: 24.0,
        total_value: 7584.0,
        currency: "USD",
        po_date: "2026-04-15",
        requested_delivery_date: "2026-04-28",
        confirmed_delivery_date: null,
        status: "proposed",
        source_exception_id: 7834,
        created_by: "planner1",
        planner_approved_by: null,
        buyer_released_by: null,
        erp_po_number: null,
      },
      {
        po_number: "DS-2026-04-002",
        line_number: 1,
        item_id: "204771",
        item_description: "Industrial Degreaser",
        loc: "2203-STD",
        supplier_id: "SUP-4821",
        supplier_name: "ABC Trading Co",
        ordered_qty: 80.0,
        unit_cost: 67.5,
        total_value: 5400.0,
        currency: "USD",
        po_date: "2026-04-15",
        requested_delivery_date: "2026-04-28",
        confirmed_delivery_date: null,
        status: "planner_approved",
        source_exception_id: 7835,
        created_by: "planner1",
        planner_approved_by: "planner1",
        buyer_released_by: null,
        erp_po_number: null,
      },
    ],
  }),
  approvePurchaseOrder: vi.fn().mockResolvedValue({
    po_number: "DS-2026-04-001", status: "planner_approved", approved_by: "planner1",
  }),
  releasePurchaseOrder: vi.fn().mockResolvedValue({
    po_number: "DS-2026-04-002", status: "buyer_released", released_by: "buyer1",
  }),
  exportPOsCSV: vi.fn().mockResolvedValue({
    filename: "PO_export_test.csv", line_count: 1, total_value: 7584, csv_content: "PO_NUMBER\n",
  }),
  fetchPOTimeline: vi.fn().mockResolvedValue({
    po_number: "DS-2026-04-001",
    current_status: "proposed",
    timeline: [
      {
        action: "proposed", performed_by: "system",
        performed_at: "2026-04-15T09:14:00Z",
        old_status: null, new_status: "proposed",
        old_qty: null, new_qty: 316.0, reason: null,
        note: "Auto-created from exception EXC-7834",
      },
    ],
  }),
  STALE: { ONE_MIN: 60000, TWO_MIN: 120000, FIVE_MIN: 300000 },
}));

describe("ProcurementPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing", () => {
    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Procurement Workflow")).toBeDefined();
  });

  it("shows status filter buttons", () => {
    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    expect(screen.getByRole("button", { name: "All" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Proposed" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Released" })).toBeDefined();
  });

  it("shows PO list items after data loads", async () => {
    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("DS-2026-04-001")).toBeDefined();
      expect(screen.getByText("DS-2026-04-002")).toBeDefined();
    });
  });

  it("shows supplier name in order queue", async () => {
    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getAllByText("ABC Trading Co").length).toBeGreaterThan(0);
    });
  });

  it("shows Approve button for proposed orders", async () => {
    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();
    });
  });

  it("shows Release button for approved orders", async () => {
    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Release" })).toBeDefined();
    });
  });

  it("shows empty state when no orders", async () => {
    const { fetchPurchaseOrders } = await import("../../api/queries");
    (fetchPurchaseOrders as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      total: 0, total_value: 0, page: 1, orders: [],
    });

    render(
      <TestQueryWrapper>
        <ProcurementPanel />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("No purchase orders in workflow")).toBeDefined();
    });
  });
});
