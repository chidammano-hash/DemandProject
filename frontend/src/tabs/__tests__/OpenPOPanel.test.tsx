import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/supply", () => ({
  fetchOpenPOs: vi.fn(),
  fetchOpenPOSummary: vi.fn(),
  fetchPastDuePOs: vi.fn(),
}));

vi.mock("@/api/queries/core", async () => {
  const actual = await vi.importActual("@/api/queries/core");
  return {
    ...actual,
    queryKeys: {
      ...(actual as any).queryKeys,
      openPOSummary: () => ["open-po-summary"],
      openPOs: (p?: unknown) => ["open-pos", p],
      pastDuePOs: () => ["past-due-pos"],
    },
  };
});

import { fetchOpenPOs, fetchOpenPOSummary, fetchPastDuePOs } from "@/api/queries/supply";
import { OpenPOPanel } from "@/tabs/inv-planning/OpenPOPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchOpenPOSummary as any).mockResolvedValue({
    total_open_value_usd: 2500000,
    total_open_lines: 450,
    total_open_qty_by_status: { open: 12000 },
    past_due_lines: 8,
    past_due_value_usd: 45000,
    suppliers_with_open_pos: 22,
    last_loaded_at: "2026-03-10T00:00:00Z",
  });
  (fetchOpenPOs as any).mockResolvedValue({
    total: 1,
    items: [
      {
        po_number: "PO-001",
        po_line_number: 1,
        item_id: "100320",
        loc: "1401-BULK",
        supplier_name: "Acme Corp",
        supplier_id: "S001",
        open_qty: 500,
        line_value: 5000,
        effective_delivery_date: "2026-03-15",
        days_past_due: 0,
        line_status: "open",
      },
    ],
  });
  (fetchPastDuePOs as any).mockResolvedValue({ total: 0, items: [] });
});

describe("OpenPOPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <OpenPOPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Total Open Value")).toBeInTheDocument();
    });
    // "Open Qty" appears in KPI card + table header
    expect(screen.getAllByText("Open Qty").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Past-Due Lines")).toBeInTheDocument();
    expect(screen.getByText("Active Suppliers")).toBeInTheDocument();
  });

  it("renders PO table rows", async () => {
    render(
      <TestQueryWrapper>
        <OpenPOPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Acme Corp")).toBeInTheDocument();
    expect(await screen.findByText("PO-001-1")).toBeInTheDocument();
  });

  it("renders PO Status legend", async () => {
    render(
      <TestQueryWrapper>
        <OpenPOPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("PO Status:", { exact: false })).toBeInTheDocument();
  });
});
