import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchPORows: vi.fn().mockResolvedValue({
      total: 1,
      rows: [
        {
          po_ck: 1, po_number: "3061972", site_id: "25", loc: "6401-OTHR",
          source: "106037-700484", item_id: "540765", ordered_qty: 10,
          net_price: 524, gross_value: 5240, closure_code: "CLOSED",
          po_hdr_status: "INACTIVE", po_line_status: "INACTIVE",
          receipt_status: "DELIVERED", supplier_id: "106037",
          supplier_name: "FRUIT OF THE VINES", carrier_name: "ADVANTAGE TRANS",
          delivery_date: "2023-08-30", original_delivery_date: "2022-03-08",
          current_ship_date: "2023-07-18", original_ship_date: "2022-01-10",
          po_type: "YO", is_closed: true, lead_time_planned: 100, lead_time_actual: 232,
        },
      ],
    }),
    fetchPOSummary: vi.fn().mockResolvedValue({
      total_lines: 5641373, closed_lines: 5354000, open_lines: 287373,
      distinct_pos: 100000, distinct_suppliers: 2278, distinct_items: 50000,
      total_value: 1000000, open_value: 50000, closed_value: 950000,
    }),
    fetchPOAging: vi.fn().mockResolvedValue({
      buckets: [
        { age_bucket: "0-30", line_count: 1000, total_value: 50000 },
        { age_bucket: "30-60", line_count: 500, total_value: 25000 },
      ],
    }),
}));

import { PurchaseOrdersPanel } from "@/tabs/inv-planning/PurchaseOrdersPanel";

describe("PurchaseOrdersPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <PurchaseOrdersPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Total PO Lines")).toBeDefined();
    expect(await screen.findByText("Open Lines")).toBeDefined();
    expect(await screen.findByText("Closed Lines")).toBeDefined();
  });

  it("renders aging buckets", async () => {
    render(
      <TestQueryWrapper>
        <PurchaseOrdersPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("0-30 days")).toBeDefined();
    expect(await screen.findByText("30-60 days")).toBeDefined();
  });

  it("renders PO table rows", async () => {
    render(
      <TestQueryWrapper>
        <PurchaseOrdersPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("3061972")).toBeDefined();
    expect(await screen.findByText("540765")).toBeDefined();
  });
});
