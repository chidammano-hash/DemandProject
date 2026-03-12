import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    supplierKeys: {
      summary: (p?: unknown) => ["supplier-summary", p],
      detail: (p?: unknown) => ["supplier-detail", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchSupplierSummary: vi.fn().mockResolvedValue({
      total_suppliers: 42,
      avg_reliability_score: 72,
      avg_lead_time_days: 18.5,
      low_reliability_count: 5,
    }),
    fetchSupplierDetail: vi.fn().mockResolvedValue({
      rows: [
        {
          supplier_no: "S001",
          supplier_name: "Acme Corp",
          supplier_reliability_score: 85,
          sku_loc_count: 120,
          avg_lt_mean_days: 14.5,
          avg_lt_cv: 0.15,
          pct_stable_lt: 0.82,
        },
      ],
    }),
  };
});

import { SupplierPanel } from "@/tabs/inv-planning/SupplierPanel";

describe("SupplierPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <SupplierPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Total Suppliers")).toBeDefined();
    expect(await screen.findByText("Avg Reliability Score")).toBeDefined();
    expect(await screen.findByText("Avg Lead Time (days)")).toBeDefined();
    expect(await screen.findByText("Low Reliability (<40)")).toBeDefined();
  });

  it("renders supplier table", async () => {
    render(
      <TestQueryWrapper>
        <SupplierPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Acme Corp")).toBeDefined();
    expect(await screen.findByText("Suppliers by Reliability (highest first)")).toBeDefined();
  });
});
