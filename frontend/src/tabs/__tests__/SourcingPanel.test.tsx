import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
  fetchSourcingRows: vi.fn().mockResolvedValue({
    total: 2,
    rows: [
      { sourcing_ck: 1, site_id: "56", item_id: "1040", loc: "8701-1000", source_cd: "104522-706901", transit_mode: "WEIGHT", supplier_id: "104522", plant_id: "706901" },
      { sourcing_ck: 2, site_id: "56", item_id: "1041", loc: "8701-1000", source_cd: "104522-706901", transit_mode: "WEIGHT", supplier_id: "104522", plant_id: "706901" },
    ],
  }),
  fetchSourcingNetwork: vi.fn().mockResolvedValue({
    total_rows: 1050000,
    supplier_count: 2278,
    item_location_count: 800000,
    single_source_count: 200000,
    multi_source_count: 600000,
    transit_modes: [{ transit_mode: "WEIGHT", count: 500000 }],
  }),
}));

import { SourcingPanel } from "@/tabs/inv-planning/SourcingPanel";

describe("SourcingPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <SourcingPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Total Mappings")).toBeDefined();
    expect(await screen.findByText("Suppliers")).toBeDefined();
    expect(await screen.findByText("Single-Source")).toBeDefined();
  });

  it("renders sourcing table rows", async () => {
    render(
      <TestQueryWrapper>
        <SourcingPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("1040")).toBeDefined();
    expect((await screen.findAllByText("104522-706901")).length).toBeGreaterThan(0);
  });
});
