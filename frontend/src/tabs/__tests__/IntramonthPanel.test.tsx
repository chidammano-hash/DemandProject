import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    intramonthKeys: {
      summary: (p?: unknown) => ["im-summary", p],
      detail: (p?: unknown) => ["im-detail", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchIntramonthSummary: vi.fn().mockResolvedValue({
      total_items: 1000,
      items_with_stockout: 42,
      items_with_extended_stockout: 8,
      total_stockout_days: 350,
      total_est_lost_sales: 15000,
    }),
    fetchIntramonthDetail: vi.fn().mockResolvedValue({
      rows: [
        {
          item_id: "100320",
          loc: "1401-BULK",
          month_start: "2026-03-01",
          stockout_days: 12,
          stockout_day_rate: 0.4,
          est_lost_sales: 450,
          had_extended_stockout: true,
        },
      ],
    }),
  };
});

import { IntramonthPanel } from "@/tabs/inv-planning/IntramonthPanel";

describe("IntramonthPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <IntramonthPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Items with Stockout")).toBeDefined();
    expect(await screen.findByText("Extended Stockouts (7d+)")).toBeDefined();
    expect(await screen.findByText("Total Stockout Days")).toBeDefined();
    // "Est. Lost Sales" appears as KPI label and table header
    expect((await screen.findAllByText("Est. Lost Sales")).length).toBeGreaterThanOrEqual(1);
  });

  it("renders top stockout items table", async () => {
    render(
      <TestQueryWrapper>
        <IntramonthPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("100320")).toBeDefined();
    expect(await screen.findByText("Top Stockout Items (current period)")).toBeDefined();
  });

  it("renders info banner", async () => {
    render(
      <TestQueryWrapper>
        <IntramonthPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Intramonth stockouts", { exact: false })).toBeDefined();
  });
});
